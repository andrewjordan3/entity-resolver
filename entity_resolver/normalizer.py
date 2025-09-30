# entity_resolver/normalizer.py
"""
TextNormalizer module for cleaning and standardizing entity names.

This module provides text normalization capabilities including lowercasing,
suffix removal, business name qualifier handling, and consolidation of
entities sharing common addresses.
"""

import re
import logging
from typing import Dict
import pandas as pd
import cudf

# Local Package Imports
from .config import NormalizationConfig, VectorizerConfig
from .utils import get_canonical_name_gpu, nfkc_normalize_series

# Set up module-level logger
logger = logging.getLogger(__name__)


class TextNormalizer:
    """
    Handle cleaning and standardization of entity names for resolution.
    
    Optimized for entity resolution and deduplication, this class balances
    standardization with preservation of distinguishing features. It handles
    common business variations while keeping proper nouns distinct.

    Attributes:
        config (NormalizationConfig): Configuration for normalization rules
        vectorizer_config (VectorizerConfig): Configuration for vectorizer settings
        _compiled_patterns (Dict): Pre-compiled regex patterns for efficiency
        _consolidation_stats (Dict): Statistics from last consolidation operation
        min_string_length (int): Minimum length to preserve during normalization
    """
    
    def __init__(
        self, 
        config: NormalizationConfig, 
        vectorizer_config: VectorizerConfig,
        min_string_length: int = 2
    ):
        """
        Initialize the TextNormalizer with configuration settings.

        Args:
            config: NormalizationConfig object containing replacement rules,
                    suffixes to remove, and other normalization parameters.
            vectorizer_config: VectorizerConfig object for downstream processing.
            min_string_length: Minimum string length to preserve. Strings shorter
                             than this after normalization will use minimal processing.
        """
        self.config = config
        self.vectorizer_config = vectorizer_config
        self.min_string_length = min_string_length
        self._consolidation_stats = {}
        
        # Convert config to dictionaries/sets for efficient lookup
        self.replacements = dict(config.replacements) if config.replacements else {}
        self.suffixes_to_remove = set(config.suffixes_to_remove) if config.suffixes_to_remove else set()
        
        # Remove overlap between replacements and suffixes
        self._dedup_config_overlap()
        
        # Compile patterns after deduplication
        self._compiled_patterns = self._compile_regex_patterns()
        
        logger.info("Initialized TextNormalizer for entity resolution")
        logger.debug(f"Replacements configured: {len(self.replacements)}")
        logger.debug(f"Suffixes to remove: {len(self.suffixes_to_remove)}")
        logger.debug(f"Minimum string length: {min_string_length}")

    def _dedup_config_overlap(self):
        """
        Remove overlapping entries between replacements and suffixes.
        
        If a word appears as both a replacement target and a suffix to remove,
        prefer the replacement (more specific intent).
        """
        if not self.replacements or not self.suffixes_to_remove:
            return
            
        # Find overlaps
        replacement_keys = set(self.replacements.keys())
        overlaps = replacement_keys & self.suffixes_to_remove
        
        if overlaps:
            logger.debug(f"Found {len(overlaps)} overlapping entries between replacements and suffixes")
            # Remove overlaps from suffixes (prefer replacements)
            self.suffixes_to_remove -= overlaps
            logger.debug(f"Removed from suffixes: {overlaps}")

    def _compile_regex_patterns(self) -> Dict[str, any]:
        """
        Pre-compile regex patterns compatible with cuDF's RE2 engine.
        
        Since RE2 doesn't support \\b word boundaries, we use character class
        patterns like (^|[^a-z0-9]) for word starts and ($|[^a-z0-9]) for word ends.
        
        Returns:
            Dictionary containing pattern strings and replacement info.
        """
        patterns = {}
        logger.debug("Compiling regex patterns for cuDF RE2 engine")

        # Business qualifier patterns - matches anywhere in string
        # Captures text after the qualifier as the main business name
        patterns['business_qualifier'] = {
            'pattern': (
                r'(?:^|.*?[^a-z0-9])'  # Non-capturing group for text before
                r'(?:'
                r'd[\/.\-\s]*b[\/.\-\s]*a|'           # dba variations
                r'f[\/.\-\s]*k[\/.\-\s]*a|'           # fka variations  
                r'a[\/.\-\s]*k[\/.\-\s]*a|'           # aka variations
                r't[\/.\-\s]*a(?:[^a-z0-9]|$)|'       # ta (trading as) - avoid matching "tax"
                r'formerly|'                          # formerly
                r'now\s+known\s+as|'                  # now known as
                r'trading\s+as|'                      # trading as
                r'doing\s+business\s+as'              # doing business as
                r')'
                r'[\s:]+(.+?)$'                       # Capture the actual business name
            ),
            'type': 'extract'
        }
        
        # Separator patterns for standardization
        patterns['ampersand'] = {'pattern': r'&+', 'replacement': ' and ', 'type': 'simple'}
        patterns['plus'] = {'pattern': r'\++', 'replacement': ' and ', 'type': 'simple'}
        patterns['forward_slash'] = {'pattern': r'/', 'replacement': ' ', 'type': 'simple'}
        patterns['pipe'] = {'pattern': r'\|', 'replacement': ' ', 'type': 'simple'}
        patterns['middle_dot'] = {'pattern': r'·', 'replacement': ' ', 'type': 'simple'}
        patterns['bullet'] = {'pattern': r'•', 'replacement': ' ', 'type': 'simple'}
        patterns['dashes'] = {'pattern': r'[–—-]+', 'replacement': ' ', 'type': 'simple'}
        
        # Common letter replacements for matching (e.g., "Triple A" -> "AAA")
        patterns['triple'] = {
            'pattern': r'(^|[^a-z0-9])triple\s+([a-z])($|[^a-z0-9])',
            'replacement': r'\1\2\2\2\3',
            'type': 'backrefs'
        }
        patterns['double'] = {
            'pattern': r'(^|[^a-z0-9])double\s+([a-z])($|[^a-z0-9])',
            'replacement': r'\1\2\2\3',
            'type': 'backrefs'
        }
        
        # Noise removal patterns
        patterns['parenthetical'] = {'pattern': r'\([^)]*\)', 'replacement': ' ', 'type': 'simple'}
        patterns['bracketed'] = {'pattern': r'\[[^\]]*\]', 'replacement': ' ', 'type': 'simple'}
        
        # OCR corrections (only in middle of words)
        patterns['ocr_zero'] = {
            'pattern': r'([a-z])0([a-z])',
            'replacement': r'\1o\2',
            'type': 'backrefs'
        }
        patterns['ocr_one'] = {
            'pattern': r'([a-z])1([a-z])',
            'replacement': r'\1l\2',
            'type': 'backrefs'
        }
        
        # Possessives
        patterns['possessive'] = {'pattern': r"'s(?:$|[^a-z0-9])", 'replacement': ' ', 'type': 'simple'}
        
        # Special characters (keep alphanumeric and spaces)
        patterns['special_chars'] = {'pattern': r'[^\w\s]', 'replacement': ' ', 'type': 'simple'}
        
        # Multiple spaces
        patterns['multi_space'] = {'pattern': r'\s+', 'replacement': ' ', 'type': 'simple'}
        
        # Build replacement patterns with proper word boundaries for RE2
        if self.replacements:
            for old_word, new_word in self.replacements.items():
                key = f'repl_{old_word}'
                # Use groups to simulate word boundaries
                patterns[key] = {
                    'pattern': r'(^|[^a-z0-9])' + re.escape(old_word) + r'($|[^a-z0-9])',
                    'replacement': r'\1' + new_word + r'\2',
                    'type': 'backrefs'
                }
            
        # Build suffix removal patterns
        if self.suffixes_to_remove:
            # Sort by length (longest first) to handle compound suffixes
            sorted_suffixes = sorted(self.suffixes_to_remove, key=len, reverse=True)
            
            # Create individual patterns for each suffix to ensure proper boundary handling
            for suffix in sorted_suffixes:
                key = f'suffix_{suffix}'
                # Match suffix with word boundary simulation
                patterns[key] = {
                    'pattern': r'(^|[^a-z0-9])' + re.escape(suffix) + r'(?:\W*)?$',
                    'replacement': r'\1',
                    'type': 'backrefs'
                }
            
        logger.info(f"Compiled {len(patterns)} normalization patterns")
        return patterns

    def normalize_text(self, gdf: cudf.DataFrame, entity_col: str) -> cudf.DataFrame:
        """
        Apply normalization rules to entity names for entity resolution.
        
        Balances standardization with preservation of distinguishing features.
        Designed to match variations like "Triple A Hardware" with "AAA Hardware"
        or "Mack's Trk" with "Mac's Truck" while keeping proper nouns distinct.
        
        Normalization pipeline:
        1. Unicode normalization (NFKC)
        2. Lowercase conversion
        3. Expand word-number patterns (triple a -> aaa)
        4. Word replacements from config (trk -> truck)
        5. Business qualifier extraction (dba, fka, etc.)
        6. Separator standardization (& -> and)
        7. Noise removal (parenthetical content)
        8. OCR error correction
        9. Suffix removal (inc, llc, etc.)
        10. Final cleanup and validation
        
        Args:
            gdf: Input cuDF DataFrame containing entity data.
            entity_col: Name of column with entity names to normalize.
            
        Returns:
            cuDF DataFrame with added 'normalized_text' column.
            
        Raises:
            KeyError: If entity_col doesn't exist in the DataFrame.
        """
        if entity_col not in gdf.columns:
            raise KeyError(f"Column '{entity_col}' not found in DataFrame")
            
        logger.info(f"Starting entity name normalization for {len(gdf):,} records")
        
        # Start with original text
        normalized = gdf[entity_col].fillna('').astype('str')
        
        # Track statistics
        initial_unique = normalized.nunique()
        initial_avg_len = normalized.str.len().mean()
        logger.debug(f"Initial: {initial_unique:,} unique, avg length {initial_avg_len:.1f}")
        
        # Step 1: Unicode normalization
        logger.debug("Step 1: Unicode normalization (NFKC)")
        normalized = nfkc_normalize_series(normalized)
        
        # Step 2: Lowercase
        logger.debug("Step 2: Converting to lowercase")
        normalized = normalized.str.lower()
        
        # Step 3: Expand word-number patterns (triple a -> aaa, double u -> uu)
        logger.debug("Step 3: Expanding word-number patterns")
        for pattern_key in ['triple', 'double']:
            if pattern_key in self._compiled_patterns:
                pattern_info = self._compiled_patterns[pattern_key]
                normalized = normalized.str.replace_with_backrefs(
                    pattern_info['pattern'],
                    pattern_info['replacement']
                )
        
        # Step 4: Apply word replacements from config
        logger.debug(f"Step 4: Applying {len(self.replacements)} word replacements")
        replacement_keys = [k for k in self._compiled_patterns.keys() if k.startswith('repl_')]
        
        for key in replacement_keys:
            pattern_info = self._compiled_patterns[key]
            normalized = normalized.str.replace_with_backrefs(
                pattern_info['pattern'],
                pattern_info['replacement']
            )
        
        # Step 5: Extract business qualifiers (dba, fka, etc.)
        logger.debug("Step 5: Processing business qualifiers")
        if 'business_qualifier' in self._compiled_patterns:
            pattern = self._compiled_patterns['business_qualifier']['pattern']
            extracted = normalized.str.extract(pattern, expand=False)
            qualifier_count = extracted.notna().sum()
            if qualifier_count > 0:
                # Use the extracted name where found, otherwise keep original
                normalized = extracted.fillna(normalized)
                logger.info(f"  Extracted business names from {qualifier_count:,} qualifiers")
        
        # Step 6: Standardize separators
        logger.debug("Step 6: Standardizing separators and symbols")
        for key in ['ampersand', 'plus', 'forward_slash', 'pipe', 'middle_dot', 'bullet', 'dashes']:
            if key in self._compiled_patterns:
                pattern_info = self._compiled_patterns[key]
                normalized = normalized.str.replace(
                    pattern_info['pattern'],
                    pattern_info['replacement'],
                    regex=True
                )
        
        # Step 7: Remove noise (parenthetical content)
        logger.debug("Step 7: Removing parenthetical and bracketed content")
        for key in ['parenthetical', 'bracketed']:
            if key in self._compiled_patterns:
                pattern_info = self._compiled_patterns[key]
                normalized = normalized.str.replace(
                    pattern_info['pattern'],
                    pattern_info['replacement'],
                    regex=True
                )
        
        # Step 8: OCR corrections
        logger.debug("Step 8: Applying OCR corrections")
        for key in ['ocr_zero', 'ocr_one']:
            if key in self._compiled_patterns:
                pattern_info = self._compiled_patterns[key]
                normalized = normalized.str.replace_with_backrefs(
                    pattern_info['pattern'],
                    pattern_info['replacement']
                )
        
        # Step 9: Remove suffixes (iteratively to handle multiple)
        logger.debug(f"Step 9: Removing {len(self.suffixes_to_remove)} business suffixes")
        suffix_keys = [k for k in self._compiled_patterns.keys() if k.startswith('suffix_')]
        
        # Apply suffix removal up to 3 times to handle cases like "inc usa llc"
        for iteration in range(3):
            changes_made = False
            for key in suffix_keys:
                pattern_info = self._compiled_patterns[key]
                before = normalized.copy()
                normalized = normalized.str.replace_with_backrefs(
                    pattern_info['pattern'],
                    pattern_info['replacement']
                )
                if not (normalized == before).all():
                    changes_made = True
            
            if not changes_made:
                break  # No more suffixes to remove
        
        # Step 10: Final cleanup
        logger.debug("Step 10: Final cleanup")
        
        # Remove possessives
        if 'possessive' in self._compiled_patterns:
            pattern_info = self._compiled_patterns['possessive']
            normalized = normalized.str.replace(
                pattern_info['pattern'],
                pattern_info['replacement'],
                regex=True
            )
        
        # Remove remaining special characters
        if 'special_chars' in self._compiled_patterns:
            pattern_info = self._compiled_patterns['special_chars']
            normalized = normalized.str.replace(
                pattern_info['pattern'],
                pattern_info['replacement'],
                regex=True
            )
        
        # Clean up multiple spaces and trim
        if 'multi_space' in self._compiled_patterns:
            pattern_info = self._compiled_patterns['multi_space']
            normalized = normalized.str.replace(
                pattern_info['pattern'],
                pattern_info['replacement'],
                regex=True
            )
        
        normalized = normalized.str.strip()
        
        # Step 11: Protect against over-normalization
        logger.debug("Step 11: Checking for over-normalization")
        
        # Create a mask for strings that became too short
        too_short_mask = (normalized.str.len() < self.min_string_length) & (normalized != '')
        
        # For strings that became too short, use minimal normalization
        if too_short_mask.any():
            count_too_short = too_short_mask.sum()
            logger.warning(f"  {count_too_short:,} strings became too short, using minimal normalization")
            
            # Apply minimal normalization (just lowercase, trim, and basic cleanup)
            original_minimal = gdf[entity_col].fillna('').astype('str')
            original_minimal = original_minimal.str.lower()
            original_minimal = original_minimal.str.replace(r'[^\w\s]', ' ', regex=True)
            original_minimal = original_minimal.str.replace(r'\s+', ' ', regex=True)
            original_minimal = original_minimal.str.strip()
            
            # Replace too-short normalized strings with minimally processed originals
            normalized[too_short_mask] = original_minimal[too_short_mask]
        
        # Handle completely empty strings
        empty_mask = (normalized == '') | normalized.isna()
        if empty_mask.any():
            empty_count = empty_mask.sum()
            logger.warning(f"  {empty_count:,} records became empty, marking as UNKNOWN")
            normalized[empty_mask] = 'unknown_entity'
        
        # Add to DataFrame
        gdf['normalized_text'] = normalized
        
        # Log statistics
        final_unique = gdf['normalized_text'].nunique()
        reduction = 1 - (final_unique / initial_unique) if initial_unique > 0 else 0
        
        # Log some example normalizations if in debug mode
        if logger.isEnabledFor(logging.DEBUG) and len(gdf) > 0:
            sample_size = min(5, len(gdf))
            sample_df = gdf[[entity_col, 'normalized_text']].head(sample_size)
            logger.debug("Sample normalizations:")
            for idx, row in enumerate(sample_df.to_pandas().itertuples()):
                logger.debug(f"  '{row[1]}' -> '{row[2]}'")
        
        logger.info(
            f"Normalization complete: {initial_unique:,} → {final_unique:,} unique "
            f"({reduction:.1%} reduction)"
        )
        
        return gdf
    
    def consolidate_by_address(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Consolidates multiple entity names at the same address using vectorized GPU operations.

        This method identifies the single best canonical name to represent all entities
        at each location, resolving name variations for the same business. All
        operations are performed on the GPU to maximize performance.

        Args:
            gdf: A cuDF DataFrame containing 'addr_normalized_key' and 'normalized_text'.

        Returns:
            A new cuDF DataFrame with the 'normalized_text' column updated with
            canonical names.

        Raises:
            KeyError: If required columns are missing from the input DataFrame.
        """
        # --- 1. Initialization and Validation ---
        self._consolidation_stats = {
            'total_addresses': 0, 'addresses_with_multiple_names': 0,
            'records_affected': 0, 'names_changed': 0, 'unique_before': 0,
            'unique_after': 0, 'empty_addresses_skipped': 0,
            'consolidation_examples': []
        }
        required_columns = ['addr_normalized_key', 'normalized_text']
        if any(col not in gdf.columns for col in required_columns):
            raise KeyError(f"Missing one or more required columns: {required_columns}")

        logger.info("=" * 60)
        logger.info("Starting address-based name consolidation")
        logger.info("=" * 60)

        # --- 2. Identify Records to Process ---
        empty_address_mask = (gdf['addr_normalized_key'] == '') | gdf['addr_normalized_key'].isna()
        num_empty_addresses = int(empty_address_mask.sum())
        self._consolidation_stats['empty_addresses_skipped'] = num_empty_addresses
        if num_empty_addresses > 0:
            logger.warning(f"Found {num_empty_addresses:,} records with empty addresses to be skipped.")

        valid_records_gdf = gdf[~empty_address_mask]
        if valid_records_gdf.empty:
            logger.warning("No valid addresses found for consolidation.")
            return gdf

        self._consolidation_stats['unique_before'] = int(gdf['normalized_text'].nunique())

        # --- 3. Find Addresses with Multiple Names ---
        logger.info("Analyzing name variations per address on GPU...")
        names_per_address = valid_records_gdf.groupby('addr_normalized_key')['normalized_text'].nunique()
        self._consolidation_stats['total_addresses'] = len(names_per_address)

        addresses_to_consolidate_index = names_per_address[names_per_address > 1].index
        num_to_consolidate = len(addresses_to_consolidate_index)
        self._consolidation_stats['addresses_with_multiple_names'] = num_to_consolidate

        if num_to_consolidate == 0:
            logger.info("No addresses found with multiple names. Skipping consolidation.")
            self._log_consolidation_summary()
            return gdf

        logger.info(f"Found {num_to_consolidate:,} addresses with multiple name variations to consolidate.")

        # --- 4. Build and Apply Canonical Mapping ---
        # Isolate only the records that are part of the consolidation effort.
        consolidation_subset_gdf = valid_records_gdf[
            valid_records_gdf['addr_normalized_key'].isin(addresses_to_consolidate_index)
        ]
        self._consolidation_stats['records_affected'] = len(consolidation_subset_gdf)
        logger.info(f"Processing {len(consolidation_subset_gdf):,} records across {num_to_consolidate:,} addresses.")

        # This is the core step. It produces a DataFrame mapping each address to its canonical name.
        canonical_map_df = self._build_canonical_mapping(consolidation_subset_gdf)

        # Merge the canonical names back into the original DataFrame.
        # This adds a 'canonical_name' column, which will be null for addresses that didn't need consolidation.
        gdf_with_canonicals = gdf.merge(canonical_map_df, on='addr_normalized_key', how='left')

        # Apply the new names and calculate final statistics.
        consolidated_gdf = self._apply_canonical_mapping(gdf_with_canonicals)

        # --- 5. Finalize and Report ---
        self._consolidation_stats['unique_after'] = int(consolidated_gdf['normalized_text'].nunique())
        self._log_consolidation_summary()

        return consolidated_gdf

    def _build_canonical_mapping(self, consolidation_subset_gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Builds a canonical name mapping using fully vectorized GPU operations.

        This method groups records by address, applies a sophisticated name selection
        function to each group in parallel on the GPU, and returns a DataFrame that
        maps each address to its single chosen canonical name.

        Args:
            consolidation_subset_gdf: A DataFrame containing only the records for
                                      addresses that require consolidation.

        Returns:
            A cuDF DataFrame with two columns: 'addr_normalized_key' and 'canonical_name'.
        """
        # --- Step 1: Group by Address and Collect All Associated Names ---
        # This is the core of the vectorization strategy. We perform a single
        # `groupby` operation on the GPU. For each 'addr_normalized_key', we
        # aggregate all 'normalized_text' values into a list. The result is a
        # Series where the index is the unique address key.
        logger.debug("Grouping names by address and applying canonical selection function on GPU...")
        grouped_names_by_address = consolidation_subset_gdf.groupby('addr_normalized_key')['normalized_text'].agg('collect')

        # --- Step 2: Apply the Canonical Name Function to Each Group ---
        # The `.apply()` method executes a function on each list of names in
        # parallel on the GPU, determining the single best name for that group.
        def select_canonical_name(names_in_group: cudf.Series) -> str:
            """UDF wrapper for the apply call for type clarity."""
            return get_canonical_name_gpu(names_in_group, self.vectorizer_config.similarity_tfidf)

        canonical_names_series = grouped_names_by_address.apply(select_canonical_name)

        # --- Step 3: Log Examples and Format the Final Output ---
        self._log_consolidation_examples(grouped_names_by_address, canonical_names_series)

        # Convert the resulting Series (index='addr_normalized_key', value='canonical_name')
        # into a two-column DataFrame, which is the final mapping table.
        canonical_map_df = canonical_names_series.reset_index(name='canonical_name')
        logger.info(f"Built canonical mapping for {len(canonical_map_df):,} addresses.")

        return canonical_map_df

    def _apply_canonical_mapping(self, gdf_with_canonicals: cudf.DataFrame) -> cudf.DataFrame:
        """
        Applies the canonical name mapping using vectorized GPU operations.

        This method updates the 'normalized_text' column with canonical names where
        available and calculates statistics about the changes made.

        Args:
            gdf_with_canonicals: The DataFrame with the 'canonical_name' column merged in.

        Returns:
            A new DataFrame with consolidated names in the 'normalized_text' column.
        """
        logger.info("Applying canonical name mapping to the dataset...")
        # Create a boolean mask to identify records that have a new canonical name.
        has_canonical_mask = gdf_with_canonicals['canonical_name'].notna()

        # Create a mask to identify where the name will actually change.
        name_changed_mask = has_canonical_mask & (
            gdf_with_canonicals['normalized_text'] != gdf_with_canonicals['canonical_name']
        )
        num_names_changed = int(name_changed_mask.sum())
        self._consolidation_stats['names_changed'] = num_names_changed

        # Use .copy() to avoid modifying the original DataFrame in place.
        result_gdf = gdf_with_canonicals.copy()

        # The core update operation. Where 'canonical_name' is not null, use it;
        # otherwise, keep the existing 'normalized_text'. This is a fast, vectorized update.
        result_gdf['normalized_text'] = result_gdf['canonical_name'].fillna(result_gdf['normalized_text'])

        # Drop the temporary helper column.
        result_gdf = result_gdf.drop(columns=['canonical_name'])
        
        logger.info(f"Applied canonical names: {num_names_changed:,} records had their names changed.")
        return result_gdf

    def _log_consolidation_examples(
        self,
        grouped_names: cudf.Series,
        canonical_names: cudf.Series,
        max_examples: int = 5
    ):
        """Logs a sample of consolidation decisions for debugging and transparency."""
        # This helper function creates a sample of "before and after" data for logging.
        if grouped_names.empty or not logger.isEnabledFor(logging.DEBUG):
            return

        examples_df = cudf.DataFrame({'all_names': grouped_names, 'canonical_name': canonical_names})
        sample_size = min(len(examples_df), max_examples)
        
        if sample_size == 0: return

        # Transfer only the small sample to pandas for easy iteration and logging.
        examples_to_log = examples_df.head(sample_size).to_pandas()

        logger.debug("--- Sample of Address Consolidation Decisions ---")
        for idx, (address, row) in enumerate(examples_to_log.iterrows(), 1):
            name_counts = pd.Series(row['all_names']).value_counts().head(3).to_dict()
            truncated_address = address[:70] + '...' if len(address) > 70 else address
            logger.debug(f"  Example {idx} | Address: {truncated_address}")
            logger.debug(f"    - Names Found:      {name_counts}")
            logger.debug(f"    - Canonical Chosen: '{row['canonical_name']}'")

            # Store the formatted example if a stats dictionary is configured.
            self._consolidation_stats['consolidation_examples'].append({
                'address': truncated_address,
                'names': name_counts,
                'canonical': row['canonical_name']
            })
    
    def _log_consolidation_summary(self) -> None:
        """Log comprehensive summary of consolidation results."""
        stats = self._consolidation_stats
        
        logger.info("=" * 60)
        logger.info("Address Consolidation Summary:")
        logger.info("=" * 60)
        
        if stats.get('total_addresses', 0) > 0:
            logger.info(f"Total unique addresses analyzed: {stats['total_addresses']:,}")
            logger.info(f"Addresses with multiple names: {stats['addresses_with_multiple_names']:,}")
            logger.info(f"Records affected by consolidation: {stats['records_affected']:,}")
            logger.info(f"Names actually changed: {stats['names_changed']:,}")
            
            if stats.get('empty_addresses_skipped', 0) > 0:
                logger.info(f"Records with empty addresses (skipped): {stats['empty_addresses_skipped']:,}")
            
            if stats.get('unique_before', 0) > 0 and stats.get('unique_after', 0) > 0:
                reduction = stats['unique_before'] - stats['unique_after']
                reduction_pct = (reduction / stats['unique_before']) * 100 if stats['unique_before'] > 0 else 0
                logger.info(f"Unique names before: {stats['unique_before']:,}")
                logger.info(f"Unique names after: {stats['unique_after']:,}")
                logger.info(f"Reduction: {reduction:,} names ({reduction_pct:.1f}%)")
            
            # Calculate effectiveness
            if stats['addresses_with_multiple_names'] > 0:
                effectiveness = (stats['names_changed'] / stats['records_affected']) * 100 if stats['records_affected'] > 0 else 0
                logger.info(f"Consolidation effectiveness: {effectiveness:.1f}% of affected records were changed to the canonical name")
        else:
            logger.info("No consolidation performed")
        
        logger.info("=" * 60)
    
    def get_consolidation_stats(self) -> Dict:
        """
        Get statistics from the last consolidation operation.
        
        Returns:
            Dictionary containing consolidation statistics
        """
        return self._consolidation_stats.copy()
    