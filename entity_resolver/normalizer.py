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
        Consolidate multiple entity names sharing the same address.

        For records sharing an address, this method identifies the best
        canonical name to represent all entities at that location. This helps
        resolve cases where the same business operates under slight name
        variations at a single address.

        Args:
            gdf: cuDF DataFrame with 'addr_normalized_key' and 'normalized_text' columns

        Returns:
            cuDF DataFrame with consolidated names in 'normalized_text' column

        Raises:
            KeyError: If required columns are missing
        """
        # Reset consolidation stats
        self._consolidation_stats = {
            'total_addresses': 0,
            'addresses_with_multiple_names': 0,
            'records_affected': 0,
            'names_changed': 0,
            'unique_before': 0,
            'unique_after': 0,
            'empty_addresses_skipped': 0,
            'consolidation_examples': []
        }
        
        # Validate required columns
        required_columns = ['addr_normalized_key', 'normalized_text']
        missing_columns = [col for col in required_columns if col not in gdf.columns]
        if missing_columns:
            raise KeyError(f"Missing required columns: {missing_columns}")
        
        logger.info("=" * 60)
        logger.info("Starting address-based name consolidation")
        logger.info("=" * 60)
        
        # First, check how many addresses are empty or invalid
        empty_addresses = (gdf['addr_normalized_key'] == '') | gdf['addr_normalized_key'].isna()
        empty_count = empty_addresses.sum()
        
        if empty_count > 0:
            logger.warning(f"Found {empty_count:,} records with empty/invalid addresses - these will be skipped")
            self._consolidation_stats['empty_addresses_skipped'] = int(empty_count)
        
        # Filter out empty addresses for consolidation
        valid_address_mask = ~empty_addresses
        valid_gdf = gdf[valid_address_mask]
        
        if len(valid_gdf) == 0:
            logger.warning("No valid addresses found for consolidation")
            return gdf
        
        # Count unique names per address (excluding empty addresses)
        logger.info("Analyzing name variations per address...")
        name_counts_per_address = valid_gdf.groupby('addr_normalized_key')['normalized_text'].nunique()
        
        # Get detailed statistics
        total_unique_addresses = len(name_counts_per_address)
        self._consolidation_stats['total_addresses'] = total_unique_addresses
        
        # Log distribution of name counts
        name_count_distribution = name_counts_per_address.value_counts().sort_index()
        logger.debug("Distribution of name counts per address:")
        distribution_pd = name_count_distribution.head(10).to_pandas()
        for count, freq in distribution_pd.items():
            logger.debug(f"  - {count} names: {freq:,} addresses")
        
        # Identify addresses with multiple names
        addresses_to_consolidate = name_counts_per_address[name_counts_per_address > 1].index
        consolidation_needed = len(addresses_to_consolidate)
        self._consolidation_stats['addresses_with_multiple_names'] = consolidation_needed
        
        if consolidation_needed == 0:
            logger.info("No addresses found with multiple names - skipping consolidation")
            self._log_consolidation_summary()
            return gdf
        
        logger.info(
            f"Found {consolidation_needed:,}/{total_unique_addresses:,} addresses "
            f"({consolidation_needed/total_unique_addresses:.1%}) with multiple names"
        )
        
        # Get statistics on name variations
        name_variations = name_counts_per_address[addresses_to_consolidate]
        max_names = int(name_variations.max())
        avg_names = float(name_variations.mean())
        median_names = float(name_variations.median())
        
        logger.info(f"Name variations statistics:")
        logger.info(f"  - Maximum names at one address: {max_names}")
        logger.info(f"  - Average names per address: {avg_names:.2f}")
        logger.info(f"  - Median names per address: {median_names:.1f}")
        
        # Filter to relevant records
        subset_gdf = gdf[gdf['addr_normalized_key'].isin(addresses_to_consolidate)]
        affected_records = len(subset_gdf)
        self._consolidation_stats['records_affected'] = affected_records
        logger.info(f"Processing {affected_records:,} records for consolidation")
        
        # Track unique names before consolidation
        unique_before = gdf['normalized_text'].nunique()
        self._consolidation_stats['unique_before'] = int(unique_before)
        
        # Build canonical name mapping
        logger.info("Determining canonical names for each address...")
        address_to_canonical_map = self._build_canonical_mapping(
            subset_gdf, 
            addresses_to_consolidate
        )
        
        if not address_to_canonical_map:
            logger.warning("No canonical names determined - returning original DataFrame")
            self._log_consolidation_summary()
            return gdf
        
        # Apply consolidation
        logger.info("Applying canonical name mapping...")
        gdf = self._apply_canonical_mapping(gdf, address_to_canonical_map)
        
        # Track unique names after consolidation
        unique_after = gdf['normalized_text'].nunique()
        self._consolidation_stats['unique_after'] = int(unique_after)
        
        # Log summary
        self._log_consolidation_summary()
        
        return gdf
    
    def _build_canonical_mapping(
        self, 
        subset_gdf: cudf.DataFrame, 
        addresses_to_consolidate: cudf.Index
    ) -> Dict[str, str]:
        """
        Build mapping from addresses to canonical names.
        
        Args:
            subset_gdf: DataFrame filtered to addresses needing consolidation
            addresses_to_consolidate: Index of addresses with multiple names
            
        Returns:
            Dictionary mapping address keys to canonical names
        """
        address_to_canonical_map = {}
        
        # Convert to list for processing
        addresses_list = addresses_to_consolidate.to_pandas()
        total_addresses = len(addresses_list)
        
        # Process in batches with progress tracking
        batch_size = 1000
        examples_collected = 0
        max_examples = 10
        
        for i in range(0, total_addresses, batch_size):
            batch = addresses_list[i:i+batch_size]
            
            for addr_key in batch:
                # Get all names at this address
                names_at_address = subset_gdf[
                    subset_gdf['addr_normalized_key'] == addr_key
                ]['normalized_text']
                
                # Skip if no names found (shouldn't happen but be safe)
                if len(names_at_address) == 0:
                    continue
                
                # Get unique names and their frequencies
                name_counts = names_at_address.value_counts()
                
                # Collect examples for logging
                if examples_collected < max_examples and len(name_counts) > 1:
                    example = {
                        'address': addr_key[:50] + '...' if len(addr_key) > 50 else addr_key,
                        'names': name_counts.head(3).to_pandas().to_dict(),
                        'canonical': None  # Will be set below
                    }
                    
                    # Determine canonical name
                    canonical_name = get_canonical_name_gpu(names_at_address, self.vectorizer_config.similarity_tfidf)
                    example['canonical'] = canonical_name
                    
                    self._consolidation_stats['consolidation_examples'].append(example)
                    examples_collected += 1
                else:
                    # Just determine canonical name without logging
                    canonical_name = get_canonical_name_gpu(names_at_address, self.vectorizer_config.similarity_tfidf)
                
                address_to_canonical_map[addr_key] = canonical_name
            
            # Progress logging
            if (i + batch_size) % 5000 == 0 or (i + batch_size) >= total_addresses:
                progress = min(i + batch_size, total_addresses)
                logger.debug(f"  - Processed {progress:,}/{total_addresses:,} addresses ({progress/total_addresses:.1%})")
        
        logger.info(f"Built canonical mapping for {len(address_to_canonical_map):,} addresses")
        
        # Log examples
        if self._consolidation_stats['consolidation_examples']:
            logger.debug("Sample consolidation decisions:")
            for idx, example in enumerate(self._consolidation_stats['consolidation_examples'][:5], 1):
                logger.debug(f"  Example {idx}:")
                logger.debug(f"    Address: {example['address']}")
                logger.debug(f"    Names found: {example['names']}")
                logger.debug(f"    Canonical chosen: {example['canonical']}")
        
        return address_to_canonical_map
    
    def _apply_canonical_mapping(
        self, 
        gdf: cudf.DataFrame, 
        address_to_canonical_map: Dict[str, str]
    ) -> cudf.DataFrame:
        """
        Apply canonical name mapping to DataFrame.
        
        Args:
            gdf: Original DataFrame
            address_to_canonical_map: Mapping from addresses to canonical names
            
        Returns:
            DataFrame with consolidated names
        """
        # Create mapping DataFrame for efficient merge
        consolidation_map_gdf = cudf.DataFrame({
            'addr_normalized_key': list(address_to_canonical_map.keys()),
            'canonical_name': list(address_to_canonical_map.values())
        })

        # Store original names as a temporary column to prevent index misalignment after merge
        gdf['original_names_temp'] = gdf['normalized_text']
        
        # Merge and update normalized_text
        gdf = gdf.merge(consolidation_map_gdf, on='addr_normalized_key', how='left')
        gdf['normalized_text'] = gdf['canonical_name'].fillna(gdf['normalized_text'])

        affected_mask = gdf['canonical_name'].notna()

        # Authoritative denominator: rows that actually had a canonical available
        self._consolidation_stats['records_affected'] = int(affected_mask.sum())
        
        # Count changes
        names_changed = (gdf['normalized_text'] != gdf['original_names_temp']).sum()
        self._consolidation_stats['names_changed'] = int(names_changed)
        
        # Clean up temporary column
        gdf = gdf.drop(columns=['canonical_name', 'original_names_temp'])
        
        return gdf
    
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
    