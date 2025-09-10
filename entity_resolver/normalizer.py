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
    
    This class implements a comprehensive text normalization pipeline that
    prepares entity names for vectorization and matching. It handles common
    variations in business names, removes noise, and consolidates entities
    that share addresses. It is specifically optimized for use with the
    rapids.ai cuDF library.

    Attributes:
        config (NormalizationConfig): Configuration for normalization rules
        vectorizer_config (VectorizerConfig): Configuration for vectorizer settings
        _compiled_patterns (Dict): Pre-compiled regex patterns for efficiency
        _consolidation_stats (Dict): Statistics from last consolidation operation
    """
    
    def __init__(self, config: NormalizationConfig, vectorizer_config: VectorizerConfig):
        """
        Initialize the TextNormalizer with configuration settings.

        Args:
            config: NormalizationConfig object containing replacement rules,
                    suffixes to remove, and other normalization parameters.
            vectorizer_config: VectorizerConfig object for downstream processing.
        """
        self.config = config
        self.vectorizer_config = vectorizer_config
        self._compiled_patterns = self._compile_regex_patterns()
        self._consolidation_stats = {}
        
        logger.info("Initialized TextNormalizer")
        logger.debug(f"Replacements configured: {len(config.replacements)}")
        logger.debug(f"Suffixes to remove: {len(config.suffixes_to_remove)}")


    def _compile_regex_patterns(self) -> Dict[str, re.Pattern]:
        """
        Pre-compile all regex patterns and define cuDF-compatible replacements.

        Pre-compiling patterns during initialization avoids repeated compilation
        when processing large datasets. This method also formats replacement strings
        with backslashes (e.g., \\1, \\2) for compatibility with cuDF's stricter
        regex engine.

        Returns:
            Dictionary mapping pattern names to compiled regex pattern objects and
            their corresponding replacement strings.
            
        Note:
            While cuDF uses Numba/CUDA for regex operations, pre-compiling patterns
            still provides benefits by validating patterns early and organizing them
            for maintainability.
        """
        patterns = {}

        logger.debug("Pre-compiling regex patterns for cuDF-based text normalization")

        # Business abbreviation patterns
        # These expand common abbreviations to their full forms. The replacement
        # strings use the '\\1', '\\2' backreference syntax required by cuDF.
        # Pattern: (boundary_start)abbrev.?(boundary_end) -> \1expansion\2
        business_abbreviations = {
            'corp': (r'(^|[^a-zA-Z0-9_])corp\.?($|[^a-zA-Z0-9_])', r'\1corporation\2'),
            'inc': (r'(^|[^a-zA-Z0-9_])inc\.?($|[^a-zA-Z0-9_])', r'\1incorporated\2'),
            'ltd': (r'(^|[^a-zA-Z0-9_])ltd\.?($|[^a-zA-Z0-9_])', r'\1limited\2'),
            'llc': (r'(^|[^a-zA-Z0-9_])llc\.?($|[^a-zA-Z0-9_])', r'\1limited liability company\2'),
            'co': (r'(^|[^a-zA-Z0-9_])co\.?($|[^a-zA-Z0-9_])', r'\1company\2'),
            'assoc': (r'(^|[^a-zA-Z0-9_])assoc\.?($|[^a-zA-Z0-9_])', r'\1associates\2'),
            'mfg': (r'(^|[^a-zA-Z0-9_])mfg\.?($|[^a-zA-Z0-9_])', r'\1manufacturing\2'),
            'intl': (r'(^|[^a-zA-Z0-9_])intl\.?($|[^a-zA-Z0-9_])', r'\1international\2'),
            'dist': (r'(^|[^a-zA-Z0-9_])dist\.?($|[^a-zA-Z0-9_])', r'\1distribution\2'),
            'svcs': (r'(^|[^a-zA-Z0-9_])svcs?\.?($|[^a-zA-Z0-9_])', r'\1services\2'),
            'mgmt': (r'(^|[^a-zA-Z0-9_])mgmt\.?($|[^a-zA-Z0-9_])', r'\1management\2'),
            'grp': (r'(^|[^a-zA-Z0-9_])grp\.?($|[^a-zA-Z0-9_])', r'\1group\2')
        }

        for abbrev_key, (pattern_str, expansion) in business_abbreviations.items():
            pattern_name = f'abbrev_{abbrev_key}'
            patterns[pattern_name] = re.compile(pattern_str, re.IGNORECASE)
            patterns[f'{pattern_name}_replacement'] = expansion

        logger.debug(f"  - Compiled {len(business_abbreviations)} business abbreviation patterns")

        # Enhanced business name qualifier pattern (dba, fka, aka, etc.)
        patterns['business_qualifier'] = re.compile(
            r'(?:\s|^)(?:'
            r'd(?:\s*[/.\-]\s*)?b(?:\s*[/.\-]\s*)?a|'    # dba variations
            r'f(?:\s*[/.\-]\s*)?k(?:\s*[/.\-]\s*)?a|'    # fka variations
            r'a(?:\s*[/.\-]\s*)?k(?:\s*[/.\-]\s*)?a|'    # aka variations
            r't(?:\s*[/.\-]\s*)?a|'                      # ta (trading as)
            r'formerly|'                                # formerly
            r'now\s+known\s+as|'                        # now known as
            r'trading\s+as|'                            # trading as
            r'doing\s+business\s+as'                    # doing business as
            r')(?:\s*:)?\s+(.*?)(?:\s*$)',               # Capture group for the actual name
            re.IGNORECASE
        )
        logger.debug("  - Compiled business qualifier pattern (dba/fka/aka/ta/formerly/etc.)")

        # Separator and conjunction patterns
        separator_patterns = {
            'ampersand': (r'&+', ' and '),
            'plus': (r'[+]+', ' and '),
            'n_word': (r'(^|[^a-zA-Z0-9_])n($|[^a-zA-Z0-9_])', r'\1 and \2'),  # Uses backreferences
            'forward_slash': (r'/', ' '),
            'backslash': (r'\\', ' '),
            'pipe': (r'\|', ' '),
            'middle_dot': (r'·', ' '),
            'bullet': (r'•', ' '),
            'dashes': (r'–|—', ' ')
        }

        for sep_key, (pattern_str, replacement) in separator_patterns.items():
            pattern_name = f'separator_{sep_key}'
            patterns[pattern_name] = re.compile(pattern_str, re.IGNORECASE if sep_key == 'n_word' else 0)
            patterns[f'{pattern_name}_replacement'] = replacement

        logger.debug(f"  - Compiled {len(separator_patterns)} separator patterns")

        # Noise removal patterns - parenthetical and quoted content
        patterns['parenthetical'] = re.compile(r'\([^)]*\)')
        patterns['bracketed'] = re.compile(r'\[[^\]]*\]')
        patterns['double_quoted'] = re.compile(r'"[^"]*"')
        patterns['single_quoted'] = re.compile(r"'[^']*'")
        logger.debug("  - Compiled noise removal patterns (parenthetical/bracketed/quoted)")

        # OCR error correction patterns, updated for cuDF backreferences
        ocr_patterns = {
            'zero_in_word': (r'([a-z])0([a-z])', r'\1o\2'),
            'one_in_word':  (r'([a-z])1([a-z])', r'\1l\2'),
            'five_in_word': (r'([a-z])5([a-z])', r'\1s\2')
        }

        for ocr_key, (pattern_str, replacement) in ocr_patterns.items():
            pattern_name = f'ocr_{ocr_key}'
            patterns[pattern_name] = re.compile(pattern_str, re.IGNORECASE)
            patterns[f'{pattern_name}_replacement'] = replacement

        logger.debug(f"  - Compiled {len(ocr_patterns)} OCR correction patterns")

        # Final cleanup patterns
        patterns['possessive_s'] = re.compile(r"'s($|[^a-zA-Z0-9_])", re.IGNORECASE)
        patterns['possessive_plural'] = re.compile(r"s'($|[^a-zA-Z0-9_])", re.IGNORECASE)
        patterns['non_alphanumeric'] = re.compile(r'[^\w\s]')
        patterns['trailing_numbers'] = re.compile(r'\s+\d+$')
        patterns['single_chars'] = re.compile(r'(^|[^a-zA-Z0-9_])([b-hj-z])($|[^a-zA-Z0-9_])', re.IGNORECASE)
        patterns['multiple_spaces'] = re.compile(r'\s+')
        logger.debug("  - Compiled final cleanup patterns")

        # Compile suffix removal pattern if suffixes are configured
        if self.config.suffixes_to_remove:
            escaped_suffixes = [re.escape(suffix) for suffix in self.config.suffixes_to_remove]
            suffix_pattern_str = r'(^|[^a-zA-Z0-9_])(?:' + '|'.join(escaped_suffixes) + r')($|[^a-zA-Z0-9_])'
            patterns['suffix_removal'] = re.compile(suffix_pattern_str, re.IGNORECASE)
            logger.debug(f"  - Compiled suffix removal pattern for {len(self.config.suffixes_to_remove)} suffixes")

        # Compile custom replacement patterns from configuration
        if self.config.replacements:
            for old_word, new_word in self.config.replacements.items():
                pattern_key = f'custom_replacement_{old_word}'
                pattern_str = r'(^|[^a-zA-Z0-9_])' + re.escape(old_word) + r'($|[^a-zA-Z0-9_])'
                patterns[pattern_key] = re.compile(pattern_str, re.IGNORECASE)
                # Store the cuDF-compatible replacement text
                patterns[f'{pattern_key}_replacement'] = r'\1' + new_word + r'\2'

            logger.debug(f"  - Compiled {len(self.config.replacements)} custom replacement patterns")

        # Validation pattern - check for remaining special characters
        patterns['validation_special_chars'] = re.compile(r'[^a-z0-9\s]')

        total_patterns = len([k for k in patterns.keys() if not k.endswith('_replacement')])
        logger.info(f"Successfully pre-compiled {total_patterns} regex patterns for normalization")

        return patterns

    def normalize_text(self, gdf: cudf.DataFrame, entity_col: str) -> cudf.DataFrame:
        """
        Apply comprehensive normalization rules to entity name column using cuDF.

        This GPU-accelerated process performs multiple cleaning steps optimized for
        business entity matching. It uses cuDF's string methods, including
        `str.replace_with_backrefs` for complex replacements, to create a
        consistent 'normalized_text' column.

        Normalization pipeline:
        1. Unicode normalization (NFKC)
        2. Basic cleaning (lowercase, whitespace)
        3. Business abbreviation expansion
        4. Separator and symbol standardization
        5. Noise removal (parenthetical content)
        6. Business qualifier handling (dba, fka, etc.)
        7. Custom word replacements
        8. Legal/organizational suffix removal
        9. Common OCR/data entry error correction
        10. Final cleanup and validation

        Args:
            gdf: Input cuDF DataFrame containing entity data.
            entity_col: Name of column with entity names to normalize.

        Returns:
            cuDF DataFrame with an added 'normalized_text' column.

        Raises:
            KeyError: If entity_col doesn't exist in the DataFrame.
            ValueError: If normalization fails or produces invalid output.
        """
        if entity_col not in gdf.columns:
            raise KeyError(f"Column '{entity_col}' not found in DataFrame")

        initial_record_count = len(gdf)
        logger.info(f"Starting text normalization for {initial_record_count:,} records in '{entity_col}'")

        normalized_series = gdf[entity_col].fillna('').astype('str')

        initial_unique_count = normalized_series.nunique()
        initial_avg_length = normalized_series.str.len().mean()
        logger.debug(f"Initial stats - Unique: {initial_unique_count:,}, Avg length: {initial_avg_length:.1f}")

        # Step 1: Unicode normalization (NFKC form)
        logger.debug("Step 1: Applying Unicode normalization (NFKC)")
        normalized_series = nfkc_normalize_series(normalized_series)

        # Step 2: Convert to lowercase
        logger.debug("Step 2: Converting to lowercase")
        normalized_series = normalized_series.str.lower()

        # Step 3: Expand common business abbreviations
        logger.debug("Step 3: Expanding business abbreviations")
        abbreviation_replacement_count = 0
        abbreviation_keys = ['corp', 'inc', 'ltd', 'llc', 'co', 'assoc', 'mfg', 'intl', 'dist', 'svcs', 'mgmt', 'grp']

        for abbrev_key in abbreviation_keys:
            pattern_name = f'abbrev_{abbrev_key}'
            if pattern_name in self._compiled_patterns:
                pattern_str = self._compiled_patterns[pattern_name].pattern
                replacement = self._compiled_patterns[f'{pattern_name}_replacement']

                before_expansion = normalized_series.copy()
                # Use replace_with_backrefs as replacements contain \1, \2
                normalized_series = normalized_series.str.replace_with_backrefs(pattern_str, replacement, regex=True)
                affected_records = (normalized_series != before_expansion).sum()
                if affected_records > 0:
                    abbreviation_replacement_count += affected_records
                    display_replacement = re.sub(r'\\\d', '', replacement)
                    logger.debug(f"  - Expanded '{abbrev_key}' to '{display_replacement}' in {affected_records:,} records")
        
        if abbreviation_replacement_count > 0:
            logger.info(f"  - Total abbreviation expansions: {abbreviation_replacement_count:,}")

        # Step 4: Standardize separators and conjunctions
        logger.debug("Step 4: Standardizing separators and conjunctions")
        separator_keys = ['ampersand', 'plus', 'n_word', 'forward_slash', 'backslash',
                          'pipe', 'middle_dot', 'bullet', 'dashes']

        for sep_key in separator_keys:
            pattern_name = f'separator_{sep_key}'
            if pattern_name in self._compiled_patterns:
                pattern_str = self._compiled_patterns[pattern_name].pattern
                replacement = self._compiled_patterns[f'{pattern_name}_replacement']
                # Conditionally use replace_with_backrefs only if needed
                if '\\' in replacement:
                    normalized_series = normalized_series.str.replace_with_backrefs(pattern_str, replacement, regex=True)
                else:
                    normalized_series = normalized_series.str.replace(pattern_str, replacement, regex=True)

        # Step 5: Remove noise - parenthetical content and special annotations
        logger.debug("Step 5: Removing parenthetical content and annotations")
        before_parenthetical_removal = normalized_series.copy()
        normalized_series = normalized_series.str.replace(self._compiled_patterns['parenthetical'].pattern, ' ', regex=True)
        affected_by_parentheses = (normalized_series != before_parenthetical_removal).sum()
        logger.debug(f"  - Removed parenthetical content from {affected_by_parentheses:,} records")

        normalized_series = normalized_series.str.replace(self._compiled_patterns['bracketed'].pattern, ' ', regex=True)
        normalized_series = normalized_series.str.replace(self._compiled_patterns['double_quoted'].pattern, ' ', regex=True)
        normalized_series = normalized_series.str.replace(self._compiled_patterns['single_quoted'].pattern, ' ', regex=True)

        # Step 6: Handle business name qualifiers (dba, fka, aka, etc.)
        logger.debug("Step 6: Processing business name qualifiers")
        qualifier_pattern = self._compiled_patterns['business_qualifier'].pattern
        # extract does not need backreference changes
        extracted_business_names = normalized_series.str.extract(qualifier_pattern, expand=False)
        qualifier_count = extracted_business_names.notna().sum()
        normalized_series = extracted_business_names.fillna(normalized_series)
        logger.info(f"  - Processed business qualifiers in {qualifier_count:,} records")

        # Step 7: Apply custom word replacements from configuration
        if self.config.replacements:
            logger.debug(f"Step 7: Applying {len(self.config.replacements)} custom replacements")
            for old_word, new_word in self.config.replacements.items():
                pattern_key = f'custom_replacement_{old_word}'
                if pattern_key in self._compiled_patterns:
                    pattern_str = self._compiled_patterns[pattern_key].pattern
                    replacement_text = self._compiled_patterns[f'{pattern_key}_replacement']
                    # Use replace_with_backrefs due to \1, \2 in replacement
                    normalized_series = normalized_series.str.replace_with_backrefs(pattern_str, replacement_text, regex=True)

        # Step 8: Remove legal and organizational suffixes
        if self.config.suffixes_to_remove and 'suffix_removal' in self._compiled_patterns:
            logger.debug(f"Step 8: Removing {len(self.config.suffixes_to_remove)} legal/org suffixes")
            suffix_pattern = self._compiled_patterns['suffix_removal'].pattern
            # Use replace_with_backrefs for the replacement string r'\1 \2'
            normalized_series = normalized_series.str.replace_with_backrefs(suffix_pattern, r'\1 \2', regex=True)

        # Step 9: Handle common OCR and data entry errors
        logger.debug("Step 9: Correcting common OCR/data entry errors")
        ocr_keys = ['zero_in_word', 'one_in_word', 'five_in_word']
        for ocr_key in ocr_keys:
            pattern_name = f'ocr_{ocr_key}'
            if pattern_name in self._compiled_patterns:
                pattern_str = self._compiled_patterns[pattern_name].pattern
                replacement = self._compiled_patterns[f'{pattern_name}_replacement']
                # Use replace_with_backrefs for OCR corrections
                normalized_series = normalized_series.str.replace_with_backrefs(pattern_str, replacement, regex=True)

        # Step 10: Final cleanup and validation
        logger.debug("Step 10: Final cleanup and validation")
        normalized_series = normalized_series.str.replace(self._compiled_patterns['possessive_s'].pattern, '', regex=True)
        normalized_series = normalized_series.str.replace(self._compiled_patterns['possessive_plural'].pattern, 's', regex=True)
        normalized_series = normalized_series.str.replace(self._compiled_patterns['non_alphanumeric'].pattern, ' ', regex=True)
        normalized_series = normalized_series.str.replace(self._compiled_patterns['trailing_numbers'].pattern, '', regex=True)
        normalized_series = normalized_series.str.replace(self._compiled_patterns['single_chars'].pattern, ' ', regex=True)
        normalized_series = normalized_series.str.replace(self._compiled_patterns['multiple_spaces'].pattern, ' ', regex=True)
        normalized_series = normalized_series.str.strip()

        # Handle edge cases - empty strings after normalization
        empty_mask = (normalized_series == '') | normalized_series.isna()
        empty_count = empty_mask.sum()
        if empty_count > 0:
            logger.warning(f"  - {empty_count:,} records became empty after normalization")
            normalized_series[empty_mask] = 'UNKNOWN_ENTITY'

        # Assign normalized text to DataFrame
        gdf['normalized_text'] = normalized_series

        # Calculate and log final statistics
        final_unique_count = gdf['normalized_text'].nunique()
        unique_reduction_ratio = 1 - (final_unique_count / initial_unique_count) if initial_unique_count > 0 else 0
        logger.info(
            f"Text normalization complete: "
            f"Unique values reduced from {initial_unique_count:,} to {final_unique_count:,} ({unique_reduction_ratio:.1%} reduction)"
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
    