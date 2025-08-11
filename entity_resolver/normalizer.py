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
import cupy as cp

# Local Package Imports
from .config import NormalizationConfig, VectorizerConfig
from . import utils

# Set up module-level logger
logger = logging.getLogger(__name__)


class TextNormalizer:
    """
    Handle cleaning and standardization of entity names for resolution.
    
    This class implements a comprehensive text normalization pipeline that
    prepares entity names for vectorization and matching. It handles common
    variations in business names, removes noise, and consolidates entities
    that share addresses.
    
    Attributes:
        config (NormalizationConfig): Configuration for normalization rules
        _compiled_patterns (Dict): Pre-compiled regex patterns for efficiency
        _consolidation_stats (Dict): Statistics from last consolidation operation
    """
    
    def __init__(self, config: NormalizationConfig, vectorizer_config: VectorizerConfig):
        """
        Initialize the TextNormalizer with configuration settings.

        Args:
            config: NormalizationConfig object containing replacement rules,
                   suffixes to remove, and other normalization parameters
        """
        self.config = config
        self.vectorizer_config = vectorizer_config
        self._compiled_patterns = self._compile_regex_patterns()
        self._consolidation_stats = {}
        
        logger.info("Initialized TextNormalizer")
        logger.debug(f"Replacements configured: {len(config.replacements)}")
        logger.debug(f"Suffixes to remove: {len(config.suffixes_to_remove)}")
    
    def _compile_regex_patterns(self) -> Dict[str, str]:
        """
        Pre-compile regex patterns for better performance.
        
        Compiling patterns once during initialization avoids repeated
        compilation during processing of large datasets.
        
        Returns:
            Dictionary of pattern names to compiled regex strings
        """
        patterns = {}
        
        # Business name qualifier pattern (dba, fka, aka)
        patterns['dba'] = (
            r'(?:\s|^)(?:'
            r'd(?:/|\s)?b(?:/|\s)?a|'  # matches dba, d/b/a, d b a
            r'f(?:/|\s)?k(?:/|\s)?a|'  # matches fka, f/k/a, f k a
            r'a(?:/|\s)?k(?:/|\s)?a'   # matches aka, a/k/a, a k a
            r')\s+(.*)'
        )
        
        # Suffix removal pattern
        if self.config.suffixes_to_remove:
            escaped_suffixes = [re.escape(s) for s in self.config.suffixes_to_remove]
            patterns['suffixes'] = r'\b(' + '|'.join(escaped_suffixes) + r')\b'
            logger.debug(f"Compiled suffix pattern for {len(self.config.suffixes_to_remove)} suffixes")
        
        # Compile replacement patterns with word boundaries
        for old_word in self.config.replacements:
            pattern_key = f'replacement_{old_word}'
            patterns[pattern_key] = r'\b' + re.escape(old_word) + r'\b'
        
        return patterns

    def normalize_text(self, gdf: cudf.DataFrame, entity_col: str) -> cudf.DataFrame:
        """
        Apply comprehensive normalization rules to entity name column.

        This GPU-accelerated process performs multiple cleaning steps including
        lowercasing, separator handling, suffix removal, and standardization
        to create a consistent 'normalized_text' column.

        Normalization steps:
        1. Basic cleaning (lowercase, separators)
        2. Remove parenthetical content
        3. Handle business qualifiers (dba, fka, aka)
        4. Apply custom word replacements
        5. Remove legal/organizational suffixes
        6. Final cleanup (punctuation, numbers, whitespace)

        Args:
            gdf: Input cuDF DataFrame
            entity_col: Name of column containing entity names to normalize

        Returns:
            cuDF DataFrame with added 'normalized_text' column

        Raises:
            KeyError: If entity_col doesn't exist in DataFrame
            ValueError: If normalization fails
        """
        # Validate input
        if entity_col not in gdf.columns:
            raise KeyError(f"Column '{entity_col}' not found in DataFrame")
        
        initial_count = len(gdf)
        logger.info(f"Starting text normalization for {initial_count:,} records in '{entity_col}'")
        
        # Start with raw entity names, handling nulls
        normalized_series = gdf[entity_col].fillna('').astype(str)
        
        # Track unique values before normalization for statistics
        initial_unique = normalized_series.nunique()
        logger.debug(f"Initial unique values: {initial_unique:,}")

        # Step 1: Basic cleaning - lowercase and common separators
        logger.debug("Step 1: Applying basic cleaning (lowercase, separators)")
        normalized_series = normalized_series.str.lower()
        normalized_series = normalized_series.str.replace('&', ' and ', n=-1)
        normalized_series = normalized_series.str.replace('+', ' and ', n=-1)
        
        # Count how many records were affected
        affected_by_separators = (normalized_series != gdf[entity_col].str.lower()).sum()
        logger.debug(f"  - Separator replacements affected {affected_by_separators:,} records")

        # Step 2: Remove parenthetical content
        logger.debug("Step 2: Removing parenthetical content")
        before_parens = normalized_series.copy()
        normalized_series = normalized_series.str.replace(r'\([^)]*\)', '')
        affected_by_parens = (normalized_series != before_parens).sum()
        logger.debug(f"  - Removed parentheses from {affected_by_parens:,} records")

        # Step 3: Handle business name qualifiers (dba, fka, aka)
        logger.debug("Step 3: Processing business name qualifiers (dba/fka/aka)")
        dba_pattern = self._compiled_patterns['dba']
        extracted_names = normalized_series.str.extract(dba_pattern)
        
        # Count records with qualifiers
        has_qualifier = extracted_names[0].notna()
        qualifier_count = has_qualifier.sum()
        
        # Use extracted name where qualifier found, otherwise keep original
        normalized_series = extracted_names[0].fillna(normalized_series)
        logger.debug(f"  - Found and processed qualifiers in {qualifier_count:,} records")

        # Step 4: Apply custom word replacements
        if self.config.replacements:
            logger.debug(f"Step 4: Applying {len(self.config.replacements)} custom replacements")
            replacement_counts = {}
            
            for old_word, new_word in self.config.replacements.items():
                pattern = self._compiled_patterns[f'replacement_{old_word}']
                before_replacement = normalized_series.copy()
                normalized_series = normalized_series.str.replace(pattern, new_word)
                
                # Track how many records were affected by each replacement
                affected = (normalized_series != before_replacement).sum()
                if affected > 0:
                    replacement_counts[old_word] = affected
            
            if replacement_counts:
                logger.debug(f"  - Top replacements: {dict(sorted(replacement_counts.items(), key=lambda x: x[1], reverse=True)[:5])}")

        # Step 5: Remove legal and organizational suffixes
        if self.config.suffixes_to_remove:
            logger.debug(f"Step 5: Removing {len(self.config.suffixes_to_remove)} suffixes")
            suffix_pattern = self._compiled_patterns['suffixes']
            before_suffix = normalized_series.copy()
            normalized_series = normalized_series.str.replace(suffix_pattern, '')
            affected_by_suffix = (normalized_series != before_suffix).sum()
            logger.debug(f"  - Removed suffixes from {affected_by_suffix:,} records")

        # Step 6: Final cleanup
        logger.debug("Step 6: Final cleanup (punctuation, trailing numbers, whitespace)")
        
        # Remove non-alphanumeric characters
        normalized_series = normalized_series.str.replace(r'[^\w\s]', ' ')
        
        # Remove trailing numbers
        normalized_series = normalized_series.str.replace(r'\s+\d+$', '')
        
        # Normalize whitespace
        normalized_series = normalized_series.str.normalize_spaces()
        normalized_series = normalized_series.str.strip()
        
        # Handle empty strings after normalization
        empty_count = (normalized_series == '').sum()
        if empty_count > 0:
            logger.warning(f"  - {empty_count:,} records became empty after normalization")
            normalized_series = normalized_series.replace('', 'UNKNOWN')

        # Assign normalized text to DataFrame
        gdf['normalized_text'] = normalized_series
        
        # Log final statistics
        final_unique = gdf['normalized_text'].nunique()
        reduction_ratio = 1 - (final_unique / initial_unique) if initial_unique > 0 else 0
        
        logger.info(
            f"Text normalization complete: "
            f"{initial_unique:,} -> {final_unique:,} unique values "
            f"({reduction_ratio:.1%} reduction)"
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
        for count, freq in name_count_distribution.head(10).items():
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
                    canonical_name = utils.get_canonical_name_gpu(names_at_address, self.vectorizer_config.similarity_tfidf)
                    example['canonical'] = canonical_name
                    
                    self._consolidation_stats['consolidation_examples'].append(example)
                    examples_collected += 1
                else:
                    # Just determine canonical name without logging
                    canonical_name = utils.get_canonical_name_gpu(names_at_address, self.vectorizer_config.similarity_tfidf)
                
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
        
        # Store original names for comparison
        original_names = gdf['normalized_text'].copy()
        
        # Merge and update normalized_text
        gdf = gdf.merge(consolidation_map_gdf, on='addr_normalized_key', how='left')
        gdf['normalized_text'] = gdf['canonical_name'].fillna(gdf['normalized_text'])
        
        # Count changes
        names_changed = (gdf['normalized_text'] != original_names).sum()
        self._consolidation_stats['names_changed'] = int(names_changed)
        
        # Clean up temporary column
        gdf = gdf.drop(columns=['canonical_name'])
        
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
                logger.info(f"Consolidation effectiveness: {effectiveness:.1f}% of affected records were consolidated")
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
