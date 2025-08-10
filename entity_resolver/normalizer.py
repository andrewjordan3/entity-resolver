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
from .config import NormalizationConfig
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
    """
    
    def __init__(self, config: NormalizationConfig):
        """
        Initialize the TextNormalizer with configuration settings.

        Args:
            config: NormalizationConfig object containing replacement rules,
                   suffixes to remove, and other normalization parameters
        """
        self.config = config
        self._compiled_patterns = self._compile_regex_patterns()
        
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
        # Validate required columns
        required_columns = ['addr_normalized_key', 'normalized_text']
        missing_columns = [col for col in required_columns if col not in gdf.columns]
        if missing_columns:
            raise KeyError(f"Missing required columns: {missing_columns}")
        
        logger.info("Starting address-based name consolidation...")
        
        # Count unique names per address
        logger.debug("Analyzing name variations per address...")
        name_counts_per_address = gdf.groupby('addr_normalized_key')['normalized_text'].nunique()
        
        # Identify addresses with multiple names
        addresses_to_consolidate = name_counts_per_address[name_counts_per_address > 1].index
        
        if len(addresses_to_consolidate) == 0:
            logger.info("No addresses found with multiple names - skipping consolidation")
            return gdf
        
        total_addresses = len(name_counts_per_address)
        consolidation_needed = len(addresses_to_consolidate)
        
        logger.info(
            f"Found {consolidation_needed:,}/{total_addresses:,} addresses "
            f"({consolidation_needed/total_addresses:.1%}) requiring name consolidation"
        )
        
        # Get statistics on name variations
        max_names = name_counts_per_address[addresses_to_consolidate].max()
        avg_names = name_counts_per_address[addresses_to_consolidate].mean()
        logger.debug(f"Name variations per address: max={max_names}, avg={avg_names:.2f}")
        
        # Filter to relevant records
        subset_gdf = gdf[gdf['addr_normalized_key'].isin(addresses_to_consolidate)]
        affected_records = len(subset_gdf)
        logger.debug(f"Processing {affected_records:,} records for consolidation")
        
        # Build canonical name mapping
        logger.debug("Determining canonical names for each address...")
        address_to_canonical_map = {}
        
        # Process in batches for better progress tracking
        addresses_list = addresses_to_consolidate.to_pandas()
        batch_size = 1000
        
        for i in range(0, len(addresses_list), batch_size):
            batch = addresses_list[i:i+batch_size]
            
            for addr_key in batch:
                # Get all names at this address
                name_series = subset_gdf[subset_gdf['addr_normalized_key'] == addr_key]['normalized_text']
                
                # Determine best canonical name
                canonical_name = utils.get_canonical_name_gpu(name_series)
                address_to_canonical_map[addr_key] = canonical_name
            
            if (i + batch_size) % 5000 == 0:
                logger.debug(f"  - Processed {min(i + batch_size, len(addresses_list)):,}/{len(addresses_list):,} addresses")
        
        if not address_to_canonical_map:
            logger.warning("No canonical names determined - returning original DataFrame")
            return gdf
        
        # Create mapping DataFrame for efficient merge
        logger.debug("Applying canonical name mapping...")
        consolidation_map_gdf = cudf.DataFrame({
            'addr_normalized_key': list(address_to_canonical_map.keys()),
            'canonical_name': list(address_to_canonical_map.values())
        })
        
        # Merge and update normalized_text
        original_col = gdf['normalized_text'].copy()
        gdf = gdf.merge(consolidation_map_gdf, on='addr_normalized_key', how='left')
        gdf['normalized_text'] = gdf['canonical_name'].fillna(gdf['normalized_text'])
        
        # Calculate statistics
        names_changed = (gdf['normalized_text'] != original_col).sum()
        final_unique = gdf['normalized_text'].nunique()
        
        logger.info(
            f"Consolidation complete: "
            f"{names_changed:,} records updated, "
            f"{final_unique:,} unique names remain"
        )
        
        # Clean up temporary column
        return gdf.drop(columns=['canonical_name'])