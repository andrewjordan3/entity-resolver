# entity_resolver/address_processor.py
"""
Address processing pipeline for parsing, cleaning, and consolidating addresses.

This module handles the complete address processing workflow:
1. Combining multiple raw address columns into a single string
2. Parsing addresses into structured components using libpostal
3. Normalizing and generating address keys for matching
4. Consolidating near-duplicate addresses (training only)
"""

import logging
from typing import Optional, Dict, List

import cudf
import re
import pandas as pd

# Local Package Imports
from .config import ValidationConfig, ColumnConfig, VectorizerConfig
from . import utils

# Set up module-level logger
logger = logging.getLogger(__name__)


class AddressProcessor:
    """
    Orchestrate parsing, cleaning, and consolidation of address data.
    
    This class manages the complete address processing pipeline, including
    parsing with libpostal, normalization, and similarity-based consolidation
    of near-duplicate addresses.
    
    Attributes:
        validation_config: Thresholds and validation settings
        column_config: Column names for input address parts
        vectorizer_config: Parameters for similarity computation
    """

    def __init__(
        self,
        validation_config: ValidationConfig,
        column_config: ColumnConfig,
        vectorizer_config: VectorizerConfig,
    ) -> None:
        """
        Initialize the AddressProcessor with configuration.
        
        Args:
            validation_config: Validation thresholds (e.g., fuzzy match ratio)
            column_config: Column names for address components
            vectorizer_config: TF-IDF and nearest-neighbor parameters
        """
        self.validation_config = validation_config
        self.column_config = column_config
        self.vectorizer_config = vectorizer_config

        # Pre-compile regex patterns for maximum efficiency. Compiling them once
        # during initialization avoids the significant overhead of repeated
        # compilation when processing large datasets.
        # This pattern identifies and removes zero-width, control, and non-standard
        # space characters that can be invisible but disrupt tokenization.
        self.ZW_CTRL_PATTERN = re.compile(r'[\u00A0\u1680\u180E\u2000-\u200D\u202F\u205F\u2060\u3000\uFEFF\u0000-\u001F\u007F]+')
        # This pattern finds common separators and is used to normalize them to a single space.
        self.SEPS_PATTERN = re.compile(r'[,\|;/]+')
        # This pattern is used to identify and discard address fragments that contain
        # only punctuation and whitespace, which are effectively noise.
        self.PUNCT_ONLY_PATTERN = re.compile(r'^[\s\.,;#\-_/\\]+$')
        
        logger.info("Initialized AddressProcessor")
        logger.debug(f"Address columns configured: {column_config.address_cols}")
        logger.debug(f"Fuzzy ratio threshold: {validation_config.address_fuzz_ratio}")

    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    def process_addresses(
        self, 
        gdf: cudf.DataFrame, 
        is_training: bool
    ) -> cudf.DataFrame:
        """
        Run the complete address processing pipeline.
        
        Processing steps:
        1. Combine configured address columns into single string
        2. Parse addresses using libpostal (CPU operation)
        3. Add parsed components and normalized key to DataFrame
        4. Consolidate similar addresses (training only)
        
        Args:
            gdf: Input DataFrame with raw address columns
            is_training: Whether to perform address consolidation
            
        Returns:
            DataFrame with parsed address components and normalized keys
        """
        # Validate input
        if gdf is None or len(gdf) == 0:
            logger.warning("Received empty DataFrame; returning as-is")
            return gdf
        
        logger.info(f"Starting address processing for {len(gdf):,} records")
        logger.debug(f"Input shape: {gdf.shape}")
        logger.debug(f"Training mode: {is_training}")
        
        # Step 1: Combine address columns
        logger.info("Step 1: Combining address columns")
        combined_address_series = self._combine_address_columns(gdf)
        
        # Check if we have any non-empty addresses
        non_empty_count = (combined_address_series != "").sum()
        logger.info(f"Found {non_empty_count:,} non-empty addresses")
        
        if non_empty_count == 0:
            logger.warning("No valid addresses found; adding empty address columns")
            return self._add_empty_address_columns(gdf)
        
        # Step 2: Parse addresses
        logger.info("Step 2: Parsing addresses with libpostal")
        parsed_addresses_df = self._parse_address_series(combined_address_series)
        
        # Step 3: Add parsed columns and create normalized key
        logger.info("Step 3: Adding parsed components and creating normalized key")
        gdf_with_parsed = self._add_parsed_columns_to_gdf(gdf, parsed_addresses_df)
        
        # Step 4: Consolidate similar addresses (training only)
        if is_training:
            logger.info("Step 4: Consolidating similar addresses (training mode)")
            gdf_with_parsed = self._consolidate_similar_addresses(gdf_with_parsed)
        else:
            logger.info("Step 4: Skipping consolidation (inference mode)")
        
        logger.info(f"Address processing complete. Output shape: {gdf_with_parsed.shape}")
        return gdf_with_parsed

    def split_canonical_address(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Split canonical_address column into component columns.
        
        This method is used for final output when a single canonical address
        string needs to be split into components for export or review.
        
        Args:
            df: DataFrame with canonical_address column
            
        Returns:
            DataFrame with additional canonical component columns
        """
        logger.info("Splitting canonical addresses into components")
        
        if "canonical_address" not in df.columns:
            logger.warning("'canonical_address' column not found; skipping split")
            return df
        
        # Parse canonical addresses
        logger.debug(f"Parsing {len(df):,} canonical addresses")
        parsed_dicts = df["canonical_address"].apply(utils.safe_parse_address)
        
        # Check for parsing failures
        failures = sum(1 for d in parsed_dicts if not d)
        if failures > 0:
            logger.warning(f"Failed to parse {failures:,} canonical addresses")
        
        parsed_components_df = pd.json_normalize(parsed_dicts)
        
        # Map parsed components to output columns
        addr_cols_map: Dict[str, str] = {
            "canonical_street_number": "address_line_1.street_number",
            "canonical_street_name": "address_line_1.street_name",
            "canonical_city": "city",
            "canonical_state": "state",
            "canonical_zip": "postal_code",
        }
        
        # Add component columns
        for out_col, src_col in addr_cols_map.items():
            if src_col in parsed_components_df.columns:
                df[out_col] = parsed_components_df[src_col].fillna("")
            else:
                df[out_col] = ""
                logger.debug(f"Component '{src_col}' not found; using empty string for '{out_col}'")
        
        logger.info(f"Added {len(addr_cols_map)} canonical component columns")
        return df

    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================
    
    def _combine_address_columns(self, gdf: cudf.DataFrame) -> cudf.Series:
        """
        Combine configured address columns into a single, clean string for libpostal.

        This function applies a robust, multi-stage preprocessing pipeline to each
        address component individually before concatenating them. This ensures a
        high-quality, standardized input string, which significantly improves the
        accuracy of libpostal's statistical parsing model.

        Preprocessing Pipeline per Component:
        1. Unicode Normalization (NFKC): Standardizes character representations.
        2. Character Removal: Eliminates invisible control characters that break parsing.
        3. Separator Normalization: Converts various separators to a standard space.
        4. Lowercasing: Ensures case-insensitivity for uniform input.
        5. Noise Filtering: Removes parts that contain only punctuation.

        Args:
            gdf: DataFrame with one or more address columns.

        Returns:
            A cuDF Series containing the combined and preprocessed address strings.
        """
        address_cols: List[str] = list(self.column_config.address_cols or [])

        if not address_cols:
            logger.warning("No address columns specified; returning empty series")
            return cudf.Series([""] * len(gdf), index=gdf.index)

        # Check which of the configured address columns are actually in the DataFrame.
        available_cols = [col for col in address_cols if col in gdf.columns]
        missing_cols = [col for col in address_cols if col not in gdf.columns]

        if missing_cols:
            logger.warning(f"Missing address columns, they will be ignored: {missing_cols}")

        if not available_cols:
            logger.error("No valid address columns found in DataFrame to combine.")
            return cudf.Series([""] * len(gdf), index=gdf.index)

        logger.debug(f"Combining {len(available_cols)} address columns: {available_cols}")

        # Process each address part with robust cleaning before concatenation.
        address_parts = []
        for col in available_cols:
            # Start with the raw column, filling nulls and ensuring string type.
            address_part = gdf[col].fillna("").astype("str")

            # Step 1: Unicode Normalization (NFKC). This is a critical first step. It
            # standardizes character representations, converting full-width characters to
            # half-width, handling ligatures (e.g., 'ﬁ' -> 'fi'), and collapsing
            # combining marks. This creates a canonical text representation for parsing.
            address_part = address_part.str.normalize_characters(form='NFKC')

            # Step 2: Remove disruptive characters and normalize separators. Zero-width,
            # control, and non-standard space characters can break tokenization logic
            # and are often invisible. Removing them prevents hard-to-debug parsing errors.
            # Common separators are then collapsed into a single space for consistency.
            address_part = address_part.str.replace(self.ZW_CTRL_PATTERN.pattern, '', regex=True)
            address_part = address_part.str.replace(self.SEPS_PATTERN.pattern, ' ', regex=True)

            # Step 3: Convert to lowercase. libpostal is largely case-insensitive, but
            # providing a consistently cased string is a best practice that removes
            # any potential ambiguity for its statistical model.
            address_part = address_part.str.lower()

            # Step 4: Trim whitespace and filter out "empty" fragments. After cleaning,
            # some parts might be left with only punctuation or spaces (e.g., a column
            # that only contained "-"). These are effectively noise and should be
            # treated as empty strings to avoid junk tokens in the final address.
            address_part = address_part.str.strip()
            # The `where` clause acts as a conditional replacement: if the part does NOT
            # match the punctuation-only pattern, keep it; otherwise, replace it with ''.
            address_part = address_part.where(~address_part.str.match(self.PUNCT_ONLY_PATTERN.pattern), other='')

            address_parts.append(address_part)

        # Combine the cleaned parts into a single string.
        if len(address_parts) == 1:
            combined_address = address_parts[0]
        else:
            # The `str.cat` method is highly efficient for this operation on the GPU.
            combined_address = address_parts[0].str.cat(others=address_parts[1:], sep=" ")

        # Step 5: Perform a final cleanup on the fully combined string. After joining,
        # there might be residual separators at the boundaries or multiple spaces if
        # an empty part was joined. This pass ensures the final output is clean,
        # compact, and has no leading/trailing whitespace.
        combined_address = (
            combined_address
            .str.replace(self.SEPS_PATTERN.pattern, ' ', regex=True)
            .str.normalize_spaces()
            .str.strip()
        )

        # Log a few samples for debugging and verification purposes.
        sample_size = min(3, len(combined_address))
        if sample_size > 0 and not combined_address.empty:
            sample = combined_address.head(sample_size).to_pandas().tolist()
            logger.debug(f"Combined address samples: {sample}")

        return combined_address
    
    def _parse_address_series(self, address_series: cudf.Series) -> cudf.DataFrame:
        """
        Parse addresses using libpostal (CPU operation).
        
        Args:
            address_series: Series of address strings
            
        Returns:
            DataFrame with parsed address components
        """
        total_addresses = len(address_series)
        logger.info(f"Parsing {total_addresses:,} addresses with libpostal")
        
        # Convert to pandas for CPU processing
        logger.debug("Converting to pandas for CPU parsing")
        address_series_pd = address_series.to_pandas()
        
        # Parse addresses
        parsed_dicts = address_series_pd.apply(utils.safe_parse_address)
        
        # Count failures
        failures = sum(1 for d in parsed_dicts if not d)
        if failures > 0:
            failure_rate = failures / total_addresses
            logger.warning(
                f"Failed to parse {failures:,} addresses ({failure_rate:.1%})"
            )
            if failure_rate > 0.1:  # More than 10% failures
                logger.error("High address parsing failure rate - check data quality")
        
        # Convert to DataFrame
        parsed_pdf = pd.json_normalize(parsed_dicts)
        
        # Log parsed components
        if not parsed_pdf.empty:
            available_components = list(parsed_pdf.columns)
            logger.debug(f"Parsed components available: {available_components}")
        
        # Convert back to GPU
        parsed_gdf = cudf.DataFrame.from_pandas(parsed_pdf)
        logger.debug(f"Parsed DataFrame shape: {parsed_gdf.shape}")
        
        return parsed_gdf
    
    def _add_parsed_columns_to_gdf(
        self, 
        gdf: cudf.DataFrame, 
        parsed_df: cudf.DataFrame
    ) -> cudf.DataFrame:
        """
        Add parsed address components to original DataFrame.
        
        Args:
            gdf: Original DataFrame
            parsed_df: DataFrame with parsed components
            
        Returns:
            DataFrame with added address columns and normalized key
        """
        logger.debug("Adding parsed address columns to DataFrame")
        
        # Define column mapping
        rename_map: Dict[str, str] = {
            "address_line_1.street_number": "addr_street_number",
            "address_line_1.street_name": "addr_street_name",
            "city": "addr_city",
            "state": "addr_state",
            "postal_code": "addr_zip",
        }
        
        # Rename columns in parsed DataFrame
        parsed_df_renamed = parsed_df.rename(columns=rename_map)
        
        # Concatenate DataFrames
        out_gdf = cudf.concat([gdf, parsed_df_renamed], axis=1)
        
        # Ensure all expected columns exist and handle nulls
        for _, new_col in rename_map.items():
            if new_col not in out_gdf.columns:
                out_gdf[new_col] = ""
                logger.debug(f"Column '{new_col}' not found; adding empty column")
            else:
                out_gdf[new_col] = out_gdf[new_col].fillna("").astype("str")
        
        # Create normalized address key
        logger.debug("Creating normalized address key")
        out_gdf["addr_normalized_key"] = utils.create_address_key_gpu(out_gdf)
        
        # Log statistics
        non_empty_keys = (out_gdf["addr_normalized_key"] != "").sum()
        logger.info(f"Created {non_empty_keys:,} non-empty address keys")
        
        return out_gdf
    
    def _add_empty_address_columns(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Add empty address columns when no addresses are available.
        
        Args:
            gdf: Input DataFrame
            
        Returns:
            DataFrame with empty address columns
        """
        addr_columns = [
            "addr_street_number",
            "addr_street_name", 
            "addr_city",
            "addr_state",
            "addr_zip",
            "addr_normalized_key"
        ]
        
        for col in addr_columns:
            gdf[col] = ""
        
        logger.debug(f"Added {len(addr_columns)} empty address columns")
        return gdf
    
    def _consolidate_similar_addresses(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Find and consolidate near-duplicate addresses.
        
        This method identifies similar addresses using TF-IDF similarity
        and graph components, then replaces variations with canonical forms.
        
        Args:
            gdf: DataFrame with parsed address components
            
        Returns:
            DataFrame with consolidated address keys
        """
        logger.info("Starting address consolidation")
        
        # Check required columns
        required_cols = [
            "addr_street_number",
            "addr_street_name",
            "addr_city",
            "addr_state",
            "addr_zip",
            "addr_normalized_key",
        ]
        
        missing_cols = [c for c in required_cols if c not in gdf.columns]
        if missing_cols:
            logger.error(f"Cannot consolidate; missing columns: {missing_cols}")
            return gdf
        
        # Get unique addresses before consolidation
        unique_before = gdf["addr_normalized_key"].nunique()
        logger.info(f"Unique addresses before consolidation: {unique_before:,}")
        
        # Extract address components
        address_components_gdf = gdf[required_cols].copy()
        
        # Get consolidation mapping
        logger.debug("Building consolidation map")
        consolidation_map = self._get_consolidation_map(
            address_gdf=address_components_gdf,
            key_col="addr_normalized_key",
            fuzz_ratio=self.validation_config.address_fuzz_ratio,
        )
        
        if consolidation_map is not None and not consolidation_map.empty:
            # Apply consolidation
            logger.info(f"Consolidating {len(consolidation_map):,} address variations")
            
            # Create a copy to avoid side effects
            out_gdf = gdf.copy()
            
            # Log sample mappings for debugging
            if len(consolidation_map) > 0:
                sample_size = min(5, len(consolidation_map))
                sample = consolidation_map.head(sample_size)
                logger.debug(f"Sample consolidation mappings:\n{sample}")
            
            # Apply mapping
            original_keys = out_gdf["addr_normalized_key"].copy()
            out_gdf["addr_normalized_key"] = out_gdf["addr_normalized_key"].replace(consolidation_map)
            
            # Count changes
            keys_changed = (out_gdf["addr_normalized_key"] != original_keys).sum()
            unique_after = out_gdf["addr_normalized_key"].nunique()
            
            logger.info(
                f"Consolidation complete: {keys_changed:,} keys changed, "
                f"{unique_before:,} -> {unique_after:,} unique addresses "
                f"({(unique_before - unique_after) / unique_before:.1%} reduction)"
            )
            
            return out_gdf
        else:
            logger.info("No similar addresses found for consolidation")
            return gdf
    
    def _get_consolidation_map(
        self, 
        address_gdf: cudf.DataFrame, 
        key_col: str, 
        fuzz_ratio: int
    ) -> Optional[cudf.Series]:
        """
        Build mapping from address keys to canonical representatives.
        
        This method:
        1. Finds similar address pairs using TF-IDF similarity
        2. Builds a graph and finds connected components
        3. Selects the best representative for each component
        
        Args:
            address_gdf: DataFrame with address components
            key_col: Column containing normalized address keys
            fuzz_ratio: Similarity threshold (0-100, higher = stricter)
            
        Returns:
            Series mapping original keys to canonical keys, or None
        """
        logger.debug(f"Building consolidation map with fuzz_ratio={fuzz_ratio}")
        
        # Get frequency of each address (for scoring)
        freq_map = address_gdf[key_col].value_counts()
        logger.debug(f"Address frequency distribution: {len(freq_map):,} unique addresses")
        
        # Get unique addresses
        unique_addresses = address_gdf.drop_duplicates(subset=[key_col]).reset_index(drop=True)
        n_unique = len(unique_addresses)
        
        if n_unique < 2:
            logger.debug("Fewer than 2 unique addresses; no consolidation needed")
            return None
        
        logger.info(f"Finding similar pairs among {n_unique:,} unique addresses")
        
        # Calculate similarity threshold
        distance_threshold = 1.0 - (float(fuzz_ratio) / 100.0)
        logger.debug(f"Using distance threshold: {distance_threshold:.3f}")
        
        # Find similar address pairs
        matched_pairs = utils.find_similar_pairs(
            string_series=unique_addresses[key_col],
            tfidf_params=self.vectorizer_config.similarity_tfidf,
            nn_params=self.vectorizer_config.similarity_nn,
            distance_threshold=distance_threshold,
        )
        
        if matched_pairs.empty:
            logger.info("No similar address pairs found")
            return None
        
        n_pairs = len(matched_pairs)
        logger.info(f"Found {n_pairs:,} similar address pairs")
        
        # Find connected components in similarity graph
        logger.debug("Finding connected components in similarity graph")
        components = utils.find_graph_components(
            edge_list_df=matched_pairs,
            output_vertex_column="unique_addr_idx",
            output_component_column="component_id"
        )
        
        n_components = components["component_id"].nunique()
        logger.info(f"Found {n_components:,} address groups (connected components)")
        
        # Merge component IDs with unique addresses
        unique_addresses = unique_addresses.merge(
            components, 
            left_index=True, 
            right_on="unique_addr_idx",
            how="inner"  # Only keep addresses that are in components
        )
        
        if unique_addresses.empty:
            logger.warning("No addresses matched to components")
            return None
        
        # Add frequency for scoring
        unique_addresses["freq"] = unique_addresses[key_col].map(freq_map).fillna(1)
        
        # Determine canonical representative for each component
        logger.debug("Selecting canonical representatives")
        canonical_map = self._determine_canonical_representatives(
            candidates_gdf=unique_addresses, 
            key_col=key_col
        )
        
        if canonical_map.empty:
            logger.warning("No canonical representatives determined")
            return None
        
        # Build final mapping
        logger.debug("Building final consolidation mapping")
        final_map_df = unique_addresses[[key_col, "component_id"]].merge(
            canonical_map, 
            on="component_id",
            how="inner"
        )
        
        # Only keep mappings where key differs from canonical
        final_map_df = final_map_df[final_map_df[key_col] != final_map_df["canonical_key"]]
        
        if final_map_df.empty:
            logger.info("All addresses are already canonical")
            return None
        
        logger.debug(f"Final consolidation map contains {len(final_map_df):,} mappings")
        
        # Return as Series with index
        return final_map_df.set_index(key_col)["canonical_key"]
    
    def _determine_canonical_representatives(
        self, 
        candidates_gdf: cudf.DataFrame, 
        key_col: str
    ) -> cudf.DataFrame:
        """
        Select the best representative for each address component.
        
        Selection criteria (in order):
        1. Completeness score (addresses with more components)
        2. Frequency (more common addresses)
        3. Alphabetical (for deterministic tie-breaking)
        
        Args:
            candidates_gdf: DataFrame with candidate addresses
            key_col: Column with address keys
            
        Returns:
            DataFrame with component_id and canonical_key columns
        """
        logger.debug("Scoring candidate addresses for canonical selection")
        
        # Calculate completeness score
        candidates_gdf["completeness_score"] = utils.calculate_address_score_gpu(candidates_gdf)
        
        # Ensure frequency column exists
        if "freq" not in candidates_gdf.columns:
            logger.warning("Frequency column missing; using default value of 1")
            candidates_gdf["freq"] = 1
        
        # Log score distribution
        logger.debug(
            f"Completeness scores: "
            f"min={candidates_gdf['completeness_score'].min():.2f}, "
            f"max={candidates_gdf['completeness_score'].max():.2f}, "
            f"mean={candidates_gdf['completeness_score'].mean():.2f}"
        )
        
        # Sort by selection criteria
        sorted_candidates = candidates_gdf.sort_values(
            by=["component_id", "completeness_score", "freq", key_col],
            ascending=[True, False, False, True]  # component_id ascending, scores descending
        )
        
        # Select first (best) candidate per component
        canonical_reps = sorted_candidates.drop_duplicates(
            subset=["component_id"], 
            keep="first"
        )
        
        logger.debug(f"Selected {len(canonical_reps):,} canonical representatives")
        
        # Return mapping
        return canonical_reps[["component_id", key_col]].rename(
            columns={key_col: "canonical_key"}
        )
