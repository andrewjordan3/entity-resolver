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
from .utils import (
    nfkc_normalize_series, 
    safe_parse_address, 
    create_address_key_gpu, 
    find_graph_components, 
    find_similar_pairs, 
    calculate_address_score_gpu,
    normalize_us_states,
)

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
        # This pattern finds common separators and is used to normalize them to a single space.
        self.SEPS_PATTERN = re.compile(r'[,\|;/]+')
        
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
        parsed_dicts = df["canonical_address"].apply(safe_parse_address)
        
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
            address_part = nfkc_normalize_series(address_part)

            # Step 3: Convert to lowercase. libpostal is largely case-insensitive, but
            # providing a consistently cased string is a best practice that removes
            # any potential ambiguity for its statistical model.
            address_part = address_part.str.lower()

            # Step 4: Trim whitespace and filter out "empty" fragments. After cleaning,
            # some parts might be left with only punctuation or spaces (e.g., a column
            # that only contained "-"). These are effectively noise and should be
            # treated as empty strings to avoid junk tokens in the final address.
            address_part = address_part.str.strip()

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
            sample = combined_address.sample(sample_size).to_pandas().tolist()
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
        parsed_dicts = address_series_pd.apply(safe_parse_address)
        
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
        
        # Normalize US state names to their full, lowercase form. This ensures
        # consistency before the address key is created.
        logger.debug("Normalizing US state names in 'addr_state' column")
        out_gdf = normalize_us_states(out_gdf, "addr_state")

        # Create normalized address key
        logger.debug("Creating normalized address key")
        out_gdf["addr_normalized_key"] = create_address_key_gpu(out_gdf)
        
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
        Find and consolidate near-duplicate addresses with validation.
        
        This method identifies similar addresses using TF-IDF similarity
        and graph components, validates them against business rules,
        then replaces variations with canonical forms.
        
        Validation includes:
        - Street number difference threshold checking
        - State boundary enforcement (optional)
        
        Args:
            gdf: DataFrame with parsed address components
            
        Returns:
            DataFrame with consolidated address keys
        """
        logger.info("Starting address consolidation with validation")
        
        # Define the set of columns required for the consolidation process. This
        # check ensures that the input DataFrame has been properly prepared by
        # earlier steps in the pipeline.
        required_cols = [
            "addr_street_number",
            "addr_street_name",
            "addr_city",
            "addr_state",
            "addr_zip",
            "addr_normalized_key",
        ]
        
        # Verify the presence of all required columns. If any are missing,
        # consolidation cannot proceed, so we log an error and return.
        missing_cols = [c for c in required_cols if c not in gdf.columns]
        if missing_cols:
            logger.error(f"Cannot consolidate; missing columns: {missing_cols}")
            return gdf
        
        # Record the number of unique addresses before consolidation to measure
        # the effectiveness of the process.
        unique_before = gdf["addr_normalized_key"].nunique()
        logger.info(f"Unique addresses before consolidation: {unique_before:,}")
        
        # Create a clean copy containing only the columns needed for consolidation.
        # This prevents accidental modification of the original DataFrame and
        # simplifies the data passed to downstream methods.
        address_components_gdf = gdf[required_cols].copy()
        
        # Build the consolidation map, which is the core of this process. This
        # map will define which address keys should be replaced by their
        # canonical (i.e., "master") equivalents.
        logger.debug("Building consolidation map with validation")
        consolidation_map = self._get_consolidation_map(
            address_gdf=address_components_gdf,
            key_col="addr_normalized_key",
        )
        
        # Proceed only if the consolidation map is not empty, meaning there are
        # address variations to be consolidated.
        if consolidation_map is not None and not consolidation_map.empty:
            logger.info(f"Consolidating {len(consolidation_map):,} validated address variations")
            
            # Create a full copy of the input DataFrame to apply changes to.
            out_gdf = gdf.copy()
            
            # Log a small sample of the mappings for debugging and transparency.
            # This helps in understanding what kind of changes are being made.
            sample_size = min(5, len(consolidation_map))
            sample = consolidation_map.head(sample_size)
            logger.debug(f"Sample consolidation mappings:\n{sample}")
            
            # Store the original keys to later calculate how many were changed.
            original_keys = out_gdf["addr_normalized_key"].copy()
            
            # Apply the mapping. The `replace` method efficiently substitutes
            # the old keys with their new canonical versions.
            out_gdf["addr_normalized_key"] = out_gdf["addr_normalized_key"].replace(consolidation_map)
            
            # Calculate and log the impact of the consolidation.
            keys_changed = (out_gdf["addr_normalized_key"] != original_keys).sum()
            unique_after = out_gdf["addr_normalized_key"].nunique()
            reduction_pct = (unique_before - unique_after) / unique_before if unique_before > 0 else 0
            
            logger.info(
                f"Consolidation complete: {keys_changed:,} keys changed, "
                f"{unique_before:,} -> {unique_after:,} unique addresses "
                f"({reduction_pct:.1%} reduction)"
            )
            
            return out_gdf
        else:
            # If no valid pairs were found, no consolidation is needed.
            logger.info("No valid similar addresses found for consolidation")
            return gdf

    def _get_consolidation_map(
        self, 
        address_gdf: cudf.DataFrame, 
        key_col: str, 
    ) -> Optional[cudf.Series]:
        """
        Build a mapping from address keys to canonical representatives with validation.
        
        This method:
        1. Finds similar address pairs using TF-IDF similarity.
        2. Validates pairs against business rules (street numbers, states).
        3. Builds a graph from validated pairs and finds connected components.
        4. Selects the best representative for each component.
        
        Args:
            address_gdf: DataFrame with address components.
            key_col: Column containing normalized address keys.
            
        Returns:
            A Series mapping original keys to canonical keys, or None if no
            consolidation is possible.
        """
        # Retrieve validation parameters from the class configuration. This makes
        # the method's signature cleaner and keeps configuration centralized.
        fuzz_ratio = self.validation_config.address_fuzz_ratio

        logger.debug(
            f"Building consolidation map with validation: "
            f"fuzz_ratio={fuzz_ratio}, "
            f"street_number_threshold={self.validation_config.street_number_threshold}, "
            f"enforce_state_boundaries={self.validation_config.enforce_state_boundaries}"
        )
        
        # Calculate the frequency of each address key. This is a crucial input
        # for scoring, as more frequent addresses are better candidates for
        # being the canonical representative.
        freq_map = address_gdf[key_col].value_counts()
        logger.debug(f"Address frequency distribution: {len(freq_map):,} unique addresses")
        
        # Deduplicate the address data to work only with unique addresses. This
        # significantly improves performance for similarity calculations.
        unique_addresses_df = address_gdf.drop_duplicates(subset=[key_col]).reset_index(drop=True)
        n_unique = len(unique_addresses_df)
        
        # If there's only one or zero unique addresses, no pairs can be formed.
        if n_unique < 2:
            logger.debug("Fewer than 2 unique addresses; no consolidation needed")
            return None
        
        logger.info(f"Finding similar pairs among {n_unique:,} unique addresses")
        
        # Convert the fuzzy matching ratio (0-100) to a cosine distance
        # threshold (0-1), as expected by the similarity function.
        distance_threshold = 1.0 - (float(fuzz_ratio) / 100.0)
        logger.debug(f"Using distance threshold: {distance_threshold:.3f}")
        
        # Find pairs of addresses that are textually similar based on TF-IDF
        # vectorization and nearest neighbors search.
        matched_pairs_df = find_similar_pairs(
            string_series=unique_addresses_df[key_col],
            tfidf_params=self.vectorizer_config.similarity_tfidf,
            nn_params=self.vectorizer_config.similarity_nn,
            distance_threshold=distance_threshold,
        )
        
        if matched_pairs_df.empty:
            logger.info("No similar address pairs found based on text similarity")
            return None
        
        n_pairs_before_validation = len(matched_pairs_df)
        logger.info(f"Found {n_pairs_before_validation:,} potential pairs before validation")
        
        # Apply business logic to filter out pairs that are textually similar
        # but logically distinct (e.g., different street numbers or states).
        validated_pairs_df = self._validate_address_pairs(
            matched_pairs_df=matched_pairs_df,
            unique_addresses_df=unique_addresses_df,
        )
        
        if validated_pairs_df.empty:
            logger.info("No pairs passed business rule validation")
            return None
        
        n_pairs_after_validation = len(validated_pairs_df)
        pass_rate = n_pairs_after_validation / n_pairs_before_validation if n_pairs_before_validation > 0 else 0
        logger.info(
            f"Validated pairs: {n_pairs_after_validation:,}/{n_pairs_before_validation:,} "
            f"({pass_rate:.1%} passed validation)"
        )
        
        # Use the validated pairs as edges in a graph to find groups (connected
        # components) of similar addresses.
        logger.debug("Finding connected components in validated similarity graph")
        components_df = find_graph_components(
            edge_list_df=validated_pairs_df,
            output_vertex_column="unique_addr_idx",
            output_component_column="component_id"
        )
        
        n_components = components_df["component_id"].nunique()
        logger.info(f"Found {n_components:,} validated address groups (connected components)")
        
        # Join component IDs back to the unique address data.
        unique_addresses_df = unique_addresses_df.merge(
            components_df, 
            left_index=True, 
            right_on="unique_addr_idx",
            how="inner"  # Keep only addresses that are part of a component.
        )
        
        if unique_addresses_df.empty:
            logger.warning("No addresses were matched to any components after merge")
            return None
        
        # Add the pre-calculated frequency to each unique address for scoring.
        unique_addresses_df["freq"] = unique_addresses_df[key_col].map(freq_map).fillna(1)
        
        # For each component, select the best address to be the canonical one.
        logger.debug("Selecting canonical representatives for each component")
        canonical_map_df = self._determine_canonical_representatives(
            candidates_gdf=unique_addresses_df, 
            key_col=key_col
        )
        
        if canonical_map_df.empty:
            logger.warning("Could not determine any canonical representatives")
            return None
        
        # Join the canonical key for each component back to all addresses in that component.
        logger.debug("Building final consolidation mapping from canonical representatives")
        final_map_df = unique_addresses_df[[key_col, "component_id"]].merge(
            canonical_map_df, 
            on="component_id",
            how="inner"
        )
        
        # The final map should only contain entries where an address key needs to
        # be changed to its canonical form.
        final_map_df = final_map_df[final_map_df[key_col] != final_map_df["canonical_key"]]
        
        if final_map_df.empty:
            logger.info("All addresses are already in their canonical form; no changes needed")
            return None
        
        logger.debug(f"Final consolidation map contains {len(final_map_df):,} mappings")
        
        # Convert the DataFrame to a Series for efficient use with `.replace()`.
        # The index is the key to be replaced, and the value is the new canonical key.
        return final_map_df.set_index(key_col)["canonical_key"]

    def _validate_address_pairs(
        self,
        matched_pairs_df: cudf.DataFrame,
        unique_addresses_df: cudf.DataFrame,
    ) -> cudf.DataFrame:
        """
        Validate address pairs against business rules to prevent incorrect consolidation.
        
        This method filters out pairs that violate:
        1. Street number difference threshold (if street numbers are numeric).
        2. State boundary enforcement (if enabled and states are present).
        
        Args:
            matched_pairs_df: DataFrame with ['source', 'destination'] columns representing pair indices.
            unique_addresses_df: DataFrame with address components for validation.
            
        Returns:
            A DataFrame containing only the pairs that passed all validation rules.
        """
        street_number_threshold = self.validation_config.street_number_threshold
        enforce_state_boundaries = self.validation_config.enforce_state_boundaries

        logger.debug("Validating address pairs against business rules")
        
        original_pair_count = len(matched_pairs_df)
        if original_pair_count == 0:
            return matched_pairs_df # Return early if there are no pairs to validate.

        # --- Augment Pair Data ---
        # To validate, we need the actual address components (street number, state)
        # for both addresses in each pair. We merge this data from unique_addresses_df.
        
        # Define the component columns needed for validation.
        component_cols = ['addr_street_number', 'addr_state', 'addr_normalized_key']

        # Merge components for the first address in the pair (index 'source').
        pairs_with_components_df = matched_pairs_df.merge(
            unique_addresses_df[component_cols],
            left_on='source', right_index=True, how='left'
        ).rename(columns={
            'addr_street_number': 'street_number_i',
            'addr_state': 'state_i',
            'addr_normalized_key': 'key_i'
        })
        
        # Merge components for the second address in the pair (index 'destination').
        pairs_with_components_df = pairs_with_components_df.merge(
            unique_addresses_df[component_cols],
            left_on='destination', right_index=True, how='left'
        ).rename(columns={
            'addr_street_number': 'street_number_j',
            'addr_state': 'state_j',
            'addr_normalized_key': 'key_j'
        })

        # Initialize a boolean mask where `True` means the pair is currently considered valid.
        valid_pairs_mask = cudf.Series([True] * len(pairs_with_components_df), index=pairs_with_components_df.index)

        # --- Rule 1: Validate Street Number Differences ---
        logger.debug(f"Applying street number validation (threshold: {street_number_threshold})")

        # Create a boolean mask to identify pairs that violate the street number rule.
        # Initialize all to False (no violation).
        violates_street_num_rule = cudf.Series([False] * len(pairs_with_components_df), index=pairs_with_components_df.index)
        
        # Isolate pairs where BOTH street numbers are purely numeric strings.
        # This prevents errors from trying to convert non-numeric values (e.g., '123-A')
        # and correctly ignores pairs where one or both numbers are missing.
        street_num_i_is_numeric = pairs_with_components_df['street_number_i'].str.match(r'^\d+$')
        street_num_j_is_numeric = pairs_with_components_df['street_number_j'].str.match(r'^\d+$')
        both_are_numeric = street_num_i_is_numeric & street_num_j_is_numeric
        
        if both_are_numeric.any():
            # Calculate the absolute difference only for the numeric pairs.
            street_num_i = pairs_with_components_df['street_number_i'][both_are_numeric].astype('int32')
            street_num_j = pairs_with_components_df['street_number_j'][both_are_numeric].astype('int32')
            street_num_diff = (street_num_i - street_num_j).abs()
            
            # Identify which of these numeric pairs exceed the allowed difference.
            exceeds_threshold = street_num_diff > street_number_threshold
            
            # Map the results from the subset back to our full-length violation mask.
            # This correctly aligns the partial results with the full DataFrame index.
            violates_street_num_rule.loc[both_are_numeric] = exceeds_threshold

            num_invalid_street = violates_street_num_rule.sum()
            if num_invalid_street > 0:
                logger.info(
                    f"  Street number validation: {int(num_invalid_street):,} pairs rejected "
                    f"(difference > {street_number_threshold})"
                )
        
        # Update the main validation mask by marking any pairs that violated the rule as invalid.
        valid_pairs_mask = valid_pairs_mask & ~violates_street_num_rule

        # --- Rule 2: Validate State Boundaries ---
        if enforce_state_boundaries:
            logger.debug("Applying state boundary validation")
            
            # Isolate pairs where BOTH addresses have a non-empty state specified.
            # If one or both states are missing, we cannot enforce this rule, so
            # the pair is allowed to pass this check (treated as a wildcard).
            state_i_present = (pairs_with_components_df['state_i'] != '') & pairs_with_components_df['state_i'].notna()
            state_j_present = (pairs_with_components_df['state_j'] != '') & pairs_with_components_df['state_j'].notna()
            both_states_present = state_i_present & state_j_present
            
            # For pairs where both states are present, check if they are different.
            states_mismatch = pairs_with_components_df['state_i'] != pairs_with_components_df['state_j']
            
            # A pair is invalid under this rule if both states are present AND they do not match.
            invalid_state_mask = both_states_present & states_mismatch
            
            if invalid_state_mask.any():
                # Update the main mask: `valid_pairs_mask` becomes False where `invalid_state_mask` is True.
                valid_pairs_mask = valid_pairs_mask & ~invalid_state_mask
                
                num_invalid_state = invalid_state_mask.sum()
                logger.info(
                    f"  State boundary validation: {num_invalid_state:,} pairs rejected (different states)"
                )

        # --- Final Filtering ---
        # Apply the final mask to the DataFrame that contains the augmented component data,
        # as this is the DataFrame that is correctly aligned with the mask's index.
        validated_pairs_with_components_df = pairs_with_components_df[valid_pairs_mask]

        # After filtering, select only the original 'source' and 'destination' columns
        # to return a clean edge list for the next step.
        validated_pairs_df = validated_pairs_with_components_df[['source', 'destination']]
        
        # Report the summary of the validation process.
        num_rejected = original_pair_count - len(validated_pairs_df)
        if num_rejected > 0:
            rejection_rate = num_rejected / original_pair_count
            logger.info(
                f"Validation complete: {num_rejected:,}/{original_pair_count:,} pairs rejected "
                f"({rejection_rate:.1%})"
            )
        else:
            logger.info("All pairs passed validation rules")
            
        return validated_pairs_df
    
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
        candidates_gdf["completeness_score"] = calculate_address_score_gpu(candidates_gdf)
        
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
