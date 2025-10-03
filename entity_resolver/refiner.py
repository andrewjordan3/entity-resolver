# entity_resolver/refiner.py
"""
This module defines the ClusterRefiner class, which is responsible for the
final stages of the entity resolution pipeline: refining cluster assignments, 
building the canonical map, and applying it to the dataset.

The refinement process includes:
- Address enrichment for incomplete records.
- Cluster splitting based on geographical or address range conflicts.
- Creation of a canonical entity profile for each cluster.
- Identification and numbering of chain entities (e.g., franchises).
"""

import cudf
import logging

# --- Local Package Imports ---
from .config import ValidationConfig, OutputConfig, VectorizerConfig
from .utils import (
    get_canonical_name_gpu,
    get_best_address_gpu,
    create_address_key_gpu,
)

# Set up a logger for this module
logger = logging.getLogger(__name__)

class ClusterRefiner:
    """
    Handles cluster refinement, canonical map creation, and final data application.

    This class performs three key functions in the entity resolution pipeline:
    
    1. **Refines Clusters**: Enriches records with missing address data and splits
       clusters that have clear conflicts (e.g., same name in different states).
       
    2. **Builds Canonical Map**: Creates the final, definitive mapping from a
       cluster ID to a single canonical entity profile (name and address).
       
    3. **Applies Map**: Merges the canonical information back onto the full dataset,
       ensuring every record has a canonical reference.
       
    All operations are optimized for GPU execution using cuDF DataFrames to maintain
    high performance with large datasets.
    """
    def __init__(
        self, 
        validation_config: ValidationConfig, 
        output_config: OutputConfig, 
        vectorizer_config: VectorizerConfig
    ):
        """
        Initializes the ClusterRefiner with necessary configuration objects.

        Args:
            validation_config: Configuration for validation rules including:
                - State boundary enforcement
                - Street number range thresholds for splitting
                - Other conflict detection parameters
            output_config: Configuration for output formatting including:
                - Desired case format (proper, upper, lower)
                - Output field specifications
            vectorizer_config: Configuration for vectorization including:
                - Similarity computation parameters
                - TF-IDF settings for canonical name selection
        """
        self.validation_config = validation_config
        self.output_config = output_config
        self.vectorizer_config = vectorizer_config
        
        # Log initialization with config details
        logger.info(
            f"ClusterRefiner initialized with: "
            f"state_boundaries={validation_config.enforce_state_boundaries}, "
            f"street_threshold={validation_config.street_number_threshold}, "
            f"output_format={output_config.output_format}"
        )

    def refine_clusters(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Refines clusters through address enrichment and conflict-based splitting.
        
        This method performs two sequential refinement operations:
        1. Address enrichment: Fills in missing street-level information for records
           that share cluster, city, state, and zip with other complete records.
        2. Conflict splitting: Breaks apart clusters that span multiple states or
           have widely disparate street numbers for the same street.
        
        Args:
            gdf: Input cuDF DataFrame containing initial cluster assignments in 
                 'cluster' column and address components.
                 
        Returns:
            cudf.DataFrame: DataFrame with added 'final_cluster' column containing
                           refined cluster assignments and 'address_was_enriched' 
                           flag for tracking enriched records.
        """
        # Check if address information is available for refinement
        if 'addr_normalized_key' not in gdf.columns:
            logger.warning(
                "No address information found (missing 'addr_normalized_key'). "
                "Skipping cluster refinement - using initial clusters as final."
            )
            gdf['final_cluster'] = gdf['cluster']
            gdf['address_was_enriched'] = False
            return gdf

        logger.info(
            f"Starting cluster refinement for {len(gdf):,} records with "
            f"{gdf['cluster'].nunique():,} initial clusters..."
        )
        
        # Initialize refinement tracking columns
        gdf['final_cluster'] = gdf['cluster']
        gdf['address_was_enriched'] = False
        
        # Perform sequential refinement steps
        initial_cluster_count = gdf['final_cluster'].nunique()
        
        gdf = self._enrich_addresses(gdf)
        enriched_record_count = gdf['address_was_enriched'].sum()
        
        gdf = self._split_clusters_by_conflict(gdf)
        final_cluster_count = gdf['final_cluster'].nunique()
        
        # Log refinement summary
        cluster_change = final_cluster_count - initial_cluster_count
        logger.info(
            f"Cluster refinement complete: "
            f"{enriched_record_count:,} records enriched, "
            f"clusters changed from {initial_cluster_count:,} to {final_cluster_count:,} "
            f"(net change: {cluster_change:+,})"
        )
        
        return gdf

    def build_canonical_map(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Builds the final canonical map from refined clusters using vectorized ops.

        This method creates a definitive mapping of each cluster ID to a
        single canonical entity profile. It determines:
        - A canonical name (most representative name).
        - A best address (most complete and reliable address).
        - The average cluster probability score.
        - Chain numbering for multi-location entities.

        Args:
            gdf: cuDF DataFrame with 'final_cluster' assignments and entity data.

        Returns:
            A cuDF DataFrame serving as the canonical map, with one row per
            unique cluster. Returns an empty DataFrame if no valid clusters exist.
        """
        logger.info("Building canonical map from refined clusters...")

        # Filter to only clustered records, excluding noise points (cluster == -1).
        clustered_records_gdf = gdf[gdf['final_cluster'] != -1].copy()

        if clustered_records_gdf.empty:
            logger.warning(
                "No valid clusters found (all records may be noise points). "
                "Cannot build canonical map."
            )
            return cudf.DataFrame()

        num_unique_clusters = clustered_records_gdf['final_cluster'].nunique()
        logger.info(f"Processing {num_unique_clusters:,} unique clusters for canonical profiles...")

        # --- Vectorized Canonical Profile Generation ---

        # 1. Get the best address profile for each cluster using groupby().apply().
        logger.debug("Step 1: Determining best address profile for each cluster via groupby-apply...")
        best_addresses_df = clustered_records_gdf.groupby('final_cluster').apply(
            get_best_address_gpu
        )
        # The result has a MultiIndex ('final_cluster', original_index). We want
        # 'final_cluster' as a column and to discard the original index.
        best_addresses_df = best_addresses_df.reset_index(level=1, drop=True).reset_index()
        logger.debug(f"Found {len(best_addresses_df):,} best address profiles.")

        # 2. Calculate average cluster probabilities for all clusters at once.
        logger.debug("Step 2: Calculating average cluster probabilities...")
        avg_probabilities_df = clustered_records_gdf.groupby('final_cluster').agg(
            avg_cluster_prob=('cluster_probability', 'mean')
        ).reset_index()
        logger.debug("Average probabilities calculated.")

        # 3. Determine the canonical name for each cluster.
        logger.debug("Step 3: Determining canonical name for each cluster...")
        canonical_names_df = self._get_canonical_names(clustered_records_gdf)
        logger.debug("Canonical names determined.")

        # 4. Merge the components into the final canonical map.
        logger.debug("Step 4: Assembling the final canonical map...")
        # Start with the addresses, which have the most columns.
        canonical_map_df = best_addresses_df.merge(
            avg_probabilities_df, on='final_cluster', how='left'
        ).merge(
            canonical_names_df, on='final_cluster', how='left'
        )

        logger.info(
            f"Created initial canonical map with {len(canonical_map_df):,} entries. "
            f"Average cluster probability: {canonical_map_df['avg_cluster_prob'].mean():.3f}"
        )

        # 5. Handle chain entities (e.g., "Starbucks - 1", "Starbucks - 2").
        canonical_map_df = self._number_chain_entities(canonical_map_df)

        logger.info(f"Canonical map build complete with {len(canonical_map_df):,} final entries.")
        return canonical_map_df

    def apply_canonical_map(
        self, 
        gdf: cudf.DataFrame, 
        canonical_map: cudf.DataFrame
    ) -> cudf.DataFrame:
        """
        Applies the final canonical map to the full dataset.
        
        Merges canonical entity information onto all records, ensuring every record
        has a canonical reference. Unclustered records (noise) receive self-references.
        
        Args:
            gdf: Full cuDF DataFrame with all records and 'final_cluster' assignments.
            canonical_map: Canonical map DataFrame with cluster-to-canonical mappings.
            
        Returns:
            cudf.DataFrame: Enhanced DataFrame with 'canonical_name' and 
                           'canonical_address' columns for every record.
        """
        logger.info(
            f"Applying canonical map to {len(gdf):,} records in final dataset..."
        )
        
        # Handle empty canonical map case
        if canonical_map is None or canonical_map.empty:
            logger.warning(
                "Canonical map is empty. Assigning self-references for all records."
            )
            gdf['canonical_name'] = gdf['normalized_text']
            gdf['canonical_address'] = gdf['addr_normalized_key']
            return gdf
        
        # Prepare canonical map for merging - select only necessary columns
        canonical_merge_columns = canonical_map[
            ['final_cluster', 'canonical_name', 'addr_normalized_key']
        ].rename(columns={'addr_normalized_key': 'canonical_address'})
        
        # Perform GPU-accelerated merge
        logger.debug(f"Merging canonical data for {len(canonical_merge_columns):,} clusters...")
        result_dataframe = gdf.merge(
            canonical_merge_columns, 
            on='final_cluster', 
            how='left'
        )
        
        # Count records that will receive canonical assignments vs self-references
        matched_record_count = result_dataframe['canonical_name'].notna().sum()
        unmatched_record_count = len(result_dataframe) - matched_record_count
        
        logger.info(
            f"Canonical assignments: {matched_record_count:,} from map, "
            f"{unmatched_record_count:,} self-references (noise/unclustered)"
        )
        
        # Fill unclustered/noise records with self-references
        result_dataframe['canonical_name'] = result_dataframe['canonical_name'].fillna(
            result_dataframe['normalized_text']
        )
        result_dataframe['canonical_address'] = result_dataframe['canonical_address'].fillna(
            result_dataframe['addr_normalized_key']
        )

        # Convert the pipe-delimited key into a more human-readable address string
        # by replacing pipes with spaces and normalizing any resulting whitespace.
        # This is the last step before the address is considered final.
        logger.debug("Formatting canonical_address for human readability...")
        result_dataframe['canonical_address'] = (
            result_dataframe['canonical_address']
            .str.replace('|', ' ', regex=False)
            .str.normalize_spaces()
            .str.strip()
        )
        
        # Apply final case formatting based on output configuration
        if self.output_config.output_format == 'proper':
            logger.debug("Applying proper case formatting to canonical names...")
            result_dataframe['canonical_name'] = result_dataframe['canonical_name'].str.title()
        elif self.output_config.output_format == 'upper':
            logger.debug("Applying uppercase formatting to canonical names...")
            result_dataframe['canonical_name'] = result_dataframe['canonical_name'].str.upper()
        elif self.output_config.output_format == 'lower':
            logger.debug("Applying lowercase formatting to canonical names...")
            result_dataframe['canonical_name'] = result_dataframe['canonical_name'].str.lower()
        
        logger.info("Canonical map application complete.")
        return result_dataframe

    # ============================================================================
    # Private Helper Methods
    # ============================================================================

    def _get_canonical_names(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Vectorized helper to determine the canonical name for each cluster.

        This method uses groupby().apply() to run the name selection logic
        for each cluster. While .apply() can involve some overhead, it is
        significantly more performant on the GPU than a Python-level loop as
        it leverages cuDF's internal dispatch mechanisms.

        Args:
            gdf: DataFrame containing clustered records.

        Returns:
            A DataFrame with 'final_cluster' and 'canonical_name' columns.
        """
        # This lambda function will be applied to the DataFrame subset for
        # each unique cluster.
        def find_canonical_name(df_group):
            return get_canonical_name_gpu(
                df_group['normalized_text'], self.vectorizer_config.similarity_tfidf
            )

        canonical_names_series = gdf.groupby('final_cluster').apply(find_canonical_name)
        canonical_names_df = canonical_names_series.to_frame('canonical_name').reset_index()

        return canonical_names_df

    def _enrich_addresses(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Enriches records with missing street info from cluster peers.

        This fully vectorized method finds the best address for each cluster and
        uses it to fill in missing street numbers/names for other records in
        the same cluster that share the same city, state, and zip.

        Args:
            gdf: DataFrame with cluster assignments and address components.

        Returns:
            DataFrame with enriched addresses and updated flags.
        """
        logger.debug("Starting vectorized address enrichment...")

        clustered_gdf = gdf[gdf['cluster'] != -1]
        if clustered_gdf.empty:
            logger.debug("No clusters available for address enrichment.")
            return gdf

        # 1. Get canonical address profiles for each cluster using groupby-apply.
        canonical_profiles_df = clustered_gdf.groupby('cluster').apply(
            get_best_address_gpu
        )
        # The result has a MultiIndex ('cluster', original_index). We want
        # 'cluster' as a column and to discard the original index.
        canonical_profiles_df = canonical_profiles_df.reset_index(level=1, drop=True).reset_index()
        logger.debug(f"Built {len(canonical_profiles_df):,} canonical address profiles.")

        # 2. Prepare for merge by selecting and renaming columns.
        canonical_merge_cols = {
            'cluster': 'cluster',
            'addr_street_number': 'canonical_street_number',
            'addr_street_name': 'canonical_street_name',
            'addr_city': 'canonical_city',
            'addr_state': 'canonical_state',
            'addr_zip': 'canonical_zip'
        }
        # Filter to only needed columns that exist in the profiles df
        cols_to_merge = [col for col in canonical_merge_cols.keys() if col in canonical_profiles_df.columns]
        profiles_for_merge = canonical_profiles_df[cols_to_merge].rename(
            columns=canonical_merge_cols
        )
        
        # 3. Merge canonical profiles back onto the main DataFrame.
        logger.debug("Merging canonical profiles with main dataset...")
        gdf = gdf.merge(profiles_for_merge, on='cluster', how='left')

        # 4. Define enrichment criteria using boolean masks.
        missing_street_mask = (gdf['addr_street_name'].isna()) | (gdf['addr_street_name'] == '')
        canonical_info_exists = (gdf['canonical_street_name'].notna()) & (gdf['canonical_street_name'] != '')
        geo_match_mask = (
            (gdf['addr_city'] == gdf['canonical_city']) &
            (gdf['addr_state'] == gdf['canonical_state']) &
            (gdf['addr_zip'] == gdf['canonical_zip'])
        )

        eligible_mask = (gdf['cluster'] != -1) & missing_street_mask & canonical_info_exists & geo_match_mask
        num_enrichable = eligible_mask.sum()

        # 5. Apply enrichment to eligible records.
        if num_enrichable > 0:
            logger.info(
                f"Enriching {num_enrichable:,} records with missing street "
                f"information from their cluster's canonical address."
            )
            gdf.loc[eligible_mask, 'addr_street_number'] = gdf.loc[eligible_mask, 'canonical_street_number']
            gdf.loc[eligible_mask, 'addr_street_name'] = gdf.loc[eligible_mask, 'canonical_street_name']
            gdf.loc[eligible_mask, 'address_was_enriched'] = True

            # Rebuild normalized keys for the records that were just changed.
            logger.debug("Rebuilding address keys for enriched records...")
            enriched_subset = gdf.loc[eligible_mask].copy()
            # The create_address_key_gpu function should return a Series with an index
            # that aligns with the enriched_subset.
            new_keys = create_address_key_gpu(enriched_subset)
            gdf.loc[eligible_mask, 'addr_normalized_key'] = new_keys
        else:
            logger.debug("No records were eligible for address enrichment.")

        # 6. Clean up temporary canonical columns.
        cols_to_drop = [c for c in canonical_merge_cols.values() if c != 'cluster']
        gdf = gdf.drop(columns=cols_to_drop)

        logger.debug("Address enrichment complete.")
        return gdf

    def _split_clusters_by_conflict(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Splits clusters that contain geographical or address range conflicts.
        
        This method identifies and splits clusters in two scenarios:
        1. Clusters spanning multiple states (if state boundary enforcement is enabled)
        2. Clusters with the same street name and zip but widely disparate street numbers
        
        Args:
            gdf: DataFrame with 'final_cluster' assignments and address components.
            
        Returns:
            cudf.DataFrame: DataFrame with updated 'final_cluster' assignments where
                           conflicting clusters have been split into smaller clusters.
        """
        logger.debug("Starting cluster conflict detection and splitting...")
        
        # Track the maximum cluster ID to ensure new splits get unique IDs
        current_max_cluster_id = int(gdf['final_cluster'].max())
        clusters_split_count = 0
        
        # ========================================================================
        # State Boundary Splitting
        # ========================================================================
        if self.validation_config.enforce_state_boundaries:
            logger.debug("Checking for clusters spanning multiple states...")
            
            # Find clusters with multiple unique states
            valid_clusters_df = gdf[gdf['final_cluster'] != -1]
            
            # Group by cluster and count unique states
            state_diversity_by_cluster = (
                valid_clusters_df
                .groupby('final_cluster')['addr_state']
                .nunique()
                .to_frame('unique_state_count')
                .reset_index()
            )
            
            # Identify multi-state clusters
            multi_state_clusters = state_diversity_by_cluster[
                state_diversity_by_cluster['unique_state_count'] > 1
            ]['final_cluster']
            
            if len(multi_state_clusters) > 0:
                logger.info(
                    f"Splitting {len(multi_state_clusters):,} clusters that span multiple states."
                )
                
                # Get records belonging to multi-state clusters
                multi_state_records_mask = gdf['final_cluster'].isin(multi_state_clusters)
                multi_state_records_df = gdf[multi_state_records_mask]
                
                # Create new cluster IDs for each (original_cluster, state) combination
                new_cluster_assignments = (
                    multi_state_records_df
                    .groupby(['final_cluster', 'addr_state'])
                    .ngroup()
                )
                
                # Apply new cluster IDs offset by current max
                gdf.loc[multi_state_records_mask, 'final_cluster'] = (
                    current_max_cluster_id + 1 + new_cluster_assignments.values
                )
                
                # Update max cluster ID for next splitting operation
                current_max_cluster_id = int(gdf['final_cluster'].max())
                clusters_split_count += len(multi_state_clusters)
            else:
                logger.debug("No clusters span multiple states.")
        
        # ========================================================================
        # Street Number Range Splitting
        # ========================================================================
        logger.debug(
            f"Checking for clusters with wide street number ranges "
            f"(threshold: {self.validation_config.street_number_threshold})..."
        )
        
        # Convert street numbers to numeric, keeping NaN for non-numeric values
        gdf['street_number_numeric'] = cudf.to_numeric(
            gdf['addr_street_number'], 
            errors='coerce'
        )
        
        # Group by cluster, street, and zip to find address ranges
        valid_clusters_with_address = gdf[
            (gdf['final_cluster'] != -1) & 
            (gdf['street_number_numeric'].notna())
        ].copy()
        
        if len(valid_clusters_with_address) > 0:
            # Calculate street number ranges for each (cluster, street, zip) group
            street_number_ranges = (
                valid_clusters_with_address
                .groupby(['final_cluster', 'addr_street_name', 'addr_zip'])
                ['street_number_numeric']
                .agg(['min', 'max', 'nunique'])
                .reset_index()
            )
            
            # Identify groups with problematic ranges
            # Criteria: Multiple unique numbers AND range exceeds threshold
            conflicting_address_ranges = street_number_ranges[
                (street_number_ranges['nunique'] > 1) &
                ((street_number_ranges['max'] - street_number_ranges['min']) > 
                 self.validation_config.street_number_threshold)
            ]
            
            if len(conflicting_address_ranges) > 0:
                conflict_group_count = len(conflicting_address_ranges)
                affected_cluster_count = conflicting_address_ranges['final_cluster'].nunique()
                
                logger.info(
                    f"Splitting {conflict_group_count:,} address groups across "
                    f"{affected_cluster_count:,} clusters with wide street number ranges."
                )
                
                # Mark conflicting groups for splitting
                conflict_markers_df = conflicting_address_ranges[
                    ['final_cluster', 'addr_street_name', 'addr_zip']
                ].copy()
                conflict_markers_df['has_range_conflict'] = True
                
                # Merge conflict markers with main DataFrame
                gdf = gdf.merge(
                    conflict_markers_df,
                    on=['final_cluster', 'addr_street_name', 'addr_zip'],
                    how='left'
                )
                
                # Fill NaN values in conflict column
                gdf['has_range_conflict'] = gdf['has_range_conflict'].fillna(False)
                
                # Get records with conflicts
                conflict_records_mask = gdf['has_range_conflict'] == True
                conflict_records_df = gdf[conflict_records_mask]
                
                if len(conflict_records_df) > 0:
                    # Create new cluster IDs for each unique address
                    new_range_based_clusters = (
                        conflict_records_df
                        .groupby([
                            'final_cluster', 
                            'addr_normalized_key'
                        ])
                        .ngroup()
                    )
                    
                    # Apply new cluster IDs
                    gdf.loc[conflict_records_mask, 'final_cluster'] = (
                        current_max_cluster_id + 1 + new_range_based_clusters.values
                    )
                    
                    clusters_split_count += affected_cluster_count
                
                # Clean up temporary conflict column
                gdf = gdf.drop(columns=['has_range_conflict'])
            else:
                logger.debug("No clusters have problematic street number ranges.")
        else:
            logger.debug("No valid addresses with numeric street numbers to check for conflicts.")
        
        # Clean up temporary numeric column
        gdf = gdf.drop(columns=['street_number_numeric'], errors='ignore')
        
        # Log final splitting summary
        if clusters_split_count > 0:
            final_cluster_count = gdf[gdf['final_cluster'] != -1]['final_cluster'].nunique()
            logger.info(
                f"Conflict splitting complete: {clusters_split_count:,} clusters were split. "
                f"Total clusters after splitting: {final_cluster_count:,}"
            )
        else:
            logger.debug("No cluster conflicts detected - no splits performed.")
        
        return gdf

    def _number_chain_entities(self, canonical_map_gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Identifies and numbers chain entities (same name at different addresses).
        
        When an entity name appears at multiple distinct addresses (e.g., franchise
        locations like "Starbucks"), this method appends location numbers to create
        unique canonical names like "Starbucks - 1", "Starbucks - 2", etc.
        
        Args:
            canonical_map_gdf: Canonical map DataFrame with entity names and addresses.
            
        Returns:
            cudf.DataFrame: Updated canonical map with numbered chain entities.
        """
        logger.info("Identifying and numbering chain entities...")
        
        if canonical_map_gdf.empty:
            logger.debug("Canonical map is empty - no chains to process.")
            return canonical_map_gdf
        
        # Count unique addresses per canonical name
        logger.debug("Counting unique addresses per entity name...")
        address_counts_per_name = (
            canonical_map_gdf
            .groupby('canonical_name')
            .agg(unique_address_count=('addr_normalized_key', 'nunique'))
            .reset_index()
        )
        
        # Identify names with multiple addresses (chains)
        chain_entity_names = address_counts_per_name.loc[
            address_counts_per_name['unique_address_count'] > 1, 
            'canonical_name'
        ]
        
        if chain_entity_names.empty:
            logger.info("No chain entities found - all entities have unique addresses.")
            return canonical_map_gdf
        
        chain_count = len(chain_entity_names)
        total_chain_locations = address_counts_per_name.loc[
            address_counts_per_name['unique_address_count'] > 1,
            'unique_address_count'
        ].sum()
        
        logger.info(
            f"Found {chain_count:,} chain entity names with "
            f"{total_chain_locations:,} total locations."
        )
        
        # Get unique name-address combinations for chains
        chain_records_mask = canonical_map_gdf['canonical_name'].isin(chain_entity_names)
        unique_chain_locations = (
            canonical_map_gdf.loc[
                chain_records_mask,
                ['canonical_name', 'addr_normalized_key']
            ]
            .drop_duplicates()
            .sort_values(['canonical_name', 'addr_normalized_key'])
            .reset_index(drop=True)
        )
        
        # Assign sequential numbers within each chain
        logger.debug("Assigning location numbers to chain entities...")
        unique_chain_locations['location_number'] = (
            unique_chain_locations
            .groupby('canonical_name')
            .cumcount()
            .astype('int32') + 1
        )
        
        # Merge location numbers back to canonical map
        enhanced_canonical_map = canonical_map_gdf.merge(
            unique_chain_locations,
            on=['canonical_name', 'addr_normalized_key'],
            how='left'
        )
        
        # Append location numbers to canonical names for chains
        has_location_number_mask = enhanced_canonical_map['location_number'].notna()
        
        if has_location_number_mask.any():
            # Create formatted location suffix
            location_suffix = (
                " - " + 
                enhanced_canonical_map.loc[has_location_number_mask, 'location_number']
                .astype('int32')
                .astype('str')
            )
            
            # Append suffix to canonical names
            enhanced_canonical_map.loc[has_location_number_mask, 'canonical_name'] = (
                enhanced_canonical_map.loc[has_location_number_mask, 'canonical_name'] + 
                location_suffix
            )
            
            logger.info(
                f"Numbered {has_location_number_mask.sum():,} chain entity locations."
            )
        
        # Remove temporary location number column
        enhanced_canonical_map = enhanced_canonical_map.drop(columns=['location_number'])
        
        logger.info("Chain entity numbering complete.")
        return enhanced_canonical_map
