# entity_resolver/merger.py
"""
GPU-Optimized Entity Resolution Cluster Merger Module

This module defines the ClusterMerger class, which handles post-clustering
refinement through intelligent cluster merging and entity consolidation.
All operations are optimized for GPU execution using cuDF's native operations.
"""

import logging

import cudf
import cupy

# --- Local Package Imports ---
from .config import (
    ClustererConfig,
    ValidationConfig,
    VectorizerConfig,
)
from .utils import (
    check_state_compatibility,
    check_street_number_compatibility,
    find_graph_components,
    find_similar_pairs,
    get_best_address_gpu,
    get_canonical_name_gpu,
)

# Set up a logger for this module with descriptive name
logger = logging.getLogger(__name__)


class ClusterMerger:
    """
    GPU-accelerated cluster merger for entity resolution pipelines.

    This class handles the sophisticated post-clustering refinement process,
    including merging over-split clusters and consolidating identical entities
    that were incorrectly separated during initial clustering.

    The merger uses graph-based algorithms to identify cluster relationships
    and applies similarity thresholds to determine which clusters should be merged.
    """

    def __init__(
        self,
        validation_config: ValidationConfig,
        vectorizer_config: VectorizerConfig,
        cluster_config: ClustererConfig,
    ):
        """
        Initialize the ClusterMerger with configuration parameters.

        Args:
            validation_config: Configuration containing validation rules and thresholds
                              including fuzzy matching ratios for names and addresses.
            vectorizer_config: Configuration for text vectorization and similarity
                              computation, including TF-IDF and nearest neighbor parameters.
        """
        self.validation_config = validation_config
        self.vectorizer_config = vectorizer_config
        self.cluster_config = cluster_config

        # Pre-compute similarity thresholds to avoid repeated division operations
        self.name_similarity_threshold = validation_config.name_fuzz_ratio / 100.0
        self.address_similarity_threshold = validation_config.address_fuzz_ratio / 100.0

    def merge_clusters(self, entity_dataframe: cudf.DataFrame) -> cudf.DataFrame:
        """
        Main orchestration method for the complete cluster merging process.

        This method executes a two-phase merging strategy:
        1. Similarity-based merging: Combines clusters with similar canonical representations
        2. Identity consolidation: Ensures identical entities share the same cluster

        Args:
            entity_dataframe: The cuDF DataFrame containing entities with initial cluster
                            assignments from the primary clustering algorithm.

        Returns:
            cudf.DataFrame: The input DataFrame with updated cluster labels after
                          all merging operations have been applied.
        """
        # Phase 1: Merge clusters based on similarity metrics
        entity_dataframe = self._merge_similar_clusters(entity_dataframe)

        # Phase 2: Consolidate any remaining identical entities
        entity_dataframe = self._consolidate_identical_entities(entity_dataframe)

        return entity_dataframe

    def _merge_similar_clusters(self, entity_dataframe: cudf.DataFrame) -> cudf.DataFrame:
        """
        Merges clusters that represent the same real-world entity using GPU-accelerated
        graph algorithms and similarity computations.

        This method is the core of the post-clustering refinement stage. It operates under
        the principle that if two clusters are highly similar in both their canonical name
        and address, they should be merged into a single entity. It finds these relationships
        transitively (i.e., if A is similar to B, and B is similar to C, then A, B, and C
        are all merged together).

        The process is as follows:
        1.  **Profile Creation**: A single, representative profile is created for each
            cluster, containing its canonical name, address, and size.
        2.  **Index Mapping**: A crucial step where a mapping is created between the
            cluster IDs and the temporary positional indices required by k-NN algorithms.
        3.  **Graph Construction**: Two similarity graphs are built—one for names and one
            for addresses. The function then finds the intersection of these graphs,
            creating a final edge list where an edge exists only if two clusters are
            similar in BOTH name AND address.
        4.  **Component Analysis**: A graph algorithm finds all connected components in the
            final similarity graph. Each component represents a group of clusters that
            should be merged.
        5.  **Mapping and Application**: For each component, a "winner" cluster is chosen
            (typically the largest), and a mapping is created to merge all other clusters
            in that component into the winner. This mapping is then applied to the main
            DataFrame in a single, efficient GPU operation.

        Args:
            entity_dataframe: The main DataFrame containing all entities and their current
                              cluster assignments from the previous stage.

        Returns:
            A cuDF DataFrame with the 'cluster' column updated to reflect the merged assignments.
        """
        logger.info('Starting GPU-accelerated cluster similarity merging process...')

        # Isolate only the records that were successfully clustered, excluding noise.
        entity_dataframe['cluster'] = entity_dataframe['cluster'].fillna(-1).astype('int32')
        clustered_entities = entity_dataframe[entity_dataframe['cluster'] != -1].copy()

        if clustered_entities.empty:
            logger.info('No clusters found for merging.')
            return entity_dataframe

        # --- Step 1: Create Cluster Profiles and Positional Index Mapping ---
        logger.debug('Building canonical profiles for each cluster...')
        cluster_profiles = self._build_cluster_profiles_gpu(clustered_entities)

        if cluster_profiles is None or cluster_profiles.empty:
            logger.info('No valid cluster profiles were created; skipping merge.')
            return entity_dataframe

        # CRITICAL: The find_similar_pairs function uses k-NN, which operates on
        # positional indices (0, 1, 2, ...). However, our cluster IDs are arbitrary
        # numbers. We must create a mapping to translate between these two systems.
        # First, ensure the profiles have a clean, sequential 0-based index.
        profiles_with_sequential_index = cluster_profiles.reset_index(drop=True)

        # Now, create a Series that maps the new positional index to the original cluster_id.
        # This will be our lookup table to convert the k-NN results back to cluster IDs.
        positional_index_to_cluster_id = profiles_with_sequential_index['cluster_id']

        # --- Step 2: Build Similarity Graph Between Clusters ---
        logger.info('Constructing cluster similarity graph for merge detection...')

        # Find pairs of clusters with similar canonical names.
        # The 'source' and 'destination' columns in the result are POSITIONAL INDICES.
        name_similarity_edges = find_similar_pairs(
            profiles_with_sequential_index['canonical_name_representation'],
            self.vectorizer_config.similarity_tfidf,
            self.vectorizer_config.similarity_nn,
            self.cluster_config.snn_clustering_params.merge_name_distance_threshold,
        )
        # Translate the positional indices back to actual cluster IDs using our mapping.
        name_edges_with_cluster_ids = cudf.DataFrame(
            {
                'source_cluster_id': positional_index_to_cluster_id.iloc[
                    name_similarity_edges['source']
                ].reset_index(drop=True),
                'destination_cluster_id': positional_index_to_cluster_id.iloc[
                    name_similarity_edges['destination']
                ].reset_index(drop=True),
            }
        )

        # Do the same for addresses.
        address_similarity_edges = find_similar_pairs(
            profiles_with_sequential_index['canonical_address_representation'],
            self.vectorizer_config.similarity_tfidf,
            self.vectorizer_config.similarity_nn,
            self.cluster_config.snn_clustering_params.merge_address_distance_threshold,
        )
        address_edges_with_cluster_ids = cudf.DataFrame(
            {
                'source_cluster_id': positional_index_to_cluster_id.iloc[
                    address_similarity_edges['source']
                ].reset_index(drop=True),
                'destination_cluster_id': positional_index_to_cluster_id.iloc[
                    address_similarity_edges['destination']
                ].reset_index(drop=True),
            }
        )

        # --- Validate and Filter Edges by State Compatibility ---
        logger.info('Validating similarity edges for state compatibility...')

        def filter_edges_by_state(
            edge_df: cudf.DataFrame, profiles_df: cudf.DataFrame
        ) -> cudf.DataFrame:
            """Helper function to filter an edge list based on state compatibility."""
            if edge_df.empty:
                return edge_df

            # Join to get the state for the source cluster
            edges_with_profiles = edge_df.merge(
                profiles_df.rename(
                    columns={
                        'cluster_id': 'source_cluster_id',
                        'canonical_state': 'source_state',
                        'canonical_street_number': 'source_num',
                    }
                ),
                on='source_cluster_id',
                how='left',
            )
            # Join to get the state for the destination cluster
            edges_with_profiles = edges_with_profiles.merge(
                profiles_df.rename(
                    columns={
                        'cluster_id': 'destination_cluster_id',
                        'canonical_state': 'dest_state',
                        'canonical_street_number': 'dest_num',
                    }
                ),
                on='destination_cluster_id',
                how='left',
            )

            # Run the compatibility check using the new standalone function
            state_ok = check_state_compatibility(
                edges_with_profiles['source_state'],
                edges_with_profiles['dest_state'],
                self.validation_config,
            )
            street_num_ok = check_street_number_compatibility(
                edges_with_profiles['source_num'],
                edges_with_profiles['dest_num'],
                self.validation_config.street_number_threshold,
            )

            # An edge is only valid if BOTH checks pass.
            compatibility_mask = state_ok & street_num_ok
            return edge_df[compatibility_mask.values]

        valid_name_edges = filter_edges_by_state(name_edges_with_cluster_ids, cluster_profiles)
        valid_address_edges = filter_edges_by_state(
            address_edges_with_cluster_ids, cluster_profiles
        )

        logger.info(
            f'State validation complete. Valid name edges: {len(valid_name_edges)}, Valid address edges: {len(valid_address_edges)}'
        )

        # The core merge requirement: an edge exists only if clusters are similar in BOTH name AND address.
        # We find this by performing an inner merge on the two edge lists.
        final_similarity_edges = valid_address_edges.merge(
            valid_name_edges, on=['source_cluster_id', 'destination_cluster_id'], how='inner'
        )

        if final_similarity_edges.empty:
            logger.info('No cluster pairs met the dual similarity thresholds for merging.')
            return entity_dataframe

        # --- Step 3: Find Connected Components in the Similarity Graph ---
        logger.debug('Identifying connected components in the final similarity graph...')

        # Each connected component represents a group of clusters that should be merged.
        graph_components = find_graph_components(
            edge_list_df=final_similarity_edges,
            source_column='source_cluster_id',
            destination_column='destination_cluster_id',
            directed=False,
            output_vertex_column='cluster_id',
            output_component_column='merge_component_id',
        )

        # Join the component IDs back to the original cluster profiles.
        profiles_with_components = cluster_profiles.merge(
            graph_components, on='cluster_id', how='left'
        )

        # --- Step 4 & 5: Create and Apply the Final Merge Mapping ---
        logger.debug('Determining winner clusters and creating the merge mapping...')

        # This helper function iterates through each component and selects a "winner"
        # (e.g., the largest cluster), creating a dictionary mapping losers to the winner.
        merge_map_df = self._create_merge_mapping_gpu(profiles_with_components)

        if merge_map_df.empty:
            logger.info('No clusters required merging after component analysis.')
            return entity_dataframe

        logger.info(f'Applying {len(merge_map_df)} cluster merges using GPU-native merge...')

        # Rename columns for a clean merge. 'cluster' is the key to join on.
        merge_map_df = merge_map_df.rename(
            columns={'loser_cluster_id': 'cluster', 'winner_cluster_id': 'new_cluster_id'}
        )

        # Perform a left merge. For rows in entity_dataframe that are losers,
        # 'new_cluster_id' will be populated. For all others, it will be null.
        entity_dataframe = entity_dataframe.merge(merge_map_df, on='cluster', how='left')

        # Coalesce the columns. If 'new_cluster_id' is null (i.e., the cluster
        # was not a loser), we keep the original 'cluster' value using fillna.
        entity_dataframe['cluster'] = (
            entity_dataframe['new_cluster_id'].fillna(entity_dataframe['cluster']).astype('int32')
        )

        # Drop the temporary 'new_cluster_id' column.
        entity_dataframe = entity_dataframe.drop(columns=['new_cluster_id'])

        return entity_dataframe

    def _build_cluster_profiles_gpu(
        self, clustered_entities: cudf.DataFrame
    ) -> cudf.DataFrame | None:
        """
        Builds canonical profiles for each cluster using a fully GPU-accelerated approach.

        This method leverages cuDF's `groupby().apply()` functionality, which
        executes a user-defined function (`_create_profile_for_group`) on each cluster's
        data in parallel on the GPU.

        The process is as follows:
        1.  Calculates the size of each cluster using a parallel `groupby().agg()`.
        2.  Executes the `_create_profile_for_group` static method for each cluster group
            on the GPU, generating the canonical name and address representations.
        3.  Cleans up the resulting DataFrame's multi-level index, which is an artifact
            of the `apply` operation.
        4.  Merges the cluster sizes and canonical representations into a final,
            comprehensive profiles DataFrame.

        Args:
            clustered_entities: A cuDF DataFrame containing only the entities that
                                have been assigned to a valid cluster (i.e., not noise).

        Returns:
            A cuDF DataFrame where each row represents a single cluster's canonical profile,
            or None if no profiles could be generated.
        """
        logger.debug('Building cluster profiles with GPU-native groupby-apply...')

        # First, calculate the size of each cluster in a single, parallel GPU operation.
        # We use 'raw_name' to count, but any non-null column would work.
        cluster_stats = (
            clustered_entities.groupby('cluster')
            .agg(cluster_entity_count=('raw_name', 'count'))
            .reset_index()
        )

        # Now, generate the canonical name and address for each cluster.
        # This is the key performance optimization: `groupby().apply()` will execute our
        # static method `_create_profile_for_group` on each group of data *in parallel on the GPU*,
        # avoiding the massive overhead of pulling data back to the CPU in a loop.
        canonical_profiles = clustered_entities.groupby('cluster').apply(
            self._create_profile_for_group
        )

        # Drop the index for a clean dataframe
        canonical_profiles = canonical_profiles.reset_index(drop=True)

        # Finally, merge the pre-calculated cluster sizes with the canonical representations.
        # This combines the two parallel computations into a single, comprehensive profiles DataFrame.
        # We join on the cluster identifiers and drop the redundant 'cluster' column from the merge.
        final_profiles = canonical_profiles.merge(
            cluster_stats, left_on='cluster_id', right_on='cluster', how='left'
        ).drop(columns=['cluster'])

        return final_profiles

    def _create_merge_mapping_gpu(self, profiles_with_components: cudf.DataFrame) -> cudf.DataFrame:
        """
        Creates a flat cluster merge-mapping DataFrame using GPU-native operations.

        For each connected component in the similarity graph, this method:
        1. Identifies the winner cluster (largest by entity count) using a
        performant sort and drop_duplicates operation.
        2. Creates a DataFrame mapping all other clusters ('losers') in that
        component directly to the winner.

        The mapping produced is inherently flat, meaning every loser maps directly
        to its final destination.

        Args:
            profiles_with_components: A cuDF DataFrame of cluster profiles that
                includes their assigned 'merge_component_id'.

        Returns:
            A cuDF DataFrame with two columns: ['loser_cluster_id', 'winner_cluster_id'],
            ready to be used in a merge operation.
        """
        # Filter to only clusters that are part of a multi-cluster component.
        clusters_in_components = profiles_with_components[
            profiles_with_components['merge_component_id'].notna()
        ]

        if clusters_in_components.empty:
            return cudf.DataFrame({'loser_cluster_id': [], 'winner_cluster_id': []})

        # Sort by component ID, then by cluster size in descending order.
        # This places the largest cluster (the winner) at the top of each group.
        sorted_clusters = clusters_in_components.sort_values(
            ['merge_component_id', 'cluster_entity_count'], ascending=[True, False]
        )

        # Use drop_duplicates to efficiently select the winner for each component.
        component_winners = sorted_clusters.drop_duplicates(
            subset=['merge_component_id'], keep='first'
        )[['merge_component_id', 'cluster_id']].copy()
        component_winners = component_winners.rename(columns={'cluster_id': 'winner_cluster_id'})

        # Join the winner ID back to all clusters in each component.
        clusters_with_winners = sorted_clusters.merge(
            component_winners, on='merge_component_id', how='left'
        )

        # A 'loser' is any cluster whose ID does not match the winner ID.
        loser_clusters = clusters_with_winners[
            clusters_with_winners['cluster_id'] != clusters_with_winners['winner_cluster_id']
        ]

        if loser_clusters.empty:
            return cudf.DataFrame({'loser_cluster_id': [], 'winner_cluster_id': []})

        # Create and return the final mapping DataFrame.
        merge_map_df = loser_clusters[['cluster_id', 'winner_cluster_id']].copy()
        merge_map_df.columns = ['loser_cluster_id', 'winner_cluster_id']

        return merge_map_df

    def _consolidate_identical_entities(self, entity_dataframe: cudf.DataFrame) -> cudf.DataFrame:
        """
        Consolidate entities with identical names and addresses using an
        end-to-end GPU-optimized pipeline.

        This method identifies cases where identical entities (based on a composite
        key of their normalized name and address) have been assigned to different
        clusters. It then merges these inconsistent assignments into a single,
        deterministically chosen "winner" cluster. The entire process, from key
        creation to final application, is designed to run on the GPU.

        Args:
            entity_dataframe: The main DataFrame with entities and their current
                              cluster assignments.

        Returns:
            A new cuDF DataFrame with the 'cluster' column updated to reflect
            the consolidated assignments.
        """
        logger.info('Starting GPU-accelerated entity consolidation...')

        # =============================================================================
        # Step 1: Data Preparation and Pre-computation
        # =============================================================================
        # Ensure the 'cluster' column is a consistent integer type and has no NaNs.
        entity_dataframe['cluster'] = entity_dataframe['cluster'].fillna(-1).astype('int32')

        # We only need to work with entities that have already been assigned a cluster.
        clustered_entities = entity_dataframe[entity_dataframe['cluster'] != -1].copy()

        if clustered_entities.empty:
            logger.info('No clustered entities found for consolidation.')
            return entity_dataframe

        # Create a single, unique key for each entity based on its name and address.
        # This key will be used to find identical entities.
        clustered_entities['composite_entity_key'] = (
            clustered_entities['normalized_text']
            + '|||'
            + clustered_entities['addr_normalized_key'].fillna('')
        )

        # =============================================================================
        # Step 2: Identify Inconsistent Entity Groups
        # =============================================================================
        # Group by the composite key to find all clusters associated with each unique entity.
        entity_cluster_groups = (
            clustered_entities.groupby('composite_entity_key')
            .agg(
                unique_clusters=('cluster', 'unique'),  # Get the list of unique cluster IDs
                cluster_count=('cluster', 'nunique'),  # Get the count of unique cluster IDs
            )
            .reset_index()
        )

        # An entity is "inconsistent" if its identical representations appear in more than one cluster.
        inconsistent_entities = entity_cluster_groups[entity_cluster_groups['cluster_count'] > 1]

        if inconsistent_entities.empty:
            logger.info(
                'No identical entities found across different clusters. Consolidation not needed.'
            )
            return entity_dataframe

        logger.warning(
            f'Found {len(inconsistent_entities)} identical entities in multiple clusters. Consolidating...'
        )

        # =============================================================================
        # Step 3: Generate the Consolidation Mapping
        # =============================================================================
        # This is the core logic step. We pass the full entity set and the identified
        # inconsistent groups to the mapping creation function. It will determine
        # the final source -> target mappings (e.g., cluster 101 -> cluster 50).
        consolidation_map_df = self._create_consolidation_mapping(
            entity_dataframe, inconsistent_entities
        )

        # =============================================================================
        # Step 4: Apply the Consolidation Map to the DataFrame
        # =============================================================================
        if consolidation_map_df.empty:
            logger.info('No cluster consolidations were necessary after analysis.')
            return entity_dataframe

        logger.info(
            f'Applying {len(consolidation_map_df)} cluster consolidations using GPU-native merge...'
        )

        # Prepare the mapping DataFrame for the merge operation.
        consolidation_map_df = consolidation_map_df.rename(
            columns={'source_cluster': 'cluster', 'target_cluster': 'new_cluster_id'}
        )

        # Use a left merge to bring the new, consolidated cluster IDs into the main DataFrame.
        # Rows that need consolidation will get a 'new_cluster_id'; others will have NaN.
        updated_df = entity_dataframe.merge(consolidation_map_df, on='cluster', how='left')

        # Coalesce the old and new cluster columns. If 'new_cluster_id' is not NaN,
        # use it; otherwise, keep the original 'cluster' value. This efficiently
        # updates only the rows that need changing.
        updated_df['cluster'] = updated_df['new_cluster_id'].fillna(updated_df['cluster'])
        updated_df['cluster'] = updated_df['cluster'].astype('int32')

        # Clean up the temporary column.
        final_df = updated_df.drop(columns=['new_cluster_id'])

        logger.info('Entity consolidation completed successfully.')
        return final_df

    def _create_consolidation_mapping(
        self, entity_dataframe: cudf.DataFrame, inconsistent_entities: cudf.DataFrame
    ) -> cudf.DataFrame:
        """
        Create a consolidation mapping using fully GPU-optimized vectorized operations.

        This method is a complete end-to-end GPU pipeline that:
        1. Explodes inconsistent entity groups into individual cluster-entity pairs.
        2. Calculates statistics for all affected clusters in parallel.
        3. Determines winner clusters using GPU sorting and grouping.
        4. Creates a flat mapping from source ('loser') to target ('winner') clusters.
        5. Resolves conflicts where a cluster appears in multiple groups.
        6. Flattens transitive mappings (e.g., A->B, B->C becomes A->C).

        Args:
            entity_dataframe: The complete DataFrame with all entities.
            inconsistent_entities: A DataFrame detailing which entities are in multiple clusters.

        Returns:
            A cuDF DataFrame with 'source_cluster' and 'target_cluster' columns
            representing the final, flattened consolidation mapping.
        """
        # =============================================================================
        # Initial Setup and Validation
        # =============================================================================
        # Filter out any entities that have not been assigned to a cluster (cluster == -1).
        # We only work with valid, clustered entities.
        valid_entities = entity_dataframe[entity_dataframe['cluster'] != -1].copy()

        # If there are no valid entities, no consolidation is possible.
        if valid_entities.empty:
            logger.debug(
                'No valid entities with cluster assignments found. Returning empty mapping.'
            )
            return cudf.DataFrame({'source_cluster': [], 'target_cluster': []})

        # Store the data type of the cluster column for consistent casting later.
        cluster_dtype = valid_entities['cluster'].dtype

        # =============================================================================
        # Step 1: Identify and Prepare All Clusters Involved in Inconsistencies
        # =============================================================================
        # The 'unique_clusters' column contains lists of clusters associated with an entity.
        # 'explode' transforms each item in the list into its own row, creating a long-format
        # DataFrame where each row is a (composite_entity_key, cluster_id) pair.
        exploded_entities = inconsistent_entities.explode('unique_clusters')
        exploded_entities = exploded_entities.dropna(subset=['unique_clusters'])

        # If exploding results in an empty DataFrame, there are no inconsistencies to resolve.
        if exploded_entities.empty:
            logger.debug('No inconsistent clusters to process. Returning empty mapping.')
            return cudf.DataFrame({'source_cluster': [], 'target_cluster': []})

        # Rename for clarity and ensure consistent data types.
        exploded_entities = exploded_entities.rename(columns={'unique_clusters': 'cluster_id'})
        exploded_entities['cluster_id'] = exploded_entities['cluster_id'].astype(cluster_dtype)

        # Get a unique series of all cluster IDs that are part of any inconsistency.
        # These are the only clusters we need to analyze further.
        affected_clusters = exploded_entities['cluster_id'].unique()
        affected_clusters = affected_clusters[affected_clusters != -1]

        if len(affected_clusters) == 0:
            logger.debug('No affected clusters found after filtering. Returning empty mapping.')
            return cudf.DataFrame({'source_cluster': [], 'target_cluster': []})

        # =============================================================================
        # Step 2: Calculate Statistics for Affected Clusters
        # =============================================================================
        # To decide a "winner" cluster, we need metrics. We calculate size and avg probability.

        # First, filter the main entity dataframe to only include rows with affected clusters.
        entities_in_affected_clusters = valid_entities[
            valid_entities['cluster'].isin(affected_clusters)
        ]

        # Calculate the size (number of entities) for each affected cluster.
        cluster_sizes_grouped = entities_in_affected_clusters.groupby('cluster').size()
        cluster_sizes = cluster_sizes_grouped.reset_index(name='cluster_size')
        cluster_sizes = cluster_sizes.rename(columns={'cluster': 'cluster_id'})

        # Initialize the cluster_stats DataFrame with the size information.
        cluster_stats = cluster_sizes

        # Dynamically find the probability column, if it exists.
        prob_cols = [c for c in valid_entities.columns if 'cluster_probability' in c.lower()]
        if prob_cols:
            prob_col_name = prob_cols[0]
            try:
                # Calculate the average cluster probability for each affected cluster.
                valid_entities[prob_col_name] = valid_entities[prob_col_name].astype('float32')
                cluster_probs_grouped = entities_in_affected_clusters.groupby('cluster')[
                    prob_col_name
                ].mean()
                cluster_probs = cluster_probs_grouped.reset_index(name='avg_probability')
                cluster_probs = cluster_probs.rename(columns={'cluster': 'cluster_id'})

                # Merge the probability stats with the size stats.
                cluster_stats = cluster_stats.merge(cluster_probs, on='cluster_id', how='left')
            except Exception as e:
                logger.warning(f'Could not process probability column {prob_col_name}: {e}')

        # Ensure 'avg_probability' column exists for consistent sorting, even if it could not be calculated.
        if 'avg_probability' not in cluster_stats.columns:
            cluster_stats['avg_probability'] = 0.0

        # Fill any missing values with 0.
        cluster_stats = cluster_stats.fillna(0)

        # =============================================================================
        # Step 3: Determine the "Winner" Cluster for Each Inconsistent Entity Group
        # =============================================================================
        # Join the calculated stats back to the exploded entity-cluster pairs.
        entities_with_stats = exploded_entities.merge(cluster_stats, on='cluster_id', how='left')
        entities_with_stats = entities_with_stats.dropna(subset=['cluster_size'])

        if entities_with_stats.empty:
            logger.debug('No entities with stats remained after merge. Returning empty mapping.')
            return cudf.DataFrame({'source_cluster': [], 'target_cluster': []})

        # Sort to determine the winner. The logic is:
        # 1. Group by each inconsistent entity (`composite_entity_key`).
        # 2. Within each group, the winner is the cluster with the largest `cluster_size`.
        # 3. If sizes are tied, the winner has the highest `avg_probability`.
        # 4. If still tied, the winner is the one with the lowest `cluster_id` for deterministic results.
        sort_order = ['composite_entity_key', 'cluster_size', 'avg_probability', 'cluster_id']
        ascending_order = [True, False, False, True]
        entities_sorted = entities_with_stats.sort_values(sort_order, ascending=ascending_order)

        # The first row for each `composite_entity_key` is now the winner.
        # We drop duplicates to isolate these winner rows.
        winner_rows = entities_sorted.drop_duplicates(subset=['composite_entity_key'], keep='first')
        winners = winner_rows[['composite_entity_key', 'cluster_id']]
        winners = winners.rename(columns={'cluster_id': 'winner_cluster_id'})

        # =============================================================================
        # Step 4: Generate Initial Source-to-Target ("Loser-to-Winner") Mappings
        # =============================================================================
        # Join the winner information back to the full list of sorted inconsistent clusters.
        clusters_with_winners = entities_sorted.merge(
            winners, on='composite_entity_key', how='left'
        )

        # Any cluster that is not the designated winner for its group is a "loser"
        # and needs to be mapped to the winner.
        loser_rows = clusters_with_winners[
            clusters_with_winners['cluster_id'] != clusters_with_winners['winner_cluster_id']
        ]
        loser_mappings = loser_rows[['cluster_id', 'winner_cluster_id']]

        if loser_mappings.empty:
            logger.debug('No loser-to-winner mappings were generated. Returning empty mapping.')
            return cudf.DataFrame({'source_cluster': [], 'target_cluster': []})

        # =============================================================================
        # Step 5: Resolve Mapping Conflicts
        # =============================================================================
        # A single cluster might be a "loser" in multiple different inconsistent groups,
        # potentially mapping it to different winners. This step resolves such conflicts.
        # For example, if we have A->B and A->C, we must choose one.
        resolved_mappings = self._resolve_cluster_conflicts(loser_mappings)

        # =============================================================================
        # Step 6: Flatten Transitive Mappings
        # =============================================================================
        # Resolve mapping chains. For example, if the mappings are A->B and B->C, this
        # step flattens them into a direct mapping: A->C. This is repeated until no
        # more chains exist.
        flattened_mappings = self._flatten_transitive_mappings_gpu(resolved_mappings)

        # =============================================================================
        # Step 7: Finalize the Consolidation Mapping
        # =============================================================================
        # Rename columns to the final desired schema: 'source_cluster' and 'target_cluster'.
        final_mapping = flattened_mappings.rename(
            columns={'cluster_id': 'source_cluster', 'winner_cluster_id': 'target_cluster'}
        )

        # As a final sanity check, remove any mappings where a cluster maps to itself.
        final_mapping = final_mapping[
            final_mapping['source_cluster'] != final_mapping['target_cluster']
        ]

        logger.debug(f'Created final consolidation mapping for {len(final_mapping)} clusters.')
        return final_mapping

    def _resolve_cluster_conflicts(self, loser_mappings: cudf.DataFrame) -> cudf.DataFrame:
        """
        Resolve conflicts where a single source cluster maps to multiple target clusters.

        A conflict arises when a single cluster_id (a "loser") is part of multiple
        inconsistent entity groups and, as a result, is mapped to different "winner"
        clusters. This function ensures that every source cluster maps to only one
        target cluster.

        The resolution strategy is deterministic:
        1. Count how many times each unique (source -> target) mapping occurs. This
           count serves as a proxy for the strength of the mapping.
        2. For each source cluster, select the target cluster that has the highest
           mapping count.
        3. If there's a tie in counts, the target cluster with the lowest numerical
           ID is chosen to ensure a consistent outcome.

        Args:
            loser_mappings: DataFrame with 'cluster_id' (source) and 'winner_cluster_id'
                            (target) columns, potentially containing conflicts.

        Returns:
            A DataFrame with conflicts resolved, containing a unique source-to-target
            mapping for each cluster.
        """
        # =============================================================================
        # Step 1: Count Occurrences of Each Unique Mapping
        # =============================================================================
        # Group by both the source ('cluster_id') and target ('winner_cluster_id') to
        # get a count for every unique mapping pair that exists.
        mapping_counts_grouped = loser_mappings.groupby(['cluster_id', 'winner_cluster_id']).size()
        mapping_counts = mapping_counts_grouped.reset_index(name='mapping_count')

        # =============================================================================
        # Step 2: Sort to Identify the Best Mapping for Each Source Cluster
        # =============================================================================
        # Sort the mappings to prepare for deduplication. The desired mapping will be
        # at the top for each 'cluster_id' group after sorting.
        #  - Primary sort by 'cluster_id' to group all mappings for the same source.
        #  - Secondary sort by 'mapping_count' descending, to prioritize the most
        #    frequent (strongest) mapping.
        #  - Tertiary sort by 'winner_cluster_id' ascending as a tie-breaker for
        #    deterministic results.
        sort_order = ['cluster_id', 'mapping_count', 'winner_cluster_id']
        ascending_order = [True, False, True]
        mappings_sorted = mapping_counts.sort_values(by=sort_order, ascending=ascending_order)

        # =============================================================================
        # Step 3: Deduplicate to Keep Only the Single Best Mapping
        # =============================================================================
        # After sorting, the first entry for each 'cluster_id' is the one we want to keep.
        # `drop_duplicates` with `keep='first'` achieves this.
        resolved_mappings_with_counts = mappings_sorted.drop_duplicates(
            subset=['cluster_id'], keep='first'
        )

        # The 'mapping_count' column is no longer needed. Select and return only the
        # final cluster_id -> winner_cluster_id pairs.
        final_resolved_mappings = resolved_mappings_with_counts[['cluster_id', 'winner_cluster_id']]

        logger.debug(
            f'Resolved conflicts for {len(loser_mappings) - len(final_resolved_mappings)} mappings.'
        )
        return final_resolved_mappings

    def _create_profile_for_group(
        self,
        cluster_group: cudf.DataFrame,
    ) -> cudf.DataFrame:
        """
        Creates a canonical profile for a single cluster group.

        This method is designed to be used within a GPU-accelerated
        `groupby().apply()` operation. It receives a DataFrame containing all
        records for a single cluster and calculates that cluster's canonical
        name and address.

        Args:
            cluster_group: A cuDF DataFrame containing all rows for one cluster.

        Returns:
            A single-row cuDF DataFrame containing the canonical representations
            for the cluster. This format is required by the `apply` function.
        """
        # =============================================================================
        # Step 1: Determine the Canonical Name Representation
        # =============================================================================
        # This step processes all 'normalized_text' entries for the cluster group
        # and calculates the single most representative name for the entire cluster.
        # The underlying `get_canonical_name_gpu` function handles the complex
        # similarity and scoring logic on the GPU.
        canonical_name = get_canonical_name_gpu(
            cluster_group['normalized_text'], self.vectorizer_config.similarity_tfidf
        )

        # =============================================================================
        # Step 2: Determine the Canonical Address Representation
        # =============================================================================
        # This step evaluates all address-related columns within the group to find the
        # single highest-quality, most complete address.
        best_address_row = get_best_address_gpu(cluster_group)

        # Extract the normalized address string from the best row found. If the
        # `get_best_address_gpu` function returned an empty DataFrame (i.e., no
        # valid addresses were found), we default to an empty string.
        if not best_address_row.empty:
            canonical_address = best_address_row['addr_normalized_key'].iloc[0]
        else:
            canonical_address = ''

        # =============================================================================
        # Step 3: Determine the Canonical State
        # =============================================================================
        # Find the most frequently occurring, non-null state within the cluster group.
        # This state will represent the canonical location of the entire cluster.
        valid_states = cluster_group['addr_state'].dropna()

        if not valid_states.empty:
            # value_counts() is a highly efficient way to get counts for unique values.
            # It returns a Series sorted by count, so the first index is the most common state.
            state_counts = valid_states.value_counts()
            canonical_state = state_counts.index[0]
        else:
            # If there are no valid states in the group, the canonical state is null.
            canonical_state = None

        # =============================================================================
        # Step 4: Canonical Street Number
        # =============================================================================
        numeric_street_numbers = cudf.to_numeric(
            cluster_group['addr_street_number'], errors='coerce'
        ).dropna()
        if not numeric_street_numbers.empty:
            # Find the most common valid street number.
            street_num_counts = numeric_street_numbers.value_counts()
            canonical_street_number = street_num_counts.index[0]
        else:
            canonical_street_number = None

        # =============================================================================
        # Step 5: Assemble the Final Profile DataFrame
        # =============================================================================
        # The `groupby().apply()` construct requires that the function returns a
        # DataFrame. We create a single-row DataFrame containing the cluster's ID
        # and its newly calculated canonical representations. These individual
        # DataFrames (one from each group) will be concatenated by cuDF into the
        # final result.

        # The cluster ID is guaranteed to be the same for all rows in this group,
        # so we can safely take it from the first row.
        cluster_id = int(cluster_group['cluster'].iloc[0])

        profile_df = cudf.DataFrame(
            {
                'cluster_id': [cluster_id],
                'canonical_name_representation': [canonical_name],
                'canonical_address_representation': [canonical_address],
                'canonical_state': [canonical_state],
                'canonical_street_number': [canonical_street_number],
            }
        )

        return profile_df

    def _flatten_transitive_mappings_gpu(self, mapping_df: cudf.DataFrame) -> cudf.DataFrame:
        """
        Flattens transitive mappings (e.g., A->B, B->C becomes A->C) on the GPU.

        This function uses a "pointer jumping" or "path doubling" algorithm, which is a
        highly efficient, parallel approach for finding the root nodes in a forest (a
        collection of directed trees). It is robust to cycles and converges in a
        logarithmic number of steps relative to the number of nodes.

        Args:
            mapping_df: A DataFrame with ['cluster_id', 'winner_cluster_id'] columns.
                        It is assumed that each 'cluster_id' maps to at most one winner.

        Returns:
            A DataFrame with the same columns, where each 'cluster_id' now maps
            directly to its ultimate final winner.
        """
        if mapping_df.empty:
            return mapping_df

        # =============================================================================
        # Step 1: Create a Dense, 0-Indexed ID Space for All Graph Nodes
        # =============================================================================
        # GPU array operations require dense, 0-based integer indices for direct memory
        # access. Our original 'cluster_id' values can be large and sparse. This step
        # maps every unique cluster ID to a new, dense ID from 0 to N-1.

        # Gather all unique cluster IDs from both source and target columns.
        source_ids = mapping_df['cluster_id']
        target_ids = mapping_df['winner_cluster_id']
        all_unique_ids = cudf.concat([source_ids, target_ids]).unique().dropna()
        all_unique_ids = all_unique_ids.astype('int32')
        n_nodes = len(all_unique_ids)

        # Create a lookup table (like a dictionary) to map from the original sparse
        # ID to the new dense ID (0, 1, 2, ...).
        id_to_dense_map = cudf.DataFrame(
            {'original_id': all_unique_ids, 'dense_id': cupy.arange(n_nodes, dtype='int32')}
        )

        # =============================================================================
        # Step 2: Map Original Mappings to the New Dense ID Space
        # =============================================================================
        # Now, translate the input mappings from original IDs to dense IDs.

        # Create a temporary DataFrame for merging.
        dense_mappings = mapping_df.copy()

        # Merge to find the dense ID for each source ('cluster_id').
        dense_mappings = dense_mappings.merge(
            id_to_dense_map, left_on='cluster_id', right_on='original_id', how='left'
        ).rename(columns={'dense_id': 'dense_source_id'})

        # Merge again to find the dense ID for each target ('winner_cluster_id').
        dense_mappings = dense_mappings.merge(
            id_to_dense_map, left_on='winner_cluster_id', right_on='original_id', how='left'
        ).rename(columns={'dense_id': 'dense_target_id'})

        # Extract the dense source and target IDs as cupy arrays for the algorithm.
        dense_sources = dense_mappings['dense_source_id'].dropna().astype('int32')
        dense_targets = dense_mappings['dense_target_id'].dropna().astype('int32')

        # =============================================================================
        # Step 3: Initialize and Run the Pointer Jumping Algorithm
        # =============================================================================
        # The 'parent' array stores the graph. parent[i] = j means node i -> node j.
        # Initially, every node is its own parent.
        parent = cupy.arange(n_nodes, dtype='int32')

        # Apply the initial mappings: each source node points to its target node.
        parent[dense_sources.values] = dense_targets.values

        # Iteratively "jump" pointers. In each step, every node updates its parent to
        # its current grandparent (parent[parent]). This is the core of the "path doubling"
        # algorithm, which doubles the path length to the root in each iteration,
        # leading to O(log N) convergence.
        max_steps = int(cupy.ceil(cupy.log2(cupy.asarray(max(n_nodes, 1), dtype='float32')))) + 2
        for _ in range(max_steps):
            new_parent = parent[parent]
            # If no pointers changed in an iteration, the graph has stabilized and we can exit early.
            if cupy.all(new_parent == parent):
                break
            parent = new_parent

        # =============================================================================
        # Step 4: Determine the Final Root/Winner for Each Original Source
        # =============================================================================
        # After the loop, `parent[i]` holds the ultimate root of the tree that node `i` belongs to.
        # We only need the final destination for the initial set of source nodes.
        final_dense_targets = parent[dense_sources.values]

        # =============================================================================
        # Step 5: Map Dense IDs Back to Original Cluster IDs
        # =============================================================================
        # We now have the final mappings in the dense space. Translate them back to
        # the original, sparse cluster IDs. The `id_to_dense_map` is now used in reverse.
        dense_to_id_map = id_to_dense_map.set_index('dense_id')

        final_sources_series = dense_to_id_map.loc[dense_sources]['original_id']
        final_targets_series = dense_to_id_map.loc[cudf.Series(final_dense_targets)]['original_id']

        # =============================================================================
        # Step 6: Assemble and Return the Final Flattened Mapping
        # =============================================================================
        flat_mapping = cudf.DataFrame(
            {'cluster_id': final_sources_series, 'winner_cluster_id': final_targets_series.values}
        )

        # Ensure the output dtypes match the original input dtypes.
        flat_mapping['cluster_id'] = flat_mapping['cluster_id'].astype('int32')
        flat_mapping['winner_cluster_id'] = flat_mapping['winner_cluster_id'].astype('int32')

        # Final cleanup to remove any duplicates that might arise and reset the index.
        return flat_mapping.drop_duplicates().reset_index(drop=True)
