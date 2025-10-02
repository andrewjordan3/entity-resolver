# entity_resolver/merger.py
"""
GPU-Optimized Entity Resolution Cluster Merger Module

This module defines the ClusterMerger class, which handles post-clustering
refinement through intelligent cluster merging and entity consolidation.
All operations are optimized for GPU execution using cuDF's native operations.
"""

import cudf
import logging
from typing import Dict, Optional, Set

# --- Local Package Imports ---
from .config import ValidationConfig, VectorizerConfig
from .utils import (
    get_canonical_name_gpu, 
    get_best_address_gpu, 
    find_similar_pairs, 
    find_graph_components,
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
    
    def __init__(self, validation_config: ValidationConfig, vectorizer_config: VectorizerConfig):
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
        
        # Pre-compute similarity thresholds to avoid repeated division operations
        self.name_similarity_threshold = validation_config.name_fuzz_ratio / 100.0
        self.address_similarity_threshold = validation_config.address_fuzz_ratio / 100.0

        ##################################################
        # Need to make the following into parameters
        ###################################################
        self.name_merging_distance_threshold: float = 0.02
        self.address_merging_distance_threshold: float = 0.01

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
        logger.info("Starting GPU-accelerated cluster similarity merging process...")
        
        # Isolate only the records that were successfully clustered, excluding noise.
        clustered_entities = entity_dataframe[entity_dataframe['cluster'] != -1].copy()
        
        if clustered_entities.empty:
            logger.info("No clusters found for merging.")
            return entity_dataframe
        
        # --- Step 1: Create Cluster Profiles and Positional Index Mapping ---
        logger.debug("Building canonical profiles for each cluster...")
        cluster_profiles = self._build_cluster_profiles_gpu(clustered_entities)
        
        if cluster_profiles is None or cluster_profiles.empty:
            logger.info("No valid cluster profiles were created; skipping merge.")
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
        logger.info("Constructing cluster similarity graph for merge detection...")
        
        # Find pairs of clusters with similar canonical names.
        # The 'source' and 'destination' columns in the result are POSITIONAL INDICES.
        name_similarity_edges = find_similar_pairs(
            profiles_with_sequential_index['canonical_name_representation'], 
            self.vectorizer_config.similarity_tfidf, 
            self.vectorizer_config.similarity_nn,
            self.name_merging_distance_threshold
        )
        # Translate the positional indices back to actual cluster IDs using our mapping.
        name_edges_with_cluster_ids = cudf.DataFrame({
            'source_cluster_id': positional_index_to_cluster_id.iloc[name_similarity_edges['source']].reset_index(drop=True),
            'destination_cluster_id': positional_index_to_cluster_id.iloc[name_similarity_edges['destination']].reset_index(drop=True)
        })

        # Do the same for addresses.
        address_similarity_edges = find_similar_pairs(
            profiles_with_sequential_index['canonical_address_representation'], 
            self.vectorizer_config.similarity_tfidf, 
            self.vectorizer_config.similarity_nn, 
            self.address_merging_distance_threshold
        )
        address_edges_with_cluster_ids = cudf.DataFrame({
            'source_cluster_id': positional_index_to_cluster_id.iloc[address_similarity_edges['source']].reset_index(drop=True),
            'destination_cluster_id': positional_index_to_cluster_id.iloc[address_similarity_edges['destination']].reset_index(drop=True)
        })
        
        # The core merge requirement: an edge exists only if clusters are similar in BOTH name AND address.
        # We find this by performing an inner merge on the two edge lists.
        final_similarity_edges = address_edges_with_cluster_ids.merge(
            name_edges_with_cluster_ids, 
            on=['source_cluster_id', 'destination_cluster_id'],
            how='inner'
        )
        
        if final_similarity_edges.empty:
            logger.info("No cluster pairs met the dual similarity thresholds for merging.")
            return entity_dataframe
            
        # --- Step 3: Find Connected Components in the Similarity Graph ---
        logger.debug("Identifying connected components in the final similarity graph...")
        
        # Each connected component represents a group of clusters that should be merged.
        graph_components = find_graph_components(
            edge_list_df=final_similarity_edges, 
            source_column='source_cluster_id', 
            destination_column='destination_cluster_id', 
            directed=False, 
            output_vertex_column='cluster_id', 
            output_component_column='merge_component_id'
        )
        
        # Join the component IDs back to the original cluster profiles.
        profiles_with_components = cluster_profiles.merge(
            graph_components,
            on='cluster_id',
            how='left'
        )
        
        # --- Step 4 & 5: Create and Apply the Final Merge Mapping ---
        logger.debug("Determining winner clusters and creating the merge mapping...")
        
        # This helper function iterates through each component and selects a "winner"
        # (e.g., the largest cluster), creating a dictionary mapping losers to the winner.
        cluster_merge_mapping = self._create_merge_mapping_gpu(profiles_with_components)
        
        if not cluster_merge_mapping:
            logger.info("No clusters required merging after component analysis.")
            return entity_dataframe
            
        logger.info(f"Applying {len(cluster_merge_mapping)} cluster merges...")
        
        # Flatten the mapping to handle transitive merges (A->B, B->C becomes A->C, B->C).
        flat_mapping = self._flatten_mapping(cluster_merge_mapping)
        
        # Apply the final mapping to the 'cluster' column in one GPU-accelerated operation.
        entity_dataframe['cluster'] = entity_dataframe['cluster'].replace(flat_mapping)
        
        logger.info("Cluster similarity merging completed successfully.")
        return entity_dataframe
    
    def _build_cluster_profiles_gpu(self, clustered_entities: cudf.DataFrame) -> Optional[cudf.DataFrame]:
        """
        Build cluster profiles entirely on GPU without loops or pandas conversions.
        
        This method creates a canonical representation for each cluster including:
        - Canonical name (most representative entity name)
        - Canonical address (best quality address)
        - Cluster size (number of entities)
        
        Args:
            clustered_entities: DataFrame containing only entities with valid cluster assignments
            
        Returns:
            cudf.DataFrame: Cluster profiles with canonical representations, or None if empty
        """
        # Use groupby aggregation to get cluster sizes in parallel
        cluster_sizes = clustered_entities.groupby('cluster').size().reset_index()
        cluster_sizes.columns = ['cluster_id', 'cluster_entity_count']
        
        # For canonical names, we'll need to call the utility function per cluster
        # But we can optimize by doing batch operations where possible
        
        # Create a function to get canonical representations for all clusters at once
        canonical_names_list = []
        canonical_addresses_list = []
        cluster_ids_list = []
        
        # Group the dataframe once and iterate through groups
        grouped_entities = clustered_entities.groupby('cluster')
        
        for cluster_id, cluster_group in grouped_entities:
            # Get canonical name for this cluster
            canonical_name = get_canonical_name_gpu(
                cluster_group['normalized_text'], 
                self.vectorizer_config.similarity_tfidf
            )
            
            # Get best address for this cluster
            best_address_row = get_best_address_gpu(cluster_group)
            
            if not best_address_row.empty:
                canonical_address = best_address_row['addr_normalized_key'].iloc[0]
            else:
                canonical_address = ""
            
            canonical_names_list.append(canonical_name)
            canonical_addresses_list.append(canonical_address)
            cluster_ids_list.append(cluster_id)
        
        if not cluster_ids_list:
            return None
        
        # Create the profiles dataframe entirely on GPU
        cluster_profiles = cudf.DataFrame({
            'cluster_id': cluster_ids_list,
            'canonical_name_representation': canonical_names_list,
            'canonical_address_representation': canonical_addresses_list
        })
        
        # Merge with cluster sizes
        cluster_profiles = cluster_profiles.merge(
            cluster_sizes, 
            on='cluster_id',
            how='left'
        )
        
        return cluster_profiles
    
    def _create_merge_mapping_gpu(self, profiles_with_components: cudf.DataFrame) -> Dict:
        """
        Create cluster merge mapping using GPU-native operations.
        
        For each connected component in the similarity graph, this method:
        1. Identifies the winner cluster (largest by entity count)
        2. Maps all other clusters in the component to the winner
        
        Args:
            profiles_with_components: Cluster profiles with component assignments
            
        Returns:
            Dict: Mapping from source cluster IDs to target (winner) cluster IDs
        """
        # Filter to only clusters that are part of a component
        clusters_in_components = profiles_with_components[
            profiles_with_components['merge_component_id'].notna()
        ]
        
        if len(clusters_in_components) == 0:
            return {}
        
        # Sort by component and size (descending) to identify winners
        sorted_clusters = clusters_in_components.sort_values(
            ['merge_component_id', 'cluster_entity_count'], 
            ascending=[True, False]
        )
        
        # The first cluster in each component is the winner (largest)
        component_winners = sorted_clusters.drop_duplicates(
            subset=['merge_component_id'], 
            keep='first'
        )[['merge_component_id', 'cluster_id']].copy()
        component_winners.columns = ['merge_component_id', 'winner_cluster_id']
        
        # Join back to get losers (non-winners) and their target winners
        clusters_with_winners = sorted_clusters.merge(
            component_winners, 
            on='merge_component_id',
            how='left'
        )
        
        # Filter to only losers (clusters that need to be remapped)
        loser_clusters = clusters_with_winners[
            clusters_with_winners['cluster_id'] != clusters_with_winners['winner_cluster_id']
        ]
        
        # Create the merge mapping dictionary
        # Convert to pandas only for the final dictionary creation
        merge_mapping = loser_clusters[['cluster_id', 'winner_cluster_id']].to_pandas()
        merge_mapping_dict = dict(zip(
            merge_mapping['cluster_id'], 
            merge_mapping['winner_cluster_id']
        ))
        
        # Flatten transitive mappings
        merge_mapping_dict = self._flatten_mapping(merge_mapping_dict)
        
        return merge_mapping_dict
    
    def _consolidate_identical_entities(self, entity_dataframe: cudf.DataFrame) -> cudf.DataFrame:
        """
        Consolidate entities with identical normalized names and addresses into 
        the same cluster using GPU-optimized operations.
        
        This cleanup step ensures consistency by detecting and fixing cases where
        identical entities ended up in different clusters due to edge cases in
        the clustering algorithm.
        
        Args:
            entity_dataframe: DataFrame with entities and their cluster assignments
            
        Returns:
            cudf.DataFrame: DataFrame with consolidated cluster assignments
        """
        logger.info("Starting entity consolidation for identical name-address pairs...")
        
        # Filter to clustered entities only
        clustered_entities = entity_dataframe[entity_dataframe['cluster'] != -1].copy()
        
        if len(clustered_entities) == 0:
            return entity_dataframe
        
        # Create composite entity key for exact matching
        # Using fillna directly on GPU to handle missing addresses
        clustered_entities['composite_entity_key'] = (
            clustered_entities['normalized_text'] + '|||' + 
            clustered_entities['addr_normalized_key'].fillna('')
        )
        
        # Group by entity key and find keys with multiple clusters
        entity_cluster_groups = clustered_entities.groupby('composite_entity_key').agg(
            unique_clusters=('cluster', 'unique'),
            cluster_count=('cluster', 'nunique')
        ).reset_index()
        
        # Filter to only entity keys that appear in multiple clusters
        inconsistent_entities = entity_cluster_groups[entity_cluster_groups['cluster_count'] > 1]
        
        if len(inconsistent_entities) == 0:
            logger.info("No identical entities found across different clusters.")
            return entity_dataframe
        
        logger.warning(f"Found {len(inconsistent_entities)} identical entities in multiple clusters. Consolidating...")
        
        # Try the GPU-optimized batch method first
        try:
            consolidation_mapping = self._create_consolidation_mapping_gpu_optimized(
                entity_dataframe,
                inconsistent_entities
            )
        except Exception as e:
            logger.debug(f"Batch consolidation failed, using fallback method: {e}")
            # Fallback to iterative method if batch method fails
            consolidation_mapping = self._create_consolidation_mapping_gpu(
                clustered_entities, 
                inconsistent_entities,
                entity_dataframe
            )
        
        # Apply consolidation mapping if any merges are needed
        if consolidation_mapping:
            logger.info(f"Consolidating {len(consolidation_mapping)} clusters...")
            # Flatten is already applied in the creation methods
            entity_dataframe['cluster'] = entity_dataframe['cluster'].replace(consolidation_mapping)
        
        logger.info("Entity consolidation completed successfully.")
        return entity_dataframe
    
    def _create_consolidation_mapping_gpu(
        self, 
        clustered_entities: cudf.DataFrame,
        inconsistent_entities: cudf.DataFrame,
        full_dataframe: cudf.DataFrame
    ) -> Dict:
        """
        Create consolidation mapping for identical entities using GPU operations.
        
        This method is the fallback when batch operations fail. It processes
        each inconsistent entity iteratively but still uses GPU operations
        for statistics calculation.
        
        Args:
            clustered_entities: DataFrame of entities with valid clusters
            inconsistent_entities: DataFrame of entity keys appearing in multiple clusters
            full_dataframe: Complete DataFrame for calculating cluster statistics
            
        Returns:
            Dict: Mapping from source cluster IDs to consolidation target cluster IDs
        """
        consolidation_map = {}
        
        # GUARDRAIL: Filter to valid clusters only
        valid_full_dataframe = full_dataframe[full_dataframe['cluster'] != -1]
        
        # Check and disambiguate cluster_probability column
        probability_columns = [col for col in valid_full_dataframe.columns 
                              if 'cluster_probability' in col.lower()]
        
        has_probability_column = len(probability_columns) > 0
        probability_col_name = None
        
        if has_probability_column:
            # Use the first matching column, or the exact match if it exists
            probability_col_name = ('cluster_probability' 
                                   if 'cluster_probability' in probability_columns 
                                   else probability_columns[0])
            
            # Ensure the column is numeric
            try:
                valid_full_dataframe[probability_col_name] = (
                    valid_full_dataframe[probability_col_name].astype('float32')
                )
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not convert {probability_col_name} to numeric: {e}")
                has_probability_column = False
        
        # Process each inconsistent entity key
        for idx, row in inconsistent_entities.iterrows():
            entity_key = row['composite_entity_key']
            affected_clusters = row['unique_clusters']
            
            # Skip if affected_clusters is null or empty
            if affected_clusters is None or len(affected_clusters) == 0:
                continue
            
            # Filter out noise clusters (-1) from affected clusters
            valid_affected_clusters = [c for c in affected_clusters if c != -1]
            
            if len(valid_affected_clusters) == 0:
                continue
            
            # Get statistics for each affected cluster
            cluster_statistics = []
            
            for cluster_id in valid_affected_clusters:
                # Ensure dtype consistency for comparison
                cluster_mask = (valid_full_dataframe['cluster'] == cluster_id)
                
                # Calculate cluster size using GPU operations
                cluster_size_series = cluster_mask.sum()
                # Properly extract scalar value
                if hasattr(cluster_size_series, 'item'):
                    cluster_size = int(cluster_size_series.item())
                elif hasattr(cluster_size_series, 'values'):
                    cluster_size = int(cluster_size_series.values[0])
                else:
                    cluster_size = int(cluster_size_series)
                
                # Skip empty clusters
                if cluster_size == 0:
                    continue
                
                # Calculate average probability if column exists
                average_probability = 0.0  # Default value
                if has_probability_column and probability_col_name:
                    cluster_probabilities = valid_full_dataframe.loc[cluster_mask, probability_col_name]
                    if len(cluster_probabilities) > 0:
                        # Properly extract scalar from cuDF Series mean
                        mean_value = cluster_probabilities.mean()
                        # Handle both cuDF Series and scalar returns
                        if hasattr(mean_value, 'item'):
                            average_probability = float(mean_value.item())
                        elif hasattr(mean_value, 'values'):
                            # For cuDF Series with single value
                            average_probability = float(mean_value.values[0])
                        else:
                            # Fallback for direct scalar
                            try:
                                average_probability = float(mean_value)
                            except (TypeError, ValueError):
                                average_probability = 0.0
                
                cluster_statistics.append({
                    'cluster_id': cluster_id,
                    'size': cluster_size,
                    'avg_probability': average_probability
                })
            
            # Skip if no valid clusters found
            if len(cluster_statistics) == 0:
                continue
            
            # Select winner based on size first, then average probability, then cluster_id for determinism
            winner_cluster = sorted(
                cluster_statistics, 
                key=lambda x: (x['size'], x['avg_probability'], -x['cluster_id']), 
                reverse=True
            )[0]
            
            winner_cluster_id = winner_cluster['cluster_id']
            
            # Map all non-winner clusters to the winner
            for cluster_id in valid_affected_clusters:
                if cluster_id != winner_cluster_id:
                    consolidation_map[cluster_id] = winner_cluster_id
        
        return consolidation_map
    
    def _flatten_mapping(self, mapping: Dict[int, int]) -> Dict[int, int]:
        """
        Flatten transitive mappings to ensure single-pass replacement works correctly.
        
        Collapses chains like A->B, B->C into A->C so a single replace() is sufficient.
        Also guards against accidental cycles by stopping when a loop is detected.
        
        Args:
            mapping: Original mapping dictionary that may contain chains
            
        Returns:
            Dict: Flattened mapping where each key maps directly to its final destination
        """
        flattened_mapping: Dict[int, int] = {}
        
        for start_cluster in mapping.keys():
            visited_clusters: Set[int] = {start_cluster}
            destination_cluster = mapping[start_cluster]
            
            # Follow the chain until we reach the end or detect a cycle
            while destination_cluster in mapping and destination_cluster not in visited_clusters:
                visited_clusters.add(destination_cluster)
                destination_cluster = mapping[destination_cluster]
            
            # If we hit a cycle (destination in visited), stop at current destination
            flattened_mapping[start_cluster] = destination_cluster
        
        return flattened_mapping
    
    def _resolve_cluster_conflicts(self, loser_mappings: cudf.DataFrame) -> cudf.DataFrame:
        """
        Resolve conflicts where a single cluster_id appears under multiple entity keys
        with different winners.
        
        When a cluster appears in multiple entity groups, we need to pick ONE winner
        deterministically. We choose the winner from the largest entity group.
        
        Args:
            loser_mappings: DataFrame with cluster_id and winner_cluster_id columns,
                          potentially containing conflicts
                          
        Returns:
            cudf.DataFrame: Deduplicated mappings with conflicts resolved
        """
        # Count how many times each loser->winner mapping appears
        mapping_counts = loser_mappings.groupby(
            ['cluster_id', 'winner_cluster_id']
        ).size().reset_index()
        mapping_counts.columns = ['cluster_id', 'winner_cluster_id', 'mapping_count']
        
        # For each cluster_id, pick the winner with the highest count
        # If tied, use winner_cluster_id as tiebreaker (smallest ID wins)
        mapping_counts_sorted = mapping_counts.sort_values(
            ['cluster_id', 'mapping_count', 'winner_cluster_id'],
            ascending=[True, False, True]
        )
        
        # Keep only the first (best) mapping for each cluster_id
        resolved_mappings = mapping_counts_sorted.drop_duplicates(
            subset=['cluster_id'],
            keep='first'
        )[['cluster_id', 'winner_cluster_id']]
        
        return resolved_mappings
    
    def _create_consolidation_mapping_gpu_optimized(
        self,
        entity_dataframe: cudf.DataFrame,
        inconsistent_entities: cudf.DataFrame
    ) -> Dict:
        """
        Create consolidation mapping using fully GPU-optimized batch operations.
        
        This method minimizes loops by using vectorized operations to process
        all inconsistent entities in parallel where possible.
        
        Args:
            entity_dataframe: Complete DataFrame for calculating cluster statistics
            inconsistent_entities: DataFrame of entity keys appearing in multiple clusters
            
        Returns:
            Dict: Mapping from source cluster IDs to consolidation target cluster IDs
        """
        # GUARDRAIL 1: Work only on valid clusters (exclude noise points)
        valid_entity_dataframe = entity_dataframe[entity_dataframe['cluster'] != -1].copy()
        
        if len(valid_entity_dataframe) == 0:
            logger.debug("No valid clusters found for consolidation statistics.")
            return {}
        
        # GUARDRAIL 2: Ensure cluster column dtype alignment
        # Cast cluster to the same dtype for consistent joining
        cluster_dtype = valid_entity_dataframe['cluster'].dtype
        
        # Explode the unique_clusters column to get individual cluster-entity pairs
        exploded_entities = inconsistent_entities.explode('unique_clusters')
        
        # GUARDRAIL 3: Handle potential nulls from explode operation
        exploded_entities = exploded_entities.dropna(subset=['unique_clusters'])
        
        if len(exploded_entities) == 0:
            logger.debug("No valid clusters after exploding inconsistent entities.")
            return {}
        
        # Rename and ensure dtype consistency
        exploded_entities = exploded_entities.rename(columns={'unique_clusters': 'cluster_id'})
        exploded_entities['cluster_id'] = exploded_entities['cluster_id'].astype(cluster_dtype)
        
        # Calculate cluster statistics for all affected clusters at once
        affected_clusters = exploded_entities['cluster_id'].unique()
        
        # Filter to only valid affected clusters (double-check no -1 values)
        affected_clusters = affected_clusters[affected_clusters != -1]
        
        if len(affected_clusters) == 0:
            logger.debug("No valid affected clusters found.")
            return {}
        
        # Get cluster sizes using groupby (only valid clusters)
        cluster_sizes = valid_entity_dataframe[
            valid_entity_dataframe['cluster'].isin(affected_clusters)
        ].groupby('cluster').size().reset_index()
        cluster_sizes.columns = ['cluster_id', 'cluster_size']
        
        # GUARDRAIL 4: Check and disambiguate cluster_probability column
        probability_columns = [col for col in valid_entity_dataframe.columns 
                              if 'cluster_probability' in col.lower()]
        
        has_probability_column = len(probability_columns) > 0
        
        if has_probability_column:
            # Use the first matching column, or the exact match if it exists
            probability_col_name = ('cluster_probability' 
                                   if 'cluster_probability' in probability_columns 
                                   else probability_columns[0])
            
            logger.debug(f"Using probability column: {probability_col_name}")
            
            # Ensure the probability column is numeric with float64 for stable means
            try:
                valid_entity_dataframe[probability_col_name] = (
                    valid_entity_dataframe[probability_col_name].astype('float64')
                )
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not convert {probability_col_name} to numeric: {e}")
                has_probability_column = False
        
        # Get average probabilities if the column exists and is valid
        if has_probability_column:
            cluster_probabilities = valid_entity_dataframe[
                valid_entity_dataframe['cluster'].isin(affected_clusters)
            ].groupby('cluster')[probability_col_name].mean().reset_index()
            cluster_probabilities.columns = ['cluster_id', 'avg_probability']
            
            # Merge statistics with explicit dtype alignment
            cluster_stats = cluster_sizes.merge(
                cluster_probabilities, 
                on='cluster_id', 
                how='left'
            )
            # Fill NaN probabilities with 0
            cluster_stats['avg_probability'] = cluster_stats['avg_probability'].fillna(0.0)
        else:
            cluster_stats = cluster_sizes.copy()
            cluster_stats['avg_probability'] = 0.0
        
        # Ensure cluster_size is not null (shouldn't happen, but being defensive)
        cluster_stats['cluster_size'] = cluster_stats['cluster_size'].fillna(0)
        
        # Join cluster statistics back to exploded entities
        entities_with_stats = exploded_entities.merge(
            cluster_stats,
            on='cluster_id',
            how='left'
        )
        
        # Drop any rows where we couldn't find cluster statistics
        entities_with_stats = entities_with_stats.dropna(subset=['cluster_size'])
        
        if len(entities_with_stats) == 0:
            logger.debug("No entities with valid cluster statistics found.")
            return {}
        
        # GUARDRAIL 5: Add deterministic tie-breaking
        # Add cluster_id as a third sort key for fully deterministic results
        # Sort by entity key, then by size (desc), probability (desc), and cluster_id (asc) for ties
        entities_sorted = entities_with_stats.sort_values(
            ['composite_entity_key', 'cluster_size', 'avg_probability', 'cluster_id'],
            ascending=[True, False, False, True]
        )
        
        # Get the first (winner) cluster for each entity key
        winners = entities_sorted.drop_duplicates(
            subset=['composite_entity_key'],
            keep='first'
        )[['composite_entity_key', 'cluster_id']].rename(
            columns={'cluster_id': 'winner_cluster_id'}
        )
        
        # Join back to get all clusters and their winners
        clusters_with_winners = entities_sorted.merge(
            winners,
            on='composite_entity_key',
            how='left'
        )
        
        # Filter to get only the losers (non-winner clusters)
        loser_mappings = clusters_with_winners[
            clusters_with_winners['cluster_id'] != clusters_with_winners['winner_cluster_id']
        ][['cluster_id', 'winner_cluster_id']]
        
        # CRITICAL: Resolve conflicts where a cluster_id maps to different winners
        # This can happen when a cluster appears in multiple entity groups
        loser_mappings = self._resolve_cluster_conflicts(loser_mappings)
        
        # Final validation: ensure we're not mapping any cluster to -1
        loser_mappings = loser_mappings[
            (loser_mappings['winner_cluster_id'] != -1) & 
            (loser_mappings['cluster_id'] != -1)
        ]
        
        # Convert to dictionary
        if len(loser_mappings) > 0:
            # Use to_pandas only for final dictionary creation
            mapping_df = loser_mappings.to_pandas()
            consolidation_map = dict(zip(
                mapping_df['cluster_id'],
                mapping_df['winner_cluster_id']
            ))
            
            # Flatten transitive mappings for single-pass replacement
            consolidation_map = self._flatten_mapping(consolidation_map)
            
            logger.debug(f"Created consolidation mapping for {len(consolidation_map)} clusters")
        else:
            consolidation_map = {}
            logger.debug("No cluster consolidations needed")
        
        return consolidation_map
