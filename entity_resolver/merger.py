# entity_resolver/merger.py
"""
This module defines the ClusterMerger class, responsible for post-clustering
refinement steps that involve merging or consolidating clusters.
"""

import cudf
import logging
from typing import Dict

# --- Local Package Imports ---
from .config import ValidationConfig, VectorizerConfig
from . import utils

# Set up a logger for this module
logger = logging.getLogger(__name__)

class ClusterMerger:
    """
    Handles the logic for merging over-split clusters and consolidating
    entities that are identical but were placed in different clusters.
    """
    def __init__(self, validation_config: ValidationConfig, vectorizer_config: VectorizerConfig):
        """
        Initializes the ClusterMerger.

        Args:
            validation_config: Configuration for validation rules (e.g., fuzz ratios).
            merger_config: Configuration for merging thresholds and parameters.
        """
        self.validation_config = validation_config
        self.vectorizer_config = vectorizer_config

    def merge_clusters(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        The main orchestration method for the merging process.

        This method first merges clusters based on similarity and then runs a
        consolidation step to fix any remaining inconsistencies.

        Args:
            gdf: The cuDF DataFrame after initial clustering and validation.

        Returns:
            The cuDF DataFrame with cluster labels updated after merging.
        """
        gdf = self._merge_similar_clusters(gdf)
        gdf = self._consolidate_identical_entities(gdf)
        return gdf

    def _merge_similar_clusters(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Intelligently merges clusters that represent the same real-world entity
        using a GPU-accelerated, graph-based approach.
        """
        logger.info("Merging similar clusters based on name and address similarity...")
        
        # --- Step 1: Create a "profile" for each cluster ---
        # A profile consists of the canonical name, address, and size of a cluster.
        logger.debug("Creating cluster profiles on GPU...")
        clustered_gdf = gdf[gdf['cluster'] != -1]
        if clustered_gdf.empty:
            logger.info("No clusters to merge.")
            return gdf
            
        unique_cluster_ids = clustered_gdf['cluster'].unique().to_pandas()
        
        profiles = []
        for cid in unique_cluster_ids:
            cluster_subset = clustered_gdf[clustered_gdf['cluster'] == cid]
            canonical_name = utils.get_canonical_name_gpu(cluster_subset['normalized_text'])
            best_addr_row = utils.get_best_address_gpu(cluster_subset)
            if not best_addr_row.empty:
                profiles.append({
                    'cluster_id': cid, 
                    'canonical_name': canonical_name, 
                    'canonical_address': best_addr_row['addr_normalized_key'].iloc[0], 
                    'size': len(cluster_subset)
                })
        
        if not profiles:
            return gdf
            
        cluster_profiles_gdf = cudf.DataFrame(profiles)

        # --- Step 2: Build a similarity graph between cluster profiles ---
        logger.info("Building similarity graph for fuzzy merging...")
        name_sim_threshold = self.validation_config.name_fuzz_ratio / 100.0
        addr_sim_threshold = self.validation_config.address_fuzz_ratio / 100.0
        
        # Find pairs of clusters with similar names.
        name_edges = utils.find_similar_pairs(
            cluster_profiles_gdf['canonical_name'], 
            self.vectorizer_config.similarity_tfidf, 
            self.vectorizer_config.similarity_nn,
            name_sim_threshold
        )
        # Find pairs of clusters with similar addresses.
        addr_edges = utils.find_similar_pairs(
            cluster_profiles_gdf['canonical_address'], 
            self.vectorizer_config.similarity_tfidf, 
            self.vectorizer_config.similarity_nn, 
            addr_sim_threshold
        )
        
        # A final edge exists only if both the name and address are similar.
        # This is an inner merge, equivalent to a logical AND.
        final_edges = addr_edges.merge(name_edges, on=['source', 'dest'])

        if final_edges.empty:
            logger.info("No fuzzy cluster merges found based on similarity graph.")
            return gdf

        # --- Step 3: Find connected components to identify merge groups ---
        # Each component in the graph represents a group of clusters that should be merged.
        components = utils.find_graph_components(
            final_edges, 'source', 'dest', 'profile_idx', 'component_id'
        )
        
        profiles_with_components = cluster_profiles_gdf.reset_index().merge(
            components, left_on='index', right_on='profile_idx'
        )
        
        # --- Step 4: Determine the "winner" for each component and create a merge map ---
        # The winner of each component is designated as the largest original cluster.
        component_winners = profiles_with_components.sort_values('size', ascending=False).drop_duplicates(
            subset=['component_id'], keep='first'
        )
        
        # Identify the "loser" clusters that need to be mapped to a winner.
        component_losers = profiles_with_components[
            ~profiles_with_components['cluster_id'].isin(component_winners['cluster_id'])
        ]
        
        # Create the final plan for which clusters get merged into which winners.
        merge_plan = component_losers.merge(
            component_winners[['component_id', 'cluster_id']], 
            on='component_id', 
            suffixes=('_loser', '_winner')
        )
        
        if merge_plan.empty:
            return gdf # No merges to perform.

        merge_map = merge_plan.to_pandas().set_index('cluster_id_loser')['cluster_id_winner'].to_dict()

        # --- Step 5: Apply the merge map ---
        logger.info(f"Performing {len(merge_map)} fuzzy merges based on graph components.")
        gdf['cluster'] = gdf['cluster'].replace(merge_map)
        
        logger.info("Merge process complete.")
        return gdf
    
    def _consolidate_identical_entities(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        A cleanup step to ensure that any identical name+address combinations
        that still exist in different clusters are forced into a single cluster.
        """
        logger.info("Consolidating entities with identical names and addresses...")
        
        clustered_gdf = gdf[gdf['cluster'] != -1].copy()
        if clustered_gdf.empty:
            return gdf
        
        # Create a unique key for each entity based on its name and address.
        clustered_gdf['entity_key'] = (
            clustered_gdf['normalized_text'] + '|||' + 
            clustered_gdf['addr_normalized_key'].fillna('')
        )
        
        # Find any entity keys that appear in more than one unique cluster.
        clusters_per_key = clustered_gdf.groupby('entity_key')['cluster'].unique()
        # Corrected: Use .list.len() for cuDF Series of lists.
        inconsistent_entities = clusters_per_key[clusters_per_key.list.len() > 1]
        
        if inconsistent_entities.empty:
            logger.info("No identical entities found in different clusters.")
            return gdf
        
        logger.warning(f"Found {len(inconsistent_entities)} identical entities in multiple clusters! Consolidating...")
        
        consolidation_map = {}
        for entity_key, cluster_list in inconsistent_entities.to_pandas().items():
            clusters_to_merge = cluster_list
            
            # Determine the best cluster to keep (the "winner").
            # The winner is chosen based on size, then average cluster probability.
            cluster_stats = []
            for cid in clusters_to_merge:
                cluster_mask = (gdf['cluster'] == cid)
                cluster_stats.append({
                    'cluster_id': cid,
                    'size': int(cluster_mask.sum()),
                    'avg_prob': float(gdf.loc[cluster_mask, 'cluster_probability'].mean())
                })
            
            winner = sorted(cluster_stats, key=lambda x: (x['size'], x['avg_prob']), reverse=True)[0]
            winner_cluster_id = winner['cluster_id']
            
            # Map all other clusters in this group to the winner.
            for cid in clusters_to_merge:
                if cid != winner_cluster_id:
                    consolidation_map[cid] = winner_cluster_id
        
        # Apply the consolidation map to the DataFrame.
        if consolidation_map:
            logger.info(f"Consolidating {len(consolidation_map)} clusters...")
            gdf['cluster'] = gdf['cluster'].replace(consolidation_map)
        
        logger.info("Consolidation complete.")
        return gdf
