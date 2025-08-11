# entity_resolver/validator.py
"""
This module defines the ClusterValidator class, which is responsible for
validating the membership of entities within clusters and reassigning any
entities that were incorrectly grouped.
"""

import cudf
import cupy
import logging
from typing import Dict, Any

# --- Local Package Imports ---
from .config import ValidationConfig, VectorizerConfig
from . import utils

# Set up a logger for this module
logger = logging.getLogger(__name__)

class ClusterValidator:
    """
    Validates cluster assignments and reassigns poorly matched entities.

    This class performs a critical cleanup step after initial clustering. It
    identifies entities that do not fit well within their assigned cluster
    (based on name, address, and state similarity) and attempts to find a
    better home for them among all other existing clusters.
    """
    def __init__(self, validation_config: ValidationConfig, vectorizer_config: VectorizerConfig):
        """
        Initializes the ClusterValidator.

        Args:
            validation_config: Configuration for validation rules (e.g., fuzz ratios).
            vectorizer_config: Configuration for vectorization, needed for similarity params.
        """
        self.config = validation_config
        # The vectorizer config is needed for the similarity TF-IDF parameters.
        self.vectorizer_config = vectorizer_config

    def validate_with_reassignment(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Evicts invalid members and reassigns them to more appropriate clusters.

        This is the main public method of the class. It orchestrates a multi-step
        process to clean up cluster assignments in a memory-efficient, GPU-accelerated way.

        Args:
            gdf: The cuDF DataFrame after the initial clustering stage.
            
        Returns:
            The cuDF DataFrame with validated and potentially reassigned clusters.
        """
        logger.info("Validating cluster membership with GPU-accelerated reassignment...")
        
        # --- Step 1: Build comprehensive profiles for all existing clusters ---
        clustered_gdf = gdf[gdf['cluster'] != -1]
        if clustered_gdf.empty:
            logger.info("No clusters to validate.")
            return gdf
        
        cluster_profiles = self._build_cluster_profiles(clustered_gdf)
        if cluster_profiles.empty:
            logger.warning("Could not build any valid cluster profiles for validation.")
            return gdf

        # --- Step 2: Identify entities that need reassignment ---
        # An entity needs reassignment if it's currently noise (cluster == -1) or
        # if it's a poor match for its current cluster.
        entities_to_reassign = self._identify_entities_for_reassignment(gdf, cluster_profiles)
        
        if entities_to_reassign.empty:
            logger.info("All entities are valid in their current clusters.")
            return gdf
        
        # --- Step 3: Find the best new cluster for each entity needing reassignment ---
        logger.debug(f"Finding better clusters for {len(entities_to_reassign)} entities...")
        best_assignments = self._find_best_assignments(entities_to_reassign, cluster_profiles)

        # --- Step 4: Apply the new assignments to the original DataFrame ---
        final_gdf = self._apply_final_assignments(gdf, best_assignments)
        
        return final_gdf

    def _build_cluster_profiles(self, clustered_gdf: cudf.DataFrame) -> cudf.DataFrame:
        """Builds a profile for each cluster containing its canonical info and stats."""
        logger.debug("Building cluster profiles on GPU...")
        
        # Get cluster size and average probability in a single pass.
        cluster_stats = clustered_gdf.groupby('cluster').agg(
            avg_probability=('cluster_probability', 'mean'),
            size=('normalized_text', 'count')
        ).reset_index()
        
        unique_clusters = cluster_stats['cluster'].unique().to_pandas()
        
        canonical_info_list = []
        for cid in unique_clusters:
            cluster_subset = clustered_gdf[clustered_gdf['cluster'] == cid]
            if cluster_subset.empty: continue
            
            c_name = utils.get_canonical_name_gpu(cluster_subset['normalized_text'])
            best_addr = utils.get_best_address_gpu(cluster_subset)
            
            if not best_addr.empty:
                canonical_info_list.append({
                    'cluster': cid,
                    'profile_canonical_name': c_name,
                    'profile_canonical_addr_key': best_addr['addr_normalized_key'].iloc[0],
                    'profile_canonical_state': best_addr['addr_state'].iloc[0]
                })

        if not canonical_info_list:
            return cudf.DataFrame()

        profiles_gdf = cudf.DataFrame(canonical_info_list)
        return profiles_gdf.merge(cluster_stats, on='cluster')

    def _identify_entities_for_reassignment(self, gdf: cudf.DataFrame, profiles: cudf.DataFrame) -> cudf.DataFrame:
        """Identifies entities that are noise or invalid in their current cluster."""
        # Merge current cluster profile info onto each entity.
        entities_with_profiles = gdf.merge(profiles, on='cluster', how='left')
        
        # Calculate similarity of each entity to its own cluster's profile.
        name_sim = utils.calculate_similarity_gpu(
            entities_with_profiles['normalized_text'], 
            entities_with_profiles['profile_canonical_name'],
            self.vectorizer_config.similarity_tfidf
        )
        addr_sim = utils.calculate_similarity_gpu(
            entities_with_profiles['addr_normalized_key'], 
            entities_with_profiles['profile_canonical_addr_key'],
            self.vectorizer_config.similarity_tfidf
        )
        
        # An assignment is valid if it meets name, address, and state criteria.
        is_valid = (
            (name_sim >= self.config.name_fuzz_ratio / 100.0) &
            (addr_sim >= self.config.address_fuzz_ratio / 100.0)
        )
        if self.config.enforce_state_boundaries:
            is_valid &= self._check_state_compatibility(
                entities_with_profiles['addr_state'], entities_with_profiles['profile_canonical_state']
            )
        
        # An entity needs reassignment if it's currently noise OR its assignment is invalid.
        needs_reassignment_mask = (gdf['cluster'] == -1) | (~is_valid)
        return gdf[needs_reassignment_mask].copy()

    def _find_best_assignments(self, entities_to_reassign: cudf.DataFrame, profiles: cudf.DataFrame) -> cudf.DataFrame:
        """Finds the best possible cluster for a batch of entities."""
        entities_to_reassign['original_index'] = entities_to_reassign.index
        
        batch_size = self.config.validate_cluster_batch_size
        all_best_assignments = []
        
        for start_idx in range(0, len(entities_to_reassign), batch_size):
            batch = entities_to_reassign.iloc[start_idx:start_idx + batch_size]
            batch_assignments = self._find_best_cluster_for_batch(batch, profiles)
            all_best_assignments.append(batch_assignments)
        
        return cudf.concat(all_best_assignments) if all_best_assignments else cudf.DataFrame()

    def _find_best_cluster_for_batch(self, batch: cudf.DataFrame, profiles: cudf.DataFrame) -> cudf.DataFrame:
        """Performs a memory-safe cross-join and scoring for a batch of entities."""
        # This is the most memory-intensive step. We cross-join a batch of entities
        # with all possible cluster profiles to evaluate every potential match.
        batch['_dummy'] = 1
        profiles['_dummy'] = 1
        pairs = batch.merge(profiles, on='_dummy', how='outer').drop(columns='_dummy')

        # Calculate similarities for all possible pairs.
        pairs['name_sim'] = utils.calculate_similarity_gpu(
            pairs['normalized_text'], pairs['profile_canonical_name'], self.vectorizer_config.similarity_tfidf
        )
        pairs['addr_sim'] = utils.calculate_similarity_gpu(
            pairs['addr_normalized_key'], pairs['profile_canonical_addr_key'], self.vectorizer_config.similarity_tfidf
        )

        # Filter down to only valid pairs that meet the thresholds.
        is_valid = (
            (pairs['name_sim'] >= self.config.name_fuzz_ratio / 100.0) &
            (pairs['addr_sim'] >= self.config.address_fuzz_ratio / 100.0)
        )
        if self.config.enforce_state_boundaries:
            is_valid &= self._check_state_compatibility(pairs['addr_state'], pairs['profile_canonical_state'])
        
        valid_pairs = pairs[is_valid].copy()
        if valid_pairs.empty:
            return cudf.DataFrame({'original_index': batch['original_index']})

        # Score each valid potential assignment.
        weights = self.config.reassignment_scoring_weights
        size_factor = (cupy.log1p(valid_pairs['size'].values) / cupy.log1p(10)).clip(max=1.0)
        valid_pairs['match_score'] = (
            weights['name_similarity'] * valid_pairs['name_sim'] +
            weights['address_similarity'] * valid_pairs['addr_sim'] +
            weights['cluster_size'] * cudf.Series(size_factor, index=valid_pairs.index) +
            weights['cluster_probability'] * valid_pairs['avg_probability']
        )

        # For each entity, find the single best cluster (the one with the highest score).
        best_matches = (
            valid_pairs.sort_values('match_score', ascending=False)
            .drop_duplicates(subset=['original_index'], keep='first')
        )
        
        return best_matches[['original_index', 'cluster', 'avg_probability', 'match_score']]

    def _apply_final_assignments(self, gdf: cudf.DataFrame, best_assignments: cudf.DataFrame) -> cudf.DataFrame:
        """Merges the best assignments back into the main DataFrame."""
        gdf_with_new = gdf.merge(
            best_assignments.rename(columns={'cluster': 'new_cluster'}),
            left_index=True,
            right_on='original_index',
            how='left'
        )

        # The final cluster is the new one if a better one was found, otherwise it's the original.
        final_cluster = gdf_with_new['new_cluster'].fillna(gdf_with_new['cluster']).astype('int32')
        
        # Calculate the new probability based on the match score.
        final_probability = (
            gdf_with_new['match_score'] * gdf_with_new['avg_probability']
        ).fillna(gdf_with_new['cluster_probability']).astype('float32')

        # Log statistics about the changes.
        reassigned_mask = (gdf['cluster'] != final_cluster) & (final_cluster != -1) & (gdf['cluster'] != -1)
        evicted_mask = (gdf['cluster'] != -1) & (final_cluster == -1)
        rescued_mask = (gdf['cluster'] == -1) & (final_cluster != -1)
        logger.info(
            f"Validation results: {int(evicted_mask.sum())} evicted, "
            f"{int(reassigned_mask.sum())} reassigned, "
            f"{int(rescued_mask.sum())} rescued from noise."
        )
        
        gdf['cluster'] = final_cluster
        gdf['cluster_probability'] = final_probability
        return gdf

    def _check_state_compatibility(self, entity_states: cudf.Series, cluster_states: cudf.Series) -> cudf.Series:
        """Checks if two state series are compatible based on config rules."""
        # States are compatible if they are identical, or if one is missing.
        states_match = (entity_states == cluster_states) | entity_states.isna() | cluster_states.isna()
        
        # Account for allowed neighboring states.
        if self.config.allow_neighboring_states:
            mismatched = ~states_match
            if mismatched.any():
                mismatched_df = cudf.DataFrame({'s1': entity_states[mismatched], 's2': cluster_states[mismatched]})
                mismatched_df['pair'] = mismatched_df.apply(lambda row: '|'.join(sorted([row.s1, row.s2])), axis=1)
                allowed_pairs = {'|'.join(sorted(p)) for p in self.config.allow_neighboring_states}
                is_allowed_neighbor = mismatched_df['pair'].isin(list(allowed_pairs))
                states_match.loc[mismatched.index[is_allowed_neighbor]] = True
                
        return states_match
