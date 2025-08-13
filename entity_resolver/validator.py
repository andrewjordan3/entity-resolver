# entity_resolver/validator.py
"""
This module defines the ClusterValidator class, which is responsible for
validating the membership of entities within clusters and reassigning any
entities that were incorrectly grouped.

This class uses memory-efficient, GPU-native strategies including:
- State-based pre-filtering to reduce candidate clusters.
- Chunked processing to avoid memory explosions with large cross-joins.
- Vectorized operations to minimize CPU-GPU data transfers.
- Early termination when good matches are found.
"""

import cudf
import cupy
import logging
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple

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

        # Cache for similarity computations, cleared after each main run.
        self._similarity_cache = {}

        # Define a consistent schema for empty results to prevent concat errors.
        self.EMPTY_ASSIGNMENT_SCHEMA = {
            'original_index': cudf.Series([], dtype='int64'),
            'cluster': cudf.Series([], dtype='int32'),
            'avg_probability': cudf.Series([], dtype='float64'),
            'match_score': cudf.Series([], dtype='float64')
        }

        # Memory management parameters
        self.max_pairs_per_chunk = min(
            self.config.profile_comparison_max_pairs_per_chunk,
            100000  # Hard limit to prevent OOM
        )

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
        logger.info("Starting GPU-efficient cluster validation with reassignment...")

        # --- Step 1: Build comprehensive profiles for all existing clusters ---
        clustered_gdf = gdf[gdf['cluster'] != -1]
        if clustered_gdf.empty:
            logger.info("No clusters to validate.")
            return gdf

        cluster_profiles = self._build_cluster_profiles(clustered_gdf)
        if cluster_profiles.empty:
            logger.warning("Could not build any valid cluster profiles for validation.")
            return gdf

        logger.info(f"Built profiles for {len(cluster_profiles)} clusters.")

        # --- Step 2: Build state-based index for efficient filtering ---
        state_to_clusters = self._build_state_index(cluster_profiles)

        # --- Step 3: Identify entities that need reassignment ---
        entities_to_reassign = self._identify_entities_for_reassignment(gdf, cluster_profiles)

        if entities_to_reassign.empty:
            logger.info("All entities are valid in their current clusters.")
            return gdf

        logger.info(f"Found {len(entities_to_reassign)} entities needing reassignment.")

        # --- Step 4: Find the best new cluster for each entity needing reassignment ---
        best_assignments = self._find_best_assignments(
            entities_to_reassign,
            cluster_profiles,
            state_to_clusters
        )

        # --- Step 5: Apply the new assignments to the original DataFrame ---
        final_gdf = self._apply_final_assignments(gdf, best_assignments)

        # Clear the similarity cache after validation to free memory.
        self._similarity_cache.clear()

        return final_gdf

    def _build_cluster_profiles(self, clustered_gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Builds a profile for each cluster containing its canonical info and stats.

        Each profile contains:
        - Canonical name (most representative entity name)
        - Canonical address
        - State information
        - Cluster statistics (size, average probability)

        Note: This method iterates through unique clusters. While a fully vectorized
        `groupby().apply()` is possible, the current approach is clear and performs
        well when the number of unique clusters is not excessively large.
        """
        logger.debug("Building cluster profiles on GPU...")

        # Get cluster size and average probability in a single pass
        cluster_stats = clustered_gdf.groupby('cluster').agg(
            avg_probability=('cluster_probability', 'mean'),
            size=('normalized_text', 'count')
        ).reset_index()

        unique_clusters = cluster_stats['cluster'].unique().to_pandas()

        canonical_info_list = []
        for cid in unique_clusters:
            cluster_subset = clustered_gdf[clustered_gdf['cluster'] == cid]
            if cluster_subset.empty:
                continue

            # Get canonical name using TF-IDF similarity
            c_name = utils.get_canonical_name_gpu(
                cluster_subset['normalized_text'],
                self.vectorizer_config.similarity_tfidf
            )

            # Get best address representation
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

        # Merge with statistics
        profiles_gdf = profiles_gdf.merge(cluster_stats, on='cluster')

        # Pre-compute normalized features for faster similarity computation
        profiles_gdf['profile_name_len'] = profiles_gdf['profile_canonical_name'].str.len()
        profiles_gdf['profile_addr_len'] = profiles_gdf['profile_canonical_addr_key'].str.len()

        return profiles_gdf

    def _build_state_index(self, profiles: cudf.DataFrame) -> Dict[str, cupy.ndarray]:
        """
        Builds an index mapping states to cluster IDs for efficient filtering.

        This allows us to quickly find which clusters are in compatible states
        for a given entity, dramatically reducing the search space.
        """
        state_index = {}

        # Handle non-null states first by grouping on the GPU
        valid_profiles = profiles.dropna(subset=['profile_canonical_state'])
        if not valid_profiles.empty:
            # Group by state and collect cluster IDs, then convert the small result
            grouped = valid_profiles.groupby('profile_canonical_state')['cluster'].agg('collect').to_pandas()
            for state, clusters in grouped.items():
                state_index[state] = cupy.asarray(clusters)

        # Handle null states separately
        null_state_clusters = profiles[profiles['profile_canonical_state'].isna()]['cluster']
        if not null_state_clusters.empty:
            # .values on a cuDF Series returns a CuPy array
            state_index[None] = null_state_clusters.values

        logger.debug(f"Built state index with {len(state_index)} unique states.")
        return state_index

    def _identify_entities_for_reassignment(
        self,
        gdf: cudf.DataFrame,
        profiles: cudf.DataFrame
    ) -> cudf.DataFrame:
        """
        Identifies entities that are noise or invalid in their current cluster.

        An entity needs reassignment if:
        1. It's currently marked as noise (cluster == -1), or
        2. It doesn't match well with its current cluster's profile.
        """
        # First, handle noise entities - they always need reassignment.
        noise_entities = gdf[gdf['cluster'] == -1].copy()

        # For clustered entities, check if they're valid in their current cluster.
        clustered_entities = gdf[gdf['cluster'] != -1].copy()

        invalid_entities_list = []
        if not clustered_entities.empty:
            # Merge current cluster profile info onto each entity
            entities_with_profiles = clustered_entities.merge(
                profiles,
                on='cluster',
                how='left'
            )

            # Calculate similarity of each entity to its own cluster's profile
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

            # An assignment is valid if it meets name, address, and state criteria
            is_valid = (
                (name_sim >= self.config.name_fuzz_ratio / 100.0) &
                (addr_sim >= self.config.address_fuzz_ratio / 100.0)
            )

            if self.config.enforce_state_boundaries:
                is_valid &= self._check_state_compatibility(
                    entities_with_profiles['addr_state'],
                    entities_with_profiles['profile_canonical_state']
                )

            # Get entities that are invalid in their current cluster
            invalid_clustered = clustered_entities[~is_valid]
            if not invalid_clustered.empty:
                invalid_entities_list.append(invalid_clustered)

        # Combine noise and invalid entities into a single DataFrame
        all_to_reassign = []
        if not noise_entities.empty:
            all_to_reassign.append(noise_entities)
        if invalid_entities_list:
            all_to_reassign.extend(invalid_entities_list)

        if not all_to_reassign:
            return cudf.DataFrame()

        return cudf.concat(all_to_reassign)

    def _find_best_assignments(
        self,
        entities_to_reassign: cudf.DataFrame,
        profiles: cudf.DataFrame,
        state_to_clusters: Dict[str, cupy.ndarray]
    ) -> cudf.DataFrame:
        """
        Memory-efficient method to find the best cluster for each entity.

        Uses several strategies to reduce memory usage:
        1. Pre-filters candidate clusters by state compatibility.
        2. Processes entities in smaller batches to avoid memory explosions.
        3. Only computes exact similarity for promising candidates.
        """
        # Preserve original index for merging results back later.
        entities_to_reassign['original_index'] = entities_to_reassign.index

        batch_size = self.config.validate_cluster_batch_size
        all_best_assignments = []

        # Process entities in batches to manage memory
        num_batches = (len(entities_to_reassign) + batch_size - 1) // batch_size
        for i, start_idx in enumerate(range(0, len(entities_to_reassign), batch_size)):
            end_idx = min(start_idx + batch_size, len(entities_to_reassign))
            batch = entities_to_reassign.iloc[start_idx:end_idx]

            logger.debug(
                f"Processing reassignment batch {i + 1}/{num_batches} "
                f"({len(batch)} entities)"
            )

            # Find best assignments for this batch
            batch_assignments = self._process_reassignment_batch_efficient(
                batch,
                profiles,
                state_to_clusters
            )

            if not batch_assignments.empty:
                all_best_assignments.append(batch_assignments)

        if all_best_assignments:
            return cudf.concat(all_best_assignments, ignore_index=True)
        else:
            return cudf.DataFrame(self.EMPTY_ASSIGNMENT_SCHEMA)

    def _process_reassignment_batch_efficient(
        self,
        batch: cudf.DataFrame,
        profiles: cudf.DataFrame,
        state_to_clusters: Dict[str, cupy.ndarray]
    ) -> cudf.DataFrame:
        """
        Process a batch of entities using memory-efficient strategies.

        Instead of creating all possible entity-cluster pairs at once, this method:
        1. Groups entities by state to reduce the number of comparisons.
        2. Finds top candidates for each group.
        3. Only does detailed scoring on these promising matches.
        """
        best_assignments_list = []

        # Group batch entities by state for efficient processing
        state_groups = self._group_entities_by_state(batch)

        for state, state_batch in state_groups:
            if state_batch.empty:
                continue

            # Get candidate clusters for this state
            candidate_clusters = self._get_candidate_clusters_for_state(
                state,
                profiles,
                state_to_clusters
            )

            if candidate_clusters.empty:
                logger.debug(f"No candidate clusters found for state: {state}")
                continue

            # Process this state group
            state_assignments = self._find_matches_for_state_group(
                state_batch,
                candidate_clusters
            )

            if not state_assignments.empty:
                best_assignments_list.append(state_assignments)

        if best_assignments_list:
            return cudf.concat(best_assignments_list, ignore_index=True)
        else:
            # Return empty DataFrame with the expected schema
            return cudf.DataFrame(self.EMPTY_ASSIGNMENT_SCHEMA)

    def _group_entities_by_state(
        self,
        batch: cudf.DataFrame
    ) -> List[Tuple[Optional[str], cudf.DataFrame]]:
        """
        Groups entities by their state using GPU-native operations.

        This reduces the number of comparisons needed when state boundaries
        are enforced. It avoids iterating in Python for better performance.
        """
        state_groups = []
        if 'addr_state' in batch.columns:
            # Use a temporary key for NA handling in groupby
            temp_key = '_temp_state_key'
            # Use a unique string to represent NA values
            na_representation = '__NA_STATE__'
            batch[temp_key] = batch['addr_state'].fillna(na_representation)

            # Group by the temporary key on the GPU
            for state_key, group in batch.groupby(temp_key):
                # Convert the representation back to the original state or None
                state_val = state_key if state_key != na_representation else None
                # Drop the temporary key from the group before appending
                state_groups.append((state_val, group.drop(columns=[temp_key])))
            # Clean up the temporary column from the original batch slice
            batch.drop(columns=[temp_key], inplace=True)
        else:
            # No state information available, treat as a single group.
            state_groups.append((None, batch))

        return state_groups

    def _get_candidate_clusters_for_state(
        self,
        state: Optional[str],
        profiles: cudf.DataFrame,
        state_to_clusters: Dict[str, cupy.ndarray]
    ) -> cudf.DataFrame:
        """
        Gets candidate clusters that are compatible with the given state.

        This pre-filtering step dramatically reduces the number of comparisons needed.
        """
        if not self.config.enforce_state_boundaries:
            # All clusters are candidates if state boundaries are not enforced.
            return profiles

        candidate_cluster_ids = []

        # Add clusters from the same state
        if state in state_to_clusters:
            candidate_cluster_ids.append(state_to_clusters[state])

        # Add clusters with no state (they can potentially match anything)
        if None in state_to_clusters:
            candidate_cluster_ids.append(state_to_clusters[None])

        # Add clusters from neighboring states if allowed
        if self.config.allow_neighboring_states and state:
            for state_pair in self.config.allow_neighboring_states:
                if state in state_pair:
                    neighbor = state_pair[0] if state_pair[1] == state else state_pair[1]
                    if neighbor in state_to_clusters:
                        candidate_cluster_ids.append(state_to_clusters[neighbor])

        if candidate_cluster_ids:
            # Combine all candidate cluster IDs and get unique values
            all_candidates = cupy.unique(cupy.concatenate(candidate_cluster_ids))

            # Filter profiles to only these clusters
            # Note: .isin is highly optimized for this operation.
            candidate_mask = profiles['cluster'].isin(all_candidates)
            return profiles[candidate_mask]
        else:
            return cudf.DataFrame()

    def _find_matches_for_state_group(
        self,
        state_batch: cudf.DataFrame,
        candidate_clusters: cudf.DataFrame
    ) -> cudf.DataFrame:
        """
        Finds best matches for a group of entities against candidate clusters.

        This method uses chunked processing to avoid memory overflow even when
        there are many entities and clusters to compare.
        """
        # Calculate the chunk size based on memory constraints
        n_entities = len(state_batch)
        n_clusters = len(candidate_clusters)
        total_pairs = n_entities * n_clusters

        if total_pairs == 0:
            return cudf.DataFrame(self.EMPTY_ASSIGNMENT_SCHEMA)

        if total_pairs <= self.max_pairs_per_chunk:
            # Can process all at once if below memory threshold
            return self._score_and_select_matches(state_batch, candidate_clusters)
        else:
            # Need to chunk the processing to avoid OOM errors
            # Ensure chunk_size is at least 1
            chunk_size = max(1, self.max_pairs_per_chunk // n_clusters)
            logger.debug(f"Chunking reassignment: {n_entities} entities in chunks of {chunk_size}")

            chunk_results = []
            for chunk_start in range(0, n_entities, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_entities)
                entity_chunk = state_batch.iloc[chunk_start:chunk_end]

                chunk_matches = self._score_and_select_matches(
                    entity_chunk,
                    candidate_clusters
                )

                if not chunk_matches.empty:
                    chunk_results.append(chunk_matches)

            if chunk_results:
                return cudf.concat(chunk_results, ignore_index=True)
            else:
                return cudf.DataFrame(self.EMPTY_ASSIGNMENT_SCHEMA)

    def _score_and_select_matches(
        self,
        entities: cudf.DataFrame,
        clusters: cudf.DataFrame
    ) -> cudf.DataFrame:
        """
        Scores all entity-cluster pairs and selects the best match for each entity.

        This is the core matching logic that evaluates similarity between
        entities and cluster profiles, applying filters sequentially to reduce
        the computational load.
        """
        # Create all pairs using a cross-join
        entities_subset = entities[['original_index', 'normalized_text',
                                    'addr_normalized_key', 'addr_state']].copy()
        entities_subset['_join_key'] = 1

        clusters_subset = clusters[['cluster', 'profile_canonical_name',
                                    'profile_canonical_addr_key', 'profile_canonical_state',
                                    'avg_probability', 'size']].copy()
        clusters_subset['_join_key'] = 1

        # Perform cross-join. This is the memory-intensive step.
        pairs = entities_subset.merge(clusters_subset, on='_join_key', how='outer')
        pairs = pairs.drop(columns=['_join_key'])

        if pairs.empty:
            return cudf.DataFrame(self.EMPTY_ASSIGNMENT_SCHEMA)

        # --- Sequential Filtering for Efficiency ---
        # 1. Calculate name similarity and filter
        pairs['name_sim'] = utils.calculate_similarity_gpu(
            pairs['normalized_text'],
            pairs['profile_canonical_name'],
            self.vectorizer_config.similarity_tfidf
        )
        name_threshold = self.config.name_fuzz_ratio / 100.0
        pairs = pairs[pairs['name_sim'] >= name_threshold]
        if pairs.empty:
            return cudf.DataFrame(self.EMPTY_ASSIGNMENT_SCHEMA)

        # 2. Calculate address similarity only for remaining pairs and filter
        pairs['addr_sim'] = utils.calculate_similarity_gpu(
            pairs['addr_normalized_key'],
            pairs['profile_canonical_addr_key'],
            self.vectorizer_config.similarity_tfidf
        )
        addr_threshold = self.config.address_fuzz_ratio / 100.0
        pairs = pairs[pairs['addr_sim'] >= addr_threshold]
        if pairs.empty:
            return cudf.DataFrame(self.EMPTY_ASSIGNMENT_SCHEMA)

        # 3. Check state compatibility if needed and filter
        if self.config.enforce_state_boundaries:
            state_compatible = self._check_state_compatibility(
                pairs['addr_state'],
                pairs['profile_canonical_state']
            )
            pairs = pairs[state_compatible]
        if pairs.empty:
            return cudf.DataFrame(self.EMPTY_ASSIGNMENT_SCHEMA)

        # --- Final Scoring ---
        # Calculate final match scores for the valid candidate pairs
        weights = self.config.reassignment_scoring_weights

        # Normalize cluster size with log scale to dampen the effect of huge clusters
        size_values = pairs['size'].values
        size_factor = (cupy.log1p(size_values) / cupy.log1p(10.0)).clip(0.0, 1.0)

        pairs['match_score'] = (
            weights['name_similarity'] * pairs['name_sim'] +
            weights['address_similarity'] * pairs['addr_sim'] +
            weights['cluster_size'] * cudf.Series(size_factor, index=pairs.index) +
            weights['cluster_probability'] * pairs['avg_probability']
        )

        # Select the best match for each entity based on the highest score
        best_matches = (
            pairs.sort_values('match_score', ascending=False)
            .drop_duplicates(subset=['original_index'], keep='first')
        )

        return best_matches[['original_index', 'cluster', 'avg_probability', 'match_score']]

    def _apply_final_assignments(
        self,
        gdf: cudf.DataFrame,
        best_assignments: cudf.DataFrame
    ) -> cudf.DataFrame:
        """
        Merges the best assignments back into the main DataFrame.

        Updates cluster assignments and probabilities based on the validation results.
        """
        if best_assignments.empty or 'original_index' not in best_assignments.columns:
            logger.info("No reassignments to apply.")
            return gdf

        gdf['original_index'] = gdf.index
        
        # Merge new assignments back to the main dataframe
        gdf_with_new = gdf.merge(
            best_assignments[['original_index', 'cluster', 'avg_probability', 'match_score']].rename(
                columns={'cluster': 'new_cluster', 'avg_probability': 'new_avg_prob'}
            ),
            on='original_index',
            how='left'
        )

        # Determine final cluster assignments
        has_new_assignment = gdf_with_new['new_cluster'].notna()

        # --- Calculate statistics for logging before applying changes ---
        # 1. Rescued: Was noise (cluster -1), now has a cluster.
        rescued_mask = (gdf_with_new['cluster'] == -1) & has_new_assignment
        # 2. Reassigned: Had a valid cluster, now has a different valid cluster.
        reassigned_mask = (
            (gdf_with_new['cluster'] != -1) &
            has_new_assignment &
            (gdf_with_new['cluster'] != gdf_with_new['new_cluster'])
        )
        # 3. Evicted: Had a cluster, but was found to be invalid and no better home was found.
        # This is implicitly handled by the logic; we calculate it here for logging.
        # An entity identified for reassignment that does NOT have a new assignment.
        initial_reassign_indices = gdf_with_new[gdf_with_new['original_index'].isin(
            best_assignments['original_index'].unique()
        )].index
        evicted_mask = initial_reassign_indices.isin(gdf_with_new[~has_new_assignment].index)


        # --- Apply new assignments ---
        # Update cluster ID: use new cluster if available, otherwise keep old.
        gdf_with_new['cluster'] = gdf_with_new['new_cluster'].fillna(gdf_with_new['cluster'])
        
        # Update probabilities for reassigned entities
        # Use a weighted combination of match score and cluster average probability
        # For reassigned entities, use the match score weighted by cluster probability
        reassigned_prob = (
            gdf_with_new.loc[has_new_assignment, 'match_score'] *
            gdf_with_new.loc[has_new_assignment, 'new_avg_prob']
        )
        # Only update probabilities for rows that got a new assignment
        gdf_with_new['cluster_probability'] = gdf_with_new['cluster_probability'].mask(
            has_new_assignment, reassigned_prob
        )
        
        # Entities that were invalid but found no new home are now noise
        # This is handled implicitly if they were passed to reassignment and got no result.
        # Let's make it explicit for clarity. An entity is noise if it was invalid and
        # does not have a new assignment.
        was_invalid_mask = gdf_with_new['original_index'].isin(best_assignments['original_index'].unique())
        is_now_noise_mask = was_invalid_mask & (~has_new_assignment)
        gdf_with_new.loc[is_now_noise_mask, 'cluster'] = -1
        gdf_with_new.loc[is_now_noise_mask, 'cluster_probability'] = 0.0

        # Log detailed statistics
        logger.info(
            "Validation complete: "
            f"{int(reassigned_mask.sum())} reassigned to different clusters, "
            f"{int(rescued_mask.sum())} rescued from noise, "
            f"{int(evicted_mask.sum())} became noise after eviction."
        )

        # Clean up temporary columns
        final_gdf = gdf_with_new.drop(columns=[
            'original_index', 'new_cluster', 'new_avg_prob', 'match_score'
        ]).astype({
             'cluster': 'int32',
             'cluster_probability': 'float32'
        })

        return final_gdf

    def _check_state_compatibility(
        self,
        entity_states: cudf.Series,
        cluster_states: cudf.Series
    ) -> cudf.Series:
        """
        Checks if two state series are compatible using a vectorized GPU approach.

        States are compatible if:
        1. They are identical.
        2. One or both are missing/null.
        3. They are in the configured `allow_neighboring_states` list.
        """
        # 1. Base case: states are compatible if they are identical or one is null.
        states_match = (
            (entity_states == cluster_states) |
            entity_states.isna() |
            cluster_states.isna()
        )

        # 2. If all pairs are already compatible, we are done.
        if states_match.all() or not self.config.allow_neighboring_states:
            return states_match

        # 3. Handle neighboring states for the remaining mismatches.
        # Isolate pairs that are not yet considered a match.
        mismatched_indices = states_match[~states_match].index
        mismatched_df = cudf.DataFrame({
            's1': entity_states[mismatched_indices],
            's2': cluster_states[mismatched_indices]
        }).dropna()  # Neighboring state comparisons require both states to be non-null.

        if mismatched_df.empty:
            return states_match

        logger.debug(f"Checking {len(mismatched_df)} mismatched state pairs against neighbors config.")

        # Create a DataFrame of allowed neighbor pairs for efficient joining.
        allowed_pairs_list = self.config.allow_neighboring_states
        allowed_df = cudf.DataFrame(allowed_pairs_list, columns=['p1', 'p2'])

        # Check for matches in both directions, e.g., (IL, WI) and (WI, IL).
        # Merge 1: (s1, s2) -> (p1, p2)
        merged1 = mismatched_df.merge(allowed_df, left_on=['s1', 's2'], right_on=['p1', 'p2'], how='inner')
        # Merge 2: (s1, s2) -> (p2, p1)
        merged2 = mismatched_df.merge(allowed_df, left_on=['s1', 's2'], right_on=['p2', 'p1'], how='inner')

        # Combine indices of all pairs found in the allowed list.
        # The index of the merged result corresponds to the index of mismatched_df,
        # which in turn corresponds to the original index in the states_match series.
        allowed_indices = cudf.concat([merged1.index, merged2.index]).unique()

        # Update the states_match series for the newly validated neighbors.
        if not allowed_indices.empty:
            states_match.loc[allowed_indices] = True

        return states_match