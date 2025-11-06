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

import logging

import cudf
import cupy as cp
from cudf.api.types import (
    is_categorical_dtype,
    is_list_dtype,
    is_string_dtype,
    is_struct_dtype,
)

# --- Local Package Imports ---
from .config import ValidationConfig, VectorizerConfig
from .utils import (
    calculate_embedding_similarity,
    check_state_compatibility,
    find_canonical_name,
    get_best_address_gpu,
    gpu_memory_cleanup,
)
from .vectorizer import EmbeddingOrchestrator

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
        self.vectorizer_config = vectorizer_config

        # Define a consistent schema for empty results to prevent concat errors.
        self.EMPTY_ASSIGNMENT_SCHEMA = {
            'original_index': cudf.Series([], dtype='int64'),
            'cluster': cudf.Series([], dtype='int32'),
            'avg_probability': cudf.Series([], dtype='float64'),
            'match_score': cudf.Series([], dtype='float64'),
        }

        # Memory management parameters
        self.max_pairs_per_chunk = min(
            self.config.profile_comparison_max_pairs_per_chunk,
            100000,  # Hard limit to prevent OOM
        )

        # Tolerance parameters to make validation less aggressive
        # These provide a buffer zone where entities can stay in their current cluster
        self.validation_tolerance = 0.01  # 1% tolerance on similarity thresholds
        self.min_improvement_threshold = 0.1  # Require 10% improvement to reassign
        self.keep_original_if_close = True  # Keep original assignment if scores are close
        self.soft_threshold_penalty = 0.05  # Penalty for being below threshold (not elimination)
        self.name_threshold = self.config.name_fuzz_ratio / 100.0
        self.addr_threshold = self.config.address_fuzz_ratio / 100.0

        # Store the instantiated EmbeddingOrchastrator for use across the class
        self.vectorizer_class: EmbeddingOrchestrator | None = None

    def validate_with_reassignment(
        self, gdf: cudf.DataFrame, vectorizer: EmbeddingOrchestrator
    ) -> cudf.DataFrame:
        """
        Evicts invalid members and reassigns them to more appropriate clusters.

        This is the main public method of the class. It orchestrates a multi-step
        process to clean up cluster assignments in a memory-efficient, GPU-accelerated way.

        Args:
            gdf: The cuDF DataFrame after the initial clustering stage.

        Returns:
            The cuDF DataFrame with validated and potentially reassigned clusters.
        """
        self.vectorizer_class = vectorizer
        logger.info('Starting GPU-efficient cluster validation with reassignment...')

        gdf['cluster'] = gdf['cluster'].fillna(-1).astype('int32')

        # --- Step 1: Build comprehensive profiles for all existing clusters ---
        clustered_gdf = gdf[gdf['cluster'] != -1]
        if clustered_gdf.empty:
            logger.info('No clusters to validate.')
            return gdf

        cluster_profiles = self._build_cluster_profiles(clustered_gdf)
        if cluster_profiles.empty:
            logger.warning('Could not build any valid cluster profiles for validation.')
            return gdf

        logger.info(f'Built profiles for {len(cluster_profiles)} clusters.')

        # --- Step 2: Build state-based index for efficient filtering ---
        state_to_clusters = self._build_state_index(cluster_profiles)

        # --- Step 3: Identify entities that need reassignment ---
        entities_to_reassign = self._identify_entities_for_reassignment(gdf, cluster_profiles)

        if entities_to_reassign.empty:
            logger.info('All entities are valid in their current clusters.')
            return gdf

        logger.info(f'Found {len(entities_to_reassign)} entities needing reassignment.')

        # --- Step 4: Find the best new cluster for each entity needing reassignment ---
        best_assignments = self._find_best_assignments(
            entities_to_reassign, cluster_profiles, state_to_clusters
        )

        # --- Step 5: Apply the new assignments to the original DataFrame ---
        final_gdf = self._apply_final_assignments(gdf, best_assignments)

        logger.info('Reassignment complete. Returning validated DataFrame.')
        return final_gdf

    def _build_cluster_profiles(self, clustered_gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Builds a profile for each cluster containing its canonical info and stats.

        Each profile contains:
        - Canonical name (most representative entity name)
        - Canonical address
        - State information
        - Cluster statistics (size, average probability)
        """
        logger.debug('Building cluster profiles on GPU...')

        # Step 1: Calculate aggregate statistics for each cluster in a single pass.
        # This is an efficient, vectorized operation that computes the mean probability
        # and size (count) for every unique cluster.
        cluster_stats = (
            clustered_gdf.groupby('cluster')
            .agg(avg_probability=('cluster_probability', 'mean'), size=('normalized_text', 'count'))
            .reset_index()
        )

        # Step 2: Define a function to process each group (i.e., each cluster).
        # This function encapsulates the logic to find the canonical name and address
        # for a single cluster's data. It will be applied to each group in parallel.
        def get_canonical_info(cluster_group):
            """
            Processes a single cluster's DataFrame to extract its canonical info.
            """
            # 1. Prepare the full name series from the group for frequency counts
            #    and any other logic that needs the complete data.
            all_names_in_group = cluster_group['normalized_text']

            # 2. Get the unique names and their corresponding aligned vectors.
            #    This is the core of the new, more efficient approach.
            unique_names_df = cluster_group.drop_duplicates(subset=['normalized_text'])

            aligned_embeddings = self.vectorizer_class.get_aligned_embeddings(unique_names_df)
            unique_name_vectors = aligned_embeddings['name']

            # The unique names in the correct order, aligned with the vectors.
            unique_names_series = unique_names_df['normalized_text']

            # Determine the most representative name for the cluster using TF-IDF.
            canonical_name = find_canonical_name(
                all_names_in_group, unique_names_series, unique_name_vectors
            )

            # Find the best address representation for the cluster.
            best_address_info = get_best_address_gpu(cluster_group)

            # If a valid address is found, create a cudf.Series with the canonical info.
            # This Series will become a single row in the resulting DataFrame after the
            # .apply() operation is complete.
            if not best_address_info.empty:
                return cudf.Series(
                    {
                        'profile_canonical_name': canonical_name,
                        'profile_canonical_addr_key': best_address_info['addr_normalized_key'].iloc[
                            0
                        ],
                        'profile_canonical_state': best_address_info['addr_state'].iloc[0],
                    }
                )
            # If no address is found, return a Series with None for address fields.
            # We still return the canonical name. This ensures a consistent output
            # schema for all groups, which is critical for the .apply() operation.
            else:
                return cudf.Series(
                    {
                        'profile_canonical_name': canonical_name,
                        'profile_canonical_addr_key': None,
                        'profile_canonical_state': None,
                    }
                )

        # Step 3: Apply the processing function to each cluster group.
        # The .groupby('cluster').apply(...) pattern is the key to vectorization here.
        # It takes each cluster's subset of data and runs the `get_canonical_info`
        # function on it, then intelligently combines the results into a new DataFrame.
        # The original 'cluster' column is preserved as the index, so we reset it.
        canonical_info_gdf = (
            clustered_gdf.groupby('cluster').apply(get_canonical_info).reset_index()
        )

        if canonical_info_gdf.empty:
            return cudf.DataFrame()

        # Step 4: Merge the canonical information with the pre-calculated cluster statistics.
        # This joins the two DataFrames based on the common 'cluster' column.
        profiles_gdf = canonical_info_gdf.merge(cluster_stats, on='cluster')

        # Step 5: Pre-compute normalized features for faster downstream similarity computations.
        # These string operations are also vectorized and executed efficiently on the GPU.
        # We fill any potential NA values in the address key with an empty string
        # to prevent errors when calculating its length.
        # NOTE: I don't think these columns are used anywhere
        # profiles_gdf['profile_name_len'] = profiles_gdf['profile_canonical_name'].str.len()
        # profiles_gdf['profile_addr_len'] = profiles_gdf['profile_canonical_addr_key'].fillna('').str.len()

        return profiles_gdf

    def _build_state_index(self, profiles: cudf.DataFrame) -> dict[str, cp.ndarray]:
        """
        Builds an index mapping states to cluster IDs for efficient filtering.

        This allows us to quickly find which clusters are in compatible states
        for a given entity, dramatically reducing the search space.

        Args:
            profiles: The cuDF DataFrame containing cluster profiles with a
            'profile_canonical_state' and 'cluster' column.

        Returns:
            A Python dictionary where keys are state strings (or None) and values
            are CuPy arrays of the corresponding cluster IDs, residing on the GPU.
        """
        state_index = {}

        # Handle non-null states first by grouping on the GPU
        valid_profiles = profiles.dropna(subset=['profile_canonical_state'])
        if not valid_profiles.empty:
            # Group by state and collect cluster IDs, then convert the small result
            grouped = (
                valid_profiles.groupby('profile_canonical_state')['cluster']
                .agg('collect')
                .to_pandas()
            )
            for state, clusters in grouped.items():
                state_index[state] = cp.asarray(clusters)

        # Handle null states separately
        null_state_clusters = profiles[profiles['profile_canonical_state'].isna()]['cluster']
        if not null_state_clusters.empty:
            # .values on a cuDF Series returns a CuPy array
            state_index[None] = null_state_clusters.values

        logger.debug(f'Built state index with {len(state_index)} unique states.')
        return state_index

    def _identify_entities_for_reassignment(
        self, entity_data_gdf: cudf.DataFrame, cluster_profiles_df: cudf.DataFrame
    ) -> cudf.DataFrame:
        """
        Identifies entities for reassignment using a robust statistical voting system.

        This method replaces a fixed-threshold approach with three complementary,
        data-driven detection techniques:
        1. Per-cluster robust outlier detection (Mahalanobis distance).
        2. Global empirical null with FDR control (Benjamini-Hochberg via shuffling).
        3. Margin test for confidence in the current assignment.

        An entity is flagged if it receives at least two "votes" from these methods,
        if its match score is exceptionally poor, or if it violates a hard business
        rule like state boundaries.

        Args:
            entity_data_gdf: DataFrame containing entity data with a 'cluster' column.
            cluster_profiles_df: DataFrame containing the canonical profiles for each cluster.

        Returns:
            A DataFrame containing all entities identified for reassignment, with
            their original schema plus a 'current_match_score' column.
        """
        logger.info('Starting entity reassignment identification process.')
        logger.debug(
            f'Input entity_data_gdf shape: {entity_data_gdf.shape}, cluster_profiles_df shape: {cluster_profiles_df.shape}'
        )

        # --- 1. Initial Data Partitioning ---
        # Noise entities are always candidates for reassignment. We separate them now
        # and will concatenate them back at the end.
        noise_entities_df = entity_data_gdf[entity_data_gdf['cluster'] == -1].copy()
        clustered_entities_df = entity_data_gdf[entity_data_gdf['cluster'] != -1].copy()
        logger.debug(
            f'Partitioned data into {len(noise_entities_df)} noise entities and {len(clustered_entities_df)} clustered entities.'
        )

        # If there are no clustered entities to analyze, we only need to handle the noise.
        if clustered_entities_df.empty:
            logger.info('No clustered entities to process. Returning only noise entities.')
            if not noise_entities_df.empty:
                noise_entities_df['current_match_score'] = 0.0
            return noise_entities_df

        # --- 2. Calculate Current Match Scores ---
        logger.debug('Calculating current match scores for clustered entities.')
        # Merge entities with their assigned cluster's profile to compute how well they
        # currently fit. This score is the baseline for all subsequent statistical tests.
        entities_with_profiles_df = clustered_entities_df.merge(
            cluster_profiles_df, on='cluster', how='left'
        )

        # Get the embeddings for each individual entity (vectors_a).
        aligned_vectors_dict = self.vectorizer_class.get_aligned_embeddings(
            entities_with_profiles_df
        )
        vectors_a_names = aligned_vectors_dict['name']
        vectors_a_address = aligned_vectors_dict['address']

        # Get the embeddings for each entity's canonical profile name (vectors_b).
        # To do this, we map the profile name string to its original vector's index.

        # Create a lookup table: map every unique name/address in the original dataset to its canonical_id.
        name_to_id_map = (
            self.vectorizer_class.canonical_gdf.reset_index()
            .drop_duplicates(subset=['normalized_text'])
            .set_index('normalized_text')['canonical_id']
        )
        address_to_id_map = (
            self.vectorizer_class.canonical_gdf.reset_index()
            .drop_duplicates(subset=['addr_normalized_key'])
            .set_index('addr_normalized_key')['canonical_id']
        )

        # 2b. Use the map to find the canonical_id for each row's profile_canonical_name/address.
        canonical_name_indices = (
            entities_with_profiles_df['profile_canonical_name'].map(name_to_id_map).to_cupy()
        )
        canonical_address_indices = (
            entities_with_profiles_df['profile_canonical_addr_key'].map(address_to_id_map).to_cupy()
        )

        # 2c. Use these indices to select the correct vectors from the master name_embeddings matrix.
        vectors_b_names = self.vectorizer_class.name_embeddings[canonical_name_indices]
        vectors_b_address = self.vectorizer_class.address_embeddings[canonical_address_indices]

        # # Calculate name and address similarity scores against the assigned profile.
        # name_similarity_scores = calculate_similarity_gpu(
        #     entities_with_profiles_df['normalized_text'],
        #     entities_with_profiles_df['profile_canonical_name'],
        #     self.vectorizer_config.similarity_tfidf
        # ).fillna(0.0)

        # addr_similarity_scores = calculate_similarity_gpu(
        #     entities_with_profiles_df['addr_normalized_key'],
        #     entities_with_profiles_df['profile_canonical_addr_key'],
        #     self.vectorizer_config.similarity_tfidf
        # ).fillna(0.0)

        name_similarity_scores_array = calculate_embedding_similarity(
            vectors_a_names, vectors_b_names
        )
        address_similarity_scores_array = calculate_embedding_similarity(
            vectors_a_address, vectors_b_address
        )

        # Convert the result back to a cuDF Series with the correct index
        name_similarity_scores = cudf.Series(
            name_similarity_scores_array, index=entities_with_profiles_df.index
        )
        addr_similarity_scores = cudf.Series(
            address_similarity_scores_array, index=entities_with_profiles_df.index
        )

        # Compute a weighted average to get the final match score.
        current_match_scores = name_similarity_scores * 0.6 + addr_similarity_scores * 0.4
        entities_with_profiles_df['name_similarity'] = name_similarity_scores
        entities_with_profiles_df['addr_similarity'] = addr_similarity_scores
        entities_with_profiles_df['current_match_score'] = current_match_scores
        logger.debug('Finished calculating match scores.')

        # --- 3. Statistical Anomaly Detection (Voting) ---
        # Each method provides a "vote" indicating if an entity is a poor fit.

        # --- DIAGNOSTIC LOGGING ---
        score_stats = entities_with_profiles_df['current_match_score'].describe()
        logger.info(f'Match score distribution for clustered entities:\n{score_stats.to_string()}')

        # Vote 1: Mahalanobis Distance - Is this entity an outlier within its own cluster's distribution?
        logger.info('Initiating Vote 1: Mahalanobis outlier detection.')
        mahalanobis_outlier_vote = self._detect_mahalanobis_outliers_per_cluster(
            entities_with_profiles_df
        )
        logger.debug(
            f'Mahalanobis vote identified {mahalanobis_outlier_vote.sum()} potential outliers.'
        )

        # Vote 2: FDR Control - Is this entity's match score statistically indistinguishable from a random match?
        # *** This method needs work ***
        # logger.info("Initiating Vote 2: FDR outlier detection via shuffling.")
        # fdr_outlier_vote = self._detect_fdr_outliers_by_shuffling(
        #     entities_with_profiles_df
        # )
        # logger.debug(f"FDR vote identified {fdr_outlier_vote.sum()} potential outliers.")
        fdr_outlier_vote = cudf.Series(False, index=entities_with_profiles_df.index, dtype='bool')

        # Vote 3: Margin Test - Is there another cluster that is almost as good a fit (or better)?
        logger.info('Initiating Vote 3: Low-margin assignment detection.')
        low_margin_vote = self._detect_low_margin_assignments_vectorized(
            entities_with_profiles_df, cluster_profiles_df
        )
        logger.debug(f'Low-margin vote identified {low_margin_vote.sum()} potential outliers.')

        # --- 4. Tally Votes and Apply Overrides ---
        logger.info('Tallying votes and applying overrides.')
        # Tally the votes from the three statistical methods.
        outlier_vote_counts = (
            mahalanobis_outlier_vote.astype('int32')
            + fdr_outlier_vote.astype('int32')
            + low_margin_vote.astype('int32')
        )
        # An entity needs at least two votes to be flagged for reassignment.
        reassignment_mask = outlier_vote_counts >= 2
        logger.debug(f'Found {reassignment_mask.sum()} entities with 2 or more votes.')

        # Override 1: Extremely poor matches are always reassigned, regardless of votes.
        # This acts as a safety net for clear mismatches.
        very_poor_match_mask = current_match_scores < 0.3
        reassignment_mask |= very_poor_match_mask
        logger.debug(
            f"Total entities flagged after 'very_poor_match' override: {reassignment_mask.sum()}"
        )

        # Override 2: Enforce hard business rule (e.g., entity state must match profile state).
        if self.config.enforce_state_boundaries:
            logger.debug('Applying state boundary enforcement override.')
            state_is_compatible = check_state_compatibility(
                entities_with_profiles_df['addr_state'],
                entities_with_profiles_df['profile_canonical_state'],
                self.config,
            )
            # If compatibility is False, it's a mismatch. `~state_is_compatible` flags it.
            # We fill NaNs with True, assuming compatibility if data is missing.
            reassignment_mask |= ~state_is_compatible.fillna(True)
            logger.debug(
                f'Total entities flagged after state compatibility override: {reassignment_mask.sum()}'
            )

        # --- 5. Final Assembly ---
        logger.info('Assembling final list of entities for reassignment.')
        # Select the entities flagged for reassignment using the final mask.
        mismatched_entities_df = entities_with_profiles_df[reassignment_mask]

        # Trim the DataFrame to match the original schema, plus the new score column.
        final_columns = list(entity_data_gdf.columns) + ['current_match_score']
        reassignment_candidates_df = mismatched_entities_df[final_columns].copy()

        # Combine the statistically-identified candidates with the original noise entities.
        all_entities_to_reassign_list = []
        if not noise_entities_df.empty:
            noise_entities_df['current_match_score'] = 0.0
            all_entities_to_reassign_list.append(noise_entities_df)
        if not reassignment_candidates_df.empty:
            all_entities_to_reassign_list.append(reassignment_candidates_df)

        if not all_entities_to_reassign_list:
            logger.info(
                'Reassignment process complete. No entities were identified for reassignment.'
            )
            return cudf.DataFrame()

        final_reassignment_df = cudf.concat(all_entities_to_reassign_list)
        logger.info(
            f'Reassignment process complete. Identified {len(final_reassignment_df)} total entities for reassignment.'
        )
        return final_reassignment_df

    # --- Statistical and Validation Helper Methods ---

    def _logit_transform_gpu(self, scores: cp.ndarray, epsilon: float = 1e-6) -> cp.ndarray:
        """
        Applies a stabilized logit transformation to similarity scores.

        This transformation stretches the score distribution from [0, 1] to [-inf, inf],
        making it more symmetric and Gaussian-like, which is a key assumption for
        methods like Mahalanobis distance.

        Args:
            scores: A cupy.ndarray of similarity scores, expected to be in the [0, 1] range.
            epsilon: A small constant to prevent log(0) or division by zero.

        Returns:
            A cupy.ndarray with the logit-transformed scores.
        """
        # Clip scores to be slightly away from 0 and 1 to ensure numerical stability.
        clipped_scores = cp.clip(scores, epsilon, 1 - epsilon)
        return cp.log(clipped_scores / (1 - clipped_scores))

    def _detect_mahalanobis_outliers_per_cluster(
        self,
        entities_with_profiles_df: cudf.DataFrame,
        min_contamination: float = 0.01,
        max_contamination: float = 0.2,
    ) -> cudf.Series:
        """
        Detects outliers within each cluster using a robust Mahalanobis distance.

        For each cluster, it calculates a robust center and covariance matrix by
        trimming the most extreme points. It then computes the Mahalanobis distance
        for all points in the cluster from this robust center. Points with a distance
        exceeding an adaptive threshold are flagged as outliers.

        Args:
            entities_with_profiles_df: DataFrame of entities with their similarity scores.
            min_contamination: The minimum assumed proportion of outliers in a cluster.
            max_contamination: The maximum assumed proportion of outliers in a cluster.

        Returns:
            A boolean cudf.Series, indexed like the input, where True marks an outlier.
        """
        logger.info('Starting Mahalanobis outlier detection for all clusters.')
        outlier_mask = cudf.Series(
            cp.zeros(len(entities_with_profiles_df), dtype=bool),
            index=entities_with_profiles_df.index,
        )

        unique_clusters = entities_with_profiles_df['cluster'].unique().to_pandas()
        logger.debug(f'Processing {len(unique_clusters)} unique clusters.')
        # Iterate over each cluster to perform localized outlier detection.
        for cluster_id in unique_clusters:
            cluster_mask = entities_with_profiles_df['cluster'] == cluster_id
            cluster_data_df = entities_with_profiles_df[cluster_mask]
            cluster_size = len(cluster_data_df)

            # We need a minimum number of points to reliably compute covariance.
            if cluster_size < 5:
                logger.debug(
                    f'Skipping cluster {cluster_id}: size ({cluster_size}) is less than 5.'
                )
                continue

            # Adaptively set the contamination rate based on cluster size.
            # Larger clusters are assumed to have a smaller proportion of outliers.
            contamination_rate = min(
                max_contamination, max(min_contamination, 1.0 / cp.sqrt(cluster_size))
            )
            logger.debug(
                f'Processing cluster {cluster_id} (size={cluster_size}), contamination_rate={contamination_rate:.4f}'
            )

            # Prepare the feature matrix (name and address similarity) and transform it.
            similarity_features_raw = cp.column_stack(
                [
                    cluster_data_df['name_similarity'].values,
                    cluster_data_df['addr_similarity'].values,
                ]
            )
            logit_features = self._logit_transform_gpu(similarity_features_raw)

            # --- Robust Covariance Estimation (MCD-like approach) ---
            # 1. Find a preliminary center using the median (robust to outliers).
            initial_center = cp.median(logit_features, axis=0)

            # 2. Calculate Euclidean distances to this center to identify a core subset.
            distances_from_center = cp.sqrt(cp.sum((logit_features - initial_center) ** 2, axis=1))

            # 3. Trim the data, keeping a high percentage of the points closest to the center.
            # This ensures the covariance matrix is not skewed by outliers.
            order = cp.argsort(distances_from_center)
            # Ensure we keep at least 3 points to avoid a singular matrix.
            num_points_to_keep = max(
                3, int(cluster_size * (1 - min(0.10, contamination_rate * 0.5)))
            )
            trimmed_indices = order[:num_points_to_keep]
            logit_features_trimmed = logit_features[trimmed_indices]

            # 4. Calculate the robust center (mean) and covariance from this trimmed subset.
            robust_center = cp.mean(logit_features_trimmed, axis=0)
            # Add a small identity matrix (regularization) to guarantee invertibility.
            robust_covariance = cp.cov(logit_features_trimmed.T) + cp.eye(2) * 1e-6

            # --- Mahalanobis Distance Calculation ---
            try:
                inv_covariance = cp.linalg.inv(robust_covariance)
                centered_features = logit_features - robust_center
                # This is the squared Mahalanobis distance calculation.
                mahalanobis_distances = cp.sqrt(
                    cp.sum((centered_features @ inv_covariance) * centered_features, axis=1)
                )

                # Set a dynamic threshold based on the contamination rate.
                distance_threshold = cp.percentile(
                    mahalanobis_distances, (1 - contamination_rate) * 100
                )

                # Use a statistical floor for the threshold. For a chi-squared distribution
                # with 2 degrees of freedom (our features), the 99th percentile is ~9.21.
                # The distance (sqrt) is ~3.04. This prevents an overly lenient threshold.
                final_threshold = max(distance_threshold, 3.05)
                cluster_outlier_flags = mahalanobis_distances > final_threshold
                logger.debug(f'Cluster {cluster_id}: found {cluster_outlier_flags.sum()} outliers.')

                # Update the main outlier mask using the boolean mask for the current cluster.
                # This is a more robust way to assign the results back, as it doesn't
                # rely on index lookups (.loc) which can be problematic with non-unique indices.
                outlier_mask[cluster_mask] = cluster_outlier_flags

            except cp.linalg.LinAlgError:
                # This can happen if the covariance matrix is singular (e.g., all points are collinear).
                # We simply skip outlier detection for this cluster.
                logger.warning(
                    f'Skipping Mahalanobis for cluster {cluster_id} due to LinAlgError (likely singular matrix).'
                )
                continue

        logger.info(f'Mahalanobis detection complete. Total outliers found: {outlier_mask.sum()}.')
        return outlier_mask

    # *************************************************************************
    # Need to refactor. I  built a random-pairing null and then computed
    # lower-tail p-values but the scores sit to the far right of the random
    # distribution so it will always come back with 0 outliers. I think I
    # need to find the centroid of each cluster, then find similarity scores
    # between each point and its centroid. Then make p values and run
    # _benjamini_hochberg_gpu. I can use the _calculate_cluster_centroids
    # function from entity_resolver/utils/clustering.py.
    # *************************************************************************
    # @gpu_memory_cleanup
    # def _detect_fdr_outliers_by_shuffling(
    #     self,
    #     entities_with_profiles_df: cudf.DataFrame,
    #     fdr_level: float = 0.05,
    #     n_shuffles: int = 10
    # ) -> cudf.Series:
    #     """
    #     Detects outliers using a non-parametric empirical null generated by
    #     shuffling cluster profiles, followed by False Discovery Rate (FDR) control.

    #     This method tests whether an entity's `current_match_score` is significantly
    #     better than scores obtained by matching it against randomly chosen cluster
    #     profiles. By shuffling many times, we create a strong "null distribution" of
    #     scores that occur by chance. We then use the Benjamini-Hochberg procedure
    #     to find a p-value threshold that controls the FDR at the specified level.

    #     Args:
    #         entities_with_profiles_df: DataFrame of entities with their scores.
    #         fdr_level: The desired False Discovery Rate (e.g., 0.05 means we accept
    #                 that up to 5% of flagged outliers may be false positives).
    #         n_shuffles: The number of times to shuffle profiles to build the null distribution.

    #     Returns:
    #         A boolean cudf.Series, where True indicates a statistically insignificant match (an outlier).
    #     """
    #     logger.info("Starting FDR outlier detection via shuffling.")
    #     num_entities = len(entities_with_profiles_df)
    #     if num_entities == 0:
    #         logger.warning("FDR detection called with an empty DataFrame.")
    #         return cudf.Series(dtype=bool)

    #     logger.debug(f"Building null distribution with {n_shuffles} shuffles for {num_entities} entities.")
    #     # Keep string columns as cuDF Series. Calling .values on a string column
    #     # is not supported as CuPy does not have a native string dtype. Numeric columns
    #     # can still be converted to CuPy arrays for performance.
    #     entity_text_series = entities_with_profiles_df['normalized_text']
    #     entity_addr_series = entities_with_profiles_df['addr_normalized_key']
    #     actual_match_scores = entities_with_profiles_df['current_match_score'].values

    #     profile_name_series = entities_with_profiles_df['profile_canonical_name']
    #     profile_addr_series = entities_with_profiles_df['profile_canonical_addr_key']

    #     # --- Build Empirical Null Distribution ---
    #     # We concatenate scores from multiple shuffles to create a more stable and
    #     # robust null distribution than a single shuffle would provide.
    #     null_score_accumulator = []
    #     for i in range(n_shuffles):
    #         logger.debug(f"Running shuffle {i+1}/{n_shuffles}...")
    #         # Shuffle the profiles by creating a random permutation of indices.
    #         # Note: .take() uses positional indices not a dataframe index.
    #         shuffled_indices = cp.random.permutation(num_entities)

    #         # Use .take() to reorder the cuDF Series according to the shuffled indices.
    #         shuffled_profile_name_series = profile_name_series.take(shuffled_indices)
    #         shuffled_profile_addr_series = profile_addr_series.take(shuffled_indices)

    #         # The .take() operation resets the index. We must reassign the original
    #         # index to the new shuffled series to ensure they align for the row-wise
    #         # similarity calculation. This resolves the ValueError.
    #         shuffled_profile_name_series.index = entity_text_series.index
    #         shuffled_profile_addr_series.index = entity_addr_series.index

    #         # Calculate similarity scores against these incorrect, random profiles.
    #         shuffled_name_sim = calculate_similarity_gpu(
    #             entity_text_series,
    #             shuffled_profile_name_series,
    #             self.vectorizer_config.similarity_tfidf
    #         ).fillna(0.0)

    #         shuffled_addr_sim = calculate_similarity_gpu(
    #             entity_addr_series,
    #             shuffled_profile_addr_series,
    #             self.vectorizer_config.similarity_tfidf
    #         ).fillna(0.0)

    #         # These scores represent what we'd expect from random pairings.
    #         null_scores_for_shuffle = shuffled_name_sim.values * 0.6 + shuffled_addr_sim.values * 0.4
    #         null_score_accumulator.append(null_scores_for_shuffle)

    #     # Combine all shuffles into one large array and sort it for fast searching.
    #     null_scores_distribution = cp.concatenate(null_score_accumulator)
    #     null_scores_distribution_sorted = cp.sort(null_scores_distribution)
    #     logger.debug(f"Null distribution created with {len(null_scores_distribution)} total scores.")

    #     # --- P-Value Calculation and FDR Control ---
    #     logger.debug(f"Calculating p-values and applying Benjamini-Hochberg correction at alpha={fdr_level}.")
    #     # For each actual score, find its rank within the null distribution.
    #     # A low rank means the score is unusually low, even compared to random matches.
    #     rank_in_null = cp.searchsorted(null_scores_distribution_sorted, actual_match_scores, side='right')

    #     # Calculate p-values using Laplace smoothing (+1) to avoid p=0 or p=1.
    #     # p-value = "probability of observing a score this low or lower by chance".
    #     p_values = (rank_in_null + 1) / (len(null_scores_distribution_sorted) + 1)

    #     # Apply Benjamini-Hochberg procedure to find which p-values are significant
    #     # while controlling for the false discovery rate.
    #     is_outlier = self._benjamini_hochberg_gpu(p_values, fdr_level)

    #     logger.info(f"FDR detection complete. Total outliers found: {is_outlier.sum()}.")
    #     return cudf.Series(is_outlier, index=entities_with_profiles_df.index)

    @gpu_memory_cleanup
    def _detect_low_margin_assignments_vectorized(
        self,
        entities_with_profiles_df: cudf.DataFrame,
        all_profiles_df: cudf.DataFrame,
        margin_threshold: float = 0.1,
        max_sample_size: int = 500,
    ) -> cudf.Series:
        """
        Identifies entities with a low assignment margin by comparing their current
        match score to the best possible score from any *other* cluster profile.

        The "margin" is `(current_score - best_alternative_score)`. A small or
        negative margin indicates low confidence in the current assignment. To make
        this computationally feasible, it operates on a random sample of the entities
        that have below-median match scores.

        Args:
            entities_with_profiles_df: DataFrame of entities with their current scores.
            all_profiles_df: DataFrame containing all available cluster profiles.
            margin_threshold: If the margin is below this value, it's a "vote" for reassignment.
            max_sample_size: The maximum number of entities to check to keep memory usage down.

        Returns:
            A boolean cudf.Series, where True indicates a low-margin assignment.
        """
        logger.info('Starting low-margin assignment detection.')
        # Initialize a mask of all Falses; we will set specific indices to True.
        low_margin_mask = cudf.Series(
            cp.zeros(len(entities_with_profiles_df), dtype=bool),
            index=entities_with_profiles_df.index,
        )

        # --- Candidate Selection for Efficiency ---
        # Pre-filter to entities with below-median scores. High-scoring entities are
        # unlikely to have a low margin, so this is a safe and effective optimization.
        median_score = entities_with_profiles_df['current_match_score'].median()
        low_score_entity_subset_df = (
            entities_with_profiles_df[
                entities_with_profiles_df['current_match_score'] <= median_score
            ]
            .reset_index()
            .rename(columns={'index': 'original_index'})
        )

        if low_score_entity_subset_df.empty:
            logger.info('No entities with below-median scores; skipping margin analysis.')
            return low_margin_mask

        logger.debug(
            f'Selected {len(low_score_entity_subset_df)} candidates for margin analysis (median score <= {median_score:.4f}).'
        )

        # If there are too many candidates, take a random sample to avoid memory explosion.
        if len(low_score_entity_subset_df) > max_sample_size:
            logger.debug(
                f'Sampling down to {max_sample_size} candidates from {len(low_score_entity_subset_df)}.'
            )
            low_score_entity_subset_df = low_score_entity_subset_df.sample(
                n=max_sample_size, random_state=42
            )

        # --- Vectorized Cross-Comparison ---
        logger.debug('Performing cross-join to compare candidates against all other profiles.')
        # Prepare for a cross-join by adding a dummy key to both DataFrames.
        # This will create all possible pairs of (candidate_entity, alternative_profile).
        low_score_entity_subset_df['dummy_key'] = 1
        profiles_alt_df = all_profiles_df.copy()
        profiles_alt_df['dummy_key'] = 1

        entity_to_all_profiles_cross_df = low_score_entity_subset_df.merge(
            profiles_alt_df, on='dummy_key', suffixes=('', '_alt')
        ).drop(columns=['dummy_key'])

        # We only care about alternative profiles, so remove pairs where the entity's
        # own cluster profile is being compared against itself.
        entity_to_all_profiles_cross_df = entity_to_all_profiles_cross_df[
            entity_to_all_profiles_cross_df['cluster']
            != entity_to_all_profiles_cross_df['cluster_alt']
        ]
        logger.debug(
            f'Cross-join created {len(entity_to_all_profiles_cross_df)} pairs for comparison.'
        )

        # Get the embeddings for each individual entity (vectors_a).
        aligned_vectors_dict = self.vectorizer_class.get_aligned_embeddings(
            entity_to_all_profiles_cross_df
        )
        vectors_a_names = aligned_vectors_dict['name']
        vectors_a_address = aligned_vectors_dict['address']

        # Get the embeddings for each entity's canonical profile name (vectors_b).
        # To do this, we map the profile name string to its original vector's index.

        # Create a lookup table: map every unique name/address in the original dataset to its canonical_id.
        name_to_id_map = (
            self.vectorizer_class.canonical_gdf.reset_index()
            .drop_duplicates(subset=['normalized_text'])
            .set_index('normalized_text')['canonical_id']
        )
        address_to_id_map = (
            self.vectorizer_class.canonical_gdf.reset_index()
            .drop_duplicates(subset=['addr_normalized_key'])
            .set_index('addr_normalized_key')['canonical_id']
        )

        # 2b. Use the map to find the canonical_id for each row's profile_canonical_name/address.
        canonical_name_indices = (
            entity_to_all_profiles_cross_df['profile_canonical_name_alt']
            .map(name_to_id_map)
            .to_cupy()
        )
        canonical_address_indices = (
            entity_to_all_profiles_cross_df['profile_canonical_addr_key_alt']
            .map(address_to_id_map)
            .to_cupy()
        )

        # 2c. Use these indices to select the correct vectors from the master name_embeddings matrix.
        vectors_b_names = self.vectorizer_class.name_embeddings[canonical_name_indices]
        vectors_b_address = self.vectorizer_class.address_embeddings[canonical_address_indices]

        # Calculate match scores for every entity against every *other* profile.
        # alt_name_sim = calculate_similarity_gpu(
        #     entity_to_all_profiles_cross_df['normalized_text'],
        #     entity_to_all_profiles_cross_df['profile_canonical_name_alt'],
        #     self.vectorizer_config.similarity_tfidf
        # ).fillna(0.0)

        # alt_addr_sim = calculate_similarity_gpu(
        #     entity_to_all_profiles_cross_df['addr_normalized_key'],
        #     entity_to_all_profiles_cross_df['profile_canonical_addr_key_alt'],
        #     self.vectorizer_config.similarity_tfidf
        # ).fillna(0.0)

        alt_name_similarity_scores_array = calculate_embedding_similarity(
            vectors_a_names, vectors_b_names
        )
        alt_address_similarity_scores_array = calculate_embedding_similarity(
            vectors_a_address, vectors_b_address
        )

        # Convert the result back to a cuDF Series with the correct index
        alt_name_sim = cudf.Series(
            alt_name_similarity_scores_array, index=entity_to_all_profiles_cross_df.index
        )
        alt_addr_sim = cudf.Series(
            alt_address_similarity_scores_array, index=entity_to_all_profiles_cross_df.index
        )

        entity_to_all_profiles_cross_df['alt_score'] = alt_name_sim * 0.6 + alt_addr_sim * 0.4

        # --- Margin Calculation ---
        logger.debug('Calculating best alternative scores and margins.')
        # For each candidate entity, find the single best score among all alternatives.
        best_alternative_scores_s = entity_to_all_profiles_cross_df.groupby('original_index')[
            'alt_score'
        ].max()

        # Merge this best alternative score back to our candidate subset.
        candidates_with_alt_scores_df = low_score_entity_subset_df.merge(
            best_alternative_scores_s.reset_index(), on='original_index', how='left'
        ).fillna({'alt_score': 0.0})  # Fill missing alt scores with 0.

        # The margin is the difference between how good the current assignment is
        # and how good the best other option is.
        candidates_with_alt_scores_df['margin'] = (
            candidates_with_alt_scores_df['current_match_score']
            - candidates_with_alt_scores_df['alt_score']
        )

        # Identify the original indices of entities whose margin is below the threshold.
        low_margin_original_indices = candidates_with_alt_scores_df[
            candidates_with_alt_scores_df['margin'] < margin_threshold
        ]['original_index']

        # Update the final mask at these specific locations.
        if len(low_margin_original_indices) > 0:
            low_margin_mask.loc[low_margin_original_indices] = True

        logger.info(
            f'Low-margin detection complete. Found {low_margin_mask.sum()} entities with low margin.'
        )
        return low_margin_mask

    # --- Low-Level GPU Helpers ---

    def _benjamini_hochberg_gpu(self, p_values: cp.ndarray, alpha: float = 0.05) -> cp.ndarray:
        """
        Performs the Benjamini-Hochberg FDR correction procedure entirely on the GPU.

        Args:
            p_values: A cupy.ndarray of p-values to be corrected.
            alpha: The desired False Discovery Rate level.

        Returns:
            A boolean cupy.ndarray of the same size as p_values, where True
            indicates that the null hypothesis can be rejected (i.e., it is a
            significant result/outlier).
        """
        num_tests = len(p_values)
        if num_tests == 0:
            return cp.array([], dtype=bool)

        # 1. Sort the p-values in ascending order while keeping track of their original indices.
        sorted_indices = cp.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]

        # 2. Find the largest k such that p-value_k <= (k / num_tests) * alpha.
        # First, create the BH critical value line to compare against.
        bh_thresholds = (cp.arange(1, num_tests + 1) / num_tests) * alpha

        # Find all p-values that fall below their corresponding threshold.
        is_below_threshold = sorted_p_values <= bh_thresholds

        # 3. If any such p-values exist, find the one with the highest rank (k).
        if cp.any(is_below_threshold):
            # `cp.where` returns indices where the condition is True. We take the last one.
            max_index_below_threshold = cp.where(is_below_threshold)[0][-1]

            # The critical value is the p-value at this highest rank.
            critical_value = sorted_p_values[max_index_below_threshold]

            # 4. Reject all null hypotheses for which the original p-value is less than or equal to this critical value.
            rejections = p_values <= critical_value
        else:
            # If no p-values were below the line, we reject none.
            rejections = cp.zeros(num_tests, dtype=bool)

        return rejections

    def _find_best_assignments(
        self,
        entities_to_reassign: cudf.DataFrame,
        profiles: cudf.DataFrame,
        state_to_clusters: dict[str, cp.ndarray],
    ) -> cudf.DataFrame:
        """
        Finds the best cluster for each entity using a state-first batching strategy.

        This method orchestrates the reassignment process by first logically partitioning
        the entities based on state (if configured) and then processing each partition
        in smaller, memory-aware batches. This ensures that state boundaries are
        respected while preventing out-of-memory errors.

        Args:
            entities_to_reassign: DataFrame of entities needing a new cluster assignment.
            profiles: DataFrame containing profile information for all candidate clusters.
            state_to_clusters: A dictionary mapping states to their associated cluster IDs.

        Returns:
            A DataFrame containing the best new assignments for the entities.
        """
        # Preserve the original index, which is critical for merging results back later.
        entities_to_reassign['original_index'] = entities_to_reassign.index
        all_best_assignments = []

        # --- Step 1: Primary Grouping Strategy ---
        # The primary grouping is based on whether state boundaries should be strictly enforced.
        if self.config.enforce_state_boundaries:
            # Group all entities by state first. This creates logical partitions.
            entity_state_groups = self._group_entities_by_state(entities_to_reassign)
            logger.info(
                f'Processing reassignments in {len(entity_state_groups)} state-based groups '
                f'due to enforce_state_boundaries=True.'
            )

            for state, state_entity_group in entity_state_groups:
                # For each state, find the relevant candidate clusters.
                candidate_clusters = self._get_candidate_clusters_for_state(
                    state, profiles, state_to_clusters
                )

                if candidate_clusters.empty:
                    logger.debug(
                        f"No candidate clusters for state '{state}'. "
                        f'Skipping {len(state_entity_group)} entities.'
                    )
                    continue

                logger.debug(
                    f"Processing {len(state_entity_group)} entities for state '{state}' "
                    f'against {len(candidate_clusters)} candidate clusters.'
                )

                # Process this specific state group in memory-managed batches.
                group_assignments = self._process_group_in_batches(
                    state_entity_group, candidate_clusters
                )

                if not group_assignments.empty:
                    all_best_assignments.append(group_assignments)
        else:
            # If not enforcing state boundaries, treat all entities as a single large group.
            logger.info(
                'Processing all reassignments in a single group (enforce_state_boundaries=False).'
            )
            # All cluster profiles are considered potential candidates.
            all_candidate_clusters = profiles

            # Process this single large group in memory-managed batches.
            group_assignments = self._process_group_in_batches(
                entities_to_reassign, all_candidate_clusters
            )

            if not group_assignments.empty:
                all_best_assignments.append(group_assignments)

        # --- Step 2: Consolidate Results ---
        if all_best_assignments:
            # Combine the results from all processed groups into a single DataFrame.
            return cudf.concat(all_best_assignments, ignore_index=True)
        else:
            # If no assignments were found, return an empty DataFrame with the correct schema.
            return cudf.DataFrame(self.EMPTY_ASSIGNMENT_SCHEMA)

    @gpu_memory_cleanup
    def _process_group_in_batches(
        self, entity_group: cudf.DataFrame, candidate_clusters: cudf.DataFrame
    ) -> cudf.DataFrame:
        """
        Processes a group of entities against candidates in memory-aware batches.

        This method takes a logical group of entities (e.g., all entities in one state)
        and processes them in smaller sub-batches defined by `validate_cluster_batch_size`.
        For each sub-batch, it performs a potentially chunked cross-join to calculate
        similarity scores, ensuring that memory usage remains under control.

        Args:
            entity_group: A DataFrame of entities to be processed (e.g., from a single state).
            candidate_clusters: The cluster profiles that are valid candidates for this group.

        Returns:
            A DataFrame with the best assignments found for the entities in the group.
        """
        # The batch size for processing entities within the larger logical group.
        batch_size = self.config.validate_cluster_batch_size
        num_batches = (len(entity_group) + batch_size - 1) // batch_size

        batch_results = []

        if num_batches > 1:
            logger.debug(
                f'Splitting group of {len(entity_group)} entities into {num_batches} sub-batches of size ~{batch_size}.'
            )

        # Iterate through the entity group in sub-batches.
        for i, start_idx in enumerate(range(0, len(entity_group), batch_size)):
            end_idx = min(start_idx + batch_size, len(entity_group))
            entity_batch = entity_group.iloc[start_idx:end_idx]

            if num_batches > 1:
                logger.debug(
                    f'Processing sub-batch {i + 1}/{num_batches} ({len(entity_batch)} entities) for the current group.'
                )

            # --- Cross-Join and Scoring Logic ---
            # This section handles the actual comparison, further chunking the cross-join
            # if the number of entity-cluster pairs is too large for memory.
            n_entities_in_batch = len(entity_batch)
            n_candidate_clusters = len(candidate_clusters)
            total_pairs_to_compare = n_entities_in_batch * n_candidate_clusters

            if total_pairs_to_compare == 0:
                continue

            # If the total number of pairs is manageable, process the entire batch at once.
            if total_pairs_to_compare <= self.max_pairs_per_chunk:
                matches = self._score_and_select_matches(entity_batch, candidate_clusters)
                if not matches.empty:
                    # Ensure the resulting DataFrame owns its memory to prevent upstream issues.
                    batch_results.append(self._own_gpu_df(matches))
            else:
                # If the cross-join is too large, chunk the entity batch even further.
                # This chunk size is dynamically calculated to respect memory limits.
                cross_join_chunk_size = max(1, self.max_pairs_per_chunk // n_candidate_clusters)
                logger.debug(
                    f'Sub-batch of {n_entities_in_batch} entities requires further chunking for '
                    f'cross-join against {n_candidate_clusters} clusters. '
                    f'Using chunk size of {cross_join_chunk_size}.'
                )

                for chunk_start in range(0, n_entities_in_batch, cross_join_chunk_size):
                    chunk_end = min(chunk_start + cross_join_chunk_size, n_entities_in_batch)
                    entity_chunk = entity_batch.iloc[chunk_start:chunk_end]

                    chunk_matches = self._score_and_select_matches(entity_chunk, candidate_clusters)

                    if not chunk_matches.empty:
                        batch_results.append(self._own_gpu_df(chunk_matches))

        if batch_results:
            return cudf.concat(batch_results, ignore_index=True)
        else:
            return cudf.DataFrame(self.EMPTY_ASSIGNMENT_SCHEMA)

    def _group_entities_by_state(
        self, batch: cudf.DataFrame
    ) -> list[tuple[str | None, cudf.DataFrame]]:
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
        self, state: str | None, profiles: cudf.DataFrame, state_to_clusters: dict[str, cp.ndarray]
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
            all_candidates = cp.unique(cp.concatenate(candidate_cluster_ids))

            # Filter profiles to only these clusters
            # Note: .isin is highly optimized for this operation.
            candidate_mask = profiles['cluster'].isin(all_candidates)
            return profiles[candidate_mask]
        else:
            return cudf.DataFrame()

    @gpu_memory_cleanup
    def _find_matches_for_state_group(
        self, state_batch: cudf.DataFrame, candidate_clusters: cudf.DataFrame
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
            # Test sync illegal memory issue
            if logger.isEnabledFor(logging.DEBUG):
                try:
                    cp.cuda.Stream.null.synchronize()
                except:
                    logger.debug(
                        '*** CUDA sync failed inside _find_matches_for_state_group single chunk branch ***'
                    )
            # Can process all at once if below memory threshold
            best_matches = self._score_and_select_matches(state_batch, candidate_clusters)
            return self._own_gpu_df(best_matches)
        else:
            # Need to chunk the processing to avoid OOM errors
            # Ensure chunk_size is at least 1
            chunk_size = max(1, self.max_pairs_per_chunk // n_clusters)
            logger.debug(f'Chunking reassignment: {n_entities} entities in chunks of {chunk_size}')

            chunk_results = []
            for chunk_start in range(0, n_entities, chunk_size):
                if logger.isEnabledFor(logging.DEBUG):
                    try:
                        cp.cuda.Stream.null.synchronize()
                    except:
                        logger.debug(
                            '*** CUDA sync failed inside _find_matches_for_state_group multi chunk branch ***'
                        )
                        logger.debug(f'*** Failed on chunk starting at {chunk_start} ***')
                chunk_end = min(chunk_start + chunk_size, n_entities)
                entity_chunk = state_batch.iloc[chunk_start:chunk_end]

                chunk_matches = self._score_and_select_matches(entity_chunk, candidate_clusters)

                if not chunk_matches.empty:
                    chunk_results.append(chunk_matches)

            if chunk_results:
                best_matches = cudf.concat(chunk_results, ignore_index=True)
                return self._own_gpu_df(best_matches)
            else:
                return cudf.DataFrame(self.EMPTY_ASSIGNMENT_SCHEMA)

    def _score_and_select_matches(
        self, entities: cudf.DataFrame, clusters: cudf.DataFrame
    ) -> cudf.DataFrame:
        """
        Scores entity-cluster pairs with SOFT SCORING instead of hard filtering.

        1. Calculates all similarities first before filtering
        2. Uses soft penalties instead of hard cutoffs
        3. Considers partial matches as valid candidates
        4. Only filters out truly incompatible matches
        """
        # Create all pairs using a cross-join
        entities_subset = entities[
            [
                'original_index',
                'normalized_text',
                'addr_normalized_key',
                'addr_state',
                'current_match_score',
            ]
        ].copy()
        entities_subset['_join_key'] = 1

        clusters_subset = clusters[
            [
                'cluster',
                'profile_canonical_name',
                'profile_canonical_addr_key',
                'profile_canonical_state',
                'avg_probability',
                'size',
            ]
        ].copy()
        clusters_subset['_join_key'] = 1

        # Perform cross-join
        pairs = entities_subset.merge(clusters_subset, on='_join_key', how='outer')
        pairs = pairs.drop(columns=['_join_key'])

        if pairs.empty:
            return cudf.DataFrame(self.EMPTY_ASSIGNMENT_SCHEMA)

        # Get the embeddings for each individual entity (vectors_a).
        aligned_vectors_dict = self.vectorizer_class.get_aligned_embeddings(pairs)
        vectors_a_names = aligned_vectors_dict['name']
        vectors_a_address = aligned_vectors_dict['address']

        # Get the embeddings for each entity's canonical profile name (vectors_b).
        # To do this, we map the profile name string to its original vector's index.

        # Create a lookup table: map every unique name/address in the original dataset to its canonical_id.
        name_to_id_map = (
            self.vectorizer_class.canonical_gdf.reset_index()
            .drop_duplicates(subset=['normalized_text'])
            .set_index('normalized_text')['canonical_id']
        )
        address_to_id_map = (
            self.vectorizer_class.canonical_gdf.reset_index()
            .drop_duplicates(subset=['addr_normalized_key'])
            .set_index('addr_normalized_key')['canonical_id']
        )

        # 2b. Use the map to find the canonical_id for each row's profile_canonical_name/address.
        canonical_name_indices = pairs['profile_canonical_name'].map(name_to_id_map).to_cupy()
        canonical_address_indices = (
            pairs['profile_canonical_addr_key'].map(address_to_id_map).to_cupy()
        )

        # 2c. Use these indices to select the correct vectors from the master name_embeddings matrix.
        vectors_b_names = self.vectorizer_class.name_embeddings[canonical_name_indices]
        vectors_b_address = self.vectorizer_class.address_embeddings[canonical_address_indices]

        name_similarity_scores_array = calculate_embedding_similarity(
            vectors_a_names, vectors_b_names
        )
        address_similarity_scores_array = calculate_embedding_similarity(
            vectors_a_address, vectors_b_address
        )

        # Convert the result back to a cuDF Series with the correct index
        pairs['name_sim'] = cudf.Series(name_similarity_scores_array, index=pairs.index)
        pairs['addr_sim'] = cudf.Series(address_similarity_scores_array, index=pairs.index)

        # --- Calculate ALL similarities first (no filtering yet) ---
        # pairs['name_sim'] = calculate_similarity_gpu(
        #     pairs['normalized_text'],
        #     pairs['profile_canonical_name'],
        #     self.vectorizer_config.similarity_tfidf
        # )

        # pairs['addr_sim'] = calculate_similarity_gpu(
        #     pairs['addr_normalized_key'],
        #     pairs['profile_canonical_addr_key'],
        #     self.vectorizer_config.similarity_tfidf
        # )

        # --- Check state compatibility ---
        if self.config.enforce_state_boundaries:
            pairs['state_compatible'] = check_state_compatibility(
                pairs['addr_state'], pairs['profile_canonical_state'], self.config
            )
        else:
            pairs['state_compatible'] = True

        # Calculate base similarity scores
        base_name_score = pairs['name_sim']
        base_addr_score = pairs['addr_sim']

        # Apply soft penalties for being below threshold (but don't eliminate)
        # If below threshold, reduce score but don't zero it out
        name_penalty = cudf.Series(
            cp.where(
                pairs['name_sim'].values < self.name_threshold,
                self.soft_threshold_penalty,  # Apply penalty
                0.0,  # No penalty if above threshold
            ),
            index=pairs.index,
        )

        addr_penalty = cudf.Series(
            cp.where(
                pairs['addr_sim'].values < self.addr_threshold,
                self.soft_threshold_penalty,  # Apply penalty
                0.0,  # No penalty if above threshold
            ),
            index=pairs.index,
        )

        # Adjusted scores with penalties
        adjusted_name_score = (base_name_score - name_penalty).clip(lower=0.0)
        adjusted_addr_score = (base_addr_score - addr_penalty).clip(lower=0.0)

        # State incompatibility is a stronger penalty but not elimination
        state_penalty = cudf.Series(
            cp.where(
                pairs['state_compatible'].values,
                0.0,  # No penalty if compatible
                0.3,  # Significant penalty if incompatible
            ),
            index=pairs.index,
        )

        # --- Calculate final match scores ---
        weights = self.config.reassignment_scoring_weights.copy()

        # Normalize cluster size with log scale
        size_values = pairs['size'].values
        size_factor = (cp.log1p(size_values) / cp.log1p(10.0)).clip(0.0, 1.0)

        # Compute raw match score
        raw_match_score = (
            weights.name_similarity * adjusted_name_score
            + weights.address_similarity * adjusted_addr_score
            + weights.cluster_size * cudf.Series(size_factor, index=pairs.index)
            + weights.cluster_probability * pairs['avg_probability']
        )

        # Apply state penalty to final score
        pairs['match_score'] = (raw_match_score - state_penalty).clip(lower=0.0)

        # --- Apply MINIMUM viable score filter ---
        # Only filter out truly terrible matches (below 20% score)
        minimum_viable_score = 0.2
        pairs = pairs[pairs['match_score'] >= minimum_viable_score]

        if pairs.empty:
            return cudf.DataFrame(self.EMPTY_ASSIGNMENT_SCHEMA)

        # --- Select best matches with improvement threshold ---
        # Group by entity and find best match
        best_matches = pairs.sort_values('match_score', ascending=False).drop_duplicates(
            subset=['original_index'], keep='first'
        )

        # Only reassign if the new match is significantly better than current
        # This prevents unnecessary reassignments for marginal improvements
        if self.min_improvement_threshold > 0:
            improvement = best_matches['match_score'] - best_matches['current_match_score']

            # For entities currently in noise (current_match_score == 0),
            # accept any reasonable match (score > 0.3)
            is_noise = best_matches['current_match_score'] == 0.0
            is_good_enough = best_matches['match_score'] > 0.3

            # For entities with existing clusters, require significant improvement
            is_improvement = improvement >= self.min_improvement_threshold

            # Keep matches that are either:
            # 1. Moving from noise to a reasonable cluster, OR
            # 2. A significant improvement over current assignment
            keep_mask = (is_noise & is_good_enough) | (~is_noise & is_improvement)

            best_matches = best_matches[keep_mask]

        return best_matches[['original_index', 'cluster', 'avg_probability', 'match_score']]

    def _apply_final_assignments(
        self, gdf: cudf.DataFrame, best_assignments: cudf.DataFrame
    ) -> cudf.DataFrame:
        """
        Applies final assignments with PROTECTION against unnecessary noise assignment.

        Key improvement: Entities stay in their original cluster if no significantly
        better match is found, rather than becoming noise.
        """
        if best_assignments.empty or 'original_index' not in best_assignments.columns:
            logger.info('No reassignments to apply.')
            return gdf

        gdf['original_index'] = gdf.index

        # Track which entities were considered for reassignment
        entities_considered = gdf[
            gdf['original_index'].isin(best_assignments['original_index'].unique())
        ]['original_index'].values

        # Merge new assignments back to the main dataframe
        gdf_with_new = gdf.merge(
            best_assignments[
                ['original_index', 'cluster', 'avg_probability', 'match_score']
            ].rename(columns={'cluster': 'new_cluster', 'avg_probability': 'new_avg_prob'}),
            on='original_index',
            how='left',
        )

        # Determine which entities have new assignments
        has_new_assignment = gdf_with_new['new_cluster'].notna()

        # --- Calculate statistics for logging ---
        rescued_mask = (gdf_with_new['cluster'] == -1) & has_new_assignment
        reassigned_mask = (
            (gdf_with_new['cluster'] != -1)
            & has_new_assignment
            & (gdf_with_new['cluster'] != gdf_with_new['new_cluster'])
        )

        # Entities keep their original cluster if no better match was found
        # This is the KEY CHANGE - we don't make them noise just because they're imperfect
        was_considered_for_reassignment = gdf_with_new['original_index'].isin(entities_considered)
        kept_original_mask = (
            was_considered_for_reassignment & ~has_new_assignment & (gdf_with_new['cluster'] != -1)
        )

        # --- Apply new assignments ---
        # Update cluster ID: use new cluster if available, otherwise keep old
        gdf_with_new['cluster'] = gdf_with_new['new_cluster'].fillna(gdf_with_new['cluster'])

        # Update probabilities for reassigned entities
        reassigned_prob = gdf_with_new['match_score'] * gdf_with_new['new_avg_prob']

        # Only update probabilities for rows that got a new assignment
        gdf_with_new['cluster_probability'] = gdf_with_new['cluster_probability'].mask(
            has_new_assignment, reassigned_prob
        )

        # Only make entities noise if they were ALREADY noise and found no match
        # Don't create new noise from previously clustered entities
        was_noise = gdf['cluster'] == -1
        still_no_match = ~has_new_assignment
        remains_noise_mask = was_noise & still_no_match & was_considered_for_reassignment

        # These are the only entities that should be noise
        gdf_with_new.loc[remains_noise_mask, 'cluster'] = -1
        gdf_with_new.loc[remains_noise_mask, 'cluster_probability'] = 0.0

        # For entities that kept their original cluster despite being checked,
        # slightly reduce their probability to reflect uncertainty
        if self.keep_original_if_close:
            gdf_with_new.loc[kept_original_mask, 'cluster_probability'] *= 0.9

        # Log detailed statistics
        logger.info(
            'Validation complete: '
            f'{int(reassigned_mask.sum())} reassigned to different clusters, '
            f'{int(rescued_mask.sum())} rescued from noise, '
            f'{int(kept_original_mask.sum())} kept in original clusters, '
            f'{int(remains_noise_mask.sum())} remain as noise.'
        )

        # Clean up temporary columns
        final_gdf = gdf_with_new.drop(
            columns=['original_index', 'new_cluster', 'new_avg_prob', 'match_score']
        ).astype({'cluster': 'int32', 'cluster_probability': 'float32'})

        return final_gdf

    @staticmethod
    def _own_gpu_df(
        input_df: cudf.DataFrame,
        *,
        copy_index: bool = True,
        prefer_numeric_astype: bool = True,
    ) -> cudf.DataFrame:
        """
        Return a DataFrame whose buffers are *owned* by cuDF (no borrowed / zero-copy views).

        Why this exists
        ---------------
        In GPU pipelines it’s common to construct cuDF objects from CuPy arrays or from
        views of upstream columns. Those columns may be backed by memory pools (CuPy/RMM)
        that can later be reclaimed. If a downstream cleanup frees a pool while a column
        still references it, the next kernel can hit an illegal device access.

        This helper defensively re-materializes each column into a fresh cuDF-owned buffer.
        - Numeric dtypes use a fast device-side reallocation via `astype(..., copy=True)`.
        - Strings / categoricals / nested (list/struct) get a `copy(deep=True)` to ensure
          all child buffers (offsets, chars, categories, children) are duplicated.
        - The index is optionally deep-copied to avoid aliasing there as well.

        Parameters
        ----------
        input_df : cudf.DataFrame
            The DataFrame to "own-ify".
        copy_index : bool, default True
            Whether to deep-copy the index. Set False if you explicitly manage index lifetime
            elsewhere and want to avoid the extra allocation.
        prefer_numeric_astype : bool, default True
            For numeric columns, prefer `astype(same_dtype, copy=True)` (fast path). If you set
            this False, numeric columns will use `copy(deep=True)` instead (slightly heavier).

        Returns
        -------
        cudf.DataFrame
            A DataFrame with the same schema and values, whose buffers are independent from
            the inputs (safe against upstream pool releases).

        Notes
        -----
        - This function keeps the **same dtypes** and **column order**.
        - For categoricals, `copy(deep=True)` preserves both codes and category values.
        - For nested types (list/struct), deep copy duplicates all child columns.
        - This does not “defragment” VRAM; it only ensures ownership & non-aliasing.

        Examples
        --------
        >>> owned = YourClass._own_gpu_df(df)
        >>> # Now safe to release upstream pools / intermediates without dangling pointers.
        """

        # Fast exit for truly empty frames (no columns or no rows).
        if input_df is None or len(input_df.columns) == 0 or len(input_df) == 0:
            # Still consider index ownership for empty-but-indexed frames
            if copy_index and input_df is not None and input_df.index is not None:
                out = input_df.copy(deep=True)
                # copy(deep=True) already owns the index; return as-is
                return out
            return input_df

        owned_columns = {}

        for col_name in input_df.columns:
            src: cudf.Series = input_df[col_name]
            dtype = src.dtype

            # Heuristic: prefer the fastest safe path for numerics;
            # use deep copies for types with child buffers or external tables.
            is_numeric = getattr(getattr(dtype, 'kind', None), 'lower', lambda: None)() in (
                'i',
                'u',
                'f',
                'b',
            )

            try:
                if is_numeric and prefer_numeric_astype:
                    # For numerics, astype with copy=True forces a new device buffer
                    # while preserving dtype and null mask semantics.
                    owned = src.astype(dtype, copy=True)
                elif (
                    is_string_dtype(dtype)
                    or is_categorical_dtype(dtype)
                    or is_list_dtype(dtype)
                    or is_struct_dtype(dtype)
                ):
                    # Strings: deep copy duplicates chars & offsets
                    # Categoricals: deep copy preserves categories & codes
                    # Lists/Structs: deep copy duplicates all child columns
                    owned = src.copy(deep=True)
                else:
                    # Fallback: deep copy covers exotic / extension dtypes.
                    owned = src.copy(deep=True)
            except Exception:
                # If any fast path fails due to dtype peculiarities, fall back to deep copy.
                owned = src.copy(deep=True)

            owned_columns[col_name] = owned

        # Reassemble the DataFrame from owned columns (preserve order).
        owned_df = cudf.DataFrame(owned_columns)

        # Optionally deep-copy the index to detach from any shared buffers.
        if copy_index:
            try:
                owned_df.index = input_df.index.copy(deep=True)
            except Exception:
                # If the index doesn’t support deep copy for some reason,
                # force a materialization through a shallow copy of the frame,
                # which typically re-allocates the index as well.
                owned_df = owned_df.copy(deep=True)

        # At this point, each column (and optionally the index) is backed by
        # cuDF-owned buffers. We *avoid* another full-frame deep copy to keep
        # performance sharp, since columns were already re-materialized above.
        return owned_df
