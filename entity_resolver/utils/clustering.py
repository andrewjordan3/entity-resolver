# entity_resolver/utils/clustering.py
"""
GPU-accelerated clustering utilities for entity resolution.

This module provides advanced clustering operations for post-processing and refinement
of initial cluster assignments. All operations are optimized for GPU execution using
cp and cuML libraries.

Key Features:
    - Noise point attachment using k-NN similarity
    - Cluster merging based on inter-cluster similarity
    - Memory-efficient processing with batching
    - GPU-native operations for maximum performance

Mathematical Foundations:
    - Cosine similarity for vector comparisons
    - Union-Find data structure for efficient merging
    - Ratio tests for disambiguation
"""

import logging

import cupy as cp
from cuml.metrics.pairwise_distances import pairwise_distances
from cuml.neighbors import NearestNeighbors

# Set up a logger for this module.
logger = logging.getLogger(__name__)


def _evaluate_noise_point_attachment(
    neighbor_cluster_labels: cp.ndarray,
    neighbor_similarity_scores: cp.ndarray,
    similarity_threshold: float,
    min_neighbor_matches: int,
    ambiguity_ratio_threshold: float,
) -> int:
    """
    Evaluates whether a single noise point should be attached to an existing cluster.

    This function implements a two-stage decision process:
    1. Ratio Test: Ensures the best candidate cluster is significantly better than alternatives
    2. Strength Test: Verifies sufficient connection strength to the candidate cluster

    The ratio test prevents ambiguous attachments when a point lies between multiple clusters,
    while the strength test ensures the attachment is meaningful rather than coincidental.

    Args:
        neighbor_cluster_labels: Array of cluster labels for the k nearest neighbors.
            Shape: (k_neighbors,). Values of -1 indicate noise neighbors.
        neighbor_similarity_scores: Cosine similarity scores to each neighbor.
            Shape: (k_neighbors,). Range: [0, 1] where 1 is identical.
        similarity_threshold: Minimum similarity required for a neighbor to be considered
            a valid match. Typically 0.7-0.9 for entity resolution.
        min_neighbor_matches: Minimum number of neighbors that must belong to the
            candidate cluster for attachment. Prevents attachment based on outliers.
        ambiguity_ratio_threshold: Minimum ratio between best and second-best cluster
            similarities. Values > 1.2 provide good disambiguation.

    Returns:
        int: The cluster ID to attach to (>= 0), or -1 if no valid attachment found.

    Example:
        >>> labels = cp.array([1, 1, 2, 1, -1])  # 3 neighbors in cluster 1, 1 in cluster 2
        >>> sims = cp.array([0.9, 0.85, 0.6, 0.88, 0.3])
        >>> result = _evaluate_noise_point_attachment(labels, sims, 0.7, 2, 1.2)
        >>> # Returns 1 (cluster 1 has 3 strong matches vs cluster 2's single weak match)
    """
    # Filter out noise neighbors as they cannot inform cluster attachment
    non_noise_neighbor_mask = neighbor_cluster_labels != -1

    if not non_noise_neighbor_mask.any():
        logger.debug('No non-noise neighbors found for attachment evaluation')
        return -1

    # Extract valid (non-noise) neighbors for analysis
    valid_neighbor_labels = neighbor_cluster_labels[non_noise_neighbor_mask]
    valid_neighbor_similarities = neighbor_similarity_scores[non_noise_neighbor_mask]

    # Count occurrences of each cluster among valid neighbors
    unique_cluster_candidates, cluster_neighbor_counts = cp.unique(
        valid_neighbor_labels, return_counts=True
    )

    # Sort clusters by frequency (most common first)
    sorted_cluster_indices = cp.argsort(cluster_neighbor_counts)[::-1]
    sorted_cluster_candidates = unique_cluster_candidates[sorted_cluster_indices]

    # The most frequent cluster is our primary candidate
    best_candidate_cluster_id = int(sorted_cluster_candidates[0])

    # === Ratio Test: Ensure clear separation from second-best option ===
    if len(sorted_cluster_candidates) > 1:
        second_best_candidate_cluster_id = sorted_cluster_candidates[1]

        # Calculate mean similarities to both candidates
        similarities_to_best = valid_neighbor_similarities[
            valid_neighbor_labels == best_candidate_cluster_id
        ]
        mean_similarity_to_best = similarities_to_best.mean()

        similarities_to_second_best = valid_neighbor_similarities[
            valid_neighbor_labels == second_best_candidate_cluster_id
        ]
        mean_similarity_to_second_best = similarities_to_second_best.mean()

        # Check if best candidate is sufficiently better than second best
        similarity_ratio = mean_similarity_to_best / (mean_similarity_to_second_best + 1e-8)

        if similarity_ratio < ambiguity_ratio_threshold:
            logger.debug(
                f'Ambiguous attachment: ratio {similarity_ratio:.2f} < threshold {ambiguity_ratio_threshold}'
            )
            return -1

    # === Strength Test: Ensure sufficient connection to candidate cluster ===
    similarities_to_best_candidate = valid_neighbor_similarities[
        valid_neighbor_labels == best_candidate_cluster_id
    ]

    # Count how many neighbors exceed the similarity threshold
    num_strong_neighbor_matches = (similarities_to_best_candidate >= similarity_threshold).sum()

    # Check all criteria for attachment
    has_enough_neighbors = similarities_to_best_candidate.size >= min_neighbor_matches
    has_sufficient_mean_similarity = similarities_to_best_candidate.mean() >= similarity_threshold
    has_enough_strong_matches = num_strong_neighbor_matches >= min_neighbor_matches

    if has_enough_neighbors and has_sufficient_mean_similarity and has_enough_strong_matches:
        logger.debug(
            f'Attaching to cluster {best_candidate_cluster_id}: '
            f'{num_strong_neighbor_matches} strong matches, '
            f'mean similarity {similarities_to_best_candidate.mean():.3f}'
        )
        return best_candidate_cluster_id

    return -1


def attach_noise_points(
    embedding_vectors: cp.ndarray,
    cluster_labels: cp.ndarray,
    k_neighbors: int,
    similarity_threshold: float,
    min_neighbor_matches: int,
    ambiguity_ratio_threshold: float,
) -> cp.ndarray:
    """
    Attaches noise points to existing clusters based on k-NN similarity.

    This GPU-accelerated function identifies points labeled as noise (cluster=-1) and
    attempts to assign them to existing clusters if they have sufficiently strong
    and unambiguous connections to cluster members.

    The algorithm:
    1. Identifies all noise points in the dataset
    2. Finds k nearest neighbors for each noise point using cosine similarity
    3. Evaluates each noise point for potential attachment using ratio and strength tests
    4. Updates labels for successfully attached points

    Args:
        embedding_vectors: Normalized embedding vectors for all data points.
            Shape: (n_samples, n_features). Must be L2-normalized for cosine similarity.
        cluster_labels: Current cluster assignments.
            Shape: (n_samples,). Noise points have label -1.
        k_neighbors: Number of nearest neighbors to consider for each noise point.
            Larger values provide more evidence but increase computation.
        similarity_threshold: Minimum cosine similarity for considering a neighbor
            as evidence for attachment. Range: [0, 1].
        min_neighbor_matches: Minimum number of neighbors required in a cluster
            for attachment. Prevents spurious attachments.
        ambiguity_ratio_threshold: Minimum ratio between best and second-best cluster
            similarities to prevent ambiguous attachments.

    Returns:
        cp.ndarray: Updated cluster labels with noise points attached where appropriate.
            Same shape as input labels.

    Raises:
        ValueError: If embedding vectors are not properly normalized.

    Example:
        >>> vectors = cp.random.randn(1000, 128)
        >>> vectors = vectors / cp.linalg.norm(vectors, axis=1, keepdims=True)
        >>> labels = cp.array([0]*400 + [1]*400 + [-1]*200)  # 200 noise points
        >>> updated_labels = attach_noise_points(vectors, labels, k=10, tau=0.7,
        ...                                      min_matching=3, ratio_threshold=1.2)
    """
    # Identify noise points that need potential attachment
    noise_point_indices = cp.where(cluster_labels == -1)[0]
    num_noise_points = len(noise_point_indices)

    if num_noise_points == 0:
        logger.info('No noise points found for attachment')
        return cluster_labels

    logger.info(f'Attempting to attach {num_noise_points:,} noise points to existing clusters')

    # Build k-NN model using cosine distance
    # Note: k+1 because the point itself will be included in results
    logger.debug(f'Building k-NN model with k={k_neighbors} for noise point attachment')
    knn_model = NearestNeighbors(
        n_neighbors=k_neighbors + 1,
        metric='cosine',
        algorithm='brute',  # Most reliable for cosine similarity
    )
    knn_model.fit(embedding_vectors)

    # Find neighbors for all noise points in a single batch operation
    logger.debug('Computing nearest neighbors for all noise points')
    cosine_distances, neighbor_indices = knn_model.kneighbors(
        embedding_vectors[noise_point_indices]
    )

    # Convert distances to similarities and exclude self-matches
    cosine_similarities = 1 - cosine_distances[:, 1:]  # Skip first column (self)
    neighbor_indices = neighbor_indices[:, 1:]  # Skip first column (self)

    # Get cluster labels for all neighbors
    neighbor_cluster_labels = cluster_labels[neighbor_indices]

    # Process each noise point for potential attachment
    updated_labels = cluster_labels.copy()
    successful_attachments = 0

    # Process noise points in batches for memory efficiency
    batch_size = min(1000, num_noise_points)

    for batch_start in range(0, num_noise_points, batch_size):
        batch_end = min(batch_start + batch_size, num_noise_points)

        logger.debug(
            f'Processing noise point batch {batch_start // batch_size + 1}/'
            f'{(num_noise_points + batch_size - 1) // batch_size}'
        )

        for relative_idx in range(batch_end - batch_start):
            absolute_idx = batch_start + relative_idx
            original_point_index = noise_point_indices[absolute_idx]

            # Evaluate attachment for this noise point
            new_cluster_label = _evaluate_noise_point_attachment(
                neighbor_cluster_labels=neighbor_cluster_labels[absolute_idx],
                neighbor_similarity_scores=cosine_similarities[absolute_idx],
                similarity_threshold=similarity_threshold,
                min_neighbor_matches=min_neighbor_matches,
                ambiguity_ratio_threshold=ambiguity_ratio_threshold,
            )

            if new_cluster_label != -1:
                updated_labels[original_point_index] = new_cluster_label
                successful_attachments += 1

    attachment_rate = successful_attachments / num_noise_points
    logger.info(
        f'Successfully attached {successful_attachments:,}/{num_noise_points:,} '
        f'noise points ({attachment_rate:.1%})'
    )

    return updated_labels


# --- Cluster Merging Helper Functions ---


def _calculate_cluster_centroids(
    embedding_vectors: cp.ndarray,
    cluster_labels: cp.ndarray,
    cluster_ids_to_process: list[int],
    max_samples_per_cluster: int,
) -> tuple[cp.ndarray, dict[int, cp.ndarray]]:
    """
    Calculates representative centroids for clusters with optional sampling.

    For large clusters, computing the exact centroid can be expensive and may not
    provide significantly better representation than a sampled centroid. This function
    uses random sampling for large clusters to balance accuracy and efficiency.

    Args:
        embedding_vectors: Normalized embedding vectors for all points.
            Shape: (n_samples, n_features).
        cluster_labels: Cluster assignment for each point.
            Shape: (n_samples,).
        cluster_ids_to_process: List of cluster IDs to compute centroids for.
            Excludes noise points (cluster=-1).
        max_samples_per_cluster: Maximum points to use for centroid calculation.
            Larger clusters will be randomly sampled.

    Returns:
        Tuple containing:
            - centroid_matrix: Matrix where row i is the centroid of cluster_ids[i].
              Shape: (n_clusters, n_features).
            - cluster_member_indices: Dictionary mapping cluster ID to member indices.

    Note:
        Centroids are NOT normalized after calculation to preserve cluster geometry.
    """
    logger.debug(f'Calculating centroids for {len(cluster_ids_to_process)} clusters')

    cluster_centroids = {}
    cluster_member_indices = {}

    for cluster_id in cluster_ids_to_process:
        # Find all points belonging to this cluster
        member_indices = cp.where(cluster_labels == cluster_id)[0]
        cluster_member_indices[cluster_id] = member_indices

        num_members = len(member_indices)

        if num_members > max_samples_per_cluster:
            # Sample points for large clusters to reduce computation
            logger.debug(
                f'Cluster {cluster_id}: Sampling {max_samples_per_cluster}/{num_members} points'
            )
            sampled_indices = cp.random.choice(
                member_indices, size=max_samples_per_cluster, replace=False
            )
            cluster_centroids[cluster_id] = cp.mean(embedding_vectors[sampled_indices], axis=0)
        else:
            # Use all points for small clusters
            cluster_centroids[cluster_id] = cp.mean(embedding_vectors[member_indices], axis=0)

    # Stack centroids into a matrix for efficient pairwise operations
    centroid_matrix = cp.vstack([cluster_centroids[cid] for cid in cluster_ids_to_process])

    logger.debug(f'Centroid matrix shape: {centroid_matrix.shape}')

    return centroid_matrix, cluster_member_indices


def _find_candidate_pairs(
    centroid_matrix: cp.ndarray,
    cluster_ids: list[int],
    centroid_similarity_threshold: float,
    computation_batch_size: int,
) -> list[tuple[int, int]]:
    """
    Efficiently identifies cluster pairs that are candidates for merging.

    This function performs a coarse-grained similarity check using cluster centroids
    to quickly eliminate pairs that are obviously dissimilar. This dramatically reduces
    the number of pairs that need detailed evaluation.

    The computation is batched to manage memory usage when dealing with many clusters.

    Args:
        centroid_matrix: Matrix of cluster centroids.
            Shape: (n_clusters, n_features).
        cluster_ids: List of cluster IDs corresponding to centroid_matrix rows.
        centroid_similarity_threshold: Minimum cosine similarity between centroids
            to consider clusters as merge candidates. Range: [0, 1].
        computation_batch_size: Number of clusters to process in each batch
            for memory-efficient pairwise distance computation.

    Returns:
        List of (cluster_id_1, cluster_id_2) tuples representing candidate pairs.
        Each pair appears only once (i.e., if (a,b) is included, (b,a) is not).
    """
    num_clusters = len(cluster_ids)
    total_possible_pairs = num_clusters * (num_clusters - 1) // 2

    logger.info(
        f'Finding merge candidates from {num_clusters} clusters '
        f'({total_possible_pairs:,} possible pairs)'
    )

    merge_candidate_pairs = []

    # Process centroids in batches to manage memory
    for batch_start_idx in range(0, num_clusters, computation_batch_size):
        batch_end_idx = min(batch_start_idx + computation_batch_size, num_clusters)
        batch_centroids = centroid_matrix[batch_start_idx:batch_end_idx]

        logger.debug(
            f'Processing centroid batch {batch_start_idx // computation_batch_size + 1}/'
            f'{(num_clusters + computation_batch_size - 1) // computation_batch_size}'
        )

        # Compute cosine similarities between batch and all centroids
        cosine_similarities = 1 - pairwise_distances(
            batch_centroids, centroid_matrix, metric='cosine'
        )

        # Find pairs exceeding the similarity threshold
        high_similarity_pairs = cp.where(cosine_similarities > centroid_similarity_threshold)

        for batch_row_idx, full_col_idx in zip(
            high_similarity_pairs[0].tolist(), high_similarity_pairs[1].tolist(), strict=True
        ):
            # Map batch index back to full cluster index
            full_row_idx = batch_start_idx + batch_row_idx

            # Only keep unique pairs (avoid both (a,b) and (b,a))
            if full_col_idx > full_row_idx:
                merge_candidate_pairs.append((cluster_ids[full_row_idx], cluster_ids[full_col_idx]))

    reduction_factor = len(merge_candidate_pairs) / max(total_possible_pairs, 1)
    logger.info(
        f'Pre-filtering reduced {total_possible_pairs:,} possible pairs to '
        f'{len(merge_candidate_pairs):,} candidates ({reduction_factor:.1%})'
    )

    return merge_candidate_pairs


def _perform_detailed_check_and_union(
    candidate_pairs: list[tuple[int, int]],
    embedding_vectors: cp.ndarray,
    cluster_member_indices: dict[int, cp.ndarray],
    union_find_structure: dict[int, int],
    merge_parameters: dict,
) -> None:
    """
    Performs detailed similarity analysis and merges qualifying cluster pairs.

    This function implements the fine-grained merge decision using sampled
    pairwise similarities between cluster members. It modifies the union-find
    structure in-place to record merge decisions.

    The detailed check samples points from each cluster and computes their
    pairwise similarities, using both median and maximum similarity as criteria.

    Args:
        candidate_pairs: List of (cluster_id_1, cluster_id_2) candidates.
        embedding_vectors: Normalized embeddings for all points.
        cluster_member_indices: Dictionary mapping cluster IDs to member indices.
        union_find_structure: Union-find data structure for tracking merges.
            Modified in-place.
        merge_parameters: Dictionary with keys:
            - 'merge_sample_size': Points to sample per cluster
            - 'merge_median_threshold': Minimum median similarity
            - 'merge_max_threshold': Minimum maximum similarity
    """

    def find_root_cluster(cluster_id: int) -> int:
        """Find root of cluster in union-find structure with path compression."""
        path_to_root = []
        current_id = cluster_id

        while union_find_structure[current_id] != current_id:
            path_to_root.append(current_id)
            current_id = union_find_structure[current_id]

        # Path compression for O(α(n)) amortized complexity
        for node_id in path_to_root:
            union_find_structure[node_id] = current_id

        return current_id

    def union_clusters(cluster_id_1: int, cluster_id_2: int) -> bool:
        """Unite two clusters if they have different roots."""
        root_1 = find_root_cluster(cluster_id_1)
        root_2 = find_root_cluster(cluster_id_2)

        if root_1 != root_2:
            union_find_structure[root_1] = root_2
            return True
        return False

    logger.info(f'Performing detailed checks on {len(candidate_pairs)} candidate pairs')

    successful_merges = 0
    sample_size = merge_parameters['merge_sample_size']
    median_threshold = merge_parameters['merge_median_threshold']
    max_threshold = merge_parameters['merge_max_threshold']

    for pair_idx, (cluster_id_1, cluster_id_2) in enumerate(candidate_pairs):
        if pair_idx % 100 == 0 and pair_idx > 0:
            logger.debug(f'Processed {pair_idx}/{len(candidate_pairs)} pairs')

        # Get member indices for both clusters
        members_cluster_1 = cluster_member_indices[cluster_id_1]
        members_cluster_2 = cluster_member_indices[cluster_id_2]

        # Sample points from each cluster
        sample_size_1 = min(len(members_cluster_1), sample_size)
        sample_size_2 = min(len(members_cluster_2), sample_size)

        sampled_indices_1 = cp.random.choice(members_cluster_1, size=sample_size_1, replace=False)
        sampled_indices_2 = cp.random.choice(members_cluster_2, size=sample_size_2, replace=False)

        sampled_vectors_1 = embedding_vectors[sampled_indices_1]
        sampled_vectors_2 = embedding_vectors[sampled_indices_2]

        # Compute pairwise similarities between samples
        pairwise_similarity_matrix = 1 - pairwise_distances(
            sampled_vectors_1, sampled_vectors_2, metric='cosine'
        )

        if pairwise_similarity_matrix.size > 0:
            median_similarity = float(cp.median(pairwise_similarity_matrix))
            max_similarity = float(cp.max(pairwise_similarity_matrix))

            # Check merge criteria
            if median_similarity >= median_threshold and max_similarity >= max_threshold:
                if union_clusters(cluster_id_1, cluster_id_2):
                    successful_merges += 1
                    logger.debug(
                        f'Merged clusters {cluster_id_1} and {cluster_id_2}: '
                        f'median_sim={median_similarity:.3f}, max_sim={max_similarity:.3f}'
                    )

    logger.info(
        f'Detailed checks complete: {successful_merges}/{len(candidate_pairs)} pairs merged'
    )


def _relabel_from_union_find(
    cluster_labels: cp.ndarray, union_find_structure: dict[int, int]
) -> cp.ndarray:
    """
    Applies cluster merges from union-find structure to create final labels.

    This function translates the union-find forest into actual cluster relabeling,
    ensuring all points in merged clusters receive the same final label.

    Args:
        cluster_labels: Original cluster labels.
            Shape: (n_samples,).
        union_find_structure: Union-find mapping from cluster IDs to roots.

    Returns:
        cp.ndarray: Updated labels with merges applied.
    """

    def find_root_cluster(cluster_id: int) -> int:
        """Find root with path compression."""
        path_to_root = []
        current_id = cluster_id

        while union_find_structure[current_id] != current_id:
            path_to_root.append(current_id)
            current_id = union_find_structure[current_id]

        for node_id in path_to_root:
            union_find_structure[node_id] = current_id

        return current_id

    logger.debug('Applying union-find merges to cluster labels')

    # Build final mapping from original cluster to merged cluster
    final_cluster_mapping = {
        cluster_id: find_root_cluster(cluster_id) for cluster_id in union_find_structure.keys()
    }

    # Apply mapping to create new labels
    updated_labels = cluster_labels.copy()

    for original_cluster_id, merged_cluster_id in final_cluster_mapping.items():
        if original_cluster_id != merged_cluster_id:
            cluster_mask = cluster_labels == original_cluster_id
            updated_labels[cluster_mask] = merged_cluster_id

            num_points_relabeled = int(cluster_mask.sum())
            logger.debug(
                f'Relabeled {num_points_relabeled} points from cluster '
                f'{original_cluster_id} to {merged_cluster_id}'
            )

    return updated_labels


def merge_snn_clusters(
    embedding_vectors: cp.ndarray, cluster_labels: cp.ndarray, merge_parameters: dict
) -> cp.ndarray:
    """
    Merges similar clusters using a multi-stage hierarchical approach.

    This function implements a scalable cluster merging algorithm:
    1. Fast pre-filtering using centroid similarity
    2. Detailed verification using sampled pairwise similarities
    3. Efficient merging using union-find data structure

    The approach balances accuracy with computational efficiency, making it
    suitable for large-scale entity resolution tasks.

    Args:
        embedding_vectors: Normalized embedding vectors for all points.
            Shape: (n_samples, n_features). Must be L2-normalized.
        cluster_labels: Initial cluster assignments.
            Shape: (n_samples,). Noise points (label=-1) are ignored.
        merge_parameters: Dictionary containing:
            - 'centroid_sample_size': Max points for centroid calculation
            - 'centroid_similarity_threshold': Threshold for candidate selection
            - 'merge_batch_size': Batch size for pairwise computations
            - 'merge_sample_size': Points to sample for detailed check
            - 'merge_median_threshold': Minimum median similarity for merge
            - 'merge_max_threshold': Minimum max similarity for merge

    Returns:
        cp.ndarray: Updated cluster labels with similar clusters merged.

    Example:
        >>> vectors = cp.random.randn(10000, 128)
        >>> vectors = vectors / cp.linalg.norm(vectors, axis=1, keepdims=True)
        >>> labels = cp.random.randint(0, 50, size=10000)
        >>> merge_params = {
        ...     'centroid_sample_size': 1000,
        ...     'centroid_similarity_threshold': 0.7,
        ...     'merge_batch_size': 100,
        ...     'merge_sample_size': 100,
        ...     'merge_median_threshold': 0.6,
        ...     'merge_max_threshold': 0.8
        ... }
        >>> merged_labels = merge_snn_clusters(vectors, labels, merge_params)
    """
    # Extract unique cluster IDs (excluding noise)
    unique_cluster_labels = cp.unique(cluster_labels)
    valid_cluster_ids = [
        int(cluster_id) for cluster_id in unique_cluster_labels.tolist() if cluster_id != -1
    ]

    num_valid_clusters = len(valid_cluster_ids)

    if num_valid_clusters < 2:
        logger.info(
            f'Insufficient clusters for merging ({num_valid_clusters} found). '
            'Returning original labels.'
        )
        return cluster_labels

    logger.info(f'Starting cluster merge process for {num_valid_clusters} clusters')

    # Stage 1: Calculate cluster centroids for fast pre-filtering
    logger.info('Stage 1: Computing cluster centroids')
    centroid_matrix, cluster_member_indices = _calculate_cluster_centroids(
        embedding_vectors=embedding_vectors,
        cluster_labels=cluster_labels,
        cluster_ids_to_process=valid_cluster_ids,
        max_samples_per_cluster=merge_parameters['centroid_sample_size'],
    )

    # Stage 2: Find candidate pairs using centroid similarity
    logger.info('Stage 2: Identifying merge candidates via centroid similarity')
    candidate_merge_pairs = _find_candidate_pairs(
        centroid_matrix=centroid_matrix,
        cluster_ids=valid_cluster_ids,
        centroid_similarity_threshold=merge_parameters['centroid_similarity_threshold'],
        computation_batch_size=merge_parameters['merge_batch_size'],
    )

    if not candidate_merge_pairs:
        logger.info('No merge candidates found. Returning original labels.')
        return cluster_labels

    # Stage 3: Perform detailed checks and merge qualifying pairs
    logger.info('Stage 3: Performing detailed merge analysis')

    # Initialize union-find structure
    union_find_structure = {cluster_id: cluster_id for cluster_id in valid_cluster_ids}

    _perform_detailed_check_and_union(
        candidate_pairs=candidate_merge_pairs,
        embedding_vectors=embedding_vectors,
        cluster_member_indices=cluster_member_indices,
        union_find_structure=union_find_structure,
        merge_parameters=merge_parameters,
    )

    # Stage 4: Apply merges to create final labels
    logger.info('Stage 4: Applying merge decisions to labels')
    final_merged_labels = _relabel_from_union_find(
        cluster_labels=cluster_labels, union_find_structure=union_find_structure
    )

    # Calculate and report merge statistics
    final_unique_clusters = len(cp.unique(final_merged_labels[final_merged_labels != -1]))
    clusters_reduced = num_valid_clusters - final_unique_clusters

    logger.info(
        f'Cluster merging complete: {num_valid_clusters} → {final_unique_clusters} clusters '
        f'({clusters_reduced} clusters merged)'
    )

    return final_merged_labels
