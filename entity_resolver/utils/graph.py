# entity_resolver/utils/graph.py
"""
GPU-Accelerated Graph Construction and Analysis Utilities

This module provides high-performance graph operations using the RAPIDS ecosystem,
including cuDF for GPU DataFrames, cuGraph for graph algorithms, and cuML for
machine learning operations. The module is designed for large-scale entity
resolution and clustering tasks where performance is critical.

Key Features:
    - Efficient edge list construction from k-nearest neighbor searches
    - Connected component detection for graph partitioning
    - Mutual rank graph construction with hybrid weighting schemes
    - Full GPU acceleration for all operations

Dependencies:
    - cupy: GPU-accelerated NumPy-like arrays
    - cudf: GPU-accelerated Pandas-like DataFrames
    - cugraph: GPU-accelerated graph algorithms
    - cuml: GPU-accelerated machine learning algorithms
"""

import logging

import cudf
import cugraph
import cupy
from cuml.neighbors import NearestNeighbors

# Set up a logger for this module.
logger = logging.getLogger(__name__)


def create_edge_list(
    neighbor_indices: cupy.ndarray,
    neighbor_distances: cupy.ndarray,
    distance_threshold: float,
    include_weights: bool = False,
) -> cudf.DataFrame:
    """
    Transforms k-nearest neighbor results into a graph edge list DataFrame.

    This function converts the dense matrix representation of k-nearest neighbors
    (as produced by scikit-learn style APIs) into a sparse edge list format
    suitable for graph construction. It filters edges based on distance threshold
    and removes self-loops to ensure a valid simple graph.

    Args:
        neighbor_indices: A 2D CuPy array of shape (n_samples, n_neighbors)
            containing the indices of nearest neighbors for each sample.
        neighbor_distances: A 2D CuPy array of shape (n_samples, n_neighbors)
            containing the distances to nearest neighbors for each sample.
        distance_threshold: Maximum distance for an edge to be included in the graph.
            Edges with distances >= this threshold are filtered out.
        include_weights: If True, includes distance as an edge weight column.
            Useful for weighted graph algorithms.

    Returns:
        A cuDF DataFrame with columns:
            - 'source': Index of the source node (int32)
            - 'destination': Index of the destination node (int32)
            - 'distance': Distance between nodes (float32, optional based on include_weights)

    Raises:
        ValueError: If input arrays have incompatible shapes or invalid dimensions.

    Example:
        >>> indices = cupy.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]])
        >>> distances = cupy.array([[0.0, 0.5, 0.8], [0.0, 0.5, 0.7], [0.0, 0.6, 0.8]])
        >>> edge_list = create_edge_list(indices, distances, threshold=0.75)
    """
    # Validate input dimensions
    if neighbor_indices.ndim != 2 or neighbor_distances.ndim != 2:
        raise ValueError(
            f'Expected 2D arrays, got shapes: indices={neighbor_indices.shape}, '
            f'distances={neighbor_distances.shape}'
        )

    if neighbor_indices.shape != neighbor_distances.shape:
        raise ValueError(
            f'Shape mismatch: indices={neighbor_indices.shape} vs '
            f'distances={neighbor_distances.shape}'
        )

    num_samples, num_neighbors_per_sample = neighbor_indices.shape
    logger.debug(
        f'Processing k-NN matrix with {num_samples} samples and '
        f'{num_neighbors_per_sample} neighbors per sample'
    )

    # Create source indices by repeating each sample index k times
    # This creates the "from" node for each edge
    source_node_indices = cupy.arange(num_samples, dtype=cupy.int32).repeat(
        num_neighbors_per_sample
    )

    # Flatten the neighbor indices to create destination nodes
    # This creates the "to" node for each edge
    destination_node_indices = neighbor_indices.flatten().astype(cupy.int32)

    # Flatten the distances for edge weights
    edge_distances = neighbor_distances.flatten().astype(cupy.float32)

    # Construct the initial edge list DataFrame
    edge_list_df = cudf.DataFrame(
        {
            'source': source_node_indices,
            'destination': destination_node_indices,
            'distance': edge_distances,
        }
    )

    # Apply filtering criteria:
    # 1. Distance must be below threshold (ensures meaningful connections)
    # 2. No self-loops allowed (source != destination)
    valid_edges_mask = (edge_list_df['distance'] < distance_threshold) & (
        edge_list_df['source'] != edge_list_df['destination']
    )
    filtered_edge_list = edge_list_df[valid_edges_mask]

    # Remove distance column if not needed to save memory
    if not include_weights:
        filtered_edge_list = filtered_edge_list[['source', 'destination']]

    num_edges = len(filtered_edge_list)
    num_possible_edges = num_samples * num_neighbors_per_sample
    retention_rate = (num_edges / num_possible_edges) * 100 if num_possible_edges > 0 else 0

    logger.info(
        f'Created edge list with {num_edges:,} edges from {num_possible_edges:,} '
        f'possible edges (retention rate: {retention_rate:.1f}%)'
    )

    return filtered_edge_list


def find_graph_components(
    edge_list_df: cudf.DataFrame,
    source_column: str = 'source',
    destination_column: str = 'destination',
    directed: bool = False,
    output_vertex_column: str = 'vertex',
    output_component_column: str = 'component_id',
) -> cudf.DataFrame:
    """
    Identifies connected components in a graph using GPU-accelerated algorithms.

    This function wraps cuGraph's connected components algorithms, providing
    a clean interface for component detection in both directed and undirected
    graphs. Components are useful for identifying clusters or groups of
    related entities in the graph.

    Args:
        edge_list_df: A cuDF DataFrame containing the graph edges with at minimum
            source and destination columns.
        source_column: Name of the column containing source node indices.
        destination_column: Name of the column containing destination node indices.
        directed: If True, treats the graph as directed and finds weakly connected
            components. If False, treats as undirected.
        output_vertex_column: Name for the vertex column in the output DataFrame.
        output_component_column: Name for the component ID column in the output DataFrame.

    Returns:
        A cuDF DataFrame with columns:
            - output_vertex_column: The vertex/node ID (matches indices from edge list)
            - output_component_column: The component ID this vertex belongs to

    Raises:
        ValueError: If the edge list is empty or required columns are missing.

    Notes:
        - Component IDs are arbitrary integers starting from 0
        - Isolated nodes (not present in edge list) will not appear in output
        - For directed graphs, uses weakly connected components
    """
    # Validate input
    if edge_list_df.empty:
        logger.warning('Received empty edge list for component detection')
        return cudf.DataFrame(
            {
                output_vertex_column: cupy.array([], dtype=cupy.int32),
                output_component_column: cupy.array([], dtype=cupy.int32),
            }
        )

    required_columns = {source_column, destination_column}
    missing_columns = required_columns - set(edge_list_df.columns)
    if missing_columns:
        raise ValueError(f'Missing required columns in edge list: {missing_columns}')

    num_edges = len(edge_list_df)
    logger.debug(
        f'Finding connected components for graph with {num_edges:,} edges (directed={directed})'
    )

    # Create and populate the cuGraph Graph object
    graph = cugraph.Graph(directed=directed)
    graph.from_cudf_edgelist(
        edge_list_df, source=source_column, destination=destination_column, renumber=False
    )

    num_vertices = graph.number_of_vertices()
    logger.debug(f'Graph contains {num_vertices:,} unique vertices')

    # Run the appropriate connected components algorithm
    if directed:
        components_df = cugraph.weakly_connected_components(graph)
        component_type = 'weakly connected'
    else:
        components_df = cugraph.connected_components(graph)
        component_type = 'connected'

    # Standardize output column names
    components_df = components_df.rename(
        columns={'labels': output_component_column, 'vertex': output_vertex_column}
    )

    # Calculate and log component statistics
    num_components = components_df[output_component_column].nunique()
    component_sizes = components_df.groupby(output_component_column).size()
    largest_component_size = component_sizes.max()
    avg_component_size = component_sizes.mean()

    logger.info(
        f'Found {num_components:,} {component_type} components. '
        f'Largest: {largest_component_size:,} vertices, '
        f'Average: {avg_component_size:.1f} vertices'
    )

    return components_df


# --- Mutual Rank Graph Helper Functions ---


def _compute_directed_knn_edges(
    embedding_vectors: cupy.ndarray, k_neighbors: int, distance_metric: str = 'cosine'
) -> cudf.DataFrame:
    """
    Computes k-nearest neighbors and formats results as directed edges with ranks.

    This internal function finds the k-nearest neighbors for each point in the
    embedding space and creates a directed edge list where each edge includes
    the rank of the neighbor (1st nearest, 2nd nearest, etc.) and the similarity
    score.

    Args:
        embedding_vectors: A 2D CuPy array of shape (n_samples, n_features)
            containing the embedding vectors for all data points.
        k_neighbors: Number of nearest neighbors to find for each point.
        distance_metric: Distance metric to use ('cosine', 'euclidean', etc.).

    Returns:
        A cuDF DataFrame with directed edges containing:
            - 'source': Source node index
            - 'destination': Destination node index
            - 'rank': Rank of destination as neighbor of source (0 to k-1)
            - 'similarity': Similarity score (1 - distance for cosine)

    Raises:
        ValueError: If k_neighbors exceeds the number of samples.
    """
    num_samples, num_features = embedding_vectors.shape

    if k_neighbors > num_samples:
        raise ValueError(
            f'k_neighbors ({k_neighbors}) cannot exceed number of samples ({num_samples})'
        )

    logger.debug(
        f'Computing {k_neighbors}-NN for {num_samples:,} vectors with '
        f'{num_features} dimensions using {distance_metric} distance'
    )

    # Fit k-NN model and find neighbors
    knn_model = NearestNeighbors(
        n_neighbors=k_neighbors,
        metric=distance_metric,
        algorithm='brute',  # Brute force is often fastest on GPU for high-dim data
    )
    knn_model.fit(embedding_vectors)
    distances_matrix, indices_matrix = knn_model.kneighbors(embedding_vectors)

    # Create source indices (each point repeated k times)
    source_indices = cupy.repeat(cupy.arange(num_samples, dtype=cupy.int32), k_neighbors)

    # Flatten neighbor indices to create destination indices
    destination_indices = indices_matrix.ravel().astype(cupy.int32)

    # Create rank array (0 to k-1 repeated for each source)
    neighbor_ranks = cupy.tile(cupy.arange(k_neighbors, dtype=cupy.int32), num_samples)

    # Convert distances to similarities (higher is better)
    if distance_metric == 'cosine':
        similarity_scores = 1.0 - distances_matrix.ravel()
    else:
        # For other metrics, use inverse distance
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        similarity_scores = 1.0 / (distances_matrix.ravel() + epsilon)

    # Construct the directed edge DataFrame
    directed_edges_df = cudf.DataFrame(
        {
            'source': source_indices,
            'destination': destination_indices,
            'rank': neighbor_ranks,
            'similarity': similarity_scores.astype(cupy.float32),
        }
    )

    # Remove self-loops (can occur when k > number of unique points)
    directed_edges_df = directed_edges_df[
        directed_edges_df['source'] != directed_edges_df['destination']
    ]

    num_directed_edges = len(directed_edges_df)
    logger.debug(f'Created {num_directed_edges:,} directed k-NN edges (excluding self-loops)')

    return directed_edges_df


def _identify_mutual_edges(directed_edges_df: cudf.DataFrame) -> cudf.DataFrame:
    """
    Identifies bidirectional (mutual) edges and returns a de-duplicated list.

    A mutual edge exists when both (i->j) and (j->i) are present in the directed
    graph. This function finds such pairs by merging the edge list with itself,
    enforces a canonical direction (source < destination) to prevent duplicates,
    and returns a clean DataFrame with data from both edge perspectives.

    Args:
        directed_edges_df: DataFrame with directed edges containing columns:
            'source', 'destination', 'rank', 'similarity'

    Returns:
        DataFrame with de-duplicated mutual edges containing:
            - 'source', 'destination': Node pair (enforced source < destination)
            - 'source_to_dest_rank': Rank of destination in source's k-NN
            - 'dest_to_source_rank': Rank of source in destination's k-NN
            - 'source_to_dest_similarity': Similarity from source perspective
            - 'dest_to_source_similarity': Similarity from destination perspective
    """
    # Perform an inner self-merge to find mutual pairs.
    # This correctly matches a forward edge (A->B) from the left side with a
    # reverse edge (B->A) from the right side.
    mutual_edges_with_duplicates = directed_edges_df.merge(
        directed_edges_df,
        left_on=['source', 'destination'],
        right_on=['destination', 'source'],
        how='inner',
        suffixes=('_fwd', '_rev'),  # Suffixes for forward and reverse edge columns
    )

    # CRITICAL: De-duplicate the mutual edges.
    # The merge creates two rows for each mutual pair (one for A->B, one for B->A).
    # We enforce a canonical direction (source < destination) to keep only one
    # edge per pair, which is correct for an undirected graph.
    deduplicated_mutual_edges = mutual_edges_with_duplicates[
        mutual_edges_with_duplicates['source_fwd'] < mutual_edges_with_duplicates['destination_fwd']
    ]

    # Select and rename columns to the final desired format for clarity.
    final_df = cudf.DataFrame(
        {
            'source': deduplicated_mutual_edges['source_fwd'],
            'destination': deduplicated_mutual_edges['destination_fwd'],
            'source_to_dest_rank': deduplicated_mutual_edges['rank_fwd'],
            'source_to_dest_similarity': deduplicated_mutual_edges['similarity_fwd'],
            'dest_to_source_rank': deduplicated_mutual_edges['rank_rev'],
            'dest_to_source_similarity': deduplicated_mutual_edges['similarity_rev'],
        }
    )

    num_mutual_pairs = len(final_df)
    num_directed_edges = len(directed_edges_df)
    # The number of directed edges involved in mutual pairs is 2 * num_mutual_pairs
    mutual_percentage = (
        (2 * num_mutual_pairs / num_directed_edges * 100) if num_directed_edges > 0 else 0
    )

    logger.info(
        f'Identified {num_mutual_pairs:,} unique mutual pairs from {num_directed_edges:,} '
        f'directed edges ({mutual_percentage:.1f}% of directed edges are part of a mutual pair)'
    )

    return final_df


def _compute_hybrid_edge_weight(
    mutual_edges_df: cudf.DataFrame,
    rank_weight_factor: float = 1.0,
    similarity_weight_factor: float = 1.0,
) -> cudf.Series:
    """
    Calculates hybrid edge weights combining mutual rank and similarity scores.

    This weighting scheme creates robust edge weights that are high only when:
    1. Both nodes rank each other highly (low rank numbers)
    2. The nodes have high similarity in the embedding space

    The combination helps filter out spurious connections that might arise from
    only considering one factor.

    Args:
        mutual_edges_df: DataFrame with mutual edges containing rank and
            similarity information for both directions.
        rank_weight_factor: Scaling factor for the rank component (default=1.0).
        similarity_weight_factor: Scaling factor for the similarity component (default=1.0).

    Returns:
        A cuDF Series containing the computed hybrid weights for each edge.

    Notes:
        - Lower ranks (closer neighbors) produce higher weights
        - The '+2' in rank calculation prevents division by zero and provides scaling
        - Final weight is the product of rank and similarity components
    """
    # Calculate rank-based weight component
    # This is inversely proportional to the sum of mutual ranks
    # Adding 2 prevents division by zero and scales appropriately
    rank_sum = mutual_edges_df['source_to_dest_rank'] + mutual_edges_df['dest_to_source_rank']
    rank_based_weight = rank_weight_factor / (rank_sum + 2.0)

    # Calculate similarity-based weight component
    # Average the bidirectional similarities for symmetry
    avg_similarity = (
        mutual_edges_df['source_to_dest_similarity'] + mutual_edges_df['dest_to_source_similarity']
    ) / 2.0
    similarity_based_weight = similarity_weight_factor * avg_similarity

    # Combine components multiplicatively
    # High weight requires BOTH high rank AND high similarity
    hybrid_weight = rank_based_weight * similarity_based_weight

    # Log weight statistics for monitoring
    weight_stats = {
        'min': float(hybrid_weight.min()),
        'max': float(hybrid_weight.max()),
        'mean': float(hybrid_weight.mean()),
        'std': float(hybrid_weight.std()),
    }
    logger.debug(
        f'Hybrid weight statistics: min={weight_stats["min"]:.4f}, '
        f'max={weight_stats["max"]:.4f}, mean={weight_stats["mean"]:.4f}, '
        f'std={weight_stats["std"]:.4f}'
    )

    return hybrid_weight


def build_mutual_rank_graph(
    embedding_vectors: cupy.ndarray,
    k_neighbors: int,
    distance_metric: str = 'cosine',
    min_edge_weight: float | None = None,
    rank_weight_factor: float = 1.0,
    similarity_weight_factor: float = 1.0,
) -> tuple[cugraph.Graph, cudf.DataFrame]:
    """
    Constructs a mutual k-nearest neighbor graph with hybrid edge weighting.

    This function creates a robust similarity graph where edges exist only between
    mutually close points (each point is in the other's k-nearest neighbors).
    Edge weights combine both rank information and similarity scores to provide
    a nuanced measure of connection strength.

    The resulting graph is particularly useful for:
    - Clustering in high-dimensional spaces
    - Entity resolution and deduplication
    - Community detection in embedding spaces
    - Outlier detection (isolated nodes)

    Args:
        embedding_vectors: A 2D CuPy array of shape (n_samples, n_features)
            containing embedding vectors for all data points.
        k_neighbors: Number of nearest neighbors to consider. Higher values
            create denser graphs but increase computation time.
        distance_metric: Distance metric for k-NN search ('cosine', 'euclidean', etc.).
            'cosine' is recommended for normalized embeddings.
        min_edge_weight: Optional minimum weight threshold. Edges with weights
            below this value are removed. None keeps all mutual edges.
        rank_weight_factor: Scaling factor for rank-based weight component.
        similarity_weight_factor: Scaling factor for similarity-based weight component.

    Returns:
        A tuple containing:
            - cugraph.Graph: The constructed mutual k-NN graph with weighted edges
            - cudf.DataFrame: Edge list with columns 'source', 'destination', 'weight'

    Raises:
        ValueError: If input vectors are not 2D or k_neighbors is invalid.

    Example:
        >>> embeddings = cupy.random.randn(1000, 128)  # 1000 samples, 128 dimensions
        >>> graph, edges = build_mutual_rank_graph(embeddings, k_neighbors=10)
        >>> components = cugraph.connected_components(graph)
    """
    # Validate inputs
    if embedding_vectors.ndim != 2:
        raise ValueError(f'Expected 2D embedding array, got shape {embedding_vectors.shape}')

    num_samples, num_features = embedding_vectors.shape
    if k_neighbors <= 0 or k_neighbors > num_samples:
        raise ValueError(f'k_neighbors must be between 1 and {num_samples}, got {k_neighbors}')

    logger.info(
        f'Building mutual rank graph for {num_samples:,} samples with '
        f"k={k_neighbors}, metric='{distance_metric}'"
    )

    # Step 1: Compute directed k-NN edges
    directed_edges_df = _compute_directed_knn_edges(embedding_vectors, k_neighbors, distance_metric)

    # Step 2: Identify mutual (bidirectional) edges
    mutual_edges_df = _identify_mutual_edges(directed_edges_df)

    if mutual_edges_df.empty:
        logger.warning(
            'No mutual edges found. Consider increasing k_neighbors or checking data distribution.'
        )
        empty_graph = cugraph.Graph(directed=False)
        empty_edges = cudf.DataFrame({'source': [], 'destination': [], 'weight': []})
        return empty_graph, empty_edges

    # Step 3: Calculate hybrid weights for mutual edges
    mutual_edges_df['weight'] = _compute_hybrid_edge_weight(
        mutual_edges_df, rank_weight_factor, similarity_weight_factor
    )

    # Step 4: Apply optional weight threshold
    if min_edge_weight is not None:
        original_edge_count = len(mutual_edges_df)
        mutual_edges_df = mutual_edges_df[mutual_edges_df['weight'] >= min_edge_weight]
        filtered_count = original_edge_count - len(mutual_edges_df)

        if filtered_count > 0:
            logger.info(
                f'Filtered {filtered_count:,} edges below weight threshold {min_edge_weight}'
            )

        if mutual_edges_df.empty:
            logger.warning(f'All edges removed by weight threshold {min_edge_weight}')
            empty_graph = cugraph.Graph(directed=False)
            empty_edges = cudf.DataFrame({'source': [], 'destination': [], 'weight': []})
            return empty_graph, empty_edges

    # Step 5: Prepare final edge list (keeping only necessary columns)
    final_edge_list = mutual_edges_df[['source', 'destination', 'weight']].copy()

    # Step 6: Construct the undirected cuGraph Graph object
    mutual_knn_graph = cugraph.Graph(directed=False)
    mutual_knn_graph.from_cudf_edgelist(
        final_edge_list,
        source='source',
        destination='destination',
        edge_attr='weight',
        renumber=False,  # Preserve original vertex IDs for mapping back to data
    )

    # Log final graph statistics
    num_vertices = mutual_knn_graph.number_of_vertices()
    num_edges = mutual_knn_graph.number_of_edges()
    avg_degree = (2 * num_edges) / num_vertices if num_vertices > 0 else 0

    logger.info(
        f'Successfully built mutual rank graph: {num_vertices:,} vertices, '
        f'{num_edges:,} edges, average degree: {avg_degree:.1f}'
    )

    return mutual_knn_graph, final_edge_list
