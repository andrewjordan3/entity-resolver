# entity_resolver/utils/graph.py
"""
This module provides GPU-accelerated utilities for graph construction and
analysis using cuDF, cuGraph, and cuML. It includes functions for creating
graphs from nearest neighbor data and finding connected components.
"""

import cupy
import cudf
import cugraph
from cuml.neighbors import NearestNeighbors
import logging

# Set up a logger for this module.
logger = logging.getLogger(__name__)


def create_edge_list(
    indices: cupy.ndarray,
    distances: cupy.ndarray,
    threshold: float
) -> cudf.DataFrame:
    """
    Creates a cuDF DataFrame of graph edges from NearestNeighbors output.

    This function transforms the 2D arrays of neighbor indices and distances
    into a "long" format edge list DataFrame, filtering out edges that exceed
    a specified distance threshold and any self-loops.

    Args:
        indices: The CuPy array of neighbor indices from kneighbors().
        distances: The CuPy array of neighbor distances from kneighbors().
        threshold: The distance threshold below which a pair is considered an edge.

    Returns:
        A cuDF DataFrame with 'source', 'destination', and 'distance' columns.
    """
    n_rows, n_neighbors = indices.shape
    logger.debug(f"Creating edge list from neighbors matrix of shape ({n_rows}, {n_neighbors}).")

    # Flatten the 2D matrices into 1D Series.
    source_nodes = cupy.arange(n_rows, dtype='int32').repeat(n_neighbors)
    destination_nodes = indices.flatten()
    distance_values = distances.flatten()

    # Create the initial edge list DataFrame.
    edge_list = cudf.DataFrame({
        'source': source_nodes,
        'destination': destination_nodes,
        'distance': distance_values
    })

    # Filter for valid edges: distance must be below the threshold and the
    # source cannot be the same as the destination (no self-loops).
    valid_edges = edge_list[
        (edge_list['distance'] < threshold) &
        (edge_list['source'] != edge_list['destination'])
    ]
    logger.info(f"Created edge list with {len(valid_edges)} edges after filtering.")
    return valid_edges


def find_graph_components(
    edge_list: cudf.DataFrame,
    source_col: str = 'source',
    destination_col: str = 'destination',
    vertex_col_name: str = 'vertex',
    component_col_name: str = 'component_id'
) -> cudf.DataFrame:
    """
    Finds connected components in a graph from an edge list using cuGraph.

    This function encapsulates the boilerplate for creating a cuGraph Graph
    object and running the weakly connected components algorithm.

    Args:
        edge_list: A cuDF DataFrame representing graph edges.
        source_col: The name of the source column in the edge_list.
        destination_col: The name of the destination column in the edge_list.
        vertex_col_name: The desired name for the output vertex column.
        component_col_name: The desired name for the output component ID column.

    Returns:
        A cuDF DataFrame mapping each vertex to its component ID.
    """
    if edge_list.empty:
        logger.warning("find_graph_components received an empty edge list.")
        return cudf.DataFrame({vertex_col_name: [], component_col_name: []})

    logger.debug(f"Finding components for graph with {len(edge_list)} edges.")
    graph = cugraph.Graph()
    graph.from_cudf_edgelist(
        edge_list,
        source=source_col,
        destination=destination_col
    )

    # Run the weakly connected components algorithm.
    components = cugraph.weakly_connected_components(graph)

    # Rename columns to the desired generic output names for consistency.
    components = components.rename(columns={
        'labels': component_col_name,
        'vertex': vertex_col_name
    })
    
    num_components = len(components[component_col_name].unique())
    logger.info(f"Found {num_components} connected components.")
    return components


# --- Mutual Rank Graph Helper Functions ---

def _get_directed_knn_edges(vectors: cupy.ndarray, k: int) -> cudf.DataFrame:
    """Finds k-nearest neighbors and formats them as a directed edge list."""
    logger.debug(f"Finding {k}-nearest neighbors for {vectors.shape[0]} vectors.")
    nn_model = NearestNeighbors(n_neighbors=k, metric='cosine').fit(vectors)
    distances, indices = nn_model.kneighbors(vectors)

    source_nodes = cupy.repeat(cupy.arange(vectors.shape[0]), k)
    destination_nodes = indices.ravel()
    
    # Create the DataFrame with ranks and similarities for each directed edge.
    directed_edges = cudf.DataFrame({
        'source': source_nodes,
        'destination': destination_nodes,
        'rank': cupy.tile(cupy.arange(k), vectors.shape[0]),
        'similarity': 1 - distances.ravel()
    })
    
    # Filter out self-loops, which can occur if k is larger than the number of unique points.
    directed_edges = directed_edges[directed_edges['source'] != directed_edges['destination']]
    logger.debug(f"Created {len(directed_edges)} directed k-NN edges.")
    return directed_edges


def _find_mutual_edges(directed_edges: cudf.DataFrame) -> cudf.DataFrame:
    """Identifies mutual neighbors by performing a self-merge."""
    # To find mutual neighbors, we need to know the rank in both directions.
    # We rename columns to prepare for a merge that finds reciprocal pairs.
    edges_renamed = directed_edges.rename(columns={
        'source': 'destination',
        'destination': 'source',
        'rank': 'rank_of_source_for_dest',
        'similarity': 'similarity_of_source_for_dest'
    })
    
    # Merge the original directed edges with the renamed (swapped) version.
    # An inner merge ensures that we only keep pairs where (i -> j) and (j -> i) both exist.
    mutual_edges = directed_edges.merge(
        edges_renamed,
        on=['source', 'destination']
    ).rename(columns={'rank': 'rank_of_dest_for_source', 'similarity': 'similarity_of_dest_for_source'})
    
    logger.info(f"Found {len(mutual_edges)} mutual edges from directed k-NN graph.")
    return mutual_edges


def _calculate_hybrid_weight(mutual_edges: cudf.DataFrame) -> cudf.Series:
    """Calculates a hybrid weight combining mutual rank and cosine similarity."""
    # 1. Rank-based component: This is robust to variations in data density.
    # The '+ 2' ensures the denominator is never zero and scales the weight.
    rank_weight = 1.0 / (
        mutual_edges['rank_of_dest_for_source'] +
        mutual_edges['rank_of_source_for_dest'] + 2
    )
    
    # 2. Similarity component: This is sensitive to the actual distance.
    # We average the two directional similarities to maintain symmetry.
    similarity_weight = (
        mutual_edges['similarity_of_dest_for_source'] +
        mutual_edges['similarity_of_source_for_dest']
    ) / 2.0
    
    # 3. Hybrid weight: The final weight is high only if points are both
    # highly ranked for each other AND are very close in the vector space.
    return rank_weight * similarity_weight


def build_mutual_rank_graph(vectors: cupy.ndarray, k: int) -> cugraph.Graph:
    """
    Builds a k-NN graph with a hybrid weighting scheme.

    This method creates a robust graph where edge weights are high only for
    pairs of points that are mutually close (i.e., each is one of the other's
    k-nearest neighbors) and have high cosine similarity.

    Args:
        vectors: The embedding vectors for all data points.
        k: The number of nearest neighbors to consider for the graph.

    Returns:
        A cugraph.Graph object with weighted, undirected edges.
    """
    # Step 1: Find all directed k-NN edges (i -> j).
    directed_edges = _get_directed_knn_edges(vectors, k)
    
    # Step 2: Identify which of those edges are mutual (i <-> j).
    mutual_edges = _find_mutual_edges(directed_edges)
    
    if mutual_edges.empty:
        logger.warning("No mutual edges found; returning an empty graph.")
        return cugraph.Graph()

    # Step 3: Calculate the hybrid weight for each mutual edge.
    mutual_edges['weight'] = _calculate_hybrid_weight(mutual_edges)

    # Step 4: Construct the final, undirected cugraph Graph object.
    snn_graph = cugraph.Graph(directed=False)
    # Set renumber=False to ensure vertex IDs are preserved, which is critical
    # for mapping graph analysis results back to the original data.
    snn_graph.from_cudf_edgelist(
        mutual_edges,
        source='source',
        destination='destination',
        edge_attr='weight',
        renumber=False
    )
    logger.info(f"Successfully built mutual rank graph with {snn_graph.number_of_edges()} edges.")
    return snn_graph
