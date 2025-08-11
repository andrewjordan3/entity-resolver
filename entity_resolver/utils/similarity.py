# entity_resolver/utils/similarity.py
"""
This module provides GPU-accelerated utilities for calculating string
similarity and finding similar pairs within a dataset using TF-IDF and
Nearest Neighbors.
"""

import cudf
import cupy
import logging
from typing import Dict, Any
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.neighbors import NearestNeighbors
from .graph import create_edge_list

# Set up a logger for this module.
logger = logging.getLogger(__name__)


def calculate_similarity_gpu(
    series_a: cudf.Series,
    series_b: cudf.Series,
    tfidf_params: Dict
) -> cudf.Series:
    """
    Calculates the row-wise cosine similarity between two cuDF string series.

    This function vectorizes two series of strings using a shared TF-IDF
    vocabulary and then computes their pairwise cosine similarity.

    Args:
        series_a: The first cuDF Series of strings.
        series_b: The second cuDF Series of strings (must have the same index).
        tfidf_params: A dictionary of parameters for TfidfVectorizer.

    Returns:
        A cuDF Series of float values representing the cosine similarity
        for each corresponding row in the input series.
    """
    logger.debug(f"Calculating row-wise similarity for two series of length {len(series_a)}.")
    # Ensure consistent data types and handle nulls before processing.
    series_a = series_a.fillna('').astype(str)
    series_b = series_b.fillna('').astype(str)

    # To ensure a fair comparison, the TF-IDF vectorizer must be fitted on the
    # combined vocabulary of both series. This guarantees that the same words
    # map to the same feature indices and have consistent IDF weights.
    combined_series = cudf.concat([series_a, series_b]).unique()
    vectorizer = TfidfVectorizer(**tfidf_params)
    vectorizer.fit(combined_series)
    logger.debug(f"TF-IDF vectorizer fitted on a combined vocabulary of size {len(combined_series)}.")

    # Transform each series into a TF-IDF matrix with L2 normalization (the default).
    vectors_a = vectorizer.transform(series_a)
    vectors_b = vectorizer.transform(series_b)

    # A key property of L2-normalized vectors is that their cosine similarity
    # is equivalent to their dot product. This is a highly efficient operation.
    # We multiply element-wise and then sum across the feature dimension (axis=1).
    similarities = vectors_a.multiply(vectors_b).sum(axis=1)

    # Convert the resulting column vector to a flat CuPy array and then to a cuDF Series.
    return cudf.Series(cupy.asarray(similarities).flatten(), index=series_a.index)


def find_similar_pairs(
    string_series: cudf.Series,
    tfidf_params: Dict[str, Any],
    nn_params: Dict[str, Any],
    distance_threshold: float
) -> cudf.DataFrame:
    """
    Finds similar pairs within a single series of strings on the GPU.

    This function encapsulates the entire process of:
    1. Vectorizing the strings using TF-IDF.
    2. Building a NearestNeighbors model to find candidates.
    3. Filtering for pairs that are closer than a given distance threshold.

    Args:
        string_series: A cuDF Series of unique strings to compare against itself.
        vectorizer_params: A dictionary of parameters for TfidfVectorizer.
        nn_params: A dictionary of parameters for NearestNeighbors.
        distance_threshold: A float distance threshold (e.g., for cosine distance).
                            A *lower* value means a *stricter* similarity requirement.

    Returns:
        A cuDF DataFrame with 'source' and 'destination' columns representing the
        indices of the matched pairs in the input string_series.
    """
    if len(string_series) < 2:
        logger.warning("find_similar_pairs requires at least two strings to compare.")
        return cudf.DataFrame({'source': [], 'destination': []})

    logger.info(f"Finding similar pairs within a series of {len(string_series)} strings.")
    logger.debug(f"Using distance threshold: {distance_threshold}")

    # Step 1: Vectorize the input strings into a TF-IDF matrix.
    vectorizer = TfidfVectorizer(**tfidf_params)
    tfidf_matrix = vectorizer.fit_transform(string_series)

    # Step 2: Build and fit the NearestNeighbors model.
    nn_model = NearestNeighbors(**nn_params)
    nn_model.fit(tfidf_matrix)
    distances, indices = nn_model.kneighbors(tfidf_matrix)
    logger.debug(f"Found neighbors for {len(indices)} items.")

    # Step 3: Use the dedicated graph utility to convert the k-NN results
    # into a filtered edge list based on the distance threshold.
    matched_pairs = create_edge_list(
        indices=indices,
        distances=distances,
        threshold=distance_threshold
    )

    # Return only the source and destination columns, which correspond to the
    # integer indices of the items in the original `string_series`.
    return matched_pairs[['source', 'destination']]
