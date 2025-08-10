# entity_resolver/utils/text.py
"""
This module provides GPU-accelerated utilities for text processing and for
selecting a canonical representation from a group of string candidates.
"""

import logging
import cudf
import cupy
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.metrics.pairwise_distances import pairwise_distances

# Set up a logger for this module.
logger = logging.getLogger(__name__)


def _calculate_centrality_score(
    unique_names: cudf.Series,
    name_counts: cudf.Series
) -> cupy.ndarray:
    """
    Calculates a centrality score for each unique name.

    This score is high for names that are both frequent and highly similar to
    other frequent names in the group. It uses TF-IDF character n-grams to
    measure similarity.

    Args:
        unique_names: A Series of unique name strings.
        name_counts: A Series containing the frequency count for each unique name.

    Returns:
        A CuPy array of centrality scores, one for each unique name.
    """
    # Using character n-grams is effective for capturing misspellings and variations.
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))
    # We must reset the index before vectorizing to avoid potential cuML errors.
    tfidf_matrix = vectorizer.fit_transform(unique_names.reset_index(drop=True))

    # Calculate the cosine similarity between all pairs of unique names.
    similarity_matrix = 1 - pairwise_distances(tfidf_matrix, metric='cosine')

    # Calculate frequency weights (proportion of total).
    total_items = name_counts.sum()
    freq_weights = name_counts / total_items

    # The core of the centrality score: for each name, sum the similarities
    # to all other names, weighted by how frequent those other names are.
    # A high score means "I am similar to other popular names."
    # The '@' operator performs matrix multiplication.
    centrality_score = similarity_matrix @ freq_weights.values

    return cupy.asarray(centrality_score)


def _calculate_length_bonus(unique_names: cudf.Series) -> cupy.ndarray:
    """
    Calculates a descriptiveness bonus based on name length.

    Longer names are often more descriptive and less likely to be abbreviations,
    so they receive a bonus. The score is logarithmic to provide diminishing
    returns, preventing absurdly long names from dominating.

    Args:
        unique_names: A Series of unique name strings.

    Returns:
        A CuPy array of length bonus scores, one for each unique name.
    """
    lengths = unique_names.str.len().astype('float32')
    # The '+ 1' avoids taking the log of zero.
    # The clip prevents the bonus from growing excessively large.
    length_bonus = cupy.log(lengths + 1).clip(max=3.5)
    return cupy.asarray(length_bonus)


def get_canonical_name_gpu(name_series: cudf.Series) -> str:
    """
    Selects the best canonical name from a Series of candidates on the GPU.

    This function identifies the best name by scoring each unique candidate based
    on a combination of three factors:
    1.  **Centrality**: How similar a name is to other frequent names.
    2.  **Frequency**: How often the name itself appears.
    3.  **Descriptiveness**: A bonus for length, favoring complete names over abbreviations.

    Args:
        name_series: A cuDF Series containing all name candidates for a single group.

    Returns:
        The string of the highest-scoring canonical name, or an empty string
        if the input is empty.
    """
    if name_series.empty:
        logger.warning("get_canonical_name_gpu received an empty series.")
        return ""

    unique_names = name_series.unique()
    if len(unique_names) == 1:
        return unique_names.iloc[0]

    logger.info(f"Finding canonical name from {len(unique_names)} unique candidates (out of {len(name_series)} total).")

    # --- Score Calculation ---
    name_counts = name_series.value_counts().reindex(unique_names).fillna(0)

    # 1. Calculate the centrality score.
    centrality_score = _calculate_centrality_score(unique_names, name_counts)

    # 2. Get the direct frequency score.
    frequency_score = (name_counts / name_counts.sum()).values

    # 3. Calculate the length bonus.
    length_bonus = _calculate_length_bonus(unique_names)

    # --- Combine Scores ---
    # The base score combines centrality and raw frequency. A name can score
    # well by being frequent itself, or by being very similar to other
    # frequent names.
    base_score = centrality_score + frequency_score

    # The final score is modulated by the length bonus. This helps break ties
    # and promotes more descriptive names.
    final_scores = base_score * length_bonus
    logger.debug(f"Calculated final scores for {len(final_scores)} unique names.")

    # Find the index of the highest score and return the corresponding name.
    best_name_index = final_scores.argmax()
    best_name = unique_names.iloc[int(best_name_index)]
    logger.info(f"Selected '{best_name}' as the canonical name.")

    return best_name
