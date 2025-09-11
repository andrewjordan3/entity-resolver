# entity_resolver/utils/similarity.py
"""
This module provides GPU-accelerated utilities for calculating string
similarity and finding similar pairs within a dataset using TF-IDF and
Nearest Neighbors.
"""

import gc
import cudf
import cupy
import logging
from typing import Dict, Any
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.neighbors import NearestNeighbors
from .graph import create_edge_list
from .text import nfkc_normalize_series

# Set up a logger for this module.
logger = logging.getLogger(__name__)


def calculate_similarity_gpu(
    series_a: cudf.Series,
    series_b: cudf.Series,
    tfidf_params: Dict
) -> cudf.Series:
    """
    Calculates the row-wise cosine similarity between two cuDF string series,
    robustly handling strings that are too short for character n-gram generation.

    This function vectorizes two series of strings using a shared TF-IDF
    vocabulary and then computes their pairwise cosine similarity. It preemptively
    filters out pairs where at least one string is too short for the specified
    n-gram size, assigning them a similarity of 0.

    Args:
        series_a: The first cuDF Series of strings.
        series_b: The second cuDF Series of strings (must have the same index).
        tfidf_params: A dictionary of parameters for TfidfVectorizer.
                      Must include 'analyzer' and 'ngram_range' if using
                      character n-grams.

    Returns:
        A cuDF Series of float values representing the cosine similarity
        for each corresponding row in the input series.
    """
    logger.debug(f"Calculating row-wise similarity for two series of length {len(series_a)}.")
    
    # Ensure consistent data types and handle nulls before processing.
    # 1) Base sanitation
    series_a = series_a.fillna('').astype('str').str.strip()
    series_b = series_b.fillna('').astype('str').str.strip()

    # 2) Unicode normalization (fold weird forms to standard ones)
    series_a = nfkc_normalize_series(series_a)
    series_b = nfkc_normalize_series(series_b)

    # 3) Remove zero-widths & odd spaces, then collapse spaces
    ZW_WEIRD = r'[\u00A0\u1680\u180E\u2000-\u200D\u202F\u205F\u2060\u3000\uFEFF]+'
    series_a = series_a.str.replace(ZW_WEIRD, '', regex=True).str.normalize_spaces()
    series_b = series_b.str.replace(ZW_WEIRD, '', regex=True).str.normalize_spaces()

    # Initialize the final result series with a default similarity of 0.0.
    # This ensures that any rows we skip will have a defined, logical similarity score.
    result_series = cudf.Series(cupy.zeros(len(series_a)), index=series_a.index, dtype='float32')

    # --- Robustness Check ---
    # Determine the minimum n-gram size from the TF-IDF parameters.
    # This logic is only necessary if the analyzer is character-based.
    min_n = 0
    if tfidf_params.get('analyzer') in ['char', 'char_wb']:
        min_n = min(tfidf_params.get('ngram_range', (1, 1)))

    # Create a mask to identify rows where BOTH strings are long enough for n-gram generation.
    # If not using character n-grams (min_n=0), all rows are considered valid initially.
    if min_n > 0:
        valid_mask = (series_a.str.len() >= min_n) & (series_b.str.len() >= min_n)
    else:
        # If not using n-grams, all rows are valid from a length perspective.
        valid_mask = cudf.Series(cupy.ones(len(series_a), dtype=bool), index=series_a.index)

    # If there are no valid rows to process, we can return the zeros series immediately.
    if not valid_mask.any():
        logger.debug("No rows with sufficient string length for n-gram generation. Returning all zeros.")
        try:
            a_sample = series_a.head(5)
            b_sample = series_b.head(5)
            sample_df = cudf.DataFrame({
                'A': a_sample,
                'A_len': a_sample.str.len(),
                'B': b_sample,
                'B_len': b_sample.str.len(),
            })
            logger.debug(f"Sample at insufficient-length condition (head 5), min_n={min_n}:\n"
                         f"{sample_df.to_pandas().to_string(index=False)}")
            a_len_min = int(series_a.str.len().min())
            b_len_min = int(series_b.str.len().min())
            a_len_max = int(series_a.str.len().max())
            b_len_max = int(series_b.str.len().max())
            logger.debug(f"Length stats — A[min={a_len_min}, max={a_len_max}], "
                         f"B[min={b_len_min}, max={b_len_max}], tfidf_params={tfidf_params}")
        except Exception as log_ex:
            logger.debug(f"Failed to log insufficient-length samples due to: {log_ex}")
        return result_series

    # Filter the series to only the valid rows that we need to process.
    series_a_valid = series_a[valid_mask]
    series_b_valid = series_b[valid_mask]
    
    # It's possible the valid series are now empty or only contain empty strings.
    # We should only proceed if there's actual text content to analyze.
    if series_a_valid.empty and series_b_valid.empty:
        logger.debug("After filtering, no valid data remains to be processed.")
        return result_series
    
    # Apply additional filtering to handle edge cases
    # Some strings might pass the length check but still cause issues in n-gram generation
    # (e.g., strings that are all spaces or special characters)
    if min_n > 0:
        # Re-strip and check length after all preprocessing
        series_a_valid = series_a_valid.str.strip()
        series_b_valid = series_b_valid.str.strip()
        
        # Create a secondary mask for strings that are still valid after stripping
        secondary_mask_a = series_a_valid.str.len() >= min_n
        secondary_mask_b = series_b_valid.str.len() >= min_n
        combined_secondary_mask = secondary_mask_a & secondary_mask_b
        
        if not combined_secondary_mask.any():
            logger.debug("No strings remain valid after secondary filtering.")
            return result_series
        
        # Apply secondary filtering
        series_a_valid = series_a_valid[combined_secondary_mask]
        series_b_valid = series_b_valid[combined_secondary_mask]

    logger.debug(f"Processing {len(series_a_valid)} valid rows.")

    # To ensure a fair comparison, the TF-IDF vectorizer must be fitted on the
    # combined vocabulary of both series. This guarantees that the same words
    # map to the same feature indices and have consistent IDF weights.
    combined_series = cudf.concat(
        [
            series_a_valid.reset_index(drop=True),
            series_b_valid.reset_index(drop=True),
        ],
        ignore_index=True,
    ).dropna().astype('str')
    
    # The combined series could be empty if the inputs only contained empty strings.
    if combined_series.empty:
        logger.debug("Combined series for fitting TF-IDF is empty. Returning.")
        return result_series

    try:
        vectorizer = TfidfVectorizer(**tfidf_params)
        vectorizer.fit(combined_series)
        vocab_size = (len(vectorizer.vocabulary_)
                    if hasattr(vectorizer, "vocabulary_") and vectorizer.vocabulary_ is not None
                    else None)
        logger.debug(
            f"TF-IDF vectorizer fitted. "
            f"{'Vocabulary size: ' + str(vocab_size) if vocab_size is not None else 'Vocabulary size unavailable'}."
        )

        # The combined series is no longer needed after fitting the vectorizer.
        del combined_series

        # Transform each valid series into a TF-IDF matrix with L2 normalization (the default).
        vectors_a = vectorizer.transform(series_a_valid)
        vectors_b = vectorizer.transform(series_b_valid)

        # A key property of L2-normalized vectors is that their cosine similarity
        # is equivalent to their dot product. This is a highly efficient operation.
        # We multiply element-wise and then sum across the feature dimension (axis=1).
        similarities_valid = vectors_a.multiply(vectors_b).sum(axis=1)

        # The large TF-IDF matrices are no longer needed.
        del vectors_a
        del vectors_b

        # Create a cuDF Series from the calculated similarities, using the correct index.
        similarities_series = cudf.Series(
            cupy.asarray(similarities_valid).flatten(), 
            index=series_a_valid.index
        )

        # The intermediate similarities array is no longer needed.
        del similarities_valid

        # Place calculated similarities back at their original positions
        # series_a_valid.index contains the indices after both filters were applied
        result_series.loc[series_a_valid.index] = similarities_series

    except RuntimeError as e:
        logger.debug(f"RuntimeError in calculate_similarity_gpu: {e}")
        logger.debug(f"tfidf_params: {tfidf_params} | min_n: {min_n}")
        try:
            a_sample = series_a_valid.head(5)
            b_sample = series_b_valid.head(5)
            sample_df = cudf.DataFrame({
                'A': a_sample,
                'A_len': a_sample.str.len(),
                'B': b_sample,
                'B_len': b_sample.str.len(),
            })
            logger.debug("Sample of valid rows at failure (head 5):\n"
                         f"{sample_df.to_pandas().to_string(index=False)}")
            if 'combined_series' in locals():
                comb_sample = combined_series.head(10)
                logger.debug(f"Combined series sample (head 10): {comb_sample.to_pandas().tolist()}")
        except Exception as log_ex:
            logger.debug(f"Failed to log samples due to: {log_ex}")

        if "Insufficient number of characters" in str(e):
            logger.warning(f"N-gram generation failed despite checks. Returning zeros. Error: {e}")
            return result_series
        else:
            raise

    finally:
        # --- Memory Management ---
        # This block ensures cleanup happens regardless of success or failure.
        # It's the most important part for preventing memory leaks over many calls.
        gc.collect()
        cupy.get_default_memory_pool().free_all_blocks()
        logger.debug("GPU memory cleanup complete.")

    return result_series

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

    logger.debug(f"Finding similar pairs within a series of {len(string_series)} strings.")
    logger.debug(f"Using distance threshold: {distance_threshold}")

    # Step 1: Vectorize the input strings into a TF-IDF matrix.
    vectorizer = TfidfVectorizer(**tfidf_params)
    tfidf_matrix = vectorizer.fit_transform(string_series)

    # Step 2: Build and fit the NearestNeighbors model.
    nn_model = NearestNeighbors(**nn_params)
    nn_model.fit(tfidf_matrix)
    distances, indices = nn_model.kneighbors(tfidf_matrix)
    logger.debug(f"Found neighbors for {len(indices)} items.")

    # We are now done with the large TF-IDF matrix, so we can delete it.
    del tfidf_matrix

    # Step 3: Use the dedicated graph utility to convert the k-NN results
    # into a filtered edge list based on the distance threshold.
    matched_pairs = create_edge_list(
        neighbor_indices=indices,
        neighbor_distances=distances,
        distance_threshold=distance_threshold
    )

    # We are now done with the large distance and index arrays.
    del distances
    del indices

    # Step 4: Force cleanup before returning.
    gc.collect()
    cupy.get_default_memory_pool().free_all_blocks()

    # Return only the source and destination columns, which correspond to the
    # integer indices of the items in the original `string_series`.
    return matched_pairs[['source', 'destination']]
