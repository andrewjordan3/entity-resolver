# entity_resolver/utils/similarity.py
"""
This module provides GPU-accelerated utilities for calculating string
similarity and finding similar pairs within a dataset using TF-IDF and
Nearest Neighbors.
"""

import cudf
import cupy
import cupyx.scipy.sparse
import logging
from typing import Dict, Any
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.neighbors import NearestNeighbors
from .graph import create_edge_list
from .text import nfkc_normalize_series
from .clean_mem import gpu_memory_cleanup
from .matrix_ops import ensure_finite_matrix, prune_sparse_matrix, scale_by_frobenius_norm
from .vector import normalize_rows
from ..config import SimilarityTfidfParams, SimilarityNnParams

# Set up a logger for this module.
logger = logging.getLogger(__name__)

def _log_series_samples(
    series_a: cudf.Series,
    series_b: cudf.Series,
    min_n: int,
    message: str
) -> None:
    """
    A helper function to log descriptive statistics and samples from two series
    for debugging purposes.

    Args:
        series_a: The first cuDF Series.
        series_b: The second cuDF Series.
        min_n: The minimum n-gram size used for filtering.
        message: A descriptive message to include in the log header.
    """
    try:
        sample_df = cudf.DataFrame({
            'series_a_sample': series_a.head(5),
            'series_a_len': series_a.head(5).str.len(),
            'series_b_sample': series_b.head(5),
            'series_b_len': series_b.head(5).str.len(),
        })
        logger.debug(
            f"{message} (min_n={min_n}):\n"
            f"{sample_df.to_pandas().to_string(index=True)}"
        )
        a_len = series_a.str.len()
        b_len = series_b.str.len()
        stats = {
            "A_min_len": int(a_len.min()),
            "A_max_len": int(a_len.max()),
            "B_min_len": int(b_len.min()),
            "B_max_len": int(b_len.max()),
        }
        logger.debug(f"Length statistics for logged series: {stats}")
    except Exception as log_ex:
        logger.debug(f"Failed to log series samples due to: {log_ex}")

def calculate_similarity_gpu(
    series_a: cudf.Series,
    series_b: cudf.Series,
    tfidf_params: SimilarityTfidfParams,
    min_unique_for_tfidf: int = 24,
) -> cudf.Series:
    """
    Calculates the row-wise cosine similarity between two cuDF string series,
    robustly handling strings that are too short for character n-gram generation.

    This function vectorizes two series of strings using a shared TF-IDF
    vocabulary and then computes their pairwise cosine similarity. It preemptively
    filters out pairs where at least one string is too short for the specified
    n-gram size after cleaning, assigning them a similarity score of 0.0.

    Args:
        series_a: The first cuDF Series of strings.
        series_b: The second cuDF Series of strings (must have the same index).
        tfidf_params: A dictionary of parameters for TfidfVectorizer.
                      Must include 'analyzer' and 'ngram_range' if using
                      character n-grams.

    Returns:
        A cuDF Series of float32 values representing the cosine similarity
        for each corresponding row in the input series.
    """
    # --- 3. TF-IDF Vectorization and Similarity Calculation ---
    # Initialize variables to None for cleanup tracking
    combined_series = None
    vectorizer = None
    vectors_a = None
    vectors_a_proc = None
    vectors_b = None
    vectors_b_proc = None
    similarities_array = None
    combined_proc_vectors = None

    if not series_a.index.equals(series_b.index):
        raise ValueError("Input series 'series_a' and 'series_b' must have the same index.")

    logger.debug(f"Starting row-wise similarity for two series of length {len(series_a)}.")

    # --- 1. Preprocessing and Cleaning ---
    def clean_series(series: cudf.Series) -> cudf.Series:
        """Applies a full cleaning pipeline to a string series."""
        # Fill nulls and ensure string type before normalization.
        s = series.fillna('').astype('str')
        # Apply Unicode and compatibility normalization. The nfkc_normalize_series
        # function handles a comprehensive cleaning pipeline including NFKC,
        # symbol folding, and all whitespace normalization (collapsing and stripping)
        s = nfkc_normalize_series(s)
        return s

    series_a_cleaned = clean_series(series_a)
    series_b_cleaned = clean_series(series_b)

    # Initialize the final result series with a default similarity of 0.0.
    # This ensures that any rows we filter out will have a defined score.
    result_series = cudf.Series(cupy.zeros(len(series_a)), index=series_a.index, dtype='float32')

    # --- Check if we should use edit distance instead of TF-IDF ---
    # Count unique strings across both series to determine approach
    combined_unique = cudf.concat([series_a_cleaned, series_b_cleaned]).unique()
    n_unique = len(combined_unique)

    # --- Use Edit Distance for Small Datasets ---
    if n_unique < min_unique_for_tfidf:
        logger.debug(f"Using vectorized edit distance for {n_unique} unique strings (< {min_unique_for_tfidf})")
        
        # 1. Filter out rows where either string is empty.
        valid_mask = (series_a_cleaned.str.len() > 0) & (series_b_cleaned.str.len() > 0)
        
        if not valid_mask.any():
            logger.debug("No valid non-empty string pairs found.")
            return result_series
            
        # 2. Get the subset of series that will be processed.
        # The original index is correctly preserved here.
        valid_a = series_a_cleaned[valid_mask]
        valid_b = series_b_cleaned[valid_mask]
        
        # 3. Perform all calculations in a vectorized way (NO LOOPS).
        # Calculate raw Levenshtein distance for all pairs at once.
        raw_distances = valid_a.str.edit_distance(valid_b)
        
        # Calculate lengths needed for normalization.
        len_a = valid_a.str.len()
        len_b = valid_b.str.len()
        max_dist = len_a + len_b # This is the denominator in your formula
        
        # 4. Calculate normalized similarity using the formula for all pairs at once.
        # We use .values to get the underlying CuPy arrays for the calculation.
        similarities_array = cupy.exp(-2.0 * raw_distances.values / max_dist.values)
        
        # Create the final series with the correct index.
        similarities_series = cudf.Series(similarities_array, index=valid_a.index, dtype='float32')

        # 5. Update the result series. This remains the same.
        result_series.update(similarities_series)
        
        logger.debug(f"Edit distance similarity calculation complete. Mean similarity: {similarities_array.mean():.4f}")
        return result_series

    # --- 2. Identify Rows Valid for N-gram Processing ---
    # Determine the minimum n-gram size from the TF-IDF parameters.
    # This is only necessary if the analyzer is character-based.
    min_n = 0
    if tfidf_params.analyzer in ['char', 'char_wb']:
        min_n = min(tfidf_params.ngram_range)

    # Create a single boolean mask to identify rows where BOTH strings are long enough.
    # This check is performed AFTER all cleaning to be as accurate as possible.
    if min_n > 0:
        valid_rows_mask = (series_a_cleaned.str.len() >= min_n) & (series_b_cleaned.str.len() >= min_n)
    else:
        # If not using character n-grams, all rows are considered valid from a length perspective.
        valid_rows_mask = cudf.Series(cupy.ones(len(series_a), dtype=bool), index=series_a.index)

    # If there are no valid rows to process, we can return the zeros series immediately.
    if not valid_rows_mask.any():
        logger.debug("No rows with sufficient string length for n-gram generation. Returning all zeros.")
        # Log the CLEANED series as well to diagnose normalization issues.
        logger.debug("--- Logging Original Series for Context ---")
        _log_series_samples(series_a, series_b, min_n, "Sample at insufficient-length condition (ORIGINAL)")
        logger.debug("--- Logging Cleaned Series That Caused Failure ---")
        _log_series_samples(series_a_cleaned, series_b_cleaned, min_n, "Sample at insufficient-length condition (CLEANED)")
        return result_series

    # Filter the series down to only the valid rows that need processing.
    # The index from the original series is preserved in these filtered views.
    series_a_to_process = series_a_cleaned[valid_rows_mask]
    series_b_to_process = series_b_cleaned[valid_rows_mask]

    logger.debug(f"Found {len(series_a_to_process)} valid rows to process for similarity.")
    
    try:
        # To ensure a fair comparison, the TF-IDF vectorizer must be fitted on the
        # combined vocabulary of both series. This guarantees that the same features
        # map to the same indices and have consistent IDF weights.
        combined_series = cudf.concat(
            [series_a_to_process, series_b_to_process],
            ignore_index=True
        ).dropna()

        # It's possible the combined series is empty if inputs only contained empty strings
        # or strings that were filtered out.
        if combined_series.empty:
            logger.debug("Combined series for TF-IDF fitting is empty. No similarities to calculate.")
            return result_series

        vectorizer = TfidfVectorizer(**tfidf_params.model_dump())
        vectorizer.fit(combined_series)
        logger.debug(f"TF-IDF vectorizer fitted. Vocabulary size: {len(getattr(vectorizer, 'vocabulary_', {}))}")

        # Transform each valid series into a TF-IDF matrix with L2 normalization (the default).
        vectors_a = vectorizer.transform(series_a_to_process)
        vectors_b = vectorizer.transform(series_b_to_process)

        # 1. Identify active rows by calculating the number of non-zero elements
        #    per row directly from the CSR matrix's indptr array. This is the
        #    officially supported and most efficient method.
        #    The difference between consecutive elements of the indptr array
        #    gives the count of non-zero elements in each row.
        row_nnz_a = vectors_a.indptr[1:] - vectors_a.indptr[:-1]
        row_nnz_b = vectors_b.indptr[1:] - vectors_b.indptr[:-1]
        active_rows_mask = (row_nnz_a > 0) & (row_nnz_b > 0)

        if not cupy.any(active_rows_mask):
            logger.debug("No rows with overlapping non-empty vectors after transform. Returning all zeros.")
            return result_series

        # 2. Get the original index for active rows to map results back later.
        active_rows_cudf_mask = cudf.Series(active_rows_mask, index=series_a_to_process.index)
        active_index = series_a_to_process.index[active_rows_cudf_mask]

        # 3. Filter matrices to only active rows.
        vectors_a_proc = vectors_a[active_rows_mask, :]
        vectors_b_proc = vectors_b[active_rows_mask, :]

        # 4. Identify active columns by taking the union of column indices
        #    that have non-zero elements in either matrix. This is the most
        #    robust way to find the shared feature set without relying on
        #    the unsupported `axis` parameter in `getnnz`.
        unique_cols_a = cupy.unique(vectors_a_proc.indices)
        unique_cols_b = cupy.unique(vectors_b_proc.indices)
        active_cols_indices = cupy.union1d(unique_cols_a, unique_cols_b)

        # 5. Filter both matrices by the common active column indices.
        #    Slicing a CSR matrix by a list of column indices is efficient.
        vectors_a_proc = vectors_a_proc[:, active_cols_indices]
        vectors_b_proc = vectors_b_proc[:, active_cols_indices]

        # 6. Apply pre-processing to the synchronized matrices. These steps are intended
        #    to prevent CUDA errors on a clean, consistent dataset.
        vectors_a_proc.sum_duplicates()
        vectors_b_proc.sum_duplicates()
        vectors_a_proc = ensure_finite_matrix(vectors_a_proc, replace_non_finite=True, copy=False)
        vectors_b_proc = ensure_finite_matrix(vectors_b_proc, replace_non_finite=True, copy=False)
        
        # To ensure column pruning is consistent, we must determine the columns
        # to keep based on the combined statistics of both matrices.
        combined_proc_vectors = cupyx.scipy.sparse.vstack([vectors_a_proc, vectors_b_proc], format='csr')

        # Run prune on the combined matrix to get a single, consistent set of columns.
        # We only care about the column mask/indices that are kept.
        _, _, kept_cols = prune_sparse_matrix(combined_proc_vectors, copy=False)

        # Apply the common column filter to both matrices.
        if kept_cols is not None:
            vectors_a_proc = vectors_a_proc[:, kept_cols]
            vectors_b_proc = vectors_b_proc[:, kept_cols]        
        
        # With columns synchronized, we can now scale each matrix independently.
        vectors_a_proc, _ = scale_by_frobenius_norm(vectors_a_proc, copy=False)
        vectors_b_proc, _ = scale_by_frobenius_norm(vectors_b_proc, copy=False)

        # 7. Row-wise L2 normalization to prepare for cosine similarity calculation.
        vectors_a_proc = normalize_rows(vectors_a_proc, copy=False)
        vectors_b_proc = normalize_rows(vectors_b_proc, copy=False)
        vectors_a_proc.sort_indices()
        vectors_b_proc.sort_indices()

        # --- Stably Format Matrices Before Final Calculation ---
        # Enforce canonical format and data types for stability and performance.
        # This conditionally creates a copy only if a matrix is a complex view
        # or has a non-standard format, preventing memory corruption errors.
        
        # Process vectors_a_proc
        vectors_a_proc.data    = vectors_a_proc.data.astype(cupy.float32, copy=False)
        vectors_a_proc.indices = vectors_a_proc.indices.astype(cupy.int32,  copy=False)
        vectors_a_proc.indptr  = vectors_a_proc.indptr.astype(cupy.int32,  copy=False)
        needs_copy_a = (
            (vectors_a_proc.data.base is not None) or
            (vectors_a_proc.indices.base is not None) or
            (vectors_a_proc.indptr.base is not None) or
            (getattr(vectors_a_proc, "has_canonical_format", True) is False)
        )
        if needs_copy_a:
            vectors_a_proc = vectors_a_proc.copy()

        # Process vectors_b_proc
        vectors_b_proc.data    = vectors_b_proc.data.astype(cupy.float32, copy=False)
        vectors_b_proc.indices = vectors_b_proc.indices.astype(cupy.int32,  copy=False)
        vectors_b_proc.indptr  = vectors_b_proc.indptr.astype(cupy.int32,  copy=False)
        needs_copy_b = (
            (vectors_b_proc.data.base is not None) or
            (vectors_b_proc.indices.base is not None) or
            (vectors_b_proc.indptr.base is not None) or
            (getattr(vectors_b_proc, "has_canonical_format", True) is False)
        )
        if needs_copy_b:
            vectors_b_proc = vectors_b_proc.copy()

        assert vectors_a_proc.shape == vectors_b_proc.shape, (
            f"Vector shapes don't match after synchronized pruning.\n"
            f"Shape A: {vectors_a_proc.shape}\n"
            f"Shape B: {vectors_b_proc.shape}"
        )

        # A key property of L2-normalized vectors is that their cosine similarity
        # is equivalent to their dot product.
        # The sum operation on a sparse matrix returns a dense ndarray of shape (n, 1)
        similarities_array = vectors_a_proc.multiply(vectors_b_proc).sum(axis=1, dtype=cupy.float32)

        # Debug synchronization check to catch CUDA errors immediately after the
        # critical sparse matrix operations. This helps identify if the corruption
        # happens during the multiply/sum operations.
        if logger.isEnabledFor(logging.DEBUG):
            try:
                cupy.cuda.Stream.null.synchronize()
                logger.debug(f"CUDA sync successful after multiply/sum for {len(series_a_to_process)} rows")
            except cupy.cuda.runtime.CUDARuntimeError as e:
                logger.error(f"CUDA corruption detected immediately after multiply/sum operation!")
                logger.error(f"Error details: {e}")
                logger.error(f"Batch size: {len(series_a_to_process)}, Vocabulary size: {len(getattr(vectorizer, 'vocabulary_', {}))}")
                _log_series_samples(series_a_to_process, series_b_to_process, min_n, 
                                  "Data that triggered CUDA corruption during multiply/sum")
                raise

        # Now continue with flattening - ravel() converts (n,1) to (n)
        similarities_array = similarities_array.ravel()
            
        # Ensure the array is contiguous in memory for efficient access
        similarities_array = cupy.ascontiguousarray(similarities_array, dtype='float32')
        
        # Verify array length matches expected number of rows
        if len(similarities_array) != len(active_index):
            logger.error(f"Shape mismatch: got {len(similarities_array)} similarities for {len(active_index)} rows.")
            # Fall back to zeros for safety
            return result_series

        # Create a cuDF Series from the calculated similarities.
        # CRITICAL: Use the copied index to ensure the results align correctly 
        # with their original positions.
        similarities_series = cudf.Series(
            similarities_array,
            index=active_index
        )

        # CRITICAL: Force cuDF to own its own device buffer instead of borrowing
        # from CuPy's memory pool. This prevents potential use-after-free issues
        # when CuPy's memory pool is cleared.
        similarities_series = similarities_series.astype("float32", copy=True)

        # Place the calculated similarities back into the full result series.
        # The `Series.update()` method modifies `result_series` in-place.
        # It aligns on the index, overwriting the default zero scores with the
        # new values from `similarities_series`.
        result_series.update(similarities_series)

    except RuntimeError as e:
        logger.error(f"A runtime error occurred in 'calculate_similarity_gpu' during TF-IDF processing: {e}", exc_info=True)
        _log_series_samples(series_a_to_process, series_b_to_process, min_n, "Sample of valid rows at failure")
        if "Insufficient number of characters" in str(e):
            logger.warning(f"N-gram generation failed despite pre-checks. Returning calculated similarities so far.")
            # The result_series will contain zeros for the rows that failed, which is the desired outcome.
        else:
            # For other unexpected errors, re-raise the exception.
            raise
    
    finally:
        # --- 4. Memory Cleanup ---
        # Clean up all intermediate objects in reverse order of creation.
        # This ensures dependencies are freed in the correct order and helps
        # prevent memory fragmentation. The 'del' statements remove Python
        # references, allowing the GPU memory manager to reclaim the memory.
        
        # Clean up the similarity calculation intermediates
        if similarities_array is not None:
            del similarities_array
        
        # Clean up the TF-IDF vectors
        if vectors_b is not None:
            del vectors_b
        if vectors_a is not None:
            del vectors_a
        if vectors_b_proc is not None:
            del vectors_b_proc
        if vectors_a_proc is not None:
            del vectors_a_proc
        
        # Clean up the vectorizer (can be large with big vocabularies)
        if vectorizer is not None:
            del vectorizer
        
        # Clean up the combined series used for fitting
        if combined_series is not None:
            del combined_series

        # Clean up combined vectors from pruning
        if combined_proc_vectors is not None:
            del combined_proc_vectors
        
        # Clean up the cleaned series that are no longer needed
        del series_b_cleaned
        del series_a_cleaned

    return result_series

@gpu_memory_cleanup
def find_similar_pairs(
    string_series: cudf.Series,
    tfidf_params: SimilarityTfidfParams,
    nn_params: SimilarityNnParams,
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
    vectorizer = TfidfVectorizer(**tfidf_params.model_dump())
    tfidf_matrix = vectorizer.fit_transform(string_series)

    # Step 2: Build and fit the NearestNeighbors model.
    nn_model = NearestNeighbors(**nn_params.model_dump())
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

    # Return only the source and destination columns, which correspond to the
    # integer indices of the items in the original `string_series`.
    return matched_pairs[['source', 'destination']]

def calculate_embedding_similarity(
    vectors_a: cupy.ndarray,
    vectors_b: cupy.ndarray
) -> cupy.ndarray:
    """
    Calculates the row-wise cosine similarity between two dense embedding matrices.

    This function provides a highly efficient, GPU-accelerated method for comparing
    two sets of vectors. It is the modern replacement for on-the-fly TF-IDF or
    edit distance calculations and is designed to work directly with the output of
    the `EmbeddingOrchestrator`. It includes robust validation to ensure inputs
    are clean, finite, and correctly shaped before calculation.

    Core Assumption
    ---------------
    This function operates under the critical assumption that the input vectors
    (`vectors_a` and `vectors_b`) are already L2-normalized (i.e., each row vector
    has a length of 1.0). This is a guarantee provided by the `EmbeddingOrchestrator`'s
    pipeline.

    When vectors are L2-normalized, their cosine similarity is mathematically
    equivalent to their dot product. This function calculates the row-wise dot
    product, which is a simple and extremely fast element-wise multiplication
    followed by a sum.

    Parameters
    ----------
    vectors_a : cupy.ndarray
        A dense, 2D CuPy array of shape (N, D) containing L2-normalized embeddings.
    vectors_b : cupy.ndarray
        A second dense, 2D CuPy array of the exact same shape (N, D) containing
        L2-normalized embeddings to be compared row-wise with `vectors_a`.

    Returns
    -------
    cupy.ndarray
        A 1D, contiguous CuPy array of float32 similarity scores of length N. The
        value at index `i` is the cosine similarity between `vectors_a[i]` and `vectors_b[i]`.

    Raises
    ------
    ValueError
        If the input arrays are not 2D CuPy ndarrays, are empty, or do not have
        the exact same shape.
    """
    logger.debug(f"Calculating row-wise embedding similarity for matrices of shape {vectors_a.shape}.")

    # --- Input Validation ---
    if not isinstance(vectors_a, cupy.ndarray) or not isinstance(vectors_b, cupy.ndarray):
        raise ValueError("Inputs `vectors_a` and `vectors_b` must be CuPy ndarrays.")

    if vectors_a.ndim != 2 or vectors_b.ndim != 2:
        raise ValueError(
            f"Input vectors must be 2-dimensional. "
            f"Got dimensions: vectors_a={vectors_a.ndim}, vectors_b={vectors_b.ndim}."
        )

    if vectors_a.shape != vectors_b.shape:
        raise ValueError(
            f"Input vector shapes do not match: "
            f"vectors_a shape is {vectors_a.shape}, vectors_b shape is {vectors_b.shape}."
        )

    if vectors_a.size == 0:
        logger.warning("Input vectors are empty. Returning an empty similarity array.")
        return cupy.array([], dtype=cupy.float32)

    # --- Data Sanitization ---
    # Ensure that both matrices contain only finite values (no NaN or infinity).
    # This is a critical step to prevent silent corruption of results or CUDA errors.
    logger.debug("Ensuring input matrices contain finite values.")
    vectors_a = ensure_finite_matrix(vectors_a, replace_non_finite=True, copy=False)
    vectors_b = ensure_finite_matrix(vectors_b, replace_non_finite=True, copy=False)

    # --- Calculation ---
    # For L2-normalized vectors, the cosine similarity is the dot product.
    # The row-wise dot product is calculated by performing an element-wise
    # multiplication and then summing across the feature dimension (axis=1).
    # This is a single, highly optimized GPU operation.
    similarity_scores = (vectors_a * vectors_b).sum(axis=1)

    logger.debug(f"Similarity calculation complete. Mean similarity: {float(similarity_scores.mean()):.4f}")
    
    # --- Finalization ---
    # Ensure the output array is in a contiguous block of memory and has the
    # standard float32 dtype for consistency in downstream operations.
    final_scores = cupy.ascontiguousarray(similarity_scores, dtype=cupy.float32)

    return final_scores