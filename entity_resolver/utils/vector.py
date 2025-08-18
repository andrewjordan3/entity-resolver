# entity_resolver/utils/vector.py
"""
This module provides GPU-accelerated utilities for vector transformations
and linear algebra operations, primarily used for creating and manipulating
embeddings.
"""

import cupy
import cupyx.scipy.sparse as cpx_sparse
import logging
from typing import Dict, List, Tuple, Union, Optional

# Set up a logger for this module
logger = logging.getLogger(__name__)


def normalize_rows(
    matrix: Union[cupy.ndarray, cpx_sparse.spmatrix],
    *,
    epsilon: Optional[float] = None,
    copy: bool = True,
    return_stats: bool = False,
) -> Union[
    Union[cupy.ndarray, cpx_sparse.csr_matrix],
    Tuple[Union[cupy.ndarray, cpx_sparse.csr_matrix], bool, float]
]:
    """
    Applies row-wise L2 normalization to a CuPy dense array or sparse matrix.

    This function scales each row of the input matrix to have a unit L2 norm.
    Rows with a norm less than or equal to the specified epsilon are set to zero
    to prevent division by zero and handle near-empty rows gracefully.

    The implementation is optimized for both dense and sparse (specifically CSR)
    formats, avoiding unnecessary data conversion or densification.

    Args:
        matrix:
            The input matrix to normalize. Can be a 2D CuPy ndarray or any
            CuPy sparse matrix format (will be converted to CSR internally).
        epsilon:
            A small threshold to identify rows with a near-zero norm. Rows
            with a norm <= epsilon will be zeroed out. If None (default), a
            sensible value is chosen based on the matrix's dtype (1e-7 for
            float32, 1e-12 for float64).
        copy:
            If True (default), the function operates on a copy of the input
            matrix. If False, the normalization is performed in-place, which
            can save memory but will modify the original matrix. Note that
            for sparse matrices not in CSR format, a copy is always made.
        return_stats:
            If True, the function returns a tuple containing the normalized
            matrix, a boolean indicating if all non-zeroed rows are
            successfully normalized within tolerance, and the maximum deviation
            from unit norm. Defaults to False.

    Returns:
        If `return_stats` is False (default):
            - The row-normalized matrix, with the same type (or CSR for sparse)
              and shape as the input.
        If `return_stats` is True:
            - A tuple containing:
                - normalized_matrix: The row-normalized matrix.
                - is_normalized: A boolean that is True if the maximum
                  deviation from unit norm is within a reasonable tolerance.
                - max_deviation: The maximum absolute difference between the
                  norm of a processed row and 1.0.
                  
    Raises:
        TypeError:
            If the input `matrix` is not a CuPy ndarray or sparse matrix.
        ValueError:
            If the input `matrix` is a dense ndarray but is not 2-dimensional.
    """
    # --- Epsilon Handling ---
    # If no epsilon is provided, choose a sensible default based on the floating
    # point precision of the input matrix's data type.
    if epsilon is None:
        if matrix.dtype == cupy.float32:
            epsilon = 1e-7
        elif matrix.dtype == cupy.float64:
            epsilon = 1e-12
        else:
            # For other dtypes, fall back to a multiple of the machine epsilon.
            epsilon = 100 * cupy.finfo(matrix.dtype).eps

    # --- Dense Matrix Path ---
    if isinstance(matrix, cupy.ndarray):
        logger.debug(f"Normalizing dense matrix of shape {matrix.shape} with epsilon={epsilon:.1e}")
        if matrix.ndim != 2:
            raise ValueError(
                f"Input dense array must be 2-dimensional, but got shape {matrix.shape}"
            )
        
        # Operate on a copy or in-place based on the 'copy' flag.
        normalized_matrix = matrix.copy() if copy else matrix

        # Calculate the L2 norm for each row. `keepdims=True` ensures the
        # result has shape (n_samples, 1) for safe broadcasting.
        row_norms = cupy.linalg.norm(normalized_matrix, axis=1, keepdims=True)

        # Create a boolean mask to identify rows with norms greater than the epsilon.
        # These are the rows that will be scaled.
        valid_rows_mask = (row_norms > epsilon)

        n_valid = int(valid_rows_mask.sum())
        logger.debug(f"Dense path: {n_valid}/{matrix.shape[0]} rows will be normalized; "
                     f"{matrix.shape[0] - n_valid} will be zeroed.")

        # Use cupy.where for conditional scaling.
        # - If a row's norm is > epsilon, scale it by its norm (matrix / row_norms).
        # - Otherwise, set all elements in that row to 0.0.
        # This is an efficient way to avoid explicit loops and handle the zero-norm case.
        normalized_matrix = cupy.where(
            valid_rows_mask,
            normalized_matrix / row_norms,
            0.0
        )

        if return_stats:
            # Calculate the deviation from unit norm for the rows that were scaled.
            # For rows that were zeroed out, the deviation is not relevant.
            deviation = cupy.abs(row_norms[valid_rows_mask] - 1.0)
            max_deviation = float(deviation.max()) if deviation.size > 0 else 0.0
            
            # Determine a reasonable tolerance based on the matrix data type.
            dtype_tolerance = 1e-7 if matrix.dtype == cupy.float64 else 1e-5
            is_normalized = max_deviation <= dtype_tolerance
            
            return normalized_matrix, is_normalized, max_deviation
        
        return normalized_matrix

    # --- Sparse Matrix Path ---
    if cpx_sparse.isspmatrix(matrix):
        logger.debug(f"Normalizing sparse matrix of shape {matrix.shape} with epsilon={epsilon:.1e}")
        # Ensure the matrix is in CSR format for efficient row-wise operations.
        # `tocsr` is efficient: if it's already CSR and copy=False, no work is done.
        # If not CSR, a conversion (which is a copy) happens regardless of the flag.
        csr_matrix = matrix.tocsr(copy=copy)

        # Calculate the squared L2 norm for each row efficiently.
        # `csr_matrix.multiply(csr_matrix)` squares each non-zero element.
        # `.sum(axis=1)` then sums these squared values row by row.
        row_norms_squared = csr_matrix.multiply(csr_matrix).sum(axis=1)
        
        # Take the square root to get the actual L2 norms.
        # cupy.asarray converts the (n, 1) CuPy matrix to a (n, 1) array,
        # and .ravel() flattens it to a 1D array of shape (n,).
        row_norms = cupy.sqrt(cupy.asarray(row_norms_squared)).ravel()

        # Create a 1D array of scale factors for each row.
        # If a row's norm is > epsilon, its scale factor is 1.0 / norm.
        # Otherwise, its scale factor is 0.0, which will zero out the row.
        scale_factors = cupy.where(row_norms > epsilon, 1.0 / row_norms, 0.0)

        n_valid = int((row_norms > epsilon).sum())
        logger.debug(f"Sparse path: {n_valid}/{matrix.shape[0]} rows will be normalized; "
                     f"{matrix.shape[0] - n_valid} will be zeroed.")

        # This is the key to efficient row-wise scaling of a CSR matrix.
        # 1. `cupy.diff(csr_matrix.indptr)` calculates the number of non-zero
        #    elements in each row. `indptr` stores the cumulative count.
        row_element_counts = cupy.diff(csr_matrix.indptr)
        
        # 2. `cupy.repeat` expands the `scale_factors` array. Each row's scale
        #    factor is repeated N times, where N is the number of non-zero
        #    elements in that row. The result is an array of shape (nnz,).
        data_scale_factors = cupy.repeat(scale_factors, row_element_counts)
        
        # 3. Finally, scale the non-zero data elements directly. This applies
        #    the correct row-specific scale factor to each element without
        #    creating any large intermediate matrices.
        csr_matrix.data *= data_scale_factors
        
        # If a row was scaled by 0.0, its data elements are now zero.
        # `eliminate_zeros()` removes these explicit zeros from the sparse structure.
        csr_matrix.eliminate_zeros()

        if return_stats:
            # Only calculate deviation for rows that were actually scaled.
            valid_rows_mask = (row_norms > epsilon)
            deviation = cupy.abs(row_norms[valid_rows_mask] - 1.0)
            max_deviation = float(deviation.max()) if deviation.size > 0 else 0.0
            
            dtype_tolerance = 1e-7 if matrix.dtype == cupy.float64 else 1e-5
            is_normalized = max_deviation <= dtype_tolerance
            
            return csr_matrix, is_normalized, max_deviation
            
        return csr_matrix

    # --- Invalid Type ---
    raise TypeError(
        f"Input must be a cupy.ndarray or cupyx.scipy.sparse.spmatrix, "
        f"but got {type(matrix)}"
    )

def center_kernel_matrix(kernel_matrix: cupy.ndarray) -> cupy.ndarray:
    """
    Applies double-centering to a symmetric kernel (Gram) matrix.

    This is a critical step in Kernel PCA, transforming the kernel matrix to
    represent dot products of feature vectors centered at the origin in the
    high-dimensional feature space. The formula is:
    K_centered = K - K_row_means - K_col_means + K_grand_mean

    Args:
        kernel_matrix: A square, symmetric CuPy array.

    Returns:
        The double-centered kernel matrix.
    """
    row_means = kernel_matrix.mean(axis=1, keepdims=True)
    col_means = kernel_matrix.mean(axis=0, keepdims=True)
    grand_mean = kernel_matrix.mean()

    centered_kernel = kernel_matrix - row_means - col_means + grand_mean

    # Enforce perfect symmetry to counteract potential floating-point drift.
    return (centered_kernel + centered_kernel.T) * 0.5


def center_kernel_vector(
    kernel_vector: cupy.ndarray,
    training_kernel_row_means: cupy.ndarray,
    training_kernel_grand_mean: float
) -> cupy.ndarray:
    """
    Centers a new kernel vector against the statistics of the training kernel.

    This projects new data points into the centered feature space defined by
    the original training data. The formula is:
    k_centered = k - training_row_means - k_col_means + training_grand_mean

    Args:
        kernel_vector: A (batch_size, n_samples) array of similarities
                       between new points and the original anchor points.
        training_kernel_row_means: Row means from the original training kernel matrix.
        training_kernel_grand_mean: Grand mean from the original training kernel.

    Returns:
        The centered kernel vector.
    """
    if kernel_vector.ndim == 1:
        kernel_vector = kernel_vector[None, :]

    new_vector_row_means = kernel_vector.mean(axis=1, keepdims=True)

    # Apply the centering formula using pre-computed training statistics.
    # training_kernel_row_means.T broadcasts across the columns of kernel_vector.
    return (
        kernel_vector
        - training_kernel_row_means.T
        - new_vector_row_means
        + training_kernel_grand_mean
    )


def get_top_k_positive_eigenpairs(
    symmetric_matrix: cupy.ndarray,
    k: int
) -> Tuple[cupy.ndarray, cupy.ndarray]:
    """
    Calculates the top-k positive eigenpairs for a symmetric matrix.

    This function performs a full eigendecomposition, sorts the results,
    filters out non-positive eigenvalues, and returns exactly k eigenpairs,
    padding with zeros if necessary.

    Args:
        symmetric_matrix: The square, symmetric CuPy array to decompose.
        k: The desired number of eigenpairs to return.

    Returns:
        A tuple containing (eigenvectors, eigenvalues).
    """
    if k > symmetric_matrix.shape[0]:
        logger.warning(
            f"Requested k={k} exceeds matrix size n={symmetric_matrix.shape[0]}; "
            "trailing dimensions will be zero-padded."
        )

    # cupy.linalg.eigh is optimized for Hermitian (symmetric real) matrices.
    eigenvalues, eigenvectors = cupy.linalg.eigh(symmetric_matrix.astype(cupy.float32))

    # Sort eigenvalues and corresponding eigenvectors in descending order.
    descending_indices = cupy.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[descending_indices]
    eigenvectors = eigenvectors[:, descending_indices]

    # Filter for numerically significant, positive eigenvalues.
    positive_mask = eigenvalues > 1e-6
    if not cupy.any(positive_mask):
        logger.warning("No positive eigenvalues found. Returning zeros.")
        return cupy.zeros((symmetric_matrix.shape[0], k)), cupy.zeros((k,))

    positive_eigenvalues = eigenvalues[positive_mask]
    positive_eigenvectors = eigenvectors[:, positive_mask]

    # Ensure exactly k components are returned by padding or trimming.
    num_positive = positive_eigenvectors.shape[1]
    if num_positive >= k:
        return positive_eigenvectors[:, :k], positive_eigenvalues[:k]
    else:
        pad_width = k - num_positive
        logger.debug(f"Found {num_positive} positive components; padding with {pad_width} zeros.")
        padded_vectors = cupy.pad(positive_eigenvectors, ((0, 0), (0, pad_width)))
        padded_values = cupy.pad(positive_eigenvalues, (0, pad_width))
        return padded_vectors, padded_values


def balance_feature_streams(
    vector_streams: Dict[str, cupy.ndarray],
    proportions: Dict[str, float]
) -> List[cupy.ndarray]:
    """
    Scales L2-normalized vector streams to their target energy proportions.

    This method assumes each input vector stream has already been row-wise
    L2 normalized (i.e., each vector has a magnitude of 1). It then scales
    each stream by the square root of its desired proportion.

    Args:
        vector_streams: Dictionary of stream names to pre-normalized CuPy arrays.
        proportions: Dictionary mapping stream names to desired energy proportions.

    Returns:
        A list of the balanced (rescaled) CuPy arrays.
    """
    balanced_vectors_list = []
    logging_dict = {}
    for name, vectors in vector_streams.items():
        proportion = proportions.get(name, 0.0)
        scaling_factor = cupy.sqrt(proportion)
        balanced_vectors_list.append(vectors * scaling_factor)
        logging_dict[name] = scaling_factor
    logger.debug(f"Balancing streams with scaling factors: {', '.join(f'{stream}: {scale:.4f}' for stream, scale in logging_dict.items())}")
        
    return balanced_vectors_list


# --- Consensus Embedding Helper Functions ---

def _compute_average_gram_matrix(
    normalized_embeddings: List[cupy.ndarray],
    sample_indices: cupy.ndarray
) -> cupy.ndarray:
    """Computes the average cosine similarity matrix over an ensemble."""
    n_samples = len(sample_indices)
    n_runs = len(normalized_embeddings)
    average_gram = cupy.zeros((n_samples, n_samples), dtype=cupy.float32)

    for embedding_run in normalized_embeddings:
        sample_vectors = embedding_run[sample_indices]
        average_gram += sample_vectors @ sample_vectors.T
    return average_gram / float(n_runs)


def _project_out_of_sample_batch(
    batch_indices: cupy.ndarray,
    normalized_embeddings: List[cupy.ndarray],
    sample_blocks: List[cupy.ndarray],
    kpca_params: Dict
) -> cupy.ndarray:
    """Projects a batch of new points into the consensus space."""
    n_samples = sample_blocks[0].shape[0]
    n_runs = len(normalized_embeddings)
    avg_kernel_vector = cupy.zeros((len(batch_indices), n_samples), dtype=cupy.float32)

    for embedding_run, sample_block in zip(normalized_embeddings, sample_blocks):
        batch_vectors = embedding_run[batch_indices]
        avg_kernel_vector += batch_vectors @ sample_block.T
    avg_kernel_vector /= float(n_runs)

    centered_k_vec = center_kernel_vector(
        avg_kernel_vector,
        kpca_params['row_means'],
        kpca_params['grand_mean']
    )
    return (centered_k_vec @ kpca_params['eigenvectors']) * kpca_params['inv_sqrt_eigenvalues']


def create_consensus_embedding(
    embeddings_list: List[cupy.ndarray],
    n_samples: int,
    batch_size: int,
    random_state: int
) -> cupy.ndarray:
    """
    Builds a consensus embedding using Kernel PCA on an average Gram matrix.

    This method provides a scale and orientation-invariant way to combine
    embeddings from an ensemble. It samples a subset of points to build an
    "anchor" model, performs Kernel PCA, and then projects all points into
    the resulting consensus space.

    Args:
        embeddings_list: A list of (n_points, n_dims) CuPy arrays.
        n_samples: The number of points to use for the anchor model.
        batch_size: The batch size for processing out-of-sample points.
        random_state: Seed for reproducible sampling.

    Returns:
        A final (n_points, n_dims) consensus embedding, row-normalized.
    """
    if not embeddings_list:
        raise ValueError("Cannot create consensus from an empty list of embeddings.")

    n_points, n_dims = embeddings_list[0].shape
    logger.info(f"Creating consensus from {len(embeddings_list)} embeddings for {n_points} points.")

    normalized_embeddings = [normalize_rows(E.astype(cupy.float32)) for E in embeddings_list]

    cupy.random.seed(random_state)
    use_sampling = n_points > n_samples
    sample_indices = cupy.sort(cupy.random.choice(n_points, n_samples, replace=False)) if use_sampling else cupy.arange(n_points)

    avg_gram = _compute_average_gram_matrix(normalized_embeddings, sample_indices)
    centered_gram = center_kernel_matrix(avg_gram)
    eigenvectors, eigenvalues = get_top_k_positive_eigenpairs(centered_gram, k=n_dims)

    sqrt_eigenvalues = cupy.sqrt(eigenvalues)
    anchor_embedding = normalize_rows(eigenvectors * sqrt_eigenvalues)

    if not use_sampling:
        return anchor_embedding

    logger.info("Extending consensus embedding to full dataset via kernel projection...")
    final_embedding = cupy.zeros((n_points, n_dims), dtype=cupy.float32)
    final_embedding[sample_indices] = anchor_embedding

    kpca_params = {
        'row_means': avg_gram.mean(axis=1, keepdims=True),
        'grand_mean': float(avg_gram.mean()),
        'eigenvectors': eigenvectors,
        'inv_sqrt_eigenvalues': 1.0 / (sqrt_eigenvalues + 1e-8)
    }
    sample_blocks = [E[sample_indices] for E in normalized_embeddings]
    remaining_indices = cupy.setdiff1d(cupy.arange(n_points), sample_indices)

    for i in range(0, len(remaining_indices), batch_size):
        batch_indices = remaining_indices[i:i + batch_size]
        batch_embedding = _project_out_of_sample_batch(batch_indices, normalized_embeddings, sample_blocks, kpca_params)
        final_embedding[batch_indices] = normalize_rows(batch_embedding)

    logger.info("Consensus extension complete.")
    return final_embedding

def create_initial_vector(
    vector_size: int,
    dtype: cupy.dtype,
    *,
    seed: Optional[int] = None,
) -> cupy.ndarray:
    """
    Creates a well-conditioned, normalized random vector for iterative solvers.

    This function generates a random vector and processes it to make it a
    suitable starting point (v0) for algorithms like Lanczos iteration (used
    in `eigsh`). The key steps are:
    1. Generate a vector from a standard normal distribution.
    2. Orthogonalize it against the constant vector (all ones) to remove any
       component in that direction, which can be a trivial eigenvector.
    3. Normalize the final vector to have a unit L2 norm.

    Args:
        vector_size:
            The desired number of elements in the vector.
        dtype:
            The CuPy data type for the vector (e.g., cupy.float32, cupy.float64).
        seed:
            An optional integer seed for the random number generator to ensure
            reproducibility. If None, the generator will be initialized with
            a random seed.

    Returns:
        A 1D CuPy array of the specified size and dtype with a unit L2 norm,
        suitable for use as an initial vector in a solver.
        
    Raises:
        ValueError: If vector_size is not a positive integer.
    """
    if not isinstance(vector_size, int) or vector_size <= 0:
        raise ValueError(f"vector_size must be a positive integer, but got {vector_size}")

    logger.debug(f"Creating initial vector of size {vector_size} with dtype={dtype} and seed={seed}")

    # 1. Initialize a random state for reproducibility
    random_state = cupy.random.RandomState(seed)

    # 2. Generate a vector of random numbers from a standard normal distribution
    initial_vector = random_state.standard_normal(vector_size, dtype=dtype)

    # 3. Orthogonalize against the all-ones vector to avoid trivial directions
    #    This is a common technique to improve the stability of Lanczos-based solvers.
    if vector_size > 1:
        # Create a normalized vector of all ones
        ones_vector = cupy.ones(vector_size, dtype=dtype) / cupy.sqrt(vector_size)
        
        # Calculate the projection of the random vector onto the ones vector
        projection = (initial_vector @ ones_vector)
        
        # Subtract the projection to make the initial_vector orthogonal to the ones_vector
        initial_vector -= projection * ones_vector

    # 4. Normalize the final vector to have a unit L2 norm
    norm = cupy.linalg.norm(initial_vector)

    if norm > 0:
        initial_vector /= norm
    else:
        # This case is extremely rare but can happen if the initial vector was
        # exactly zero or perfectly aligned with the ones_vector.
        # We re-initialize to a safe, non-zero vector.
        logger.warning("Initial vector had zero norm after orthogonalization. Re-initializing.")
        initial_vector = random_state.standard_normal(vector_size, dtype=dtype)
        norm = cupy.linalg.norm(initial_vector)
        if norm > 0:
            initial_vector /= norm
        else: # If still zero, create a deterministic vector
            initial_vector[0] = 1.0

    return initial_vector
