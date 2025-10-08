# entity_resolver/utils/vector.py
"""
This module provides GPU-accelerated utilities for vector transformations
and linear algebra operations, primarily used for creating and manipulating
embeddings.
"""

import cupy
import cupyx.scipy.sparse as cpx_sparse
import numpy as np
import logging
from typing import Dict, List, Tuple, Union, Optional, Literal, Any

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

        # Apply the scaling to the matrix data using the most efficient GPU-native method.
        nnz = csr_matrix.data.size
        
        # Find the row index for every non-zero element. `indptr` is a sorted
        # array marking where each row's data begins. `searchsorted` uses this
        # to efficiently map every element in `.data` back to its original row.
        row_idx = cupy.searchsorted(csr_matrix.indptr, cupy.arange(nnz), side="right") - 1

        # Use fancy indexing to expand `scale_factors` to match the `.data` array,
        # then perform a direct element-wise multiplication.
        csr_matrix.data *= scale_factors[row_idx]
        
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

def center_kernel_matrix(
    kernel_matrix: cupy.ndarray,
    verify_symmetry: bool = True,
    symmetry_tolerance: float = 1e-6,
    enforce_symmetry: bool = True,
    return_centering_params: bool = False
) -> Union[cupy.ndarray, Tuple[cupy.ndarray, Dict[str, Any]]]:
    """
    Applies double-centering transformation to a kernel (Gram) matrix.

    Double-centering is essential for Kernel PCA and other kernel methods that require
    centered data in the implicit feature space. This transformation ensures that the
    feature vectors (which we never explicitly compute) have zero mean in the
    high-dimensional feature space induced by the kernel.

    Mathematical Foundation:
    Given kernel matrix K where K[i,j] = φ(x_i)·φ(x_j) for some feature map φ,
    the centered kernel K_c corresponds to centered features φ_c(x) = φ(x) - μ_φ,
    where μ_φ is the mean of all φ(x_i).

    The centering formula is:
    K_c = K - 1_n @ K - K @ 1_n + 1_n @ K @ 1_n
    where 1_n is the (1/n) * ones(n,n) matrix.

    Simplified implementation:
    K_c = K - row_means - col_means + grand_mean

    Args:
        kernel_matrix: Square symmetric matrix of shape (n, n) containing kernel values.
        verify_symmetry: If True, checks that input matrix is symmetric.
        symmetry_tolerance: Maximum allowed deviation from symmetry when verifying.
        enforce_symmetry: If True, forces output to be exactly symmetric by averaging
                         with its transpose. Recommended to avoid numerical drift.
        return_centering_params: If True, returns a tuple of (centered_kernel, centering_params)
                                where centering_params contains the means needed to center
                                new test data.

    Returns:
        If return_centering_params is False:
            centered_kernel: Double-centered kernel matrix of shape (n, n).
        If return_centering_params is True:
            Tuple of (centered_kernel, centering_params) where centering_params is a dict.

    Raises:
        ValueError: If matrix is not square, not symmetric, or contains invalid values.
        TypeError: If input is not a CuPy ndarray.
    """
    # Step 1: Validate input type and shape
    if not isinstance(kernel_matrix, cupy.ndarray):
        raise TypeError(
            f"kernel_matrix must be a CuPy ndarray, got {type(kernel_matrix).__name__}"
        )
    if kernel_matrix.ndim != 2:
        raise ValueError(
            f"Kernel matrix must be 2-dimensional, got shape {kernel_matrix.shape}"
        )
    n_rows, n_cols = kernel_matrix.shape
    if n_rows != n_cols:
        raise ValueError(
            f"Kernel matrix must be square. Got shape ({n_rows}, {n_cols})"
        )
    if n_rows == 0:
        raise ValueError("Cannot center an empty matrix")

    # Preserve the original dtype for computations, promoting if necessary
    working_dtype = kernel_matrix.dtype
    if working_dtype not in [cupy.float32, cupy.float64]:
        logger.warning(
            f"Converting kernel matrix from {working_dtype} to float64 for numerical stability"
        )
        working_dtype = cupy.float64
        kernel_matrix = kernel_matrix.astype(working_dtype)

    # Step 2: Verify symmetry if requested
    if verify_symmetry:
        asymmetry_matrix = kernel_matrix - kernel_matrix.T
        symmetry_error = float(cupy.max(cupy.abs(asymmetry_matrix)))
        matrix_scale = float(cupy.max(cupy.abs(kernel_matrix)))
        adaptive_tolerance = max(symmetry_tolerance, symmetry_tolerance * matrix_scale)

        if symmetry_error > adaptive_tolerance:
            relative_error = symmetry_error / max(1e-10, matrix_scale)
            if relative_error < 1e-3:  # Less than 0.1% relative error
                logger.warning(
                    f"Kernel matrix has small asymmetry (absolute: {symmetry_error:.2e}, "
                    f"relative: {relative_error:.2e}). Symmetrizing before centering."
                )
                kernel_matrix = (kernel_matrix + kernel_matrix.T) * 0.5
            else:
                raise ValueError(
                    f"Kernel matrix is not symmetric. Maximum asymmetry: {symmetry_error:.2e} "
                    f"(relative: {relative_error:.2e}, tolerance: {adaptive_tolerance:.2e})."
                )
    
    # Step 3: Check for NaN or Inf values
    if cupy.any(cupy.isnan(kernel_matrix)):
        raise ValueError("Kernel matrix contains NaN values")
    if cupy.any(cupy.isinf(kernel_matrix)):
        raise ValueError("Kernel matrix contains infinite values")

    # Step 4: Check for potentially problematic constant-like Gram matrices
    # This detects a specific case where a cosine similarity matrix has collapsed,
    # leading to a near-zero centered matrix, which is usually not intended.
    diag_is_one = bool(cupy.allclose(cupy.diag(kernel_matrix), 1.0, rtol=1e-3, atol=1e-3))
    if n_rows > 1 and diag_is_one:
        offdiag_mask = ~cupy.eye(n_rows, dtype=bool)
        scale = float(cupy.max(cupy.abs(kernel_matrix)))
        # Use float64 for std dev calculation for better precision
        offdiag_std = float(cupy.std(kernel_matrix[offdiag_mask], dtype=cupy.float64))

        # Check if the standard deviation of off-diagonal elements is negligible
        if offdiag_std <= (1e-6 * (scale if scale > 0.0 else 1.0)):
            logger.warning(
                "Kernel matrix resembles a cosine Gram matrix with nearly constant "
                "off-diagonals and a diagonal of 1s. Centering this matrix will "
                "result in a near-zero matrix. This may indicate collapsed embeddings "
                "or poorly chosen kernel parameters."
            )

    # Step 5: Compute means using high-precision float64 for accuracy
    row_means = kernel_matrix.mean(axis=1, keepdims=True, dtype=cupy.float64)
    col_means = kernel_matrix.mean(axis=0, keepdims=True, dtype=cupy.float64)
    grand_mean = float(kernel_matrix.mean(dtype=cupy.float64))

    logger.debug(
        f"Centering statistics: grand_mean={grand_mean:.6f}, "
        f"row_mean_range=[{float(row_means.min()):.6f}, {float(row_means.max()):.6f}]"
    )

    # Step 6: Apply the double-centering formula
    centered_kernel = (
        kernel_matrix
        - row_means.astype(working_dtype)
        - col_means.astype(working_dtype)
        + working_dtype(grand_mean)
    )

    # Step 7: Enforce perfect symmetry if requested
    if enforce_symmetry:
        centered_kernel = (centered_kernel + centered_kernel.T) * 0.5
        logger.debug("Symmetry enforced via averaging with transpose")

    # Step 8: Verify centering was successful (for debugging)
    if logger.isEnabledFor(logging.DEBUG):
        max_row_mean = float(cupy.abs(centered_kernel.mean(axis=1)).max())
        max_col_mean = float(cupy.abs(centered_kernel.mean(axis=0)).max())
        verification_tolerance = (1e-5 if working_dtype == cupy.float32 else 1e-10) * max(1.0, float(cupy.abs(kernel_matrix).max()))
        if max_row_mean > verification_tolerance or max_col_mean > verification_tolerance:
            logger.warning(
                f"Centered kernel has larger than expected mean values "
                f"(max_row: {max_row_mean:.2e}, max_col: {max_col_mean:.2e}, "
                f"tolerance: {verification_tolerance:.2e})."
            )

    # Step 9: Return results
    if return_centering_params:
        centering_params = {
            'row_means': row_means.squeeze(),
            'grand_mean': grand_mean,
            'n_samples': n_rows
        }
        return centered_kernel, centering_params
    else:
        return centered_kernel


def center_kernel_vector(
    kernel_vector: cupy.ndarray,
    training_kernel_row_means: cupy.ndarray,
    training_kernel_grand_mean: float,
    validate_dimensions: bool = True
) -> cupy.ndarray:
    """
    Centers kernel vectors for new/test samples using training kernel statistics.
    
    This function is essential for projecting new data points into the centered
    feature space established during training in kernel methods (especially Kernel PCA).
    It ensures that transformations applied to new samples are consistent with the
    centering applied to the training data.
    
    Mathematical Foundation:
    For new samples x_new, we need to compute centered kernel values with training samples.
    If K_train was centered using statistics μ_row and μ_grand, then for consistency:
    
    k_centered[i,j] = k[i,j] - μ_row[j] - mean(k[i,:]) + μ_grand
    
    Where:
    - k[i,j] = kernel(x_new[i], x_train[j])
    - μ_row[j] = mean over training samples of kernel with x_train[j]
    - mean(k[i,:]) = mean kernel value between x_new[i] and all training samples
    - μ_grand = grand mean of training kernel matrix
    
    This ensures the implicit feature vectors φ(x_new) are centered with respect
    to the same origin as the training features.
    
    Use Cases:
    - Transforming test data in Kernel PCA
    - Out-of-sample extension in kernel methods
    - Online/incremental kernel learning
    - Cross-validation in kernel methods
    
    Args:
        kernel_vector: Kernel values between new and training samples.
                      Shape: (batch_size, n_train_samples) for batched input
                      OR (n_train_samples,) for single sample.
                      Element [i,j] = kernel(x_new[i], x_train[j]).
        training_kernel_row_means: Pre-computed row means from centered training kernel.
                                  Shape: (n_train_samples,) or (n_train_samples, 1).
                                  Element [j] = mean(K_train[j, :]).
        training_kernel_grand_mean: Pre-computed grand mean of training kernel matrix.
                                   Scalar value = mean(K_train).
        validate_dimensions: If True, performs dimension compatibility checks.
                           Set to False only in performance-critical loops where
                           dimensions are guaranteed correct.
    
    Returns:
        centered_kernel_vector: Centered kernel values of shape (batch_size, n_train_samples).
                              If input was 1D, output maintains batch dimension of 1.
                              Properties:
                              - Consistent with training kernel centering
                              - Row sums ≈ 0 (equivalently, n*grand_mean - Σ row_means)
    
    Raises:
        ValueError: If dimensions are incompatible or inputs are invalid.
        TypeError: If inputs are not of expected types.
    
    Example:
        >>> # Training phase
        >>> X_train = cupy.random.randn(100, 20)  # 100 training samples
        >>> K_train = compute_rbf_kernel(X_train, X_train)
        >>> K_train_centered = center_kernel_matrix(K_train)
        >>> 
        >>> # Save statistics for test phase
        >>> train_row_means = K_train.mean(axis=1)
        >>> train_grand_mean = K_train.mean()
        >>> 
        >>> # Test phase - single sample
        >>> x_test = cupy.random.randn(20)  # Single test sample
        >>> k_test = compute_rbf_kernel(x_test.reshape(1, -1), X_train).ravel()
        >>> k_test_centered = center_kernel_vector(k_test, train_row_means, train_grand_mean)
        >>> 
        >>> # Test phase - batch
        >>> X_test = cupy.random.randn(10, 20)  # 10 test samples
        >>> K_test = compute_rbf_kernel(X_test, X_train)
        >>> K_test_centered = center_kernel_vector(K_test, train_row_means, train_grand_mean)
    
    Implementation Notes:
        - Handles both single samples and batches transparently
        - Broadcasting is used for efficient computation
        - Original input dimension (1D vs 2D) is preserved in shape
    """
    
    # Step 1: Type validation
    if not isinstance(kernel_vector, cupy.ndarray):
        raise TypeError(
            f"kernel_vector must be a CuPy ndarray, got {type(kernel_vector).__name__}"
        )
    
    if not isinstance(training_kernel_row_means, cupy.ndarray):
        raise TypeError(
            f"training_kernel_row_means must be a CuPy ndarray, "
            f"got {type(training_kernel_row_means).__name__}"
        )
    
    # Convert grand mean to float if necessary
    try:
        training_kernel_grand_mean = float(training_kernel_grand_mean)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"training_kernel_grand_mean must be convertible to float: {e}"
        )
    
    # Step 2: Handle input dimensionality
    # Store original dimension to maintain API consistency
    input_was_1d = (kernel_vector.ndim == 1)
    
    if input_was_1d:
        # Convert 1D input to 2D with batch size of 1
        # Shape: (n_train_samples,) -> (1, n_train_samples)
        kernel_vector = kernel_vector[None, :]
        logger.debug("Converted 1D input to 2D with batch_size=1")
    
    elif kernel_vector.ndim != 2:
        raise ValueError(
            f"kernel_vector must be 1D or 2D, got shape {kernel_vector.shape}"
        )
    
    # Extract dimensions for clarity
    batch_size, n_train_samples = kernel_vector.shape
    
    # Step 3: Prepare training statistics for broadcasting
    # Ensure training_kernel_row_means is 1D for consistent handling
    if training_kernel_row_means.ndim == 2:
        if training_kernel_row_means.shape[1] != 1:
            raise ValueError(
                f"training_kernel_row_means must be 1D or column vector, "
                f"got shape {training_kernel_row_means.shape}"
            )
        training_kernel_row_means = training_kernel_row_means.ravel()
    elif training_kernel_row_means.ndim != 1:
        raise ValueError(
            f"training_kernel_row_means must be 1D or 2D column vector, "
            f"got {training_kernel_row_means.ndim}D array"
        )
    
    # Step 4: Validate dimension compatibility
    if validate_dimensions:
        if len(training_kernel_row_means) != n_train_samples:
            raise ValueError(
                f"Dimension mismatch: kernel_vector has {n_train_samples} training samples, "
                f"but training_kernel_row_means has {len(training_kernel_row_means)} elements. "
                "These must match."
            )
        
        # Sanity check on the grand mean value
        if not (-1e10 < training_kernel_grand_mean < 1e10):
            logger.warning(
                f"Unusual training_kernel_grand_mean value: {training_kernel_grand_mean}. "
                "This might indicate an error in computing training statistics."
            )
    
    # Step 5: Compute row means for the new kernel vectors
    # This represents the average kernel value between each new sample and all training samples
    # Shape: (batch_size, 1)
    new_vector_row_means = kernel_vector.mean(axis=1, keepdims=True)
    
    # Step 6: Apply the centering formula
    # Formula: k_centered = k - training_row_means - new_row_means + training_grand_mean
    # 
    # Detailed breakdown:
    # - kernel_vector: Shape (batch_size, n_train_samples)
    # - training_kernel_row_means: Shape (n_train_samples,), broadcasts to (1, n_train_samples)
    #   then to (batch_size, n_train_samples)
    # - new_vector_row_means: Shape (batch_size, 1), broadcasts to (batch_size, n_train_samples)
    # - training_kernel_grand_mean: Scalar, broadcasts to all elements
    
    centered_kernel_vector = (
        kernel_vector
        - training_kernel_row_means[None, :]  # Subtract training row means (per column)
        - new_vector_row_means                 # Subtract new sample means (per row)
        + training_kernel_grand_mean           # Add back the grand mean
    )
    
    # Step 7: Log statistics for debugging
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"Centering statistics: "
            f"batch_size={batch_size}, "
            f"n_train_samples={n_train_samples}, "
            f"new_row_means_range=[{float(new_vector_row_means.min()):.4f}, "
            f"{float(new_vector_row_means.max()):.4f}], "
            f"output_range=[{float(centered_kernel_vector.min()):.4f}, "
            f"{float(centered_kernel_vector.max()):.4f}]"
        )
        
        # This check verifies the mathematical integrity of the centering process,
        # accounting for the nuances of floating-point arithmetic.

        # Calculate the actual sum of each row in the centered matrix.
        row_sums = centered_kernel_vector.sum(axis=1)

        # In exact arithmetic, the sum of each centered row should equal this value,
        # assuming the training statistics were computed from the UN-centered training kernel.
        expected_sum = (n_train_samples * training_kernel_grand_mean -
                        training_kernel_row_means.sum())

        # The residual is the difference between the actual and theoretical sums.
        # This value should be very close to zero.
        residual = row_sums - expected_sum

        # --- Data-Aware Tolerance Calculation ---
        # Instead of a fixed tolerance, we compute a dynamic tolerance based on
        # the properties of the input data, which is a more robust approach.

        # Get the machine epsilon for the data's precision (e.g., float32 or float64).
        eps = cupy.finfo(centered_kernel_vector.dtype).eps

        # The potential for floating-point error accumulation is proportional to the
        # sum of the absolute values of the numbers involved (L1 norm). We use the
        # uncentered input vector for this scaling factor.
        row_l1_norm = cupy.abs(kernel_vector).sum(axis=1)

        # A safety constant, typically between 16 and 32 in numerical analysis,
        # to provide a safe margin for the error estimation.
        safety_constant = 16.0

        # The per-row tolerance scales with the data type's precision and the row's magnitude.
        per_row_tolerance = safety_constant * eps * row_l1_norm

        # For rows with very small or zero values, the relative tolerance can become
        # meaninglessly small. We enforce a small absolute floor to prevent this.
        atol_floor = 1e-5 if centered_kernel_vector.dtype == cupy.float32 else 1e-8
        per_row_tolerance = cupy.maximum(per_row_tolerance, atol_floor)

        # Find the single largest deviation and the largest allowed tolerance in the batch.
        max_deviation = float(cupy.max(cupy.abs(residual)))
        max_allowed_tolerance = float(cupy.max(per_row_tolerance))

        # Check if the largest observed deviation exceeds the largest allowed tolerance.
        if max_deviation > max_allowed_tolerance:
            # This log message provides clear, actionable information for debugging.
            logger.warning(
                "Large deviation in centered kernel row sum. "
                f"Max Deviation: {max_deviation:.6f}, "
                f"Expected Sum: {float(expected_sum):.6f}, "
                f"Max Allowed Tolerance: {max_allowed_tolerance:.6f}. "
                "Check that training statistics come from the UNcentered training kernel."
            )
    
    # Step 8: Return with original dimensionality
    if input_was_1d:
        # Convert back to 1D if input was 1D
        # Shape: (1, n_train_samples) -> (n_train_samples,)
        centered_kernel_vector = centered_kernel_vector.ravel()
        logger.debug("Converted output back to 1D to match input dimensionality")
    
    return centered_kernel_vector


def get_top_k_positive_eigenpairs(
    symmetric_matrix: cupy.ndarray,
    k: int,
    eigenvalue_threshold: float = 1e-8,
    verify_symmetry: bool = True
) -> Tuple[cupy.ndarray, cupy.ndarray]:
    """
    Extracts the top-k eigenpairs with positive eigenvalues from a symmetric matrix.
    
    This function performs full eigendecomposition using the optimized routine for
    Hermitian matrices, then filters and sorts to return exactly k eigenpairs.
    Zero-padding is applied if fewer than k positive eigenvalues exist.
    
    Mathematical Background:
    - For symmetric matrix A: A = V * Λ * V^T, where V contains eigenvectors and Λ contains eigenvalues
    - Positive eigenvalues indicate positive semi-definite components
    - Top eigenvalues capture the most variance/energy in the matrix
    - Eigenvectors form an orthonormal basis for the eigenspace

    Notes:
      - Uses cupy.linalg.eigh (Hermitian solver). Eigenvalues are returned in ascending order,
        so we flip once (no redundant argsort).
      - If verify_symmetry=True and asymmetry is detected but small, the matrix is symmetrized 
        as (A + A^T)/2 to improve numerical stability
      - Padded columns in eigenvectors are zero columns (not orthonormal by definition).
    
    Use Cases:
    - Principal Component Analysis (PCA)
    - Spectral clustering
    - Dimensionality reduction
    - Covariance matrix analysis
    - Kernel methods and spectral embeddings
    
    Args:
        symmetric_matrix: Square symmetric matrix of shape (n, n). 
                         Must be real-valued and symmetric (A = A^T).
        k: Number of eigenpairs to return. Can exceed matrix dimension 
           (will be zero-padded).
        eigenvalue_threshold: Minimum eigenvalue to consider as "positive".
                             Values below this are treated as numerical zeros.
                             Default: 1e-8 (suitable for float32/float64 precision).
        verify_symmetry: If True, verifies matrix symmetry before decomposition.
                        Set to False if you're certain the matrix is symmetric
                        to save computation time.
    
    Returns:
        Tuple of (eigenvectors, eigenvalues):
        - eigenvectors: Matrix of shape (n, k) where each column is an eigenvector.
                       Eigenvectors are orthonormal (unit length, orthogonal).
        - eigenvalues: Array of shape (k,) containing corresponding eigenvalues
                      in descending order.
        
        Note: If fewer than k positive eigenvalues exist, the remaining columns/values
              are zero-padded.
    
    Raises:
        ValueError: If matrix is not square, not symmetric (when verify_symmetry=True),
                   or if k < 1.
        TypeError: If input is not a CuPy ndarray.
        
    Example:
        >>> # Create a positive semi-definite matrix
        >>> A = cupy.random.randn(100, 20)
        >>> gram_matrix = A @ A.T  # Guaranteed positive semi-definite
        >>> eigvecs, eigvals = get_top_k_positive_eigenpairs(gram_matrix, k=10)
        >>> print(f"Top eigenvalue: {eigvals[0]:.4f}")
        >>> print(f"Eigenvectors shape: {eigvecs.shape}")  # (100, 10)
        
    Performance Notes:
        - cupy.linalg.eigh is optimized for Hermitian/symmetric matrices (O(n^3))
        - For very large matrices (n > 5000), consider iterative methods if only
          a few eigenpairs are needed (e.g., cupy.sparse.linalg.eigsh)
        - Memory usage: O(n^2) for the eigenvector matrix
    """
    
    # Step 1: Input validation - catch errors early
    if not isinstance(symmetric_matrix, cupy.ndarray):
        raise TypeError("symmetric_matrix must be a CuPy ndarray")
    
    if symmetric_matrix.ndim != 2:
        raise ValueError(f"Matrix must be 2D, got shape {symmetric_matrix.shape}")
    
    n_rows, n_cols = symmetric_matrix.shape
    if n_rows != n_cols:
        raise ValueError(f"Matrix must be square, got shape ({n_rows}, {n_cols})")
    
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    
    # Determine the working dtype (preserve original precision)
    working_dtype = symmetric_matrix.dtype
    if working_dtype not in [cupy.float32, cupy.float64]:
        # If not already float, convert to float64 for better precision
        working_dtype = cupy.float64
        symmetric_matrix = symmetric_matrix.astype(working_dtype)
    
    # Step 2: Verify and potentially fix symmetry if requested
    if verify_symmetry:
        # Check if A ≈ A^T within numerical tolerance
        asymmetry_matrix = symmetric_matrix - symmetric_matrix.T
        max_asymmetry = float(cupy.max(cupy.abs(asymmetry_matrix)))
        
        # Set tolerance based on dtype and matrix scale
        matrix_scale = float(cupy.max(cupy.abs(symmetric_matrix)))
        relative_tolerance = 1e-5 if working_dtype == cupy.float32 else 1e-10
        absolute_tolerance = relative_tolerance * max(1.0, matrix_scale)
        
        if max_asymmetry > absolute_tolerance:
            # Check if we can symmetrize (small asymmetry)
            if max_asymmetry < 1e-3 * matrix_scale:
                logger.warning(
                    f"Matrix has small asymmetry (max={max_asymmetry:.2e}). "
                    f"Symmetrizing as (A + A^T)/2 for numerical stability."
                )
                symmetric_matrix = (symmetric_matrix + symmetric_matrix.T) / 2.0
            else:
                raise ValueError(
                    f"Matrix is not symmetric. Maximum asymmetry: {max_asymmetry:.2e} "
                    f"(relative: {max_asymmetry/max(1e-10, matrix_scale):.2e}). "
                    "Set verify_symmetry=False to skip this check if you're certain."
                )
    
    # Step 3: Warn if k exceeds matrix dimension (will require padding)
    matrix_size = symmetric_matrix.shape[0]
    if k > matrix_size:
        logger.warning(
            f"Requested k={k} exceeds matrix dimension n={matrix_size}. "
            f"Result will be zero-padded for the last {k - matrix_size} components."
        )
    
    # Step 4: Perform eigendecomposition
    # eigh is specifically optimized for Hermitian (symmetric real) matrices
    try:
        eigenvalues, eigenvectors = cupy.linalg.eigh(symmetric_matrix)
    except cupy.linalg.LinAlgError as e:
        logger.error(f"Eigendecomposition failed: {e}")
        raise ValueError(
            f"Failed to decompose matrix. It may be singular or ill-conditioned: {e}"
        )
    
    # Step 5: Sort eigenvalues and eigenvectors in descending order
    # eigh returns eigenvalues in ascending order, so we reverse
    # Use negative indices for more efficient reversal
    sorted_eigenvalues = eigenvalues[::-1]
    sorted_eigenvectors = eigenvectors[:, ::-1]
    
    # Step 6: Filter for positive eigenvalues (above numerical threshold)
    positive_mask = sorted_eigenvalues > eigenvalue_threshold
    num_positive = int(cupy.sum(positive_mask))
    
    # Step 7: Handle case where no positive eigenvalues exist
    if num_positive == 0:
        logger.warning(
            f"No eigenvalues above threshold {eigenvalue_threshold} found. "
            f"Maximum eigenvalue: {float(sorted_eigenvalues[0]):.2e}. "
            "Returning zero matrices."
        )
        return (
            cupy.zeros((matrix_size, k), dtype=working_dtype),
            cupy.zeros(k, dtype=working_dtype)
        )
    
    # Step 8: Extract positive eigenpairs efficiently
    # Instead of boolean indexing, use explicit slicing for better performance
    positive_eigenvalues = sorted_eigenvalues[:num_positive]
    positive_eigenvectors = sorted_eigenvectors[:, :num_positive]
    
    # Log the eigenvalue spectrum for debugging
    logger.debug(
        f"Eigenvalue spectrum: {num_positive} positive "
        f"(max={float(positive_eigenvalues[0]):.2e}, "
        f"min={float(positive_eigenvalues[-1]):.2e}), "
        f"{matrix_size - num_positive} non-positive"
    )
    
    # Step 9: Return exactly k components (trim or pad as needed)
    if num_positive >= k:
        # We have enough positive eigenpairs - return the top k
        final_eigenvectors = positive_eigenvectors[:, :k].astype(working_dtype)
        final_eigenvalues = positive_eigenvalues[:k].astype(working_dtype)
        
        logger.debug(f"Returning top {k} of {num_positive} positive eigenpairs")
        
    else:
        # We need to pad with zeros to reach k components
        pad_width = k - num_positive
        
        logger.info(
            f"Found only {num_positive} positive eigenpairs, "
            f"padding with {pad_width} zero components to reach k={k}"
        )
        
        # Pad eigenvectors with zero columns
        final_eigenvectors = cupy.pad(
            positive_eigenvectors,
            pad_width=((0, 0), (0, pad_width)),
            mode='constant',
            constant_values=0
        ).astype(working_dtype)
        
        # Pad eigenvalues with zeros
        final_eigenvalues = cupy.pad(
            positive_eigenvalues,
            pad_width=(0, pad_width),
            mode='constant',
            constant_values=0
        ).astype(working_dtype)
    
    # Optional: Verify orthonormality of non-padded eigenvectors (for debugging)
    if logger.isEnabledFor(logging.DEBUG) and num_positive > 0:
        actual_k = min(k, num_positive)
        orthogonality_check = cupy.eye(actual_k, dtype=working_dtype) - (
            final_eigenvectors[:, :actual_k].T @ final_eigenvectors[:, :actual_k]
        )
        max_orthogonality_error = float(cupy.max(cupy.abs(orthogonality_check)))
        if max_orthogonality_error > 1e-5:
            logger.warning(
                f"Eigenvectors may not be perfectly orthonormal. "
                f"Max error: {max_orthogonality_error:.2e}"
            )
    
    return final_eigenvectors, final_eigenvalues


def balance_feature_streams(
    vector_streams: Dict[str, cupy.ndarray],
    proportions: Dict[Literal['semantic', 'tfidf', 'phonetic'], float]
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

def compute_average_gram_matrix(
    normalized_embeddings: List[cupy.ndarray],
    sample_indices: cupy.ndarray,
    verify_normalization: bool = True
) -> cupy.ndarray:
    """
    Computes the average Gram matrix (cosine similarity) from multiple embedding runs.

    This function is optimized for GPUs by stacking all embedding runs into a single
    3D array and using batch matrix multiplication to compute all Gram matrices
    simultaneously, avoiding Python for-loops for maximum performance.

    Mathematical Foundation:
    - For normalized vectors u and v: cos(θ) = u·v / (||u|| * ||v||) = u·v
    - Gram matrix G_i for run i = X_i @ X_i^T, where X_i is the matrix of samples.
    - Average Gram = (1/n_runs) * Σ(G_i), computed via cupy.mean on the batch.
    - The resulting matrix is symmetric with diagonal elements ≈ 1.0.

    Args:
        normalized_embeddings: A list of L2-normalized embedding matrices. Each
                               matrix must be a CuPy array of shape
                               (n_total_samples, embedding_dim).
        sample_indices: A 1D CuPy or NumPy array of integer indices specifying which
                        samples to include in the Gram matrix calculation.
        verify_normalization: If True, a random subset of embeddings is checked for
                              proper L2 normalization. Disable for performance if
                              you are certain they are normalized.

    Returns:
        A symmetric CuPy array of shape (n_samples, n_samples) representing the
        average cosine similarity matrix. Element [i, j] is the average cosine
        similarity between sample i and sample j across all runs.

    Raises:
        ValueError: If inputs are empty, have inconsistent shapes, contain invalid
                    indices, or if embeddings are found to be poorly normalized.
        TypeError: If inputs are not of the expected types (list of CuPy arrays).

    Performance Notes:
        - This vectorized approach is significantly faster than a looped approach.
        - Memory usage is higher, as it creates an intermediate 3D array of shape
          (n_runs, n_samples, n_samples). For extremely large `n_runs` or `n_samples`
          where memory is a constraint, a looped approach may be necessary.
        - Complexity: O(n_runs * n_samples^2 * embedding_dim), but with much better
          constants due to GPU parallelism.
    """
    # --- Step 1: Comprehensive Input Validation ---
    if not isinstance(normalized_embeddings, list) or not normalized_embeddings:
        raise ValueError("normalized_embeddings must be a non-empty list of CuPy arrays.")

    if not isinstance(sample_indices, (cupy.ndarray, np.ndarray)):
        raise TypeError("sample_indices must be a CuPy or NumPy array.")

    # Ensure sample_indices is a 1D CuPy array of integers
    if isinstance(sample_indices, np.ndarray):
        sample_indices = cupy.asarray(sample_indices)
    if sample_indices.ndim != 1 or len(sample_indices) == 0:
        raise ValueError("sample_indices must be a non-empty 1D array.")
    if not cupy.issubdtype(sample_indices.dtype, cupy.integer):
        logger.warning(f"Casting sample_indices from {sample_indices.dtype} to int32.")
        sample_indices = sample_indices.astype(cupy.int32)
    
    # --- Step 2: Dimension and Type Validation ---
    n_selected_samples = len(sample_indices)
    first_embedding = normalized_embeddings[0]

    if not isinstance(first_embedding, cupy.ndarray):
        raise TypeError("Elements of normalized_embeddings must be CuPy arrays.")

    # Establish reference dimensions from the first embedding matrix.
    n_total_samples, embedding_dimension = first_embedding.shape
    working_dtype = first_embedding.dtype
    if working_dtype not in [cupy.float32, cupy.float64]:
        working_dtype = cupy.float32
        logger.info(f"Embeddings are not float32/64. Using {working_dtype} for computation.")

    # Validate index bounds and embedding consistency across all runs.
    max_index = int(cupy.max(sample_indices))
    if int(cupy.min(sample_indices)) < 0:
        raise ValueError("sample_indices cannot contain negative values.")

    for i, emb in enumerate(normalized_embeddings):
        if not isinstance(emb, cupy.ndarray):
            raise TypeError(f"Element {i} in normalized_embeddings is not a CuPy array.")
        
        # Check for consistent row and column counts.
        if emb.shape[0] != n_total_samples:
            raise ValueError(
                f"Inconsistent number of samples at index {i}. Expected {n_total_samples}, "
                f"but got {emb.shape[0]}."
            )
        if emb.shape[1] != embedding_dimension:
            raise ValueError(
                f"Inconsistent embedding dimensions at index {i}. Expected {embedding_dimension}, "
                f"but got {emb.shape[1]}."
            )
        if max_index >= emb.shape[0]:
            raise ValueError(
                f"Index {max_index} is out of bounds for embedding {i} "
                f"(size: {emb.shape[0]})."
            )
        
        # Optional: Verify that a sample of vectors are normalized
        if verify_normalization:
            norms = cupy.linalg.norm(emb[cupy.random.choice(emb.shape[0], 10)], axis=1)
            if cupy.max(cupy.abs(norms - 1.0)) > 0.01:
                logger.warning(f"Normalization issue detected in embedding {i}.")

    # --- Step 3: Vectorized Gram Matrix Computation ---
    # Stack the list of 2D arrays into a single 3D array.
    # This is the key step that enables batch processing on the GPU.
    # Shape: (n_runs, n_total_samples, embedding_dim)
    try:
        stacked_embeddings = cupy.stack(normalized_embeddings).astype(working_dtype)
    except ValueError as e:
        raise ValueError("Failed to stack embeddings. Ensure all matrices have the same shape.") from e
    
    # Select the specified samples from all runs in a single, batched operation.
    # Shape: (n_runs, n_selected_samples, embedding_dim)
    selected_embeddings = stacked_embeddings[:, sample_indices, :]

    # Compute all Gram matrices at once using batched matrix multiplication.
    # This performs (X @ X.T) for each run in parallel.
    # Shape: (n_runs, n_selected_samples, n_selected_samples)
    all_gram_matrices = selected_embeddings @ selected_embeddings.transpose((0, 2, 1))

    # Average the Gram matrices along the 'runs' axis (axis=0).
    average_gram_matrix = cupy.mean(all_gram_matrices, axis=0)
    
    # --- Step 4: Finalization and Output Validation ---
    # Clip to handle any minor floating-point errors.
    average_gram_matrix = cupy.clip(average_gram_matrix, -1.0, 1.0)
    
    # Enforce perfect symmetry, correcting for potential numerical inaccuracies.
    average_gram_matrix = (average_gram_matrix + average_gram_matrix.T) / 2.0
    
    # Validate output properties
    diagonal_mean = float(cupy.diag(average_gram_matrix).mean())
    if abs(diagonal_mean - 1.0) > 0.05:
        logger.warning(
            f"Diagonal mean is {diagonal_mean:.4f}, which deviates from the expected 1.0. "
            "This may indicate normalization issues in the input embeddings."
        )
    logger.info(f"Average Gram matrix computed with shape {average_gram_matrix.shape} and diagonal mean {diagonal_mean:.4f}.")

    return average_gram_matrix


def project_out_of_sample_batch(
    batch_indices: cupy.ndarray,
    normalized_embeddings: List[cupy.ndarray],
    anchor_samples: List[cupy.ndarray],
    kpca_params: Dict,
    validate_inputs: bool = True,
    numerical_stability_epsilon: float = 1e-10
) -> cupy.ndarray:
    """
    Projects new samples into a consensus Kernel PCA space using ensemble averaging.
    
    This function performs out-of-sample projection for a consensus/ensemble kernel PCA
    approach. It computes kernel values between new samples and anchor points across
    multiple embedding runs, averages them, then projects into the pre-computed PCA space.
    
    Mathematical Foundation:
    1. Ensemble kernel computation: K_avg = (1/R) * Σ_r K_r(x_new, anchors_r)
    2. Kernel centering: K_centered = center(K_avg) using training statistics
    3. PCA projection: Z = K_centered @ V @ Λ^(-1/2)
    
    Where:
    - R = number of embedding runs (ensemble size)
    - K_r = kernel matrix for run r
    - V = eigenvectors from training kernel PCA
    - Λ = eigenvalues from training kernel PCA
    
    The consensus approach improves robustness by averaging over multiple embedding
    representations, reducing sensitivity to initialization or random variations.
    
    Use Cases:
    - Ensemble kernel PCA for robust dimensionality reduction
    - Multi-view kernel learning
    - Consensus clustering in kernel space
    - Robust anomaly detection using kernel methods
    
    Args:
        batch_indices: Indices of samples to project from the full embedding matrices.
                      Shape: (batch_size,), values in range [0, total_samples).
                      These index into the normalized_embeddings arrays.
        
        normalized_embeddings: List of L2-normalized embedding matrices, one per ensemble run.
                             Each has shape (total_samples, embedding_dim).
                             Must be pre-normalized for cosine similarity computation.
                             Length = n_runs (ensemble size).
        
        anchor_samples: List of anchor/reference samples for each embedding run.
                       Each has shape (n_anchors, embedding_dim).
                       These are the samples used to train the kernel PCA.
                       Must match the samples used to compute kpca_params.
                       Length must equal len(normalized_embeddings).
        
        kpca_params: Dictionary containing pre-computed Kernel PCA parameters:
                    - 'row_means': Training kernel row means, shape (n_anchors,)
                    - 'grand_mean': Training kernel grand mean (scalar)
                    - 'eigenvectors': PCA eigenvectors, shape (n_anchors, n_components)
                    - 'inv_sqrt_eigenvalues': 1/sqrt(eigenvalues), shape (n_components,)
                    
        validate_inputs: If True, performs comprehensive input validation.
                        Set to False only in performance-critical inner loops.
        
        numerical_stability_epsilon: Small value added to eigenvalues before computing
                                    1/sqrt to prevent division by zero.
                                    Only used if not already applied in kpca_params.
    
    Returns:
        projected_samples: Projected coordinates in the kernel PCA space.
                         Shape: (batch_size, n_components).
                         Each row is the low-dimensional representation of a sample.
    
    Raises:
        ValueError: If dimensions are incompatible, indices out of bounds,
                   or required parameters missing.
        TypeError: If inputs are not of expected types.
        KeyError: If required keys missing from kpca_params.
    
    Example:
        >>> # Training phase - compute kernel PCA on anchor samples
        >>> embeddings_train = [normalize(E) for E in embedding_runs]  # R embedding runs
        >>> anchors = [E[anchor_indices] for E in embeddings_train]
        >>> 
        >>> # Compute average kernel and its PCA
        >>> K_avg = compute_average_gram_matrix(embeddings_train, anchor_indices)
        >>> K_centered = center_kernel_matrix(K_avg)
        >>> eigvecs, eigvals = get_top_k_positive_eigenpairs(K_centered, k=50)
        >>> 
        >>> # Store parameters
        >>> kpca_params = {
        ...     'row_means': K_avg.mean(axis=1),
        ...     'grand_mean': K_avg.mean(),
        ...     'eigenvectors': eigvecs,
        ...     'inv_sqrt_eigenvalues': 1.0 / cupy.sqrt(eigvals + 1e-10)
        ... }
        >>> 
        >>> # Test phase - project new samples
        >>> test_indices = cupy.array([100, 101, 102])  # Indices of test samples
        >>> projections = project_out_of_sample_batch(
        ...     test_indices, embeddings_train, anchors, kpca_params
        ... )
        >>> print(projections.shape)  # (3, 50) - 3 samples, 50 components
    
    Algorithm Details:
        1. For each embedding run, compute dot products between new samples and anchors
        2. Average these kernel matrices across runs (consensus approach)
        3. Center using training statistics (critical for consistency)
        4. Project using eigenvectors and scale by inverse sqrt eigenvalues
        
    Performance Notes:
        - Time: O(batch_size * n_anchors * embedding_dim * n_runs)
        - Memory: O(batch_size * n_anchors) for kernel matrix
        - For large batches, consider processing in smaller chunks
    """
    
    # Step 1: Validate input types
    if not isinstance(batch_indices, cupy.ndarray):
        raise TypeError(f"batch_indices must be CuPy array, got {type(batch_indices).__name__}")
    
    if not isinstance(normalized_embeddings, list):
        raise TypeError("normalized_embeddings must be a list of CuPy arrays")
    
    if not isinstance(anchor_samples, list):
        raise TypeError("anchor_samples must be a list of CuPy arrays")
    
    if not isinstance(kpca_params, dict):
        raise TypeError(f"kpca_params must be a dictionary, got {type(kpca_params).__name__}")
    
    # Step 2: Extract and validate dimensions
    n_runs = len(normalized_embeddings)
    if n_runs == 0:
        raise ValueError("normalized_embeddings cannot be empty")
    
    if len(anchor_samples) != n_runs:
        raise ValueError(
            f"Mismatch: {n_runs} embedding runs but {len(anchor_samples)} anchor blocks. "
            "Must have one anchor block per embedding run."
        )
    
    batch_size = len(batch_indices)
    if batch_size == 0:
        raise ValueError("batch_indices cannot be empty")
    
    # Get dimensions from first anchor block
    n_anchors = anchor_samples[0].shape[0]
    anchor_dim = anchor_samples[0].shape[1]
    
    # Step 3: Validate kpca_params contains required keys
    required_keys = ['row_means', 'grand_mean', 'eigenvectors', 'inv_sqrt_eigenvalues']
    missing_keys = [key for key in required_keys if key not in kpca_params]
    if missing_keys:
        raise KeyError(f"kpca_params missing required keys: {missing_keys}")
    
    # Extract parameters for easier access
    row_means = kpca_params['row_means']
    grand_mean = kpca_params['grand_mean']
    eigenvectors = kpca_params['eigenvectors']
    inv_sqrt_eigenvalues = kpca_params['inv_sqrt_eigenvalues']
    
    # Step 4: Comprehensive dimension validation if requested
    if validate_inputs:
        # Check index bounds
        max_index = int(batch_indices.max())
        min_index = int(batch_indices.min())
        
        if min_index < 0:
            raise ValueError(f"batch_indices contains negative values: min={min_index}")
        
        # Validate all embedding matrices
        for i, (embedding, anchor_block) in enumerate(zip(normalized_embeddings, anchor_samples)):
            if not isinstance(embedding, cupy.ndarray):
                raise TypeError(f"normalized_embeddings[{i}] must be CuPy array")
            
            if not isinstance(anchor_block, cupy.ndarray):
                raise TypeError(f"anchor_samples[{i}] must be CuPy array")
            
            if embedding.ndim != 2:
                raise ValueError(
                    f"normalized_embeddings[{i}] must be 2D, got shape {embedding.shape}"
                )
            
            if max_index >= embedding.shape[0]:
                raise ValueError(
                    f"Index {max_index} out of bounds for embedding[{i}] "
                    f"with {embedding.shape[0]} samples"
                )
            
            if anchor_block.shape != (n_anchors, anchor_dim):
                raise ValueError(
                    f"anchor_samples[{i}] has shape {anchor_block.shape}, "
                    f"expected ({n_anchors}, {anchor_dim})"
                )
            
            if embedding.shape[1] != anchor_dim:
                raise ValueError(
                    f"Dimension mismatch: embedding[{i}] has {embedding.shape[1]} dims, "
                    f"anchors have {anchor_dim} dims"
                )
        
        # Validate KPCA parameters dimensions
        if len(row_means) != n_anchors:
            raise ValueError(
                f"row_means length {len(row_means)} doesn't match "
                f"number of anchors {n_anchors}"
            )
        
        if eigenvectors.shape[0] != n_anchors:
            raise ValueError(
                f"eigenvectors has {eigenvectors.shape[0]} rows, "
                f"expected {n_anchors} (number of anchors)"
            )
        
        n_components = eigenvectors.shape[1]
        if len(inv_sqrt_eigenvalues) != n_components:
            raise ValueError(
                f"inv_sqrt_eigenvalues length {len(inv_sqrt_eigenvalues)} "
                f"doesn't match number of components {n_components}"
            )
    
    # Step 5: Initialize accumulator for averaging kernel matrices
    # This will store the sum of kernel matrices across all runs
    avg_kernel_matrix = cupy.zeros(
        (batch_size, n_anchors), 
        dtype=cupy.float32
    )
    
    # Step 6: Compute and accumulate kernel matrices for each embedding run
    for run_idx, (embedding_matrix, anchor_block) in enumerate(
        zip(normalized_embeddings, anchor_samples)
    ):
        # Extract embeddings for the batch samples
        # Shape: (batch_size, embedding_dim)
        batch_embeddings = embedding_matrix[batch_indices]
        
        # Compute kernel (cosine similarity) between batch and anchors
        # Since embeddings are normalized: K = batch @ anchors^T
        # Shape: (batch_size, n_anchors)
        kernel_matrix = batch_embeddings @ anchor_block.T
        
        # Accumulate for averaging
        avg_kernel_matrix += kernel_matrix
        
        logger.debug(
            f"Run {run_idx}: kernel range [{float(kernel_matrix.min()):.4f}, "
            f"{float(kernel_matrix.max()):.4f}]"
        )
    
    # Step 7: Complete the averaging
    avg_kernel_matrix /= float(n_runs)
    
    logger.debug(
        f"Average kernel computed: shape={avg_kernel_matrix.shape}, "
        f"range=[{float(avg_kernel_matrix.min()):.4f}, {float(avg_kernel_matrix.max()):.4f}]"
    )
    
    # Step 8: Center the kernel matrix using training statistics
    # This ensures new samples are centered consistently with training data
    centered_kernel_matrix = center_kernel_vector(
        avg_kernel_matrix,
        row_means,
        grand_mean
    )
    
    # Step 9: Project into PCA space
    # First multiply by eigenvectors: (batch_size, n_anchors) @ (n_anchors, n_components)
    # This gives coordinates in eigenvector basis
    projected_coordinates = centered_kernel_matrix @ eigenvectors
    
    # Step 10: Scale by inverse square root of eigenvalues
    # This normalizes the projection by the variance in each component
    # Whitening transformation: makes all components have unit variance
    scaled_projections = projected_coordinates * inv_sqrt_eigenvalues[None, :]
    
    # Log projection statistics
    logger.debug(
        f"Projection complete: shape={scaled_projections.shape}, "
        f"range=[{float(scaled_projections.min()):.4f}, {float(scaled_projections.max()):.4f}], "
        f"mean={float(scaled_projections.mean()):.4f}, "
        f"std={float(scaled_projections.std()):.4f}"
    )
    
    # Step 11: Validate output for numerical issues
    if cupy.any(cupy.isnan(scaled_projections)) or cupy.any(cupy.isinf(scaled_projections)):
        logger.error(
            "NaN or Inf detected in projections! Check eigenvalues for near-zero values "
            "and ensure input embeddings are properly normalized."
        )
        raise ValueError("Numerical instability detected in projection")
    
    return scaled_projections


def create_consensus_embedding(
    embeddings_list: List[cupy.ndarray],
    n_anchor_samples: int,
    batch_size: int = 1000,
    random_state: Optional[int] = None,
    n_components: Optional[int] = None,
    eigenvalue_threshold: float = 1e-6,
    normalize_output: bool = True
) -> cupy.ndarray:
    """
    Creates a robust consensus embedding by combining multiple embedding runs via Kernel PCA.
    
    This function implements a sophisticated ensemble approach that is invariant to:
    - Scale differences between embedding runs
    - Rotational/reflection differences (orientation)
    - Random initialization effects
    - Local optima in individual runs
    
    The consensus is achieved by:
    1. Computing average cosine similarities across all runs
    2. Using Kernel PCA to find the principal components of similarity structure
    3. Projecting all points into this consensus space
    
    Mathematical Foundation:
    The method finds the embedding Y that best preserves the average kernel matrix:
    K_avg = (1/R) * Σ_r normalize(X_r) @ normalize(X_r)^T
    where X_r is embedding from run r, and R is the number of runs.
    
    Kernel PCA then finds Y such that Y @ Y^T ≈ K_avg (in the principal subspace).
    
    Applications:
    - Combining multiple runs of stochastic embedding algorithms (t-SNE, UMAP, etc.)
    - Multi-view learning where each view provides an embedding
    - Robust dimensionality reduction resistant to outliers
    - Ensemble clustering in embedded space
    
    Args:
        embeddings_list: List of embedding matrices from different runs/views.
                        Each array has shape (n_points, embedding_dim).
                        All arrays must have the same number of points (rows).
                        Embedding dimensions (columns) can vary between runs.
        
        n_anchor_samples: Number of anchor/reference points for Kernel PCA.
                         If less than total points, uses sampling strategy.
                         Larger values = more accurate but slower.
                         Recommended: min(5000, n_points) for efficiency.
        
        batch_size: Number of points to process simultaneously when projecting
                   out-of-sample points. Larger = faster but more memory.
                   Default: 1000 (good for most GPUs).
        
        random_state: Seed for reproducible anchor point sampling.
                     If None, uses current random state.
                     Set for reproducible results across runs.
        
        n_components: Number of dimensions in consensus embedding.
                     If None, uses the minimum embedding dimension from input list.
                     Cannot exceed n_anchor_samples.
        
        eigenvalue_threshold: Minimum eigenvalue to consider as positive.
                            Smaller values retain more components but may be noisy.
                            Default: 1e-6 (suitable for float32).
        
        normalize_output: If True, L2-normalizes each output embedding vector.
                         Recommended for downstream cosine similarity tasks.
    
    Returns:
        consensus_embedding: Combined embedding matrix of shape (n_points, n_components).
                           Each row is the consensus representation of a point.
                           If normalize_output=True, rows are unit vectors.
    
    Raises:
        ValueError: If embeddings_list is empty, dimensions incompatible,
                   or parameters invalid.
        MemoryError: If batch processing still exceeds GPU memory.
    
    Example:
        >>> # Run t-SNE multiple times with different initializations
        >>> embeddings = []
        >>> for seed in range(10):
        ...     tsne = TSNE(random_state=seed)
        ...     embedding = tsne.fit_transform(data)
        ...     embeddings.append(cupy.asarray(embedding))
        >>> 
        >>> # Create consensus embedding
        >>> consensus = create_consensus_embedding(
        ...     embeddings,
        ...     n_anchor_samples=2000,
        ...     batch_size=500,
        ...     random_state=42
        ... )
        >>> print(f"Consensus shape: {consensus.shape}")
    
    Algorithm Complexity:
        Time: O(R * n_anchors^2 * d + n_points * n_anchors * d)
        Memory: O(n_anchors^2 + n_points * d)
        where R = number of runs, d = embedding dimension
    
    Implementation Notes:
        - Uses subsampling for efficiency when n_points > n_anchor_samples
        - Anchor points define the kernel PCA model
        - Remaining points are projected as out-of-sample data
        - Batch processing prevents GPU memory overflow
    """
    
    # Step 1: Validate inputs
    if not embeddings_list:
        raise ValueError("embeddings_list cannot be empty")
    
    if not all(isinstance(E, cupy.ndarray) for E in embeddings_list):
        raise TypeError("All elements in embeddings_list must be CuPy arrays")
    
    # Extract dimensions and validate consistency
    n_points = embeddings_list[0].shape[0]
    embedding_dims = [E.shape[1] for E in embeddings_list]
    n_runs = len(embeddings_list)
    
    # Validate all embeddings have same number of points
    for i, E in enumerate(embeddings_list):
        if E.ndim != 2:
            raise ValueError(f"Embedding {i} must be 2D, got shape {E.shape}")
        if E.shape[0] != n_points:
            raise ValueError(
                f"All embeddings must have same number of points. "
                f"Embedding 0 has {n_points}, embedding {i} has {E.shape[0]}"
            )
    
    # Determine output dimensions
    min_dim = min(embedding_dims)
    max_dim = max(embedding_dims)
    if n_components is None:
        n_components = min_dim
        if min_dim != max_dim:
            logger.info(
                f"Embedding dimensions vary from {min_dim} to {max_dim}. "
                f"Using minimum dimension {min_dim} for consensus."
            )
    else:
        if n_components > min_dim:
            logger.warning(
                f"Requested n_components={n_components} exceeds minimum "
                f"embedding dimension {min_dim}. Using {min_dim} instead."
            )
            n_components = min_dim
    
    # Validate anchor samples parameter
    if n_anchor_samples <= 0:
        raise ValueError(f"n_anchor_samples must be positive, got {n_anchor_samples}")
    
    if n_anchor_samples > n_points:
        logger.warning(
            f"n_anchor_samples ({n_anchor_samples}) exceeds n_points ({n_points}). "
            f"Using all {n_points} points as anchors."
        )
        n_anchor_samples = n_points
    
    if n_components > n_anchor_samples:
        raise ValueError(
            f"n_components ({n_components}) cannot exceed "
            f"n_anchor_samples ({n_anchor_samples})"
        )
    
    # Step 2: Log configuration
    logger.debug(
        f"Creating consensus embedding:\n"
        f"  - Input: {n_runs} embeddings, {n_points} points\n"
        f"  - Dimensions: {embedding_dims}\n"
        f"  - Output: {n_components} components\n"
        f"  - Anchors: {n_anchor_samples} samples\n"
        f"  - Batch size: {batch_size}"
    )
    
    # Step 3: Normalize all embeddings for cosine similarity computation
    logger.debug("Normalizing embeddings for cosine similarity...")
    
    normalized_embeddings = []
    for i, E in enumerate(embeddings_list):
        E_normalized = normalize_rows(E)
        normalized_embeddings.append(E_normalized)
        
        # Check for numerical issues
        norms = cupy.linalg.norm(E_normalized, axis=1)
        if cupy.any(cupy.isnan(norms)) or cupy.any(norms == 0):
            raise ValueError(f"Embedding {i} contains zero or NaN vectors")
    
    # Step 4: Sample anchor points for Kernel PCA
    if random_state is not None:
        cupy.random.seed(random_state)
    
    use_sampling = (n_points > n_anchor_samples)
    
    if use_sampling:
        # Sample without replacement for anchors
        logger.debug(
            f"Sampling {n_anchor_samples} anchor points from {n_points} total points..."
        )
        anchor_indices = cupy.sort(
            cupy.random.choice(n_points, n_anchor_samples, replace=False)
        )
    else:
        # Use all points as anchors
        logger.debug(f"Using all {n_points} points as anchors (no sampling needed)")
        anchor_indices = cupy.arange(n_points)
        n_anchor_samples = n_points
    
    # Step 5: Compute average Gram matrix over ensemble
    logger.debug(f"Computing average Gram matrix from {n_runs} embedding runs...")
    
    avg_gram = compute_average_gram_matrix(normalized_embeddings, anchor_indices)
    
    # Log Gram matrix statistics
    gram_min = float(avg_gram.min())
    gram_max = float(avg_gram.max())
    gram_mean = float(avg_gram.mean())
    
    logger.debug(
        f"Gram matrix statistics: min={gram_min:.4f}, "
        f"max={gram_max:.4f}, mean={gram_mean:.4f}"
    )
    
    # Step 6: Center the Gram matrix for Kernel PCA
    logger.debug("Centering Gram matrix...")
    
    centered_gram = center_kernel_matrix(avg_gram)
    
    # Step 7: Compute eigendecomposition for Kernel PCA
    logger.debug(f"Computing top {n_components} eigenpairs...")
    
    eigenvectors, eigenvalues = get_top_k_positive_eigenpairs(
        centered_gram, 
        k=n_components,
        eigenvalue_threshold=eigenvalue_threshold
    )
    
    # Log eigenvalue spectrum
    n_positive = int(cupy.sum(eigenvalues > eigenvalue_threshold))
    logger.debug(
        f"Eigenvalue spectrum: {n_positive} positive eigenvalues, "
        f"top 5: {eigenvalues[:5].tolist() if len(eigenvalues) >= 5 else eigenvalues.tolist()}"
    )
    
    if n_positive < n_components:
        logger.warning(
            f"Only {n_positive} positive eigenvalues found, but {n_components} "
            f"components requested. Remaining components will be zero."
        )
    
    # Step 8: Compute anchor embedding from eigenvectors
    # Y = V * Λ^(1/2) gives embedding where Y @ Y^T = V @ Λ @ V^T
    sqrt_eigenvalues = cupy.sqrt(cupy.maximum(eigenvalues, 0))  # Ensure non-negative
    anchor_embedding = eigenvectors * sqrt_eigenvalues[None, :]
    
    if normalize_output:
        anchor_embedding = normalize_rows(anchor_embedding)
    
    # Step 9: Handle full vs. sampled case
    if not use_sampling:
        # All points were anchors - we're done!
        return anchor_embedding
    
    # Step 10: Project out-of-sample points into consensus space
    logger.debug(
        f"Projecting {n_points - n_anchor_samples} out-of-sample points "
        f"in batches of {batch_size}..."
    )
    
    # Initialize full embedding matrix
    final_embedding = cupy.zeros((n_points, n_components), dtype=cupy.float32)
    
    # Place anchor embeddings in their positions
    final_embedding[anchor_indices] = anchor_embedding
    
    # Prepare parameters for out-of-sample projection
    kpca_params = {
        'row_means': avg_gram.mean(axis=1),  # Shape: (n_anchors,)
        'grand_mean': float(avg_gram.mean()),
        'eigenvectors': eigenvectors,
        'inv_sqrt_eigenvalues': 1.0 / (sqrt_eigenvalues + 1e-10)
    }
    
    # Extract anchor samples from each embedding run
    anchor_blocks = [E[anchor_indices] for E in normalized_embeddings]
    
    # Find indices of points that need projection
    all_indices = cupy.arange(n_points)
    remaining_indices = cupy.setdiff1d(all_indices, anchor_indices)
    n_remaining = len(remaining_indices)
    
    # Process in batches to manage memory
    n_batches = (n_remaining + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_remaining)
        batch_indices = remaining_indices[start_idx:end_idx]
        
        if batch_idx % max(1, n_batches // 10) == 0:
            progress = (batch_idx + 1) / n_batches * 100
            logger.debug(f"  Projection progress: {progress:.1f}%")
        
        # Project batch into consensus space
        try:
            batch_embedding = project_out_of_sample_batch(
                batch_indices,
                normalized_embeddings,
                anchor_blocks,
                kpca_params,
                validate_inputs=(batch_idx == 0)  # Only validate first batch
            )
            
            if normalize_output:
                batch_embedding = normalize_rows(batch_embedding)
            
            final_embedding[batch_indices] = batch_embedding
            
        except Exception as e:
            logger.error(f"Failed to project batch {batch_idx}: {e}")
            raise
    
    # Step 11: Final validation
    if cupy.any(cupy.isnan(final_embedding)) or cupy.any(cupy.isinf(final_embedding)):
        raise ValueError(
            "NaN or Inf detected in final embedding. "
            "Check input data for anomalies or adjust eigenvalue_threshold."
        )
    
    # Log final statistics
    final_min = float(final_embedding.min())
    final_max = float(final_embedding.max())
    final_mean = float(final_embedding.mean())
    final_std = float(final_embedding.std())
    variance_explained = eigenvalues[:n_components] / eigenvalues.sum()
    
    logger.debug(
        f"Consensus embedding completed successfully\n"
        f"  Output shape: {final_embedding.shape}\n"
        f"  Statistics: min={final_min:.4f}, max={final_max:.4f}, "
        f"mean={final_mean:.4f}, std={final_std:.4f}\n"
        f"Variance explained by {n_components} components: {variance_explained.sum():.2%}"
    )
    
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
