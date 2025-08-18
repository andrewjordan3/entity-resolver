# entity_resolver/utils/matrix_ops.py
"""
GPU-Accelerated Utility Functions for Dense and Sparse Matrix Operations.

This module provides a collection of stateless, reusable utility functions for
cleaning, normalizing, and conditioning CuPy matrices. These operations are
fundamental pre-processing steps for numerical algorithms, particularly for
improving the stability and performance of iterative solvers used in machine
learning components like Singular Value Decomposition (SVD).

The functions are designed to handle both dense `cupy.ndarray` and sparse
`cupyx.scipy.sparse` matrices efficiently, performing all computations on the
GPU. They serve as the core building blocks for the pre-processing pipelines
in the entity resolution components.

Key Functions:
- ensure_finite_matrix: Validates and optionally cleans NaN/Inf values.
- winsorize_matrix: Clips matrix values at specified quantiles to handle outliers.
- scale_by_frobenius_norm: Normalizes the entire matrix to have a unit norm.
- prune_sparse_matrix: Removes uninformative rows and columns to improve stability.
"""

import cupy
import cupyx.scipy.sparse as cpx_sparse
import logging
from typing import Union, Tuple, Optional

# Set up a logger for this module
logger = logging.getLogger(__name__)

def ensure_finite_matrix(
    matrix: Union[cupy.ndarray, cpx_sparse.spmatrix],
    *,
    replace_non_finite: bool = False,
    copy: bool = True,
) -> Union[cupy.ndarray, cpx_sparse.spmatrix]:
    """
    Checks for and optionally replaces non-finite values in a CuPy matrix.

    This function validates that a matrix does not contain any NaN (Not a Number)
    or infinite values. It can either raise an error upon detection or replace
    these values with zero. For sparse matrices, operations are performed only
    on the non-zero elements stored in the `.data` attribute.

    Args:
        matrix:
            The input matrix to validate/clean. Can be a CuPy ndarray or any
            CuPy sparse matrix format.
        replace_non_finite:
            If False (default), the function will raise a FloatingPointError if
            any non-finite values are found. If True, it will replace all NaN
            and infinite values with 0.0.
        copy:
            If True (default), the function operates on a copy of the input
            matrix. If False, the operation is performed in-place, which
            can save memory but will modify the original matrix.

    Returns:
        The validated or cleaned matrix.

    Raises:
        FloatingPointError:
            If `replace_non_finite` is False and any non-finite values are
            detected in the matrix.
        TypeError:
            If the input `matrix` is not a CuPy ndarray or sparse matrix.
    """
    # --- Dense Matrix Path ---
    if isinstance(matrix, cupy.ndarray):
        target_matrix = matrix.copy() if copy else matrix
        is_finite = cupy.isfinite(target_matrix).all()

        if is_finite:
            logger.debug("Confirmed all values are finite in the dense matrix.")
            return target_matrix

        if replace_non_finite:
            logger.debug("Replacing non-finite values with 0 in the dense matrix.")
            # cupy.nan_to_num replaces NaN with 0, pos-inf with large number, neg-inf with small number.
            # We will use it to handle NaN, then handle inf separately for a clean 0 replacement.
            cupy.nan_to_num(target_matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            return target_matrix
        else:
            raise FloatingPointError("Non-finite (NaN or Inf) values detected in the dense matrix.")

    # --- Sparse Matrix Path ---
    elif cpx_sparse.isspmatrix(matrix):
        target_matrix = matrix.copy() if copy else matrix
        is_finite = cupy.isfinite(target_matrix.data).all()

        if is_finite:
            logger.debug("Confirmed all values are finite in the sparse matrix data.")
            return target_matrix
        
        if replace_non_finite:
            logger.debug("Replacing non-finite values with 0 in the sparse matrix data.")
            cupy.nan_to_num(target_matrix.data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            # After replacing, some stored elements might be zero. Clean them up.
            target_matrix.eliminate_zeros()
            return target_matrix
        else:
            raise FloatingPointError("Non-finite (NaN or Inf) values detected in the sparse matrix data.")

    # --- Invalid Type ---
    else:
        raise TypeError(
            f"Input must be a cupy.ndarray or cupyx.scipy.sparse.spmatrix, "
            f"but got {type(matrix)}"
        )

def winsorize_matrix(
    matrix: Union[cupy.ndarray, cpx_sparse.spmatrix],
    *,
    limits: Tuple[Optional[float], Optional[float]] = (0.01, 0.99),
    copy: bool = True,
) -> Union[cupy.ndarray, cpx_sparse.spmatrix]:
    """
    Clips matrix values at specified lower and upper quantiles (Winsorization).

    This function limits extreme values in a dataset to reduce the effect of
    spurious outliers. It computes the values at the q-th quantiles and
    caps any values below or above these thresholds.

    For sparse matrices, winsorization is applied only to the non-zero elements.

    Args:
        matrix:
            The input matrix to winsorize. Can be a CuPy ndarray or any
            CuPy sparse matrix format.
        limits:
            A tuple of two floats (lower, upper) representing the quantile
            limits. Values must be between 0.0 and 1.0. For example, (0.01, 0.99)
            caps the lowest 1% and highest 1% of values. Use `None` for a limit
            to disable clipping on that side (e.g., (0.01, None) only clips the
            lower tail).
        copy:
            If True (default), the function operates on a copy of the input
            matrix. If False, the operation is performed in-place, which
            can save memory but will modify the original matrix.

    Returns:
        The winsorized matrix, with the same type and shape as the input.

    Raises:
        ValueError:
            If the quantile limits are invalid (e.g., not between 0 and 1).
        TypeError:
            If the input `matrix` is not a CuPy ndarray or sparse matrix.
    """
    lower_limit, upper_limit = limits
    
    # --- Validate Limits ---
    if lower_limit is not None and not (0.0 <= lower_limit <= 1.0):
        raise ValueError(f"Lower limit must be between 0.0 and 1.0, but got {lower_limit}")
    if upper_limit is not None and not (0.0 <= upper_limit <= 1.0):
        raise ValueError(f"Upper limit must be between 0.0 and 1.0, but got {upper_limit}")
    if lower_limit is not None and upper_limit is not None and lower_limit >= upper_limit:
        raise ValueError(
            f"Lower limit ({lower_limit}) cannot be greater than or equal to upper limit ({upper_limit})"
        )

    # --- Dense Matrix Path ---
    if isinstance(matrix, cupy.ndarray):
        if matrix.size == 0:
            return matrix.copy() if copy else matrix # Nothing to do

        logger.debug(f"Winsorizing dense matrix of shape {matrix.shape} with limits={limits}")
        
        target_matrix = matrix.copy() if copy else matrix
        
        # Calculate the values at the specified quantiles
        lower_bound = cupy.quantile(target_matrix, lower_limit) if lower_limit is not None else None
        upper_bound = cupy.quantile(target_matrix, upper_limit) if upper_limit is not None else None
        
        logger.debug(f"Dense path bounds: lower={lower_bound}, upper={upper_bound}")

        # Clip the matrix values. cupy.clip handles None for bounds correctly.
        cupy.clip(target_matrix, a_min=lower_bound, a_max=upper_bound, out=target_matrix)
        
        return target_matrix

    # --- Sparse Matrix Path ---
    elif cpx_sparse.isspmatrix(matrix):
        if matrix.nnz == 0:
            return matrix.copy() if copy else matrix # Nothing to do
            
        logger.debug(f"Winsorizing sparse matrix of shape {matrix.shape} (nnz={matrix.nnz}) with limits={limits}")

        target_matrix = matrix.copy() if copy else matrix
        
        # For sparse matrices, we only winsorize the non-zero values.
        data = target_matrix.data
        
        # Calculate quantile values from the .data array
        lower_bound = cupy.quantile(data, lower_limit) if lower_limit is not None else None
        upper_bound = cupy.quantile(data, upper_limit) if upper_limit is not None else None

        logger.debug(f"Sparse path bounds: lower={lower_bound}, upper={upper_bound}")

        # Clip the .data array in-place
        cupy.clip(data, a_min=lower_bound, a_max=upper_bound, out=data)
        
        return target_matrix

    # --- Invalid Type ---
    else:
        raise TypeError(
            f"Input must be a cupy.ndarray or cupyx.scipy.sparse.spmatrix, "
            f"but got {type(matrix)}"
        )
    
def scale_by_frobenius_norm(
    matrix: Union[cupy.ndarray, cpx_sparse.spmatrix],
    *,
    epsilon: Optional[float] = None,
    copy: bool = True,
) -> Tuple[Union[cupy.ndarray, cpx_sparse.spmatrix], float]:
    """
    Scales a matrix by the inverse of its Frobenius norm.

    This operation normalizes the entire matrix so that its new Frobenius norm
    is 1.0. It's a common pre-conditioning step for numerical algorithms to
    prevent issues with very large or small singular values. If the norm is
    below a small epsilon, the matrix is not scaled to avoid division by zero.

    Args:
        matrix:
            The input matrix to scale. Can be a CuPy ndarray or any
            CuPy sparse matrix format.
        epsilon:
            A small threshold. If the calculated Frobenius norm is less than
            this value, the matrix is considered to be effectively zero and
            is not scaled. If None, a default is chosen based on dtype.
        copy:
            If True (default), the function operates on a copy of the input
            matrix. If False, the scaling is performed in-place where possible.

    Returns:
        A tuple containing:
            - scaled_matrix: The matrix scaled by the inverse of its norm.
            - scale_factor: The computed scale factor (1.0 / norm). Returns
              1.0 if the matrix was not scaled.

    Raises:
        TypeError:
            If the input `matrix` is not a CuPy ndarray or sparse matrix.
    """
    # --- Epsilon Handling ---
    if epsilon is None:
        if matrix.dtype == cupy.float32:
            epsilon = 1e-7
        else: # Default for float64 and others
            epsilon = 1e-12
            
    target_matrix = matrix.copy() if copy else matrix
    frobenius_norm = 0.0

    # --- Dense Matrix Path ---
    if isinstance(target_matrix, cupy.ndarray):
        if target_matrix.size == 0:
            return target_matrix, 1.0 # Nothing to scale
        # .item() safely converts the 0-dim CuPy array to a Python float
        frobenius_norm = float(cupy.linalg.norm(target_matrix, 'fro'))

    # --- Sparse Matrix Path ---
    elif cpx_sparse.isspmatrix(target_matrix):
        if target_matrix.nnz == 0:
            return target_matrix, 1.0 # Nothing to scale
        # Efficiently calculate norm from the .data attribute
        norm_squared = cupy.sum(target_matrix.data ** 2)
        frobenius_norm = float(cupy.sqrt(norm_squared))

    # --- Invalid Type ---
    else:
        raise TypeError(
            f"Input must be a cupy.ndarray or cupyx.scipy.sparse.spmatrix, "
            f"but got {type(matrix)}"
        )

    # --- Apply Scaling ---
    if frobenius_norm > epsilon:
        scale_factor = 1.0 / frobenius_norm
        logger.debug(
            f"Scaling matrix of shape {matrix.shape} by Frobenius norm. "
            f"Norm={frobenius_norm:.4e}, Scale Factor={scale_factor:.4e}"
        )
        # Scale the data directly for clarity and efficiency
        if isinstance(target_matrix, cupy.ndarray):
            target_matrix *= scale_factor
        else:
            target_matrix.data *= scale_factor
            
        return target_matrix, scale_factor
    else:
        logger.debug(
            f"Matrix of shape {matrix.shape} has near-zero Frobenius norm "
            f"({frobenius_norm:.4e}). Skipping scaling."
        )
        return target_matrix, 1.0
    
def prune_sparse_matrix(
    matrix: cpx_sparse.csr_matrix,
    *,
    min_row_sum: float = 1e-9,
    min_df: int = 2,
    max_df_ratio: float = 0.98,
    energy_cutoff_ratio: float = 0.995,
    copy: bool = True,
) -> Tuple[cpx_sparse.csr_matrix, cupy.ndarray, cupy.ndarray]:
    """
    Prunes rows and columns from a sparse matrix to improve numerical stability.

    This function applies a series of conditioning steps:
    1. Row Pruning: Removes rows that are nearly empty (sum of values is near zero).
    2. Column Pruning (by Document Frequency): Removes columns (features) that
       are either too rare (min_df) or too common (max_df_ratio).
    3. Column Pruning (by Energy): From the remaining columns, it keeps the
       smallest subset whose cumulative energy exceeds the specified cutoff ratio.

    Args:
        matrix:
            The input CSR sparse matrix to be pruned. Must be in CSR format.
        min_row_sum:
            Rows with a total sum of values less than this threshold will be removed.
        min_df:
            Columns that appear in fewer than `min_df` documents (rows) will be removed.
        max_df_ratio:
            Columns that appear in more than `max_df_ratio * n_rows` documents
            will be removed.
        energy_cutoff_ratio:
            The ratio of total energy to preserve in the DF-filtered columns.
        copy:
            If True (default), operates on a copy. If False, performs operations
            in-place where possible.

    Returns:
        A tuple containing:
            - pruned_matrix: The smaller, pruned CSR matrix.
            - kept_row_indices: A 1D CuPy array of the original row indices that were kept.
            - kept_column_indices: A 1D CuPy array of the original column indices that were kept.
            
    Raises:
        TypeError: If the input is not a CuPy CSR sparse matrix.
        ValueError: If pruning results in an empty matrix or parameters are invalid.
    """
    # --- Step 0: Validate Inputs and Data Type ---
    if not cpx_sparse.isspmatrix_csr(matrix):
        raise TypeError("Input matrix must be a cupyx.scipy.sparse.csr_matrix.")
    if min_df < 1:
        raise ValueError(f"min_df must be >= 1, got {min_df}")
    if not (0.0 <= min_row_sum):
        raise ValueError(f"min_row_sum must be >= 0, got {min_row_sum}")
    if not (0.0 < max_df_ratio <= 1.0):
        raise ValueError(f"max_df_ratio must be in (0, 1], got {max_df_ratio}")
    if not (0.0 < energy_cutoff_ratio <= 1.0):
        raise ValueError(f"energy_cutoff_ratio must be in (0, 1], got {energy_cutoff_ratio}")

    target_matrix = matrix.copy() if copy else matrix
    
    # Ensure floating point math for energy calculations
    if not cupy.issubdtype(target_matrix.dtype, cupy.floating):
        target_matrix = target_matrix.astype(cupy.float64, copy=False)

    n_samples, n_features = target_matrix.shape

    # --- Step 1: Row Pruning ---
    row_sums = cupy.asarray(target_matrix.sum(axis=1)).ravel()
    valid_rows_mask = row_sums > min_row_sum
    kept_row_indices = cupy.where(valid_rows_mask)[0]
    
    if kept_row_indices.size < n_samples:
        logger.debug(
            f"Pruning rows: removing {n_samples - kept_row_indices.size} rows with sum < {min_row_sum:.1e}."
        )
        target_matrix = target_matrix[kept_row_indices, :]
        n_samples = target_matrix.shape[0]
    
    if n_samples == 0:
        raise ValueError("All rows were removed during pruning. Matrix is empty.")

    # --- Step 2: Filter Columns by Document Frequency ---
    doc_frequency = cupy.bincount(target_matrix.indices, minlength=n_features)
    max_df = int(max_df_ratio * n_samples)
    
    df_mask = (doc_frequency >= min_df) & (doc_frequency <= max_df)
    df_keepable_cols = cupy.where(df_mask)[0]

    if df_keepable_cols.size == 0:
        raise ValueError("No columns survived document frequency pruning.")

    # --- Step 3: Filter DF-eligible columns by Energy ---
    col_energy = cupy.bincount(
        target_matrix.indices,
        weights=(target_matrix.data ** 2),
        minlength=n_features
    )
    
    # Consider only the energy of columns that passed the DF filter
    energy_in_keepable_cols = col_energy[df_keepable_cols]
    total_energy_in_subset = float(energy_in_keepable_cols.sum())

    if energy_in_keepable_cols.size == 0 or total_energy_in_subset == 0.0:
        raise ValueError("Zero keepable energy after DF pruning; cannot proceed.")

    # Sort the keepable columns by their energy
    order = cupy.argsort(energy_in_keepable_cols)[::-1]
    
    # Find the smallest number of these columns that meet the energy cutoff
    cumulative_energy = cupy.cumsum(energy_in_keepable_cols[order])
    cutoff_point = energy_cutoff_ratio * total_energy_in_subset
    num_cols_for_cutoff = int(cupy.searchsorted(cumulative_energy, cutoff_point)) + 1

    # Final indices are the top-energy columns from the DF-filtered set
    kept_column_indices = df_keepable_cols[order[:num_cols_for_cutoff]]

    if kept_column_indices.size == 0:
        raise ValueError("All columns were removed during energy pruning.")

    logger.debug(
        "Pruning columns: kept %d/%d (DF: min_df=%d, max_df=%d) | energy_cutoff=%.3f",
        int(kept_column_indices.size), int(n_features), int(min_df), int(max_df), float(energy_cutoff_ratio)
    )
    
    # --- Apply Final Pruning ---
    pruned_matrix = target_matrix[:, kept_column_indices]
    pruned_matrix.eliminate_zeros()
    
    # Ensure the final matrix is in canonical CSR format
    pruned_matrix.sort_indices()

    return pruned_matrix, kept_row_indices, kept_column_indices