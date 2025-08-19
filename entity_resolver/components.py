# entity_resolver/components.py
"""
Custom GPU-accelerated components for entity resolution pipeline.

This module provides custom implementations of machine learning components
that are optimized for GPU execution and sparse matrix operations.
"""

import logging
from typing import Optional, Dict, Any, Union, Tuple

import cupy
from cupyx.scipy.sparse import csr_matrix, spmatrix, isspmatrix_csr
from cupyx.scipy.sparse.linalg import svds, eigsh, LinearOperator

from .utils import (
    normalize_rows,
    ensure_finite_matrix,
    prune_sparse_matrix,
    winsorize_matrix,
    scale_by_frobenius_norm,
    create_initial_vector
)

# Set up module-level logger
logger = logging.getLogger(__name__)


class GPUTruncatedSVD:
    """
    GPU-accelerated Truncated SVD for sparse matrix dimensionality reduction.

    This class provides a scikit-learn-style interface for GPU-accelerated
    Singular Value Decomposition on sparse matrices. It is designed to be a
    performant and robust tool for feature extraction in high-dimensional
    sparse datasets, such as text corpora represented as TF-IDF matrices.

    The primary solver is `cupyx.scipy.sparse.linalg.svds`. If this solver
    fails due to numerical instability, the class automatically falls back to
    a more robust method based on `eigsh` applied to an "augmented matrix."
    This fallback path includes a comprehensive pre-processing pipeline to
    condition the matrix for the solver.

    Attributes:
        n_components (int):
            The target number of dimensions for the output.
        components_ (cupy.ndarray):
            The principal axes in feature space, representing the directions of
            maximum variance (also known as V^T in SVD). Shape: (n_components, n_features).
        singular_values_ (cupy.ndarray):
            The singular values corresponding to each of the selected components,
            sorted in descending order.
        explained_variance_ (cupy.ndarray):
            The amount of variance explained by each of the selected components.
            Calculated as `singular_values_**2 / (n_samples - 1)`.
        explained_variance_ratio_ (cupy.ndarray):
            The percentage of variance explained by each of the selected components.
            This is a relative measure, normalized to sum to 1.0 over the
            computed components.
    """

    # --- Model State (populated during fit) ---
    # Type hints for attributes that will be populated by fit()
    components_: Optional[cupy.ndarray] = None
    singular_values_: Optional[cupy.ndarray] = None
    explained_variance_: Optional[cupy.ndarray] = None
    explained_variance_ratio_: Optional[cupy.ndarray] = None
    total_variance_: Optional[float] = None
    true_explained_variance_ratio_: Optional[float] = None

    def __init__(
            self, 
            n_components: int = 256, 
            fallback_config: Optional[Dict[str, Any]] = None, 
            **svds_kwargs: Any
        ):
        """
        Initialize the GPUTruncatedSVD transformer.

        Args:
            n_components:
                The target number of dimensions for the output (the 'k' in SVD).
            fallback_config:
                A dictionary of parameters to control the robust SVD fallback mechanism.
            **svds_kwargs:
                Additional keyword arguments to pass to the primary `svds` solver.
                Common arguments include 'tol' for tolerance and 'maxiter' for
                maximum iterations.
        """
        self.n_components = n_components
        self.svds_kwargs = svds_kwargs

        # Ensure fallback_config is a dict to prevent errors on .get() if it's None
        safe_fallback_config = fallback_config or {}

        # --- Solver & Pre-processing Parameters ---
        # Use .get() to safely retrieve each parameter from the config dictionary,
        # providing a hardcoded default value if the key is not present.
        # Use high precision for the stable solver.
        self.fallback_dtype = safe_fallback_config.get('fallback_dtype', cupy.float64)
        # Number of times to retry eigsh on failure.
        self.eigsh_restarts = safe_fallback_config.get('eigsh_restarts', 3)
        # Threshold for removing near-empty rows.
        self.prune_min_row_sum = safe_fallback_config.get('prune_min_row_sum', 1e-9)
        # Min document frequency for column pruning.
        self.prune_min_df = safe_fallback_config.get('prune_min_df', 2)
        # Max document frequency ratio for column pruning.
        self.prune_max_df_ratio = safe_fallback_config.get('prune_max_df_ratio', 0.98)
        # Preserve 99.5% of variance during column pruning.
        self.prune_energy_cutoff = safe_fallback_config.get('prune_energy_cutoff', 0.995)
        # Clip top 0.1% of values to handle extreme outliers.
        self.winsorize_limits: Tuple[Optional[float], Optional[float]] = safe_fallback_config.get('winsorize_limits', (None, 0.999))  

    def _prepare_input_matrix(self, X: spmatrix) -> csr_matrix:
        """
        Validates, cleans, and standardizes the input sparse matrix.

        This helper method performs several critical checks and conversions:
        1. Ensures all matrix values are finite, replacing NaN/Inf if necessary.
        2. Converts the matrix to the CSR format, which is efficient for the
           operations used in this class.
        3. Ensures the matrix has a floating-point data type.
        4. Validates that `n_components` is a feasible value for the matrix shape.

        Args:
            X: The raw input sparse matrix.

        Returns:
            A clean, validated CSR matrix ready for SVD.
        """
        logger.debug("Preparing input matrix.")
        # Use the utility function to handle non-finite values first.
        clean_matrix = ensure_finite_matrix(X, replace_non_finite=True, copy=True)
        
        # Ensure CSR format for efficient row slicing and matrix multiplication.
        if not isspmatrix_csr(clean_matrix):
            clean_matrix = clean_matrix.tocsr(copy=False)
            
        # Ensure a floating-point dtype for all subsequent calculations.
        if not cupy.issubdtype(clean_matrix.dtype, cupy.floating):
             clean_matrix = clean_matrix.astype(self.fallback_dtype, copy=False)
        
        # Validate shape constraints for SVD. `k` must be less than the smallest dimension.
        n_samples, n_features = clean_matrix.shape
        if self.n_components >= min(n_samples, n_features):
            raise ValueError(
                f"n_components ({self.n_components}) must be < min(n_samples, n_features) "
                f"({min(n_samples, n_features)})"
            )
        return clean_matrix

    def _get_eigsh_restart_params(self, current_params: Dict[str, Any], n_aug: int) -> Dict[str, Any]:
        """
        Generates tweaked parameters for an eigsh restart attempt.

        If `eigsh` fails, this method provides a new set of slightly relaxed
        parameters to increase the chance of convergence on the next attempt.

        Args:
            current_params: A dictionary of the `ncv`, `tol`, and `maxiter` from
                            the failed attempt.
            n_aug: The dimension of the augmented matrix, used to cap `ncv`.

        Returns:
            A dictionary with the new, relaxed parameters.
        """
        # Gently reduce the number of Lanczos vectors (`ncv`).
        new_ncv = min(max(self.n_components + 2, int(0.9 * current_params['ncv'])), n_aug - 1)
        # Relax the tolerance by a factor of 10, but not beyond a reasonable floor.
        new_tol = max(current_params['tol'] * 10, 1e-7)
        # Increase the maximum number of iterations.
        new_maxiter = int(current_params['maxiter'] * 1.5)
        
        logger.warning(
            f"Restarting eigsh with new params: ncv={new_ncv}, tol={new_tol:.1e}, maxiter={new_maxiter}"
        )
        return {'ncv': new_ncv, 'tol': new_tol, 'maxiter': new_maxiter}

    def fit(self, X: spmatrix, y: Optional[Any] = None) -> 'GPUTruncatedSVD':
        """
        Fit the model to the data X. This is a convenience method that calls
        `fit_transform` and returns the fitted instance.
        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X: spmatrix, y: Optional[Any] = None) -> cupy.ndarray:
        """
        Fit the model to the data and perform dimensionality reduction.

        This is the main method where the SVD computation happens. It first
        attempts the fast `svds` solver and falls back to a more robust `eigsh`
        -based method if necessary.
        """
        logger.info(f"Starting fit_transform for GPUTruncatedSVD with n_components={self.n_components}")

        # --- Step 1: Prepare and Validate Input Matrix ---
        clean_matrix = self._prepare_input_matrix(X)
        n_samples, n_features = clean_matrix.shape
        decomposed_matrix = clean_matrix
        kept_row_indices = cupy.arange(n_samples)

        # --- Step 2: Attempt Standard SVD ---
        try:
            logger.debug("Attempting standard 'svds' solver.")
            U, s, V_T = svds(clean_matrix, k=self.n_components, **self.svds_kwargs)
            if cupy.all(s == 0):
                raise RuntimeError("svds solver returned all zero singular values, indicating non-convergence.")
            logger.info("Standard 'svds' solver succeeded.")

        # --- Step 3: Fallback to Robust Augmented Eigendecomposition ---
        except Exception as e:
            logger.warning(f"Standard svds failed: {e}. Falling back to robust eigsh method.")

            # --- 3a: Pre-process the matrix for stability ---
            # This pipeline of functions conditions the matrix to make it more
            # amenable to the iterative eigsh solver.
            pruned_matrix, kept_row_indices, kept_col_indices = prune_sparse_matrix(
                clean_matrix.astype(self.fallback_dtype),
                min_row_sum=self.prune_min_row_sum,
                min_df=self.prune_min_df,
                max_df_ratio=self.prune_max_df_ratio,
                energy_cutoff_ratio=self.prune_energy_cutoff,
            )
            winsorized_matrix = winsorize_matrix(pruned_matrix, limits=self.winsorize_limits, copy=False)
            normalized_matrix = normalize_rows(winsorized_matrix, copy=False)
            scaled_matrix, scale_factor = scale_by_frobenius_norm(normalized_matrix, copy=False)
            decomposed_matrix = scaled_matrix # Use this highly processed matrix for variance calculation.
            
            n_pruned_samples, n_pruned_features = scaled_matrix.shape
            n_aug = n_pruned_samples + n_pruned_features

            # --- 3b: Define the Augmented Matrix Operator ---
            # This defines the matrix B = [[0, A], [A.T, 0]] implicitly through
            # its matrix-vector product. This avoids ever forming the large
            # (n+m) x (n+m) matrix in memory.
            def _augmented_matvec(vector_z: cupy.ndarray) -> cupy.ndarray:
                vec_u, vec_v = vector_z[:n_pruned_samples], vector_z[n_pruned_samples:]
                final_result = cupy.concatenate([scaled_matrix @ vec_v, scaled_matrix.T @ vec_u])
                if not cupy.isfinite(final_result).all():
                    max_val = float(cupy.abs(final_result[cupy.isfinite(final_result)]).max()) if cupy.isfinite(final_result).any() else 0
                    logger.error(
                        f"Non-finite values in matvec result. Max finite: {max_val:.2e}, "
                        f"NaNs: {int(cupy.isnan(final_result).sum())}, Infs: {int(cupy.isinf(final_result).sum())}"
                    )
                    raise FloatingPointError("NaN or Inf generated during matrix-vector product.")
                return final_result

            aug_op = LinearOperator(shape=(n_aug, n_aug), matvec=_augmented_matvec, dtype=self.fallback_dtype)

            # --- 3c: Run Eigsh with Restarts ---
            # Set initial parameters, favoring user-provided ones.
            eigsh_params: Dict[str, Any] = {
                'tol': self.svds_kwargs.get('tol', 1e-8),
                'maxiter': self.svds_kwargs.get('maxiter', 20000),
                'ncv': self.svds_kwargs.get('ncv', min(max(2 * self.n_components + 1, 20), n_aug - 1))
            }
            
            eigenvalues, eigenvectors = None, None
            for attempt in range(self.eigsh_restarts):
                try:
                    logger.debug(f"Starting eigsh attempt {attempt + 1}/{self.eigsh_restarts} with params: {eigsh_params}")
                    initial_vec = create_initial_vector(n_aug, dtype=self.fallback_dtype, seed=attempt)
                    eigenvalues, eigenvectors = eigsh(aug_op, k=self.n_components, v0=initial_vec, **eigsh_params)
                    logger.info("Eigsh solver succeeded.")
                    break # Exit loop on success
                except Exception as eigsh_e:
                    logger.warning(f"Eigsh attempt {attempt + 1} failed: {eigsh_e}")
                    if attempt < self.eigsh_restarts - 1:
                        eigsh_params = self._get_eigsh_restart_params(eigsh_params, n_aug)
                    else: # Last attempt failed
                        raise RuntimeError("Eigsh fallback failed after all restarts.") from eigsh_e

            # --- 3d: Reconstruct SVD components from Eigenpairs ---
            # The singular values of A are the positive eigenvalues of B.
            s = cupy.maximum(eigenvalues, 0.0) / scale_factor
            # The right singular vectors (V) are in the bottom part of the eigenvectors.
            V_pruned = eigenvectors[n_pruned_samples:, :]
            V_pruned /= cupy.linalg.norm(V_pruned, axis=0, keepdims=True)
            # Reconstruct the full V matrix, inserting zeros for pruned columns.
            V_full = cupy.zeros((n_features, self.n_components), dtype=self.fallback_dtype)
            V_full[kept_col_indices, :] = V_pruned
            V_T = V_full.T

        # --- Step 4: Finalize and Store Results ---
        # Sort all components by singular value in descending order.
        descending_indices = cupy.argsort(s)[::-1]
        self.singular_values_ = s[descending_indices]
        self.components_ = V_T[descending_indices, :]
        self._calculate_variance_explained(decomposed_matrix, decomposed_matrix.shape[0])
        logger.info(f"GPUTruncatedSVD fit complete. Top singular value: {float(self.singular_values_[0]):.4e}")

        # --- Step 5: Transform the Data ---
        # The final transformation must be performed on the original, full-sized matrix
        # to ensure the output dimensions match the input sample count.
        transformed_matrix = self.transform(clean_matrix)
        
        # If rows were pruned during the fallback, we must re-insert zero-rows
        # into the output to maintain the original sample dimension.
        if kept_row_indices.size < n_samples:
            logger.debug("Re-inserting zero rows for pruned samples in the output.")
            full_transformed_matrix = cupy.zeros((n_samples, self.n_components), dtype=transformed_matrix.dtype)
            full_transformed_matrix[kept_row_indices] = transformed_matrix
            transformed_matrix = full_transformed_matrix

        return transformed_matrix

    def transform(self, X: spmatrix) -> cupy.ndarray:
        """
        Transform a matrix using the fitted SVD components.

        This projects the input data onto the principal components (V) learned
        during the `fit` stage. The formula is `X_transformed = X @ V`.
        """
        if self.components_ is None:
            raise RuntimeError("This SVD instance has not been fitted yet. Call 'fit' or 'fit_transform' first.")
        
        # Prepare matrix
        # Prepare the matrix for transformation (ensures correct dtype and no NaNs).
        prepared_matrix = self._prepare_input_matrix(X)
        
        # Validate that the number of features matches the fitted model.
        if prepared_matrix.shape[1] != self.components_.shape[1]:
            raise ValueError(f"Input has {prepared_matrix.shape[1]} features, but model was fitted with {self.components_.shape[1]} features.")
        
        logger.debug(f"Transforming matrix of shape {X.shape}")
        # The transformation is a matrix multiplication: X @ V, which is equivalent to X @ (V^T)^T.
        return prepared_matrix @ self.components_.T

    def reset(self) -> None:
        """
        Resets the fitted model state to its initial, unfitted condition.

        This method clears all attributes that are populated during the `fit` or
        `fit_transform` process, such as the SVD components and singular values.
        After calling `reset`, the instance is ready to be refitted on new data.

        Note: This method does NOT explicitly free GPU memory. For that, call
        `cupy.get_default_memory_pool().free_all_blocks()` separately after you
        are finished with the SVD object and other large CuPy arrays.
        """
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.total_variance_ = None
        self.true_explained_variance_ratio_ = None
        logger.debug("Model state has been reset.")

    def _calculate_variance_explained(self, original_matrix: csr_matrix, n_samples: int) -> None:
        """
        Calculate variance explained by the SVD components.
        
        For TruncatedSVD on sparse matrices (which are NOT centered), the variance
        calculation differs from PCA. The total variance is the Frobenius norm squared
        of the matrix divided by (n-1).
        
        Args:
            original_matrix: The matrix that was actually decomposed (post-processing)
            n_samples: Number of samples in the matrix
        """
        # For sparse matrices in TruncatedSVD, we DON'T center the data
        # Total variance is simply ||X||_F^2 / (n-1)
        # where ||X||_F is the Frobenius norm
        
        # Safeguard against edge cases
        n_samples_adjusted = max(n_samples - 1, 1)

        # Calculate Frobenius norm squared efficiently for sparse matrix
        frobenius_norm_squared = float((original_matrix.data ** 2).sum())
        
        # Total variance estimate
        # Note: This is the total variance of the matrix we actually decomposed
        # If we pruned columns or normalized, this reflects that processed matrix
        total_variance = frobenius_norm_squared / n_samples_adjusted
        
        # Component variance: s_i^2 / (n-1)
        self.explained_variance_ = (self.singular_values_ ** 2) / n_samples_adjusted
        
        # For Truncated SVD, we can only explain variance up to the sum of
        # the k singular values we computed. The true total variance would require
        # ALL singular values, which we don't have.
        
        # Two approaches for explained variance ratio:
        
        # Approach 1: Conservative (what sklearn does)
        # Use the sum of computed variances as denominator
        # This ensures ratios sum to 1.0 but doesn't reflect true % of total variance
        sum_explained_variance = float(self.explained_variance_.sum())
        
        # Approach 2: Approximate total variance from Frobenius norm
        # This can give ratios > 1 if we're missing significant singular values
        # but is more truthful about what % of total variance we captured
        
        # Use Approach 1 (conservative, always sums to ≤ 1.0)
        eps = 1e-18
        if sum_explained_variance > eps:
            # This ensures the ratios sum to exactly 1.0 for the components we have
            self.explained_variance_ratio_ = self.explained_variance_ / sum_explained_variance
        else:
            self.explained_variance_ratio_ = cupy.zeros_like(self.explained_variance_)
        
        # Store the actual total variance for diagnostic purposes
        self.total_variance_ = total_variance
        
        # Calculate what fraction of TRUE total variance we captured
        # This can be > 1.0 if matrix was poorly conditioned, but usually should be < 1.0
        self.true_explained_variance_ratio_ = sum_explained_variance / max(total_variance, eps)
        
        # Log diagnostic information
        cumulative_variance = float(self.explained_variance_ratio_.sum())
        true_variance_captured = float(self.true_explained_variance_ratio_)
        
        logger.info(
            f"Variance explained (relative): {cumulative_variance:.4f}, "
            f"Variance captured (absolute): {true_variance_captured:.4f}"
        )
        
        if true_variance_captured > 1.0:
            logger.warning(
                f"Captured variance ratio > 1.0 ({true_variance_captured:.4f}). "
                "This suggests numerical instability in the matrix. "
                "The explained_variance_ratio_ has been normalized to sum to 1.0."
            )
        elif true_variance_captured < 0.5:
            logger.warning(
                f"Only {true_variance_captured:.2%} of true variance captured. "
                f"Consider increasing n_components (current: {self.n_components})."
            )
