# entity_resolver/components.py
"""
Custom GPU-accelerated components for entity resolution pipeline.

This module provides custom implementations of machine learning components
that are optimized for GPU execution and sparse matrix operations.
"""

import logging
from typing import Tuple, Optional

import cupy
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import svds, eigsh, LinearOperator

# Set up module-level logger
logger = logging.getLogger(__name__)


class GPUTruncatedSVD:
    """
    GPU-accelerated Truncated SVD for sparse matrix dimensionality reduction.
    
    This class provides a scikit-learn-style interface for the GPU-accelerated
    `svds` function, enabling efficient dimensionality reduction directly on
    sparse matrices. It serves as a replacement for `cuml.TruncatedSVD`, which
    lacks sparse matrix support.
    
    The class performs Singular Value Decomposition (SVD) to extract the most
    significant principal components from high-dimensional sparse data, which
    is crucial for the initial dimensionality reduction in feature processing.
    
    Attributes:
        n_components (int): Number of components to extract
        components_ (cupy.ndarray): Principal axes in feature space (V^T from SVD)
        singular_values_ (cupy.ndarray): Singular values in descending order
        explained_variance_ (cupy.ndarray): Variance explained by each component
        explained_variance_ratio_ (cupy.ndarray): Percentage of variance explained
    """
    
    def __init__(self, n_components: int = 256, **svds_kwargs):
        """
        Initialize the GPUTruncatedSVD transformer.

        Args:
            n_components: Target number of dimensions for output (the 'k' in SVD).
                         Must be less than min(n_samples, n_features).
            **svds_kwargs: Additional arguments for cupyx.scipy.sparse.linalg.svds:
                          - maxiter: Maximum number of iterations (default: None)
                          - tol: Tolerance for convergence (default: 0)
                          - which: Which k singular values to find (default: 'LM')
                          - return_singular_vectors: Whether to compute U and V (default: True)
        """
        self.n_components = n_components
        self.svds_kwargs = svds_kwargs
        
        # Model state - will be populated during fit
        self.components_ = None  # V^T matrix: (n_components, n_features)
        self.singular_values_ = None  # Top k singular values
        self.explained_variance_ = None  # Variance explained by each component
        self.explained_variance_ratio_ = None  # Relative variance explained
        
        logger.debug(
            f"Initialized GPUTruncatedSVD with n_components={n_components}, "
            f"svds_kwargs={svds_kwargs}"
        )

    def _ensure_finite(self, array: cupy.ndarray, context: str):
        """Checks if all elements in a CuPy array are finite, raising an error if not."""
        if not cupy.isfinite(array).all():
            raise FloatingPointError(
                f"Non-finite (NaN or Inf) values detected in '{context}'."
            )
        logger.debug(f"Confirmed all values are finite in '{context}'.")

    def _l2_normalize_rows(self, matrix: csr_matrix) -> csr_matrix:
        """
        Applies row-wise L2 normalization to a sparse CSR matrix.

        This helper function ensures each row vector has a Euclidean norm of 1,
        a standard pre-processing step for algorithms based on cosine similarity.

        Args:
            matrix: The input CSR matrix to normalize.

        Returns:
            The row-normalized CSR matrix.
        """
        logger.debug("Performing row-wise L2 normalization.")
        # Calculate the L2 norm for each row.
        row_norms = cupy.sqrt(matrix.power(2).sum(axis=1)).ravel()
        
        # Create scale factors, guarding against division by zero for empty rows.
        eps = 1e-8
        scale_factors = cupy.where(row_norms > eps, 1.0 / row_norms, 0.0)

        # Apply scaling efficiently to the non-zero data of the CSR matrix.
        # This 'searchsorted' trick maps each non-zero element in `matrix.data`
        # to its corresponding row index, allowing us to apply the correct
        # scale factor without iterating or creating a dense diagonal matrix.
        data_row_ids = cupy.searchsorted(
            matrix.indptr,
            cupy.arange(matrix.nnz, dtype=matrix.indptr.dtype),
            side="right"
        ) - 1
        
        normalized_matrix = matrix.copy()
        cupy.multiply(
            normalized_matrix.data, 
            scale_factors[data_row_ids], 
            out=normalized_matrix.data
        )
        self._ensure_finite(normalized_matrix.data, "L2 Normalized Rows")
        
        logger.debug(
            f"Row-normalized matrix: {int(cupy.sum(row_norms > eps))}/{matrix.shape[0]} "
            "non-zero rows processed."
        )
        return normalized_matrix

    def _prepare_input_matrix(self, sparse_matrix: cupy.sparse.spmatrix) -> cupy.sparse.csr_matrix:
        """
        Validates and prepares the input sparse matrix for SVD.
        
        Ensures the matrix is a CSR matrix, contains no NaN/inf values, and has 
        a dtype of float64 for high-precision calculations.
        
        Args:
            sparse_matrix: The input CuPy sparse matrix.
            
        Returns:
            A sparse CSR matrix with dtype=cupy.float64.
        """
        if not isinstance(sparse_matrix, cupy.sparse.spmatrix):
            raise TypeError(f"Input must be a CuPy sparse matrix, not {type(sparse_matrix)}")

        # Check for invalid values in the sparse data array
        if cupy.isnan(sparse_matrix.data).any():
            raise ValueError("Input sparse matrix contains NaN values.")
        if cupy.isinf(sparse_matrix.data).any():
            raise ValueError("Input sparse matrix contains infinity values.")
            
        # Ensure CSR format for efficient row slicing and matrix multiplication
        if not isinstance(sparse_matrix, csr_matrix):
            sparse_matrix = sparse_matrix.tocsr()
            
        # Ensure float64 for high-precision SVD and variance calculations
        if sparse_matrix.dtype != cupy.float64:
            sparse_matrix = sparse_matrix.astype(cupy.float64)
        
        logger.debug(f"Prepared matrix: shape={sparse_matrix.shape}, dtype={sparse_matrix.dtype}")
        return sparse_matrix

    def _prune_and_normalize_matrix(
        self, sparse_matrix: csr_matrix
    ) -> Tuple[csr_matrix, cupy.ndarray]:
        """
        Prunes and normalizes a sparse matrix to improve numerical stability for SVD.

        This helper function applies a series of steps to condition the matrix:
        1.  Removes microscopic values that are likely numerical noise.
        2.  Filters out columns (features) that are either too common or contribute very little.
        3.  Applies row-wise L2 normalization via a dedicated helper.

        Args:
            sparse_matrix: The input CSR matrix to be processed.

        Returns:
            A tuple containing:
            - The pruned and row-normalized sparse matrix.
            - A 1D CuPy array containing the indices of the columns that were kept.
        """
        logger.debug("Starting matrix pruning and normalization for SVD fallback.")
        n_samples, n_features = sparse_matrix.shape
        matrix_to_prune = sparse_matrix.copy()

        # Step 1: Prune microscopic values
        pruning_threshold = 1e-12
        matrix_to_prune.data[cupy.abs(matrix_to_prune.data) < pruning_threshold] = 0
        matrix_to_prune.eliminate_zeros()
        logger.debug(f"Pruned microscopic values below {pruning_threshold:.1e}.")

        # Step 1b: Zero-out lower-tail tiny values (bottom 0.1%)
        if matrix_to_prune.nnz > 0:
            # Percentile expects q in [0, 100]; use 0.1 for the 0.1th percentile
            lb = cupy.percentile(matrix_to_prune.data, 1.0)

            # If the percentile threshold is above the microscopic threshold, apply it
            # (otherwise this step is redundant with the 1e-15 prune)
            if float(lb) > pruning_threshold:
                mask = matrix_to_prune.data < lb
                n_zeroed = int(mask.sum().item())
                if n_zeroed:
                    matrix_to_prune.data[mask] = 0
                    matrix_to_prune.eliminate_zeros()
                    logger.debug(
                        f"Zeroed bottom 0.1% values (< {float(lb):.3e}); "
                        f"removed {n_zeroed:,} entries."
                    )

        # Step 2: Prune columns by document frequency and energy
        row_sums = cupy.asarray(matrix_to_prune.sum(axis=1)).ravel()
        row_threshold = 1e-10  # Rows with total sum below this are problematic
        valid_rows = row_sums > row_threshold
        n_valid_rows = int(valid_rows.sum())
        
        if n_valid_rows < n_samples:
            logger.warning(
                f"Removing {n_samples - n_valid_rows} near-empty rows "
                f"(sum < {row_threshold:.1e}) to prevent numerical issues."
            )
            matrix_to_prune = matrix_to_prune[valid_rows, :]
            n_samples = n_valid_rows
        
        if n_samples == 0:
            raise ValueError("All rows were removed during pruning. Matrix is essentially empty.")

        doc_frequency = cupy.bincount(matrix_to_prune.indices, minlength=n_features)
        max_df_ratio = 0.98
        df_mask = doc_frequency <= int(max_df_ratio * n_samples)

        # Also remove columns that appear in too few documents (min_df)
        min_df = max(2, int(0.001 * n_samples))  # At least 0.1% of documents
        df_mask &= doc_frequency >= min_df

        col_energy = cupy.bincount(
            matrix_to_prune.indices,
            weights=(matrix_to_prune.data ** 2),
            minlength=n_features
        )
        total_energy = float(col_energy.sum())
        if total_energy == 0.0:
            raise ValueError("All columns have zero energy after microscopic pruning; cannot proceed.")
        
        # Keep columns that contribute meaningful energy
        energy_threshold = 1e-10 * total_energy / n_features
        energy_mask = col_energy > energy_threshold

        energy_cutoff_ratio = 0.995
        energy_desc_order = cupy.argsort(col_energy)[::-1]
        cumulative_energy = cupy.cumsum(col_energy[energy_desc_order])
        num_cols_for_cutoff = int(cupy.searchsorted(
            cumulative_energy, energy_cutoff_ratio * cumulative_energy[-1]
        ) + 1)
        energy_cols_to_keep = energy_desc_order[:num_cols_for_cutoff]
        cumulative_energy_mask = cupy.zeros(n_features, dtype=bool)
        cumulative_energy_mask[energy_cols_to_keep] = True

        final_keep_mask = df_mask & energy_mask & cumulative_energy_mask
        kept_column_indices = cupy.where(final_keep_mask)[0]

        if kept_column_indices.size == 0:
            raise ValueError("Matrix pruning removed all columns. Cannot proceed.")

        logger.debug(
            f"Pruned columns: kept {len(kept_column_indices)}/{n_features} "
            f"(DF ratio < {max_df_ratio}, Energy ratio > {energy_cutoff_ratio})."
        )
        pruned_matrix = matrix_to_prune[:, kept_column_indices]
        pruned_matrix.eliminate_zeros()

        # Pre-normalization outlier removal (BEFORE L2 normalization)
        # This is crucial - outliers can cause huge problems during normalization
        if pruned_matrix.nnz > 0:
            # Use a more aggressive percentile for outlier detection
            upper_percentile = 99.0  # Changed from 99.9
            ub = cupy.percentile(pruned_matrix.data, upper_percentile)
            
            if cupy.isfinite(ub) and float(ub) > 0:
                outlier_mask = pruned_matrix.data > ub
                n_outliers = int(outlier_mask.sum())
                if n_outliers > 0:
                    # Cap outliers to the threshold
                    pruned_matrix.data[outlier_mask] = ub
                    logger.debug(
                        f"Capped {n_outliers} outliers (>{float(ub):.3e}) "
                        f"at {upper_percentile}th percentile."
                    )

        # Step 3: Row-wise L2 Normalization
        normalized_matrix = self._l2_normalize_rows(pruned_matrix)

        # Ensure no infinite or NaN values crept in
        if not cupy.isfinite(normalized_matrix.data).all():
            # Find and fix problematic values
            bad_mask = ~cupy.isfinite(normalized_matrix.data)
            n_bad = int(bad_mask.sum())
            if n_bad > 0:
                logger.warning(f"Found {n_bad} non-finite values after normalization. Replacing with 0.")
                normalized_matrix.data[bad_mask] = 0
                normalized_matrix.eliminate_zeros()
        
        # Final validation
        if normalized_matrix.nnz == 0:
            raise ValueError("Matrix became empty after normalization.")
        
        # If dealing with the augmented matrix, also need to handle the valid_rows mapping
        if n_valid_rows < sparse_matrix.shape[0]:
            # Return the valid row indices as well for proper reconstruction
            return normalized_matrix, kept_column_indices, cupy.where(valid_rows)[0]
        
        return normalized_matrix, kept_column_indices

    def _run_eigsh_with_restarts(
        self,
        augmented_matrix_operator: LinearOperator,
        n_components: int,
        n_augmented_dims: int,
        *,
        which: str = 'LA',
        ncv: int = None,
        tol: float = 1e-8,
        maxiter: int = 50000,
        restarts: int = 3,
        seed: int = 0,
        dtype: cupy.dtype = cupy.float64
    ) -> Tuple[cupy.ndarray, cupy.ndarray]:
        """Run eigsh with safe restarts if the Lanczos algorithm breaks down."""

        random_state = cupy.random.RandomState(seed)

        def make_initial_vector():
            """Build an initial v0 that is not aligned with trivial directions."""
            initial_vector = random_state.standard_normal(n_augmented_dims, dtype=dtype)
            # Orthogonalize against all-ones to avoid an easy pathology
            ones = cupy.ones(n_augmented_dims, dtype=dtype) / cupy.sqrt(n_augmented_dims)
            projection = (initial_vector @ ones).item()
            initial_vector = initial_vector - projection * ones
            norm = cupy.linalg.norm(initial_vector)
            return initial_vector / (norm if norm > 0 else 1.0)

        attempt = 0
        current_ncv = ncv
        current_tol = tol
        current_maxiter = maxiter
        last_error = None

        while attempt < restarts:
            try:
                initial_vector = make_initial_vector()
                eigenvalues, eigenvectors = eigsh(
                    augmented_matrix_operator, k=n_components, which=which,
                    ncv=current_ncv, tol=current_tol, maxiter=current_maxiter,
                    return_eigenvectors=True, v0=initial_vector
                )
                return eigenvalues, eigenvectors
            except FloatingPointError as e:
                last_error = e
                attempt += 1
                # Gentle tweaks: reduce ncv, relax tol, give more iterations
                current_ncv = min(max(n_components + 2, int(0.9 * current_ncv)), n_augmented_dims - 1)
                current_tol = max(current_tol, 1e-7)  # Don’t go *tighter* on restart
                current_maxiter = int(current_maxiter * 1.5)
                logger.warning(
                    f"eigsh breakdown (attempt {attempt}/{restarts}): {e}. "
                    f"Restarting with ncv={current_ncv}, tol={current_tol}, maxiter={current_maxiter}."
                )
        # If we get here, all restarts failed
        raise RuntimeError(f"eigsh failed after {restarts} restarts") from last_error

    def _eigsh_augmented_tsvd(
        self,
        sparse_matrix: csr_matrix,
        n_components: int,
        ncv: int = None,
        tol: float = 1.0e-8,
        maxiter: int = 20000
    ) -> Tuple[cupy.ndarray, cupy.ndarray, cupy.ndarray]:
        """
        Numerically stable SVD using an augmented matrix approach. Fallback for when svds fails.

        This method computes the SVD by finding the eigenpairs of a larger,
        symmetric "augmented matrix" B = [[0, X], [X.T, 0]]. This avoids forming
        X.T @ X, which can be unstable.

        Args:
            sparse_matrix: The input data matrix (X).
            n_components: The number of singular values to compute (k).
            ncv: Number of Lanczos vectors for the eigsh solver.
            tol: Tolerance for convergence for the eigsh solver.
            maxiter: Maximum iterations for the eigsh solver.

        Returns:
            A tuple (U, s, V_T) containing the SVD components.
        """
        logger.warning(
            "The standard 'svds' solver failed. Falling back to the numerically stable "
            "augmented matrix (eigsh) method with pre-processing."
        )
        original_n_features = sparse_matrix.shape[1]
        dtype = cupy.float64

        # Step 1: Prune and Normalize the Matrix for stability
        processed_matrix, kept_column_indices = self._prune_and_normalize_matrix(sparse_matrix)
        n_samples, n_features_pruned = processed_matrix.shape
        n_aug = n_samples + n_features_pruned

        if n_components >= n_aug - 1:
            raise ValueError(f"n_components ({n_components}) must be < n_samples + n_features_pruned - 1 ({n_aug-1})")

        # --- Robust upper-tail clipping BEFORE alpha ---
        # Winsorize only the top tail to tame rare heavy entries; do NOT raise small ones.
        # Choose a conservative percentile (e.g., 99.9) so you only touch true outliers.
        winsor_p = 99.9
        if processed_matrix.nnz > 0:
            ub = cupy.percentile(processed_matrix.data, winsor_p)
            # Skip if ub is non-finite or non-positive (shouldn't happen for TF-IDF)
            if cupy.isfinite(ub) and float(ub) > 0.0:
                # Cap only the upper tail in-place
                cupy.minimum(processed_matrix.data, ub, out=processed_matrix.data)

                # Optionally log how much was clipped
                clipped_frac = float((processed_matrix.data == ub).sum()) / float(processed_matrix.nnz)
                logger.debug(f"Winsorized upper tail at {winsor_p}th pct: "
                             f"{clipped_frac:.4%} of nonzeros clipped to {float(ub):.3e}")

                # Remove any explicit zeros that may have appeared
                processed_matrix.eliminate_zeros()

                # Re-normalize rows to restore unit length (spherical geometry)
                processed_matrix = self._l2_normalize_rows(processed_matrix)

        self._ensure_finite(processed_matrix.data, "processed_matrix after winsorizing")

        # Step 2: Global Scaling by Frobenius norm
        frobenius_norm_sq = float((processed_matrix.data ** 2).sum())
        if frobenius_norm_sq <= 1e-10:
            raise ValueError("Matrix has zero Frobenius norm after processing.")
        alpha = 1.0 / cupy.sqrt(frobenius_norm_sq)
        scaled_matrix = processed_matrix * alpha
        self._ensure_finite(scaled_matrix.data, "scaled_matrix after global scaling")
        logger.debug(f"Applied global scaling alpha = {alpha:.2e}.")

        # Step 3: Define Augmented Matrix Operator
        def _augmented_matvec(vector_z):
            vector_z = cupy.asarray(vector_z, dtype=dtype)
            vector_u, vector_v = vector_z[:n_samples], vector_z[n_samples:]
            result_upper = scaled_matrix @ vector_v
            result_lower = scaled_matrix.T @ vector_u
            final_result = cupy.concatenate([result_upper, result_lower])
            if not cupy.isfinite(final_result).all():
                max_val = float(cupy.abs(final_result[cupy.isfinite(final_result)]).max()) if cupy.isfinite(final_result).any() else 0
                logger.error(
                    f"Non-finite values in final result. "
                    f"Max finite value: {max_val:.2e}, "
                    f"Num NaN: {int(cupy.isnan(final_result).sum())}, "
                    f"Num Inf: {int(cupy.isinf(final_result).sum())}"
                )
                raise FloatingPointError("NaN or Inf generated during matrix-vector product.")
            # Check for values that are getting too large (early warning)
            max_abs_val = float(cupy.abs(final_result).max())
            if max_abs_val > 1e6:
                logger.warning(f"Large values detected in matvec result: max={max_abs_val:.2e}")
            return final_result

        augmented_operator = LinearOperator(
            shape=(n_aug, n_aug), matvec=_augmented_matvec, dtype=dtype
        )

        # Step 4: Configure and Run Eigsh Solver
        if ncv is None:
            max_ncv = n_aug - 1
            desired_ncv = max(2 * n_components + 1, min(n_components + 50, n_components * 2))
            ncv = min(max_ncv, desired_ncv)
        logger.debug(f"Using ncv={ncv} for augmented matrix eigsh.")

        try:
            eigenvalues, eigenvectors = self._run_eigsh_with_restarts(
                augmented_matrix_operator=augmented_operator,
                n_components=n_components,
                n_augmented_dims=n_aug,
                ncv=ncv,
                tol=tol,
                maxiter=maxiter
            )
        except (FloatingPointError, Exception) as e:
            logger.error(f"eigsh fallback failed with error: {e}", exc_info=True)
            raise RuntimeError("SVD computation failed even with the robust fallback method.") from e

        # Step 5: Process and Validate Results
        if not cupy.isfinite(eigenvalues).all() or not cupy.isfinite(eigenvectors).all():
            raise RuntimeError("eigsh fallback returned non-finite (NaN or Inf) eigenpairs.")

        descending_indices = cupy.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[descending_indices]
        eigenvectors = eigenvectors[:, descending_indices]
        
        singular_values = eigenvalues / alpha

        V_pruned = eigenvectors[n_samples:, :]
        col_norms = cupy.linalg.norm(V_pruned, axis=0, keepdims=True)
        V_pruned = V_pruned / cupy.maximum(col_norms, 1e-12)

        Z = processed_matrix @ V_pruned
        s_inv = 1.0 / cupy.maximum(singular_values, 1e-18)
        U = Z * s_inv[None, :]

        V_full = cupy.zeros((original_n_features, n_components), dtype=dtype)
        V_full[kept_column_indices, :] = V_pruned
        V_transpose = V_full.T

        logger.info(
            f"Augmented matrix SVD complete. Singular values range: "
            f"[{float(singular_values[-1]):.2e}, {float(singular_values[0]):.2e}]"
        )
        # The U matrix from this method is not returned directly but is used
        # to ensure the decomposition is valid. The final transform will still
        # be X @ V.T.
        return U, singular_values, V_transpose

    def fit(self, X: csr_matrix, y=None) -> 'GPUTruncatedSVD':
        """
        Fit the model to the data X.

        This is a convenience method that calls fit_transform and discards the
        transformed matrix, returning the fitted estimator.

        Args:
            X: Input data as a CuPy CSR sparse matrix.
            y: Ignored. Present for API compatibility.

        Returns:
            The fitted GPUTruncatedSVD instance.
        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X: csr_matrix, y=None) -> cupy.ndarray:
        """
        Fit the model and transform data to a lower-dimensional space.

        This method centralizes the SVD computation. It decomposes the input
        matrix X, stores the components, and returns the transformed data.

        Args:
            X: Input sparse matrix with shape (n_samples, n_features).
            y: Ignored. Present for API compatibility.

        Returns:
            Dense array of transformed data with shape (n_samples, n_components).
        """
        logger.debug("Starting fit_transform on sparse matrix.")
        prepared_matrix = self._prepare_input_matrix(X)
        n_samples, n_features = prepared_matrix.shape

        max_components = min(n_samples, n_features) - 1
        if self.n_components > max_components:
            raise ValueError(
                f"n_components ({self.n_components}) must be <= "
                f"min(n_samples, n_features) - 1 = {max_components}"
            )

        logger.info(
            f"Fitting GPUTruncatedSVD on sparse matrix with shape {prepared_matrix.shape}, "
            f"density={prepared_matrix.nnz / (n_samples * n_features):.4f}"
        )

        try:
            U, s, V_T = svds(
                prepared_matrix,
                k=self.n_components,
                return_singular_vectors=True,
                **self.svds_kwargs
            )
            # Check for svds convergence failure (often returns zeros)
            if cupy.all(s == 0):
                raise RuntimeError("svds solver failed to converge (returned all zeros).")
        except Exception as e:
            logger.warning(f"Standard svds failed: {e}. Attempting fallback.")
            fallback_maxiter = min(n_samples * 10, 50_000)
            U, s, V_T = self._eigsh_augmented_tsvd(
                prepared_matrix,
                n_components=self.n_components,
                tol=self.svds_kwargs.get('tol', 1.0e-8),
                maxiter=self.svds_kwargs.get('maxiter', fallback_maxiter),
                ncv=self.svds_kwargs.get('ncv')
            )

        # Sort results in descending order of singular values
        descending_indices = cupy.argsort(s)[::-1]
        self.singular_values_ = s[descending_indices]
        self.components_ = V_T[descending_indices, :]
        U_sorted = U[:, descending_indices]

        # --- Calculate Explained Variance ---
        # This is the mathematically correct way to calculate total variance for
        # a data matrix, accounting for the mean of each feature.
        sum_sq = float((prepared_matrix.data ** 2).sum())
        col_sum = prepared_matrix.sum(axis=0)
        mu = cupy.asarray(col_sum).ravel().astype(cupy.float64, copy=False) / float(n_samples)
        mu_sq_norm = float(cupy.inner(mu, mu))
        
        numerator = sum_sq - n_samples * mu_sq_norm
        numerator = max(numerator, 0.0) # Guard against floating point inaccuracies
        total_variance = numerator / max(n_samples - 1, 1)
        
        self.explained_variance_ = (self.singular_values_ ** 2) / max(n_samples - 1, 1)
        
        eps = 1e-18
        if total_variance > eps:
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        else:
            self.explained_variance_ratio_ = cupy.zeros_like(self.explained_variance_)

        cumulative_variance = float(cupy.sum(self.explained_variance_ratio_))
        logger.info(
            f"GPUTruncatedSVD fit complete. "
            f"Top singular value: {float(self.singular_values_[0]):.6e}, "
            f"Total explained variance: {cumulative_variance:.4f}"
        )
        if cumulative_variance < 0.8:
            logger.warning(
                f"Only {cumulative_variance:.2%} of variance explained. "
                f"Consider increasing n_components (current: {self.n_components})."
            )

        # The transformed matrix is U * Sigma
        transformed_matrix = U_sorted * self.singular_values_
        
        logger.debug(
            f"fit_transform complete. Output shape: {transformed_matrix.shape}, "
            f"dtype: {transformed_matrix.dtype}"
        )
        return transformed_matrix

    def transform(self, X: csr_matrix) -> cupy.ndarray:
        """
        Transform data using the fitted model.
        
        Projects the input data onto the principal components learned during fit().
        
        Args:
            X: Input sparse matrix with shape (n_samples, n_features).
               Must have the same n_features as the data used in fit().
        
        Returns:
            Dense array of transformed data with shape (n_samples, n_components).
        """
        if self.components_ is None:
            raise RuntimeError(
                "GPUTruncatedSVD has not been fitted. Call fit() or fit_transform() first."
            )
        
        prepared_matrix = self._prepare_input_matrix(X)
        n_samples, n_features = prepared_matrix.shape
        expected_features = self.components_.shape[1]
        
        if n_features != expected_features:
            raise ValueError(
                f"Input has {n_features} features, but GPUTruncatedSVD "
                f"was fitted with {expected_features} features."
            )

        logger.debug(f"Transforming sparse matrix with shape {prepared_matrix.shape}")
        
        # Project data onto principal components: X @ V
        transformed_matrix = prepared_matrix @ self.components_.T
        
        return transformed_matrix
    
    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this estimator.
        
        Returns:
            Dictionary of parameter names to values.
        """
        return {'n_components': self.n_components, **self.svds_kwargs}
    
    def set_params(self, **params) -> 'GPUTruncatedSVD':
        """
        Set parameters for this estimator.
        
        Args:
            **params: Parameter names and values.
            
        Returns:
            self
        """
        for key, value in params.items():
            if key == 'n_components':
                self.n_components = value
            else:
                self.svds_kwargs[key] = value
        return self