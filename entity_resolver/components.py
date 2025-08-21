# entity_resolver/components.py
"""
GPU-accelerated Truncated SVD for sparse matrix analysis.

This module provides the `GPUTruncatedSVD` class, a high-performance tool for
dimensionality reduction on sparse datasets, designed to run on NVIDIA GPUs
via the CuPy library. It serves as a powerful alternative to CPU-based methods,
offering significant speed advantages for large-scale data.

Core Features:
- **Scikit-learn Compatible API**: Implements the familiar `fit`, `transform`,
  and `fit_transform` methods for easy integration into existing ML pipelines.
- **Robust Two-Stage Solver**: Employs a fast `svds`-based solver for typical
  cases and automatically falls back to a more stable, regularized eigensolver
  (`eigsh`) for numerically challenging or ill-conditioned matrices.
- **Advanced Pre-processing**: The fallback mechanism includes a comprehensive
  pre-processing pipeline (pruning, winsorizing, normalization) to condition
  the data, maximizing the likelihood of a successful decomposition.

Primary Use Cases:
This class is particularly effective for tasks involving high-dimensional
sparse feature spaces, such as:
- Latent Semantic Analysis (LSA) on TF-IDF matrices from text corpora.
- Feature extraction and noise reduction in large datasets.
- Collaborative filtering and recommendation systems.

Example:
    >>> import cupy
    >>> from cupyx.scipy.sparse import csr_matrix
    >>> # Assume X is a large sparse CuPy matrix
    >>> X = csr_matrix(cupy.random.rand(5000, 10000))
    >>> svd = GPUTruncatedSVD(svds_kwargs={'n_components': 100})
    >>> X_reduced = svd.fit_transform(X)
    >>> print(X_reduced.shape)
    (5000, 100)
"""

import logging
from typing import Optional, Dict, Any, Tuple

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
            fallback_config: Optional[Dict[str, Any]] = None, 
            svds_kwargs: Optional[Dict[str, Any]] = None
        ):
        """
        Initialize the GPUTruncatedSVD transformer.

        Args:
            fallback_config:
                A dictionary of parameters to control the robust SVD fallback mechanism.
            svds_kwargs:
                Additional keyword arguments to pass to the primary `svds` solver.
                Common arguments include 'tol' for tolerance and 'maxiter' for
                maximum iterations. `n_components` should also be passed here.
        """
        self.svds_kwargs = svds_kwargs if svds_kwargs is not None else {}
        self.n_components = self.svds_kwargs.pop('n_components', 256)

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

    def fit(self, input_matrix: spmatrix, target_variable: Optional[Any] = None) -> 'GPUTruncatedSVD':
        """
        Fits the SVD model to the data matrix.

        This method is a convenience wrapper around `fit_transform`. It performs the
        SVD computation on the input matrix, stores the resulting components
        (`singular_values_`, `components_`, etc.) in the instance, and then returns
        the fitted instance itself. This is standard practice for scikit-learn
        compatible estimators, allowing for method chaining (e.g., `svd.fit(X).transform(X)`).

        Args:
            input_matrix (spmatrix): The sparse matrix of shape (n_samples, n_features)
                                     to be decomposed.
            target_variable (Optional[Any]): Ignored. This parameter is included for API
                                             compatibility.

        Returns:
            The fitted GPUTruncatedSVD instance.
        """
        # The core logic is delegated to `fit_transform`. We call it for its side effect:
        # fitting the model and populating the instance attributes. The transformed
        # matrix it returns is discarded here.
        self.fit_transform(input_matrix)

        # Return the instance itself to allow for method chaining.
        return self
    
    def transform(self, input_matrix: spmatrix) -> cupy.ndarray:
        """
        Transforms a matrix using the already-fitted SVD components.

        This method projects the input data onto the principal component space that was
        learned during the `fit` or `fit_transform` stage. The mathematical operation is
        `X_transformed = X @ V`, where V is the matrix of right singular vectors
        (self.components_.T). The result is a new matrix with `n_components` features.

        Args:
            input_matrix: The sparse matrix (n_samples, n_features) to be transformed.

        Returns:
            A CuPy ndarray of shape (n_samples, n_components) representing the data in the
            reduced-dimensional space.

        Raises:
            RuntimeError: If the method is called before the SVD model has been fitted.
            ValueError: If the number of features in the input matrix does not match the
                        number of features the model was trained on.
        """
        # First, check if the model has been fitted. The `components_` attribute is only
        # populated after a successful fit. Calling transform before fit is a logical error.
        if self.components_ is None:
            raise RuntimeError(
                "This SVD instance has not been fitted yet. Call 'fit' or 'fit_transform' first."
            )

        # We reuse the same preparation logic to ensure the input matrix is clean
        # (correct dtype, no NaNs, etc.) before the multiplication. Note that the
        # `n_components` check inside `_prepare_input_matrix` is based on the shape of
        # this new matrix, not the one used for fitting, which is correct behavior.
        prepared_matrix = self._prepare_input_matrix(input_matrix)

        # A critical validation step: the number of columns (features) in the matrix to be
        # transformed must exactly match the number of columns in the matrix used to fit
        # the model. The dimensions of `self.components_` are (n_components, n_features).
        if prepared_matrix.shape[1] != self.components_.shape[1]:
            raise ValueError(
                f"Input matrix has {prepared_matrix.shape[1]} features, but the model "
                f"was fitted with {self.components_.shape[1]} features."
            )

        logger.debug(f"Transforming matrix of shape {input_matrix.shape}")
        # The core transformation is a matrix multiplication. We multiply the prepared input
        # matrix by the transpose of the components matrix.
        # Shape: (n_samples, n_features) @ (n_features, n_components) -> (n_samples, n_components)
        return prepared_matrix @ self.components_.T

    def fit_transform(self, input_matrix: spmatrix, target_variable: Optional[Any] = None) -> cupy.ndarray:
        """
        Fits the SVD model to the input data and applies dimensionality reduction.

        This method serves as the primary entry point for the SVD computation. It follows a robust
        two-stage process:
        1.  It first attempts to use the fast and efficient `svds` solver, which is suitable for
            well-conditioned matrices.
        2.  If `svds` fails to converge or encounters numerical issues, it triggers a fallback
            mechanism. This robust path employs a series of matrix conditioning steps
            (pruning, winsorizing, scaling) and then uses a regularized, augmented matrix
            with the `eigsh` (eigensolver for Hermitian matrices) to ensure a stable solution.

        Args:
            input_matrix (spmatrix): The sparse matrix of shape (n_samples, n_features) to be
                                     decomposed.
            target_variable (Optional[Any]): Ignored. This parameter is included for API
                                             compatibility with scikit-learn pipelines.

        Returns:
            cupy.ndarray: The transformed matrix of shape (n_samples, n_components) after
                          dimensionality reduction.
        """
        logger.info(f"Starting fit_transform for GPUTruncatedSVD with n_components={self.n_components}")

        # --- Step 1: Prepare and Validate the Input Matrix ---
        cleaned_input_matrix = self._prepare_input_matrix(input_matrix)
        number_of_samples, number_of_features = cleaned_input_matrix.shape
        matrix_for_decomposition = cleaned_input_matrix
        kept_row_indices = cupy.arange(number_of_samples)

        # --- Step 2: Attempt Standard SVD with the 'svds' Solver ---
        try:
            logger.debug("Attempting the standard 'svds' solver for SVD.")
            _, singular_values, right_singular_vectors_transposed = svds(
                cleaned_input_matrix, k=self.n_components, **self.svds_kwargs
            )
            if cupy.all(singular_values == 0):
                raise RuntimeError("The 'svds' solver returned all-zero singular values, indicating non-convergence.")
            logger.info("Standard 'svds' solver completed successfully.")

        # --- Step 3: Fallback to Robust Augmented Eigendecomposition Method ---
        except Exception as svds_exception:
            logger.warning(f"Standard 'svds' solver failed: {svds_exception}. Falling back to the robust 'eigsh' method.")

            # --- Step 3a: Pre-process the Matrix for Enhanced Stability ---
            pruned_matrix, kept_row_indices, kept_column_indices = prune_sparse_matrix(
                cleaned_input_matrix.astype(self.fallback_dtype),
                min_row_sum=self.prune_min_row_sum,
                min_df=self.prune_min_df,
                max_df_ratio=self.prune_max_df_ratio,
                energy_cutoff_ratio=self.prune_energy_cutoff,
            )
            finite_matrix = ensure_finite_matrix(pruned_matrix, replace_non_finite=True, copy=False)
            winsorized_matrix = winsorize_matrix(finite_matrix, limits=self.winsorize_limits, copy=False)
            scaled_matrix, frobenius_scale_factor = scale_by_frobenius_norm(winsorized_matrix, copy=False)
            normalized_row_matrix = normalize_rows(scaled_matrix, copy=False)
            matrix_for_decomposition = normalized_row_matrix
            n_pruned_samples, n_pruned_features = normalized_row_matrix.shape
            augmented_matrix_dimension = n_pruned_samples + n_pruned_features

            # --- Step 3b: Define the Regularized Augmented Matrix Operator ---
            alpha_regularization = 1e-3
            regularization_term = self.calculate_initial_regularization(normalized_row_matrix, alpha=alpha_regularization)

            # Create a LinearOperator. This object encapsulates the matrix-vector product function.
            # A lambda function is used here to pass the required contextual arguments
            # (the matrix, regularization term, etc.) to our class method.
            augmented_matrix_operator = LinearOperator(
                shape=(augmented_matrix_dimension, augmented_matrix_dimension),
                matvec=lambda vec: self._regularized_augmented_matvec(
                    vec,
                    normalized_row_matrix=normalized_row_matrix,
                    regularization_term=regularization_term,
                    n_pruned_samples=n_pruned_samples
                ),
                dtype=self.fallback_dtype
            )

            # --- Step 3c: Run `eigsh` with a Restart Mechanism ---
            eigsh_parameters: Dict[str, Any] = {
                'tol': self.svds_kwargs.get('tol', 1e-8),
                'maxiter': self.svds_kwargs.get('maxiter', 20000),
                'ncv': self.svds_kwargs.get('ncv', min(max(2 * self.n_components + 1, 20), augmented_matrix_dimension - 1))
            }

            eigenvalues, eigenvectors = None, None
            for attempt in range(self.eigsh_restarts):
                try:
                    logger.debug(f"Starting eigsh attempt {attempt + 1}/{self.eigsh_restarts} with params: {eigsh_parameters}")
                    initial_vector = create_initial_vector(augmented_matrix_dimension, dtype=self.fallback_dtype, seed=attempt)
                    eigenvalues, eigenvectors = eigsh(
                        augmented_matrix_operator, k=self.n_components, v0=initial_vector, which='LA', **eigsh_parameters
                    )
                    logger.info("The 'eigsh' solver completed successfully.")
                    break
                except Exception as eigsh_exception:
                    logger.warning(f"Eigsh attempt {attempt + 1} failed: {eigsh_exception}")
                    if attempt < self.eigsh_restarts - 1:
                        regularization_term *= 10
                        eigsh_parameters = self._get_eigsh_restart_params(eigsh_parameters, augmented_matrix_dimension)
                    else:
                        raise RuntimeError("The 'eigsh' fallback solver failed after all restart attempts.") from eigsh_exception

            # --- Step 3d: Reconstruct SVD Components from Eigenpairs ---
            adjusted_eigenvalues = cupy.maximum(eigenvalues - regularization_term, 0.0)
            singular_values = cupy.sqrt(adjusted_eigenvalues) / frobenius_scale_factor
            right_singular_vectors_pruned = eigenvectors[n_pruned_samples:, :]
            right_singular_vectors_pruned /= cupy.linalg.norm(right_singular_vectors_pruned, axis=0, keepdims=True)
            right_singular_vectors_full = cupy.zeros((number_of_features, self.n_components), dtype=self.fallback_dtype)
            right_singular_vectors_full[kept_column_indices, :] = right_singular_vectors_pruned
            right_singular_vectors_transposed = right_singular_vectors_full.T

        # --- Step 4: Finalize and Store SVD Results ---
        descending_sort_indices = cupy.argsort(singular_values)[::-1]
        self.singular_values_ = singular_values[descending_sort_indices]
        self.components_ = right_singular_vectors_transposed[descending_sort_indices, :]
        self._calculate_variance_explained(matrix_for_decomposition, matrix_for_decomposition.shape[0])
        logger.info(f"GPUTruncatedSVD fit complete. Top singular value: {float(self.singular_values_[0]):.4e}")

        # --- Step 5: Transform the Data into the Lower-Dimensional Space ---
        transformed_matrix = self.transform(cleaned_input_matrix)

        if kept_row_indices.size < number_of_samples:
            logger.debug("Re-inserting zero-rows into the output for the pruned samples.")
            full_transformed_matrix = cupy.zeros((number_of_samples, self.n_components), dtype=transformed_matrix.dtype)
            full_transformed_matrix[kept_row_indices] = transformed_matrix
            transformed_matrix = full_transformed_matrix

        return transformed_matrix

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

    def _prepare_input_matrix(self, input_matrix: spmatrix) -> csr_matrix:
        """
        Validates, cleans, and standardizes the input sparse matrix for SVD.

        This essential helper method performs several critical checks and conversions to
        ensure the input matrix is in an optimal state for the SVD algorithms. The
        pipeline is ordered to prevent unnecessary operations and ensure correctness.

        Pipeline Steps:
        1.  **Ensure Finiteness**: Replaces any non-finite values (NaN, Inf) with zeros.
            This is done first as it can affect the matrix sparsity and values.
        2.  **Convert to CSR Format**: Converts the matrix to Compressed Sparse Row (CSR)
            format. This format is highly efficient for the row-based operations and
            matrix-vector products frequently used in iterative SVD solvers.
        3.  **Ensure Floating-Point DType**: Casts the matrix to a floating-point data
            type if it isn't already. SVD is a numerical algorithm that requires floats.
        4.  **Validate `n_components`**: Checks that the number of requested components is
            mathematically feasible given the matrix dimensions.

        Args:
            input_matrix (spmatrix): The raw input sparse matrix from the user.

        Returns:
            csr_matrix: A clean, validated CSR matrix ready for SVD computation.

        Raises:
            ValueError: If `self.n_components` is not strictly less than the smaller
                        dimension of the input matrix.
        """
        logger.debug("Preparing and validating the input matrix.")

        # Step 1: Handle non-finite values (NaN, infinity). This is the first and most
        # critical cleaning step. A copy is made to avoid modifying the user's original data.
        prepared_matrix = ensure_finite_matrix(input_matrix, replace_non_finite=True, copy=True)

        # Step 2: Ensure the matrix is in CSR format. CSR is optimized for fast
        # matrix-vector products and row slicing, which are core operations in the
        # Lanczos-based algorithms used by `svds` and `eigsh`.
        if not isspmatrix_csr(prepared_matrix):
            logger.debug("Input matrix is not in CSR format. Converting.")
            prepared_matrix = prepared_matrix.tocsr(copy=False)

        # Step 3: Ensure a floating-point dtype for all subsequent numerical calculations.
        # SVD algorithms are fundamentally based on floating-point arithmetic.
        if not cupy.issubdtype(prepared_matrix.dtype, cupy.floating):
            logger.debug(f"Input matrix dtype is not float. Casting to {self.fallback_dtype}.")
            prepared_matrix = prepared_matrix.astype(self.fallback_dtype, copy=False)

        # Step 4: Validate shape constraints for SVD. The number of singular values to
        # compute (`k`, or `n_components`) must be strictly less than the minimum of the
        # number of samples and features.
        number_of_samples, number_of_features = prepared_matrix.shape
        if self.n_components >= min(number_of_samples, number_of_features):
            raise ValueError(
                f"n_components ({self.n_components}) must be < min(n_samples, n_features) "
                f"which is {min(number_of_samples, number_of_features)} for the given matrix."
            )

        logger.debug("Input matrix preparation complete.")
        return prepared_matrix

    def _get_eigsh_restart_params(self, previous_parameters: Dict[str, Any], augmented_matrix_dimension: int) -> Dict[str, Any]:
        """
        Generates a tweaked set of parameters for an `eigsh` restart attempt.

        When an `eigsh` computation fails to converge, this method is called to provide
        a new, slightly more relaxed set of parameters. The strategy is to broaden the
        conditions for convergence in a controlled manner to increase the likelihood of
        finding a solution in the next attempt.

        Args:
            previous_parameters: A dictionary containing the `ncv`, `tol`, and `maxiter`
                                 from the previously failed attempt.
            augmented_matrix_dimension: The full dimension of the augmented matrix, which is
                                        used to set a hard upper limit on `ncv`.

        Returns:
            A dictionary containing the new, relaxed parameters for the next `eigsh` call.
        """
        # Strategy 1: Adjust the number of Lanczos vectors (`ncv`).
        # We slightly reduce `ncv` from the previous attempt. A smaller subspace can
        # sometimes help the algorithm stabilize by avoiding difficult-to-converge directions.
        # It's capped to be at least `n_components + 2` and no more than the matrix dimension minus one.
        updated_lanczos_vectors = min(
            max(self.n_components + 2, int(0.9 * previous_parameters['ncv'])),
            augmented_matrix_dimension - 1
        )

        # Strategy 2: Relax the convergence tolerance (`tol`).
        # We make the tolerance 10 times larger (less strict), making it easier for the
        # algorithm to satisfy the convergence criteria. A floor of 1e-7 is set to
        # prevent it from becoming unreasonably large.
        relaxed_tolerance = max(previous_parameters['tol'] * 10, 1e-7)

        # Strategy 3: Increase the maximum number of iterations (`maxiter`).
        # We give the algorithm 50% more iterations to find a solution, in case the
        # previous attempt was simply cut short before it could converge.
        increased_max_iterations = int(previous_parameters['maxiter'] * 1.5)

        logger.warning(
            f"Restarting eigsh with new params: ncv={updated_lanczos_vectors}, "
            f"tol={relaxed_tolerance:.1e}, maxiter={increased_max_iterations}"
        )

        return {
            'ncv': updated_lanczos_vectors,
            'tol': relaxed_tolerance,
            'maxiter': increased_max_iterations
        }
    
    def _regularized_augmented_matvec(
        self,
        vector_z: cupy.ndarray,
        normalized_row_matrix: spmatrix,
        regularization_term: float,
        n_pruned_samples: int
    ) -> cupy.ndarray:
        """
        Computes the matrix-vector product for the regularized augmented matrix.

        This operation is the core of the iterative `eigsh` solver. It computes B @ z, where
        B is the augmented matrix and z is partitioned into [u, v]. The product is:
        - Top block:    λ*u + A*v
        - Bottom block: A.T*u + λ*v

        Args:
            vector_z (cupy.ndarray): The input vector to be multiplied.
            normalized_row_matrix (spmatrix): The conditioned matrix 'A' for the operation.
            regularization_term (float): The regularization value 'λ'.
            n_pruned_samples (int): The number of rows in 'A', used for partitioning `vector_z`.

        Returns:
            cupy.ndarray: The result of the matrix-vector product.
        """
        # Explicitly cast the regularization term to the vector's dtype for numerical safety.
        regularization_lambda = vector_z.dtype.type(regularization_term)

        # Split the input vector `z` into its two components `u` and `v`.
        vector_u_component, vector_v_component = vector_z[:n_pruned_samples], vector_z[n_pruned_samples:]

        # Compute the matrix-vector products for each block of the augmented matrix.
        top_block_result = regularization_lambda * vector_u_component + normalized_row_matrix @ vector_v_component
        bottom_block_result = normalized_row_matrix.T @ vector_u_component + regularization_lambda * vector_v_component

        # Concatenate the results to form the final output vector.
        final_result_vector = cupy.concatenate([top_block_result, bottom_block_result])

        # Sanity check for numerical issues like NaN or Inf, which can derail the solver.
        if not cupy.isfinite(final_result_vector).all():
            max_finite_val = float(cupy.abs(final_result_vector[cupy.isfinite(final_result_vector)]).max()) if cupy.isfinite(final_result_vector).any() else 0
            logger.error(
                f"Non-finite values detected in regularized matvec result. Max finite value: {max_finite_val:.2e}, "
                f"NaN count: {int(cupy.isnan(final_result_vector).sum())}, Inf count: {int(cupy.isinf(final_result_vector).sum())}, "
                f"Regularization value: {regularization_term:.2e}"
            )
            raise FloatingPointError("NaN or Inf generated during the regularized matrix-vector product.")

        return final_result_vector

    def _calculate_variance_explained(self, decomposed_matrix: csr_matrix, n_samples: int) -> None:
        """
        Calculate and store the variance explained by the SVD components.

        Unlike PCA, Truncated SVD on sparse matrices does not center the data.
        Consequently, the total variance is not the sum of variances of each feature.
        Instead, it is defined as the squared Frobenius norm of the matrix, divided by
        the degrees of freedom (n_samples - 1).

        This method calculates two types of variance ratios:
        1. `explained_variance_ratio_`: A relative measure where each component's
           variance is divided by the sum of variances of *only the computed components*.
           This ensures the ratio sums to 1.0 and is consistent with scikit-learn.
        2. `true_explained_variance_ratio_`: An absolute measure where the sum of
           component variances is divided by the total variance of the entire
           (potentially processed) matrix. This gives a more realistic picture of
           how much of the total information is captured by the `n_components`.

        Args:
            decomposed_matrix: The matrix that was actually used for the SVD computation
                               (this may be a pruned/processed version of the original).
            n_samples: The number of samples in the `decomposed_matrix`.
        """
        # For sparse matrices in TruncatedSVD, we DON'T center the data
        # Total variance is simply ||X||_F^2 / (n-1)
        # where ||X||_F is the Frobenius norm
        
        # Degrees of freedom is n_samples - 1. Safeguard against n_samples <= 1.
        degrees_of_freedom = max(n_samples - 1, 1)

        # The total variance is defined as ||X||_F^2 / (n - 1), where ||X||_F is the
        # Frobenius norm. For a sparse matrix, the squared norm is the sum of its
        # squared non-zero elements.
        frobenius_norm_squared = float((decomposed_matrix.data ** 2).sum())

        # Total variance estimate
        # Note: This is the total variance of the matrix we actually decomposed
        # If we pruned columns or normalized, this reflects that processed matrix
        total_variance_of_decomposed_matrix = frobenius_norm_squared / degrees_of_freedom
        self.total_variance_ = total_variance_of_decomposed_matrix
        
        # Component variance: s_i^2 / (n-1)
        # The variance explained by each individual component is its corresponding
        # singular value squared, divided by the degrees of freedom.
        self.explained_variance_ = (self.singular_values_ ** 2) / degrees_of_freedom
        
        # For Truncated SVD, we can only explain variance up to the sum of
        # the k singular values we computed. The true total variance would require
        # ALL singular values, which we don't have.
        
        # Two approaches for explained variance ratio:
        
        # Approach 1: Conservative (what sklearn does)
        # Use the sum of computed variances as denominator
        # This ensures ratios sum to 1.0 but doesn't reflect true % of total variance
        sum_of_component_variances = float(self.explained_variance_.sum())
        
        # Use Approach 1 (conservative, always sums to ≤ 1.0)
        epsilon = 1e-18  # A small constant to avoid division by zero.

        # `explained_variance_ratio_` (Relative):
        # This is the standard scikit-learn approach. It normalizes by the sum of the
        # variances we found, ensuring the ratios for the found components sum to 1.0.
        # This is useful for understanding the relative importance of the components.
        if sum_of_component_variances > epsilon:
            self.explained_variance_ratio_ = self.explained_variance_ / sum_of_component_variances
        else:
            self.explained_variance_ratio_ = cupy.zeros_like(self.explained_variance_)
        
        # Approach 2: Approximate total variance from Frobenius norm
        # This can give ratios > 1 if we're missing significant singular values
        # but is more truthful about what % of total variance we captured

        # `true_explained_variance_ratio_` (Absolute):
        # This provides a more insightful diagnostic, showing what fraction of the
        # *total* matrix variance is captured by our selected components.
        # This can be > 1.0 if matrix was poorly conditioned, but usually should be < 1.0
        self.true_explained_variance_ratio_ = sum_of_component_variances / max(total_variance_of_decomposed_matrix, epsilon)
        
        # Log diagnostic information
        cumulative_relative_variance = float(self.explained_variance_ratio_.sum())
        true_variance_captured = float(self.true_explained_variance_ratio_)

        logger.info(
            f"Variance explained (relative): {cumulative_relative_variance:.4f}, "
            f"Variance captured (absolute): {true_variance_captured:.4f}"
        )

        # A ratio > 1.0 is a red flag for numerical instability, suggesting the sum of
        # squared singular values found is greater than the total squared norm of the matrix.
        if true_variance_captured > 1.0:
            logger.warning(
                f"Captured variance ratio > 1.0 ({true_variance_captured:.4f}). "
                "This suggests numerical instability in the matrix. "
                "The `explained_variance_ratio_` has been normalized to sum to 1.0."
            )
        # A low ratio indicates that more components may be needed to represent the data.
        elif true_variance_captured < 0.5:
            logger.warning(
                f"Only {true_variance_captured:.2%} of true variance captured. "
                f"Consider increasing n_components (current: {self.n_components})."
            )

    @staticmethod
    def calculate_initial_regularization(
        matrix: spmatrix,
        alpha: float = 1e-3
    ) -> float:
        """
        Calculates an initial regularization term based on the matrix's properties.

        A good regularization term is crucial for the stability of the `eigsh` solver.
        This method estimates a suitable value by relating it to the average singular
        value of the matrix. For a matrix that has been pre-processed (e.g., via
        Frobenius and row normalization), this provides a data-driven heuristic that
        scales appropriately with the matrix's content.

        The estimation follows this logic:
        1.  Calculate the squared Frobenius norm (||A||_F^2), which is the sum of
            the squared singular values.
        2.  Estimate the average singular value: σ_avg ≈ ||A||_F / sqrt(rank). We
            approximate the rank with min(n_samples, n_features).
        3.  The regularization term (lambda) is set as a small fraction (`alpha`) of
            this estimated average singular value.
        4.  The result is clamped to a reasonable range to prevent it from being
            too small (causing instability) or too large (over-regularizing).

        Args:
            matrix: The preprocessed sparse matrix for which to calculate regularization.
            alpha: A coefficient determining the fraction of the average singular value
                   to use as the regularization term.

        Returns:
            A float value for the initial regularization term.
        """
        number_of_rows, number_of_columns = matrix.shape

        # Calculate the squared Frobenius norm. For a sparse matrix, this is simply the
        # sum of the squares of its non-zero elements.
        frobenius_norm_squared = float((matrix.multiply(matrix)).sum())
        # To prevent division by zero in edge cases with all-zero matrices, ensure the
        # norm is at least the smallest positive number for the given dtype.
        frobenius_norm_squared = max(frobenius_norm_squared, cupy.finfo(matrix.dtype).tiny)

        # Estimate the average singular value. The Frobenius norm is the square root of the
        # sum of squared singular values. By dividing by the square root of the matrix
        # rank (approximated by min(rows, cols)), we get a rough estimate of the average σ.
        estimated_rank = max(min(number_of_rows, number_of_columns), 1)
        estimated_average_singular_value = (frobenius_norm_squared ** 0.5) / (estimated_rank ** 0.5)

        # Set the initial regularization term as a small fraction of the average singular value.
        regularization_lambda = alpha * estimated_average_singular_value

        # Clamp the regularization term to a safe and reasonable range.
        # Lower bound: A very small number to prevent underflow and maintain numerical stability.
        # Upper bound: A fraction (10%) of the average singular value to avoid overly
        # biasing the results, which would suppress the true smaller singular values.
        lower_bound = 1e-12 if matrix.dtype == cupy.float64 else 1e-7
        upper_bound = 0.1 * estimated_average_singular_value

        # Use standard Python's max() and min() for clamping a scalar float.
        # This is the correct approach and avoids the cupy.clip type error.
        clamped_lambda = float(max(lower_bound, min(regularization_lambda, upper_bound)))

        logger.debug(
            f"Calculated regularization={clamped_lambda:.2e} from ||A||_F="
            f"{frobenius_norm_squared**0.5:.4f}, avg_sigma≈{estimated_average_singular_value:.2e}, "
            f"alpha={alpha}, shape=({number_of_rows}, {number_of_columns})"
        )

        return clamped_lambda