# entity_resolver/components.py
"""
Custom GPU-accelerated components for entity resolution pipeline.

This module provides custom implementations of machine learning components
that are optimized for GPU execution and sparse matrix operations.
"""

import logging

from typing import Tuple
import cupy
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
        if not isinstance(sparse_matrix, cupy.sparse.csr_matrix):
            sparse_matrix = sparse_matrix.tocsr()
            
        # Ensure float64 for high-precision SVD and variance calculations
        if sparse_matrix.dtype != cupy.float64:
            sparse_matrix = sparse_matrix.astype(cupy.float64)
        
        logger.debug(f"Prepared matrix: shape={sparse_matrix.shape}, dtype={sparse_matrix.dtype}")
        return sparse_matrix

    def _eigsh_augmented_tsvd(
            self, 
            sparse_matrix: cupy.sparse.csr_matrix, 
            n_components: int, 
            ncv: int = None, 
            tol: float = 1.0e-8, 
            maxiter: int = 20000
        ) -> Tuple[None, cupy.ndarray, cupy.ndarray]:
        """
        Computes the exact top-k singular values/vectors of the 
        row-L2-normalized input (spherical TF-IDF) via the augmented operator.
        
        Computes SVD of X by finding eigenvalues of the augmented matrix:
            B = [[0,   αX ],
                [αXᵀ, 0  ]]
        where α = 1/‖X‖_F for numerical stability.
        
        The eigenvalues are ±ασ (where σ are singular values of X).
        Using 'LA' returns the positive eigenvalues directly.
        
        Args:
            sparse_matrix: Input CSR matrix (already float64 from _prepare_input_matrix).
            n_components: Number of singular values to compute.
            ncv: Number of Lanczos vectors (must satisfy k+1 < ncv < n_aug).
            tol: Convergence tolerance.
            maxiter: Maximum iterations.
        
        Returns:
            (None, singular_values, V_transpose) matching svds format.
        """
        logger.warning(
            "Standard svds failed. Using exact augmented matrix method (eigsh). "
            "This preserves mathematical exactness while improving stability."
        )
        
        n_samples, n_features = sparse_matrix.shape
        n_aug = n_samples + n_features
        dtype = cupy.float64
        
        # === Step 1: Check finiteness of input ===
        if not cupy.isfinite(sparse_matrix.data).all():
            raise ValueError("Input sparse matrix contains non-finite values")
        
        # === Step 2: Row L2-normalize (unconditional for cosine similarity) ===
        # Create a copy to preserve original
        X_norm = sparse_matrix.copy()
        
        # Compute row norms
        row_norms_sq = cupy.asarray(X_norm.power(2).sum(axis=1)).ravel()
        row_norms = cupy.sqrt(row_norms_sq)
        
        # Find rows with non-zero norm (use small epsilon for numerical safety)
        eps = 1.0e-12
        nonzero_mask = row_norms > eps
        
        if not cupy.any(nonzero_mask):
            raise ValueError("All rows have zero norm - cannot proceed with SVD")
        
        # Apply row normalization
        scale_factors = cupy.zeros(n_samples, dtype=dtype)
        scale_factors[nonzero_mask] = 1.0 / row_norms[nonzero_mask]
        
        # Apply scaling to sparse matrix data via CSR structure
        row_indices = cupy.repeat(cupy.arange(n_samples), cupy.diff(X_norm.indptr))
        X_norm.data *= scale_factors[row_indices]
        
        logger.debug(f"Row-normalized matrix: {cupy.sum(nonzero_mask)}/{n_samples} non-zero rows")
        
        # === Step 3: Global scaling by α = 1/‖X‖_F (always applied) ===
        frobenius_norm_sq = float((X_norm.data ** 2).sum())
        if frobenius_norm_sq <= 0:
            raise ValueError("Matrix has zero Frobenius norm after row normalization")
        
        frobenius_norm = cupy.sqrt(frobenius_norm_sq)
        alpha = 1.0 / frobenius_norm
        
        # Apply global scaling
        X_norm.data *= alpha
        
        logger.debug(f"Applied global scaling α = {alpha:.2e} (‖X‖_F = {frobenius_norm:.2e})")
        
        # === Step 4: Define exact augmented matrix operator ===
        def augmented_matvec(v: cupy.ndarray) -> cupy.ndarray:
            """
            Exact matrix-vector product for B = [[0, αX], [αXᵀ, 0]].
            No diagonal regularization - preserves exactness.
            """
            v = cupy.asarray(v, dtype=dtype)
            
            # Check input validity
            if not cupy.isfinite(v).all():
                raise ValueError("augmented_matvec received non-finite input")
            
            # Split vector
            v_upper = v[:n_samples]
            v_lower = v[n_samples:]
            
            # Exact block matrix multiplication (no regularization!)
            result_upper = X_norm @ v_lower       # αX @ v_lower
            result_lower = X_norm.T @ v_upper     # αXᵀ @ v_upper
            
            result = cupy.concatenate([result_upper, result_lower])
            
            # Verify output is finite
            if not cupy.isfinite(result).all():
                raise ValueError("augmented_matvec produced non-finite output")
            
            return result
        
        # === Step 5: Set ncv with proper bounds ===
        if ncv is None:
            # Must satisfy: k+1 < ncv < n_aug
            min_ncv = n_components + 2  # Strict lower bound
            max_ncv = n_aug - 1          # Strict upper bound
            
            # Use generous ncv for better convergence (you have the GPU memory!)
            desired_ncv = min(
                max(n_components * 2, n_components + 50),  # At least 2x or +50
                n_components + 200,                         # But not excessive
                max_ncv                                      # Must fit bounds
            )
            
            # Ensure bounds are satisfied
            ncv = max(min_ncv, min(desired_ncv, max_ncv))
            
            logger.info(
                f"Set ncv={ncv} (bounds: {min_ncv} < ncv < {max_ncv}, "
                f"k={n_components}, matrix size={n_aug})"
            )
        else:
            # Validate user-provided ncv
            if ncv <= n_components + 1:
                raise ValueError(f"ncv ({ncv}) must be > k+1 ({n_components + 1})")
            if ncv >= n_aug:
                raise ValueError(f"ncv ({ncv}) must be < matrix size ({n_aug})")
        
        # === Step 6: Create augmented operator and solve ===
        augmented_operator = LinearOperator(
            shape=(n_aug, n_aug),
            matvec=augmented_matvec,
            dtype=dtype
        )
        
        try:
            logger.debug(f"Starting eigsh: k={n_components}, ncv={ncv}, which='LA'")
            
            eigenvalues, eigenvectors = eigsh(
                augmented_operator,
                k=n_components,
                which='LA',  # Largest algebraic → positive eigenvalues ασ
                ncv=ncv,
                tol=tol,
                maxiter=maxiter,
                return_eigenvectors=True
            )
            
        except Exception as e:
            logger.error(f"eigsh failed: {e}")
            raise RuntimeError(f"Augmented matrix eigsh failed: {e}") from e
        
        # === Step 7: Validate and process results ===
        # Check finiteness
        if not cupy.isfinite(eigenvalues).all() or not cupy.isfinite(eigenvectors).all():
            raise ValueError("eigsh returned non-finite values")
        
        # Handle complex (shouldn't happen for symmetric real matrix)
        if cupy.iscomplexobj(eigenvalues):
            max_imag = float(cupy.abs(eigenvalues.imag).max())
            if max_imag > 1.0e-10:
                raise ValueError(f"Eigenvalues have significant imaginary parts: {max_imag:.2e}")
            eigenvalues = eigenvalues.real
        
        # Sort by magnitude (descending)
        order = cupy.argsort(cupy.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        # Extract singular values (undo α scaling)
        singular_values = cupy.abs(eigenvalues) / alpha
        
        # Extract right singular vectors V from bottom block
        V = eigenvectors[n_samples:, :]  # Shape: (n_features, n_components)
        
        # === Step 8: Normalize V columns to unit length ===
        col_norms = cupy.linalg.norm(V, axis=0)
        
        # Check for zero norms
        if cupy.any(col_norms < 1.0e-10):
            raise ValueError(f"Found {cupy.sum(col_norms < 1.0e-10)} singular vectors with zero norm")
        
        V_normalized = V / col_norms[cupy.newaxis, :]
        V_transpose = V_normalized.T
        
        # === Step 9: Validation - Check quality of decomposition ===
        # 9a. Check orthonormality of V^T rows
        VVT = V_transpose @ V_transpose.T
        I_approx = VVT
        I_exact = cupy.eye(n_components, dtype=dtype)
        ortho_error = float(cupy.linalg.norm(I_approx - I_exact, 'fro'))
        
        if ortho_error > 1.0e-6:
            logger.warning(f"V^T orthonormality error: {ortho_error:.2e}")
            if ortho_error > 1.0e-3:
                raise ValueError(f"V^T is not orthonormal (error: {ortho_error:.2e})")
        
        # 9b. Check residuals: X^T X v_i ≈ σ_i^2 v_i (sample a few)
        n_check = min(3, n_components)
        logger.debug(f"Checking SVD quality for first {n_check} components...")
        
        # We need X (unscaled but row-normalized) for this check
        X_check = X_norm  
        
        for i in range(n_check):
            v_i = V_normalized[:, i]
            sigma_i = singular_values[i]
            
            # Compute X^T X v_i
            Xv = X_check @ v_i
            XTXv = X_check.T @ Xv
            
            # Expected: σ^2 v_i
            expected = (sigma_i ** 2) * v_i
            
            # Relative residual
            residual_norm = float(cupy.linalg.norm(XTXv - expected))
            expected_norm = float(cupy.linalg.norm(expected))
            
            if expected_norm > 1.0e-10:
                relative_residual = residual_norm / expected_norm
                logger.debug(f"  Component {i}: σ={sigma_i:.2e}, relative residual={relative_residual:.2e}")
                
                if relative_residual > 0.1:  # 10% tolerance
                    logger.warning(f"High residual for component {i}: {relative_residual:.2e}")
        
        # === Step 10: Final checks and logging ===
        # Check all values are finite
        if not cupy.isfinite(singular_values).all():
            raise ValueError("Final singular values contain non-finite values")
        if not cupy.isfinite(V_transpose).all():
            raise ValueError("Final V^T contains non-finite values")
        
        # Check we got meaningful singular values
        if cupy.all(singular_values < 1.0e-10):
            raise ValueError("All singular values are near zero")
        
        # Compute condition number and variance explained
        condition_number = singular_values[0] / (singular_values[-1] + 1.0e-12)
        variance_explained = cupy.cumsum(singular_values ** 2) / cupy.sum(singular_values ** 2)
        n_90_percent = int(cupy.searchsorted(variance_explained, 0.9) + 1)
        
        logger.info(
            f"Augmented SVD successful:\n"
            f"  - Computed {n_components} components\n"
            f"  - Singular values: [{singular_values[-1]:.2e}, {singular_values[0]:.2e}]\n"
            f"  - Condition number: {condition_number:.2e}\n"
            f"  - 90% variance: {n_90_percent} components\n"
            f"  - V^T orthonormality error: {ortho_error:.2e}"
        )
        
        return None, singular_values, V_transpose

    def fit(self, sparse_matrix_csr: cupy.sparse.csr_matrix) -> 'GPUTruncatedSVD':
        """
        Compute singular value decomposition on the sparse matrix.
        
        This method decomposes the input matrix X into three matrices:
        X ≈ U * S * V^T, where U and V are orthogonal and S is diagonal.
        
        Args:
            sparse_matrix_csr: Input data as a CuPy CSR sparse matrix
                              with shape (n_samples, n_features)
        
        Returns:
            Self (fitted GPUTruncatedSVD instance)
            
        Raises:
            ValueError: If n_components >= min(n_samples, n_features)
        """
        n_samples, n_features = sparse_matrix_csr.shape
        
        # Validate n_components
        max_components = min(n_samples, n_features) - 1
        if self.n_components >= max_components:
            raise ValueError(
                f"n_components ({self.n_components}) must be < "
                f"min(n_samples, n_features) - 1 = {max_components}"
            )
        
        logger.info(
            f"Fitting GPUTruncatedSVD on sparse matrix with shape {sparse_matrix_csr.shape}, "
            f"density={sparse_matrix_csr.nnz / (n_samples * n_features):.4f}"
        )

        # --- Total variance of X (calculated efficiently for sparse matrices) ---
        # Frobenius norm squared: sum of squares of all non-zero entries
        sum_sq = float((sparse_matrix_csr.multiply(sparse_matrix_csr)).sum())

        # Column means vector μ (length = n_features)
        col_sum = sparse_matrix_csr.sum(axis=0)
        mu = cupy.asarray(col_sum).ravel().astype(cupy.float64, copy=False) / float(n_samples)

        # Squared L2 norm of the mean vector: ||μ||^2
        mu_sq_norm = float(cupy.inner(mu, mu))

        # Total variance = (||X||_F^2 - n * ||μ||^2) / (n - 1)
        numerator = sum_sq - n_samples * mu_sq_norm
        numerator = max(numerator, 0.0) # Guard against tiny negative from roundoff
        total_variance = numerator / max(n_samples - 1, 1)
        
        # Perform sparse SVD using ARPACK solver
        # Note: cupyx.scipy.sparse.linalg.svds is deterministic (no random_state needed)
        _, singular_values, V_transpose = svds(
            sparse_matrix_csr,
            k=self.n_components,
            return_singular_vectors=True,
            **self.svds_kwargs
        )
        
        # If the solver fails to converge, it may return zeros instead of raising
        # an error. We check for this explicitly and use the stable fallback method if needed.
        if cupy.all(singular_values == 0):
            fallback_maxiter = min(n_samples * 10, 50_000)
            _, singular_values, V_transpose = self._eigsh_augmented_tsvd(
                sparse_matrix_csr,
                n_components=self.n_components,
                tol=self.svds_kwargs.get('tol', 1.0e-8),
                maxiter=self.svds_kwargs.get('maxiter', fallback_maxiter),
                ncv=self.svds_kwargs.get('ncv')
            )

        # Sort singular values in descending order (svds doesn't guarantee order)
        # Guard small values with float64
        sorted_values = cupy.asarray(singular_values, dtype=cupy.float64)
        sorted_values = cupy.abs(sorted_values)
        descending_indices = cupy.argsort(sorted_values)[::-1]
        
        self.singular_values_ = sorted_values[descending_indices]
        self.components_ = V_transpose[descending_indices, :]
        
        # Calculate explained variance (singular values squared, normalized by n_samples)
        self.explained_variance_ = (self.singular_values_ ** 2) / max(n_samples - 1, 1)
        
        # Calculate explained variance ratio
        eps = 1e-18 # Guard for division by zero
        if total_variance > eps:
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        else:
            self.explained_variance_ratio_ = cupy.zeros_like(self.explained_variance_)
        
        # Log summary statistics
        cumulative_variance = float(cupy.sum(self.explained_variance_ratio_))
        logger.info(
            f"GPUTruncatedSVD fit complete. "
            f"Top singular value: {float(self.singular_values_[0]):.6e}, "
            f"Explained variance ratio: {cumulative_variance:.4f}"
        )
        
        if cumulative_variance < 0.8:
            logger.warning(
                f"Only {cumulative_variance:.2%} of variance explained. "
                f"Consider increasing n_components (current: {self.n_components})"
            )
        
        return self

    def fit_transform(self, sparse_matrix_csr: cupy.sparse.csr_matrix) -> cupy.ndarray:
        """
        Fit the model and transform data to lower-dimensional space.
        
        This is a convenience method that combines fit() and transform()
        in a single call, which can be more efficient than calling them separately.
        
        Args:
            sparse_matrix_csr: Input sparse matrix with shape (n_samples, n_features)
        
        Returns:
            Dense array of transformed data with shape (n_samples, n_components) of dtype cupy.float64
        """
        logger.debug("Starting fit_transform on sparse matrix")

        prepared_matrix = self._prepare_input_matrix(sparse_matrix_csr)
        
        # Fit the model
        self.fit(prepared_matrix)
        
        # Transform the data: Z = X * V (where V = components_.T)
        # This projects the original data onto the principal components
        transformed_matrix = prepared_matrix @ self.components_.T
        
        logger.debug(
            f"fit_transform complete. Output shape: {transformed_matrix.shape}, "
            f"dtype: {transformed_matrix.dtype}"
        )
        
        return transformed_matrix

    def transform(self, sparse_matrix_csr: cupy.sparse.csr_matrix) -> cupy.ndarray:
        """
        Transform data using the fitted model.
        
        Projects the input data onto the principal components learned during fit().
        
        Args:
            sparse_matrix_csr: Input sparse matrix with shape (n_samples, n_features)
                              Must have the same n_features as the data used in fit()
        
        Returns:
            Dense array of transformed data with shape (n_samples, n_components)
            
        Raises:
            RuntimeError: If the model hasn't been fitted yet
            ValueError: If input has wrong number of features
        """
        if self.components_ is None:
            raise RuntimeError(
                "GPUTruncatedSVD has not been fitted. "
                "Call fit() or fit_transform() first."
            )
        
        _, n_features = sparse_matrix_csr.shape
        expected_features = self.components_.shape[1]
        
        if n_features != expected_features:
            raise ValueError(
                f"Input has {n_features} features, but GPUTruncatedSVD "
                f"was fitted with {expected_features} features"
            )
        
        prepared_matrix = self._prepare_input_matrix(sparse_matrix_csr)

        logger.debug(
            f"Transforming sparse matrix with shape {prepared_matrix.shape}"
        )
        
        # Project data onto principal components
        transformed_matrix = prepared_matrix @ self.components_.T
        
        return transformed_matrix
    
    def get_params(self) -> dict:
        """
        Get parameters for this estimator.
        
        Returns:
            Dictionary of parameter names to values
        """
        params = {
            'n_components': self.n_components,
            **self.svds_kwargs
        }
        return params
    
    def set_params(self, **params) -> 'GPUTruncatedSVD':
        """
        Set parameters for this estimator.
        
        Args:
            **params: Parameter names and values
            
        Returns:
            Self
        """
        if 'n_components' in params:
            self.n_components = params.pop('n_components')
        
        # Remaining params go to svds_kwargs
        self.svds_kwargs.update(params)
        
        return self