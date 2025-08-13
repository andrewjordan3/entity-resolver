# entity_resolver/components.py
"""
Custom GPU-accelerated components for entity resolution pipeline.

This module provides custom implementations of machine learning components
that are optimized for GPU execution and sparse matrix operations.
"""

import logging

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

    def _eigsh_augmented_tsvd(self, sparse_matrix, n_components: int, ncv=None, tol=1.0e-8, maxiter=20000):
        """
        Numerically stable SVD using an augmented matrix approach. Fallback for when svds fails.

        This method computes the SVD by finding the eigenvalues of a larger,
        symmetric "augmented matrix" B, defined as:
            B = [[0,   X],
                [X.T, 0]]
        
        The eigenvalues of B are ±σ where σ are the singular values of X.
        The eigenvectors of B contain both left (U) and right (V) singular vectors of X:
        - Top n_samples rows → left singular vectors (U)
        - Bottom n_features rows → right singular vectors (V)
        
        This avoids explicitly forming X.T @ X, which can be numerically unstable
        for ill-conditioned matrices.

        Args:
            sparse_matrix (cupy.sparse.csr_matrix): The input data matrix (X).
            n_components (int): The number of singular values to compute (k).
            ncv (int, optional): Number of Lanczos vectors to generate.
            tol (float, optional): Tolerance for convergence (default: 1.0e-8).
            maxiter (int, optional): Maximum number of iterations (default: 20000).

        Returns:
            A tuple (None, singular_values, V_transpose) to match svds output format.
            We return None for U since we don't need it for dimensionality reduction.
        """
        logger.warning(
            "The standard 'svds' solver failed (likely returned all zeros). "
            "Falling back to the numerically stable augmented matrix (eigsh) method. "
            "This may be slower but more robust for ill-conditioned matrices."
        )
        
        n_samples, n_features = sparse_matrix.shape
        dtype = cupy.float64
        
        # Ensure the matrix is in CSR format and float64 for numerical stability
        if not isinstance(sparse_matrix, cupy.sparse.csr_matrix):
            sparse_matrix = sparse_matrix.tocsr()
        if sparse_matrix.dtype != dtype:
            sparse_matrix = sparse_matrix.astype(dtype)
        
        # Define the matrix-vector product for the augmented matrix B.
        # This function is the core of the LinearOperator, allowing eigsh
        # to work without ever explicitly forming the (n+m)×(n+m) matrix B.
        def _augmented_matvec(vector_z):
            """Compute B @ z where B = [[0, X], [X.T, 0]]"""
            # Ensure input is float64 for consistency
            vector_z = cupy.asarray(vector_z, dtype=dtype)
            
            # Split the input vector z into its two parts
            vector_u = vector_z[:n_samples]   # First n_samples elements (corresponds to U space)
            vector_v = vector_z[n_samples:]   # Remaining n_features elements (corresponds to V space)
            
            # Perform the block-matrix multiplication:
            # [0,   X] @ [u] = [X @ v]
            # [X.T, 0]   [v]   [X.T @ u]
            result_upper = sparse_matrix @ vector_v      # X @ v
            result_lower = sparse_matrix.T @ vector_u    # X.T @ u
            
            return cupy.concatenate([result_upper, result_lower])
        
        # Set the number of Lanczos vectors if not specified.
        # This affects memory usage and convergence speed.
        # We need at least 2k+1 vectors, but more can improve convergence.
        if ncv is None:
            # Ensure ncv is within valid bounds for eigsh
            max_ncv = n_samples + n_features - 1  # Maximum possible ncv
            desired_ncv = max(2 * n_components + 1, min(n_components + 20, n_components * 2))
            ncv = min(max_ncv, desired_ncv)
            logger.debug(f"Using ncv={ncv} for augmented matrix eigsh (matrix size: {max_ncv + 1})")
        
        # Create the LinearOperator representing the augmented matrix B.
        # This object encapsulates the matrix-vector product logic.
        augmented_operator = LinearOperator(
            shape=(n_samples + n_features, n_samples + n_features),
            matvec=_augmented_matvec,
            dtype=dtype
        )
        
        # Use 'LA' to get the largest algebraic eigenvalues.
        # For the augmented matrix, we want the largest positive eigenvalues,
        # which correspond to the largest singular values.
        try:
            eigenvalues, eigenvectors = eigsh(
                augmented_operator,
                k=n_components,
                which='LA',  # Largest algebraic eigenvalues
                ncv=ncv,
                tol=tol,
                maxiter=maxiter,
                return_eigenvectors=True
            )
        except Exception as e:
            logger.error(f"eigsh failed with error: {e}")
            raise RuntimeError(
                f"Both svds and eigsh failed. This may indicate severely ill-conditioned data. "
                f"Consider preprocessing your data or using fewer components. Error: {e}"
            ) from e
        
        # The eigenvalues of the augmented matrix are ±σ where σ are the singular values.
        # We take absolute values to get the singular values.
        # Note: Due to numerical errors, eigenvalues might have tiny imaginary parts
        if cupy.iscomplexobj(eigenvalues):
            # Log if we have significant imaginary components (shouldn't happen for symmetric matrix)
            max_imag = cupy.abs(eigenvalues.imag).max()
            if max_imag > 1.0e-10:
                logger.warning(f"Eigenvalues have imaginary components (max: {max_imag:.2e}). Taking real part.")
            eigenvalues = eigenvalues.real
        
        singular_values = cupy.abs(eigenvalues).astype(dtype, copy=False)
        
        # Extract the right singular vectors (V) from the bottom n_features rows of eigenvectors
        # The eigenvectors have shape (n_samples + n_features, n_components)
        right_singular_vectors = eigenvectors[n_samples:, :]  # Shape: (n_features, n_components)
        
        # Sort singular values and corresponding vectors in descending order
        descending_indices = cupy.argsort(singular_values)[::-1]
        singular_values = singular_values[descending_indices]
        right_singular_vectors = right_singular_vectors[:, descending_indices]
        
        # Normalize the right singular vectors to have unit length
        # This is important because eigenvectors from eigsh might not be perfectly normalized
        vector_norms = cupy.linalg.norm(right_singular_vectors, axis=0)
        
        # Check for any zero norms (shouldn't happen, but let's be safe)
        zero_norm_mask = vector_norms < 1.0e-10
        if cupy.any(zero_norm_mask):
            n_zero = int(cupy.sum(zero_norm_mask))
            logger.warning(f"Found {n_zero} right singular vectors with near-zero norm. Setting to random unit vectors.")
            # Replace zero vectors with random unit vectors
            for idx in cupy.where(zero_norm_mask)[0]:
                random_vec = cupy.random.randn(n_features, dtype=dtype)
                random_vec /= cupy.linalg.norm(random_vec)
                right_singular_vectors[:, idx] = random_vec
                vector_norms[idx] = 1.0
        
        # Normalize all vectors
        right_singular_vectors /= vector_norms[cupy.newaxis, :]
        
        # Additional validation: check that we got reasonable singular values
        if cupy.all(singular_values < 1.0e-10):
            logger.error("All singular values are near zero. Data might be all zeros or severely rank-deficient.")
            raise ValueError(
                "Failed to compute meaningful singular values. "
                "Check that your input matrix is not all zeros or extremely sparse."
            )
        
        # Check for convergence quality by looking at the ratio of smallest to largest singular value
        condition_number = singular_values[0] / (singular_values[-1] + 1.0e-10)
        if condition_number > 1.0e10:
            logger.warning(
                f"Extremely high condition number ({condition_number:.2e}). "
                f"Results may be numerically unstable."
            )
        
        logger.info(
            f"Augmented matrix SVD complete. "
            f"Singular values range: [{singular_values[-1]:.2e}, {singular_values[0]:.2e}], "
            f"Condition number: {condition_number:.2e}"
        )
        
        # Return in format consistent with svds: (U, s, V.T)
        # We don't compute U since it's not needed for dimensionality reduction
        return None, singular_values, right_singular_vectors.T

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
            fallback_ncv = min(
                (n_samples + n_features) - 1, 
                max(2*self.n_components + 1, self.n_components + 64)
            )
            _, singular_values, V_transpose = self._eigsh_augmented_tsvd(
                sparse_matrix_csr,
                n_components=self.n_components,
                tol=self.svds_kwargs.get('tol', 1e-8),
                maxiter=self.svds_kwargs.get('maxiter', fallback_maxiter),
                ncv=self.svds_kwargs.get('ncv', fallback_ncv)
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