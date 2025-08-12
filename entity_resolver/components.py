# entity_resolver/components.py
"""
Custom GPU-accelerated components for entity resolution pipeline.

This module provides custom implementations of machine learning components
that are optimized for GPU execution and sparse matrix operations.
"""

import logging

import cupy
from cupyx.scipy.sparse.linalg import svds

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