# entity_resolver/utils/vector.py
"""
This module provides GPU-accelerated utilities for vector transformations
and linear algebra operations, primarily used for creating and manipulating
embeddings.
"""

import cupy
import logging
from typing import Dict, List, Tuple

# Set up a logger for this module
logger = logging.getLogger(__name__)


def normalize_rows(vectors: cupy.ndarray, eps: float = 1e-8) -> cupy.ndarray:
    """
    Performs a safe, row-wise L2 normalization on a CuPy array.

    This ensures each row vector has a magnitude of 1, which is essential
    for cosine similarity calculations. A small epsilon is added to the norm
    to prevent division by zero for any all-zero row vectors.

    Args:
        vectors: A 2D CuPy array.
        eps: A small float to prevent division by zero.

    Returns:
        The row-normalized CuPy array.
    """
    norms = cupy.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + eps)


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
    proportions: Dict[str, float],
    eps: float = 1e-8
) -> List[cupy.ndarray]:
    """
    Balances the energy of feature streams to match target proportions.

    This method calculates the average energy (mean squared L2 norm) of
    each feature stream and computes a scaling factor to adjust its energy
    to a desired proportion. This ensures each stream contributes a controlled
    amount of variance to the final combined vector.

    Args:
        vector_streams: Dictionary of stream names to CuPy arrays.
        proportions: Dictionary mapping stream names to desired energy proportions.
        eps: A small epsilon to prevent division by zero.

    Returns:
        A list of the balanced (rescaled) CuPy arrays.
    """
    def block_energy(Z: cupy.ndarray) -> float:
        return float((Z * Z).sum(axis=1).mean().get())

    energies = {name: block_energy(vec) for name, vec in vector_streams.items()}
    scaling_factors = {
        name: (proportions[name] / (energies[name] + eps)) ** 0.5
        for name in vector_streams.keys()
    }
    logger.debug(f"Balancing streams with scaling factors: {scaling_factors}")
    return [vec * scaling_factors[name] for name, vec in vector_streams.items()]


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
