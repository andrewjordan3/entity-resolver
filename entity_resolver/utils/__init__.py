# entity_resolver/utils/__init__.py
"""
Utility Package for the Entity Resolution Pipeline.

This package consolidates all low-level, reusable helper functions for the
entity resolution pipeline, organized into domain-specific modules. This
`__init__.py` file exposes the public functions from each module, allowing for
clean and convenient access, e.g., `from entity_resolver.utils import get_canonical_name_gpu`.

The `__all__` variable explicitly defines the public API of this package,
controlling which names are imported when a user performs `from .utils import *`.

Modules:
- text: Text processing and canonical name selection.
- address: Address parsing, normalization, and scoring.
- similarity: TF-IDF vectorization and similarity calculations.
- vector: Vector transformations and linear algebra operations.
- graph: Graph construction and component analysis.
- clustering: Post-processing and refinement for clustering results.
- validation: Data validation and consistency checks.
"""

# Text processing
from .text import (
    get_canonical_name_gpu,
    nfkc_normalize_series,
    find_canonical_name,
)

# Address operations
from .address import (
    safe_parse_address,
    create_address_key_gpu,
    calculate_address_score_gpu,
    get_best_address_gpu,
    normalize_us_states,
)

# Similarity calculations
from .similarity import (
    calculate_similarity_gpu,
    find_similar_pairs,
    calculate_embedding_similarity,
)

# Vector operations
from .vector import (
    normalize_rows,
    balance_feature_streams,
    center_kernel_matrix,
    center_kernel_vector,
    get_top_k_positive_eigenpairs,
    create_consensus_embedding,
    create_initial_vector,
)

# Graph operations
from .graph import (
    create_edge_list,
    find_graph_components,
    build_mutual_rank_graph,
)

# Clustering operations
from .clustering import (
    attach_noise_points,
    merge_snn_clusters,
)

# Validation
from .validation import (
    validate_no_duplicates,
    validate_canonical_consistency,
    check_state_compatibility,
    check_street_number_compatibility,
)

# Matrix operations
from .matrix_ops import (
    ensure_finite_matrix,
    winsorize_matrix,
    scale_by_frobenius_norm,
    prune_sparse_matrix,
)

# GPU memory operations
from .clean_mem import gpu_memory_cleanup

# String preparation for vectorization
from .embedding_streams import (
    prepare_text_streams,
    TextStreamSet,
    AllTextStreams,
)

# The public API of the 'utils' package.
__all__ = [
    # text.py
    'get_canonical_name_gpu',
    'nfkc_normalize_series',
    'find_canonical_name',

    # address.py
    'safe_parse_address',
    'create_address_key_gpu',
    'calculate_address_score_gpu',
    'get_best_address_gpu',
    'normalize_us_states',

    # similarity.py
    'calculate_similarity_gpu',
    'find_similar_pairs',
    'calculate_embedding_similarity',

    # vector.py
    'normalize_rows',
    'balance_feature_streams',
    'center_kernel_matrix',
    'center_kernel_vector',
    'get_top_k_positive_eigenpairs',
    'create_consensus_embedding',
    'create_initial_vector',

    # graph.py
    'create_edge_list',
    'find_graph_components',
    'build_mutual_rank_graph',

    # clustering.py
    'attach_noise_points',
    'merge_snn_clusters',

    # validation.py
    'validate_no_duplicates',
    'validate_canonical_consistency',
    'check_state_compatibility',
    'check_street_number_compatibility',

    # matrix_ops.py
    'ensure_finite_matrix',
    'winsorize_matrix',
    'scale_by_frobenius_norm',
    'prune_sparse_matrix',

    # clean_mem.py
    'gpu_memory_cleanup',

    # embedding_streams.py
    'prepare_text_streams',
    'TextStreamSet',
    'AllTextStreams'
]
