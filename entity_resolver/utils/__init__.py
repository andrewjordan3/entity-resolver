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
    get_canonical_name_gpu
)

# Address operations
from .address import (
    safe_parse_address,
    create_address_key_gpu,
    calculate_address_score_gpu,
    get_best_address_gpu
)

# Similarity calculations
from .similarity import (
    calculate_similarity_gpu,
    find_similar_pairs
)

# Vector operations
from .vector import (
    normalize_rows,
    balance_feature_streams,
    center_kernel_matrix,
    center_kernel_vector,
    get_top_k_positive_eigenpairs,
    create_consensus_embedding
)

# Graph operations
from .graph import (
    create_edge_list,
    find_graph_components,
    build_mutual_rank_graph
)

# Clustering operations
from .clustering import (
    attach_noise_points,
    merge_snn_clusters
)

# Validation
from .validation import (
    validate_no_duplicates,
    validate_canonical_consistency
)

# The public API of the 'utils' package.
__all__ = [
    # text.py
    'get_canonical_name_gpu',

    # address.py
    'safe_parse_address',
    'create_address_key_gpu',
    'calculate_address_score_gpu',
    'get_best_address_gpu',

    # similarity.py
    'calculate_similarity_gpu',
    'find_similar_pairs',

    # vector.py
    'normalize_rows',
    'balance_feature_streams',
    'center_kernel_matrix',
    'center_kernel_vector',
    'get_top_k_positive_eigenpairs',
    'create_consensus_embedding',

    # graph.py
    'create_edge_list',
    'find_graph_components',
    'build_mutual_rank_graph',

    # clustering.py
    'attach_noise_points',
    'merge_snn_clusters',

    # validation.py
    'validate_no_duplicates',
    'validate_canonical_consistency'
]
