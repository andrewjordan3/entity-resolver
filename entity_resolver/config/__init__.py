# entity_resolver/config/__init__.py
"""
Initializes the config sub-package.

This file makes the most important components of the configuration system
directly available when importing from `entity_resolver.config`, simplifying
access for other parts of the application.
"""

from .loader import load_config
from .schema import (
    ClustererConfig,
    ColumnConfig,
    ConfidenceScoringConfig,
    EnsembleParams,
    NormalizationConfig,
    OutputConfig,
    ResolverConfig,
    SimilarityNnParams,
    SimilarityTfidfParams,
    ValidationConfig,
    VectorizerConfig,
)

# Defines the public API of this sub-package.
# When a user does `from entity_resolver.config import *`, only these
# names will be imported.
__all__ = [
    'ResolverConfig',
    'VectorizerConfig',
    'NormalizationConfig',
    'ClustererConfig',
    'ValidationConfig',
    'ConfidenceScoringConfig',
    'OutputConfig',
    'ColumnConfig',
    'EnsembleParams',
    'SimilarityTfidfParams',
    'SimilarityNnParams',
    'load_config',
]
