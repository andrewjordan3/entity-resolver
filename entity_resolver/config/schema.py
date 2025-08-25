# entity_resolver/config/schema.py
"""
Entity Resolution Pipeline Configuration Schema

This module defines all Pydantic-based configuration models for the GPU-accelerated 
entity resolution pipeline using RAPIDS (cuML, cuDF, cuPy).

Using Pydantic models provides:
- Type-hinting for better IDE support and code clarity
- Automatic validation with descriptive error messages
- Self-documentation through field descriptions
- Easy serialization/deserialization to/from YAML
- Default values with optional overrides
- Strict validation preventing unknown fields (extra='forbid')

The main ResolverConfig class composes all subordinate configurations for each stage 
of the entity resolution process, creating a single source of truth for all parameters.
"""

from pathlib import Path
from typing import List, Dict, Set, Any, Optional, Literal, Union, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import cupy

class SvdEigshFallbackConfig(BaseModel):
    """Parameters for the robust SVD fallback mechanism."""
    # No arbitrary types allowed; all fields must be serializable.
    model_config = ConfigDict(extra='forbid')

    fallback_dtype: Any = Field(
        default="float64",
        description="String name of the CuPy dtype for the eigsh solver ('float64' is recommended for stability)."
    )
    eigsh_restarts: int = Field(
        default=3,
        ge=0,
        description="Number of times to restart the eigsh solver on failure before raising an error."
    )
    prune_min_row_sum: float = Field(
        default=1e-9,
        ge=0.0,
        description="Rows with a total sum of values less than this threshold will be removed before SVD."
    )
    prune_min_df: int = Field(
        default=2,
        ge=1,
        description="Columns that appear in fewer than `min_df` documents (rows) will be removed."
    )
    prune_max_df_ratio: float = Field(
        default=0.98,
        gt=0.0,
        le=1.0,
        description="Columns that appear in more than `max_df_ratio * n_rows` documents will be removed."
    )
    prune_energy_cutoff: float = Field(
        default=0.995,
        gt=0.0,
        le=1.0,
        description="Keeps the smallest set of columns whose cumulative energy exceeds this ratio of the total."
    )
    winsorize_limits: Tuple[Optional[float], Optional[float]] = Field(
        default=(None, 0.999),
        description="Quantile limits for clipping extreme values. Use None to disable a limit on one side."
    )

    @field_validator('fallback_dtype', mode='before')
    @classmethod
    def validate_dtype_string(cls, v: Any) -> str:
        """Ensures the input is a string and represents a valid CuPy dtype name."""
        if not isinstance(v, str):
            raise TypeError("fallback_dtype must be provided as a string (e.g., 'float64').")
        try:
            cupy.dtype(v)
        except TypeError:
            raise ValueError(f"'{v}' is not a valid cupy dtype name.")
        return v

    @field_validator('fallback_dtype', mode='after')
    @classmethod
    def convert_string_to_dtype(cls, v: str) -> Any:
        """Converts the validated string into a cupy.dtype object."""
        return cupy.dtype(v)

    @field_validator('winsorize_limits')
    @classmethod
    def validate_winsorize_limits(cls, v: Tuple[Optional[float], Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
        """Validates the winsorize limits tuple."""
        lower, upper = v
        if lower is not None and not (0.0 <= lower <= 1.0):
            raise ValueError(f"Lower winsorize limit must be between 0.0 and 1.0, got {lower}")
        if upper is not None and not (0.0 <= upper <= 1.0):
            raise ValueError(f"Upper winsorize limit must be between 0.0 and 1.0, got {upper}")
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError(f"Lower limit ({lower}) cannot be >= upper limit ({upper})")
        return v

# === Core Data and I/O Configurations ===

class ColumnConfig(BaseModel):
    """
    Specifies the names of the input DataFrame columns to be used in the resolution process.
    
    This configuration tells the resolver which columns contain the entity names and 
    address components. The resolver will look for these exact column names in your 
    input cuDF DataFrame.
    """
    model_config = ConfigDict(extra='forbid')
    
    entity_col: str = Field(
        default='raw_name', 
        min_length=1,
        description=(
            "The name of the column containing the primary entity name to be resolved. "
            "This should be the main business/organization name column in your data. "
            "Examples: 'company_name', 'vendor_name', 'organization', 'business_name'"
        )
    )
    
    address_cols: List[str] = Field(
        default_factory=lambda: ['Maintenance Service Vendor Address', 'Maintenance Service Vendor City', 
                                'Maintenance Service Vendor State/Province', 'Maintenance Service Vendor Zip/Postal Code'], 
        min_length=1,
        description=(
            "A list of column names that together form the complete address. "
            "These columns will be concatenated in order to create the full address string. "
            "Can be a single column (e.g., ['full_address']) or multiple columns "
            "(e.g., ['street', 'city', 'state', 'zip']). Order matters for concatenation."
        )
    )

    @field_validator('address_cols')
    @classmethod
    def validate_address_columns(cls, v: List[str]) -> List[str]:
        """
        Validates that address columns are non-empty strings with no duplicates.
        
        Args:
            v: List of address column names
            
        Returns:
            Validated list of column names
            
        Raises:
            ValueError: If columns are empty strings or duplicates exist
        """
        seen = set()
        for i, col in enumerate(v):
            if not isinstance(col, str) or not col.strip():
                raise ValueError(f"address_cols[{i}] must be a non-empty string, got: '{col}'")
            if col in seen:
                raise ValueError(f"Duplicate column name in address_cols: '{col}'")
            seen.add(col)
        return v


class OutputConfig(BaseModel):
    """
    Configuration for output formatting, logging verbosity, and manual review thresholds.
    
    Controls how the final resolved entities are formatted and which entities should
    be flagged for human review based on confidence scores.
    """
    model_config = ConfigDict(extra='forbid')
    
    output_format: Literal['proper', 'raw', 'upper', 'lower'] = Field(
        default='proper',
        description=(
            "The case style for the final canonical entity names:\n"
            "- 'proper': Title Case (e.g., 'Acme Corporation')\n"
            "- 'raw': Keep original case as-is from the most representative cluster member\n"
            "- 'upper': UPPERCASE (e.g., 'ACME CORPORATION')\n"
            "- 'lower': lowercase (e.g., 'acme corporation')"
        )
    )
    
    review_confidence_threshold: float = Field(
        default=0.75, 
        ge=0.0, 
        le=1.0,
        description=(
            "The confidence score threshold for flagging matches for manual review. "
            "Matches with confidence below this value will be marked for human verification. "
            "Range: 0.0 (flag everything) to 1.0 (flag nothing). "
            "Recommended values: 0.7-0.8 for high precision requirements, 0.5-0.6 for high recall."
        )
    )
    
    log_level: int = Field(
        default=10,  # logging.DEBUG
        description=(
            "The logging verbosity level for console output:\n"
            "- 10 (DEBUG): Detailed diagnostic output, useful for troubleshooting\n"
            "- 20 (INFO): General informational messages about progress\n"
            "- 30 (WARNING): Only warnings and errors\n"
            "- 40 (ERROR): Only error messages\n"
            "- 50 (CRITICAL): Only critical failures"
        )
    )
    
    split_address_components: bool = Field(
        default=False,
        description=(
            "If True, the output cuDF DataFrame will include separate columns for each "
            "canonical address component (street, city, state, zip). "
            "If False, only a single concatenated canonical_address column is included. "
            "Useful when downstream processes need structured address data."
        )
    )


# === Preprocessing Configurations ===

class NormalizationConfig(BaseModel):
    """
    Defines rules for cleaning and standardizing entity names before matching.
    
    This preprocessing step is crucial for matching variations of the same entity.
    It handles common abbreviations, misspellings, and removes legal suffixes that
    don't contribute to entity identity. All operations are GPU-accelerated using cuDF.
    """
    model_config = ConfigDict(extra='forbid')
    
    replacements: Dict[str, str] = Field(
        default_factory=lambda: {
            # Common misspellings
            "traiier": "trailer",
            # Abbreviations to expand
            "rpr": "repair",
            "svcs": "service",
            "svc": "service",
            "ctr": "center",
            "ctrs": "centers", 
            "cntr": "center",
            "trk": "truck",
            "auto": "automotive",
            "auth": "authorized",
            "dist": "distribution",
            "mfg": "manufacturing",
            "mfr": "manufacturing",
            "equip": "equipment",
            "natl": "national",
            "mgmt": "management",
            "assoc": "associates"
        },
        description=(
            "A dictionary mapping common abbreviations, acronyms, or misspellings to their "
            "standardized forms. Applied before suffix removal. Keys should be lowercase. "
            "The replacement is case-insensitive but preserves the original case pattern. "
            "Example: 'Svc' becomes 'Service', 'SVC' becomes 'SERVICE'."
        )
    )
    
    suffixes_to_remove: Set[str] = Field(
        default_factory=lambda: {
            # Legal entity types
            "inc", "incorporated", "llc", "ll", "lp", "llp", 
            "ltd", "limited", "corp", "corporation", "co", "company",
            "plc", "pllc", "pa", "pc", "sc",
            # Doing Business As indicators
            "dba", "fka", "aka", "etal", "et al",
            # Geographic/scope indicators
            "international", "intl", "usa", "america", "us",
            # Organizational structure
            "group", "grp", "holdings", "ent"
        },
        description=(
            "A set of common legal and organizational suffixes to remove from entity names. "
            "These are removed after replacements are applied. Removal is case-insensitive. "
            "Suffixes are only removed from the end of the name, not from the middle. "
            "Example: 'Acme Corp LLC' becomes 'Acme', but 'LLC Acme Services' becomes 'LLC Acme Services'."
        )
    )


# === Modeling Stage Configurations ===

class VectorizerConfig(BaseModel):
    """
    Parameters for GPU-accelerated feature extraction and dimensionality reduction.
    
    The vectorizer creates multiple complementary representations (streams) of each entity:
    - TF-IDF: Captures character-level patterns and spelling variations (using cuML)
    - Phonetic: Captures how names sound, helping match phonetically similar entities
    - Semantic: Captures meaning using transformer embeddings (GPU-accelerated)
    
    These streams are combined to create a rich representation for clustering.
    All operations use RAPIDS (cuML, cuDF, cuPy) for GPU acceleration.
    """
    model_config = ConfigDict(extra='forbid')
    
    # === Feature Stream Selection ===
    encoders: List[Literal['tfidf', 'phonetic', 'semantic']] = Field(
        default_factory=lambda: ['tfidf', 'phonetic', 'semantic'],
        description=(
            "List of feature extraction methods to use. Each creates a different view of the data:\n"
            "- 'tfidf': Character n-gram features for spelling similarity (GPU-accelerated with cuML)\n"
            "- 'phonetic': Sound-based encoding for phonetic matching\n"
            "- 'semantic': Transformer embeddings for meaning similarity (GPU-accelerated)\n"
            "Using all three provides the best accuracy but increases computation time."
        )
    )
    
    sparse_reducers: List[Literal['svd', 'pca']] = Field(
        default_factory=lambda: ['svd', 'pca'],
        description=(
            "Dimensionality reduction techniques applied to sparse feature matrices:\n"
            "- 'svd': Custom GPUTruncatedSVD using cupyx.scipy.sparse.linalg.svds\n"
            "- 'pca': cuML PCA (requires dense conversion but captures more variance)\n"
            "Both are applied and their results concatenated for richer representations."
        )
    )
    
    use_address_in_encoding: bool = Field(
        default=True,
        description=(
            "If True, concatenates the normalized address to the entity name for TF-IDF encoding. "
            "This helps distinguish entities with similar names but different locations. "
            "Has no effect on phonetic or semantic streams."
        )
    )
    
    # === Spectral Preprocessing ===
    damping_beta: float = Field(
        default=0.4, 
        ge=0.0, 
        le=1.0,
        description=(
            "Controls spectral damping (variance normalization) strength:\n"
            "- 0.0: No damping (preserve original variance structure)\n"
            "- 1.0: Full whitening (all features have equal variance)\n"
            "- 0.3-0.5: Balanced damping (recommended)\n"
            "Helps prevent dominant features from overshadowing others."
        )
    )
    
    epsilon: float = Field(
        default=1.0e-8, 
        gt=0,
        description=(
            "Small constant added to denominators to prevent division by zero. "
            "Should be much smaller than your data scale (typically 1.0e-8 to 1.0e-10)."
        )
    )
    
    # === Semantic Stream Configuration ===
    semantic_model: str = Field(
        default='BAAI/bge-base-en-v1.5', 
        min_length=1,
        description=(
            "HuggingFace sentence-transformer model for semantic encoding. "
            "Should be a model name from https://huggingface.co/sentence-transformers. "
            "The model is loaded on GPU for accelerated inference. Examples:\n"
            "- 'all-mpnet-base-v2': General purpose, 768 dims (good default)\n"
            "- 'all-MiniLM-L6-v2': Faster, 384 dims (for speed)\n"
            "- 'BAAI/bge-base-en-v1.5': High quality, 768 dims (for accuracy)"
        )
    )
    
    semantic_batch_size: int = Field(
        default=1024, 
        gt=0,
        description=(
            "Batch size for GPU semantic encoding. Larger values are faster but use more GPU memory. "
            "Adjust based on GPU memory: 256-512 for small GPUs, 1024-2048 for large GPUs."
        )
    )
    
    # === TF-IDF Stream Configuration ===
    tfidf_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "analyzer": "char",
            "ngram_range": [3, 5],
            "max_features": 10000,
            "min_df": 2,
            "sublinear_tf": True,
            "dtype": "float64",
        },
        description=(
            "Parameters passed to cuML's TfidfVectorizer (GPU-accelerated):\n"
            "- analyzer: 'char' for character n-grams, 'word' for word n-grams\n"
            "- ngram_range: [min_n, max_n] size of n-grams to extract\n"
            "- max_features: Maximum number of features (vocabulary size)\n"
            "- min_df: Minimum document frequency for a feature\n"
            "- sublinear_tf: Use log(TF) instead of raw TF\n"
            "- dtype: Data type for the matrix ('float32' saves GPU memory, 'float64' is more precise)"
        )
    )
    
    tfidf_svd_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            'n_components': 1536,
            'tol': 1.0e-7,
            'ncv': 6144,
            'maxiter': 100000
        },
        description=(
            "Parameters for custom GPUTruncatedSVD (handles sparse matrices on GPU):\n"
            "- n_components: Number of dimensions to reduce to (typically 512-2048)\n"
            "- tol: Convergence tolerance for cupyx.scipy.sparse.linalg.svds\n"
            "- ncv: Number of Lanczos vectors for svds (should be > n_components)\n"
            "- maxiter: Maximum iterations for svds convergence\n"
            "Note: This uses a custom implementation since cuML doesn't support sparse matrices"
        )
    )
    
    tfidf_pca_params: Dict[str, Any] = Field(
        default_factory=lambda: {'n_components': 1024},
        description=(
            "Parameters for cuML PCA applied to TF-IDF features:\n"
            "- n_components: Target dimensionality (typically 256-1024)\n"
            "Note: PCA will automatically receive random_state from global config"
        )
    )
    
    # === Phonetic Stream Configuration ===
    phonetic_max_words: int = Field(
        default=5, 
        ge=1,
        le=10,
        description=(
            "Maximum number of words from entity name to encode phonetically. "
            "Using more words increases accuracy but also computation time. "
            "Typically 3-5 words capture the most important parts of a name."
        )
    )
    
    phonetic_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            'analyzer': 'word',
            'binary': True,
            'max_features': 2000
        },
        description=(
            "Parameters for CountVectorizer on phonetic codes (GPU-accelerated with cuML):\n"
            "- analyzer: Should be 'word' to treat each phonetic code as a token\n"
            "- binary: If True, use presence/absence rather than counts\n"
            "- max_features: Maximum vocabulary size for phonetic codes"
        )
    )
    
    phonetic_svd_params: Dict[str, Any] = Field(
        default_factory=lambda: {'n_components': 256},
        description=(
            "GPUTruncatedSVD parameters for phonetic features. Typically needs fewer components "
            "than TF-IDF since phonetic features are less complex."
        )
    )
    
    phonetic_pca_params: Dict[str, Any] = Field(
        default_factory=lambda: {'n_components': 160},
        description=(
            "cuML PCA parameters for phonetic features. Further reduces dimensionality "
            "after SVD for a compact phonetic representation."
        )
    )
    
    # === Stream Balancing Configuration ===
    stream_proportions: Dict[Literal['semantic', 'tfidf', 'phonetic'], float] = Field(
        default_factory=lambda: {
            'semantic': 0.45,
            'tfidf': 0.45,
            'phonetic': 0.10
        },
        description=(
            "Relative importance weights for each feature stream. Must sum to 1.0.\n"
            "These control how much each stream contributes to the final representation:\n"
            "- semantic: Weight for meaning-based features (0.3-0.5 typical)\n"
            "- tfidf: Weight for spelling-based features (0.3-0.5 typical)\n"
            "- phonetic: Weight for sound-based features (0.1-0.2 typical)\n"
            "Adjust based on your data characteristics."
        )
    )
    
    # === String Similarity Fallback Configuration ===
    similarity_tfidf: Dict[str, Any] = Field(
        default_factory=lambda: {
            'analyzer': 'char',
            'ngram_range': [3, 5],
            'min_df': 2,
            'sublinear_tf': True,
            'norm': 'l2',
            'max_features': 50000
        },
        description=(
            "TF-IDF parameters for the string similarity fallback matcher (cuML). "
            "Used when entities can't be confidently assigned to clusters. "
            "Typically uses more features than the main TF-IDF for higher precision."
        )
    )
    
    similarity_nn: Dict[str, Any] = Field(
        default_factory=lambda: {
            'n_neighbors': 24,
            'metric': 'cosine'
        },
        description=(
            "cuML nearest neighbor search parameters for similarity fallback:\n"
            "- n_neighbors: Number of similar entities to retrieve (10-50 typical)\n"
            "- metric: Distance metric ('cosine' for normalized, 'euclidean' for raw)"
        )
    )

    # === SVD Eigsh Fallback Configuration ===
    eigsh_fallback_params: SvdEigshFallbackConfig = Field(
        default_factory=SvdEigshFallbackConfig,
        description="Parameters for the robust SVD fallback mechanism used when the standard solver fails."
    )

    @field_validator('stream_proportions')
    @classmethod
    def validate_proportions_sum(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Ensures stream proportions sum to 1.0 within floating point tolerance."""
        total = sum(v.values())
        if abs(total - 1.0) > 1.0e-6:
            raise ValueError(
                f"stream_proportions must sum to 1.0, but got {total:.6f}. "
                f"Current values: {v}"
            )
        return v
    
    @field_validator('tfidf_params', mode='before')
    @classmethod
    def validate_tfidf_dtype_string(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validates that the dtype in tfidf_params is a valid string."""
        if 'dtype' in v:
            dtype_str = v['dtype']
            if not isinstance(dtype_str, str):
                raise TypeError(f"tfidf_params['dtype'] must be a string, but got {type(dtype_str)}")
            try:
                cupy.dtype(dtype_str)
            except TypeError:
                raise ValueError(f"'{dtype_str}' is not a valid cupy dtype name for tfidf_params.")
        return v

    @field_validator('tfidf_params', mode='after')
    @classmethod
    def convert_tfidf_dtype(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Converts the validated dtype string in tfidf_params to a cupy.dtype object."""
        if 'dtype' in v:
            v['dtype'] = cupy.dtype(v['dtype'])
        return v


class ClustererConfig(BaseModel):
    """
    Configuration for GPU-accelerated manifold learning (UMAP) and clustering (HDBSCAN/SNN).
    
    This is the core of the entity resolution process. UMAP learns a low-dimensional
    representation that preserves entity similarity, then clustering algorithms
    group similar entities together. All operations use cuML for GPU acceleration.
    """
    model_config = ConfigDict(extra='forbid')
    
    # === UMAP Ensemble Configuration ===
    umap_n_runs: int = Field(
        default=12, 
        ge=1,
        le=50,
        description=(
            "Number of cuML UMAP models to train with different parameters. "
            "Multiple runs with parameter variation improves robustness. "
            "More runs = better quality but longer runtime. "
            "Recommended: 3-5 for speed, 10-20 for quality, 30+ for production."
        )
    )
    
    umap_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "n_neighbors": 15,
            "n_components": 48,
            "min_dist": 0.05,
            "spread": 0.5,
            "metric": "cosine",
            "init": "spectral",
            "n_epochs": 400,
            "negative_sample_rate": 7,
            "repulsion_strength": 1.0,
            "learning_rate": 0.5,
        },
        description=(
            "Base cuML UMAP parameters (some will be randomized in ensemble):\n"
            "- n_neighbors: Size of local neighborhood (5-50, higher preserves more global structure)\n"
            "- n_components: Output dimensions (20-100, higher preserves more information)\n"
            "- min_dist: Minimum distance between points (0-0.99, lower allows tighter clusters)\n"
            "- spread: Scale of embedded points (must be > min_dist, typically 0.5-2.0)\n"
            "- metric: Distance metric ('cosine' for normalized, 'euclidean' for raw)\n"
            "- init: Initialization ('spectral' for deterministic, 'random' for stochastic)\n"
            "- n_epochs: Training iterations (200-500 typical)\n"
            "- negative_sample_rate: Negative sampling rate (5-20, higher prevents overfitting)\n"
            "- repulsion_strength: Force between points (0.5-2.0, higher spreads clusters)\n"
            "- learning_rate: Optimization step size (0.5-1.5 typical)"
        )
    )
    
    umap_ensemble_sampling_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            'local_view_ratio': 0.7,
            'n_neighbors_local': [10, 35],
            'n_neighbors_global': [40, 70],
            'min_dist': [0.0, 0.15],
            'spread': [0.5, 2.0],
            'n_epochs': [200, 500],
            'learning_rate': [0.5, 1.5],
            'repulsion_strength': [0.5, 1.5],
            'negative_sample_rate': [5, 20],
            'init_strategies': ['spectral', 'random']
        },
        description=(
            "Parameter ranges for randomized UMAP ensemble:\n"
            "- local_view_ratio: Fraction of models using local vs global neighborhoods\n"
            "- n_neighbors_local: Range for local neighborhood models [min, max]\n"
            "- n_neighbors_global: Range for global neighborhood models [min, max]\n"
            "- min_dist: Range for minimum distance parameter\n"
            "- spread: Range for spread parameter\n"
            "- n_epochs: Range for training iterations\n"
            "- learning_rate: Range for learning rate\n"
            "- repulsion_strength: Range for repulsion force\n"
            "- negative_sample_rate: Range for negative sampling\n"
            "- init_strategies: Initialization methods to sample from"
        )
    )
    
    # === Consensus Embedding Configuration ===
    cosine_consensus_n_samples: int = Field(
        default=8192, 
        ge=1000,
        le=50000,
        description=(
            "Number of samples for kernel PCA consensus embedding (GPU-accelerated). "
            "Higher values give better consensus but use more GPU memory. "
            "Should be min(total_entities, 8192) for most datasets."
        )
    )
    
    cosine_consensus_batch_size: int = Field(
        default=2048, 
        ge=128,
        le=8192,
        description=(
            "Batch size for GPU consensus embedding computation. "
            "Larger batches are faster but use more GPU memory. "
            "Adjust based on available VRAM."
        )
    )
    
    # === HDBSCAN Clustering Configuration ===
    max_noise_rate_warn: float = Field(
        default=0.95, 
        ge=0.0, 
        le=1.0,
        description=(
            "Threshold for warning about high noise rates in cuML HDBSCAN. "
            "If more than this fraction of points are noise, a warning is issued. "
            "High noise might indicate parameters need adjustment."
        )
    )
    
    hdbscan_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            'min_cluster_size': 2,
            'min_samples': 1,
            'cluster_selection_epsilon': 0.0,
            'prediction_data': True,
            'alpha': 0.9,
            'cluster_selection_method': 'leaf'
        },
        description=(
            "cuML HDBSCAN clustering parameters:\n"
            "- min_cluster_size: Minimum size for a cluster (2+ for entity resolution)\n"
            "- min_samples: Conservative parameter (usually 1 for ER)\n"
            "- cluster_selection_epsilon: Distance threshold for cluster selection\n"
            "- prediction_data: Generate data for soft clustering (keep True)\n"
            "- alpha: Distance scaling parameter (0.8-1.0 typical)\n"
            "- cluster_selection_method: 'leaf' for fine-grained, 'eom' for stable clusters"
        )
    )
    
    # === SNN (Shared Nearest Neighbor) Configuration ===
    snn_clustering_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            'k_neighbors': 48,
            'louvain_resolution': 0.60
        },
        description=(
            "SNN graph clustering parameters (GPU-accelerated with cugraph):\n"
            "- k_neighbors: Number of neighbors for graph construction (30-100 typical)\n"
            "- louvain_resolution: Community detection resolution (0.4-1.0, higher = smaller clusters)"
        )
    )
    
    noise_attachment_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            'k': 15,
            'tau': 0.82,
            'min_matching': 2,
            'ratio_threshold': 1.5
        },
        description=(
            "Parameters for attaching noise points to clusters:\n"
            "- k: Number of neighbors to check for attachment\n"
            "- tau: Similarity threshold for attachment (0.7-0.9 typical)\n"
            "- min_matching: Minimum shared neighbors required\n"
            "- ratio_threshold: Ratio test for confident attachment"
        )
    )
    
    # === Cluster Merging Configuration ===
    merge_median_threshold: float = Field(
        default=0.84, 
        ge=0.0, 
        le=1.0,
        description=(
            "Minimum median similarity between clusters to consider merging. "
            "Higher values = more conservative merging. "
            "0.80-0.85 typical for entity resolution."
        )
    )
    
    merge_max_threshold: float = Field(
        default=0.90, 
        ge=0.0, 
        le=1.0,
        description=(
            "Minimum maximum similarity (best pair) required for merging. "
            "Ensures at least one very strong connection exists. "
            "Should be higher than merge_median_threshold."
        )
    )
    
    merge_sample_size: int = Field(
        default=32, 
        ge=5,
        le=2048,
        description=(
            "Number of points to sample from large clusters for merge checks. "
            "Prevents O(n²) comparison complexity. Higher = more accurate but slower."
        )
    )
    
    centroid_similarity_threshold: float = Field(
        default=0.75, 
        ge=0.0, 
        le=1.0,
        description=(
            "Pre-filter threshold for cluster centroid similarity. "
            "Only cluster pairs with centroids above this are checked in detail. "
            "Lower values check more pairs but take longer."
        )
    )
    
    merge_batch_size: int = Field(
        default=2048, 
        ge=128,
        description="Batch size for vectorized GPU merge computations."
    )
    
    centroid_sample_size: int = Field(
        default=2048, 
        ge=128,
        description="Sample size for computing cluster centroids on GPU."
    )
    
    # === Ensemble Configuration ===
    ensemble_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            'purity_min': 0.75,
            'min_overlap': 3,
            'allow_new_snn_clusters': True,
            'min_newcluster_size': 4,
            'default_rescue_conf': 0.60
        },
        description=(
            "Parameters for combining HDBSCAN and SNN results:\n"
            "- purity_min: Minimum purity to accept a cluster without validation\n"
            "- min_overlap: Minimum entities shared to match clusters\n"
            "- allow_new_snn_clusters: Whether SNN can introduce new clusters\n"
            "- min_newcluster_size: Minimum size for new SNN clusters\n"
            "- default_rescue_conf: Default confidence for rescued noise points"
        )
    )
    
    @model_validator(mode='after')
    def validate_umap_spread(self) -> 'ClustererConfig':
        """Ensures UMAP spread is strictly greater than min_dist to prevent errors."""
        min_dist = self.umap_params.get('min_dist', 0)
        spread = self.umap_params.get('spread', 1)
        if spread <= min_dist:
            raise ValueError(
                f"UMAP 'spread' ({spread}) must be strictly greater than 'min_dist' ({min_dist}). "
                f"Recommended: spread = min_dist + 0.1 to 1.0"
            )
        return self
    
    @model_validator(mode='after')
    def validate_neighbor_ranges(self) -> 'ClustererConfig':
        """Ensures n_neighbors_local and n_neighbors_global ranges don't overlap."""
        sampling = self.umap_ensemble_sampling_config
        
        # Extract ranges
        local_min, local_max = sampling.get('n_neighbors_local', [10, 35])
        global_min, global_max = sampling.get('n_neighbors_global', [40, 70])
        
        # Check for overlap
        if local_max >= global_min:
            raise ValueError(
                f"n_neighbors_local range [{local_min}, {local_max}] overlaps with "
                f"n_neighbors_global range [{global_min}, {global_max}]. "
                f"The maximum of n_neighbors_local ({local_max}) must be less than "
                f"the minimum of n_neighbors_global ({global_min}) to maintain distinct "
                f"local and global neighborhood views."
            )
        
        return self


class ValidationConfig(BaseModel):
    """
    Configuration for validating and refining cluster assignments.
    
    After initial clustering, this stage checks that entities in the same cluster
    are truly the same real-world entity by validating names, addresses, and other
    business rules. All operations use GPU-accelerated string matching where possible.
    """
    model_config = ConfigDict(extra='forbid')
    
    street_number_threshold: int = Field(
        default=30, 
        ge=0,
        le=1000,
        description=(
            "Maximum allowed difference between street numbers in the same cluster. "
            "For example, with threshold=50, addresses '123 Main St' and '150 Main St' "
            "could be in the same cluster, but '123 Main St' and '200 Main St' could not. "
            "Set to 0 to require exact street number matches."
        )
    )
    
    address_fuzz_ratio: int = Field(
        default=89, 
        ge=0, 
        le=100,
        description=(
            "Minimum fuzzy string match score (0-100) for addresses in the same cluster. "
            "Uses GPU-accelerated Levenshtein distance. Higher = stricter matching. "
            "85-90 allows minor typos, 95+ requires near-exact matches."
        )
    )
    
    name_fuzz_ratio: int = Field(
        default=89, 
        ge=0, 
        le=100,
        description=(
            "Minimum fuzzy string match score (0-100) for entity names in the same cluster. "
            "Similar to address_fuzz_ratio but applied to entity names. "
            "Can be different from address threshold if names are more/less reliable."
        )
    )
    
    enforce_state_boundaries: bool = Field(
        default=True,
        description=(
            "If True, entities in different states cannot be in the same cluster. "
            "Prevents matching 'Acme Corp' in Texas with 'Acme Corp' in California. "
            "Set to False if entities legitimately span state boundaries."
        )
    )
    
    allow_neighboring_states: List[List[str]] = Field(
        default_factory=list,
        description=(
            "List of state pairs that can be matched across borders. "
            "Example: [['IL', 'WI'], ['NY', 'NJ']] allows Illinois-Wisconsin "
            "and New York-New Jersey matches. Only used if enforce_state_boundaries=True. "
            "Useful for metro areas that span state lines."
        )
    )
    
    profile_comparison_max_pairs_per_chunk: int = Field(
        default=1_000_000, 
        ge=10000,
        le=10_000_000,
        description=(
            "Maximum number of entity pairs to compare in each batch during validation. "
            "Prevents GPU memory overflow on large datasets. Lower values use less VRAM "
            "but take longer. Adjust based on available GPU memory."
        )
    )
    
    reassignment_scoring_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            'name_similarity': 0.40,
            'address_similarity': 0.40,
            'cluster_size': 0.10,
            'cluster_probability': 0.10
        },
        description=(
            "Weights for scoring potential cluster reassignments. Must sum to 1.0.\n"
            "- name_similarity: Weight for name match quality (0.3-0.5 typical)\n"
            "- address_similarity: Weight for address match quality (0.3-0.5 typical)\n"
            "- cluster_size: Prefer larger clusters (0.05-0.15 typical)\n"
            "- cluster_probability: Weight for clustering confidence (0.05-0.15 typical)"
        )
    )
    
    validate_cluster_batch_size: int = Field(
        default=1024, 
        ge=32,
        le=10000,
        description=(
            "Batch size for GPU cluster validation operations. "
            "Larger batches are faster but use more GPU memory."
        )
    )

    @field_validator('reassignment_scoring_weights')
    @classmethod
    def validate_weights_sum(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Ensures reassignment weights sum to 1.0."""
        total = sum(v.values())
        if abs(total - 1.0) > 1.0e-6:
            raise ValueError(
                f"reassignment_scoring_weights must sum to 1.0, but got {total:.6f}. "
                f"Current values: {v}"
            )
        return v
    
    @field_validator('allow_neighboring_states')
    @classmethod
    def validate_state_pairs(cls, v: List[List[str]]) -> List[List[str]]:
        """Validates that state pairs are properly formatted."""
        for i, pair in enumerate(v):
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(
                    f"allow_neighboring_states[{i}] must be a list of exactly 2 state codes, "
                    f"got: {pair}"
                )
            if not all(isinstance(state, str) and len(state) == 2 for state in pair):
                raise ValueError(
                    f"allow_neighboring_states[{i}] must contain 2-letter state codes, "
                    f"got: {pair}"
                )
        return v


class ConfidenceScoringConfig(BaseModel):
    """
    Configuration for calculating final confidence scores for each entity match.
    
    Confidence scores help identify matches that may need manual review and
    provide transparency about match quality. Calculations are GPU-accelerated.
    """
    model_config = ConfigDict(extra='forbid')
    
    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            'cluster_probability': 0.25,
            'name_similarity': 0.20,
            'address_confidence': 0.25,
            'cohesion_score': 0.15,
            'cluster_size_factor': 0.15
        },
        description=(
            "Component weights for final confidence score. Must sum to 1.0.\n"
            "Components:\n"
            "- cluster_probability: HDBSCAN's soft clustering probability (0.2-0.3 typical)\n"
            "- name_similarity: Average name similarity within cluster (0.15-0.25 typical)\n"
            "- address_confidence: Address match quality score (0.2-0.3 typical)\n"
            "- cohesion_score: How tightly clustered entities are (0.1-0.2 typical)\n"
            "- cluster_size_factor: Bonus for larger clusters (0.05-0.15 typical)\n"
            "Adjust weights based on which signals are most reliable in your data."
        )
    )

    @field_validator('weights')
    @classmethod
    def validate_weights_sum(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Ensures confidence weights sum to 1.0."""
        total = sum(v.values())
        if abs(total - 1.0) > 1.0e-6:
            raise ValueError(
                f"Confidence weights must sum to 1.0, but got {total:.6f}. "
                f"Current values: {v}"
            )
        return v


# === Master Configuration ===

class ResolverConfig(BaseModel):
    """
    Master configuration for the GPU-accelerated Entity Resolution pipeline.
    
    This is the main configuration object that orchestrates all stages of the
    entity resolution process using RAPIDS (cuML, cuDF, cuPy) for GPU acceleration.
    It composes all subordinate configurations and ensures consistency across the pipeline.
    
    Usage:
        # Load from YAML file
        config = ResolverConfig.from_yaml('config.yaml')
        
        # Or create with custom parameters
        config = ResolverConfig(
            columns=ColumnConfig(entity_col='company_name'),
            clusterer=ClustererConfig(umap_n_runs=10),
            random_state=42
        )
        
        # Initialize resolver
        resolver = EntityResolver(config)
    
    The configuration follows a hierarchical structure:
    - columns: Input data specification
    - output: Output formatting and review thresholds
    - normalization: Text preprocessing rules
    - vectorizer: Feature extraction and encoding (GPU-accelerated)
    - clusterer: Manifold learning and clustering (GPU-accelerated)
    - validation: Cluster validation and refinement
    - scoring: Confidence score calculation
    - random_state: Global seed for reproducibility
    
    All computationally intensive operations leverage GPU acceleration through RAPIDS.
    """
    model_config = ConfigDict(extra='forbid')
    
    # === Sub-configurations ===
    columns: ColumnConfig = Field(
        default_factory=ColumnConfig,
        description="Configuration for input cuDF DataFrame columns"
    )
    
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Configuration for output formatting and review"
    )
    
    normalization: NormalizationConfig = Field(
        default_factory=NormalizationConfig,
        description="Text normalization and cleaning rules"
    )
    
    vectorizer: VectorizerConfig = Field(
        default_factory=VectorizerConfig,
        description="GPU-accelerated feature extraction and encoding parameters"
    )
    
    clusterer: ClustererConfig = Field(
        default_factory=ClustererConfig,
        description="GPU-accelerated clustering algorithm parameters"
    )
    
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig,
        description="Cluster validation and refinement rules"
    )
    
    scoring: ConfidenceScoringConfig = Field(
        default_factory=ConfidenceScoringConfig,
        description="Confidence scoring weights"
    )
    
    # === Global Configuration ===
    random_state: Optional[int] = Field(
        default=42,
        ge=0,
        le=2**32-1,
        description=(
            "Global random seed for reproducibility. "
            "Set to None for non-deterministic behavior. "
            "This seed is propagated to all stochastic components "
            "(cuML PCA, cuML UMAP, sampling, etc.) to ensure consistent results "
            "across runs with the same data and configuration."
        )
    )

    @model_validator(mode='after')
    def propagate_random_state(self) -> 'ResolverConfig':
        """
        Propagates the global random_state to all components that need it.
        
        This ensures reproducibility across the entire pipeline by using
        the same seed for all stochastic operations in cuML and other libraries.
        """
        if self.random_state is not None:
            # Propagate to cuML PCA components
            self.vectorizer.tfidf_pca_params["random_state"] = self.random_state
            self.vectorizer.phonetic_pca_params["random_state"] = self.random_state
            
            # Propagate to cuML UMAP
            self.clusterer.umap_params["random_state"] = self.random_state
        
        return self

# === Public API ===
__all__ = [
    # Main configuration
    'ResolverConfig',
    # Sub-configurations
    'ColumnConfig', 
    'OutputConfig', 
    'NormalizationConfig', 
    'VectorizerConfig', 
    'ClustererConfig', 
    'ValidationConfig',
    'ConfidenceScoringConfig'
]