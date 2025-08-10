# entity_resolver/config/schema.py
"""
This module defines all Pydantic-based configuration models for the
entity resolution pipeline.

Using Pydantic models provides type-hinting, self-documentation, automatic
validation, and easy serialization. The main ResolverConfig composes all
the subordinate configurations for each stage of the process.
"""

import logging
from typing import List, Dict, Set, Any, Tuple
from pydantic import BaseModel, Field, validator, model_validator

# === Core Data and I/O Configurations ===

class ColumnConfig(BaseModel):
    """Specifies the names of the input DataFrame columns to be used."""
    entity_col: str = Field(
        'company_name', 
        min_length=1,
        description="The column containing the primary entity name (e.g., 'company_name')."
    )
    address_cols: List[str] = Field(
        default_factory=lambda: ['address'], 
        min_items=1,
        description="A list of columns that together form the address (e.g., ['address_line_1', 'city_state_zip'])."
    )

    @validator('address_cols')
    def address_cols_must_be_unique_and_non_empty(cls, v):
        """Ensures address columns are non-empty strings and have no duplicates."""
        seen = set()
        for i, col in enumerate(v):
            if not isinstance(col, str) or not col.strip():
                raise ValueError(f"address_cols[{i}] must be a non-empty string.")
            if col in seen:
                raise ValueError(f"address_cols contains a duplicate column name: '{col}'.")
            seen.add(col)
        return v

class OutputConfig(BaseModel):
    """Configuration for the output format, logging, and review process."""
    output_format: str = Field(
        'proper',
        description="The case style for the final canonical name. 'proper' for Title Case, 'raw' for as-is."
    )
    review_confidence_threshold: float = Field(
        0.75, 
        ge=0.0, 
        le=1.0,
        description="The confidence score below which a match will be flagged for manual review."
    )
    log_level: int = Field(
        logging.INFO,
        description="The logging level for console output (e.g., logging.INFO, logging.DEBUG)."
    )
    split_address_components: bool = Field(
        False,
        description="If True, the final output DataFrame will include separate columns for canonical address components."
    )

# === Preprocessing Configurations ===

class NormalizationConfig(BaseModel):
    """Defines the rules for cleaning and standardizing entity names."""
    replacements: Dict[str, str] = Field(
        default_factory=lambda: {
            "traiier": "trailer", "rpr": "repair", "svcs": "service", "svc": "service",
            "ctr": "center", "ctrs": "centers", "cntr": "center", "trk": "truck",
            "auto": "automotive", "auth": "authorized", "dist": "distribution",
            "mfg": "manufacturing", "mfr": "manufacturing", "equip": "equipment",
            "natl": "national", "mgmt": "management", "assoc": "associates"
        },
        description="A dictionary of common abbreviations or misspellings and their standard form."
    )
    suffixes_to_remove: Set[str] = Field(
        default_factory=lambda: {
            "inc", "incorporated", "llc", "ll", "lp", "llp", "ltd", "limited",
            "corp", "corporation", "co", "company", "plc", "pllc",
            "pa", "pc", "sc", "dba", "fka", "aka", "etal", "et al",
            "international", "intl", "usa", "america", "us",
            "group", "grp", "holdings", "ent"
        },
        description="A set of common legal and organizational suffixes to be removed from names."
    )

# === Modeling Stage Configurations ===

class VectorizerConfig(BaseModel):
    """Parameters for the vectorization and initial dimensionality reduction stages."""
    encoders: List[str] = Field(
        default_factory=lambda: ['tfidf', 'phonetic', 'semantic'],
        description="A list of feature streams to generate. Options: 'tfidf', 'phonetic', 'semantic'."
    )
    sparse_reducers: List[str] = Field(
        default_factory=lambda: ['svd', 'pca'],
        description="A list of dimensionality reduction techniques for sparse vectors. Options: 'svd', 'pca'."
    )
    use_address_in_encoding: bool = Field(
        True,
        description="If True, the normalized address will be appended to the entity name for the TF-IDF stream."
    )
    damping_beta: float = Field(
        0.4, 
        ge=0.0, 
        le=1.0,
        description="Controls the strength of the spectral damping. beta=1.0 is full whitening, beta=0.0 is no whitening. 0.4 is a good default."
    )
    epsilon: float = Field(
        1e-8, 
        gt=0,
        description="Small constant to prevent division by zero in numerical operations."
    )
    
    # --- Semantic Stream ---
    semantic_model: str = Field(
        'all-mpnet-base-v2', 
        min_length=1,
        description="Sentence-transformer model for semantic encoding. Produces 768 features."
    )
    semantic_batch_size: int = Field(512, gt=0)

    # --- TF-IDF Stream ---
    tfidf_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "analyzer": "char", "ngram_range": (3, 5), "max_features": 10000,
            "min_df": 3, "sublinear_tf": True, "dtype": "float64",
        },
        description="Parameters for the TfidfVectorizer."
    )
    tfidf_svd_params: Dict[str, Any] = Field(
        default_factory=lambda: {'n_components': 2048},
        description="Parameters for TruncatedSVD applied to TF-IDF vectors."
    )
    tfidf_pca_params: Dict[str, Any] = Field(
        default_factory=lambda: {'n_components': 768},
        description="Parameters for PCA applied to TF-IDF vectors."
    )

    # --- Phonetic Stream ---
    phonetic_max_words: int = Field(5, gt=0, description="Maximum number of words to consider for phonetic encoding.")
    phonetic_params: Dict[str, Any] = Field(
        default_factory=lambda: {'analyzer': 'word', 'binary': True, 'max_features': 2000},
        description="Parameters for CountVectorizer on phonetic codes."
    )
    phonetic_svd_params: Dict[str, Any] = Field(
        default_factory=lambda: {'n_components': 384},
        description="Parameters for TruncatedSVD applied to phonetic vectors."
    )
    phonetic_pca_params: Dict[str, Any] = Field(
        default_factory=lambda: {'n_components': 256},
        description="Parameters for PCA applied to phonetic vectors."
    )

    # --- Stream Balancing ---
    stream_proportions: Dict[str, float] = Field(
        default_factory=lambda: {'semantic': 0.45, 'tfidf': 0.40, 'phonetic': 0.15},
        description="Defines the desired final variance contribution of each stream. Must sum to 1.0."
    )
    
    # --- String Matching ---
    similarity_tfidf: Dict[str, Any] = Field(
        default_factory=lambda: {
            'analyzer': 'char', 'ngram_range': (3, 5), 'min_df': 2,
            'sublinear_tf': True, 'norm': 'l2'
        },
        description="TF-IDF parameters for the string similarity search fallback."
    )
    similarity_nn: Dict[str, Any] = Field(
        default_factory=lambda: {'n_neighbors': 24, 'metric': 'cosine'},
        description="Nearest neighbor parameters for the string similarity search fallback."
    )

    @validator('encoders')
    def check_encoders(cls, v):
        """Validates that encoders are from the allowed set."""
        allowed = {"tfidf", "phonetic", "semantic"}
        if not set(v).issubset(allowed):
            raise ValueError(f"Encoders must be a subset of {allowed}.")
        return v

    @validator('stream_proportions')
    def check_proportions_sum(cls, v):
        """Validates that stream proportions sum to 1.0."""
        if abs(sum(v.values()) - 1.0) > 1e-6:
            raise ValueError(f"stream_proportions must sum to 1.0, but sums to {sum(v.values())}")
        return v

class ClustererConfig(BaseModel):
    """Parameters for the manifold learning and clustering stages."""
    # --- UMAP Ensemble ---
    umap_n_runs: int = Field(3, ge=1, description="Number of UMAP models in the ensemble.")
    umap_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "n_neighbors": 15, "n_components": 48, "min_dist": 0.05, "spread": 0.5,
            "metric": "cosine", "init": "spectral", "n_epochs": 400,
            "negative_sample_rate": 7, "repulsion_strength": 1.0, "learning_rate": 0.5,
        },
        description="Default parameters for the UMAP models in the ensemble."
    )
    umap_ensemble_sampling_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            'local_view_ratio': 0.7, 'n_neighbors_local': (10, 35),
            'n_neighbors_global': (50, 70), 'min_dist': (0.0, 0.15),
            'spread': (0.5, 2.0), 'n_epochs': (200, 500), 'learning_rate': (0.5, 1.5),
            'repulsion_strength': (0.5, 1.5), 'negative_sample_rate': (5, 20),
            'init_strategies': ['spectral', 'random']
        },
        description="Defines the search space for randomized UMAP parameters in the ensemble."
    )
    
    # --- Kernel PCA Consensus Embedding ---
    cosine_consensus_n_samples: int = Field(8192, gt=0)
    cosine_consensus_batch_size: int = Field(2048, gt=0)
    
    # --- HDBSCAN Clustering ---
    max_noise_rate_warn: float = Field(0.95, ge=0.0, le=1.0, description="High-noise warning threshold.")
    hdbscan_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            'min_cluster_size': 3, 'min_samples': 1, 'cluster_selection_epsilon': 0.15,
            'prediction_data': True, 'alpha': 0.8, 'cluster_selection_method': 'leaf'
        },
        description="Parameters for the HDBSCAN clustering algorithm."
    )

    # --- SNN Graph Clustering Engine ---
    snn_clustering_params: Dict[str, Any] = Field(
        default_factory=lambda: {'k_neighbors': 40, 'louvain_resolution': 0.65}
    )
    noise_attachment_params: Dict[str, Any] = Field(
        default_factory=lambda: {'k': 15, 'tau': 0.82, 'min_matching': 2, 'ratio_threshold': 1.5}
    )

    # --- SNN Cluster Merging ---
    merge_median_threshold: float = Field(0.84, ge=0.0, le=1.0, description="The median similarity between two clusters must exceed this to be considered for a merge.")
    merge_max_threshold: float = Field(0.90, ge=0.0, le=1.0, description="The maximum similarity (single most similar pair) must also exceed this.")
    merge_sample_size: int = Field(20, gt=0, description="To avoid O(n^2) comparisons, sample this many points from large clusters for the check.")
    centroid_similarity_threshold: float = Field(0.75, ge=0.0, le=1.0, description="In the pre-filtering step, only cluster pairs with centroid similarity above this are checked in detail.")
    merge_batch_size: int = Field(2048, gt=0)
    centroid_sample_size: int = Field(2048, gt=0)

    # --- Ensemble of HDBSCAN and SNN ---
    ensemble_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            'purity_min': 0.75, 'min_overlap': 2, 'allow_new_snn_clusters': True,
            'min_newcluster_size': 4, 'default_rescue_conf': 0.60
        }
    )
    
    @model_validator(mode='after')
    def check_umap_spread(self) -> 'ClustererConfig':
        """Validates that UMAP spread is greater than min_dist."""
        if self.umap_params.get('spread') <= self.umap_params.get('min_dist'):
            raise ValueError("In umap_params, 'spread' must be strictly greater than 'min_dist'.")
        return self

class ValidationConfig(BaseModel):
    """Parameters for validating cluster membership and refining clusters."""
    street_number_threshold: int = Field(50, ge=0, description="Maximum difference allowed between street numbers within the same cluster.")
    address_fuzz_ratio: int = Field(87, ge=0, le=100, description="Minimum fuzzy match ratio (0-100) required for an address to be considered valid in a cluster.")
    name_fuzz_ratio: int = Field(89, ge=0, le=100, description="Minimum fuzzy match ratio (0-100) required for a name to be considered valid in a cluster.")
    enforce_state_boundaries: bool = Field(True, description="If True, entities in different states will not be placed in the same cluster.")
    allow_neighboring_states: List[Tuple[str, str]] = Field(default_factory=list, description="A list of state pairs that are allowed to be matched across borders (e.g., [('IL', 'WI')]).")
    profile_comparison_max_pairs_per_chunk: int = Field(1_000_000, gt=0, description="To prevent OOM errors, this limits the size of the cross-join during reassignment.")
    
    reassignment_scoring_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            'name_similarity': 0.40, 'address_similarity': 0.40,
            'cluster_size': 0.10, 'cluster_probability': 0.10
        },
        description="Weights for scoring potential new clusters during the reassignment step. Must sum to 1.0."
    )
    validate_cluster_batch_size: int = Field(2000, gt=0)

    @validator('reassignment_scoring_weights')
    def check_weights_sum(cls, v):
        """Validates that reassignment weights sum to 1.0."""
        if abs(sum(v.values()) - 1.0) > 1e-6:
            raise ValueError(f"reassignment_scoring_weights must sum to 1.0, but sums to {sum(v.values())}")
        return v

class ConfidenceScoringConfig(BaseModel):
    """Weights for calculating the final confidence score for each match."""
    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            'cluster_probability': 0.25, 'name_similarity': 0.20, 'address_confidence': 0.25,
            'cohesion_score': 0.15, 'cluster_size_factor': 0.15
        },
        description="These weights determine the contribution of each factor to the final score. Must sum to 1.0."
    )

    @validator('weights')
    def check_weights_sum(cls, v):
        """Validates that confidence weights sum to 1.0."""
        if abs(sum(v.values()) - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, but sums to {sum(v.values())}")
        return v

# === Master Configuration ===

class ResolverConfig(BaseModel):
    """
    Master configuration object that composes all subordinate configurations.
    
    An instance of this class is the single object needed to initialize the
    main EntityResolver, providing access to all pipeline parameters.
    """
    columns: ColumnConfig = Field(default_factory=ColumnConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    vectorizer: VectorizerConfig = Field(default_factory=VectorizerConfig)
    clusterer: ClustererConfig = Field(default_factory=ClustererConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    scoring: ConfidenceScoringConfig = Field(default_factory=ConfidenceScoringConfig)
    random_state: int = 42

    @model_validator(mode='after')
    def propagate_random_state(self) -> 'ResolverConfig':
        """
        Propagates the global random_state to components that require it
        for deterministic, reproducible results. This replaces __post_init__.
        """
        self.vectorizer.tfidf_pca_params["random_state"] = self.random_state
        self.vectorizer.phonetic_pca_params["random_state"] = self.random_state
        self.clusterer.umap_params["random_state"] = self.random_state
        return self

    class Config:
        # This is needed to allow types like logging.INFO which are not native Pydantic types
        arbitrary_types_allowed = True

__all__ = [
    'ColumnConfig', 
    'OutputConfig', 
    'NormalizationConfig', 
    'VectorizerConfig', 
    'ClustererConfig', 
    'ValidationConfig',
    'ConfidenceScoringConfig', 
    'ResolverConfig'
]