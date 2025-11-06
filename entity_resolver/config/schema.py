# entity_resolver/config/schema.py
"""
Pydantic Schema for GPU-Accelerated Entity Resolution Pipeline Configuration.

This module serves as the single source of truth for all configuration parameters
governing the entity resolution and deduplication pipeline. It leverages Pydantic
to define a clear, hierarchical, and type-safe schema. All computationally intensive
stages of the pipeline are designed to be accelerated on NVIDIA GPUs using the
RAPIDS ecosystem (cuDF, cuML, cuPy).

The use of Pydantic provides several key advantages:
1.  **Type Safety & Validation**: All configuration values are automatically parsed,
    validated, and cast to their correct types at runtime. This prevents common
    configuration errors and ensures that downstream components receive data in
    the expected format.
2.  **Rich Error Messages**: When validation fails, Pydantic raises descriptive
    errors that pinpoint exactly which parameter is incorrect and why, dramatically
    simplifying debugging.
3.  **Self-Documentation**: The schema itself, with its type hints, descriptions,
    and default values, acts as comprehensive documentation for every available
    pipeline parameter.
4.  **IDE Support**: The strongly-typed nature of the configuration objects provides
    excellent autocompletion, type-checking, and navigation in modern IDEs like
    VS Code or PyCharm.
5.  **Serialization**: Configurations can be easily loaded from and saved to common
    formats like YAML or JSON, facilitating reproducibility and experiment tracking.
6.  **Strictness**: By setting `extra='forbid'`, the schema prevents unknown or
    misspelled parameters from being passed in, catching potential typos early.

The primary entry point is the `ResolverConfig` class, which aggregates all
stage-specific configurations (e.g., `VectorizerConfig`, `ClustererConfig`) into
a single, cohesive object.
"""

# ======================================================================================
# Core Library Imports
# ======================================================================================

# --- Standard Library Imports ---
# The `typing` module provides support for type hints, which are crucial for making
# the Pydantic models explicit and enabling static analysis.
from typing import Any, Literal

# RAPIDS cuPy is the GPU-accelerated equivalent of NumPy. It is imported here
# primarily to handle and validate GPU-specific data types (e.g., `cupy.dtype`)
# that are passed to other RAPIDS libraries like cuML.
import cupy

# --- Third-Party Library Imports ---
# Pydantic is the core library used for data validation and settings management.
# Its models define the structure, types, and constraints of the configuration.
from pydantic import (
    BaseModel,  # The base class for all configuration models.
    ConfigDict,  # A dictionary-like object for configuring model behavior.
    Field,  # Used to customize model fields with defaults, validation, etc.
    field_validator,  # A decorator for creating custom per-field validation logic.
    model_validator,  # A decorator for creating custom validation logic for the entire model.
)

# ======================================================================================
# Re-usable Helper Models & Validators
# ======================================================================================


def _validate_dtype_string(v: Any) -> str:
    """
    Shared helper to validate that a raw input value is a valid CuPy float dtype string.

    This is a 'before' validator, so it accepts `Any` to handle potentially
    invalid user input (e.g., an integer or None). It then enforces that the
    input is a string representing a floating-point type.
    """
    if not isinstance(v, str):
        raise TypeError("Dtype must be provided as a string (e.g., 'float64').")
    try:
        dtype = cupy.dtype(v)
        # Enforce floating-point types, as integer dtypes are unsuitable for these matrices.
        if dtype.kind != 'f':
            raise ValueError(f"Dtype must be a floating-point type, but '{v}' is not.")
    except TypeError as e:
        raise ValueError(f"'{v}' is not a valid cupy dtype name.") from e
    return v


def _convert_string_to_dtype(v: str) -> Any:
    """
    Shared helper to convert a validated string into a cupy.dtype object.

    This is an 'after' validator. It runs after _validate_dtype_string, so it
    can safely assume `v` is a valid dtype string and proceeds with conversion.
    The return type is `Any` because `cupy.dtype` is not a standard type.
    """
    return cupy.dtype(v)


# ======================================================================================
# Helper & Utility Configurations
# ======================================================================================


class SvdEigshFallbackConfig(BaseModel):
    """
    Configuration for the robust SVD fallback mechanism.

    This configuration governs a series of preprocessing and solver steps that are
    invoked when the primary sparse SVD solver (`cupyx.scipy.sparse.linalg.svds`)
    fails to converge. Such failures can occur with ill-conditioned or rank-deficient
    matrices. This fallback uses a more stable, albeit potentially slower, method based
    on the eigenvalue decomposition of the covariance matrix (eigsh), combined with
    aggressive data pruning and cleaning to improve numerical stability.
    """

    # Configure the Pydantic model.
    # - `extra='forbid'`: Disallows any fields not explicitly defined in the model.
    # - `arbitrary_types_allowed=True`: Explicitly permits non-standard types like
    #   `cupy.dtype` to be stored in the model's fields after validation. This is
    #   necessary because we convert the string 'float64' into a cupy.dtype object.
    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)

    # --- Solver Parameters ---
    fallback_dtype: Any = Field(
        default='float64',
        description=(
            "The CuPy dtype to use for the eigsh solver. 'float64' (double precision) "
            'is strongly recommended as it provides greater numerical stability, which '
            'is critical in a fallback scenario where the matrix may be ill-conditioned. '
            "'float32' can be used to conserve memory but increases the risk of non-convergence."
        ),
    )
    eigsh_restarts: int = Field(
        default=3,
        ge=0,
        le=10,
        description=(
            'Number of times to restart the `eigsh` solver on convergence failure. '
            'Each restart can use a slightly different internal state, sometimes '
            'allowing it to find a solution. Setting this above 0 adds resilience. '
            'A value between 1 and 5 is typical.'
        ),
    )

    # --- Pruning & Clipping Parameters ---
    prune_min_row_sum: float = Field(
        default=1e-9,
        ge=0.0,
        description=(
            'Rows in the feature matrix with a total sum of values less than this '
            'threshold will be removed before attempting SVD. This helps eliminate '
            'near-empty or zero-energy records that contribute little information '
            'but can cause numerical instability.'
        ),
    )
    prune_min_df: int = Field(
        default=2,
        ge=1,
        description=(
            'The minimum document frequency (DF). Columns (features) that appear '
            'in fewer than `min_df` documents (rows) will be removed. This is a standard '
            'technique to filter out rare, noisy features that are unlikely to be useful.'
        ),
    )
    prune_max_df_ratio: float = Field(
        default=0.98,
        gt=0.0,
        le=1.0,
        description=(
            'The maximum document frequency (DF) as a ratio of the total number of '
            'documents. Columns that appear in more than this fraction of documents '
            'will be removed. This filters out overly common, non-discriminative '
            "features (e.g., character n-grams like ' co' in company names)."
        ),
    )
    prune_energy_cutoff: float = Field(
        default=0.995,
        gt=0.0,
        le=1.0,
        description=(
            'A threshold for pruning columns based on their cumulative energy (sum of '
            'squared values). The process keeps the smallest set of columns whose '
            'cumulative energy exceeds this ratio of the total, effectively removing '
            'low-energy, low-information features.'
        ),
    )
    winsorize_limits: tuple[float | None, float | None] = Field(
        default=(None, 0.999),
        description=(
            'Quantile-based limits for clipping (Winsorizing) extreme values in the matrix. '
            'This is a robust way to handle outliers that can destabilize SVD. For example, '
            '`(0.01, 0.99)` clips all values below the 1st percentile and above the 99th. '
            'Use `None` to disable clipping on one side (e.g., `(None, 0.999)` only clips high values).'
        ),
    )

    # Assign the shared validators to this model's field.
    _validate_dtype_str: classmethod = field_validator('fallback_dtype', mode='before')(
        _validate_dtype_string
    )
    _convert_to_dtype: classmethod = field_validator('fallback_dtype', mode='after')(
        _convert_string_to_dtype
    )

    @field_validator('winsorize_limits')
    @classmethod
    def validate_winsorize_limits(
        cls,
        v: tuple[float | None, float | None],
    ) -> tuple[float | None, float | None]:
        """
        Validates that the winsorize limits tuple is logical and within the [0, 1] range.
        """
        lower, upper = v
        if lower is not None and not (0.0 <= lower <= 1.0):
            raise ValueError(f'Lower winsorize limit must be between 0.0 and 1.0, got {lower}')
        if upper is not None and not (0.0 <= upper <= 1.0):
            raise ValueError(f'Upper winsorize limit must be between 0.0 and 1.0, got {upper}')
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError(
                f'Lower limit ({lower}) cannot be greater than or equal to the upper limit ({upper}).'
            )
        return v


# ======================================================================================
# Core Data and I/O Configurations
# ======================================================================================


class ColumnConfig(BaseModel):
    """
    Specifies the names of the input DataFrame columns for the resolution process.

    This crucial configuration acts as a map, telling the pipeline where to find the
    primary entity identifier (e.g., a company name) and the associated address
    component(s) within the input cuDF DataFrame. The pipeline will raise an error
    during initialization if these columns are not found in the provided data.
    """

    model_config = ConfigDict(extra='forbid')

    entity_col: str = Field(
        default='entity',
        min_length=1,
        max_length=256,
        pattern=r'^[A-Za-z](?:[A-Za-z0-9_\s]*[A-Za-z0-9_])?$',
        description=(
            'The name of the column containing the primary entity identifier to be resolved. '
            'This is typically the main business or organization name. All matching and '
            'clustering logic is centered around this column. The name must be a valid '
            'identifier (letters, numbers, underscores) and cannot start with a number.'
            "Common examples: 'company_name', 'Vendor Name', 'organization'."
        ),
    )

    address_cols: list[str] = Field(
        default_factory=lambda: ['address'],
        min_length=1,
        max_length=10,  # Limit to a reasonable number of address components
        description=(
            'A list of one or more column names that constitute the full address. '
            'These columns are concatenated in the provided order, separated by spaces, '
            'to form a single address string for processing. The order is critical. '
            "For structured data, a list like `['street', 'city', 'state', 'zip']` "
            "is appropriate. For unstructured data, a single column `['full_address']` "
            'can be used.'
        ),
    )

    @field_validator('address_cols')
    @classmethod
    def validate_address_columns(cls, v: list[str]) -> list[str]:
        """
        Validates that address columns are non-empty, unique, valid identifier strings.

        This check prevents common errors such as empty strings, leading or trailing whitespace, or duplicate
        column names being passed in the address column list, which could lead to
        unexpected behavior or errors during processing.

        Args:
            v: The list of address column names from the configuration.

        Returns:
            The validated list of address column names.

        Raises:
            ValueError: If the list is empty, any column name is invalid, or if duplicates exist.
        """
        if not v:
            raise ValueError('`address_cols` cannot be an empty list.')

        seen = set()
        validated_list = []
        for i, col in enumerate(v):
            if not isinstance(col, str):
                raise TypeError(
                    f'address_cols[{i}] must be a string, but got type {type(col).__name__}'
                )

            stripped_col = col.strip()
            if not stripped_col:
                raise ValueError(
                    f"address_cols[{i}] ('{col}') cannot be empty or contain only whitespace."
                )
            if col != stripped_col:
                raise ValueError(
                    f"address_cols[{i}] ('{col}') cannot have leading or trailing whitespace."
                )

            if stripped_col in seen:
                raise ValueError(
                    f"Duplicate column name found in address_cols: '{stripped_col}'. All column names must be unique."
                )

            seen.add(stripped_col)
            validated_list.append(stripped_col)
        return validated_list

    @model_validator(mode='after')
    def check_column_overlap(self) -> 'ColumnConfig':
        """
        Ensures the entity_col is not duplicated in the address_cols list.

        The entity column must be distinct from the address columns to maintain a clear
        separation between the primary identifier and its location attributes.
        This validation prevents logical errors in the downstream feature engineering.
        """
        if self.entity_col in self.address_cols:
            raise ValueError(
                f"The `entity_col` ('{self.entity_col}') cannot also be present in the "
                f'`address_cols` list: {self.address_cols}'
            )
        return self


class OutputConfig(BaseModel):
    """
    Configuration for output formatting, logging verbosity, and manual review thresholds.

    Controls how the final resolved entities are formatted, which entities should be
    flagged for human review based on confidence scores, and also manages the
    verbosity of the pipeline's logging output during its run.
    """

    model_config = ConfigDict(extra='forbid')

    output_format: Literal['proper', 'raw', 'upper', 'lower'] = Field(
        default='proper',
        description=(
            'The case style for the final canonical entity names:\n'
            "- 'proper': Applies title casing (e.g., 'Acme Corporation').\n"
            "- 'raw': Preserves the original casing from the most representative cluster member.\n"
            "- 'upper': Converts all names to UPPERCASE (e.g., 'ACME CORPORATION').\n"
            "- 'lower': Converts all names to lowercase (e.g., 'acme corporation')."
        ),
    )

    review_confidence_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description=(
            'The confidence score threshold for flagging matches for manual review. '
            'Matches with a confidence score *below* this value will be marked. '
            'This is a key parameter for controlling the precision-recall trade-off. '
            'Set higher (e.g., 0.8-0.9) for high precision (fewer, more certain matches). '
            'Set lower (e.g., 0.6-0.7) for high recall (more matches, some potentially incorrect).'
        ),
    )

    log_level: int | str = Field(
        default='INFO',
        description=(
            'The logging verbosity level. Can be specified as a standard logging integer '
            'or a case-insensitive string:\n'
            "- 'DEBUG' (10): Detailed diagnostic output, useful for troubleshooting.\n"
            "- 'INFO' (20): General messages about pipeline progress (recommended default).\n"
            "- 'WARNING' (30): Only warnings about potential issues and errors.\n"
            "- 'ERROR' (40): Only error messages.\n"
            "- 'CRITICAL' (50): Only critical failures that halt execution."
        ),
    )

    split_address_components: bool = Field(
        default=False,
        description=(
            'If True, the output cuDF DataFrame will include separate columns for each '
            'canonical address component (e.g., street, city, state, zip), parsed from the '
            'canonical address. If False, only a single concatenated `canonical_address` '
            'column is included. Enable this if downstream processes require structured addresses.'
        ),
    )

    @field_validator('log_level', mode='before')
    @classmethod
    def validate_and_normalize_log_level(cls, v: int | str) -> int:
        """Converts string log levels to their integer equivalents and validates them."""

        log_level_map = {'DEBUG': 10, 'INFO': 20, 'WARNING': 30, 'ERROR': 40, 'CRITICAL': 50}

        if isinstance(v, str):
            upper_v = v.upper()
            if upper_v in log_level_map:
                return log_level_map[upper_v]
            raise ValueError(
                f"Invalid log level string: '{v}'. Must be one of {list(log_level_map.keys())}"
            )

        if isinstance(v, int):
            if v in log_level_map.values():
                return v
            raise ValueError(
                f'Invalid log level integer: {v}. Must be one of {list(log_level_map.values())}'
            )

        raise TypeError(f'log_level must be a string or an integer, not {type(v).__name__}')


# ======================================================================================
# Preprocessing Configurations
# ======================================================================================


class NormalizationConfig(BaseModel):
    """
    Defines rules for cleaning and standardizing entity names before matching.

    This preprocessing step is crucial for matching variations of the same entity.
    It handles common abbreviations and removes "noise" terms like legal suffixes
    that often don't contribute to the core identity of an entity. All operations
    are GPU-accelerated using cuDF string methods for high performance.
    """

    model_config = ConfigDict(extra='forbid')

    replacements: dict[str, str] = Field(
        default_factory=lambda: {
            'svc': 'service',
            'svcs': 'services',
            'equip': 'equipment',
            'mfg': 'manufacturing',
            'dist': 'distribution',
        },
        description=(
            'A dictionary mapping common abbreviations, acronyms, or misspellings to their '
            'standardized forms. This operation is case-insensitive as it is applied '
            'AFTER the entire string has been converted to lowercase. '
            "Example: Both 'Svc' and 'SVC' will become 'service'."
        ),
    )

    suffixes_to_remove: set[str] = Field(
        default_factory=lambda: {
            # Standard legal entity types
            'inc',
            'incorporated',
            'llc',
            'l l c',
            'limited liability company',
            'lp',
            'l p',
            'llp',
            'l l p',
            'ltd',
            'limited',
            'corp',
            'corporation',
            'co',
            'company',
            'plc',
            'pc',
            'pllc',
            # Common business indicators
            'group',
            'holding',
            'holdings',
            'associates',
            'partners',
            'sons',
            # Common "Doing Business As" indicators
            'dba',
            'fka',
            'aka',
            'c o',
            'o b o',
        },
        description=(
            'A set of common legal and organizational suffixes to remove from entity names. '
            'Removal is case-insensitive and happens after custom replacements. The logic '
            'removes these terms only when they appear as whole words. '
            "Example: 'Acme Corp LLC' becomes 'acme'."
        ),
    )

    @field_validator('replacements')
    @classmethod
    def validate_and_clean_replacements(cls, v: dict[str, str]) -> dict[str, str]:
        """Ensures replacement keys/values are clean, lowercase, non-empty strings."""
        cleaned_replacements = {}
        for key, value in v.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(
                    f"Replacement dictionary key must be a non-empty string, but got: '{key}'"
                )
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Replacement dictionary value for key '{key}' must be a non-empty string, but got: '{value}'"
                )

            cleaned_key = key.lower().strip()
            cleaned_value = value.lower().strip()

            if not cleaned_key or not cleaned_value:
                raise ValueError(
                    'Replacement keys and values cannot be empty after stripping whitespace.'
                )

            cleaned_replacements[cleaned_key] = cleaned_value
        return cleaned_replacements

    @field_validator('suffixes_to_remove')
    @classmethod
    def validate_and_clean_suffixes(cls, v: set[str]) -> set[str]:
        """Ensures all suffixes are clean, lowercase, non-empty strings."""
        cleaned_suffixes = set()
        for item in v:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"Each item in `suffixes_to_remove` must be a non-empty string, but got: '{item}'"
                )

            cleaned_item = item.lower().strip()
            if not cleaned_item:
                raise ValueError('Suffixes cannot be empty after stripping whitespace.')

            cleaned_suffixes.add(cleaned_item)
        return cleaned_suffixes

    @model_validator(mode='after')
    def check_replacement_suffix_overlap(self) -> 'NormalizationConfig':
        """
        Ensures that no term exists in both replacements and suffixes_to_remove.

        A term cannot be both replaced and removed, as this creates ambiguity in the
        preprocessing pipeline. This validator prevents such logical conflicts in the
        configuration, forcing a clear choice for each normalization term.
        """
        replacement_keys = set(self.replacements.keys())
        overlap = replacement_keys.intersection(self.suffixes_to_remove)

        if overlap:
            raise ValueError(
                'The following terms were found in both `replacements` and `suffixes_to_remove`, '
                f'which is not allowed: {sorted(list(overlap))}. Please specify each term '
                'in only one of the two configurations.'
            )
        return self


# ======================================================================================
# Typed Sub-Models for Vectorizer Configuration
# ======================================================================================


class TfidfParams(BaseModel):
    """Parameters for the cuML TfidfVectorizer, which converts text into a matrix of TF-IDF features."""

    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)
    analyzer: Literal['char', 'word', 'char_wb'] = Field(
        default='char',
        description="Level at which to generate features. 'char' is robust to typos. "
        "'char_wb' is similar but only creates n-grams from text inside word boundaries. "
        "The cuML TfidfVectorizer has had bugs with 'char_wb', so we default to 'char'",
    )
    ngram_range: tuple[int, int] = Field(
        default=(3, 5),
        description='Range of n-gram sizes to extract. E.g., (3, 5) extracts 3-grams, 4-grams, and 5-grams.',
    )
    max_features: int | None = Field(
        default=10000,
        gt=0,
        description='Build a vocabulary that only considers the top `max_features` ordered by term frequency.',
    )
    max_df: float | int = Field(
        default=0.99,
        description='Ignore terms with a document frequency strictly higher than the given threshold (corpus-specific stop words).',
    )
    min_df: float | int = Field(
        default=2,
        description='Minimum document frequency. A feature must appear in at least this many documents to be included.',
    )
    sublinear_tf: bool = Field(
        default=True,
        description='Apply sublinear TF scaling (replaces tf with log(tf)). Dampens the effect of very frequent terms.',
    )
    norm: Literal['l1', 'l2', None] = Field(
        default='l2',
        description="Vector normalization type. 'l2' (Euclidean) is standard for cosine similarity.",
    )
    dtype: Any = Field(
        default='float64',
        description="Data type for the output matrix. 'float32' saves GPU memory, 'float64' offers more precision.",
    )

    _validate_dtype_str: classmethod = field_validator('dtype', mode='before')(
        _validate_dtype_string
    )
    _convert_to_dtype: classmethod = field_validator('dtype', mode='after')(
        _convert_string_to_dtype
    )

    @field_validator('ngram_range')
    @classmethod
    def validate_ngram_range(cls, v: tuple[int, int]) -> tuple[int, int]:
        """Ensures the n-gram range is logically ordered and contains positive integers."""
        low, high = v
        if not (isinstance(low, int) and isinstance(high, int) and low > 0 and high > 0):
            raise ValueError('ngram_range values must be positive integers.')
        if low > high:
            raise ValueError(
                f'In ngram_range, the lower bound ({low}) cannot be greater than the upper bound ({high}).'
            )
        return v

    @field_validator('min_df', 'max_df')
    @classmethod
    def validate_df(cls, v: float | int, field_info) -> float | int:
        """Validates that min_df and max_df values are within their logical ranges."""
        name = field_info.field_name
        if isinstance(v, float):
            if not (0.0 <= v <= 1.0):
                raise ValueError(f'If `{name}` is a float, it must be in the range [0.0, 1.0].')
        elif isinstance(v, int):
            if v < 1:
                raise ValueError(f'If `{name}` is an int, it must be >= 1.')
        return v

    def to_kwargs(self) -> dict:
        """Return parameters for TfidfVectorizer initialization."""
        return {
            'analyzer': self.analyzer,
            'ngram_range': self.ngram_range,
            'max_features': self.max_features,
            'max_df': self.max_df,
            'min_df': self.min_df,
            'sublinear_tf': self.sublinear_tf,
            'norm': self.norm,
            'dtype': self.dtype,  # Will pass the actual cupy.dtype object
        }


class TfidfSvdParams(BaseModel):
    """
    Parameters for the custom GPUTruncatedSVD on TF-IDF features.
    SVD is excellent for topic modeling and latent semantic analysis.
    """

    model_config = ConfigDict(extra='forbid')
    n_components: int = Field(
        default=512, gt=0, description='The target number of dimensions for the reduced data.'
    )
    tol: float = Field(
        default=1.0e-7,
        gt=0,
        description="Tolerance for the SVD solver's convergence. Smaller is more precise but slower.",
    )
    ncv: int | None = Field(
        default=None,
        gt=0,
        description='Number of Lanczos vectors. Controls memory usage and convergence speed.',
    )
    maxiter: int | None = Field(
        default=None, gt=0, description='Maximum number of iterations for the SVD solver.'
    )

    @model_validator(mode='after')
    def set_and_validate_ncv(self) -> 'TfidfSvdParams':
        """Sets a robust default for `ncv` if not provided and ensures it's valid."""
        if self.ncv is None:
            # Set a robust default ncv based on n_components, capped to avoid excessive memory use.
            self.ncv = min(max(2 * self.n_components + 1, 32), 4096)

        if self.ncv <= self.n_components:
            raise ValueError(
                f'`ncv` ({self.ncv}) must be strictly greater than `n_components` '
                f'({self.n_components}) for the underlying ARPACK SVD solver to function correctly.'
            )
        return self


class TfidfPcaParams(BaseModel):
    """
    Parameters for cuML PCA on TF-IDF features. PCA is effective at finding
    directions of maximum variance.
    """

    model_config = ConfigDict(extra='forbid')
    n_components: int = Field(
        default=128, gt=0, description='The target number of dimensions for the reduced data.'
    )
    random_state: int | None = Field(
        default=None,
        exclude=True,
        description='Seed for reproducibility, propagated from global config.',
    )


class PhoneticParams(BaseModel):
    """
    Parameters for the cuML CountVectorizer on phonetic codes, creating a
    feature space based on how words sound.
    """

    model_config = ConfigDict(extra='forbid')
    analyzer: Literal['word'] = Field(
        default='word',
        description="Must be 'word' to treat each phonetic code as a distinct token.",
    )
    binary: bool = Field(
        default=True,
        description='If True, all non-zero counts are set to 1. Measures presence, not frequency.',
    )
    max_features: int | None = Field(
        default=2000, gt=0, description='Maximum number of unique phonetic codes to consider.'
    )
    min_df: float | int = Field(
        default=1,
        description='Ignore terms with a document frequency strictly lower than the given threshold.',
    )

    @field_validator('min_df')
    @classmethod
    def validate_min_df(cls, v: float | int) -> float | int:
        """Validates that min_df value is within its logical range."""
        if isinstance(v, float):
            if not (0.0 <= v <= 1.0):
                raise ValueError('If `min_df` is a float, it must be in the range [0.0, 1.0].')
        elif isinstance(v, int):
            if v < 1:
                raise ValueError('If `min_df` is an int, it must be >= 1.')
        return v


class PhoneticSvdParams(BaseModel):
    """Parameters for GPUTruncatedSVD on phonetic features."""

    model_config = ConfigDict(extra='forbid')
    n_components: int = Field(
        default=256,
        gt=0,
        description='Target dimensionality. Phonetic space is usually less complex than TF-IDF.',
    )


class PhoneticPcaParams(BaseModel):
    """Parameters for cuML PCA on phonetic features."""

    model_config = ConfigDict(extra='forbid')
    n_components: int = Field(
        default=160, gt=0, description='Target dimensionality for the PCA reduction stage.'
    )
    random_state: int | None = Field(
        default=None,
        exclude=True,
        description='Seed for reproducibility, propagated from global config.',
    )


class SimilarityTfidfParams(BaseModel):
    """
    Parameters for the fallback TF-IDF model, used for high-precision
    pairwise string similarity checks.
    """

    model_config = ConfigDict(extra='forbid')

    # Explicitly define all fields you want (excluding dtype)
    analyzer: Literal['char', 'word', 'char_wb'] = Field(
        default='char', description='Level at which to generate features.'
    )
    ngram_range: tuple[int, int] = Field(
        default=(3, 5), description='Range of n-gram sizes to extract.'
    )
    max_features: int | None = Field(
        default=50000,
        gt=0,
        description='Uses a larger vocabulary for more granular similarity comparisons.',
    )
    max_df: float | int = Field(
        default=0.99,
        description='Ignore terms with a document frequency strictly higher than the given threshold.',
    )
    min_df: float | int = Field(default=2, description='Minimum document frequency.')
    sublinear_tf: bool = Field(default=True, description='Apply sublinear TF scaling.')
    norm: Literal['l1', 'l2', None] = Field(
        default='l2',
        description="Vector normalization type. 'l2' (Euclidean) is standard for cosine similarity.",
    )

    # Add the same validators you had in TfidfParams (minus dtype-related ones)
    @field_validator('ngram_range')
    @classmethod
    def validate_ngram_range(cls, v: tuple[int, int]) -> tuple[int, int]:
        low, high = v
        if not (isinstance(low, int) and isinstance(high, int) and low > 0 and high > 0):
            raise ValueError('ngram_range values must be positive integers.')
        if low > high:
            raise ValueError(
                f'In ngram_range, the lower bound ({low}) cannot be greater than the upper bound ({high}).'
            )
        return v

    @field_validator('min_df', 'max_df')
    @classmethod
    def validate_df(cls, v: float | int, field_info) -> float | int:
        name = field_info.field_name
        if isinstance(v, float):
            if not (0.0 <= v <= 1.0):
                raise ValueError(f'If `{name}` is a float, it must be in the range [0.0, 1.0].')
        elif isinstance(v, int):
            if v < 1:
                raise ValueError(f'If `{name}` is an int, it must be >= 1.')
        return v


class SimilarityNnParams(BaseModel):
    """
    Parameters for the fallback cuML NearestNeighbors model, used to find
    the most similar candidates for unassigned entities.
    """

    model_config = ConfigDict(extra='forbid')
    n_neighbors: int = Field(
        default=24, gt=0, description='Number of nearest neighbors to find for each entity.'
    )
    metric: Literal['cosine', 'euclidean', 'manhattan', 'minkowski'] = Field(
        default='cosine', description="'cosine' is ideal for comparing TF-IDF vectors."
    )


# ======================================================================================
# Main Vectorizer Configuration
# ======================================================================================


class VectorizerConfig(BaseModel):
    """
    Parameters for GPU-accelerated feature extraction and dimensionality reduction.

    The vectorizer creates multiple complementary representations (streams) of each entity:
    - TF-IDF: Captures character-level patterns and spelling variations (syntactic similarity).
    - Phonetic: Captures how names sound, helping match phonetically similar entities.
    - Semantic: Captures meaning using transformer embeddings (semantic similarity).

    These streams are combined to create a rich, multi-faceted representation that is
    then used for clustering. All operations leverage RAPIDS (cuML, cuDF, cuPy).
    """

    model_config = ConfigDict(extra='forbid')

    # --- Feature Stream Selection ---
    encoders: set[Literal['tfidf', 'phonetic', 'semantic']] = Field(
        default_factory=lambda: {'tfidf', 'phonetic', 'semantic'},
        min_length=1,
        description='Set of feature extraction methods to use. Using multiple streams '
        'provides a more robust and accurate representation of the data.',
    )

    sparse_reducers: set[Literal['svd', 'pca']] = Field(
        default_factory=lambda: {'svd'},
        min_length=1,
        description='Dimensionality reduction techniques to apply to sparse feature '
        "matrices (TF-IDF and Phonetic). 'svd' is generally faster and "
        "more memory-efficient for sparse data than 'pca'.",
    )

    use_address_in_encoding: bool = Field(
        default=True,
        description='If True, concatenates the normalized address to the entity name '
        'for TF-IDF encoding. Helps distinguish entities with similar names '
        'but different locations. Does not affect phonetic or semantic streams.',
    )

    # --- Spectral Preprocessing ---
    damping_beta: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description='Controls spectral damping (variance normalization) strength. '
        '0.0 means no damping, 1.0 means full whitening. A value '
        'between 0.3-0.5 is recommended to prevent dominant features from '
        'overshadowing others while preserving important variance.',
    )

    epsilon: float = Field(
        default=1.0e-8,
        gt=0,
        description='A small constant added to denominators during normalization to '
        'prevent division-by-zero errors, ensuring numerical stability.',
    )

    # --- Semantic Stream Configuration ---
    semantic_model: str = Field(
        default='BAAI/bge-base-en-v1.5',
        min_length=1,
        description='HuggingFace sentence-transformer model for semantic encoding. The '
        'model is downloaded on first use and loaded onto the GPU for '
        'high-throughput inference.',
    )

    semantic_batch_size: int = Field(
        default=1024,
        gt=0,
        description='Batch size for GPU semantic encoding. Adjust based on available '
        'GPU memory (VRAM). Larger is faster but uses more VRAM.',
    )

    # --- TF-IDF Stream Configuration ---
    tfidf_params: TfidfParams = Field(
        default_factory=TfidfParams, description='Base parameters for TF-IDF vectorization.'
    )
    tfidf_svd_params: TfidfSvdParams = Field(
        default_factory=TfidfSvdParams, description='SVD parameters for TF-IDF features.'
    )
    tfidf_pca_params: TfidfPcaParams = Field(
        default_factory=TfidfPcaParams, description='PCA parameters for TF-IDF features.'
    )

    # --- Phonetic Stream Configuration ---
    phonetic_max_words: int = Field(
        default=10,
        ge=1,
        le=20,
        description='Maximum number of words from the start of an entity name to use '
        'for phonetic encoding. Capping this can improve performance with long names.',
    )
    phonetic_params: PhoneticParams = Field(
        default_factory=PhoneticParams, description='Base parameters for phonetic vectorization.'
    )
    phonetic_svd_params: PhoneticSvdParams = Field(
        default_factory=PhoneticSvdParams, description='SVD parameters for phonetic features.'
    )
    phonetic_pca_params: PhoneticPcaParams = Field(
        default_factory=PhoneticPcaParams, description='PCA parameters for phonetic features.'
    )

    # --- Stream Balancing Configuration ---
    stream_proportions: dict[Literal['semantic', 'tfidf', 'phonetic'], float] = Field(
        default_factory=lambda: {'semantic': 0.45, 'tfidf': 0.45, 'phonetic': 0.10},
        description='Relative importance weights for each active feature stream. These '
        'control how much each stream contributes to the final representation. '
        'Must sum to 1.0 and only contain keys for active encoders.',
    )

    # --- String Similarity Fallback Configuration ---
    similarity_tfidf: SimilarityTfidfParams = Field(
        default_factory=SimilarityTfidfParams,
        description='TF-IDF settings for the high-precision fallback similarity matcher.',
    )
    similarity_nn: SimilarityNnParams = Field(
        default_factory=SimilarityNnParams,
        description='Nearest neighbor settings for the fallback similarity matcher.',
    )

    # --- SVD Eigsh Fallback Configuration ---
    eigsh_fallback_params: SvdEigshFallbackConfig = Field(
        default_factory=SvdEigshFallbackConfig,
        description='Parameters for the robust SVD fallback mechanism, used only if the primary solver fails.',
    )

    final_svd_components: int = Field(
        default=512, gt=128, description='Target dimensionality for canonical embedding matrices'
    )

    @model_validator(mode='after')
    def validate_stream_proportions(self) -> 'VectorizerConfig':
        """
        Ensures stream_proportions are consistent with the selected encoders and sum to 1.0.

        This critical validation step prevents logical errors where the weights for feature
        streams do not match the streams that are actually enabled, ensuring the final
        feature combination is correctly weighted.
        """
        active_encoders = self.encoders
        proportion_keys = set(self.stream_proportions.keys())

        if active_encoders != proportion_keys:
            raise ValueError(
                'The keys in `stream_proportions` must exactly match the values in `encoders`.\n'
                f'Currently enabled encoders: {sorted(list(active_encoders))}\n'
                f'Proportion keys provided: {sorted(list(proportion_keys))}'
            )

        total = sum(self.stream_proportions.values())
        if abs(total - 1.0) > 1.0e-6:
            raise ValueError(
                f'The values in `stream_proportions` must sum to 1.0, but got {total:.6f} for values: '
                f'{self.stream_proportions}'
            )

        return self

    @model_validator(mode='after')
    def validate_sparse_reducer_logic(self) -> 'VectorizerConfig':
        """
        Ensures the sparse reducer configuration is logically valid.

        Rules enforced:
        1. 'svd' must always be present, as it is the primary reducer and is
           required to densify the matrix before PCA can be run.
        2. If 'pca' is used, its `n_components` must be less than the `n_components`
           of the preceding 'svd' step, as it is a further dimensionality reduction.
        """
        # Rule 1: 'svd' is mandatory for sparse reduction.
        if 'svd' not in self.sparse_reducers:
            raise ValueError(
                "`sparse_reducers` must always include 'svd'. It is the primary reducer "
                'for sparse matrices and is required before PCA can be applied.'
            )

        # Rule 2: If using PCA, ensure its dimensionality is less than SVD's.
        if 'pca' in self.sparse_reducers:
            # --- Check TF-IDF dimensionality chain ---
            svd_dims_tfidf = self.tfidf_svd_params.n_components
            pca_dims_tfidf = self.tfidf_pca_params.n_components
            if svd_dims_tfidf <= pca_dims_tfidf:
                raise ValueError(
                    'When using PCA on TF-IDF features, `tfidf_svd_params.n_components` '
                    f'({svd_dims_tfidf}) must be strictly greater than `tfidf_pca_params.n_components` '
                    f'({pca_dims_tfidf}) because PCA reduces the output of SVD.'
                )

            # --- Check Phonetic dimensionality chain ---
            svd_dims_phonetic = self.phonetic_svd_params.n_components
            pca_dims_phonetic = self.phonetic_pca_params.n_components
            if svd_dims_phonetic <= pca_dims_phonetic:
                raise ValueError(
                    'When using PCA on phonetic features, `phonetic_svd_params.n_components` '
                    f'({svd_dims_phonetic}) must be strictly greater than `phonetic_pca_params.n_components` '
                    f'({pca_dims_phonetic}) because PCA reduces the output of SVD.'
                )

        return self


# ======================================================================================
# Typed Sub-Models for Clusterer Configuration
# ======================================================================================


class UmapParams(BaseModel):
    """
    Base parameters for the cuML UMAP algorithm, which performs non-linear manifold learning.
    """

    model_config = ConfigDict(extra='forbid')
    n_neighbors: int = Field(
        default=15,
        gt=0,
        description='Size of the local neighborhood. Larger values preserve more global structure.',
    )
    n_components: int = Field(
        default=48, gt=0, description='The target dimensionality of the embedded space.'
    )
    min_dist: float = Field(
        default=0.05,
        ge=0.0,
        lt=1.0,
        description='Minimum distance between embedded points. Lower values create tighter clusters.',
    )
    spread: float = Field(
        default=0.5,
        gt=0.0,
        description='The effective scale of the embedded points. Determines how far apart clusters are.',
    )
    metric: Literal['cosine', 'euclidean', 'manhattan', 'minkowski'] = Field(
        default='cosine', description='The distance metric to use.'
    )
    init: Literal['spectral', 'random'] = Field(
        default='spectral', description="Initialization method. 'spectral' is more deterministic."
    )
    n_epochs: int = Field(
        default=400,
        gt=0,
        description='Number of training epochs. More epochs can lead to a better embedding but take longer.',
    )
    negative_sample_rate: int = Field(
        default=7,
        ge=0,
        description='Number of negative samples per positive sample. Higher values increase optimization accuracy.',
    )
    repulsion_strength: float = Field(
        default=1.0,
        ge=0.0,
        description='Weighting of repulsive forces in the embedding. Higher values push points apart more.',
    )
    learning_rate: float = Field(
        default=0.5, gt=0.0, description='The initial learning rate for the optimization algorithm.'
    )
    random_state: int | None = Field(
        default=None,
        exclude=True,
        description='Seed for reproducibility, propagated from global config.',
    )

    @model_validator(mode='after')
    def validate_spread_vs_min_dist(self) -> 'UmapParams':
        """Ensures that `spread` is strictly greater than `min_dist`, a requirement for UMAP."""
        if self.spread <= self.min_dist:
            raise ValueError(
                f'UMAP parameter `spread` ({self.spread}) must be strictly greater than `min_dist` ({self.min_dist}).'
            )
        return self


class UmapEnsembleSamplingConfig(BaseModel):
    """
    Parameter ranges for creating a robust UMAP ensemble by training multiple models with randomized hyperparameters.
    """

    model_config = ConfigDict(extra='forbid')
    local_view_ratio: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Fraction of UMAP runs that will focus on local structure by using the 'local' n_neighbors range.",
    )
    n_neighbors_local: tuple[int, int] = Field(
        default=(10, 35),
        description='[min, max] range for `n_neighbors` for local-focused UMAP runs.',
    )
    n_neighbors_global: tuple[int, int] = Field(
        default=(40, 70),
        description='[min, max] range for `n_neighbors` for global-focused UMAP runs.',
    )
    min_dist: tuple[float, float] = Field(
        default=(0.0, 0.15), description='[min, max] range to sample `min_dist` from.'
    )
    spread: tuple[float, float] = Field(
        default=(0.5, 2.0), description='[min, max] range to sample `spread` from.'
    )
    n_epochs: tuple[int, int] = Field(
        default=(200, 500), description='[min, max] range to sample `n_epochs` from.'
    )
    learning_rate: tuple[float, float] = Field(
        default=(0.5, 1.5), description='[min, max] range to sample `learning_rate` from.'
    )
    repulsion_strength: tuple[float, float] = Field(
        default=(0.5, 1.5), description='[min, max] range to sample `repulsion_strength` from.'
    )
    negative_sample_rate: tuple[int, int] = Field(
        default=(5, 20), description='[min, max] range to sample `negative_sample_rate` from.'
    )
    init_strategies: list[Literal['spectral', 'random']] = Field(
        default_factory=lambda: ['spectral', 'random'],
        description='List of initialization strategies to sample from.',
    )

    @field_validator('*')
    @classmethod
    def validate_range_logic(cls, v: Any, field_info) -> Any:
        """Generic validator to ensure that for any tuple `(min, max)`, `min <= max`."""
        if isinstance(v, tuple) and len(v) == 2:
            low, high = v
            if low > high:
                raise ValueError(
                    f"In parameter range for '{field_info.field_name}', the lower bound ({low}) cannot be greater than the upper bound ({high})."
                )
        return v

    @model_validator(mode='after')
    def validate_neighbor_ranges(self) -> 'UmapEnsembleSamplingConfig':
        """Ensures the local and global n_neighbors ranges are distinct and do not overlap."""
        local_min, local_max = self.n_neighbors_local
        global_min, global_max = self.n_neighbors_global
        if local_max >= global_min:
            raise ValueError(
                f'The `n_neighbors_local` range [{local_min}, {local_max}] must not overlap with '
                f'the `n_neighbors_global` range [{global_min}, {global_max}]. '
                f'The max of local must be less than the min of global.'
            )
        return self


class HdbscanParams(BaseModel):
    """
    Parameters for the cuML HDBSCAN clustering algorithm, a density-based algorithm excellent for discovering clusters of varying shapes.
    """

    model_config = ConfigDict(extra='forbid')
    min_cluster_size: int = Field(
        default=2,
        ge=2,
        description='The minimum number of samples in a group for it to be considered a cluster.',
    )
    min_samples: int = Field(
        default=1,
        ge=1,
        description='Number of samples in a neighborhood for a point to be considered a core point.',
    )
    cluster_selection_epsilon: float = Field(
        default=0.0,
        ge=0.0,
        description='A distance threshold. Two clusters will be merged if they are closer than this value.',
    )
    prediction_data: bool = Field(
        default=True,
        description='Must be True to generate data needed for soft clustering probabilities and predicting new points.',
    )
    alpha: float = Field(
        default=0.9,
        gt=0.0,
        description='A parameter for scaling distance. Affects how density is measured.',
    )
    cluster_selection_method: Literal['eom', 'leaf'] = Field(
        default='leaf',
        description="'leaf' selects smaller, more granular clusters. 'eom' (Excess of Mass) selects more stable, prominent clusters.",
    )


class SnnClusteringParams(BaseModel):
    """
    Parameters for Shared Nearest Neighbor (SNN) graph clustering, performed with cuGraph. Often reveals different structural aspects than HDBSCAN.
    """

    model_config = ConfigDict(extra='forbid')
    k_neighbors: int = Field(
        default=48, gt=0, description='Number of neighbors to use when constructing the SNN graph.'
    )
    louvain_resolution: float = Field(
        default=0.60,
        gt=0.0,
        description='Resolution parameter for the Louvain community detection algorithm. Higher values lead to more, smaller communities.',
    )
    merge_name_distance_threshold: float = Field(
        default=0.02,
        gt=0.0,
        lt=1.0,
        description='Maximum k-NN distance for two cluster NAMES to be considered for merging.',
    )
    merge_address_distance_threshold: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description='Maximum k-NN distance for two cluster ADDRESSES to be considered for merging.',
    )


class NoiseAttachmentParams(BaseModel):
    """
    Parameters for the algorithm that attempts to assign noise points (outliers) from HDBSCAN to existing clusters.
    """

    model_config = ConfigDict(extra='forbid')
    k_neighbors: int = Field(
        default=15,
        gt=0,
        description='Number of neighbors to consider when checking for a potential cluster attachment.',
    )
    similarity_threshold: float = Field(
        default=0.82,
        ge=0.0,
        le=1.0,
        description='Minimum similarity score required to attach a noise point to a cluster.',
    )
    min_neighbor_matches: int = Field(
        default=2,
        ge=1,
        description='The noise point must have at least this many neighbors that belong to the same target cluster.',
    )
    ambiguity_ratio_threshold: float = Field(
        default=1.5,
        ge=1.0,
        description='A ratio to prevent ambiguous attachments. The score for the best cluster must be this much higher than the score for the second-best cluster.',
    )


class EnsembleParams(BaseModel):
    """
    Parameters for the final ensembling logic that combines the results from HDBSCAN and SNN clustering to produce a final, stable set of clusters.
    """

    model_config = ConfigDict(extra='forbid')
    purity_min: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description='Minimum purity score for a cluster to be accepted without further validation.',
    )
    min_overlap: int = Field(
        default=3,
        ge=1,
        description='Minimum number of shared entities required to consider two clusters (one from HDBSCAN, one from SNN) as representing the same group.',
    )
    allow_new_snn_clusters: bool = Field(
        default=True,
        description='If True, allows clusters found by SNN but not HDBSCAN to be included in the final result.',
    )
    min_newcluster_size: int = Field(
        default=4,
        ge=2,
        description='The minimum size for a new cluster introduced by SNN to be considered valid.',
    )
    default_rescue_conf: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="The default confidence score to assign to noise points that are 'rescued' and attached to a cluster.",
    )


# ======================================================================================
# Main Clusterer Configuration
# ======================================================================================


class ClustererConfig(BaseModel):
    """
    Configuration for GPU-accelerated manifold learning (UMAP) and clustering (HDBSCAN/SNN).

    This is the core of the entity resolution process. UMAP learns a low-dimensional
    representation that preserves entity similarity, then multiple clustering algorithms
    (HDBSCAN and SNN) group similar entities together. A final ensemble step combines
    these results for a robust final clustering. All operations use cuML and cuGraph
    for maximum GPU acceleration.
    """

    model_config = ConfigDict(extra='forbid')

    # --- UMAP Ensemble Configuration ---
    umap_n_runs: int = Field(
        default=12,
        ge=1,
        le=50,
        description='Number of cuML UMAP models to train in the ensemble. More runs improve robustness but increase runtime.',
    )
    umap_params: UmapParams = Field(
        default_factory=UmapParams, description='Base parameters for each UMAP run.'
    )
    umap_ensemble_sampling_config: UmapEnsembleSamplingConfig = Field(
        default_factory=UmapEnsembleSamplingConfig,
        description='Hyperparameter ranges for the UMAP ensemble.',
    )

    # --- Consensus Embedding Configuration ---
    cosine_consensus_n_samples: int = Field(
        default=8192,
        ge=1000,
        le=50000,
        description='Number of samples for kernel PCA consensus embedding. Higher values give better results but use more GPU memory.',
    )
    cosine_consensus_batch_size: int = Field(
        default=2048,
        ge=128,
        le=8192,
        description='Batch size for GPU consensus embedding computation. Adjust based on available VRAM.',
    )

    # --- HDBSCAN Clustering Configuration ---
    max_noise_rate_warn: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description='Threshold for warning about high noise rates in HDBSCAN. A high rate may indicate suboptimal UMAP parameters.',
    )
    hdbscan_params: HdbscanParams = Field(
        default_factory=HdbscanParams,
        description='Parameters for the primary HDBSCAN clustering algorithm.',
    )

    # --- SNN (Shared Nearest Neighbor) Configuration ---
    snn_clustering_params: SnnClusteringParams = Field(
        default_factory=SnnClusteringParams,
        description='Parameters for the secondary SNN graph clustering algorithm.',
    )
    noise_attachment_params: NoiseAttachmentParams = Field(
        default_factory=NoiseAttachmentParams,
        description='Parameters for re-attaching HDBSCAN noise points to clusters.',
    )

    # --- Cluster Merging Configuration ---
    merge_median_threshold: float = Field(
        default=0.84,
        ge=0.0,
        le=1.0,
        description='Minimum median similarity between two clusters to consider merging them.',
    )
    merge_max_threshold: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description='Minimum maximum similarity (i.e., the single best pair) between two clusters to consider merging.',
    )
    merge_sample_size: int = Field(
        default=32,
        ge=5,
        le=2048,
        description='Number of points to sample from large clusters for merge similarity checks to avoid O(n^2) complexity.',
    )
    centroid_similarity_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description='Pre-filter threshold for cluster centroid similarity. Only pairs above this threshold are considered for full merge checks.',
    )
    merge_batch_size: int = Field(
        default=2048, ge=128, description='Batch size for vectorized GPU merge computations.'
    )
    centroid_sample_size: int = Field(
        default=2048,
        ge=128,
        description='Sample size for computing cluster centroids on GPU for large clusters.',
    )

    # --- Ensemble Configuration ---
    ensemble_params: EnsembleParams = Field(
        default_factory=EnsembleParams,
        description='Parameters for combining HDBSCAN and SNN clustering results.',
    )

    @model_validator(mode='after')
    def validate_merge_thresholds(self) -> 'ClustererConfig':
        """Ensures the merge thresholds are logically consistent."""
        if self.merge_max_threshold < self.merge_median_threshold:
            raise ValueError(
                f'`merge_max_threshold` ({self.merge_max_threshold}) cannot be less than '
                f'`merge_median_threshold` ({self.merge_median_threshold}). A high maximum '
                f'similarity should be a stricter requirement than a high median.'
            )
        return self


# ======================================================================================
# Typed Sub-Models for Validation Configuration
# ======================================================================================


class ReassignmentWeights(BaseModel):
    """
    Defines the weights for scoring potential cluster reassignments for entities that
    were flagged during validation. The scores determine the best new cluster for an entity.
    """

    model_config = ConfigDict(extra='forbid')

    name_similarity: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Weight for the fuzzy string similarity score of the entity's name against the target cluster's canonical name.",
    )
    address_similarity: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Weight for the fuzzy string similarity of the entity's address against the target cluster's canonical address.",
    )
    cluster_size: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description='Weight for the size of the target cluster. This acts as a tie-breaker, preferring larger, more established clusters.',
    )
    cluster_probability: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Weight for the entity's soft clustering probability (from HDBSCAN) for the target cluster.",
    )

    @model_validator(mode='after')
    def validate_weights_sum_to_one(self) -> 'ReassignmentWeights':
        """Ensures the weights sum to 1.0, maintaining a normalized scoring system."""
        total = sum(self.dict().values())
        if abs(total - 1.0) > 1.0e-6:
            raise ValueError(
                f'Reassignment weights must sum to 1.0, but got {total:.6f} for values: {self.dict()}'
            )
        return self


# ======================================================================================
# Main Validation Configuration
# ======================================================================================


class ValidationConfig(BaseModel):
    """
    Configuration for validating and refining cluster assignments.

    After initial clustering, this stage checks that entities within the same cluster
    are truly the same real-world entity by applying a series of checks on names,
    addresses, and other business rules. All string comparisons are GPU-accelerated.
    """

    model_config = ConfigDict(extra='forbid')

    street_number_threshold: int = Field(
        default=30,
        ge=0,
        le=1000,
        description='Maximum allowed absolute difference between street numbers for two '
        'entities to be in the same cluster. Set to 0 to require exact matches.',
    )

    address_fuzz_ratio: int = Field(
        default=89,
        ge=0,
        le=100,
        description='Minimum fuzzy string match score (0-100) for addresses in the same '
        'cluster, based on Levenshtein distance. Higher is stricter.',
    )

    name_fuzz_ratio: int = Field(
        default=89,
        ge=0,
        le=100,
        description='Minimum fuzzy string match score (0-100) for entity names in the same '
        'cluster. Can be different from the address threshold if names are '
        'more or less reliable in the source data.',
    )

    enforce_state_boundaries: bool = Field(
        default=True,
        description='If True, entities with addresses in different states cannot be in '
        'the same cluster, unless explicitly allowed by `allow_neighboring_states`.',
    )

    allow_neighboring_states: list[list[str]] = Field(
        default_factory=list,
        description='A list of state pairs that are allowed to be matched across borders, '
        "e.g., [['NY', 'NJ']] for the NYC metro area. Only has an effect "
        'if `enforce_state_boundaries` is True.',
    )

    profile_comparison_max_pairs_per_chunk: int = Field(
        default=1_000_000,
        ge=10000,
        le=10_000_000,
        description='Maximum number of entity pairs to compare in a single batch during '
        'validation. A crucial parameter to prevent GPU out-of-memory errors.',
    )

    reassignment_scoring_weights: ReassignmentWeights = Field(
        default_factory=ReassignmentWeights,
        description='The weighted components used to score and choose the best new cluster for a reassigned entity.',
    )

    validate_cluster_batch_size: int = Field(
        default=1024,
        ge=32,
        le=10000,
        description='The number of clusters to process in a single batch during validation operations on the GPU.',
    )

    @field_validator('allow_neighboring_states')
    @classmethod
    def validate_state_pairs(cls, v: list[list[str]]) -> list[list[str]]:
        """Validates that state pairs are properly formatted as pairs of 2-letter uppercase strings."""
        validated_pairs = []
        for i, pair in enumerate(v):
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(
                    f'allow_neighboring_states[{i}] must be a list of exactly 2 state codes, got: {pair}'
                )

            validated_pair = []
            for state in pair:
                if not isinstance(state, str):
                    raise ValueError(
                        f'State codes must be strings, but got {type(state)} in pair {i}.'
                    )

                state_upper = state.strip().upper()
                if len(state_upper) != 2:
                    raise ValueError(
                        f"State codes must be 2 letters long, but got '{state}' in pair {i}."
                    )
                validated_pair.append(state_upper)

            # Ensure the pair is sorted to make lookups deterministic, e.g., ['NY', 'NJ'] is the same as ['NJ', 'NY']
            validated_pairs.append(sorted(validated_pair))

        return validated_pairs


# ======================================================================================
# Typed Sub-Models for Confidence Scoring Configuration
# ======================================================================================


class ConfidenceWeights(BaseModel):
    """
    Defines the weighted components used to calculate the final confidence score for
    each entity's cluster assignment. A higher score indicates a more reliable match.
    """

    model_config = ConfigDict(extra='forbid')

    cluster_probability: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for the soft clustering probability from HDBSCAN. Represents the model's fundamental belief in the assignment.",
    )
    name_similarity: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description='Weight for the average fuzzy name similarity of an entity against all other members of its cluster.',
    )
    address_confidence: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description='Weight for a composite score based on address similarity and geographic consistency within the cluster.',
    )
    cohesion_score: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description='Weight for the geometric cohesion of the cluster in the UMAP embedding space. Tighter clusters are more confident.',
    )
    cluster_size_factor: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description='Weight for a factor that rewards larger clusters, as they represent stronger evidence of a real-world entity.',
    )

    @model_validator(mode='after')
    def validate_weights_sum_to_one(self) -> 'ConfidenceWeights':
        """Ensures the weights sum to 1.0, maintaining a normalized scoring system."""
        total = sum(self.dict().values())
        if abs(total - 1.0) > 1.0e-6:
            raise ValueError(
                f'Confidence weights must sum to 1.0, but got {total:.6f} for values: {self.dict()}'
            )
        return self


# ======================================================================================
# Main Confidence Scoring Configuration
# ======================================================================================


class ConfidenceScoringConfig(BaseModel):
    """
    Configuration for calculating final confidence scores for each entity match.

    Confidence scores help identify matches that may need manual review and
    provide transparency about match quality. Calculations are GPU-accelerated.
    """

    model_config = ConfigDict(extra='forbid')

    weights: ConfidenceWeights = Field(
        default_factory=ConfidenceWeights,
        description='The weighted components that are combined to produce the '
        'final confidence score for each resolved entity.',
    )


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

    # --- Sub-configurations ---
    columns: ColumnConfig = Field(
        default_factory=ColumnConfig, description='Configuration for input cuDF DataFrame columns.'
    )

    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description='Configuration for output formatting, logging, and review thresholds.',
    )

    normalization: NormalizationConfig = Field(
        default_factory=NormalizationConfig,
        description='Configuration for text normalization and cleaning rules.',
    )

    vectorizer: VectorizerConfig = Field(
        default_factory=VectorizerConfig,
        description='Configuration for GPU-accelerated feature extraction and encoding.',
    )

    clusterer: ClustererConfig = Field(
        default_factory=ClustererConfig,
        description='Configuration for GPU-accelerated clustering and manifold learning.',
    )

    validation: ValidationConfig = Field(
        default_factory=ValidationConfig,
        description='Configuration for cluster validation and refinement rules.',
    )

    scoring: ConfidenceScoringConfig = Field(
        default_factory=ConfidenceScoringConfig,
        description='Configuration for final confidence scoring weights.',
    )

    # --- Global Configuration ---
    random_state: int | None = Field(
        default=42,
        ge=0,
        le=2**32 - 1,
        description='Global random seed for reproducibility. Propagated to '
        'all stochastic components (e.g., PCA, UMAP) to ensure '
        'consistent results. Set to None for non-deterministic behavior.',
    )

    @model_validator(mode='after')
    def propagate_random_state(self) -> 'ResolverConfig':
        """
        Propagates the global `random_state` to all sub-components that require it.

        This crucial step ensures that the entire pipeline is reproducible by seeding
        all stochastic algorithms (like PCA and UMAP) with the same value.
        """
        if self.random_state is not None:
            # Propagate to cuML PCA components in the vectorizer
            if self.vectorizer and self.vectorizer.tfidf_pca_params:
                self.vectorizer.tfidf_pca_params.random_state = self.random_state
            if self.vectorizer and self.vectorizer.phonetic_pca_params:
                self.vectorizer.phonetic_pca_params.random_state = self.random_state

            # Propagate to cuML UMAP in the clusterer
            if self.clusterer and self.clusterer.umap_params:
                self.clusterer.umap_params.random_state = self.random_state

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
    'ConfidenceScoringConfig',
]
