# entity_resolver/context_vectorizer.py
"""
A low-level, reusable vectorizer that processes a single set of text streams
(TF-IDF, semantic, phonetic) into one canonical embedding matrix. It follows a
standard Scikit-learn API (`fit_transform`, `transform`) and manages its own
set of trained models.
"""

import logging
from typing import Any

import cudf
import cupy
import phonetics
from cuml.decomposition import PCA, TruncatedSVD
from cuml.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer

from .components import GPUTruncatedSVD

# Local package imports
from .config import VectorizerConfig
from .utils import (
    TextStreamSet,
    balance_feature_streams,
    normalize_rows,
)

# Set up module-level logger
logger = logging.getLogger(__name__)


class SingleContextVectorizer:
    """
    Processes a single set of text streams into a canonical embedding matrix.

    This "worker" class is a self-contained vectorization engine. It takes a
    `TextStreamSet` (containing tfidf, semantic, and phonetic text series),
    applies the full multi-stream vectorization pipeline, and returns a single
    `cupy.ndarray` embedding matrix. It maintains its own state of trained
    encoders and reduction models.
    """

    def __init__(self, config: VectorizerConfig, context_name: str):
        """
        Initialize the vectorizer with configuration settings.

        Args:
            config: VectorizerConfig object containing all vectorization parameters
                   including encoder settings, reduction parameters, and stream weights
        """
        self.config = config
        self.context_name = context_name
        self.trained_encoders: dict[str, Any] = {}
        self.reduction_models: dict[str, Any] = {}
        self._is_fitted: bool = False

        logger.info(f'Initialized MultiStreamVectorizer with encoders: {config.encoders}')
        logger.debug(f'Stream proportions: {config.stream_proportions}')
        logger.debug(f'Semantic model: {config.semantic_model}')

    def fit(self, streams: TextStreamSet) -> 'SingleContextVectorizer':
        """
        Fits all vectorization and reduction models on the provided text streams.

        This method follows the Scikit-learn API convention. It trains the internal
        models but does not return the transformed data. For efficiency, it calls
        `fit_transform` internally and discards the result.

        Args:
            streams: The `TextStreamSet` containing the data to train on.

        Returns:
            The fitted vectorizer instance (`self`).
        """
        self.fit_transform(streams)
        return self

    def fit_transform(self, streams: TextStreamSet) -> cupy.ndarray:
        """
        Fit all vectorization models and transform data in one step.

        This method trains all specified encoders and reduction models on the
        input data, then returns the vectorized representation.

        Args:
            gdf: Pre-processed cuDF DataFrame with 'normalized_text' column

        Returns:
            Combined vector matrix as cupy.ndarray

        Raises:
            ValueError: If no valid encoders are specified
            RuntimeError: If vectorization fails
        """
        logger.info('Starting fit_transform for vectorization')
        return self._process_encoding(streams, is_training=True)

    def transform(self, streams: TextStreamSet) -> cupy.ndarray:
        """
        Transform data using pre-fitted models.

        This method applies already-trained encoders and reduction models to
        new data, maintaining consistency with the training phase.

        Args:
            gdf: Pre-processed cuDF DataFrame with 'normalized_text' column

        Returns:
            Combined vector matrix as cupy.ndarray

        Raises:
            RuntimeError: If models haven't been fitted yet
            ValueError: If input data format is incompatible
        """
        logger.info('Starting transform for vectorization')
        return self._process_encoding(streams, is_training=False)

    def _process_encoding(self, streams: TextStreamSet, is_training: bool) -> cupy.ndarray:
        """
        Core logic for encoding entity data into numerical vectors.

        This method coordinates the entire vectorization pipeline, processing
        each stream independently then combining them into a unified representation.

        Args:
            gdf: Input cuDF DataFrame with entity data
            is_training: Whether to fit new models (True) or use existing ones (False)

        Returns:
            Tuple of (input DataFrame, combined vector matrix)

        Raises:
            ValueError: If no valid encoders are configured or processed
        """
        operation_mode = 'Training' if is_training else 'Transforming'
        logger.info(
            f'{operation_mode} vectorization with {len(streams.tfidf):,} records '
            f'using streams: {self.config.encoders}'
        )

        # Step 2: Process each encoder stream independently
        vector_streams = {}

        # TF-IDF stream for syntactic similarity
        if 'tfidf' in self.config.encoders:
            logger.debug('Processing TF-IDF stream...')
            vector_streams['tfidf'] = self._encode_tfidf_stream(streams.tfidf, is_training)
            logger.info(f'TF-IDF stream shape: {vector_streams["tfidf"].shape}')

        # Phonetic stream for sound-based similarity
        if 'phonetic' in self.config.encoders and streams.phonetic is not None:
            logger.debug('Processing phonetic stream...')
            # Phonetic encoding works best on entity names only
            vector_streams['phonetic'] = self._encode_phonetic_stream(streams.phonetic, is_training)
            logger.info(f'Phonetic stream shape: {vector_streams["phonetic"].shape}')
        else:
            if 'phonetic' in self.config.encoders:
                logger.info(
                    f"Phonetic stream requested but not provided for '{self.context_name}' "
                    'context (expected for address-only). Skipping phonetic encoding.'
                )

        # Semantic stream for meaning-based similarity
        if 'semantic' in self.config.encoders:
            logger.debug('Processing semantic stream...')
            # Semantic encoding also works best on entity names only
            vector_streams['semantic'] = self._encode_semantic_stream(streams.semantic, is_training)
            logger.info(f'Semantic stream shape: {vector_streams["semantic"].shape}')

        # Validate that at least one stream was processed
        if not vector_streams:
            raise ValueError(
                'No valid encoders were specified or processed. '
                f'Check configuration: {self.config.encoders}'
            )

        # Step 3: Balance and combine all streams into final vectors
        combined_vectors = self._combine_streams(vector_streams, is_training)

        logger.info(
            f'Vectorization complete. Final shape: {combined_vectors.shape} '
            f'({combined_vectors.nbytes / 1e6:.2f} MB)'
        )

        self._is_fitted = True

        return combined_vectors

    def _encode_tfidf_stream(self, text_series: cudf.Series, is_training: bool) -> cupy.ndarray:
        """
        Encode text using TF-IDF vectorization with dimensionality reduction.

        This stream captures syntactic similarity through character n-grams,
        making it robust to minor spelling variations and typos.

        Args:
            text_series: Text data to encode
            is_training: Whether to fit new models

        Returns:
            Dense array of TF-IDF vectors after reduction
        """
        logger.info('Processing TF-IDF (syntactic) stream...')

        # Create high-dimensional sparse vectors from character n-grams
        sparse_vectors = self._apply_tfidf_vectorizer(text_series, is_training)
        logger.debug(
            f'TF-IDF sparse vectors: shape={sparse_vectors.shape}, '
            f'density={sparse_vectors.nnz / (sparse_vectors.shape[0] * sparse_vectors.shape[1]):.4f}'
        )

        # Reduce to dense, lower-dimensional representation
        dense_vectors = self._reduce_feature_stream(
            vectors=sparse_vectors, stream_name='tfidf', is_training=is_training
        )

        return dense_vectors

    def _encode_phonetic_stream(self, text_series: cudf.Series, is_training: bool) -> cupy.ndarray:
        """
        Encode text using phonetic algorithms with dimensionality reduction.

        This stream captures sound-based similarity using the Metaphone algorithm,
        helpful for matching entities with similar pronunciations but different
        spellings (e.g., "Smith" vs "Smythe").

        Args:
            text_series: Text data to encode
            is_training: Whether to fit new models

        Returns:
            Dense array of phonetic vectors after reduction
        """
        logger.info('Processing Phonetic (sound-based) stream...')

        # Convert text to phonetic representation
        phonetic_text_series = self._convert_to_phonetic(text_series)
        logger.debug('Phonetic conversion complete')

        # Create sparse vectors from phonetic words
        sparse_vectors = self._apply_phonetic_vectorizer(phonetic_text_series, is_training)
        logger.debug(
            f'Phonetic sparse vectors: shape={sparse_vectors.shape}, '
            f'density={sparse_vectors.nnz / (sparse_vectors.shape[0] * sparse_vectors.shape[1]):.4f}'
        )

        # Reduce to dense representation
        dense_vectors = self._reduce_feature_stream(
            vectors=sparse_vectors, stream_name='phonetic', is_training=is_training
        )

        return dense_vectors

    def _encode_semantic_stream(self, text_series: cudf.Series, is_training: bool) -> cupy.ndarray:
        """
        Encode text using semantic embeddings from transformer models.

        This stream captures meaning-based similarity using pre-trained language
        models, helpful for matching entities with similar meanings but different
        surface forms (e.g., "IBM" vs "International Business Machines").

        Args:
            text_series: Text data to encode
            is_training: Whether to load new model

        Returns:
            Dense array of semantic embeddings

        Raises:
            RuntimeError: If model not found in transform mode
        """
        logger.info('Processing Semantic (meaning-based) stream...')

        encoder_key = 'semantic_model'

        if is_training:
            # Load the specified model from HuggingFace on first use
            logger.info(f'Loading semantic model: {self.config.semantic_model}')
            model = SentenceTransformer(self.config.semantic_model)
            self.trained_encoders[encoder_key] = model
            logger.debug(f'Model embedding dimension: {model.get_sentence_embedding_dimension()}')
        else:
            # Use already loaded model
            model = self.trained_encoders.get(encoder_key)
            if not model:
                raise RuntimeError(
                    'Semantic model not found. Call fit_transform first to load the model.'
                )

        # Convert to list for sentence transformer (runs on CPU)
        text_list = text_series.to_pandas().tolist()
        logger.debug(
            f'Encoding {len(text_list):,} texts with batch size {self.config.semantic_batch_size}'
        )

        # Generate embeddings (CPU operation)
        vectors_cpu = model.encode(
            text_list,
            batch_size=self.config.semantic_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Transfer to GPU
        dense_vectors = cupy.asarray(vectors_cpu)
        logger.info(f'Semantic stream complete. Shape: {dense_vectors.shape}')

        return dense_vectors

    def _apply_tfidf_vectorizer(
        self, text_series: cudf.Series, is_training: bool
    ) -> cupy.sparse.csr_matrix:
        """
        Fit or transform data using TF-IDF vectorizer.

        Args:
            text_series: Text data to vectorize
            is_training: Whether to fit new vectorizer

        Returns:
            Sparse CSR matrix of TF-IDF features

        Raises:
            RuntimeError: If vectorizer not found in transform mode
        """
        encoder_key = 'tfidf_vectorizer'

        if is_training:
            # Configure and fit new TF-IDF vectorizer
            tfidf_params = self.config.tfidf_params.to_kwargs()

            logger.debug(f'Fitting TF-IDF with params: {tfidf_params}')
            model = TfidfVectorizer(**tfidf_params)
            sparse_vectors = model.fit_transform(text_series)

            self.trained_encoders[encoder_key] = model
            logger.debug(f'TF-IDF vocabulary size: {len(model.vocabulary_)}')
        else:
            # Use existing vectorizer
            model = self.trained_encoders.get(encoder_key)
            if not model:
                raise RuntimeError(
                    'TF-IDF vectorizer not found. Call fit_transform first to train the model.'
                )
            sparse_vectors = model.transform(text_series)

        return sparse_vectors

    def _apply_phonetic_vectorizer(
        self, text_series: cudf.Series, is_training: bool
    ) -> cupy.sparse.csr_matrix:
        """
        Fit or transform data using phonetic CountVectorizer.

        Args:
            text_series: Phonetic text data to vectorize
            is_training: Whether to fit new vectorizer

        Returns:
            Sparse CSR matrix of phonetic features

        Raises:
            RuntimeError: If vectorizer not found in transform mode
        """
        encoder_key = 'phonetic_vectorizer'

        if is_training:
            # Configure and fit new phonetic vectorizer
            logger.debug(f'Fitting phonetic vectorizer with params: {self.config.phonetic_params}')
            model = CountVectorizer(**self.config.phonetic_params.model_dump())
            sparse_vectors = model.fit_transform(text_series)

            self.trained_encoders[encoder_key] = model
            logger.debug(f'Phonetic vocabulary size: {len(model.vocabulary_)}')
        else:
            # Use existing vectorizer
            model = self.trained_encoders.get(encoder_key)
            if not model:
                raise RuntimeError(
                    'Phonetic vectorizer not found. Call fit_transform first to train the model.'
                )
            sparse_vectors = model.transform(text_series)

        return sparse_vectors

    def _convert_to_phonetic(self, text_series: cudf.Series) -> cudf.Series:
        """
        Convert text to phonetic representation using Metaphone algorithm.

        This method processes each word individually and combines their phonetic
        representations, limiting to the first few words to focus on core entity names.

        Args:
            text_series: Text data to convert

        Returns:
            cudf.Series containing phonetic representations
        """

        def multi_word_metaphone(text: str) -> str:
            """
            Convert multi-word string to phonetic form.

            Args:
                text: Input text string

            Returns:
                Space-separated phonetic representation
            """
            if not isinstance(text, str) or not text:
                return 'EMPTY'

            # Process only first few words to capture core entity name
            words = text.split()[: self.config.phonetic_max_words]
            phonetic_words = []

            for word in words:
                if word:  # Skip empty strings
                    try:
                        # dmetaphone returns a tuple (primary, alternate)
                        primary, alternate = phonetics.dmetaphone(word)

                        # Add the primary code if it exists
                        if primary:
                            phonetic_words.append(primary)

                        # Add the alternate code if it exists and is different
                        if alternate and alternate != primary:
                            phonetic_words.append(alternate)
                    except Exception as e:
                        logger.debug(f"Phonetic conversion failed for '{word}': {e}")
                        # Use original word if phonetic conversion fails
                        phonetic_words.append(word)

            return ' '.join(phonetic_words) if phonetic_words else 'EMPTY'

        # Must run on CPU due to Python library dependency
        logger.debug(f'Converting {len(text_series):,} texts to phonetic representation')
        phonetic_text_pandas = text_series.to_pandas().apply(multi_word_metaphone)

        # Transfer back to GPU
        return cudf.Series(phonetic_text_pandas)

    def _combine_streams(
        self, vector_streams: dict[str, cupy.ndarray], is_training: bool = True
    ) -> cupy.ndarray | tuple[cupy.ndarray, TruncatedSVD]:
        """
        Normalize, balance, and fuse multiple feature streams; then reduce with cuML
        TruncatedSVD and row-normalize the projection for cosine-metric downstream
        steps (e.g., kNN/UMAP/HDBSCAN).

        Processing stages:
        1) Per-stream row-wise L2 normalization (via `normalize_rows`) to put each
            stream into a cosine-stable geometry.
        2) Stream balancing per `self.config.stream_proportions` to prevent any
            single stream from dominating by scale.
        3) Dense float32 concatenation into a single fused matrix of shape (N, D_total).
        4) Row-wise L2 normalization of the fused matrix (again with `normalize_rows`)
            so mixed streams share a common cosine space prior to SVD.
        5) TruncatedSVD (GPU, no mean-centering) to `svd_n_components` dimensions,
            producing a dense (N, K) projection.
        6) Post-SVD row-wise L2 normalization to ensure cosine distances/similarities
            are directly usable in the reduced space.

        Args:
            vector_streams:
                Mapping {stream_name -> CuPy dense array} where each array has shape
                (N, d_i). Examples: {"tfidf": ..., "phonetic": ..., "semantic": ...}.
            svd_n_components:
                Target dimensionality K for TruncatedSVD (typ. 384–512 for text-heavy data).
            return_svd_model:
                If True, also return the fitted TruncatedSVD model for later transform-only use.

        Returns:
            If `return_svd_model` is False:
                projected_normalized_vectors: cupy.ndarray of shape (N, svd_n_components),
                row-L2 normalized (unit norm per row).
            If `return_svd_model` is True:
                (projected_normalized_vectors, svd_model)

        Notes:
            • TruncatedSVD does NOT mean-center, which preserves cosine geometry.
            • Keep float32 unless you encounter numerical issues that justify float64.
            • For reproducibility and tuning, you can add random_state / n_iter later.
        """
        logger.info(
            f'Normalizing, balancing, and combining {len(vector_streams)} feature streams...'
        )

        # ---- 1) Per-stream normalization -------------------------------------------------
        # Each stream is row-L2 normalized using the project's robust `normalize_rows`.
        per_stream_normalized: dict[str, cupy.ndarray] = {
            stream_name: normalize_rows(stream_vectors)
            for stream_name, stream_vectors in vector_streams.items()
        }

        # ---- 2) Balance streams by configured proportions --------------------------------
        # Scale each normalized stream by its configured weight before concatenation.
        balanced_stream_arrays: list[cupy.ndarray] = balance_feature_streams(
            vector_streams=per_stream_normalized,
            proportions=self.config.stream_proportions,
        )

        # ---- 3) Dense float32 concatenation ----------------------------------------------
        # Standardize dtype for memory/speed and concatenate along feature axis.
        balanced_stream_arrays = [
            arr.astype(cupy.float32, copy=False) for arr in balanced_stream_arrays
        ]
        fused_dense_vectors: cupy.ndarray = cupy.concatenate(balanced_stream_arrays, axis=1)

        # ---- 4) Pre-SVD row-wise normalization ------------------------------------------
        # Normalize rows of the fused matrix (again via `normalize_rows`) so the fused
        # representation sits in a clean cosine space prior to SVD.
        fused_dense_vectors = normalize_rows(fused_dense_vectors)
        logger.debug('Fused matrix ready for SVD: shape=%s', fused_dense_vectors.shape)

        # ---- 5) TruncatedSVD (cuML, GPU) -------------------------------------------------
        # Reduce dimensionality without mean-centering (key for cosine geometry).
        svd_key = 'final_svd_reducer'

        if is_training:
            svd_model = TruncatedSVD(n_components=self.config.final_svd_components)
            projected_vectors = svd_model.fit_transform(fused_dense_vectors)
            self.reduction_models[svd_key] = svd_model
        else:
            svd_model = self.reduction_models.get(svd_key)
            if not svd_model:
                raise RuntimeError('Final SVD model not found. Call fit_transform first.')
            projected_vectors = svd_model.transform(fused_dense_vectors)

        # Optional: log cumulative explained variance ratio if available (attribute varies by cuML version).
        evr = getattr(svd_model, 'explained_variance_ratio_', None)
        if evr is not None:
            try:
                cum_evr = float(cupy.asarray(evr).sum())
                logger.info(
                    'TruncatedSVD: n_components=%d, cumulative explained variance ratio=%.4f',
                    self.config.final_svd_components,
                    cum_evr,
                )
            except Exception:
                # Non-fatal; attribute shape/type varies across versions.
                pass

        # ---- 6) Post-SVD row-wise normalization -----------------------------------------
        # Normalize rows of the SVD projection so cosine similarities/distances in the
        # reduced space are ready for kNN/UMAP/HDBSCAN without further scaling.
        projected_normalized_vectors = normalize_rows(projected_vectors)
        logger.debug(
            'Projected matrix for downstream cosine metrics: shape=%s',
            projected_normalized_vectors.shape,
        )

        return projected_normalized_vectors

    def _reduce_feature_stream(
        self, vectors: cupy.sparse.csr_matrix | cupy.ndarray, stream_name: str, is_training: bool
    ) -> cupy.ndarray:
        """
        Reduce dimensionality of feature stream using SVD and/or PCA.

        This method applies a two-stage reduction pipeline:
        1. SVD for initial high-ratio reduction from sparse to dense
        2. Optional PCA for further denoising and dimension reduction

        Args:
            vectors: Input vectors (sparse or dense)
            stream_name: Name of the stream for configuration lookup
            is_training: Whether to fit new reduction models

        Returns:
            Dense array of reduced vectors
        """
        logger.info(
            f"Reducing '{stream_name}' stream from shape {vectors.shape} "
            f'using reducers: {self.config.sparse_reducers}'
        )

        current_vectors = vectors

        # Get stream-specific reduction parameters
        if stream_name == 'tfidf':
            svd_params = self.config.tfidf_svd_params
            pca_params = self.config.tfidf_pca_params
        elif stream_name == 'phonetic':
            svd_params = self.config.phonetic_svd_params
            pca_params = self.config.phonetic_pca_params
        else:
            # It's good practice to handle unexpected cases.
            raise ValueError(f"Invalid stream name provided: '{stream_name}'")

        # Stage 1: SVD for initial reduction from sparse matrix
        if 'svd' in self.config.sparse_reducers:
            current_vectors = self._apply_svd_reduction(
                current_vectors, stream_name, svd_params, is_training
            )
            logger.debug(f'After SVD: shape={current_vectors.shape}')

        # Stage 2: Optional PCA for further reduction and denoising
        if 'pca' in self.config.sparse_reducers and pca_params is not None:
            current_vectors = self._apply_pca_reduction(
                current_vectors, stream_name, pca_params, is_training
            )
            logger.debug(f'After PCA: shape={current_vectors.shape}')

        logger.info(f"Reduction for '{stream_name}' complete. Final shape: {current_vectors.shape}")

        # Ensure output is dense array
        if hasattr(current_vectors, 'toarray'):
            return current_vectors.toarray()
        return current_vectors

    def _apply_svd_reduction(
        self,
        sparse_vectors: cupy.sparse.csr_matrix | cupy.ndarray,
        stream_name: str,
        svd_params: dict[str, Any],
        is_training: bool,
    ) -> cupy.ndarray:
        """
        Apply Truncated SVD reduction with spectral damping.

        This method uses SVD for dimensionality reduction and applies spectral
        damping to balance component contributions, avoiding over-emphasis on
        high-variance components while not excessively amplifying noise.

        Spectral Damping:
        - beta=1.0: Full whitening (equalizes all components, high noise risk)
        - beta=0.0: No whitening (preserves original variance distribution)
        - beta=0.4: Balanced approach (reduces dominance without noise amplification)

        Args:
            sparse_vectors: Input vectors (sparse or dense)
            stream_name: Name of stream for model storage
            svd_params: SVD configuration parameters
            is_training: Whether to fit new SVD model

        Returns:
            Dense array of reduced and damped vectors

        Raises:
            RuntimeError: If SVD model not found in transform mode
        """
        svd_key = f'{stream_name}_svd_reducer'

        if is_training:
            logger.debug(f"Fitting TruncatedSVD for '{stream_name}' with params: {svd_params}")
            svd_model = GPUTruncatedSVD(self.config.eigsh_fallback_params, svd_params)
            dense_vectors = svd_model.fit_transform(sparse_vectors)
            self.reduction_models[svd_key] = svd_model

            # Log explained variance information
            if hasattr(svd_model, 'explained_variance_ratio_'):
                cumsum_variance = cupy.cumsum(svd_model.explained_variance_ratio_)
                variance_90 = int(cupy.searchsorted(cumsum_variance, cupy.array([0.9]))) + 1
                logger.debug(
                    f'SVD explained variance: '
                    f'{float(cumsum_variance[-1]):.4f} total, '
                    f'{variance_90} components for 90% variance'
                )
        else:
            svd_model = self.reduction_models.get(svd_key)
            if not svd_model:
                raise RuntimeError(
                    f'SVD reducer for {stream_name} not found. '
                    f'Call fit_transform first to train the model.'
                )
            dense_vectors = svd_model.transform(sparse_vectors)

        # Apply spectral damping to balance component contributions
        beta = self.config.damping_beta
        logger.debug(f'Applying spectral damping with beta={beta} to SVD output')

        # Calculate scaling factors based on singular values
        # Clamp to epsilon for numerical stability
        safe_singular_values = cupy.maximum(svd_model.singular_values_, self.config.epsilon)
        scaling_factors = safe_singular_values ** (-beta)

        # Apply damping to vectors
        damped_vectors = dense_vectors * scaling_factors

        logger.debug(
            f"SVD reduction for '{stream_name}' complete. "
            f'Shape: {damped_vectors.shape}, '
            f'Mean norm after damping: {float(cupy.linalg.norm(damped_vectors, axis=1).mean()):.4f}'
        )

        return damped_vectors

    def _apply_pca_reduction(
        self,
        dense_vectors: cupy.ndarray,
        stream_name: str,
        pca_params: dict[str, Any],
        is_training: bool,
    ) -> cupy.ndarray:
        """
        Apply PCA reduction to dense vectors for further denoising.

        PCA is applied as a second stage after SVD to further reduce dimensions
        and remove noise while preserving the most important variance directions.

        Args:
            dense_vectors: Dense input vectors
            stream_name: Name of stream for model storage
            pca_params: PCA configuration parameters
            is_training: Whether to fit new PCA model

        Returns:
            Further reduced dense vectors

        Raises:
            RuntimeError: If PCA model not found in transform mode
        """
        pca_key = f'{stream_name}_pca_reducer'

        if is_training:
            logger.debug(f"Fitting PCA for '{stream_name}' with params: {pca_params}")
            pca_model = PCA(**pca_params)
            reduced_vectors = pca_model.fit_transform(dense_vectors)
            self.reduction_models[pca_key] = pca_model

            # Log explained variance information
            if hasattr(pca_model, 'explained_variance_ratio_'):
                total_variance = float(cupy.sum(pca_model.explained_variance_ratio_))
                logger.debug(f"PCA explained variance for '{stream_name}': {total_variance:.4f}")
        else:
            pca_model = self.reduction_models.get(pca_key)
            if not pca_model:
                raise RuntimeError(
                    f'PCA reducer for {stream_name} not found. '
                    f'Call fit_transform first to train the model.'
                )
            reduced_vectors = pca_model.transform(dense_vectors)

        logger.debug(
            f"Successfully applied PCA to '{stream_name}' stream with output dtype: {reduced_vectors.dtype}"
        )

        return reduced_vectors

    def get_models(self) -> dict[str, Any]:
        """
        Get trained encoders and reduction models for persistence.

        Returns:
            Dictionary containing all trained models for saving
        """
        return {'encoders': self.trained_encoders, 'reduction_models': self.reduction_models}

    def set_models(self, models: dict[str, Any]) -> None:
        """
        Load pre-trained encoders and reduction models.

        This method is used when loading a saved model to restore the trained
        state of all vectorization components.

        Args:
            trained_encoders: Dictionary of trained encoder models
            reduction_models: Dictionary of trained reduction models
        """
        self.trained_encoders = models.get('encoders', {})
        self.reduction_models = models.get('reduction_models', {})
        self._is_fitted = True  # Mark as fitted when loading models
        logger.info(
            f'Loaded {len(self.trained_encoders)} encoder models and '
            f'{len(self.reduction_models)} reduction models'
        )
        logger.debug(f'Encoder models: {list(self.trained_encoders.keys())}')
        logger.debug(f'Reduction models: {list(self.reduction_models.keys())}')
