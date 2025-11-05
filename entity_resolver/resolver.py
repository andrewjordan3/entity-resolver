# entity_resolver/resolver.py
"""
Main EntityResolver class that orchestrates the entire entity resolution pipeline.

This module provides the primary interface for entity resolution, coordinating
all pipeline components including normalization, vectorization, clustering,
validation, merging, refinement, and scoring.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

# ============================================================================
# GPU AVAILABILITY CHECK AND CONDITIONAL IMPORTS
# ============================================================================
# Check for GPU libraries at module load time to provide early feedback
try:
    import cudf

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    # Use module logger for consistency
    logging.getLogger(__name__).warning(
        'GPU libraries (cuDF/cuML/CuPy) not found. GPU acceleration is disabled. '
        'Install rapids-ai packages to enable GPU support.'
    )

# ============================================================================
# LOCAL PACKAGE IMPORTS
# ============================================================================
# Configuration imports
# Persistence functionality
from . import persistence
from .address_processor import AddressProcessor
from .clusterer import EntityClusterer
from .config import ResolverConfig, load_config
from .merger import ClusterMerger

# Pipeline component imports - each handles a specific stage
from .normalizer import TextNormalizer
from .predictor import EntityPredictor
from .refiner import ClusterRefiner
from .reporter import ResolutionReporter
from .scorer import ConfidenceScorer

# Utility imports - be specific about what we're using
from .utils import validate_canonical_consistency, validate_no_duplicates
from .validator import ClusterValidator
from .vectorizer import EmbeddingOrchestrator


class EntityResolver:
    """
    Main orchestrator for GPU-accelerated entity resolution pipeline.

    This class coordinates all individual components (normalizer, vectorizer,
    clusterer, etc.) to execute the full entity resolution workflow for both
    training (fit) and prediction (transform) operations.

    Attributes:
        config (ResolverConfig): Configuration object containing all settings
        logger (logging.Logger): Logger instance for this resolver
        canonical_map_ (Dict): Mapping of cluster IDs to canonical entities (set after fitting)
        resolved_gdf_ (cudf.DataFrame): Final resolved dataframe (set after transform)
        _is_fitted (bool): Whether the resolver has been fitted to data
    """

    # Package version for tracking
    __version__ = '0.1.0'

    def __init__(self, config_path: str | None = None, *, config: ResolverConfig | None = None):
        """
        Initialize the EntityResolver and all sub-components.

        This constructor supports three initialization modes:
        1. Load configuration from YAML file (provide config_path)
        2. Use pre-loaded configuration object (provide config)
        3. Use default configuration (provide neither)

        Args:
            config_path: Optional path to YAML configuration file
            config: Pre-loaded ResolverConfig object (used internally for model loading)

        Raises:
            ImportError: If GPU libraries are not available
            ValueError: If both config_path and config are provided
        """
        # Enforce GPU requirement upfront with clear error message
        if not GPU_AVAILABLE:
            raise ImportError(
                'EntityResolver requires GPU acceleration. '
                'Please install cuDF, cuML, and CuPy: '
                'conda install -c rapidsai -c nvidia -c conda-forge rapids'
            )

        # Validate mutually exclusive parameters
        if config_path and config:
            raise ValueError(
                "Provide either 'config_path' or 'config', not both. "
                'config_path is for loading from file, config is for internal use.'
            )

        # Load or use provided configuration
        self.config = config if config else load_config(config_path)

        # Set up logging for this instance
        self.logger = self._setup_logger()
        self.logger.info(f'Initializing EntityResolver v{self.__version__}')

        # Initialize all pipeline components
        self._initialize_components()

        # Initialize state attributes that will be populated during fitting
        self.canonical_map_ = None
        self.resolved_gdf_ = None
        self._is_fitted = False

        self.logger.debug('EntityResolver initialization complete')

    def _initialize_components(self) -> None:
        """
        Instantiate all pipeline component classes with their configurations.

        This method creates instances of each component in the pipeline,
        passing the appropriate configuration sections to each.
        """
        self.logger.debug('Initializing pipeline components...')

        # Text normalization component
        self.normalizer = TextNormalizer(self.config.normalization, self.config.vectorizer)
        self.logger.debug('TextNormalizer initialized')

        # Address processing component (needs multiple configs for full functionality)
        self.address_processor = AddressProcessor(
            validation_config=self.config.validation,
            column_config=self.config.columns,
            vectorizer_config=self.config.vectorizer,  # For similarity parameters
        )
        self.logger.debug('AddressProcessor initialized')

        # Vectorization component for creating embeddings
        self.vectorizer = EmbeddingOrchestrator(self.config.vectorizer)
        self.logger.debug('EmbeddingOrchestrator initialized')

        # Clustering component for grouping similar entities
        self.clusterer = EntityClusterer(self.config.clusterer)
        self.logger.debug('EntityClusterer initialized')

        # Validation component for cluster quality checks
        self.validator = ClusterValidator(
            validation_config=self.config.validation,
            vectorizer_config=self.config.vectorizer,  # For similarity parameters
        )
        self.logger.debug('ClusterValidator initialized')

        # Merging component for combining related clusters
        self.merger = ClusterMerger(
            validation_config=self.config.validation,
            vectorizer_config=self.config.vectorizer,  # For similarity parameters
            cluster_config=self.config.clusterer,
        )
        self.logger.debug('ClusterMerger initialized')

        # Refinement component for final cluster processing
        self.refiner = ClusterRefiner(
            validation_config=self.config.validation,
            output_config=self.config.output,  # For formatting rules
            vectorizer_config=self.config.vectorizer,
        )
        self.logger.debug('ClusterRefiner initialized')

        # Scoring component for confidence metrics
        self.scorer = ConfidenceScorer(
            scoring_config=self.config.scoring,
            output_config=self.config.output,
            vectorizer_config=self.config.vectorizer,
            column_config=self.config.columns,
        )
        self.logger.debug('ConfidenceScorer initialized')

        # Prediction component for new data
        self.predictor = EntityPredictor()
        self.logger.debug('EntityPredictor initialized')

        # Reporting component for analysis and summaries
        self.reporter = ResolutionReporter(self.config)
        self.logger.debug('ResolutionReporter initialized')

        self.logger.info('All pipeline components initialized successfully')

    # ========================================================================
    # MAIN PUBLIC API METHODS
    # ========================================================================

    def fit(self, df: pd.DataFrame) -> EntityResolver:
        """
        Fit the resolver on training data, learning all model parameters.

        This method executes the complete training pipeline, including:
        - Text normalization and address processing
        - Vectorization model training
        - Clustering model training
        - Cluster validation and refinement
        - Canonical entity mapping creation

        Args:
            df: Input pandas DataFrame with entity data to resolve

        Returns:
            Self (fitted EntityResolver instance) for method chaining

        Raises:
            RuntimeError: If fitting fails at any pipeline stage
        """
        self.logger.info(f'{"=" * 60}')
        self.logger.info(f'Starting training on {len(df):,} records')
        self.logger.info(f'{"=" * 60}')

        # Convert pandas DataFrame to GPU-accelerated cuDF DataFrame
        gpu_dataframe = self._prepare_gpu_dataframe(df)

        # Execute the complete training pipeline
        self._execute_training_pipeline(gpu_dataframe)

        # Mark the resolver as fitted
        self._is_fitted = True

        # Log training completion statistics
        self.logger.info(f'{"=" * 60}')
        self.logger.info('Training complete')
        if self.canonical_map_ is not None:
            unique_entities = len(self.canonical_map_)
            self.logger.info(f'Built canonical map with {unique_entities:,} unique entities')
            self.logger.debug(f'Compression ratio: {len(df) / unique_entities:.2f}:1')
        self.logger.info(f'{"=" * 60}')

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the fitted resolver.

        This method applies the learned model to new records, assigning them
        to existing canonical entities based on similarity.

        Args:
            df: pandas DataFrame with new records to resolve

        Returns:
            pandas DataFrame with resolved entity information including:
            - canonical_name: The standardized entity name
            - cluster_id: The assigned cluster identifier
            - confidence_score: Resolution confidence (0-1)
            - Additional fields based on configuration

        Raises:
            RuntimeError: If resolver has not been fitted
        """
        if not self._is_fitted:
            raise RuntimeError('Resolver has not been fitted. Call fit() or fit_transform() first.')

        self.logger.info(f'{"=" * 60}')
        self.logger.info(f'Transforming {len(df):,} new records')
        self.logger.info(f'{"=" * 60}')

        # Convert to GPU DataFrame
        gpu_dataframe = self._prepare_gpu_dataframe(df)

        # Execute prediction pipeline
        resolved_gpu_df = self._execute_prediction_pipeline(gpu_dataframe)

        # Store GPU DataFrame for potential reporting
        self.resolved_gdf_ = resolved_gpu_df

        # Convert back to pandas DataFrame for return
        resolved_df = resolved_gpu_df.to_pandas()
        self.logger.debug(f'Converted {len(resolved_df):,} records back to pandas DataFrame')

        # Optionally split canonical address into components
        if self.config.output.split_address_components:
            self.logger.debug('Splitting canonical addresses into components')
            resolved_df = self.address_processor.split_canonical_address(resolved_df)

        self.logger.info('Transform complete')
        return resolved_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the resolver and transform data in a single operation.

        This convenience method combines fit() and transform() for cases where
        you want to both train the model and get resolved results for the
        training data.

        Args:
            df: pandas DataFrame to both train on and resolve

        Returns:
            pandas DataFrame with resolved entity information
        """
        self.logger.info(f'{"=" * 60}')
        self.logger.info(f'Starting fit & transform on {len(df):,} records')
        self.logger.info(f'{"=" * 60}')

        # Fit the model (this populates self.resolved_gdf_)
        self.fit(df)

        # The resolved data is already in self.resolved_gdf_ from fitting
        resolved_df = self.resolved_gdf_.to_pandas()
        self.logger.debug(f'Converted {len(resolved_df):,} records back to pandas DataFrame')

        # Optionally split canonical address into components
        if self.config.output.split_address_components:
            self.logger.debug('Splitting canonical addresses into components')
            resolved_df = self.address_processor.split_canonical_address(resolved_df)

        self.logger.info(f'{"=" * 60}')
        self.logger.info('Fit & transform complete')
        self.logger.info(f'{"=" * 60}')

        return resolved_df

    # ========================================================================
    # PERSISTENCE METHODS
    # ========================================================================

    def save_model(self, directory_path: str) -> None:
        """
        Save the fitted resolver to disk for later use.

        This method saves all trained components and configuration to the
        specified directory, allowing the model to be loaded and used later
        without retraining.

        Args:
            directory_path: Directory where model components will be saved

        Raises:
            RuntimeError: If resolver has not been fitted
            IOError: If save operation fails
        """
        if not self._is_fitted:
            raise RuntimeError('Cannot save unfitted resolver. Call fit() first.')

        self.logger.info(f'Saving resolver model to {directory_path}')
        persistence.save_model(self, directory_path)
        self.logger.info('Model saved successfully')

    @classmethod
    def load_model(cls, directory_path: str) -> EntityResolver:
        """
        Load a saved resolver from disk.

        This class method reconstructs a complete EntityResolver instance
        from saved components, ready for immediate use without retraining.

        Args:
            directory_path: Directory containing saved model components

        Returns:
            Fully reconstructed and functional EntityResolver instance

        Raises:
            IOError: If model files cannot be loaded
            ValueError: If saved model is incompatible or corrupted
        """
        # Use class-level logger for loading messages
        logger = logging.getLogger(__name__)
        logger.info(f'Loading resolver model from {directory_path}')

        # Load all saved components
        components = persistence.load_model_components(directory_path)

        # Create new instance with saved configuration
        resolver = cls(config=components['config'])

        # Restore component states
        resolver._restore_component_states(components)

        # Restore canonical map and fitted status
        resolver.canonical_map_ = components.get('canonical_map')
        resolver._is_fitted = True

        resolver.logger.info(f'Model successfully loaded from {directory_path}')
        return resolver

    # ========================================================================
    # PIPELINE EXECUTION METHODS
    # ========================================================================

    def _execute_training_pipeline(self, gdf: cudf.DataFrame) -> None:
        """
        Execute the complete training pipeline step-by-step.

        This method coordinates all training stages in sequence, with detailed
        logging and validation at each step.

        Args:
            gdf: GPU DataFrame with training data
        """
        # Step 1 & 2: Text normalization and address processing
        self.logger.info('Step 1/7: Normalizing text fields...')
        gdf = self.normalizer.normalize_text(gdf, self.config.columns.entity_col)
        self.logger.debug(f'Text normalization complete for {len(gdf):,} records')

        # Process addresses if configured
        if self.config.columns.address_cols:
            self.logger.info('Step 2/7: Processing and standardizing addresses...')
            gdf = self.address_processor.process_addresses(gdf, is_training=True)
            self.logger.debug('Address processing complete')

            # Consolidate entities by normalized address
            gdf = self.normalizer.consolidate_by_address(gdf)
            self.logger.debug(f'Consolidated to {len(gdf):,} unique address-entity combinations')
        else:
            self.logger.info(
                'Step 2/7: Skipping address processing (no address columns configured)'
            )

        # Step 3: Create vector embeddings
        self.logger.info('Step 3/7: Creating vector embeddings...')
        self.vectorizer.fit_transform(gdf)
        canonical_gdf = self.vectorizer.canonical_gdf.copy()
        vectors = self.vectorizer.combined_embeddings
        self.logger.debug(f'Created embeddings with shape {vectors.shape}')

        # Step 4: Cluster similar entities
        self.logger.info('Step 4/7: Clustering similar entities...')
        canonical_gdf, _ = self.clusterer.fit_transform(canonical_gdf, vectors)
        unique_clusters = canonical_gdf['cluster'].nunique()
        self.logger.debug(f'Initial clustering produced {unique_clusters:,} clusters')

        # Step 5: Validate and merge clusters
        self.logger.info('Step 5/7: Validating and merging clusters...')

        # Validate cluster quality and reassign outliers
        canonical_gdf = self.validator.validate_with_reassignment(canonical_gdf, self.vectorizer)
        self.logger.debug('Cluster validation and reassignment complete')

        # Merge highly similar clusters
        canonical_gdf = self.merger.merge_clusters(canonical_gdf)
        self.logger.debug(
            f'Cluster merging complete, {canonical_gdf["cluster"].nunique():,} clusters remain'
        )

        # Verify no duplicate cluster assignments
        validate_no_duplicates(canonical_gdf, 'cluster', 'merging')

        # Step 6: Refine clusters and build canonical mapping
        self.logger.info('Step 6/7: Refining clusters and building canonical map...')

        # Apply final refinement rules
        canonical_gdf = self.refiner.refine_clusters(canonical_gdf)
        self.logger.debug('Cluster refinement complete')

        # Verify no duplicate final cluster assignments
        validate_no_duplicates(canonical_gdf, 'final_cluster', 'refining')

        # Build mapping of clusters to canonical entities
        self.canonical_map_ = self.refiner.build_canonical_map(canonical_gdf)
        self.logger.debug(f'Built canonical map with {len(self.canonical_map_):,} entries')

        # Step 7: Apply canonical names and calculate scores
        self.logger.info('Step 7/7: Applying canonical names and scoring...')

        # Apply canonical mapping to all records
        canonical_gdf = self.refiner.apply_canonical_map(canonical_gdf, self.canonical_map_)

        # Validate canonical name consistency
        validate_canonical_consistency(canonical_gdf)
        self.logger.debug('Canonical name consistency validated')

        # Calculate confidence scores and quality flags
        canonical_gdf = self.scorer.score_and_flag(canonical_gdf)
        self.logger.debug('Confidence scoring complete')

        # Store final resolved DataFrame
        self.resolved_gdf_ = canonical_gdf
        self.logger.info('Training pipeline execution complete')

    def _execute_prediction_pipeline(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Execute the prediction pipeline for new data.

        This method applies the trained model to new records, using learned
        parameters to assign them to canonical entities.

        Args:
            gdf: GPU DataFrame with new data to resolve

        Returns:
            GPU DataFrame with resolution results
        """
        # Step 1 & 2: Text normalization and address processing
        self.logger.info('Step 1/5: Normalizing text fields...')
        gdf = self.normalizer.normalize_text(gdf, self.config.columns.entity_col)
        self.logger.debug(f'Text normalization complete for {len(gdf):,} records')

        # Process addresses if configured
        if self.config.columns.address_cols:
            self.logger.info('Step 2/5: Processing and standardizing addresses...')
            gdf = self.address_processor.process_addresses(gdf, is_training=False)
            self.logger.debug('Address processing complete')
        else:
            self.logger.info(
                'Step 2/5: Skipping address processing (no address columns configured)'
            )

        # Step 3: Create vector embeddings using trained models
        self.logger.info('Step 3/5: Creating vector embeddings...')
        gdf, vectors = self.vectorizer.transform(gdf)
        self.logger.debug(f'Created embeddings with shape {vectors.shape}')

        # Step 4: Predict cluster assignments
        self.logger.info('Step 4/5: Predicting cluster assignments...')
        gdf = self.predictor.predict(
            gdf, vectors, self.clusterer.cluster_model, self.canonical_map_
        )
        assigned_clusters = gdf['cluster'].nunique()
        self.logger.debug(f'Assigned records to {assigned_clusters:,} clusters')

        # Step 5: Apply canonical names and calculate scores
        self.logger.info('Step 5/5: Applying canonical names and scoring...')

        # Apply canonical mapping
        gdf = self.refiner.apply_canonical_map(gdf, self.canonical_map_)
        self.logger.debug('Canonical names applied')

        # Calculate confidence scores and quality flags
        gdf = self.scorer.score_and_flag(gdf)
        self.logger.debug('Confidence scoring complete')

        self.logger.info('Prediction pipeline execution complete')
        return gdf

    # ========================================================================
    # ANALYSIS AND REPORTING METHODS
    # ========================================================================

    def get_review_dataframe(self) -> pd.DataFrame:
        """
        Get a summary DataFrame for reviewing resolution results.

        This method provides a condensed view of the resolution results,
        useful for quality review and validation.

        Returns:
            pandas DataFrame with summary statistics and sample results

        Raises:
            RuntimeError: If no resolved data is available
        """
        if self.resolved_gdf_ is None:
            raise RuntimeError(
                'No resolved data available. Run fit_transform() or transform() first.'
            )

        self.logger.info('Generating review DataFrame')
        review_df = self.reporter.get_review_dataframe(self.resolved_gdf_)
        self.logger.debug(f'Review DataFrame contains {len(review_df):,} summary rows')
        return review_df

    def generate_report(self, original_df: pd.DataFrame) -> dict[str, Any]:
        """
        Generate detailed statistical report on resolution results.

        This method analyzes the resolution results and provides comprehensive
        statistics on clustering quality, data reduction, and other metrics.

        Args:
            original_df: Original input DataFrame for comparison

        Returns:
            Dictionary containing detailed resolution statistics:
            - summary: Overall resolution metrics
            - cluster_stats: Per-cluster statistics
            - quality_metrics: Resolution quality indicators
            - data_reduction: Compression and efficiency metrics

        Raises:
            RuntimeError: If no resolved data is available
        """
        if self.resolved_gdf_ is None:
            raise RuntimeError(
                'No resolved data available. Run fit_transform() or transform() first.'
            )

        self.logger.info('Generating comprehensive resolution report')
        report = self.reporter.generate_report(original_df, self.resolved_gdf_, self.canonical_map_)
        self.logger.debug('Report generation complete')
        return report

    # ========================================================================
    # UTILITY AND HELPER METHODS
    # ========================================================================

    def _restore_component_states(self, components: dict[str, Any]) -> None:
        """
        Restore component states from loaded model data.

        This method is used during model loading to restore the trained state
        of each pipeline component.

        Args:
            components: Dictionary containing saved component states
        """
        self.logger.debug('Restoring component states from loaded models...')

        # Restore vectorizer models if present
        if components.get('vectorizer_models'):
            self.logger.debug('Restoring vectorizer models')
            self.vectorizer.set_models(**components['vectorizer_models'])

        # Restore clusterer models if present
        if components.get('clusterer_models'):
            self.logger.debug('Restoring clusterer models')
            self.clusterer.set_models(**components['clusterer_models'])

        self.logger.debug('Component state restoration complete')

    def _setup_logger(self) -> logging.Logger:
        """
        Set up logging for the entire entity_resolver package.

        This method configures the package-level logger so that all modules
        inherit the same log level and handler configuration. This ensures
        consistent logging throughout the pipeline.

        Returns:
            Logger instance for this specific module
        """
        # Get the package-level logger (parent of all module loggers)
        package_logger = logging.getLogger('entity_resolver')

        # Set the log level on the package logger
        # This will apply to all child loggers (normalizer, vectorizer, etc.)
        package_logger.setLevel(self.config.output.log_level)

        # Stop messages from propagating to the root logger,
        # which prevents the duplicate output seen in Colab.
        package_logger.propagate = False

        # Only add a handler if one doesn't already exist
        # This prevents duplicate log messages when creating multiple EntityResolver instances
        if not package_logger.handlers:
            console_handler = logging.StreamHandler()

            # Set formatter for consistent log format
            log_format = logging.Formatter(
                fmt='%(asctime)s - %(levelname)-8s - [%(name)s] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
            )
            console_handler.setFormatter(log_format)

            # Set handler level to match package level
            console_handler.setLevel(self.config.output.log_level)

            # Add handler to package logger
            package_logger.addHandler(console_handler)

        # If log level changed, update existing handler
        elif package_logger.handlers:
            package_logger.handlers[0].setLevel(self.config.output.log_level)

        # Return a module-specific logger for the resolver's own messages
        return logging.getLogger(__name__)

    def _prepare_gpu_dataframe(self, df: pd.DataFrame) -> cudf.DataFrame:
        """
        Convert pandas DataFrame to GPU-accelerated cuDF DataFrame.

        This method handles the data transfer from CPU to GPU memory,
        with appropriate logging of the operation.

        Args:
            df: pandas DataFrame to convert

        Returns:
            cuDF DataFrame ready for GPU processing

        Raises:
            MemoryError: If insufficient GPU memory is available
        """
        self.logger.info(f'Transferring {len(df):,} records to GPU memory...')

        try:
            gpu_df = cudf.from_pandas(df)
            self.logger.debug(
                f'Successfully allocated {gpu_df.memory_usage().sum() / 1e6:.2f} MB on GPU'
            )
            return gpu_df
        except Exception as e:
            self.logger.error(f'Failed to transfer data to GPU: {e}')
            raise MemoryError(
                f'Unable to transfer {len(df):,} records to GPU. '
                f'Consider reducing batch size or upgrading GPU memory.'
            ) from e
