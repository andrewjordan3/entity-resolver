# entity_resolver/vectorizer.py
"""
High-Level Orchestrator for Multi-Context Entity Embedding.

This module provides the `EmbeddingOrchestrator`, the primary user-facing class
for the entire vectorization pipeline. Its role is to manage the complex process
of converting a DataFrame of raw entity data into three distinct, analysis-ready
embedding matrices.

Architectural Role
------------------
The orchestrator acts as a "manager" or "controller" in the vectorization process.
It does not perform the low-level vectorization itself. Instead, it coordinates
the work of three specialized "worker" classes (`SingleContextVectorizer`), one
for each embedding context (combined, name, and address).

Its key responsibilities are:
1.  **Data Ingestion**: Accepts a `cudf.DataFrame` of entity records.
2.  **Indexing and Alignment**: Creates a `canonical_gdf` with a stable, sequential
    index (`canonical_id`) to guarantee a one-to-one correspondence between the
    rows in the DataFrame and the rows in the output embedding matrices. This is
    critical for preventing data misalignment issues.
3.  **Text Stream Preparation**: Coordinates with the `embedding_streams` utility to
    prepare all the specialized text inputs required by the downstream vectorizers.
4.  **Process Delegation**: Manages three instances of `SingleContextVectorizer` and
    delegates the `fit_transform` task for each context to the appropriate worker.
5.  **State Management**: Stores the final, canonical embedding matrices as public
    attributes (`combined_embeddings`, `name_embeddings`, `address_embeddings`) for
    easy access.
6.  **Model Persistence**: Provides `get_models` and `set_models` methods to
    seamlessly save and load the trained state of all underlying models.

Workflow
--------
1.  Instantiate `EmbeddingOrchestrator` with a `VectorizerConfig`.
2.  Call the `.fit_transform(gdf)` method with your data.
3.  The orchestrator performs all internal steps.
4.  Access the results via `orchestrator.canonical_gdf`,
    `orchestrator.combined_embeddings`, etc.

Usage Example
-------------
```python
from entity_resolver.config import VectorizerConfig
from entity_resolver.embedding_orchestrator import EmbeddingOrchestrator
import cudf

# 1. Create a configuration object
config = VectorizerConfig()

# 2. Instantiate the orchestrator
orchestrator = EmbeddingOrchestrator(config)

# 3. Load your data into a cuDF DataFrame
gdf = cudf.read_csv("your_entity_data.csv")

# 4. Run the end-to-end vectorization process
orchestrator.fit_transform(gdf)

# 5. Access the aligned results
aligned_data = orchestrator.canonical_gdf
combined_vectors = orchestrator.combined_embeddings
name_vectors = orchestrator.name_embeddings
address_vectors = orchestrator.address_embeddings

# `aligned_data.loc[i]` corresponds to `combined_vectors[i]`
```
"""

import logging
from typing import Any

import cudf
import cupy

from .config import VectorizerConfig
from .context_vectorizer import SingleContextVectorizer
from .utils import prepare_text_streams

# Set up module-level logger
logger = logging.getLogger(__name__)


class EmbeddingOrchestrator:
    """
    Orchestrates multi-context vectorization of entity data.

    This high-level class manages the end-to-end process of converting a DataFrame
    of entities into three distinct, canonical embedding matrices: `combined`, `name`,
    and `address`. It ensures perfect alignment between the data and matrices via a
    stable `canonical_gdf`.

    Attributes:
        config (VectorizerConfig): The configuration object.
        canonical_gdf (cudf.DataFrame): A DataFrame with a stable 'canonical_id' index,
            ensuring alignment with the embedding matrices.
        combined_embeddings (cupy.ndarray): Embeddings from name + address data.
        name_embeddings (cupy.ndarray): Embeddings from name-only data.
        address_embeddings (cupy.ndarray): Embeddings from address-only data.
    """

    def __init__(self, config: VectorizerConfig):
        """
        Initializes the EmbeddingOrchestrator.

        Args:
            config: A `VectorizerConfig` object containing all parameters for
                    the vectorization and reduction pipeline.
        """
        self.config = config
        self.canonical_gdf: cudf.DataFrame | None = None
        self.combined_embeddings: cupy.ndarray | None = None
        self.name_embeddings: cupy.ndarray | None = None
        self.address_embeddings: cupy.ndarray | None = None

        # Instantiate three dedicated "worker" vectorizers, one for each context.
        # This encapsulates the state (trained models) for each context cleanly.
        self.combined_vectorizer = SingleContextVectorizer(config, 'combined')
        self.name_vectorizer = SingleContextVectorizer(config, 'name')
        self.address_vectorizer = SingleContextVectorizer(config, 'address')

        logger.info('Initialized EmbeddingOrchestrator with 3 distinct context vectorizers.')
        logger.debug(
            f'Orchestrator configured with final SVD components: {config.final_svd_components}'
        )

    def fit_transform(self, gdf: cudf.DataFrame) -> None:
        """
        Fits all models and transforms the data to create the three embedding matrices.

        This is the main entry point for the class. It executes the entire
        vectorization pipeline from start to finish. The results (the canonical
        DataFrame and the three embedding matrices) are stored as attributes on
        the instance upon completion.

        Args:
            gdf: The input `cudf.DataFrame` containing the raw, pre-processed
                 entity data. It must contain the columns required by the
                 `prepare_text_streams` function.
        """
        if not isinstance(gdf, cudf.DataFrame) or gdf.empty:
            logger.error('Input to fit_transform must be a non-empty cuDF DataFrame.')
            raise ValueError('Input must be a non-empty cuDF DataFrame.')

        logger.info(f'Starting embedding orchestration for {len(gdf):,} records.')

        # Step 1: Create the canonical DataFrame. This is a critical step for
        # ensuring that the rows of the output matrices can always be reliably
        # mapped back to the original data, even if the input gdf had a
        # non-standard or non-unique index.
        self.canonical_gdf = gdf.copy(deep=True)
        self.canonical_gdf['canonical_id'] = cupy.arange(len(gdf))
        self.canonical_gdf = self.canonical_gdf.set_index('canonical_id')
        logger.debug(
            "Created canonical_gdf with stable 'canonical_id' index for data-matrix alignment."
        )

        # Step 2: Prepare all text streams for all three contexts using the utility.
        # This call creates the specialized string inputs for every stream and context.
        logger.info('Preparing text streams for all three contexts (combined, name, address)...')
        all_streams = prepare_text_streams(
            self.canonical_gdf, use_address_in_encoding=self.config.use_address_in_encoding
        )
        logger.debug('Completed text stream preparation.')

        # Step 3: Sequentially process each context. The orchestrator delegates the
        # actual vectorization work to the specialized SingleContextVectorizer instances.
        logger.info("--- Starting 'combined' context vectorization ---")
        self.combined_embeddings = self.combined_vectorizer.fit_transform(all_streams.combined)
        logger.info(
            f"--- Completed 'combined' context. Matrix shape: {self.combined_embeddings.shape} ---"
        )

        logger.info("--- Starting 'name' context vectorization ---")
        self.name_embeddings = self.name_vectorizer.fit_transform(all_streams.name)
        logger.info(f"--- Completed 'name' context. Matrix shape: {self.name_embeddings.shape} ---")

        logger.info("--- Starting 'address' context vectorization ---")
        self.address_embeddings = self.address_vectorizer.fit_transform(all_streams.address)
        logger.info(
            f"--- Completed 'address' context. Matrix shape: {self.address_embeddings.shape} ---"
        )

        logger.info(
            'Embedding orchestration complete. All 3 embedding matrices are now available as attributes.'
        )

    def transform(self, gdf: cudf.DataFrame) -> tuple[cupy.ndarray, cupy.ndarray, cupy.ndarray]:
        """
        Transforms new data using the already-fitted models.

        This method is for inference. It takes a new DataFrame of entity data and
        applies the pre-existing, trained vectorization pipeline to it. It does not
        modify the internal state of the orchestrator (e.g., `canonical_gdf`).

        Args:
            gdf: A `cudf.DataFrame` containing new, unseen entity data to be transformed.

        Returns:
            A tuple containing the three new embedding matrices in the order:
            (combined_embeddings, name_embeddings, address_embeddings).

        Raises:
            RuntimeError: If the method is called before `fit_transform` has been
                          run or before models have been loaded via `set_models`.
        """
        if not self._is_fitted:
            raise RuntimeError(
                'The orchestrator has not been fitted yet. '
                "Call 'fit_transform' or 'set_models' before calling 'transform'."
            )

        if not isinstance(gdf, cudf.DataFrame) or gdf.empty:
            logger.error('Input to transform must be a non-empty cuDF DataFrame.')
            raise ValueError('Input must be a non-empty cuDF DataFrame.')

        logger.info(f'Transforming {len(gdf):,} new records using pre-fitted models.')

        # Step 1: Prepare text streams for the new, incoming data.
        logger.debug('Preparing text streams for the new data...')
        all_streams = prepare_text_streams(
            gdf, use_address_in_encoding=self.config.use_address_in_encoding
        )
        logger.debug('Text stream preparation for new data complete.')

        # Step 2: Delegate the transform task to each of the worker vectorizers.
        # These calls will use the models that were trained during the `fit_transform` phase.
        logger.info("--- Transforming 'combined' context ---")
        combined_vectors = self.combined_vectorizer.transform(all_streams.combined)

        logger.info("--- Transforming 'name' context ---")
        name_vectors = self.name_vectorizer.transform(all_streams.name)

        logger.info("--- Transforming 'address' context ---")
        address_vectors = self.address_vectorizer.transform(all_streams.address)

        logger.info('Transformation of new data complete.')

        return (combined_vectors, name_vectors, address_vectors)

    def get_aligned_embeddings(
        self,
        data_slice: cudf.DataFrame | cudf.Series,
        validate_indices: bool = True,
        include_data: bool = False,
    ) -> dict[str, cupy.ndarray] | tuple[cudf.DataFrame, dict[str, cupy.ndarray]]:
        """
        Retrieves embedding matrix rows that align perfectly with a given data slice.

        This is a critical utility for post-clustering analysis and downstream operations.
        After running `fit_transform`, the `canonical_gdf` is typically filtered based on
        cluster assignments, quality scores, or other criteria. This method ensures you get
        the exact embedding vectors corresponding to your filtered data, maintaining perfect
        row-to-row alignment.

        The method is robust to different data slice formats:
        - DataFrame with `canonical_id` as index
        - DataFrame with `canonical_id` as a column (after reset_index)
        - Series with `canonical_id` as index

        Performance Characteristics
        ---------------------------
        This method uses advanced integer array indexing on GPU arrays, which is highly
        efficient even for large slices. The operation is O(n) where n is the size of
        the data slice, not the original dataset size.

        Typical workflow:
        1. orchestrator.fit_transform(gdf) creates canonical_gdf with canonical_id index
        2. User performs clustering/filtering: filtered_gdf = canonical_gdf[canonical_gdf['cluster'] == 5]
        3. User retrieves aligned embeddings: embeddings = orchestrator.get_aligned_embeddings(filtered_gdf)
        4. Embeddings dict contains vectors in same order as filtered_gdf rows

        Parameters
        ----------
        data_slice : cudf.DataFrame or cudf.Series
            A subset of the orchestrator's `canonical_gdf`. Must contain `canonical_id`
            values either as the index or as a column. The order of rows in this slice
            determines the order of vectors in the returned embedding matrices.

        validate_indices : bool, default=True
            If True, performs validation checks on the canonical_id values:
            - Ensures all indices are within valid bounds [0, N) where N is dataset size
            - Checks for negative indices (indicates data corruption)
            - Warns if duplicate indices are present (may indicate user error)
            - Verifies indices exist in the original canonical_gdf
            Set to False to skip validation for performance (only if you're certain indices are valid).

        include_data : bool, default=False
            If True, returns a tuple of (aligned_data, aligned_embeddings) where aligned_data
            is the data_slice with canonical_id as the index (if it wasn't already). This is
            convenient for keeping data and embeddings together in subsequent operations.
            If False, returns only the aligned_embeddings dictionary.

        Returns
        -------
        If include_data=False (default):
            dict of {str: cupy.ndarray}
                dictionary mapping context names to their aligned embedding matrices:
                - 'combined': Embeddings from name + address (shape: n_slice × d_combined)
                - 'name': Embeddings from name only (shape: n_slice × d_name)
                - 'address': Embeddings from address only (shape: n_slice × d_address)

        If include_data=True:
            tuple of (cudf.DataFrame, dict of {str: cupy.ndarray})
                First element: The data_slice with canonical_id guaranteed as index
                Second element: The aligned embeddings dictionary (as above)

        Raises
        ------
        RuntimeError
            If the orchestrator has not been fitted yet (no embeddings available).

        ValueError
            - If input is not a cudf.DataFrame or cudf.Series
            - If input is empty (no rows to align)
            - If canonical_id is not found as index or column
            - If any source embedding matrix is None/missing
            - If validation is enabled and indices are invalid:
                * Indices out of bounds (< 0 or >= dataset size)
                * Indices not found in original canonical_gdf

        Warnings
        --------
        If validation is enabled and duplicate canonical_id values are detected, a warning
        is logged. Duplicates will cause the same embedding vector to appear multiple times
        in the output, which may indicate an error in your filtering logic.

        Examples
        --------
        Basic usage after clustering:

        >>> # After fit_transform and clustering
        >>> orchestrator.fit_transform(gdf)
        >>> # ... perform clustering ...
        >>>
        >>> # Filter to cluster 3
        >>> cluster_3_data = orchestrator.canonical_gdf[
        ...     orchestrator.canonical_gdf['cluster_id'] == 3
        ... ]
        >>>
        >>> # Get aligned embeddings for this cluster
        >>> embeddings = orchestrator.get_aligned_embeddings(cluster_3_data)
        >>>
        >>> # Now you can compute intra-cluster distances
        >>> from cuml.metrics import pairwise_distances
        >>> distances = pairwise_distances(embeddings['combined'], metric='cosine')

        With data included:

        >>> # Get both data and embeddings together
        >>> data, embeddings = orchestrator.get_aligned_embeddings(
        ...     cluster_3_data,
        ...     include_data=True
        ... )
        >>>
        >>> # Data and embeddings guaranteed to align
        >>> assert len(data) == embeddings['combined'].shape[0]
        >>> assert data.index.name == 'canonical_id'

        Skip validation for performance (use carefully):

        >>> # If you know indices are valid, skip validation
        >>> embeddings = orchestrator.get_aligned_embeddings(
        ...     large_slice,
        ...     validate_indices=False  # Faster but no safety checks
        ... )

        Notes
        -----
        - The returned embedding matrices maintain the exact row order of the input data_slice
        - This method does NOT modify the orchestrator's state or stored embeddings
        - Memory usage: Returns new arrays (not views), so large slices require memory
        - For very small slices (< 100 rows), validation overhead is negligible
        - For very large slices (> 1M rows), consider validate_indices=False after testing

        See Also
        --------
        EmbeddingOrchestrator.fit_transform : Creates the canonical_gdf and embeddings
        EmbeddingOrchestrator.transform : Transform new data without refitting
        """
        # -------------------------------------------------------------------------
        # Phase 1: Input validation and state checking
        # -------------------------------------------------------------------------
        logger.info(f'Aligning embeddings for a data slice with {len(data_slice):,} records.')

        # Check if orchestrator has been fitted
        if not self._is_fitted:
            logger.error('Alignment requested but orchestrator is not fitted.')
            raise RuntimeError(
                'Cannot align embeddings: the orchestrator has not been fitted yet. '
                'Call fit_transform() first to create embeddings.'
            )

        # Validate input type and non-emptiness
        if not isinstance(data_slice, (cudf.DataFrame, cudf.Series)):
            logger.error(f'Invalid input type: {type(data_slice)}')
            raise ValueError(
                'Input data_slice must be a cudf.DataFrame or cudf.Series. '
                f'Received: {type(data_slice)}'
            )

        if data_slice.empty:
            logger.error('Alignment requested for empty data slice.')
            raise ValueError('Input data_slice is empty. Cannot align embeddings for zero records.')

        # Verify all embedding matrices exist
        if self.combined_embeddings is None:
            raise ValueError('Combined embeddings are missing. Run fit_transform() first.')
        if self.name_embeddings is None:
            raise ValueError('Name embeddings are missing. Run fit_transform() first.')
        if self.address_embeddings is None:
            raise ValueError('Address embeddings are missing. Run fit_transform() first.')

        logger.debug(
            f'Source embedding shapes: '
            f'combined={self.combined_embeddings.shape}, '
            f'name={self.name_embeddings.shape}, '
            f'address={self.address_embeddings.shape}'
        )

        # -------------------------------------------------------------------------
        # Phase 2: Extract canonical_id values from data slice
        # -------------------------------------------------------------------------
        # Robustly find canonical_id whether it's in the index or a column.
        # This handles cases where users have reset_index() on their slice.

        canonical_id_source = None  # Track where we found it for logging

        if data_slice.index.name == 'canonical_id':
            logger.debug("Found 'canonical_id' as the index of the data slice.")
            alignment_indices = data_slice.index.to_cupy()
            canonical_id_source = 'index'
            aligned_data_with_index = data_slice  # Already has correct index

        elif 'canonical_id' in getattr(data_slice, 'columns', []):
            logger.debug("Found 'canonical_id' as a column in the data slice.")
            alignment_indices = data_slice['canonical_id'].to_cupy()
            canonical_id_source = 'column'

            # If user wants data back, set canonical_id as index for consistency
            if include_data:
                aligned_data_with_index = data_slice.set_index('canonical_id')
            else:
                aligned_data_with_index = None

        else:
            logger.error(
                "Alignment failed: 'canonical_id' not found as index or column. "
                f'Available columns: {getattr(data_slice, "columns", []).tolist() if hasattr(data_slice, "columns") else "N/A (Series)"}, '
                f'Index name: {data_slice.index.name}'
            )
            raise ValueError(
                "Input data_slice must have 'canonical_id' as either its index or as a column. "
                'The canonical_id is created during fit_transform and is required for alignment.'
            )

        num_indices = len(alignment_indices)
        logger.debug(
            f'Extracted {num_indices:,} canonical_id values from data_slice {canonical_id_source}.'
        )

        # -------------------------------------------------------------------------
        # Phase 3: Validate indices (optional but recommended)
        # -------------------------------------------------------------------------
        if validate_indices:
            logger.debug('Performing validation checks on canonical_id indices...')

            # Check 1: Ensure indices are within valid bounds
            max_valid_index = len(self.combined_embeddings) - 1
            min_index = int(alignment_indices.min())
            max_index = int(alignment_indices.max())

            if min_index < 0:
                logger.error(f'Negative canonical_id detected: {min_index}')
                raise ValueError(
                    f'Invalid canonical_id values: found negative index {min_index}. '
                    'canonical_id must be non-negative integers. This indicates data corruption.'
                )

            if max_index > max_valid_index:
                logger.error(
                    f'Out-of-bounds canonical_id: {max_index} exceeds maximum valid index {max_valid_index}'
                )
                raise ValueError(
                    f'Invalid canonical_id values: index {max_index} is out of bounds. '
                    f'Valid range is [0, {max_valid_index}]. '
                    'This data_slice contains indices not present in the original canonical_gdf.'
                )

            logger.debug(
                f'Index range validation passed: [{min_index}, {max_index}] within [0, {max_valid_index}]'
            )

            # Check 2: Warn about duplicate indices (may indicate user error)
            unique_indices = cupy.unique(alignment_indices)
            num_unique = len(unique_indices)

            if num_unique < num_indices:
                num_duplicates = num_indices - num_unique
                logger.warning(
                    f'Duplicate canonical_id values detected: {num_duplicates:,} duplicates among {num_indices:,} total indices. '
                    'This will cause the same embedding vectors to appear multiple times in the output. '
                    'If this is unintentional, check your data filtering logic.'
                )
            else:
                logger.debug(
                    f'No duplicate indices detected (all {num_indices:,} indices are unique).'
                )

            # Check 3: Verify indices exist in original canonical_gdf (if available)
            if self.canonical_gdf is not None and 'canonical_id' in [
                self.canonical_gdf.index.name
            ] + (
                self.canonical_gdf.columns.tolist()
                if hasattr(self.canonical_gdf, 'columns')
                else []
            ):
                # Get canonical_id values from stored canonical_gdf
                if self.canonical_gdf.index.name == 'canonical_id':
                    valid_canonical_ids = self.canonical_gdf.index.to_cupy()
                elif 'canonical_id' in self.canonical_gdf.columns:
                    valid_canonical_ids = self.canonical_gdf['canonical_id'].to_cupy()
                else:
                    valid_canonical_ids = None

                if valid_canonical_ids is not None:
                    # Check if all alignment_indices exist in valid_canonical_ids
                    # Using isin-style check with unique values for efficiency
                    valid_set = cupy.unique(valid_canonical_ids)
                    are_valid = cupy.isin(unique_indices, valid_set)

                    if not cupy.all(are_valid):
                        invalid_indices = unique_indices[~are_valid]
                        num_invalid = len(invalid_indices)
                        logger.error(
                            f"Found {num_invalid} canonical_id values that don't exist in canonical_gdf"
                        )
                        raise ValueError(
                            f'Invalid canonical_id values: {num_invalid} indices from data_slice '
                            'do not exist in the original canonical_gdf. '
                            'This data_slice was not derived from the same fit_transform run.'
                        )

                    logger.debug(
                        'All canonical_id values verified to exist in original canonical_gdf.'
                    )

            logger.info('Validation checks passed: all canonical_id values are valid.')
        else:
            logger.debug('Skipping validation checks (validate_indices=False).')

        # -------------------------------------------------------------------------
        # Phase 4: Perform advanced integer array indexing
        # -------------------------------------------------------------------------
        # GPU-accelerated indexing operation - highly efficient even for large slices
        logger.debug('Performing advanced integer array indexing on embedding matrices...')

        try:
            aligned_combined = self.combined_embeddings[alignment_indices]
            aligned_name = self.name_embeddings[alignment_indices]
            aligned_address = self.address_embeddings[alignment_indices]
        except IndexError as e:
            logger.error(f'Indexing failed during alignment: {e}')
            raise ValueError(
                f'Failed to index embedding matrices with provided canonical_id values. '
                f'This should not happen after validation. Original error: {e}'
            ) from e

        # Verify output shapes match expectations
        expected_shape_0 = num_indices
        actual_shapes = {
            'combined': aligned_combined.shape,
            'name': aligned_name.shape,
            'address': aligned_address.shape,
        }

        for context_name, shape in actual_shapes.items():
            if shape[0] != expected_shape_0:
                logger.error(
                    f'Shape mismatch for {context_name}: expected {expected_shape_0} rows, got {shape[0]}'
                )
                raise RuntimeError(
                    f'Internal error: aligned {context_name} embeddings have incorrect shape {shape}. '
                    f'Expected ({expected_shape_0}, ?).'
                )

        logger.info(
            f'Successfully aligned all three embedding contexts to the data slice. '
            f'Output shapes: combined={aligned_combined.shape}, '
            f'name={aligned_name.shape}, address={aligned_address.shape}'
        )

        # -------------------------------------------------------------------------
        # Phase 5: Package and return results
        # -------------------------------------------------------------------------
        aligned_embeddings = {
            'combined': aligned_combined,
            'name': aligned_name,
            'address': aligned_address,
        }

        if include_data:
            if aligned_data_with_index is None:
                # Edge case: shouldn't happen, but handle gracefully
                logger.warning(
                    'include_data=True but aligned_data_with_index is None. Returning data_slice as-is.'
                )
                aligned_data_with_index = data_slice

            logger.debug('Returning both aligned data and embeddings.')
            return aligned_data_with_index, aligned_embeddings
        else:
            logger.debug('Returning aligned embeddings only.')
            return aligned_embeddings

    def get_models(self) -> dict[str, Any]:
        """
        Gathers all trained models from all contexts for persistence.

        This method collects the dictionaries of trained encoders and reduction
        models from each of the three `SingleContextVectorizer` instances and
        packages them into a single, top-level dictionary, keyed by context.

        Returns:
            A dictionary containing the trained models for all three contexts,
            structured as {'combined': {...}, 'name': {...}, 'address': {...}}.
        """
        logger.info('Gathering trained models from all contexts for persistence.')
        return {
            'combined': self.combined_vectorizer.get_models(),
            'name': self.name_vectorizer.get_models(),
            'address': self.address_vectorizer.get_models(),
        }

    def set_models(self, all_models: dict[str, Any]):
        """
        Loads pre-trained models into all context vectorizers.

        This method allows you to restore a previously trained state. It takes a
        dictionary (typically loaded from a file) and distributes the model
        dictionaries to the appropriate `SingleContextVectorizer` instance based
        on the context keys ('combined', 'name', 'address').

        Args:
            all_models: A dictionary containing the trained models for each context.
                        Should match the structure returned by `get_models`.
        """
        logger.info('Loading pre-trained models into all context vectorizers.')

        # Safely load models for each context, logging a warning if a context is missing.
        if 'combined' in all_models:
            self.combined_vectorizer.set_models(all_models['combined'])
        else:
            logger.warning("No models found for 'combined' context during set_models call.")

        if 'name' in all_models:
            self.name_vectorizer.set_models(all_models['name'])
        else:
            logger.warning("No models found for 'name' context during set_models call.")

        if 'address' in all_models:
            self.address_vectorizer.set_models(all_models['address'])
        else:
            logger.warning("No models found for 'address' context during set_models call.")

        logger.info('Finished loading models for all contexts.')

    @property
    def _is_fitted(self) -> bool:
        """Check if all context vectorizers have been fitted."""
        return (
            self.combined_vectorizer._is_fitted
            and self.name_vectorizer._is_fitted
            and self.address_vectorizer._is_fitted
        )

    @property
    def embeddings(self) -> dict[str, cupy.ndarray]:
        """Convenient access to all embedding matrices."""
        return {
            'combined': self.combined_embeddings,
            'name': self.name_embeddings,
            'address': self.address_embeddings,
        }
