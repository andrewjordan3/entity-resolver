# entity_resolver/clusterer.py
"""
Entity clustering pipeline combining UMAP manifold learning and HDBSCAN density clustering.

This module implements a sophisticated clustering approach that combines:
1. UMAP ensemble for robust dimensionality reduction
2. HDBSCAN for density-based clustering
3. SNN graph clustering for noise rescue and high recall

The ensemble approach balances precision (HDBSCAN) with recall (SNN) to achieve
robust entity resolution even with challenging data distributions.

Mathematical Foundation:
- UMAP ensemble: Creates multiple manifold projections with diverse parameters,
  then combines them via kernel PCA consensus for stability
- HDBSCAN: Finds dense regions in the reduced space as core clusters
- SNN rescue: Builds a mutual k-NN graph to recover noise points and find
  additional structure missed by density-based methods
- Purity-based ensemble: Maps SNN clusters to HDBSCAN clusters based on
  overlap purity, ensuring high precision while improving recall
"""

import logging
from typing import Any, Dict, List, Tuple, Optional

import cupy as cp
import cudf
import numpy as np

# GPU Library Imports
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
import cugraph

# Local Package Imports
from .config import ClustererConfig, EnsembleParams
from .utils import (
    normalize_rows,
    build_mutual_rank_graph,
    attach_noise_points,
    merge_snn_clusters,
    create_consensus_embedding,
    gpu_memory_cleanup,
)

# Set up module-level logger
logger = logging.getLogger(__name__)


class EntityClusterer:
    """
    Orchestrate entity clustering using UMAP ensemble and HDBSCAN with SNN rescue.

    This class implements a multi-stage clustering pipeline:
    
    Training Pipeline:
    1. UMAP ensemble → consensus embedding
    2. HDBSCAN → core clusters + noise points
    3. SNN graph clustering → rescue noise / mint new clusters
    
    Inference Pipeline:
    1. Transform via fitted UMAP ensemble
    2. Predict cluster membership with fitted HDBSCAN
    
    Attributes:
        config (ClustererConfig): Configuration for all clustering parameters
        cluster_model (HDBSCAN): Fitted HDBSCAN model
        umap_ensemble (List[UMAP]): Collection of fitted UMAP models
        random_state (int): Random seed for reproducibility
    """

    def __init__(self, config: ClustererConfig) -> None:
        """
        Initialize the clusterer with configuration.
        
        Args:
            config: ClustererConfig containing all clustering parameters
        """
        self.config = config
        self.cluster_model: Optional[HDBSCAN] = None
        self.umap_ensemble: List[UMAP] = []
        
        # Extract random state for consistent seeding
        self.random_state = self.config.umap_params.random_state 

        # Initialize inverse vector mapping for UMAP input
        self._current_inverse_indices: cp.ndarray | None = None
        
        logger.info("Initialized EntityClusterer")
        logger.debug(f"UMAP ensemble size: {self.config.umap_n_runs}")
        logger.debug(f"HDBSCAN min_cluster_size: {self.config.hdbscan_params.min_cluster_size}")
        logger.debug(f"SNN k_neighbors: {self.config.snn_clustering_params.k_neighbors}")

    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    def fit_transform(
        self, 
        gdf: cudf.DataFrame, 
        vectors: cp.ndarray
    ) -> Tuple[cudf.DataFrame, cp.ndarray]:
        """
        Fit clustering models and transform data.
        
        This method performs the complete training pipeline:
        1. Ensemble UMAP for robust dimensionality reduction
        2. HDBSCAN for core density-based clustering
        3. SNN graph clustering for noise rescue
        4. Ensemble combination for final cluster assignments
        
        Args:
            gdf: Input DataFrame to add cluster labels to.
                 Must not contain 'cluster' or 'cluster_probability' columns.
            vectors: High-dimensional feature vectors to cluster.
                    Shape: (n_samples, n_features).
                    Should be normalized if using cosine similarity.
            
        Returns:
            Tuple of (DataFrame with cluster labels, reduced vectors)
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If clustering fails
        """
        logger.info(f"Starting fit_transform with {len(gdf):,} records, {vectors.shape[1]} features")
        self._validate_inputs(gdf, vectors)

        return self._process_clustering(gdf, vectors, is_training=True)

    def transform(
        self, 
        gdf: cudf.DataFrame, 
        vectors: cp.ndarray
    ) -> Tuple[cudf.DataFrame, cp.ndarray]:
        """
        Transform new data using fitted models.
        
        This method applies the trained UMAP and HDBSCAN models to new data,
        assigning cluster memberships based on the learned structure.
        
        Args:
            gdf: Input DataFrame to add cluster predictions to
            vectors: Feature vectors to assign to clusters.
                    Must have same number of features as training data.
            
        Returns:
            Tuple of (DataFrame with cluster predictions, reduced vectors)
            
        Raises:
            RuntimeError: If models haven't been fitted (call fit_transform first)
            ValueError: If inputs are invalid or dimensions don't match training
        """
        logger.info(f"Starting transform with {len(gdf):,} records")
        
        # Check models are fitted
        if not self.umap_ensemble or not self.cluster_model:
            raise RuntimeError(
                "Models not fitted. Call fit_transform() before transform()."
            )
        
        # Validate inputs
        self._validate_inputs(gdf, vectors)
        
        # Process clustering in inference mode
        result_df, reduced_vectors = self._process_clustering(gdf, vectors, is_training=False)

        return result_df, reduced_vectors

    # ========================================================================
    # CORE ORCHESTRATION
    # ========================================================================
    
    def _process_clustering(
        self,
        gdf: cudf.DataFrame,
        vectors: cp.ndarray,
        is_training: bool,
    ) -> Tuple[cudf.DataFrame, cp.ndarray]:
        """
        Execute clustering pipeline for training or inference.
        
        Args:
            gdf: Input DataFrame
            vectors: Feature vectors
            is_training: Whether to fit new models or use existing
            
        Returns:
            Tuple of (clustered DataFrame, reduced vectors)
        """
        operation_mode = "Training" if is_training else "Inference"
        logger.info(f"Processing clustering in {operation_mode} mode")
        
        # Work on a copy to avoid mutating input
        gdf = gdf.copy()

        # Ensure proper dtype and memory layout for GPU operations
        vectors = self._prepare_vectors(vectors)
        
        # Step 1: Dimensionality reduction via UMAP ensemble
        logger.info("Step 1: Dimensionality reduction with UMAP ensemble")
        reduced_vectors = self._run_umap_ensemble(vectors, is_training)
        
        if is_training:
            # Training path: Full clustering pipeline
            logger.info("Step 2: Core clustering with HDBSCAN")
            gdf_clustered = self._run_hdbscan(gdf, reduced_vectors)
            
            logger.info("Step 3: SNN graph clustering for noise rescue")
            snn_labels = self._run_snn_engine(reduced_vectors, gdf_clustered["cluster"].values)
            
            logger.info("Step 4: Ensemble HDBSCAN and SNN results")
            gdf_final = self._ensemble_cluster_labels(gdf_clustered, snn_labels)
            
            # Log final clustering statistics
            self._log_clustering_stats(gdf_final)
            
        else:
            # Inference path: Predict using fitted models
            logger.info("Step 2: Predicting cluster assignments")
            gdf_final = self._predict_clusters(gdf, reduced_vectors)
        
        return gdf_final, reduced_vectors
    
    def _prepare_vectors(self, vectors: cp.ndarray) -> cp.ndarray:
        """
        Prepare vectors for GPU operations (float32, C-contiguous).
        
        Ensures vectors are in the optimal format for GPU computation:
        - float32 dtype for efficiency
        - C-contiguous memory layout for coalesced memory access
        
        Args:
            vectors: Input vectors of any dtype/layout
            
        Returns:
            Properly formatted vectors (may be same object if already optimal)
        """
        # Convert to float32 if needed
        if vectors.dtype != cp.float32:
            logger.debug(f"Converting vectors from {vectors.dtype} to float32")
            vectors = vectors.astype(cp.float32, copy=False)
        
        # Ensure C-contiguous memory layout for GPU efficiency
        if not vectors.flags.c_contiguous:
            logger.debug("Making vectors C-contiguous for GPU operations")
            vectors = cp.ascontiguousarray(vectors)
        
        logger.debug(f"Prepared vectors: shape={vectors.shape}, dtype={vectors.dtype}")
        return vectors
    
    def _predict_clusters(
        self, 
        gdf: cudf.DataFrame, 
        reduced_vectors: cp.ndarray
    ) -> cudf.DataFrame:
        """
        Predict cluster assignments for new data.
        
        Args:
            gdf: DataFrame to add predictions to
            reduced_vectors: Reduced feature vectors
            
        Returns:
            DataFrame with cluster predictions
            
        Raises:
            RuntimeError: If models aren't fitted
            NotImplementedError: If HDBSCAN doesn't support predict
        """
        if not self.cluster_model:
            raise RuntimeError(
                "HDBSCAN model not fitted. Call fit_transform() first."
            )
        
        # Check if predict is available
        if not hasattr(self.cluster_model, "predict"):
            raise NotImplementedError(
                "This HDBSCAN build doesn't support predict(). "
                "Consider using approximate predict or nearest-centroid assignment."
            )
        
        # Assign default confidence for predictions
        # (HDBSCAN predict doesn't provide probabilities)
        gdf["cluster"] = self.cluster_model.predict(reduced_vectors)
        
        # Assign default confidence for predictions
        default_conf = float(self.config.ensemble_params.default_rescue_conf)
        gdf["cluster_probability"] = cp.where(
            gdf["cluster"] != -1, 
            default_conf, 
            0.0
        )
        
        # Log prediction statistics
        n_assigned = int((gdf["cluster"] != -1).sum())
        logger.info(f"Predicted {n_assigned:,}/{len(gdf):,} records to clusters")
        
        return gdf

    # ========================================================================
    # CLUSTERING COMPONENTS
    # ========================================================================
    
    def _run_hdbscan(
        self, 
        gdf: cudf.DataFrame, 
        reduced_vectors: cp.ndarray
    ) -> cudf.DataFrame:
        """
        Run HDBSCAN density-based clustering.
        
        HDBSCAN finds dense regions in the reduced space as core clusters,
        leaving sparse regions as noise. This provides high-precision clusters
        but may have lower recall.
        
        Args:
            gdf: DataFrame to add cluster labels to
            reduced_vectors: Low-dimensional embeddings from UMAP
            
        Returns:
            DataFrame with HDBSCAN cluster labels and probabilities
            
        Raises:
            RuntimeError: If HDBSCAN fails to find any clusters
        """
        logger.info("Running HDBSCAN for core clustering")
        logger.debug(f"HDBSCAN parameters: {self.config.hdbscan_params}")
        
        try:
            # Fit HDBSCAN
            clusterer = HDBSCAN(**self.config.hdbscan_params.model_dump())
            clusterer.fit(reduced_vectors)
            
        except Exception as e:
            logger.error(f"HDBSCAN failed: {e}")
            raise RuntimeError(f"HDBSCAN clustering failed: {e}")
        
        # Store fitted model for later use
        self.cluster_model = clusterer
        
        # Add cluster assignments to DataFrame
        gdf["cluster"] = clusterer.labels_
        gdf["cluster_probability"] = clusterer.probabilities_
        
        # Calculate and log statistics
        n_clusters = int(gdf["cluster"][gdf["cluster"] != -1].nunique())
        n_noise = int((gdf["cluster"] == -1).sum())
        noise_rate = n_noise / max(len(gdf), 1)
        
        logger.info(
            f"HDBSCAN results: {n_clusters} clusters, "
            f"{n_noise:,} noise points ({noise_rate:.1%})"
        )
        
        # Log cluster size distribution
        if n_clusters > 0:
            cluster_sizes = gdf[gdf["cluster"] != -1]["cluster"].value_counts()
            logger.debug(
                f"Cluster sizes: min={cluster_sizes.min()}, "
                f"max={cluster_sizes.max()}, "
                f"mean={cluster_sizes.mean():.1f}"
                f"median={cluster_sizes.median():.0f}"
            )
        
        # Warn if noise rate is too high
        max_noise_warn = float(self.config.max_noise_rate_warn)
        if noise_rate > max_noise_warn:
            logger.warning(
                f"High noise rate ({noise_rate:.1%} > {max_noise_warn:.1%}). "
                f"Consider adjusting min_cluster_size or UMAP parameters."
            )
        
        return gdf

    def _run_snn_engine(self, reduced_vectors: cp.ndarray, hdbscan_labels: cp.ndarray = None) -> cp.ndarray:
        """
        Run SNN graph clustering for noise rescue.
        
        This three-stage process:
        1. Build mutual rank graph and detect communities
        2. Attach noise points to nearby communities
        3. Merge over-split clusters
        
        Args:
            reduced_vectors: Low-dimensional embeddings
            
        Returns:
            Array of SNN cluster labels (-1 for noise)
        """
        logger.info("Starting SNN Graph Clustering Engine")
        
        # Normalize vectors for cosine similarity - UMAP output is not normalized
        logger.debug("Normalizing vectors for cosine similarity")
        vectors_norm = normalize_rows(reduced_vectors, copy=True)
        
        # Stage 1: Community detection
        logger.info("SNN Stage 1: Building graph and finding communities")
        k_neighbors = self.config.snn_clustering_params.k_neighbors
        logger.debug(f"Building mutual rank graph with k={k_neighbors}")
        
        snn_graph, _ = build_mutual_rank_graph(vectors_norm, k_neighbors)
        
        if snn_graph.number_of_edges() == 0:
            logger.warning("SNN graph has no edges. All points treated as noise.")
            return cp.full(reduced_vectors.shape[0], -1, dtype=cp.int32)
        
        logger.debug(
            f"Graph statistics: {snn_graph.number_of_vertices()} vertices, "
            f"{snn_graph.number_of_edges()} edges"
        )
        
        # Run Louvain community detection
        resolution = self.config.snn_clustering_params.louvain_resolution
        logger.debug(f"Running Louvain with resolution={resolution}")
        
        partitions_df, modularity = cugraph.louvain(snn_graph, resolution=resolution)
        
        # Convert to label array
        initial_labels = cp.full(reduced_vectors.shape[0], -1, dtype=cp.int32)
        initial_labels[partitions_df["vertex"].values] = partitions_df["partition"].values
        
        n_communities = int(initial_labels[initial_labels != -1].max() + 1)
        logger.info(f"Found {n_communities} initial communities (modularity={modularity:.3f})")
        
        # Stage 2: Attach noise points
        logger.info("SNN Stage 2: Attaching noise points to communities")
        # If we have HDBSCAN labels, use those as the starting point
        if hdbscan_labels is not None:
            labels_to_use = hdbscan_labels.copy()
        else:
            labels_to_use = initial_labels

        initial_noise = int((labels_to_use == -1).sum())
        
        labels_after_attachment = attach_noise_points(
            vectors_norm, 
            labels_to_use, 
            **self.config.noise_attachment_params.model_dump()
        )
        
        attached_noise = initial_noise - int((labels_after_attachment == -1).sum())
        logger.debug(f"Attached {attached_noise:,}/{initial_noise:,} noise points")
        
        # Stage 3: Merge over-split clusters
        logger.info("SNN Stage 3: Merging over-split clusters")
        
        # Build merge parameters from config
        merge_params = {
            'merge_median_threshold': self.config.merge_median_threshold,
            'merge_max_threshold': self.config.merge_max_threshold,
            'merge_sample_size': self.config.merge_sample_size,
            'centroid_similarity_threshold': self.config.centroid_similarity_threshold,
            'merge_batch_size': self.config.merge_batch_size,
            'centroid_sample_size': self.config.centroid_sample_size,
        }
        
        final_labels = merge_snn_clusters(
            vectors_norm, 
            labels_after_attachment, 
            merge_params
        )
        
        # Log final SNN statistics
        n_final_clusters = int(final_labels[final_labels != -1].max() + 1) if (final_labels != -1).any() else 0
        final_noise = int((final_labels == -1).sum())
        
        logger.info(
            f"SNN complete: {n_final_clusters} clusters, "
            f"{final_noise:,} noise points ({final_noise/len(final_labels):.1%})"
        )
        
        return final_labels

    @gpu_memory_cleanup
    def _run_umap_ensemble(
        self, 
        vectors: cp.ndarray, 
        is_training: bool
    ) -> cp.ndarray:
        """
        Perform manifold learning using UMAP ensemble with vector deduplication.
        
        This method includes deduplication to handle the strict UMAP 
        zero-distance neighbor check in newer cuML versions. The process:
        1. Deduplicate input vectors (with epsilon tolerance for floating-point)
        2. Run UMAP ensemble on unique vectors only
        3. Expand results back to original size
        4. Create consensus embedding
        
        Args:
            vectors: High-dimensional feature vectors
            is_training: Whether to fit new models
            
        Returns:
            Low-dimensional consensus embedding for all input vectors
            
        Raises:
            RuntimeError: If ensemble fails or no models available
        """
        logger.info(f"Running UMAP ensemble on shape {vectors.shape}")

        # Step 1: Deduplicate vectors to avoid zero-distance neighbor errors
        # We use a small epsilon (1e-9) to handle floating-point comparison issues
        dedup_epsilon = 1e-9
        unique_vectors, unique_indices, inverse_indices = self._deduplicate_vectors(
            vectors, 
            epsilon=dedup_epsilon
        )
        
        # Store the inverse mapping for later expansion
        # We need this to expand UMAP results back to original size
        self._current_inverse_indices = inverse_indices
        
        n_runs = int(self.config.umap_n_runs)
        umap_embeddings: List[cp.ndarray] = []
        
        if is_training:
            # Training: Fit ensemble of UMAP models
            trained_umaps: List[UMAP] = []
            rng = np.random.default_rng(self.random_state)
            
            for run_idx in range(n_runs):
                # Generate diverse parameters for this run
                umap_params = self._generate_run_parameters(run_index=run_idx, rng=rng)
                
                logger.debug(f"UMAP run {run_idx + 1}/{n_runs} with umap parameters {umap_params}")
                
                try:
                    reducer = UMAP(**umap_params)
                    
                    # Fit UMAP on deduplicated vectors only
                    # This avoids the zero-distance neighbor error
                    unique_embedding = reducer.fit_transform(unique_vectors)
                    
                    # Validate embedding
                    if cp.isnan(unique_embedding).any():
                        logger.warning(f"UMAP run {run_idx + 1} produced NaN values, skipping")
                        continue
                    
                    # Expand embedding back to original size
                    # Duplicate vectors get identical embeddings
                    expanded_embedding = self._expand_umap_embeddings(
                        unique_embedding, 
                        inverse_indices
                    )

                    umap_embeddings.append(expanded_embedding)
                    trained_umaps.append(reducer)
                    
                except Exception as e:
                    logger.error(f"UMAP run {run_idx + 1}/{n_runs} failed: {e}")
                    continue
            
            if not trained_umaps:
                raise RuntimeError("All UMAP ensemble runs failed")
            
            self.umap_ensemble = trained_umaps
            logger.info(f"Successfully fitted {len(trained_umaps)}/{n_runs} UMAP models")
            
        else:
            # Inference: Transform with existing models
            if not self.umap_ensemble:
                raise RuntimeError("No pre-trained UMAP models found")
            
            logger.info(
                f"Transforming {len(unique_vectors):,} unique vectors "
                f"with {len(self.umap_ensemble)} UMAP models"
            )
            
            for idx, reducer in enumerate(self.umap_ensemble):
                try:
                    # Transform deduplicated vectors
                    unique_embedding = reducer.transform(unique_vectors)
                    
                    if cp.isnan(unique_embedding).any():
                        logger.warning(f"UMAP model {idx + 1} produced NaN values, skipping")
                        continue
                    
                    # Expand to original size
                    expanded_embedding = self._expand_umap_embeddings(
                        unique_embedding,
                        inverse_indices
                    )
                    
                    umap_embeddings.append(expanded_embedding)
                    
                except Exception as e:
                    logger.error(f"UMAP transform {idx + 1} failed: {e}")
                    continue
        
        if not umap_embeddings:
            raise RuntimeError("UMAP ensemble produced no valid embeddings")
        
        # Combine embeddings into consensus
        logger.info(f"Creating consensus from {len(umap_embeddings)} embeddings")
        
        final_vectors = create_consensus_embedding(
            embeddings_list=umap_embeddings,
            n_anchor_samples=self.config.cosine_consensus_n_samples,
            batch_size=self.config.cosine_consensus_batch_size,
            random_state=self.random_state
        )

        # Clean up temporary deduplication mapping
        del self._current_inverse_indices

        # --- GPU Memory Cleanup ---
        # The umap_embeddings list can be very large. We delete it and
        # the decorator frees up GPU memory immediately.
        del umap_embeddings
        logger.debug("Cleaned up UMAP embedding list from GPU memory.")
        
        logger.info(f"UMAP ensemble complete: shape {final_vectors.shape}")
        return final_vectors

    # ========================================================================
    # ENSEMBLE COMBINATION
    # ========================================================================
    
    def _ensemble_cluster_labels(
        self, 
        gdf_hdbscan: cudf.DataFrame, 
        snn_labels: cp.ndarray
    ) -> cudf.DataFrame:
        """
        Combine HDBSCAN and SNN results via purity-based mapping.
        
        This method:
        1. Maps SNN clusters to HDBSCAN clusters based on overlap purity
        2. Rescues HDBSCAN noise points using mapped SNN clusters
        3. Optionally creates new clusters from large unmapped SNN groups
        
        Args:
            gdf_hdbscan: DataFrame with HDBSCAN labels
            snn_labels: Array of SNN cluster labels
            
        Returns:
            DataFrame with ensembled cluster labels
        """
        assert len(gdf_hdbscan) == len(snn_labels), f"Length mismatch: {len(gdf_hdbscan)} vs {len(snn_labels)}"

        params = self.config.ensemble_params
        logger.info("Ensembling HDBSCAN (precision) with SNN (recall)")
        
        hdb_labels = gdf_hdbscan["cluster"].values
        hdb_probs = gdf_hdbscan["cluster_probability"].values
        
        # Find mapping between SNN and HDBSCAN clusters
        snn_to_hdb_map = self._find_snn_to_hdbscan_mapping(hdb_labels, snn_labels, params)
        
        # Initialize final labels with HDBSCAN results
        final_labels = hdb_labels.copy()
        final_probs = hdb_probs.copy()
        noise_mask = final_labels == -1
        
        # Apply mapping to rescue noise points
        rescued_count = self._rescue_noise_points(
            final_labels, 
            final_probs, 
            noise_mask,
            snn_labels, 
            snn_to_hdb_map, 
            params
        )
        
        # Optionally mint new clusters
        if params.allow_new_snn_clusters:
            new_clusters = self._mint_new_clusters(
                final_labels, 
                final_probs,
                noise_mask, 
                snn_labels, 
                snn_to_hdb_map, 
                params
            )
        else:
            new_clusters = 0
        
        # Update DataFrame
        gdf_hdbscan["cluster"] = final_labels
        gdf_hdbscan["cluster_probability"] = final_probs
        
        logger.info(
            f"Ensemble complete: rescued {rescued_count:,} noise points, "
            f"created {new_clusters} new clusters"
        )
        
        return gdf_hdbscan
    
    def _find_snn_to_hdbscan_mapping(
        self, 
        hdb_labels: cp.ndarray, 
        snn_labels: cp.ndarray,
        params: EnsembleParams
    ) -> cudf.DataFrame:
        """
        Find best mapping from SNN clusters to HDBSCAN clusters.
        
        Args:
            hdb_labels: HDBSCAN cluster labels
            snn_labels: SNN cluster labels
            params: Ensemble parameters
            
        Returns:
            DataFrame with 'snn' and 'hdb' columns for valid mappings
        """
        logger.debug("Finding SNN to HDBSCAN cluster mappings")
        
        # Create overlap DataFrame
        overlap_df = cudf.DataFrame({"hdb": hdb_labels, "snn": snn_labels})
        clustered_both = overlap_df[(overlap_df["hdb"] != -1) & (overlap_df["snn"] != -1)]
        
        if clustered_both.empty:
            logger.warning("No overlap between HDBSCAN and SNN clusters")
            return cudf.DataFrame({"snn": [], "hdb": []})
        
        # Calculate overlap statistics
        overlap_counts = clustered_both.groupby(["snn", "hdb"]).size().reset_index(name="overlap")
        snn_total_sizes = overlap_counts.groupby("snn")["overlap"].sum().reset_index(name="snn_total")
        overlap_counts = overlap_counts.merge(snn_total_sizes, on="snn")
        overlap_counts["purity"] = overlap_counts["overlap"] / overlap_counts["snn_total"]
        
        # Find best match for each SNN cluster
        best_matches = (
            overlap_counts
            .sort_values(["snn", "overlap"], ascending=[True, False])
            .drop_duplicates(subset=["snn"])
        )
        
        # Filter by purity and overlap thresholds
        valid_mappings = best_matches[
            (best_matches["purity"] >= params.purity_min) &
            (best_matches["overlap"] >= params.min_overlap)
        ][["snn", "hdb"]]
        
        logger.debug(f"Found {len(valid_mappings)} valid SNN->HDBSCAN mappings")
        return valid_mappings
    
    def _rescue_noise_points(
        self,
        final_labels: cp.ndarray,
        final_probs: cp.ndarray,
        noise_mask: cp.ndarray,
        snn_labels: cp.ndarray,
        snn_to_hdb_map: cudf.DataFrame,
        params: EnsembleParams
    ) -> int:
        """
        Rescue HDBSCAN noise points using SNN clusters.
        
        Returns:
            Number of rescued points
        """
        # Map SNN labels to HDBSCAN clusters
        snn_labels_df = cudf.DataFrame({"snn": snn_labels})
        snn_labels_df = snn_labels_df.merge(snn_to_hdb_map, on="snn", how="left")
        mapped_hdb_labels = snn_labels_df["hdb"].fillna(-1).astype("int32").values
        
        # Rescue noise points that SNN assigns to mapped clusters
        rescue_mask = noise_mask & (mapped_hdb_labels != -1)
        final_labels[rescue_mask] = mapped_hdb_labels[rescue_mask]
        
        # Set confidence for rescued points
        default_conf = float(params.default_rescue_conf)
        final_probs[rescue_mask] = default_conf
        
        return int(rescue_mask.sum())
    
    def _mint_new_clusters(
        self,
        final_labels: cp.ndarray,
        final_probs: cp.ndarray,
        noise_mask: cp.ndarray,
        snn_labels: cp.ndarray,
        snn_to_hdb_map: cudf.DataFrame,
        params: EnsembleParams
    ) -> int:
        """
        Create new clusters from large unmapped SNN groups.
        
        Returns:
            Number of new clusters created
        """
        # Find SNN clusters that aren't mapped to HDBSCAN
        snn_sizes = cudf.Series(snn_labels).value_counts().reset_index()
        snn_sizes.columns = ["snn", "size"]
        
        unmapped_snn = snn_sizes.merge(snn_to_hdb_map, on="snn", how="left")
        new_cluster_candidates = unmapped_snn[
            (unmapped_snn["snn"] != -1) & 
            unmapped_snn["hdb"].isnull() &
            (unmapped_snn["size"] >= params.min_newcluster_size)
        ]
        
        if new_cluster_candidates.empty:
            return 0
        
        # Assign new cluster IDs
        next_id = int(final_labels.max()) + 1 if int(final_labels.max()) >= 0 else 0
        candidate_ids = new_cluster_candidates["snn"].values
        
        new_id_map = cudf.DataFrame({
            "snn": candidate_ids,
            "new_id": cp.arange(next_id, next_id + len(candidate_ids), dtype=cp.int32),
        })
        
        # Apply new cluster assignments
        snn_labels_df = cudf.DataFrame({"snn": snn_labels})
        snn_labels_df = snn_labels_df.merge(new_id_map, on="snn", how="left")
        new_ids = snn_labels_df["new_id"].fillna(-1).astype("int32").values
        
        assign_new_mask = noise_mask & (new_ids != -1)
        final_labels[assign_new_mask] = new_ids[assign_new_mask]
        
        # Set confidence for new clusters
        default_conf = float(params.default_rescue_conf)
        final_probs[assign_new_mask] = default_conf
        
        return len(candidate_ids)

    # ========================================================================
    # PARAMETER GENERATION
    # ========================================================================
    @staticmethod
    def _get_sub_range(
        full_range: Tuple[float, float],
        start_percent: float,
        end_percent: float,
        use_log_scale: bool = False
    ) -> Tuple[float, float]:
        """
        Extract a sub-range from a full parameter range using linear or logarithmic scaling.
        
        This utility method enables consistent sub-range selection across different UMAP
        hyperparameters. Logarithmic scaling is essential for parameters with multiplicative
        effects (like min_dist), while linear scaling suits parameters with additive effects.
        
        Args:
            full_range: The complete (min, max) range to select from.
            start_percent: Starting position in the range as a fraction (0.0 to 1.0).
            end_percent: Ending position in the range as a fraction (0.0 to 1.0).
            use_log_scale: If True, calculations are performed in log10 space, which is
                        appropriate for parameters where relative differences matter more
                        than absolute differences.
        
        Returns:
            A new (min, max) tuple representing the specified sub-range.
            
        Raises:
            ValueError: If percentages are invalid (not in [0,1] or start >= end).
        """
        if not (0.0 <= start_percent < end_percent <= 1.0):
            raise ValueError("Percentages must be between 0.0 and 1.0, with start < end.")
        
        lower_bound, upper_bound = full_range
        
        if use_log_scale:
            # Logarithmic scaling for multiplicative parameters
            # Add epsilon to prevent log(0) errors
            epsilon = 1e-9
            log_lower = np.log10(lower_bound + epsilon)
            log_upper = np.log10(upper_bound + epsilon)
            log_range_size = log_upper - log_lower
            
            new_log_lower = log_lower + (log_range_size * start_percent)
            new_log_upper = log_lower + (log_range_size * end_percent)
            
            return (10**new_log_lower, 10**new_log_upper)
        else:
            # Linear scaling for additive parameters
            range_size = upper_bound - lower_bound
            
            new_lower = lower_bound + (range_size * start_percent)
            new_upper = lower_bound + (range_size * end_percent)
            
            return (new_lower, new_upper)

    def _sample_min_dist_and_spread(
        self, 
        rng: np.random.Generator,
        min_dist_range: Tuple[float, float],
        spread_range: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Sample correlated (min_dist, spread) parameters for UMAP with constraint enforcement.
        
        These parameters control the balance between local and global structure in the embedding:
        - min_dist: Minimum distance between points in the low-dimensional embedding space.
                    Controls how tightly UMAP packs connected points together.
                    Smaller values create tighter clumps; larger values create more even distribution.
        - spread: The effective scale/dispersion of embedded points around each other.
                Controls how spread out the embedding is overall.
        
        Mathematical constraint: spread must always be > min_dist to ensure valid UMAP behavior.
        
        For entity resolution, the relationship between these parameters is critical:
        - Tight packing (small min_dist) helps group name variants together
        - But sufficient spread prevents over-compression that loses discriminative power
        
        Args:
            rng: Random number generator for reproducible sampling
            min_dist_range: Tuple of (lower_bound, upper_bound) for min_dist parameter
            spread_range: Tuple of (lower_bound, upper_bound) for spread parameter
            
        Returns:
            Tuple of (min_dist, spread) values satisfying the spread > min_dist constraint
        """
        min_dist_lower_bound, min_dist_upper_bound = min_dist_range
        spread_lower_bound, spread_upper_bound = spread_range
        
        # Use log-uniform sampling for min_dist to better explore the parameter space.
        # This is crucial because min_dist effects are multiplicative rather than additive:
        # the difference between 0.001 and 0.01 is more significant than 0.1 and 0.11.
        # We add epsilon to handle the edge case where lower bound is exactly 0.
        epsilon_for_log_safety = 1.0e-9
        log_min_dist_lower = np.log10(abs(min_dist_lower_bound) + epsilon_for_log_safety)
        log_min_dist_upper = np.log10(abs(min_dist_upper_bound) + epsilon_for_log_safety)
        
        # Sample in log space then transform back to linear space
        log_sampled_min_dist = rng.uniform(log_min_dist_lower, log_min_dist_upper)
        sampled_min_dist = 10.0 ** log_sampled_min_dist
        
        # To satisfy the mathematical constraint spread > min_dist, we must adjust
        # the valid sampling range for spread based on the sampled min_dist value.
        # We add a small buffer (1e-4) to avoid numerical edge cases where spread ≈ min_dist.
        constraint_buffer = 1.0e-4
        constrained_spread_lower = max(spread_lower_bound, sampled_min_dist + constraint_buffer)
        
        # Handle the edge case where our constraint pushes the lower bound above the upper bound.
        # This can happen when min_dist is sampled near its upper range and spread range is tight.
        if constrained_spread_lower >= spread_upper_bound:
            # In this case, we set spread to the minimum valid value
            sampled_spread = constrained_spread_lower
        else:
            # Normal case: sample spread uniformly from the valid constrained range.
            # Linear sampling is appropriate here as spread effects are more linear.
            sampled_spread = rng.uniform(constrained_spread_lower, spread_upper_bound)
        
        return float(sampled_min_dist), float(sampled_spread)

    def _generate_run_parameters(
        self, 
        run_index: int, 
        rng: np.random.Generator
    ) -> Dict[str, Any]:
        """
        Generate diverse yet correlated UMAP parameters for ensemble run with view-based coupling.
        
        This method implements a sophisticated parameter generation strategy that creates
        two distinct "modes" of UMAP embeddings:
        1. Local view: Emphasizes fine-grained structure, good for capturing entity variants
        2. Global view: Emphasizes broad structure, good for maintaining entity boundaries
        
        Parameter Correlation Strategy:
        - Core parameters (n_neighbors, min_dist, spread) define the view type
        - Supporting parameters are adjusted to reinforce the chosen view
        - Parameters with multiplicative effects use logarithmic scaling
        - Parameters with additive effects use linear scaling
        
        The first run (index 0) always uses stable base parameters as an anchor point,
        ensuring at least one consistent embedding in the ensemble.
        
        Args:
            run_index: Index of current ensemble run (0-based)
            rng: Random number generator for reproducibility
            
        Returns:
            Dictionary of UMAP parameters optimized for the chosen view type
        """
        # Start with base UMAP parameters as template
        sampling_config = self.config.umap_ensemble_sampling_config
        umap_params = self.config.umap_params.model_dump()
        
        if run_index == 0:
            # First run: Use stable base parameters as anchor for ensemble.
            # This provides a consistent reference point across different random seeds
            # and helps prevent the ensemble from being too unstable.
            logger.debug("UMAP run 1: Using base parameters as stable anchor")
            umap_params['random_state'] = self._get_run_seed(run_index)
            return umap_params
        
        # For subsequent runs, generate diverse parameters with view-based correlation
        
        # Determine view type using configured probability
        local_view_probability = sampling_config.local_view_ratio
        is_local_view = rng.random() < local_view_probability
        view_type = "LOCAL" if is_local_view else "GLOBAL"
        
        # Define the sub-range percentages for each view type
        # Local view uses lower portions of ranges for tighter embeddings
        # Global view uses upper portions for better separation
        if is_local_view:
            # Local view parameter range selections
            n_neighbors_range = sampling_config.n_neighbors_local
            
            # Use lower 60% of min_dist range (logarithmic scale for multiplicative effect)
            min_dist_range = self._get_sub_range(
                sampling_config.min_dist,
                start_percent=0.0,
                end_percent=0.6,
                use_log_scale=True
            )
            
            # Use lower 60% of spread range (linear scale for additive effect)
            spread_range = self._get_sub_range(
                sampling_config.spread,
                start_percent=0.0,
                end_percent=0.6,
                use_log_scale=False
            )
            
            # Supporting parameters: use lower halves to reinforce local structure
            repulsion_percent_range = (0.0, 0.5)
            negative_sample_percent_range = (0.0, 0.5)
            epochs_percent_range = (0.0, 0.5)
            
        else:
            # Global view parameter range selections
            n_neighbors_range = sampling_config.n_neighbors_global
            
            # Use upper 60% of min_dist range (starting at 40%)
            min_dist_range = self._get_sub_range(
                sampling_config.min_dist,
                start_percent=0.4,
                end_percent=1.0,
                use_log_scale=True
            )
            
            # Use upper 60% of spread range
            spread_range = self._get_sub_range(
                sampling_config.spread,
                start_percent=0.4,
                end_percent=1.0,
                use_log_scale=False
            )
            
            # Supporting parameters: use upper halves to reinforce global structure
            repulsion_percent_range = (0.5, 1.0)
            negative_sample_percent_range = (0.5, 1.0)
            epochs_percent_range = (0.5, 1.0)
        
        # --- Sample Core Parameters ---
        
        # n_neighbors: uniformly sampled within the view-specific range
        umap_params['n_neighbors'] = int(
            rng.integers(low=n_neighbors_range[0], high=n_neighbors_range[1], endpoint=True)
        )
        
        # min_dist and spread: sampled with constraint enforcement
        min_dist, spread = self._sample_min_dist_and_spread(rng, min_dist_range, spread_range)
        umap_params["min_dist"] = min_dist
        umap_params["spread"] = spread
        
        # --- Sample Supporting Parameters ---
        
        # Repulsion strength: linear scaling (additive force effect)
        repulsion_sub_range = self._get_sub_range(
            sampling_config.repulsion_strength,
            start_percent=repulsion_percent_range[0],
            end_percent=repulsion_percent_range[1],
            use_log_scale=False
        )
        umap_params["repulsion_strength"] = float(
            rng.uniform(repulsion_sub_range[0], repulsion_sub_range[1])
        )
        
        # Negative sample rate: logarithmic scaling (multiplicative optimization effect)
        negative_sample_sub_range = self._get_sub_range(
            sampling_config.negative_sample_rate,
            start_percent=negative_sample_percent_range[0],
            end_percent=negative_sample_percent_range[1],
            use_log_scale=True
        )
        umap_params["negative_sample_rate"] = int(
            rng.uniform(negative_sample_sub_range[0], negative_sample_sub_range[1])
        )
        
        # Number of epochs: linear scaling (additive iteration effect)
        epochs_sub_range = self._get_sub_range(
            sampling_config.n_epochs,
            start_percent=epochs_percent_range[0],
            end_percent=epochs_percent_range[1],
            use_log_scale=False
        )
        # Use integers for epoch count
        epochs_sub_range_int = (int(epochs_sub_range[0]), int(epochs_sub_range[1]))
        umap_params["n_epochs"] = int(
            rng.integers(epochs_sub_range_int[0], epochs_sub_range_int[1], endpoint=True)
        )
        
        # --- Sample Independent Parameters ---
        # These benefit from full-range random variation regardless of view type
        
        # Learning rate: controls optimization step size
        # Full range sampling for diverse convergence behavior
        learning_rate_range = sampling_config.learning_rate
        umap_params["learning_rate"] = float(
            rng.uniform(learning_rate_range[0], learning_rate_range[1])
        )
        
        # Initialization strategy: starting point for optimization
        # Random selection helps explore different local optima
        init_strategies = sampling_config.init_strategies
        umap_params["init"] = rng.choice(init_strategies).item()
        
        # Set unique but deterministic random seed for this run
        umap_params["random_state"] = self._get_run_seed(run_index)
        
        # Log the parameter choices for debugging and analysis
        logger.debug(
            f"UMAP run {run_index + 1}: {view_type} view with "
            f"n_neighbors={umap_params['n_neighbors']}, "
            f"min_dist={umap_params['min_dist']:.4f}, "
            f"spread={umap_params['spread']:.3f}, "
            f"repulsion={umap_params['repulsion_strength']:.2f}, "
            f"neg_samples={umap_params['negative_sample_rate']}, "
            f"epochs={umap_params['n_epochs']}"
        )
        
        return umap_params

    def _get_run_seed(self, run_index: int) -> int:
        """
        Generate deterministic seed for a specific run.
        
        Uses a large prime multiplier to ensure seeds are well-separated
        in the random number space.
        
        Args:
            run_index: Index of the run
            
        Returns:
            Unique seed for this run
        """
        # Use large prime to ensure good separation
        prime_multiplier = 104729
        seed = int(self.random_state + run_index * prime_multiplier) % (2**31)
        return seed

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _validate_inputs(self, gdf: cudf.DataFrame, vectors: cp.ndarray) -> None:
        """
        Validate input DataFrame and vectors.
        
        Args:
            gdf: Input DataFrame
            vectors: Feature vectors
            
        Raises:
            ValueError: If inputs are invalid
            TypeError: If inputs are wrong type
        """
        # Type validation
        if not isinstance(gdf, cudf.DataFrame):
            raise TypeError(f"gdf must be cudf.DataFrame, got {type(gdf)}")
        
        if not isinstance(vectors, cp.ndarray):
            raise TypeError(f"vectors must be cupy.ndarray, got {type(vectors)}")
        
        # Shape validation
        if len(gdf) != vectors.shape[0]:
            raise ValueError(
                f"DataFrame length ({len(gdf)}) doesn't match "
                f"vector rows ({vectors.shape[0]})"
            )
        
        if vectors.ndim != 2:
            raise ValueError(f"Vectors must be 2D, got shape {vectors.shape}")
        if vectors.shape[1] == 0:
            raise ValueError("Feature vectors have zero dimensions")
        
        if cp.isnan(vectors).any():
            n_nan = int(cp.isnan(vectors).sum())
            raise ValueError(f"Feature vectors contain {n_nan} NaN values")
        if cp.isinf(vectors).any():
            n_inf = int(cp.isinf(vectors).sum())
            raise ValueError(f"Feature vectors contain {n_inf} Inf values")
    
    def _deduplicate_vectors(
        self, 
        vectors: cp.ndarray, 
        epsilon: float = 1e-9
    ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        Deduplicate vectors with floating-point tolerance to avoid UMAP zero-distance errors.
        
        This method identifies and removes duplicate vectors (within epsilon tolerance) to prevent
        UMAP from encountering the "all neighbors at distance 0" error. It maintains a mapping
        to reconstruct the original vector indices after UMAP transformation.
        
        The deduplication process:
        1. Round vectors to a precision determined by epsilon to handle floating-point errors
        2. Use cuDF DataFrame for efficient row-wise deduplication
        3. Return deduplicated vectors and mappings for reconstruction
        
        Args:
            vectors: Input vectors of shape (n_samples, n_features)
            epsilon: Tolerance for considering vectors as duplicates. Vectors are rounded
                    to approximately -log10(epsilon) decimal places before comparison.
                    Default 1e-9 provides ~9 decimal places of precision.
        
        Returns:
            Tuple containing:
                - unique_vectors: Deduplicated vectors of shape (n_unique, n_features)
                - unique_indices: Indices of unique vectors in original array
                - inverse_indices: Mapping from original indices to unique indices,
                                 shape (n_samples,). Used to expand results back.
        
        Example:
            If input has vectors [A, B, A, C, B], returns:
            - unique_vectors: [A, B, C]
            - unique_indices: [0, 1, 3] (first occurrence of each unique vector)
            - inverse_indices: [0, 1, 0, 2, 1] (maps each original to its unique index)
        """
        n_samples, n_features = vectors.shape
        logger.debug(f"Deduplicating {n_samples:,} vectors with {n_features} features, epsilon={epsilon}")
        
        # Determine rounding precision based on epsilon
        # For epsilon=1e-9, this gives decimal_places=9
        decimal_places = int(-np.log10(epsilon))
        
        # Round vectors to handle floating-point comparison issues
        # This ensures that vectors differing by less than epsilon are treated as identical
        vectors_rounded = cp.around(vectors, decimals=decimal_places)
        
        # Method 1: Use cuDF for row-wise deduplication (most reliable for GPU)
        # Convert rounded vectors to DataFrame for efficient deduplication
        # Create column names for the DataFrame
        column_names = [f"f{i}" for i in range(n_features)]
        
        # Create DataFrame from vectors
        # We add an index column to track original positions
        vectors_df = cudf.DataFrame(vectors_rounded, columns=column_names)
        vectors_df['original_idx'] = cp.arange(n_samples, dtype=cp.int32)
        
        # Drop duplicates, keeping first occurrence
        # This gives us the unique vectors and their original indices
        unique_df = vectors_df.drop_duplicates(subset=column_names, keep='first')
        unique_df = unique_df.sort_values('original_idx')  # Maintain order
        
        # Extract unique indices (positions of first occurrence in original array)
        unique_indices = unique_df['original_idx'].values
        
        # Get the unique vectors from the original (non-rounded) array
        # This preserves full precision
        unique_vectors = vectors[unique_indices]
        
        # Create inverse mapping: for each original vector, find its unique index
        # We need to map each original vector to its corresponding unique vector index
        # First, create a mapping from unique vectors back to their new indices
        unique_df_for_merge = unique_df.copy()
        unique_df_for_merge['unique_idx'] = cp.arange(len(unique_df), dtype=cp.int32)
        
        # Merge with original DataFrame to get mapping
        merged_df = vectors_df[column_names].merge(
            unique_df_for_merge[column_names + ['unique_idx']], 
            on=column_names, 
            how='left'
        )
        inverse_indices = merged_df['unique_idx'].values
        
        n_unique = len(unique_vectors)
        reduction_ratio = 1.0 - (n_unique / n_samples)

        if logger.isEnabledFor(logging.DEBUG):
            # Float32 has ~7 decimal digits; avoid pretending we have more
            decimal_places = max(0, min(int(-np.log10(epsilon)), 7))

            # 1) Rounded uniqueness — reuse the rounded DataFrame we already built
            # `vectors_df` = DataFrame of the *rounded* vectors; `column_names` are your f0..fN
            n_unique_rounded = int(len(vectors_df.drop_duplicates(subset=column_names, keep="first")))
            
            # 2) Exact uniqueness — only build if problem size is reasonable
            #    This can be memory-heavy (201k x 1312). Guard it.
            n_unique_exact = None
            try:
                if n_samples * n_features <= 5_000_000:  # ~ heuristic to avoid huge frames
                    df_exact = cudf.DataFrame(vectors, columns=column_names)
                    n_unique_exact = int(len(df_exact.drop_duplicates(subset=column_names, keep="first")))
                else:
                    logger.debug(
                        "Skipping exact-uniqueness count due to matrix size; "
                        f"n_samples={n_samples:,}, n_features={n_features:,}."
                    )
            except Exception as diag_ex:
                logger.debug(f"Exact-uniqueness diagnostic skipped due to: {diag_ex}")
                n_unique_exact = None

            # 3) Log concise comparison
            if n_unique_exact is not None:
                safe_den = max(n_unique_exact, 1)
                extra_reduction = 1.0 - (n_unique_rounded / safe_den)
                logger.debug(
                    f"Uniqueness — exact: {n_unique_exact:,}, rounded(@{decimal_places}dp): {n_unique_rounded:,} "
                    f"(additional reduction from rounding: {extra_reduction:.1%})"
                )
            else:
                logger.debug(
                    f"Uniqueness — rounded(@{decimal_places}dp): {n_unique_rounded:,} "
                    "(exact count skipped)"
                )

            # 4) Show tiny diagnostic for merged groups after rounding
            try:
                if n_unique_exact is not None and n_unique_rounded < n_unique_exact:
                    exact_hash = cudf.hash(df_exact[column_names])
                    rounded_hash = cudf.hash(vectors_df[column_names])  # rounded already
                    key_df = cudf.DataFrame({"rounded_key": rounded_hash, "exact_key": exact_hash})
                    merged = key_df.groupby("rounded_key").agg({"exact_key": "nunique"})
                    merged = merged.rename(columns={"exact_key": "merged_exact_rows"})
                    merged = merged[merged["merged_exact_rows"] > 1].reset_index().head(5)
                    if len(merged) > 0:
                        logger.debug(
                            "Rounded groups that merged multiple exact rows (up to 5 shown):\n"
                            f"{merged.to_pandas().to_string(index=False)}"
                        )
            except Exception as diag_ex:
                logger.debug(f"Dedup diagnostics (merge preview) skipped due to: {diag_ex}")
            finally:
                # Clean up heavy temporaries if we created them
                try:
                    del df_exact
                except Exception:
                    pass
        
        logger.info(
            f"Vector deduplication: {n_samples:,} → {n_unique:,} unique vectors "
            f"({reduction_ratio:.1%} reduction)"
        )
        
        if reduction_ratio > 0.5:
            logger.info(
                f"High duplication rate detected ({reduction_ratio:.1%}). "
                f"This is common for categorical embeddings or repeated entities."
            )
        
        # Clean up temporary DataFrames to free GPU memory
        del vectors_df, unique_df, unique_df_for_merge, merged_df
        cp.get_default_memory_pool().free_all_blocks()
        
        return unique_vectors, unique_indices, inverse_indices
    
    def _expand_umap_embeddings(
        self, 
        unique_embeddings: cp.ndarray, 
        inverse_indices: cp.ndarray
    ) -> cp.ndarray:
        """
        Expand deduplicated UMAP embeddings back to original size.
        
        After UMAP processes unique vectors, this method maps the results back to all
        original vector positions, ensuring duplicate vectors get identical embeddings.
        
        Args:
            unique_embeddings: UMAP embeddings for unique vectors, shape (n_unique, n_dims)
            inverse_indices: Mapping from original to unique indices, shape (n_samples,)
        
        Returns:
            Expanded embeddings matching original vector count, shape (n_samples, n_dims)
        
        Example:
            If unique_embeddings has 3 embeddings [E1, E2, E3] and 
            inverse_indices is [0, 1, 0, 2, 1], returns [E1, E2, E1, E3, E2]
        """
        # Use fancy indexing to expand embeddings
        # Each position in inverse_indices tells us which unique embedding to use
        expanded_embeddings = unique_embeddings[inverse_indices]
        
        logger.debug(
            f"Expanded embeddings from {unique_embeddings.shape[0]:,} unique "
            f"to {expanded_embeddings.shape[0]:,} total"
        )
        
        return expanded_embeddings

    def _log_clustering_stats(self, gdf: cudf.DataFrame) -> None:
        """
        Log comprehensive clustering statistics.
        
        Args:
            gdf: Clustered DataFrame
        """
        cluster_col = gdf["cluster"]
        n_clusters = int(cluster_col[cluster_col != -1].nunique())
        n_noise = int((cluster_col == -1).sum())
        
        logger.info(f"{'='*60}")
        logger.info(f"Final clustering statistics:")
        logger.info(f"  Total records: {len(gdf):,}")
        logger.info(f"  Number of clusters: {n_clusters}")
        logger.info(f"  Noise points: {n_noise:,} ({n_noise/len(gdf):.1%})")
        
        if n_clusters > 0:
            cluster_sizes = cluster_col[cluster_col != -1].value_counts()
            logger.info(f"  Cluster size range: {cluster_sizes.min()}-{cluster_sizes.max()}")
            logger.info(f"  Mean cluster size: {cluster_sizes.mean():.1f}")
            logger.info(f"  Median cluster size: {cluster_sizes.median():.0f}")
        
        # Confidence statistics
        probs = gdf["cluster_probability"]
        logger.info(f"  Mean confidence: {probs.mean():.3f}")
        logger.info(f"  High confidence (>0.9): {(probs > 0.9).sum():,} records")
        logger.info(f"  Low confidence (<0.5): {(probs < 0.5).sum():,} records")
        logger.info(f"{'='*60}")
    
    def get_models(self) -> Dict[str, Any]:
        """
        Get trained models for persistence.
        
        Returns:
            Dictionary containing cluster_model and UMAP ensemble
        
        Raises:
            RuntimeError: If models haven't been fitted
        """
        if not self.cluster_model or not self.umap_ensemble:
            raise RuntimeError("Models not fitted. Call fit_transform() first.")
        
        return {
            "cluster_model": self.cluster_model,
            "umap_reducer_ensemble": self.umap_ensemble
        }
    
    def set_models(
        self, 
        cluster_model: HDBSCAN, 
        umap_reducer_ensemble: List[UMAP]
    ) -> None:
        """
        Load pre-trained models.
        
        Args:
            cluster_model: Fitted HDBSCAN model
            umap_reducer_ensemble: List of fitted UMAP models

        Raises:
            ValueError: If models are invalid
        """
        if not cluster_model or not umap_reducer_ensemble:
            raise ValueError("Both cluster_model and umap_ensemble required")
        
        if not isinstance(umap_reducer_ensemble, list):
            raise ValueError("umap_reducer_ensemble must be a list")
        
        self.cluster_model = cluster_model
        self.umap_ensemble = umap_reducer_ensemble
        logger.info(
            f"Loaded pre-trained models: "
            f"HDBSCAN and {len(umap_reducer_ensemble)} UMAP models"
        )