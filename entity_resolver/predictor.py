# entity_resolver/predictor.py
"""
This module defines the EntityPredictor class, which is responsible for
assigning new, unseen entities to existing clusters defined by a fitted model.
"""

import cudf
import cupy
import logging
from typing import Dict, Any

# --- GPU Library Imports ---
import cuml

# Set up a logger for this module
logger = logging.getLogger(__name__)

class EntityPredictor:
    """
    Handles the prediction of cluster assignments for new entities using a
    pre-fitted clustering model and a canonical map.
    """
    def predict(
        self, 
        gdf: cudf.DataFrame, 
        vectors: cupy.ndarray, 
        cluster_model: cuml.cluster.HDBSCAN, 
        canonical_map: cudf.DataFrame
    ) -> cudf.DataFrame:
        """
        Predicts cluster membership for new records and attaches canonical data.

        This method performs two main functions:
        1.  Uses the fitted HDBSCAN model to predict which cluster each new
            entity belongs to.
        2.  Merges the pre-calculated average cluster probability from the
            canonical map to provide a baseline confidence for the prediction.

        Args:
            gdf: The pre-processed cuDF DataFrame with new entities to classify.
            vectors: The low-dimensional vectors corresponding to the new entities.
            cluster_model: The fitted HDBSCAN model from the training stage.
            canonical_map: The DataFrame containing the profiles of all known
                           canonical entities, including their average cluster probability.

        Returns:
            The input DataFrame with 'cluster' and 'cluster_probability' columns added.
        """
        if cluster_model is None or canonical_map is None:
            raise ValueError("A fitted cluster_model and canonical_map must be provided for prediction.")
        
        logger.info(f"Predicting cluster assignments for {len(gdf)} new records...")
        
        # --- Step 1: Predict Cluster Assignments ---
        # Use the pre-fitted HDBSCAN model to get the cluster label for each vector.
        # Points that don't fit well into any existing cluster will be labeled -1 (noise).
        predicted_cluster_labels = cluster_model.predict(vectors)
        gdf['cluster'] = predicted_cluster_labels
        
        # --- Step 2: Add Cluster Probabilities ---
        # Attach the average probability associated with each predicted cluster. This
        # provides a measure of how cohesive the original cluster was, which serves
        # as a good proxy for prediction confidence.
        gdf = self._add_cluster_probabilities(gdf, canonical_map)
        
        return gdf
    
    def _add_cluster_probabilities(
        self, 
        gdf: cudf.DataFrame, 
        canonical_map: cudf.DataFrame
    ) -> cudf.DataFrame:
        """
        Merges the average cluster probability from the canonical map onto the
        DataFrame of new entities.

        Args:
            gdf: The DataFrame with predicted cluster labels.
            canonical_map: The DataFrame containing canonical entity profiles.

        Returns:
            The DataFrame with an added 'cluster_probability' column.
        """
        # Ensure the canonical map has the required probability column.
        if 'avg_cluster_prob' not in canonical_map.columns:
            logger.warning(
                "'avg_cluster_prob' not found in canonical map. "
                "Assigning a default probability of 0.75 to all predictions."
            )
            # Create a default probability for each known cluster if the column is missing.
            prob_map = canonical_map[['final_cluster']].drop_duplicates()
            prob_map['avg_cluster_prob'] = 0.75
        else:
            # Create a clean mapping of cluster ID to its average probability.
            prob_map = canonical_map[['final_cluster', 'avg_cluster_prob']].drop_duplicates()
        
        # Perform a left merge to attach the probability to each record.
        gdf = gdf.merge(
            prob_map, 
            on='final_cluster', 
            how='left'
        ).rename(columns={'avg_cluster_prob': 'cluster_probability'})
        
        # For any records assigned to a new cluster (or noise), the probability will be NaN.
        # Fill these with 0.0, as there is no pre-existing confidence for these assignments.
        gdf['cluster_probability'] = gdf['cluster_probability'].fillna(0.0)
        
        return gdf
