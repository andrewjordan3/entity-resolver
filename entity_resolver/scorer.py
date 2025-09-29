# entity_resolver/scorer.py
"""
This module defines the ConfidenceScorer class, responsible for calculating
a nuanced confidence score for each entity match and flagging records for review.
"""

import cudf
import cupy
import logging

# --- Local Package Imports ---
from .config import ConfidenceScoringConfig, OutputConfig, VectorizerConfig, ColumnConfig
from .utils import calculate_similarity_gpu

# Set up a logger for this module
logger = logging.getLogger(__name__)

class ConfidenceScorer:
    """
    Calculates confidence scores and flags records for manual review.

    This class uses a weighted model that considers multiple factors to produce
    a score representing the quality of each entity match.
    """
    def __init__(
        self, 
        scoring_config: ConfidenceScoringConfig, 
        output_config: OutputConfig,
        vectorizer_config: VectorizerConfig,
        column_config: ColumnConfig
    ):
        """
        Initializes the ConfidenceScorer.

        Args:
            scoring_config: Configuration for confidence score weights.
            output_config: Configuration for review thresholds.
            vectorizer_config: Configuration for vectorization, needed for similarity params.
            column_config: Configuration for input column names.
        """
        self.scoring_config = scoring_config
        self.output_config = output_config
        self.vectorizer_config = vectorizer_config
        self.column_config = column_config

    def score_and_flag(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        The main public method to run the full scoring and flagging pipeline.

        Args:
            gdf: The cuDF DataFrame after the canonical map has been applied.

        Returns:
            The DataFrame with added confidence and review flag columns.
        """
        gdf = self._score_confidence(gdf)
        gdf = self._flag_for_review(gdf)
        return gdf

    def _score_confidence(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """Calculates comprehensive, nuanced confidence scores on the GPU."""
        logger.info("Scoring confidence of assignments with detailed metrics...")
        
        # --- Step 1: Calculate individual similarity components ---
        # These form the base of the confidence score.
        gdf['name_similarity'] = calculate_similarity_gpu(
            gdf['normalized_text'], 
            gdf['canonical_name'].str.lower(),
            self.vectorizer_config.similarity_tfidf
        )
        gdf['address_confidence'] = calculate_similarity_gpu(
            gdf['addr_normalized_key'], 
            gdf['canonical_address'],
            self.vectorizer_config.similarity_tfidf
        )
        
        # --- Step 2: Calculate cluster-level metrics ---
        # These metrics measure the quality and cohesion of the clusters themselves.
        valid_clusters_gdf = gdf[gdf['final_cluster'] >= 0]
        if not valid_clusters_gdf.empty:
            cluster_metrics = valid_clusters_gdf.groupby('final_cluster').agg(
                cluster_size=('normalized_text', 'count'),
                avg_cluster_prob=('cluster_probability', 'mean'),
                name_variation=('name_similarity', 'std')
            ).fillna(0)
            # Cohesion is high if the name similarity of members to their canonical name is consistent (low std dev).
            cluster_metrics['cohesion_score'] = (1 - cluster_metrics['name_variation']).clip(0, 1)
            gdf = gdf.merge(cluster_metrics, on='final_cluster', how='left')
        else: # Handle case with no valid clusters.
            gdf['cluster_size'] = 1
            gdf['cohesion_score'] = 1.0

        # Fill metrics for noise points (which were not in a cluster).
        gdf['cluster_size'] = gdf['cluster_size'].fillna(1)
        gdf['cohesion_score'] = gdf['cohesion_score'].fillna(1.0)
        
        # --- Step 3: Calculate the final weighted score ---
        weights = self.scoring_config.weights
        
        # Calculate a cluster size factor using a logarithmic scale. This gives diminishing
        # returns, so a cluster of 100 is not considered 10x better than a cluster of 10.
        log_size = cupy.log1p(gdf['cluster_size'].values) / cupy.log1p(10)
        cluster_size_factor = cudf.Series(log_size, index=gdf.index).clip(upper=1.0)
        
        # Combine all weighted components into a single base score.
        base_score = (
            gdf['cluster_probability'].fillna(0) * weights.cluster_probability +
            gdf['name_similarity'] * weights.name_similarity +
            gdf['address_confidence'] * weights.address_confidence +
            gdf['cohesion_score'] * weights.cohesion_score +
            cluster_size_factor * weights.cluster_size_factor
        )
        
        # --- Step 4: Apply penalties for specific conditions ---
        # Calculate how much the original name changed to become the canonical name.
        change_magnitude = 1 - calculate_similarity_gpu(
            gdf[self.column_config.entity_col], gdf['canonical_name'], self.vectorizer_config.similarity_tfidf
        )
        # Apply a small penalty for significant name changes.
        base_score = base_score.where(~(change_magnitude > 0.5), base_score * 0.9)
        
        # Apply a penalty if the address was enriched, as this indicates initially sparse data.
        if 'address_was_enriched' in gdf.columns:
            base_score = base_score.where(~gdf['address_was_enriched'], base_score * 0.95)
            
        # Apply a larger penalty for small clusters that also had large name changes.
        small_cluster_penalty_mask = (gdf['cluster_size'] <= 2) & (change_magnitude > 0.7)
        base_score = base_score.where(~small_cluster_penalty_mask, base_score * 0.85)

        # --- Step 5: Finalize and categorize the score ---
        # For unclustered noise points, the score is based only on name similarity.
        unclustered_mask = gdf['final_cluster'] == -1
        final_score = base_score.where(~unclustered_mask, gdf['name_similarity'] * 0.5)
        
        gdf['confidence_score'] = final_score.clip(0, 1)
        
        # Assign a human-readable category based on the final score.
        bins = [0, 0.5, 0.7, 0.85, 1.0]
        labels = ['Low', 'Medium', 'High', 'Very High']
        gdf['confidence_category'] = cudf.cut(gdf['confidence_score'], bins=bins, labels=labels, include_lowest=True)
        
        logger.info(f"Confidence scoring complete. Average score: {gdf['confidence_score'].mean():.3f}")
        return gdf
    
    def _flag_for_review(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """Flags records that may require manual review based on a set of rules."""
        logger.info("Flagging records for manual review...")
        
        if 'confidence_score' not in gdf.columns:
            raise ValueError("Must run scoring before flagging. Call score_and_flag().")

        # Calculate the magnitude of change between the original and canonical names.
        change_magnitude = 1 - calculate_similarity_gpu(
            gdf[self.column_config.entity_col], gdf['canonical_name'], self.vectorizer_config.similarity_tfidf
        )
        
        # Define all conditions that would trigger a review flag.
        review_conditions = {
            'low_confidence': gdf['confidence_score'] < self.output_config.review_confidence_threshold,
            'drastic_name_change': change_magnitude > 0.7,
            'singleton_name_change': (gdf['cluster_size'] == 1) & (change_magnitude > 0.01),
        }
        if 'address_was_enriched' in gdf.columns:
            review_conditions['enriched_low_confidence'] = gdf['address_was_enriched'] & (gdf['confidence_score'] < 0.8)

        # Combine all conditions into a single boolean mask.
        needs_review_mask = cudf.Series(False, index=gdf.index)
        for condition_mask in review_conditions.values():
            needs_review_mask |= condition_mask
            
        gdf['needs_review'] = needs_review_mask
        
        # Build a comma-separated string of reasons for the review flag.
        gdf['review_reason'] = ''
        for reason, mask in review_conditions.items():
            gdf['review_reason'] = gdf['review_reason'].where(~mask, gdf['review_reason'] + ',' + reason)
            
        gdf['review_reason'] = gdf['review_reason'].str.lstrip(',')

        review_count = int(gdf['needs_review'].sum())
        logger.info(f"{review_count} records flagged for review.")
        return gdf
