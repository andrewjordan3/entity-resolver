# entity_resolver/reporter.py
"""
This module defines the ResolutionReporter class, which is responsible for
generating human-readable reports and summary DataFrames from the results
of the entity resolution pipeline.
"""

import pandas as pd
import cudf
import logging
from typing import Dict, Any

# --- Local Package Imports ---
from .config import ResolverConfig

# Set up a logger for this module
logger = logging.getLogger(__name__)

class ResolutionReporter:
    """
    Generates reports and summary DataFrames from resolution results.

    This class is stateless; it takes the final DataFrames from the main
    resolver and transforms them into formats suitable for analysis and review.
    """
    def __init__(self, config: ResolverConfig):
        """
        Initializes the ResolutionReporter.

        Args:
            config: The main ResolverConfig object, used to access column names.
        """
        self.config = config

    def get_review_dataframe(self, resolved_gdf: cudf.DataFrame) -> pd.DataFrame:
        """
        Generates a sorted summary DataFrame for easy manual review.

        This method creates a clean, deduplicated mapping from each original
        entity to its final canonical form, sorted for readability.

        Args:
            resolved_gdf: The final, resolved GPU DataFrame from the pipeline.

        Returns:
            A pandas DataFrame with 'original_name', 'original_address',
            'canonical_name', and 'canonical_address' columns.
        """
        if resolved_gdf is None or resolved_gdf.empty:
            logger.error("Cannot generate review frame from empty or None resolved_gdf.")
            return pd.DataFrame(columns=[
                'original_name', 'original_address', 
                'canonical_name', 'canonical_address'
            ])

        logger.info("Generating review DataFrame...")

        # --- Step 1: Reconstruct the original full address string ---
        # The original address might have been split across multiple columns.
        original_address_cols = self.config.columns.address_cols
        resolved_gdf['original_address'] = resolved_gdf[original_address_cols[0]].fillna('').astype(str)
        for col in original_address_cols[1:]:
            resolved_gdf['original_address'] = resolved_gdf['original_address'] + ' ' + resolved_gdf[col].fillna('').astype(str)
        resolved_gdf['original_address'] = resolved_gdf['original_address'].str.normalize_spaces()

        # --- Step 2: Select, rename, and deduplicate the key columns ---
        original_name_col = self.config.columns.entity_col
        review_gdf = resolved_gdf[[
            original_name_col,
            'original_address',
            'canonical_name',
            'canonical_address'
        ]].rename(columns={original_name_col: 'original_name'})
        
        unique_mappings = review_gdf.drop_duplicates()

        # --- Step 3: Sort for readability ---
        # This groups all original entities under their final canonical form.
        sorted_mappings = unique_mappings.sort_values(
            by=['canonical_name', 'canonical_address', 'original_name']
        )

        logger.info(f"Found {len(sorted_mappings)} unique original -> canonical mappings.")
        return sorted_mappings.to_pandas()
    
    def generate_report(self, original_df: pd.DataFrame, resolved_gdf: cudf.DataFrame, canonical_map: cudf.DataFrame) -> Dict[str, Any]:
        """
        Generates a dictionary of detailed statistics about the resolution process.

        Args:
            original_df: The original pandas DataFrame that was input to the pipeline.
            resolved_gdf: The final, resolved GPU DataFrame.
            canonical_map: The final canonical map DataFrame.

        Returns:
            A dictionary containing detailed statistics about the resolution process.
        """
        logger.info("Generating enhanced final report...")
        
        # --- Calculate Base Statistics ---
        unique_before = original_df[self.config.columns.entity_col].nunique()
        unique_after = resolved_gdf['canonical_name'].nunique()
        
        # --- Calculate Distributional and Detailed Statistics ---
        cluster_sizes = resolved_gdf[resolved_gdf['final_cluster'] != -1].groupby('final_cluster').size()
        size_stats = cluster_sizes.describe().to_pandas().to_dict() if not cluster_sizes.empty else {}
        confidence_stats = resolved_gdf['confidence_score'].describe().to_pandas().to_dict() if 'confidence_score' in resolved_gdf.columns else {}
        
        # --- Breakdown of Review Reasons ---
        review_reasons_breakdown = {}
        if 'needs_review' in resolved_gdf.columns and resolved_gdf['needs_review'].sum() > 0:
            review_reasons = resolved_gdf[resolved_gdf['needs_review']]['review_reason'].to_pandas()
            review_reasons_breakdown = review_reasons.str.get_dummies(sep=',').sum().to_dict()

        # --- Assemble the Final Report Dictionary ---
        report_dict = {
            'summary': {
                'total_records_processed': len(resolved_gdf),
                'unique_entities_before': int(unique_before),
                'unique_entities_after': int(unique_after),
                'reduction_rate': 1 - (unique_after / max(unique_before, 1)),
            },
            'clustering_details': {
                'canonical_entities_found': len(canonical_map) if canonical_map is not None else 0,
                'unclustered_records (noise)': int((resolved_gdf['final_cluster'] == -1).sum()),
                'chain_entities_found': int(resolved_gdf['canonical_name'].str.contains(r' - \d+$').sum()),
                'enriched_addresses': int(resolved_gdf['address_was_enriched'].sum()) if 'address_was_enriched' in resolved_gdf.columns else 0,
            },
            'cluster_size_distribution': size_stats,
            'confidence_distribution': confidence_stats,
            'review_summary': {
                'total_records_for_review': int(resolved_gdf['needs_review'].sum()) if 'needs_review' in resolved_gdf.columns else 0,
                'review_reasons_breakdown': review_reasons_breakdown
            }
        }
        
        # Convert all GPU-based scalar values to standard Python numbers for clean output.
        for section, content in report_dict.items():
            if isinstance(content, dict):
                for key, val in content.items():
                    if hasattr(val, 'item'):
                        content[key] = val.item()
        
        # Log the formatted report to the console.
        self._log_report(report_dict)
                
        return report_dict

    def _log_report(self, report_dict: Dict[str, Any]) -> None:
        """Formats and logs the generated report dictionary."""
        logger.info("--- Resolution Report ---")
        for key, val in report_dict['summary'].items():
            val_str = f"{val:.2%}" if 'rate' in key else str(val)
            logger.info(f"{key.replace('_', ' ').title():<28}: {val_str}")
        
        logger.info("\n--- Clustering Details ---")
        for key, val in report_dict['clustering_details'].items():
            logger.info(f"{key.replace('_', ' ').title():<28}: {val}")

        if report_dict['cluster_size_distribution']:
            logger.info("\n--- Cluster Size Distribution ---")
            dist = report_dict['cluster_size_distribution']
            logger.info(f"{'Mean Size':<28}: {dist.get('mean', 0):.2f}")
            logger.info(f"{'Std Dev Size':<28}: {dist.get('std', 0):.2f}")
            logger.info(f"{'Min / Max Size':<28}: {int(dist.get('min', 0))} / {int(dist.get('max', 0))}")

        if report_dict['confidence_distribution']:
            logger.info("\n--- Confidence Score Distribution ---")
            dist = report_dict['confidence_distribution']
            logger.info(f"{'Mean Confidence':<28}: {dist.get('mean', 0):.3f}")
            logger.info(f"{'Std Dev Confidence':<28}: {dist.get('std', 0):.3f}")
            logger.info(f"{'Min / Max Confidence':<28}: {dist.get('min', 0):.3f} / {dist.get('max', 0):.3f}")

        if report_dict['review_summary']['total_records_for_review'] > 0:
            logger.info("\n--- Review Summary ---")
            logger.info(f"{'Total For Review':<28}: {report_dict['review_summary']['total_records_for_review']}")
            for reason, count in report_dict['review_summary']['review_reasons_breakdown'].items():
                logger.info(f"  - {reason.replace('_', ' ').title():<25}: {count}")
