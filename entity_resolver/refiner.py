# entity_resolver/refiner.py
"""
This module defines the ClusterRefiner class, which is responsible for the
final stages of the pipeline: refining cluster assignments, building the
canonical map, and applying it to the dataset.
"""

import cudf
import logging
from typing import Dict, Any

# --- Local Package Imports ---
from .config import ValidationConfig, OutputConfig
from . import utils

# Set up a logger for this module
logger = logging.getLogger(__name__)

class ClusterRefiner:
    """
    Handles cluster refinement, canonical map creation, and final data application.

    This class performs three key functions:
    1.  **Refines Clusters**: Enriches records with missing address data and splits
        clusters that have clear conflicts (e.g., same name in different states).
    2.  **Builds Canonical Map**: Creates the final, definitive mapping from a
        cluster ID to a single canonical entity profile (name and address).
    3.  **Applies Map**: Merges the canonical information back onto the full dataset.
    """
    def __init__(self, validation_config: ValidationConfig, output_config: OutputConfig):
        """
        Initializes the ClusterRefiner.

        Args:
            validation_config: Configuration for validation rules (e.g., splitting thresholds).
            output_config: Configuration for output formatting.
        """
        self.validation_config = validation_config
        self.output_config = output_config

    def refine_clusters(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Refines clusters through address enrichment and conflict-based splitting.
        """
        if 'addr_normalized_key' not in gdf.columns:
            gdf['final_cluster'] = gdf['cluster']
            return gdf

        logger.info("Refining clusters with address enrichment and conflict splitting...")
        # Start with the 'final_cluster' being the same as the initial cluster.
        gdf['final_cluster'] = gdf['cluster']
        
        # This new column will track which records were modified.
        gdf['address_was_enriched'] = False
        
        gdf = self._enrich_addresses(gdf)
        gdf = self._split_clusters_by_conflict(gdf)

        logger.info("Cluster refinement complete.")
        return gdf

    def build_canonical_map(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Builds the final canonical map from the refined clusters.
        """
        logger.info("Building canonical map from refined clusters...")
        
        # --- Step 1: Get the best profile for each final cluster ---
        clustered_gdf = gdf[gdf['final_cluster'] != -1]
        if clustered_gdf.empty:
            logger.warning("No valid clusters found to build canonical map.")
            return cudf.DataFrame()
            
        unique_clusters = clustered_gdf['final_cluster'].unique().to_pandas()
        
        canonical_profiles = []
        for cid in unique_clusters:
            cluster_subset = clustered_gdf[clustered_gdf['final_cluster'] == cid]
            
            canonical_name = utils.get_canonical_name_gpu(cluster_subset['normalized_text'])
            best_addr_row = utils.get_best_address_gpu(cluster_subset)
            avg_prob = cluster_subset['cluster_probability'].mean()
            
            if not best_addr_row.empty:
                profile = best_addr_row.to_pandas().iloc[0].to_dict()
                profile.update({
                    'final_cluster': cid,
                    'canonical_name': canonical_name,
                    'avg_cluster_prob': float(avg_prob)
                })
                canonical_profiles.append(profile)

        if not canonical_profiles:
            return cudf.DataFrame()

        canonical_map_gdf = cudf.DataFrame(canonical_profiles)
        
        # --- Step 2: Handle "Chain" Entities ---
        # A chain is when one entity name (e.g., "Starbucks") exists at multiple
        # distinct addresses. We need to give each a unique canonical name.
        canonical_map_gdf = self._number_chain_entities(canonical_map_gdf)

        logger.info(f"Canonical map with {len(canonical_map_gdf)} entries built successfully.")
        return canonical_map_gdf

    def apply_canonical_map(self, gdf: cudf.DataFrame, canonical_map: cudf.DataFrame) -> cudf.DataFrame:
        """
        Applies the final canonical map to the full dataset.
        """
        logger.info("Applying canonical map to produce final results...")
        
        if canonical_map is None or canonical_map.empty:
            logger.warning("Canonical map is empty. Assigning self-references.")
            gdf['canonical_name'] = gdf['normalized_text']
            gdf['canonical_address'] = gdf['addr_normalized_key']
            return gdf

        # Select only the necessary columns for the merge.
        map_to_merge = canonical_map[['final_cluster', 'canonical_name', 'addr_normalized_key']].rename(
            columns={'addr_normalized_key': 'canonical_address'}
        )

        # Merge the canonical information onto the main DataFrame.
        gdf_result = gdf.merge(map_to_merge, on='final_cluster', how='left')
        
        # For unclustered records (noise), fill in their own info as the canonical form.
        gdf_result['canonical_name'] = gdf_result['canonical_name'].fillna(gdf_result['normalized_text'])
        gdf_result['canonical_address'] = gdf_result['canonical_address'].fillna(gdf_result['addr_normalized_key'])
        
        # Apply final case formatting based on the output configuration.
        if self.output_config.output_format == 'proper':
            gdf_result['canonical_name'] = gdf_result['canonical_name'].str.title()
            
        return gdf_result

    # === Private Helper Methods ===

    def _enrich_addresses(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """Fills in missing street-level info for records in a cluster."""
        logger.debug("Performing address enrichment...")
        # Get the canonical address profile for each cluster.
        unique_clusters = gdf[gdf['cluster'] != -1]['cluster'].unique().to_pandas()
        canonical_profiles = []
        for cid in unique_clusters:
            cluster_subset = gdf[gdf['cluster'] == cid]
            best_addr_row = utils.get_best_address_gpu(cluster_subset)
            if not best_addr_row.empty:
                profile = best_addr_row.iloc[0:1]
                profile['cluster'] = cid
                canonical_profiles.append(profile)

        if not canonical_profiles:
            return gdf

        canonical_gdf = cudf.concat(canonical_profiles).rename(columns={
            'addr_street_number': 'c_street_number', 'addr_street_name': 'c_street_name',
            'addr_city': 'c_city', 'addr_state': 'c_state', 'addr_zip': 'c_zip'
        })
        
        gdf = gdf.merge(canonical_gdf[['cluster', 'c_street_number', 'c_street_name', 'c_city', 'c_state', 'c_zip']], on='cluster', how='left')
        
        # An address can be enriched if it's in a cluster, is missing street info,
        # but matches the city, state, and zip of the cluster's canonical address.
        enrich_mask = (
            (gdf['cluster'] != -1) &
            (gdf['addr_street_name'].isna() | (gdf['addr_street_name'] == '')) &
            (gdf['addr_city'] == gdf['c_city']) &
            (gdf['addr_state'] == gdf['c_state']) &
            (gdf['addr_zip'] == gdf['c_zip'])
        )
        
        enriched_count = int(enrich_mask.sum())
        if enriched_count > 0:
            logger.info(f"Enriching {enriched_count} records with missing street info.")
            gdf.loc[enrich_mask, 'addr_street_number'] = gdf.loc[enrich_mask, 'c_street_number']
            gdf.loc[enrich_mask, 'addr_street_name'] = gdf.loc[enrich_mask, 'c_street_name']
            gdf.loc[enrich_mask, 'address_was_enriched'] = True
            # Rebuild the normalized key for any records that were enriched.
            gdf.loc[enrich_mask, 'addr_normalized_key'] = utils.create_address_key_gpu(gdf.loc[enrich_mask])

        return gdf.drop(columns=['c_street_number', 'c_street_name', 'c_city', 'c_state', 'c_zip'])

    def _split_clusters_by_conflict(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """Splits clusters that contain clear geographical or address range conflicts."""
        logger.debug("Splitting clusters based on address conflicts...")
        max_cluster_id = gdf['final_cluster'].max()
        
        # Split clusters that span multiple states.
        if self.validation_config.enforce_state_boundaries:
            state_groups = gdf[gdf['final_cluster'] != -1].groupby('final_cluster')['addr_state'].nunique()
            clusters_to_split = state_groups[state_groups > 1].index
            if not clusters_to_split.empty:
                logger.info(f"Splitting {len(clusters_to_split)} clusters with multiple states.")
                split_gdf = gdf[gdf['final_cluster'].isin(clusters_to_split)]
                new_ids = split_gdf.groupby(['final_cluster', 'addr_state']).ngroup()
                gdf.loc[split_gdf.index, 'final_cluster'] = max_cluster_id + 1 + new_ids.values
                max_cluster_id = gdf['final_cluster'].max()

        # Split clusters with wide street number ranges.
        gdf['addr_street_number_numeric'] = cudf.to_numeric(gdf['addr_street_number'], errors='coerce')
        groups = gdf[gdf['final_cluster'] != -1].groupby(['final_cluster', 'addr_street_name', 'addr_zip'])
        ranges = groups['addr_street_number_numeric'].agg(['min', 'max', 'nunique'])
        
        conflicts = ranges[
            (ranges['nunique'] > 1) &
            ((ranges['max'] - ranges['min']) > self.validation_config.street_number_threshold)
        ]
        
        if not conflicts.empty:
            logger.info(f"Splitting {len(conflicts)} sub-groups with wide street number ranges.")
            gdf = gdf.merge(conflicts.reset_index()[['final_cluster', 'addr_street_name', 'addr_zip']].assign(is_conflict=True),
                            on=['final_cluster', 'addr_street_name', 'addr_zip'], how='left')
            gdf['is_conflict'] = gdf['is_conflict'].fillna(False)

            conflict_gdf = gdf[gdf['is_conflict']]
            new_ids = conflict_gdf.groupby(['final_cluster', 'addr_street_name', 'addr_zip', 'addr_street_number']).ngroup()
            gdf.loc[conflict_gdf.index, 'final_cluster'] = max_cluster_id + 1 + new_ids.values
            gdf = gdf.drop(columns=['is_conflict'])
        
        return gdf.drop(columns=['addr_street_number_numeric'], errors='ignore')

    def _number_chain_entities(self, canonical_map_gdf: cudf.DataFrame) -> cudf.DataFrame:
        """Finds and numbers entities with the same name at different addresses."""
        logger.info("Identifying and numbering chain entities...")
        
        # Count how many unique addresses each canonical name is associated with.
        name_to_address_counts = canonical_map_gdf.groupby('canonical_name')['canonical_address'].nunique()
        
        # Identify names that appear at more than one address.
        chain_names = name_to_address_counts[name_to_address_counts > 1].index
        
        if chain_names.empty:
            logger.info("No chain entities found.")
            return canonical_map_gdf

        logger.info(f"Found {len(chain_names)} entity names with multiple addresses.")
        
        chains_gdf = canonical_map_gdf[canonical_map_gdf['canonical_name'].isin(chain_names)].copy()
        non_chains_gdf = canonical_map_gdf[~canonical_map_gdf['canonical_name'].isin(chain_names)]
        
        # Sort for deterministic numbering.
        chains_gdf = chains_gdf.sort_values(['canonical_name', 'canonical_address'])
        
        # Assign a unique number to each address within a name group.
        chains_gdf['chain_num'] = chains_gdf.groupby('canonical_name').cumcount() + 1
        
        # Append the number to the canonical name.
        chains_gdf['canonical_name'] = chains_gdf['canonical_name'] + " - " + chains_gdf['chain_num'].astype(str)
        
        # Recombine with the non-chain entities.
        return cudf.concat([non_chains_gdf, chains_gdf.drop(columns=['chain_num'])])
