# entity_resolver/utils/validation.py
"""
This module provides GPU-accelerated utilities for data validation and
consistency checking, crucial for ensuring the integrity of the final
resolved entities.
"""

import cudf
import logging

# Set up a logger for this module.
logger = logging.getLogger(__name__)


def _find_cross_cluster_duplicates(gdf: cudf.DataFrame, cluster_col: str) -> cudf.Series:
    """
    Identifies entities that incorrectly appear in multiple clusters.

    This helper creates a composite key from name and address, then groups by
    this key to find any instances that have been assigned to more than one
    cluster ID.

    Args:
        gdf: The DataFrame containing clustered records.
        cluster_col: The name of the cluster column to check.

    Returns:
        A cuDF Series containing the entity keys that appear in multiple
        clusters, with their corresponding cluster counts.
    """
    # Create a composite key to uniquely identify an entity by its name and address.
    # The '|||' separator is used to reliably split the key back apart for logging.
    entity_keys = (
        gdf['normalized_text'] + '|||' +
        gdf['addr_normalized_key'].fillna('')
    )

    # Create a temporary DataFrame for the check.
    key_cluster_df = cudf.DataFrame({
        'entity_key': entity_keys,
        'cluster': gdf[cluster_col]
    })

    # Group by the unique entity key and count the number of distinct cluster IDs.
    unique_clusters_per_key = key_cluster_df.groupby('entity_key')['cluster'].nunique()

    # Any key with a count greater than 1 is a duplicate.
    return unique_clusters_per_key[unique_clusters_per_key > 1]


def validate_no_duplicates(gdf: cudf.DataFrame, cluster_col: str = 'final_cluster') -> bool:
    """
    Validates that no identical name+address combination exists in different clusters.

    Args:
        gdf: The cuDF DataFrame to validate.
        cluster_col: The name of the cluster column ('cluster' or 'final_cluster').

    Returns:
        True if validation passes, False otherwise.
    """
    logger.info(f"Validating for cross-cluster duplicates in column '{cluster_col}'...")
    if cluster_col not in gdf.columns:
        logger.error(f"Validation FAILED: Column '{cluster_col}' not found in DataFrame.")
        return False

    # We only need to check records that have been assigned to a cluster.
    clustered_gdf = gdf[gdf[cluster_col] != -1].copy()
    if clustered_gdf.empty:
        logger.info("No clustered records to validate. Validation PASSED.")
        return True

    duplicates = _find_cross_cluster_duplicates(clustered_gdf, cluster_col)

    if not duplicates.empty:
        logger.error(f"❌ VALIDATION FAILED: {len(duplicates)} entities appear in multiple clusters!")
        # Log the first few examples for quick debugging.
        for entity_key, cluster_count in duplicates.head(5).to_pandas().items():
            name, addr = entity_key.split('|||', 1)
            logger.error(f"  '{name}' at '{addr}' appears in {cluster_count} different clusters.")
        return False

    logger.info("✅ Validation PASSED: No cross-cluster duplicates found.")
    return True


def _log_canonical_consistency_errors(
    clustered_gdf: cudf.DataFrame,
    inconsistent_names: cudf.Series
):
    """Logs detailed diagnostic information for inconsistent canonical mappings."""
    logger.error(f"❌ CRITICAL: Canonical consistency FAILED! {len(inconsistent_names)} names are linked to multiple addresses!")

    # Log detailed examples of the first 5 inconsistent names.
    for name, count in inconsistent_names.head(5).to_pandas().items():
        problem_records = clustered_gdf[clustered_gdf['canonical_name'] == name][
            ['canonical_address', 'final_cluster', 'normalized_text']
        ].drop_duplicates()

        logger.error(f"\n  '{name}' appears with {count} different addresses:")
        for _, row in problem_records.to_pandas().iterrows():
            logger.error(f"    - Address: '{row['canonical_address']}'")
            logger.error(f"      Cluster: {row['final_cluster']}, Original Text: '{row['normalized_text']}'")

        # Specifically check for the problematic case of mixed null/non-null addresses.
        addr_values = clustered_gdf[clustered_gdf['canonical_name'] == name]['canonical_address']
        if addr_values.isnull().any() and addr_values.notna().any():
            null_count = addr_values.isnull().sum()
            total_count = len(addr_values)
            logger.error(f"  WARNING: This name has both populated and missing addresses ({null_count}/{total_count} missing).")


def validate_canonical_consistency(final_gdf: cudf.DataFrame) -> bool:
    """
    Ensures that for any given canonical name, all records share the same canonical address.

    This is a critical final validation check. A failure indicates a serious
    flaw in the upstream clustering or canonicalization logic.

    Args:
        final_gdf: The final DataFrame after canonical values have been applied.

    Returns:
        True if the output is consistent, False otherwise.
    """
    logger.info("Performing final validation of canonical name/address consistency...")
    required_cols = ['canonical_name', 'canonical_address', 'final_cluster']
    if not all(col in final_gdf.columns for col in required_cols):
        logger.warning("Canonical columns not found, skipping canonical consistency check.")
        return True

    # We only check entities that were successfully clustered.
    clustered_gdf = final_gdf[final_gdf['final_cluster'] != -1]

    # For each canonical_name, count how many unique canonical_addresses it has.
    # CRITICAL: `dropna=False` is essential to catch cases where a name is linked
    # to both a real address and a null/missing address.
    addresses_per_name = clustered_gdf.groupby('canonical_name')['canonical_address'].nunique(dropna=False)

    # In a consistent output, this number should always be 1 for every name.
    inconsistent_names = addresses_per_name[addresses_per_name > 1]

    if not inconsistent_names.empty:
        _log_canonical_consistency_errors(clustered_gdf, inconsistent_names)
        return False

    logger.info("✅ Final validation PASSED: Canonical names and addresses are consistent.")
    return True
