# entity_resolver/utils/validation.py
"""
This module provides GPU-accelerated utilities for data validation and
consistency checking, crucial for ensuring the integrity of the final
resolved entities.
"""

import logging

import cudf

from ..config import ValidationConfig

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
    entity_keys = gdf['normalized_text'] + '|||' + gdf['addr_normalized_key'].fillna('')

    # Create a temporary DataFrame for the check.
    key_cluster_df = cudf.DataFrame({'entity_key': entity_keys, 'cluster': gdf[cluster_col]})

    # Group by the unique entity key and count the number of distinct cluster IDs.
    unique_clusters_per_key = key_cluster_df.groupby('entity_key')['cluster'].nunique()

    # Any key with a count greater than 1 is a duplicate.
    return unique_clusters_per_key[unique_clusters_per_key > 1]


def validate_no_duplicates(
    gdf: cudf.DataFrame,
    cluster_col: str = 'final_cluster',
    context: str | None = None,
) -> bool:
    """
    Validates that no identical name+address combination exists in different clusters.

    Args:
        gdf: The cuDF DataFrame to validate.
        cluster_col: The name of the cluster column ('cluster' or 'final_cluster').
        context: Optional label for the pipeline stage (e.g. 'merging', 'refining')
                 used only for clearer log messages.

    Returns:
        True if validation passes, False otherwise.
    """
    phase = f' during {context}' if context else ''
    logger.info(f"Validating for cross-cluster duplicates in column '{cluster_col}'{phase}...")
    if cluster_col not in gdf.columns:
        logger.error(f"Validation FAILED{phase}: Column '{cluster_col}' not found in DataFrame.")
        return False

    # We only need to check records that have been assigned to a cluster.
    clustered_gdf = gdf[gdf[cluster_col] != -1].copy()
    if clustered_gdf.empty:
        logger.info(f'No clustered records to validate{phase}. Validation PASSED.')
        return True

    duplicates = _find_cross_cluster_duplicates(clustered_gdf, cluster_col)

    if not duplicates.empty:
        logger.error(
            f'❌ VALIDATION FAILED{phase}: {len(duplicates)} entities appear in multiple clusters!'
        )
        # Log the first few examples for quick debugging.
        for entity_key, cluster_count in duplicates.head(5).to_pandas().items():
            name, addr = entity_key.split('|||', 1)
            logger.error(f"  '{name}' at '{addr}' appears in {cluster_count} different clusters.")
        return False

    logger.info(f'✅ Validation PASSED{phase}: No cross-cluster duplicates found.')
    return True


def _log_canonical_consistency_errors(
    clustered_gdf: cudf.DataFrame, inconsistent_names: cudf.Series
):
    """Logs detailed diagnostic information for inconsistent canonical mappings."""
    logger.error(
        f'❌ CRITICAL: Canonical consistency FAILED! {len(inconsistent_names)} names are linked to multiple addresses!'
    )

    # Log detailed examples of the first 5 inconsistent names.
    for name, count in inconsistent_names.head(5).to_pandas().items():
        problem_records = clustered_gdf[clustered_gdf['canonical_name'] == name][
            ['canonical_address', 'final_cluster', 'normalized_text']
        ].drop_duplicates()

        logger.error(f"\n  '{name}' appears with {count} different addresses:")
        for _, row in problem_records.to_pandas().iterrows():
            logger.error(f"    - Address: '{row['canonical_address']}'")
            logger.error(
                f"      Cluster: {row['final_cluster']}, Original Text: '{row['normalized_text']}'"
            )

        # Specifically check for the problematic case of mixed null/non-null addresses.
        addr_values = clustered_gdf[clustered_gdf['canonical_name'] == name]['canonical_address']
        if addr_values.isnull().any() and addr_values.notna().any():
            null_count = addr_values.isnull().sum()
            total_count = len(addr_values)
            logger.error(
                f'  WARNING: This name has both populated and missing addresses ({null_count}/{total_count} missing).'
            )


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
    logger.info('Performing final validation of canonical name/address consistency...')
    required_cols = ['canonical_name', 'canonical_address', 'final_cluster']
    if not all(col in final_gdf.columns for col in required_cols):
        logger.warning('Canonical columns not found, skipping canonical consistency check.')
        return True

    # We only check entities that were successfully clustered.
    clustered_gdf = final_gdf[final_gdf['final_cluster'] != -1]

    # For each canonical_name, count how many unique canonical_addresses it has.
    # CRITICAL: `dropna=False` is essential to catch cases where a name is linked
    # to both a real address and a null/missing address.
    addresses_per_name = clustered_gdf.groupby('canonical_name')['canonical_address'].nunique(
        dropna=False
    )

    # In a consistent output, this number should always be 1 for every name.
    inconsistent_names = addresses_per_name[addresses_per_name > 1]

    if not inconsistent_names.empty:
        _log_canonical_consistency_errors(clustered_gdf, inconsistent_names)
        return False

    logger.info('✅ Final validation PASSED: Canonical names and addresses are consistent.')
    return True


def check_state_compatibility(
    entity_states: cudf.Series, cluster_states: cudf.Series, config: ValidationConfig
) -> cudf.Series:
    """
    Checks if two state series are compatible using a vectorized GPU approach.

    Return a boolean Series (aligned to `entity_states.index`) indicating whether
    each (entity_state, cluster_state) pair is considered compatible.

    Compatibility definition:
      • True if states are exactly equal.
      • True if either state is missing/null (treat missing as compatible).
      • True if the pair appears in `config.allow_neighboring_states` (either order).
      • Otherwise False.

    Notes/guarantees:
      • Output index == input indices; no reordering.
      • Neighbor checks are applied only if `config.enforce_state_boundaries` is True.
      • Pairs with any null are treated as already compatible and are not sent to neighbor checks
    """
    # 1. Base case: states are compatible if they are identical or one is null.
    states_match = (entity_states == cluster_states) | entity_states.isna() | cluster_states.isna()

    # 2. If all pairs are already compatible, we are done.
    if states_match.all() or not config.enforce_state_boundaries:
        return states_match

    # 3. Handle neighboring states for the remaining mismatches.
    # Isolate pairs that are not yet considered a match.
    # Get mismatched indices
    mismatched_indices = states_match[~states_match].index

    # Create DataFrame BUT PRESERVE THE INDEX
    mismatched_df = cudf.DataFrame(
        {'s1': entity_states.loc[mismatched_indices], 's2': cluster_states.loc[mismatched_indices]},
        index=mismatched_indices,
    )  # KEEP THE ORIGINAL INDEX

    # Now track which indices have non-null values before dropna
    non_null_mask = ~(mismatched_df['s1'].isna() | mismatched_df['s2'].isna())

    # Only process non-null pairs
    mismatched_df_clean = (
        mismatched_df[non_null_mask].reset_index().rename(columns={'index': 'original_index'})
    )

    if mismatched_df_clean.empty:
        return states_match

    logger.debug(f'Checking {len(mismatched_df)} mismatched state pairs against neighbors config.')

    # Create a DataFrame of allowed neighbor pairs for efficient joining.
    allowed_pairs_list = config.allow_neighboring_states
    allowed_df = cudf.DataFrame(allowed_pairs_list, columns=['p1', 'p2'])

    # Check for matches in both directions, e.g., (IL, WI) and (WI, IL).
    # Merge 1: (s1, s2) -> (p1, p2)
    merged1 = mismatched_df_clean.merge(
        allowed_df, left_on=['s1', 's2'], right_on=['p1', 'p2'], how='inner'
    )
    # Merge 2: (s1, s2) -> (p2, p1)
    merged2 = mismatched_df_clean.merge(
        allowed_df, left_on=['s1', 's2'], right_on=['p2', 'p1'], how='inner'
    )

    # Use the preserved 'original_index' column ---
    # Instead of using the incorrect .index attribute of the merged DataFrames,
    # we now use the values from the column that faithfully tracked the original index.
    allowed_indices_from_merge1 = merged1['original_index']
    allowed_indices_from_merge2 = merged2['original_index']

    # Combine indices of all pairs found in the allowed list.
    # The index of the merged result corresponds to the index of mismatched_df,
    # which in turn corresponds to the original index in the states_match series.
    allowed_indices = cudf.concat(
        [allowed_indices_from_merge1, allowed_indices_from_merge2]
    ).unique()

    # Update the states_match series for the newly validated neighbors.
    if not allowed_indices.empty:
        allowed_idx = allowed_indices.astype(states_match.index.dtype).values
        states_match.loc[allowed_idx] = True

    return states_match


def check_street_number_compatibility(
    source_numbers: cudf.Series, target_numbers: cudf.Series, threshold: int
) -> cudf.Series:
    """
    Return a boolean Series (aligned to `source_numbers.index`) indicating whether
    each pair of street numbers is considered compatible.

    Compatibility rule (inclusive threshold):
      • True if |source - target| ≤ `threshold`.
      • True if either value is missing or non-numeric (treat unknown as compatible).
      • False otherwise.

    Inputs:
      • `source_numbers`, `target_numbers` may be strings or numerics; messy values
        like '123A' or '' are coerced to null.
      • `threshold` is an integer tolerance in house-number units. The comparison is inclusive.

    Notes:
      • Output index == input indices; no reordering.
      • This function is vectorized and GPU-friendly (cuDF/CuPy under the hood).

    Returns:
      cudf.Series[bool] of the same length as inputs.
    """
    # Convert series to numeric, coercing any non-numeric values (e.g., '123-A') to NaT.
    # This is a robust way to handle potentially messy street number data.
    source_numeric = cudf.to_numeric(source_numbers, errors='coerce')
    target_numeric = cudf.to_numeric(target_numbers, errors='coerce')

    # Check for nulls. If either number is null, we consider them compatible
    # as we lack sufficient information to rule out a match.
    is_either_null = source_numeric.isna() | target_numeric.isna()

    # Calculate the absolute difference for the non-null pairs.
    # We fillna with a value outside the threshold to ensure they evaluate to False.
    diff = (source_numeric - target_numeric).abs().fillna(threshold + 1)

    # A pair is compatible if the difference is within the threshold OR if one was null.
    return (diff <= threshold) | is_either_null
