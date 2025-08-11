# entity_resolver/utils/address.py
"""
This module provides utilities for parsing, scoring, and manipulating physical
addresses. It includes CPU-bound parsing using the `libpostal` library and
GPU-accelerated functions for normalization and scoring using `cudf`.
"""

import re
import cudf
import logging
from typing import Dict

# Third-party libraries for address parsing
from postal.parser import parse_address
from postal.expand import expand_address

# Set up a logger for this module to provide visibility into its operations.
# This is preferable to using print() for logging warnings or errors.
logger = logging.getLogger(__name__)

# --- Module-level Constants ---

# Define the standard address component columns used throughout the GPU functions.
# Using a constant ensures consistency and makes it easier to update the schema
# in one place, reducing the risk of typos or mismatches between functions.
ADDRESS_COMPONENT_COLUMNS = [
    'addr_street_number',
    'addr_street_name',
    'addr_city',
    'addr_state',
    'addr_zip'
]


# --- CPU-Bound Address Parsing Functions ---

def _expand_and_parse_address(address_string: str) -> Dict[str, str]:
    """
    Internal helper to expand and parse an address string using libpostal.

    This function first expands common abbreviations (e.g., "St" -> "Street")
    to create a more standardized representation, then parses the result into
    labeled components (e.g., 'house_number', 'road').

    Args:
        address_string: The raw address string to parse.

    Returns:
        A dictionary mapping libpostal labels to their corresponding values.
        Returns an empty dictionary if the input is invalid.
    """
    if not address_string or not isinstance(address_string, str):
        return {}

    # `expand_address` returns a list of possible expansions. We assume the first
    # one is the most likely and use it for parsing.
    expanded_address_list = expand_address(address_string)
    most_likely_expansion = expanded_address_list[0] if expanded_address_list else address_string

    # `parse_address` returns a list of (value, label) tuples.
    parsed_tuples = parse_address(most_likely_expansion)

    # Convert the list of tuples into a more accessible dictionary.
    # This makes looking up components by their label (e.g., 'city') trivial.
    return {label: value for value, label in parsed_tuples}


def _format_parsed_components(parsed_components: Dict[str, str], original_address: str) -> Dict[str, str]:
    """
    Internal helper to format libpostal's output into a standardized schema.

    This function handles special cases like P.O. Boxes and ensures that the
    final output dictionary has a consistent structure.

    Args:
        parsed_components: The dictionary of components from libpostal.
        original_address: The original address string, used for P.O. Box parsing.

    Returns:
        A dictionary with standardized keys for each address component.
    """
    # Handle P.O. Boxes as a special case. If libpostal detects a 'po_box',
    # we use a regular expression to robustly extract the box number, as
    # libpostal's standard parsing can sometimes be inconsistent for this.
    if 'po_box' in parsed_components:
        # Search for "box" followed by an optional hash and the number.
        po_box_match = re.search(r'box\s*#?\s*(\d+)', original_address, re.IGNORECASE)
        street_name = f"PO BOX {po_box_match.group(1)}" if po_box_match else "PO BOX"
        street_number = ''  # P.O. Boxes do not have a separate street number.
    else:
        street_number = parsed_components.get('house_number', '')
        street_name = parsed_components.get('road', '')

    # Map the parsed (and potentially corrected) components to our final schema.
    # This ensures every address has the same set of keys, even if some values are empty.
    return {
        'address_line_1.street_number': street_number,
        'address_line_1.street_name': street_name,
        'city': parsed_components.get('city', ''),
        'state': parsed_components.get('state', '').upper(),
        # Standardize postal codes to the 5-digit format.
        'postal_code': parsed_components.get('postcode', '')[:5]
    }


def safe_parse_address(address_string: str) -> Dict[str, str]:
    """
    Parses a raw address string into a structured dictionary of components.

    This is the main public-facing parsing function. It orchestrates the
    expansion, parsing, and formatting steps, wrapping them in an exception
    handler to ensure the function never fails on malformed input. This is a
    CPU-bound operation.

    Args:
        address_string: The single, unparsed address string.

    Returns:
        A dictionary containing the standardized address components, or an
        empty dictionary if parsing fails.
    """
    logger.debug(f"Attempting to parse address: '{address_string}'")
    try:
        parsed_components = _expand_and_parse_address(address_string)
        if not parsed_components:
            logger.debug(f"Could not parse '{address_string}' into components.")
            return {}
        
        formatted_address = _format_parsed_components(parsed_components, address_string)
        logger.debug(f"Successfully parsed and formatted address: {formatted_address}")
        return formatted_address
    except Exception as e:
        # If any unexpected error occurs during parsing, log it for debugging
        # and return an empty dict to prevent downstream failures.
        logger.warning(f"libpostal parse error for '{address_string}': {e}")
        return {}


# --- GPU-Accelerated Address Utility Functions ---

def _is_series_present(series: cudf.Series) -> cudf.Series:
    """
    Helper to create a boolean mask for non-empty, non-null strings in a Series.

    This utility function reduces code duplication in the scoring logic.

    Args:
        series: A cuDF Series of strings.

    Returns:
        A boolean cuDF Series, where True indicates a valid, present string.
    """
    return series.notna() & (series != '')


def create_address_key_gpu(gdf: cudf.DataFrame) -> cudf.Series:
    """
    Creates a normalized address key from component columns on the GPU.

    This function serves as the single source of truth for the address key's
    format. It concatenates address components into a single, clean string
    that can be used for grouping and joining.

    Args:
        gdf: A cuDF DataFrame containing the component address columns defined
             in `ADDRESS_COMPONENT_COLUMNS`.

    Returns:
        A cuDF Series containing the combined and normalized address key.
    """
    logger.debug(f"Creating address key for DataFrame with shape {gdf.shape}")
    # Ensure all required columns exist in the DataFrame to prevent errors.
    # If a column is missing, it's added and filled with empty strings.
    for col in ADDRESS_COMPONENT_COLUMNS:
        if col not in gdf.columns:
            gdf[col] = ''

    # Convert all component columns to string type after filling nulls.
    key_components = [gdf[col].fillna('').astype(str) for col in ADDRESS_COMPONENT_COLUMNS]

    # Concatenate all components into a single string series, separated by spaces.
    normalized_key = key_components[0].str.cat(key_components[1:], sep=' ')

    # Apply final cleaning: convert to lowercase and collapse multiple spaces.
    return normalized_key.str.lower().str.normalize_spaces()


def calculate_address_score_gpu(gdf: cudf.DataFrame) -> cudf.Series:
    """
    Calculates a data completeness score for each address on the GPU.

    This function assigns weighted points based on the presence and validity
    of different address components. The score is used to identify the most
    descriptive address representation within a group of candidates.

    Args:
        gdf: A cuDF DataFrame containing the address component columns.

    Returns:
        A cuDF Series of integer scores, one for each row in the input DataFrame.
    """
    logger.debug(f"Calculating address scores for DataFrame with shape {gdf.shape}")
    # Initialize a Series of zeros with the same index as the input DataFrame.
    # This ensures correct alignment and serves as the base for our score.
    score = cudf.Series(0, index=gdf.index, dtype='int32')

    # --- Scoring Weights ---
    # The weights reflect the relative importance of each address component
    # in uniquely identifying a location.

    # A street name is the most critical component.
    score += _is_series_present(gdf['addr_street_name']).astype('int32') * 2

    # Street number, city, and a valid state are also highly important.
    score += _is_series_present(gdf['addr_street_number']).astype('int32') * 1
    score += _is_series_present(gdf['addr_city']).astype('int32') * 1
    score += (gdf['addr_state'].notna() & (gdf['addr_state'].str.len() == 2)).astype('int32') * 1

    # A valid 5-digit zip code adds confidence to the address.
    is_valid_zip = (
        gdf['addr_zip'].notna() &
        (gdf['addr_zip'].str.len() == 5) &
        gdf['addr_zip'].str.isdigit()
    )
    score += is_valid_zip.astype('int32') * 1

    return score


def get_best_address_gpu(address_gdf: cudf.DataFrame) -> cudf.DataFrame:
    """
    Selects the single best address from a DataFrame of candidates on the GPU.

    This function identifies the best address by first scoring each unique
    address based on its completeness. It then uses the frequency of each
    address as a primary tie-breaker, assuming the most common version is
    the most likely to be correct.

    Args:
        address_gdf: A cuDF DataFrame of address records for a single entity.
                     Must include the component columns and 'addr_normalized_key'.

    Returns:
        A single-row cuDF DataFrame representing the best address, or an empty
        DataFrame if the input is empty.
    """
    if address_gdf.empty:
        logger.debug("get_best_address_gpu received an empty DataFrame. Returning.")
        return address_gdf
    
    logger.info(f"Finding best address from {len(address_gdf)} candidates.")

    # First, calculate the frequency of each unique normalized address key.
    # This will be our tie-breaker.
    frequency_map = address_gdf['addr_normalized_key'].value_counts()
    logger.debug(f"Found {len(frequency_map)} unique addresses.")

    # To avoid scoring every single row, we work with only the unique addresses.
    unique_candidates = address_gdf.drop_duplicates(subset=['addr_normalized_key'])

    # Map the frequency counts to the unique candidates.
    # We use a temporary index for an efficient join-like operation.
    unique_candidates = unique_candidates.set_index('addr_normalized_key')
    unique_candidates['frequency'] = frequency_map
    unique_candidates = unique_candidates.reset_index()

    # Calculate the completeness score for each unique address.
    unique_candidates['score'] = calculate_address_score_gpu(unique_candidates)

    # --- Determine the Best Candidate ---
    # We sort by three criteria to get the most stable and reliable result:
    # 1. `score` (descending): The most complete address is best.
    # 2. `frequency` (descending): For addresses with the same score, the most common one wins.
    # 3. `addr_normalized_key` (ascending): A final, deterministic tie-breaker.
    best_candidate = unique_candidates.sort_values(
        by=['score', 'frequency', 'addr_normalized_key'],
        ascending=[False, False, True]
    ).head(1)
    
    logger.info(f"Selected best address with score {best_candidate['score'].iloc[0]} and frequency {best_candidate['frequency'].iloc[0]}.")

    return best_candidate
