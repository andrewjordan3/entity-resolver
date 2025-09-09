# entity_resolver/utils/address.py
"""
Address parsing and scoring utilities for entity resolution.

This module provides a hybrid approach to processing physical addresses. It uses
the CPU-based libpostal library for detailed parsing and normalization, and leverages
the GPU-accelerated cuDF library for high-performance scoring and selection
of addresses in large datasets.
"""

import logging
import re
from typing import Dict

import cudf
from postal.expand import expand_address
from postal.parser import parse_address

# ============================================================================
# Module-level Configuration
# ============================================================================

# Set up a dedicated logger for this module to provide contextualized output.
logger = logging.getLogger(__name__)

# Define standard address component columns used in GPU operations.
# This ensures consistency and avoids magic strings in function bodies.
ADDRESS_COMPONENT_COLUMNS = [
    'addr_street_number',
    'addr_street_name',
    'addr_city',
    'addr_state',
    'addr_zip'
]

# Define weights for address completeness scoring.
# Centralizing these constants makes the scoring logic easier to understand and tune.
# The weights reflect the relative importance of each component for uniquely
# identifying an address.
ADDRESS_SCORE_WEIGHTS = {
    'street_name': 2,  # Most critical component.
    'street_number': 1,
    'city': 1,
    'state': 1,       # A valid 2-character state code is required.
    'zip': 1          # A valid 5-digit zip code adds confidence.
}


# ============================================================================
# CPU-Based Address Parsing (libpostal)
# ============================================================================

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
        Returns an empty dictionary if the input is invalid or empty.
    """
    if not address_string or not isinstance(address_string, str):
        logger.debug("Input to _expand_and_parse_address was empty or not a string.")
        return {}

    # expand_address returns a list of possible expansions. We use the first,
    # most likely expansion for parsing to ensure consistency.
    expanded_address_list = expand_address(address_string)
    most_likely_expansion = expanded_address_list[0] if expanded_address_list else address_string

    # parse_address returns a list of (value, label) tuples.
    parsed_tuples = parse_address(most_likely_expansion)

    # Convert the list of tuples into a dictionary for easier access by label.
    return {label: value for value, label in parsed_tuples}


def _format_parsed_components(parsed_components: Dict[str, str], original_address: str) -> Dict[str, str]:
    """
    Internal helper to format libpostal's output into a standardized schema.

    This function maps the varied output from libpostal to a consistent
    dictionary structure. It handles special cases like P.O. Boxes and ensures
    that the final output dictionary has a predictable set of keys.

    Args:
        parsed_components: The dictionary of components from libpostal.
        original_address: The original address string, used for P.O. Box parsing.

    Returns:
        A dictionary with standardized keys for each address component.
    """
    # Handle P.O. Boxes as a special case, as they don't have a typical
    # street number and name. We extract the box number from the original string.
    if 'po_box' in parsed_components:
        # Use regex to robustly find the P.O. Box number.
        po_box_match = re.search(r'box\s*#?\s*(\d+)', original_address, re.IGNORECASE)
        street_name = f"PO BOX {po_box_match.group(1)}" if po_box_match else "PO BOX"
        street_number = ''  # P.O. Boxes do not have a separate street number.
    else:
        street_number = parsed_components.get('house_number', '')
        street_name = parsed_components.get('road', '')

    # Map parsed components to our internal, standardized schema.
    return {
        'address_line_1.street_number': street_number,
        'address_line_1.street_name': street_name,
        'city': parsed_components.get('city', ''),
        'state': parsed_components.get('state', ''),
        # Standardize postal codes to the 5-digit US format.
        # Note: This may truncate non-US postal codes.
        'postal_code': parsed_components.get('postcode', '')[:5]
    }


def safe_parse_address(address_string: str) -> Dict[str, str]:
    """
    Parses a raw address string into a structured dictionary of components.

    This is the main public-facing CPU parsing function. It orchestrates the
    expansion, parsing, and formatting steps, wrapping them in an exception
    handler to ensure robust operation even with malformed input.

    Args:
        address_string: The single, unparsed address string.

    Returns:
        A dictionary containing the standardized address components, or an
        empty dictionary if parsing fails.
    """
    try:
        parsed_components = _expand_and_parse_address(address_string)
        if not parsed_components:
            logger.debug(f"Could not parse address into components: '{address_string[:50]}...'")
            return {}

        formatted_address = _format_parsed_components(parsed_components, address_string)
        return formatted_address

    except Exception as e:
        # Catch any unexpected errors from the libpostal library.
        logger.warning(f"libpostal parse error for '{address_string[:50]}...': {e}")
        return {}


# ============================================================================
# GPU-Accelerated Address Utilities (cuDF)
# ============================================================================

def _is_series_present(series: cudf.Series) -> cudf.Series:
    """
    Helper to create a boolean mask for non-empty, non-null strings in a Series.

    Args:
        series: A cuDF Series of strings.

    Returns:
        A boolean cuDF Series, where True indicates a valid, present string.
    """
    return series.notna() & (series != '')


def _ensure_address_columns(gdf: cudf.DataFrame) -> cudf.DataFrame:
    """
    Ensures all standard address component columns exist in the DataFrame.

    Args:
        gdf: The input cuDF DataFrame.

    Returns:
        The DataFrame with missing address columns added and filled with
        empty strings.
    """
    for col in ADDRESS_COMPONENT_COLUMNS:
        if col not in gdf.columns:
            logger.debug(f"Column '{col}' not found. Adding it as an empty column.")
            gdf[col] = ''
    return gdf


def create_address_key_gpu(address_dataframe: cudf.DataFrame) -> cudf.Series:
    """
    Creates a robust, normalized address key from component columns on the GPU.

    This function is designed to produce a stable and consistent identifier for
    each address, even when the source data has minor variations. It achieves this
    by individually normalizing each critical component of the address before
    concatenating them into a single key string. This component-wise approach
    is far more reliable than cleaning the concatenated string.

    The normalization pipeline includes:
    1.  Standardizing street numbers (e.g., handling ranges like "123-125").
    2.  Cleaning street names (e.g., removing inconsistent directional suffixes).
    3.  Ensuring consistent ZIP code format (first 5 digits).

    Args:
        address_dataframe: A cuDF DataFrame containing the component address
                           columns (e.g., 'addr_street_name', 'addr_city').

    Returns:
        A cuDF Series containing the combined and highly normalized address key,
        suitable for use in grouping, joining, or as a unique entity identifier.
    """
    # Work on a copy of the DataFrame to prevent unintended side effects on the
    # original DataFrame passed to the function. This is a safe practice.
    address_dataframe_copy = _ensure_address_columns(address_dataframe.copy())
    
    # --- Component-wise Normalization ---

    # Normalize the street number.
    # First, fill any nulls with an empty string and ensure it's a string type.
    normalized_street_number = address_dataframe_copy['addr_street_number'].fillna('').astype(str)
    # Then, remove any non-numeric characters that appear after the first sequence
    # of numbers. This effectively handles ranges (e.g., "123-125" becomes "123")
    # and extraneous text (e.g., "456 Apt B" becomes "456").
    normalized_street_number = normalized_street_number.str.replace(r'[^0-9].*', '', regex=True)
    
    # Normalize the street name.
    # Fill nulls, convert to string, and lowercase for case-insensitive matching.
    standardized_street_name = address_dataframe_copy['addr_street_name'].fillna('').astype(str).str.lower()
    # Remove common directional suffixes (e.g., n, s, e, w and their full-word
    # counterparts). This is crucial because their usage can be inconsistent in
    # source data ("Main St" vs "Main St N").
    standardized_street_name = standardized_street_name.str.replace(r'\s+(n|s|e|w|north|south|east|west)$', '', regex=True)
    
    # Standardize the city name.
    # Fill nulls, convert to string, and lowercase. City name abbreviations
    # are handled by the upstream libpostal process, so no replacements are needed here.
    standardized_city = address_dataframe_copy['addr_city'].fillna('').astype(str).str.lower()
    
    # --- Key Assembly ---

    # Create a list of the cleaned, standardized component Series.
    # For state and zip, we perform a simple null fill, type conversion, and lowercase.
    # The ZIP code is truncated to the first 5 digits to standardize formats like ZIP+4.
    address_key_components = [
        normalized_street_number,
        standardized_street_name,
        standardized_city,
        address_dataframe_copy['addr_state'].fillna('').astype(str).str.lower(),
        address_dataframe_copy['addr_zip'].fillna('').astype(str).str[:5]
    ]
    
    # Concatenate all the processed components into a single string Series.
    # A pipe character '|' is used as a delimiter. This is a robust choice
    # because it is not a character that typically appears in address data,
    # preventing ambiguity that a space or comma could cause.
    final_address_key = address_key_components[0].str.cat(address_key_components[1:], sep='|')
    
    # As a final cleanup step, normalize all whitespace to single spaces and
    # trim any leading/trailing whitespace. This handles any extraneous spaces
    # that may have been introduced during the replacement steps.
    return final_address_key.str.normalize_spaces()


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
    gdf = _ensure_address_columns(gdf.copy())
    score = cudf.Series(0, index=gdf.index, dtype='int32')

    # Street name is most critical.
    score += _is_series_present(gdf['addr_street_name']).astype('int32') * ADDRESS_SCORE_WEIGHTS['street_name']

    # Street number, city, and state are also important.
    score += _is_series_present(gdf['addr_street_number']).astype('int32') * ADDRESS_SCORE_WEIGHTS['street_number']
    score += _is_series_present(gdf['addr_city']).astype('int32') * ADDRESS_SCORE_WEIGHTS['city']

    # For state, we check for a valid 2-character abbreviation.
    is_valid_state = gdf['addr_state'].notna() & (gdf['addr_state'].str.len() == 2)
    score += is_valid_state.astype('int32') * ADDRESS_SCORE_WEIGHTS['state']

    # For ZIP code, we validate a 5-digit format.
    is_valid_zip = (
        gdf['addr_zip'].notna() &
        (gdf['addr_zip'].str.len() == 5) &
        gdf['addr_zip'].str.isdigit()
    )
    score += is_valid_zip.astype('int32') * ADDRESS_SCORE_WEIGHTS['zip']

    return score


def get_best_address_gpu(address_gdf: cudf.DataFrame) -> cudf.DataFrame:
    """
    Selects the single best address from a DataFrame of candidates on the GPU.

    This function identifies the best address by first scoring each unique address
    based on data completeness, then using the frequency of the address as a
    tie-breaker. This ensures the most complete and most common representation
    is chosen.

    Args:
        address_gdf: A cuDF DataFrame of address records for a single entity.
                     Must include the 'addr_normalized_key' column generated by
                     `create_address_key_gpu`.

    Returns:
        A single-row cuDF DataFrame representing the best address, or an empty
        DataFrame if the input is empty.
    """
    if address_gdf.empty:
        logger.debug("get_best_address_gpu received an empty DataFrame, returning empty.")
        return address_gdf

    logger.debug(f"Finding best address from {len(address_gdf)} candidates.")

    # Step 1: Calculate frequency of each unique normalized address key.
    frequency_map = address_gdf['addr_normalized_key'].value_counts()
    frequency_map.name = 'frequency'

    # Step 2: Work with unique addresses only for efficiency.
    unique_candidates = address_gdf.drop_duplicates(subset=['addr_normalized_key'])

    # Step 3: Map frequency counts back to the unique candidates.
    unique_candidates = unique_candidates.merge(
        frequency_map,
        left_on='addr_normalized_key',
        right_index=True,
        how='left'
    )

    # Step 4: Calculate completeness score for each unique address.
    unique_candidates['score'] = calculate_address_score_gpu(unique_candidates)

    # Step 5: Sort to find the best candidate.
    # The logic is: highest score wins. If scores are tied, highest frequency wins.
    # If both are tied, sort by key for a deterministic result.
    best_candidate = unique_candidates.sort_values(
        by=['score', 'frequency', 'addr_normalized_key'],
        ascending=[False, False, True]
    ).head(1)

    if not best_candidate.empty:
        # Use .item() to extract the scalar value for logging.
        score = best_candidate['score'].iloc[0].item()
        freq = best_candidate['frequency'].iloc[0].item()
        logger.debug(f"Selected best address with score {score} and frequency {freq}.")

    # Drop intermediate columns used for ranking before returning.
    return best_candidate.drop(columns=['score', 'frequency'])
