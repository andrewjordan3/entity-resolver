# entity_resolver/utils/text.py
"""
Core text processing utilities for the entity resolution pipeline.

This module provides two primary, high-performance functionalities essential for
preparing and analyzing entity data:

1.  **Advanced Unicode Normalization:**
    A robust, multi-stage function (`nfkc_normalize_series`) for cleaning and
    canonicalizing strings. It goes beyond simple lowercasing by applying the
    formal NFKC Unicode standard, followed by a custom ASCII-folding pass to
    handle a comprehensive set of visual and formatting ambiguities. The process
    is highly optimized for performance on large datasets by using a
    "factorize-apply-remap" pattern.

2.  **Canonical Name Selection:**
    A sophisticated, GPU-accelerated function (`get_canonical_name_gpu`) that
    intelligently selects the best representative name from a cluster of
    candidate strings. It uses a multi-faceted scoring model that considers
    a name's frequency, its syntactic similarity to other frequent names
    (centrality), and its length (descriptiveness).

These utilities are foundational to ensuring that input data is clean,
consistent, and ready for accurate vectorization and matching.
"""

import logging
import pandas as pd
import re
import cudf
import cupy
import unicodedata2
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.metrics.pairwise_distances import pairwise_distances
from ..config import SimilarityTfidfParams

# Set up a logger for this module.
logger = logging.getLogger(__name__)

# --- Module-level constants for ASCII Punctuation Folding ---
# This dictionary maps a comprehensive set of Unicode punctuation, symbols,
# and control characters to their basic ASCII equivalents. This is a crucial
# post-processing step after NFKC normalization to ensure a canonical string
# representation for matching against messy, real-world data.
_ASCII_COMPAT_MAP = {
    # --- Punctuation and Symbols (fold to ASCII) ---
    "\u2044": "/",  # FRACTION SLASH → SOLIDUS
    "\u2215": "/",  # DIVISION SLASH → SOLIDUS
    "\u00F7": "/",  # DIVISION SIGN → SOLIDUS (keep for part numbers)
    "\u00D7": "x",  # MULTIPLICATION SIGN → letter x
    "\u2212": "-",  # MINUS SIGN → HYPHEN-MINUS
    "\u2010": "-",  # HYPHEN
    "\u2011": "-",  # NON-BREAKING HYPHEN
    "\u2012": "-",  # FIGURE DASH
    "\u2013": "-",  # EN DASH
    "\u2014": "-",  # EM DASH
    "\u2015": "-",  # HORIZONTAL BAR
    "\u2043": "-",  # HYPHEN BULLET → hyphen
    "\u2022": "-",  # BULLET → hyphen (your existing choice)
    "\u2219": ".",  # BULLET OPERATOR → period
    "\u00B7": " ",  # MIDDLE DOT → space (acts as separator)
    "\u2218": "o",  # RING OPERATOR → 'o' (seen in part numbers)
    "\u2026": "...",# HORIZONTAL ELLIPSIS → three periods
    "\u00AE": "",   # REGISTERED SIGN (®) → remove
    "\u2122": "",   # TRADE MARK SIGN (™) → remove
    "\u00A9": "",   # COPYRIGHT SIGN (©) → remove
    "\u2120": "",   # SERVICE MARK (℠) → remove
    "\u00B0": "",   # DEGREE SIGN (°) → remove
    "\u02DA": "",   # RING ABOVE (˚) → remove
    
    # --- Currency Symbols (remove for entity matching) ---
    "\u00A2": "",   # CENT SIGN (¢) → remove
    "\u00A3": "",   # POUND SIGN (£) → remove
    "\u00A4": "",   # CURRENCY SIGN (¤) → remove
    "\u00A5": "",   # YEN SIGN (¥) → remove
    "\u20AC": "",   # EURO SIGN (€) → remove
    "\u2030": "",   # PER MILLE SIGN (‰) → remove
    "\u2031": "",   # PER TEN THOUSAND SIGN → remove
    
    # --- Mathematical and Technical Symbols ---
    "\u00B1": "+-", # PLUS-MINUS SIGN → plus-minus
    "\u2248": "~",  # ALMOST EQUAL TO → tilde
    "\u2260": "!=", # NOT EQUAL TO → != 
    "\u2264": "<=", # LESS-THAN OR EQUAL TO → <=
    "\u2265": ">=", # GREATER-THAN OR EQUAL TO → >=
    "\u00AC": "!",  # NOT SIGN → exclamation
    "\u221E": "",   # INFINITY → remove
    
    # --- Ordinal Indicators (important for addresses) ---
    "\u00AA": "a",  # FEMININE ORDINAL INDICATOR (ª) → 'a'
    "\u00BA": "o",  # MASCULINE ORDINAL INDICATOR (º) → 'o'
    "\u02E2": "s",  # MODIFIER LETTER SMALL S (superscript s)
    "\u02E3": "x",  # MODIFIER LETTER SMALL X (superscript x)
    
    # --- Quotes and Primes (unify to ASCII) ---
    "\u2018": "'",  # LEFT SINGLE QUOTATION MARK
    "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK
    "\u201A": "'",  # SINGLE LOW-9 QUOTATION MARK
    "\u201B": "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    "\u2032": "'",  # PRIME (feet/minutes) → apostrophe
    "\u2035": "'",  # REVERSED PRIME → apostrophe
    "\u201C": '"',  # LEFT DOUBLE QUOTATION MARK
    "\u201D": '"',  # RIGHT DOUBLE QUOTATION MARK
    "\u201E": '"',  # DOUBLE LOW-9 QUOTATION MARK
    "\u201F": '"',  # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    "\u2033": '"',  # DOUBLE PRIME (inches/seconds) → quote
    "\u2036": '"',  # REVERSED DOUBLE PRIME → quote
    "\u2034": "'''", # TRIPLE PRIME → three apostrophes
    "\u2037": "'''", # REVERSED TRIPLE PRIME → three apostrophes
    "\u00AB": '"',  # LEFT-POINTING DOUBLE ANGLE QUOTATION MARK («)
    "\u00BB": '"',  # RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK (»)
    "\u2039": "'",  # SINGLE LEFT-POINTING ANGLE QUOTATION MARK (‹)
    "\u203A": "'",  # SINGLE RIGHT-POINTING ANGLE QUOTATION MARK (›)
    
    # --- Whitespace Variants (map to standard space) ---
    "\t": " ",      # TAB → space
    "\n": " ",      # NEWLINE → space (for multi-line addresses)
    "\r": " ",      # CARRIAGE RETURN → space
    "\u00A0": " ",  # NO-BREAK SPACE
    "\u2000": " ",  # EN QUAD
    "\u2001": " ",  # EM QUAD
    "\u2002": " ",  # EN SPACE
    "\u2003": " ",  # EM SPACE
    "\u2004": " ",  # THREE-PER-EM SPACE
    "\u2005": " ",  # FOUR-PER-EM SPACE
    "\u2006": " ",  # SIX-PER-EM SPACE
    "\u2007": " ",  # FIGURE SPACE
    "\u2008": " ",  # PUNCTUATION SPACE
    "\u2009": " ",  # THIN SPACE
    "\u200A": " ",  # HAIR SPACE
    "\u202F": " ",  # NARROW NO-BREAK SPACE
    "\u205F": " ",  # MEDIUM MATHEMATICAL SPACE
    "\u3000": " ",  # IDEOGRAPHIC SPACE
    "\u2028": " ",  # LINE SEPARATOR
    "\u2029": " ",  # PARAGRAPH SEPARATOR
    
    # --- Zero-Width and Invisible Characters (remove completely) ---
    "\u00AD": "",   # SOFT HYPHEN
    "\u200B": "",   # ZERO WIDTH SPACE
    "\u200C": "",   # ZERO WIDTH NON-JOINER
    "\u200D": "",   # ZERO WIDTH JOINER
    "\u2060": "",   # WORD JOINER
    "\u180E": "",   # MONGOLIAN VOWEL SEPARATOR
    "\uFEFF": "",   # ZERO WIDTH NO-BREAK SPACE (BOM)
    "\u034F": "",   # COMBINING GRAPHEME JOINER
    "\u061C": "",   # ARABIC LETTER MARK
    "\u115F": "",   # HANGUL CHOSEONG FILLER
    "\u1160": "",   # HANGUL JUNGSEONG FILLER
    "\u17B4": "",   # KHMER VOWEL INHERENT AQ
    "\u17B5": "",   # KHMER VOWEL INHERENT AA
    
    # --- BiDi / Directional Controls (remove completely) ---
    "\u200E": "",   # LEFT-TO-RIGHT MARK
    "\u200F": "",   # RIGHT-TO-LEFT MARK
    "\u202A": "",   # LEFT-TO-RIGHT EMBEDDING
    "\u202B": "",   # RIGHT-TO-LEFT EMBEDDING
    "\u202C": "",   # POP DIRECTIONAL FORMATTING
    "\u202D": "",   # LEFT-TO-RIGHT OVERRIDE
    "\u202E": "",   # RIGHT-TO-LEFT OVERRIDE
    "\u2066": "",   # LEFT-TO-RIGHT ISOLATE
    "\u2067": "",   # RIGHT-TO-LEFT ISOLATE
    "\u2068": "",   # FIRST STRONG ISOLATE
    "\u2069": "",   # POP DIRECTIONAL ISOLATE
    "\u206A": "",   # INHIBIT SYMMETRIC SWAPPING
    "\u206B": "",   # ACTIVATE SYMMETRIC SWAPPING
    "\u206C": "",   # INHIBIT ARABIC FORM SHAPING
    "\u206D": "",   # ACTIVATE ARABIC FORM SHAPING
    "\u206E": "",   # NATIONAL DIGIT SHAPES
    "\u206F": "",   # NOMINAL DIGIT SHAPES
    
    # --- Additional Symbols and Punctuation ---
    "\u00A6": "|",  # BROKEN BAR → vertical bar
    "\u00A7": "",   # SECTION SIGN (§) → remove
    "\u00B6": "",   # PILCROW SIGN (¶) → remove
    "\u2020": "",   # DAGGER (†) → remove
    "\u2021": "",   # DOUBLE DAGGER (‡) → remove
    "\u2023": "-",  # TRIANGULAR BULLET → hyphen
    "\u2024": ".",  # ONE DOT LEADER → period
    "\u2025": "..", # TWO DOT LEADER → two periods
    "\u2027": "-",  # HYPHENATION POINT → hyphen
    "\u203B": "*",  # REFERENCE MARK → asterisk
    "\u203C": "!!", # DOUBLE EXCLAMATION MARK → two exclamations
    "\u203D": "?!", # INTERROBANG → question-exclamation
    "\u2047": "??", # DOUBLE QUESTION MARK → two questions
    "\u2048": "?!", # QUESTION EXCLAMATION MARK → question-exclamation
    "\u2049": "!?", # EXCLAMATION QUESTION MARK → exclamation-question
    "\u204B": "",   # REVERSED PILCROW SIGN → remove
    "\u204C": "",   # BLACK LEFTWARDS BULLET → remove
    "\u204D": "",   # BLACK RIGHTWARDS BULLET → remove
    "\u2052": "%",  # COMMERCIAL MINUS SIGN → percent
    "\u2053": "~",  # SWUNG DASH → tilde
}

# Create the translation table once when the module is imported.
# This is a significant performance optimization.
_ASCII_TRANSLATION_TABLE = str.maketrans(_ASCII_COMPAT_MAP)

def _calculate_centrality_score(
    unique_names: cudf.Series,
    name_counts: cudf.Series,
    tfidf_params: SimilarityTfidfParams,
    min_unique_for_similarity: int = 5,
) -> cupy.ndarray:
    """
    Calculates a centrality score for each unique name.

    This score is high for names that are both frequent and highly similar to
    other frequent names in the group. It uses TF-IDF character n-grams to
    measure similarity.

    Args:
        unique_names: A Series of unique name strings.
        name_counts: A Series containing the frequency count for each unique name.

    Returns:
        A CuPy array of centrality scores, one for each unique name.
    """
    min_for_df = 20 # Need to make this a parameter eventually
    n_unique = len(unique_names)
    
    # If we have too few unique names, similarity is meaningless
    if n_unique < min_unique_for_similarity:
        # Return just the frequency weights as the centrality score
        # This makes the most frequent name the most "central"
        total_items = name_counts.sum()
        freq_weights = name_counts / total_items
        logger.debug(
            f"Only {n_unique} unique names (< {min_unique_for_similarity}), "
            f"using frequency weights as centrality scores"
        )
        return cupy.asarray(freq_weights.values)
    
    exclude_keys = {'min_df', 'max_df'} if n_unique < min_for_df else set()

    vec_params = tfidf_params.model_dump(
        mode="python",
        round_trip=True,
        exclude=exclude_keys,
        exclude_none=True,
    )
    # Using character n-grams is effective for capturing misspellings and variations.
    vectorizer = TfidfVectorizer(**vec_params)
    # We must reset the index before vectorizing to avoid potential cuML errors.
    tfidf_matrix = vectorizer.fit_transform(unique_names.reset_index(drop=True))

    # Check if we got meaningful features
    if tfidf_matrix.shape[1] == 0:
        logger.warning("TF-IDF produced no features, falling back to frequency weights")
        total_items = name_counts.sum()
        freq_weights = name_counts / total_items
        return cupy.asarray(freq_weights.values)

    # Calculate the cosine similarity between all pairs of unique names.
    similarity_matrix = 1 - pairwise_distances(tfidf_matrix, metric='cosine')

    # Calculate frequency weights (proportion of total).
    total_items = name_counts.sum()
    freq_weights = name_counts / total_items

    # The core of the centrality score: for each name, sum the similarities
    # to all other names, weighted by how frequent those other names are.
    # A high score means "I am similar to other popular names."
    # The '@' operator performs matrix multiplication.
    centrality_score = similarity_matrix @ freq_weights.values

    return cupy.asarray(centrality_score)

def _calculate_length_bonus(unique_names: cudf.Series) -> cupy.ndarray:
    """
    Calculates a descriptiveness bonus based on name length.

    Longer names are often more descriptive and less likely to be abbreviations,
    so they receive a bonus. The score is logarithmic to provide diminishing
    returns, preventing absurdly long names from dominating.

    Args:
        unique_names: A Series of unique name strings.

    Returns:
        A CuPy array of length bonus scores, one for each unique name.
    """
    lengths = unique_names.str.len().astype('float32')
    # The '+ 1' avoids taking the log of zero.
    # The clip prevents the bonus from growing excessively large.
    length_bonus = cupy.log(lengths + 1).clip(max=3.5)
    return cupy.asarray(length_bonus)


def get_canonical_name_gpu(
        name_series: cudf.Series, 
        tfidf_params: SimilarityTfidfParams,
    ) -> str:
    """
    Selects the best canonical name from a Series of candidates on the GPU.

    This function identifies the best name by scoring each unique candidate based
    on a combination of three factors:
    1.  **Centrality**: How similar a name is to other frequent names.
    2.  **Frequency**: How often the name itself appears.
    3.  **Descriptiveness**: A bonus for length, favoring complete names over abbreviations.

    Args:
        name_series: A cuDF Series containing all name candidates for a single group.
        tfidf_params: Parameters for the TfidfVectorizer.

    Returns:
        The string of the highest-scoring canonical name, or an empty string
        if the input is empty.
    """
    if name_series.empty:
        logger.warning("get_canonical_name_gpu received an empty series.")
        return ""

    unique_names = name_series.unique()
    if len(unique_names) == 1:
        return unique_names.iloc[0]

    logger.debug(f"Finding canonical name from {len(unique_names)} unique candidates (out of {len(name_series)} total).")

    # --- Score Calculation ---
    name_counts = name_series.value_counts().reindex(unique_names).fillna(0)

    # 1. Calculate the centrality score.
    centrality_score = _calculate_centrality_score(unique_names, name_counts, tfidf_params)

    # 2. Get the direct frequency score.
    frequency_score = (name_counts / name_counts.sum()).values

    # 3. Calculate the length bonus.
    length_bonus = _calculate_length_bonus(unique_names)

    # --- Combine Scores ---
    # The base score combines centrality and raw frequency. A name can score
    # well by being frequent itself, or by being very similar to other
    # frequent names.
    base_score = centrality_score + frequency_score

    # The final score is modulated by the length bonus. This helps break ties
    # and promotes more descriptive names.
    final_scores = base_score * length_bonus
    logger.debug(f"Calculated final scores for {len(final_scores)} unique names.")

    # Find the index of the highest score and return the corresponding name.
    best_name_index = final_scores.argmax()
    best_name = unique_names.iloc[int(best_name_index)]
    logger.debug(f"Selected '{best_name}' as the canonical name.")

    return best_name

def _nfkc_normalize(text_element: str) -> str:
    """
    Applies a three-stage normalization for maximum text canonicalization.

    This function is the core, element-wise normalization logic. It is designed
    to be applied to a single string. For applying to a full Series, use the
    optimized `nfkc_normalize_series` function.

    Stage 1: NFKC Normalization
    ---------------------------
    Handles "compatibility" characters like full-width variants, ligatures, and
    superscripts, converting them to their canonical forms.

    Stage 2: ASCII Punctuation Folding
    ----------------------------------
    Performs an additional step to convert a comprehensive set of visually
    similar punctuation, symbols, and whitespace variants into their basic ASCII
    counterparts. It also removes invisible control and zero-width characters.

    Stage 3: Whitespace Normalization
    ---------------------------------
    Collapses any sequences of multiple whitespace characters into a single
    space and strips leading/trailing whitespace from the final string.

    Args:
        text_element: A single string to be normalized.

    Returns:
        A new string that has been fully normalized, folded, and cleaned.
    """
    # Handle None/NaN values gracefully
    if pd.isna(text_element) or text_element is None:
        return ""
    
    # Ensure we're working with a string
    text_element = str(text_element)
    
    # Stage 1: Apply the formal Unicode NFKC normalization.
    normalized_text = unicodedata2.normalize('NFKC', text_element)

    # Stage 2: Apply the fast, manual translation for remaining compatibility characters.
    normalized_text = normalized_text.translate(_ASCII_TRANSLATION_TABLE)
    
    # Stage 3: Collapse all whitespace to single spaces and strip ends.
    return re.sub(r'\s+', ' ', normalized_text).strip()


def nfkc_normalize_series(input_series: cudf.Series) -> cudf.Series:
    """
    Applies NFKC normalization to a cudf.Series in a highly optimized manner.

    This function uses the "factorize-apply-remap" pattern to significantly
    speed up the application of the Python-based `_nfkc_normalize` UDF. Instead
    of applying the function to every row, it applies it only to the unique
    values in the Series and then maps the results back.

    This method is substantially faster than `series.apply(_nfkc_normalize)`
    for any data with repeated string values.

    Args:
        input_series: A cuDF Series of strings to be normalized.

    Returns:
        A new cuDF Series with the normalized string values.
    """
    # Store original null mask to preserve it in the output
    original_null_mask = input_series.isna()
    
    # Move the data to the CPU for pandas' highly optimized string operations.
    # Nulls are handled and type is guaranteed before processing.
    pandas_series = input_series.fillna("").astype("str").to_pandas()

    # Factorize the Series to get the unique string values and the integer codes
    # that map each original row to its unique value.
    category_codes, unique_values = pd.factorize(pandas_series, sort=False)

    # Apply the expensive normalization function ONLY to the smaller set of unique values.
    normalized_unique_values = [_nfkc_normalize(unique_value) for unique_value in unique_values]

    # Reconstruct the full-length series by using the category codes to "take"
    # the normalized values from the unique array.
    remapped_series = pd.Series(normalized_unique_values).take(category_codes)

    # Move the final, normalized series back to the GPU, preserving the original index.
    result_series = cudf.Series(remapped_series.values, index=input_series.index)
    
    # Restore original nulls (important for data integrity)
    result_series[original_null_mask] = None
    
    return result_series