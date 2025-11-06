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
import re

import cudf
import cupy
import pandas as pd
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
    '\u2044': '/',  # FRACTION SLASH → SOLIDUS
    '\u2215': '/',  # DIVISION SLASH → SOLIDUS
    '\u00f7': '/',  # DIVISION SIGN → SOLIDUS (keep for part numbers)
    '\u00d7': 'x',  # MULTIPLICATION SIGN → letter x
    '\u2212': '-',  # MINUS SIGN → HYPHEN-MINUS
    '\u2010': '-',  # HYPHEN
    '\u2011': '-',  # NON-BREAKING HYPHEN
    '\u2012': '-',  # FIGURE DASH
    '\u2013': '-',  # EN DASH
    '\u2014': '-',  # EM DASH
    '\u2015': '-',  # HORIZONTAL BAR
    '\u2043': '-',  # HYPHEN BULLET → hyphen
    '\u2022': '-',  # BULLET → hyphen (your existing choice)
    '\u2219': '.',  # BULLET OPERATOR → period
    '\u00b7': ' ',  # MIDDLE DOT → space (acts as separator)
    '\u2218': 'o',  # RING OPERATOR → 'o' (seen in part numbers)
    '\u2026': '...',  # HORIZONTAL ELLIPSIS → three periods
    '\u00ae': '',  # REGISTERED SIGN (®) → remove
    '\u2122': '',  # TRADE MARK SIGN (™) → remove
    '\u00a9': '',  # COPYRIGHT SIGN (©) → remove
    '\u2120': '',  # SERVICE MARK (℠) → remove
    '\u00b0': '',  # DEGREE SIGN (°) → remove
    '\u02da': '',  # RING ABOVE (˚) → remove
    # --- Currency Symbols (remove for entity matching) ---
    '\u00a2': '',  # CENT SIGN (¢) → remove
    '\u00a3': '',  # POUND SIGN (£) → remove
    '\u00a4': '',  # CURRENCY SIGN (¤) → remove
    '\u00a5': '',  # YEN SIGN (¥) → remove
    '\u20ac': '',  # EURO SIGN (€) → remove
    '\u2030': '',  # PER MILLE SIGN (‰) → remove
    '\u2031': '',  # PER TEN THOUSAND SIGN → remove
    # --- Mathematical and Technical Symbols ---
    '\u00b1': '+-',  # PLUS-MINUS SIGN → plus-minus
    '\u2248': '~',  # ALMOST EQUAL TO → tilde
    '\u2260': '!=',  # NOT EQUAL TO → !=
    '\u2264': '<=',  # LESS-THAN OR EQUAL TO → <=
    '\u2265': '>=',  # GREATER-THAN OR EQUAL TO → >=
    '\u00ac': '!',  # NOT SIGN → exclamation
    '\u221e': '',  # INFINITY → remove
    # --- Ordinal Indicators (important for addresses) ---
    '\u00aa': 'a',  # FEMININE ORDINAL INDICATOR (ª) → 'a'
    '\u00ba': 'o',  # MASCULINE ORDINAL INDICATOR (º) → 'o'
    '\u02e2': 's',  # MODIFIER LETTER SMALL S (superscript s)
    '\u02e3': 'x',  # MODIFIER LETTER SMALL X (superscript x)
    # --- Quotes and Primes (unify to ASCII) ---
    '\u2018': "'",  # LEFT SINGLE QUOTATION MARK
    '\u2019': "'",  # RIGHT SINGLE QUOTATION MARK
    '\u201a': "'",  # SINGLE LOW-9 QUOTATION MARK
    '\u201b': "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    '\u2032': "'",  # PRIME (feet/minutes) → apostrophe
    '\u2035': "'",  # REVERSED PRIME → apostrophe
    '\u201c': '"',  # LEFT DOUBLE QUOTATION MARK
    '\u201d': '"',  # RIGHT DOUBLE QUOTATION MARK
    '\u201e': '"',  # DOUBLE LOW-9 QUOTATION MARK
    '\u201f': '"',  # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    '\u2033': '"',  # DOUBLE PRIME (inches/seconds) → quote
    '\u2036': '"',  # REVERSED DOUBLE PRIME → quote
    '\u2034': "'''",  # TRIPLE PRIME → three apostrophes
    '\u2037': "'''",  # REVERSED TRIPLE PRIME → three apostrophes
    '\u00ab': '"',  # LEFT-POINTING DOUBLE ANGLE QUOTATION MARK («)
    '\u00bb': '"',  # RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK (»)
    '\u2039': "'",  # SINGLE LEFT-POINTING ANGLE QUOTATION MARK (‹)
    '\u203a': "'",  # SINGLE RIGHT-POINTING ANGLE QUOTATION MARK (›)
    # --- Whitespace Variants (map to standard space) ---
    '\t': ' ',  # TAB → space
    '\n': ' ',  # NEWLINE → space (for multi-line addresses)
    '\r': ' ',  # CARRIAGE RETURN → space
    '\u00a0': ' ',  # NO-BREAK SPACE
    '\u2000': ' ',  # EN QUAD
    '\u2001': ' ',  # EM QUAD
    '\u2002': ' ',  # EN SPACE
    '\u2003': ' ',  # EM SPACE
    '\u2004': ' ',  # THREE-PER-EM SPACE
    '\u2005': ' ',  # FOUR-PER-EM SPACE
    '\u2006': ' ',  # SIX-PER-EM SPACE
    '\u2007': ' ',  # FIGURE SPACE
    '\u2008': ' ',  # PUNCTUATION SPACE
    '\u2009': ' ',  # THIN SPACE
    '\u200a': ' ',  # HAIR SPACE
    '\u202f': ' ',  # NARROW NO-BREAK SPACE
    '\u205f': ' ',  # MEDIUM MATHEMATICAL SPACE
    '\u3000': ' ',  # IDEOGRAPHIC SPACE
    '\u2028': ' ',  # LINE SEPARATOR
    '\u2029': ' ',  # PARAGRAPH SEPARATOR
    # --- Zero-Width and Invisible Characters (remove completely) ---
    '\u00ad': '',  # SOFT HYPHEN
    '\u200b': '',  # ZERO WIDTH SPACE
    '\u200c': '',  # ZERO WIDTH NON-JOINER
    '\u200d': '',  # ZERO WIDTH JOINER
    '\u2060': '',  # WORD JOINER
    '\u180e': '',  # MONGOLIAN VOWEL SEPARATOR
    '\ufeff': '',  # ZERO WIDTH NO-BREAK SPACE (BOM)
    '\u034f': '',  # COMBINING GRAPHEME JOINER
    '\u061c': '',  # ARABIC LETTER MARK
    '\u115f': '',  # HANGUL CHOSEONG FILLER
    '\u1160': '',  # HANGUL JUNGSEONG FILLER
    '\u17b4': '',  # KHMER VOWEL INHERENT AQ
    '\u17b5': '',  # KHMER VOWEL INHERENT AA
    # --- BiDi / Directional Controls (remove completely) ---
    '\u200e': '',  # LEFT-TO-RIGHT MARK
    '\u200f': '',  # RIGHT-TO-LEFT MARK
    '\u202a': '',  # LEFT-TO-RIGHT EMBEDDING
    '\u202b': '',  # RIGHT-TO-LEFT EMBEDDING
    '\u202c': '',  # POP DIRECTIONAL FORMATTING
    '\u202d': '',  # LEFT-TO-RIGHT OVERRIDE
    '\u202e': '',  # RIGHT-TO-LEFT OVERRIDE
    '\u2066': '',  # LEFT-TO-RIGHT ISOLATE
    '\u2067': '',  # RIGHT-TO-LEFT ISOLATE
    '\u2068': '',  # FIRST STRONG ISOLATE
    '\u2069': '',  # POP DIRECTIONAL ISOLATE
    '\u206a': '',  # INHIBIT SYMMETRIC SWAPPING
    '\u206b': '',  # ACTIVATE SYMMETRIC SWAPPING
    '\u206c': '',  # INHIBIT ARABIC FORM SHAPING
    '\u206d': '',  # ACTIVATE ARABIC FORM SHAPING
    '\u206e': '',  # NATIONAL DIGIT SHAPES
    '\u206f': '',  # NOMINAL DIGIT SHAPES
    # --- Additional Symbols and Punctuation ---
    '\u00a6': '|',  # BROKEN BAR → vertical bar
    '\u00a7': '',  # SECTION SIGN (§) → remove
    '\u00b6': '',  # PILCROW SIGN (¶) → remove
    '\u2020': '',  # DAGGER (†) → remove
    '\u2021': '',  # DOUBLE DAGGER (‡) → remove
    '\u2023': '-',  # TRIANGULAR BULLET → hyphen
    '\u2024': '.',  # ONE DOT LEADER → period
    '\u2025': '..',  # TWO DOT LEADER → two periods
    '\u2027': '-',  # HYPHENATION POINT → hyphen
    '\u203b': '*',  # REFERENCE MARK → asterisk
    '\u203c': '!!',  # DOUBLE EXCLAMATION MARK → two exclamations
    '\u203d': '?!',  # INTERROBANG → question-exclamation
    '\u2047': '??',  # DOUBLE QUESTION MARK → two questions
    '\u2048': '?!',  # QUESTION EXCLAMATION MARK → question-exclamation
    '\u2049': '!?',  # EXCLAMATION QUESTION MARK → exclamation-question
    '\u204b': '',  # REVERSED PILCROW SIGN → remove
    '\u204c': '',  # BLACK LEFTWARDS BULLET → remove
    '\u204d': '',  # BLACK RIGHTWARDS BULLET → remove
    '\u2052': '%',  # COMMERCIAL MINUS SIGN → percent
    '\u2053': '~',  # SWUNG DASH → tilde
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
    min_for_df = 20  # Need to make this a parameter eventually
    n_unique = len(unique_names)
    # Calculate frequency weights (proportion of total).
    total_items = name_counts.sum()
    freq_weights = (name_counts / total_items).values

    # For very small groups, use edit distance instead of TF-IDF
    if n_unique < min_unique_for_similarity:
        if n_unique == 1:
            # Single name - it's 100% central by definition
            return cupy.ones(1)

        # Calculate edit distance matrix
        similarity_matrix = cupy.zeros((n_unique, n_unique))

        for i in range(n_unique):
            # Get edit distances from name i to all names
            distances = unique_names.str.edit_distance(unique_names.iloc[i])

            # Convert distances to similarities (0 to 1 scale)
            # Using exponential decay: similarity = exp(-distance/max_len)
            max_len = unique_names.str.len().max()
            similarities = cupy.exp(-cupy.asarray(distances.values) / max_len)
            similarity_matrix[i, :] = similarities

        # Weight similarities by frequency
        centrality_score = similarity_matrix @ freq_weights

        logger.debug(
            f'Using edit distance for {n_unique} unique names (< {min_unique_for_similarity})'
        )
        return centrality_score

    exclude_keys = {'min_df', 'max_df'} if n_unique < min_for_df else set()

    vec_params = tfidf_params.model_dump(
        mode='python',
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
        logger.warning('TF-IDF produced no features, falling back to frequency weights')
        return cupy.asarray(freq_weights)

    # Calculate the cosine similarity between all pairs of unique names.
    similarity_matrix = 1 - pairwise_distances(tfidf_matrix, metric='cosine')

    # The core of the centrality score: for each name, sum the similarities
    # to all other names, weighted by how frequent those other names are.
    # A high score means "I am similar to other popular names."
    # The '@' operator performs matrix multiplication.
    centrality_score = similarity_matrix @ freq_weights

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
        logger.warning('get_canonical_name_gpu received an empty series.')
        return ''

    unique_names = name_series.unique()
    if len(unique_names) == 1:
        return unique_names.iloc[0]

    logger.debug(
        f'Finding canonical name from {len(unique_names)} unique candidates (out of {len(name_series)} total).'
    )

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
    logger.debug(f'Calculated final scores for {len(final_scores)} unique names.')

    # Find the index of the highest score and return the corresponding name.
    best_name_index = final_scores.argmax()
    best_name = unique_names.iloc[int(best_name_index)]
    logger.debug(f"Selected '{best_name}' as the canonical name.")

    return best_name


def find_canonical_name(
    all_names_in_group: cudf.Series,
    unique_names_series: cudf.Series,
    unique_name_vectors: cupy.ndarray,
) -> str:
    """
    Selects the best canonical name from a group using pre-computed embeddings.

    This function implements a sophisticated scoring model to identify the most
    representative name within a cluster. It operates on the GPU and is designed
    for use after canonical embeddings have been generated. The final choice is
    based on a combination of three factors, ensuring a robust selection that
    balances popularity, similarity, and descriptiveness.

    Scoring Model
    -------------
    The winning name is the one with the highest final score, calculated as:
    `final_score = (centrality_score + frequency_score) * length_bonus`

    1.  **Centrality Score**:
        This is the most powerful component. It leverages the rich, multi-modal
        "name-only" embeddings to determine how similar a candidate name is to all
        other *frequent* names in the group. A high centrality score means a name
        is at the "center of gravity" of the cluster's name space, making it a
        strong candidate. This is calculated via `calculate_centrality_score`.

    2.  **Frequency Score**:
        A straightforward but strong signal based on the raw frequency of each
        unique name within the group. The most common name is often the correct one,
        and this score directly reflects that.

    3.  **Length Bonus**:
        A simple heuristic that acts as a tie-breaker and favors more descriptive
        names. It provides a small bonus to longer names, helping to select fully
        spelled-out company names over abbreviations (e.g., "Crystal Clean LLC"
        over "Crystal Clean").

    Parameters
    ----------
    all_names_in_group : cudf.Series
        A Series containing ALL name candidates for a single group, including
        duplicates. This is used to accurately calculate `name_counts`.

    unique_names_series : cudf.Series
        A Series containing only the UNIQUE name candidates for the group. The
        final scores are calculated for each name in this series. Must be
        aligned row-for-row with `unique_name_vectors`.

    unique_name_vectors : cupy.ndarray
        A dense CuPy array of shape (N, D) containing the pre-computed, L2-normalized,
        "name-only" canonical embeddings for each name in `unique_names_series`.
        Must be aligned row-for-row with `unique_names_series`.

    Returns
    -------
    str
        The string of the highest-scoring canonical name. Returns an empty string
        if the input `all_names_in_group` is empty.

    See Also
    --------
    entity_resolver.centrality_score.calculate_centrality_score : The function
        used to compute the centrality score component.
    entity_resolver.embedding_orchestrator.EmbeddingOrchestrator : The class
        used to generate the canonical vectors required by this function.
    """
    if all_names_in_group.empty:
        logger.warning('find_canonical_name received an empty series.')
        return ''

    if len(unique_names_series) == 1:
        return unique_names_series.iloc[0]

    logger.debug(
        f'Finding canonical name from {len(unique_names_series)} unique candidates (out of {len(all_names_in_group)} total).'
    )

    # --- Score Calculation ---
    # Ensure the name counts are aligned with the unique names series.
    name_counts = all_names_in_group.value_counts().reindex(unique_names_series).fillna(0)

    # 1. Calculate the centrality score using the canonical vectors.
    centrality_score = _calculate_centrality_score_with_vectors(unique_name_vectors, name_counts)

    # 2. Get the direct frequency score (normalized by total count).
    total_count = name_counts.sum()
    if total_count == 0:
        # Avoid division by zero if counts are all zero for some reason
        frequency_score = cupy.zeros(len(name_counts), dtype=cupy.float32)
    else:
        frequency_score = (name_counts.values / total_count).astype(cupy.float32)

    # 3. Calculate the length bonus.
    length_bonus = _calculate_length_bonus(unique_names_series)

    # --- Combine Scores ---
    # The base score combines centrality and raw frequency. A name can score
    # well by being frequent itself, or by being very similar to other
    # frequent names.
    base_score = centrality_score + frequency_score

    # The final score is modulated by the length bonus. This helps break ties
    # and promotes more descriptive names.
    final_scores = base_score * length_bonus
    logger.debug(f'Calculated final scores for {len(final_scores)} unique names.')

    # Find the index of the highest score and return the corresponding name.
    best_name_index = int(cupy.argmax(final_scores))
    best_name = unique_names_series.iloc[best_name_index]
    logger.debug(
        f"Selected '{best_name}' as the canonical name with score {final_scores[best_name_index]:.4f}."
    )

    return best_name


def _calculate_centrality_score_with_vectors(
    aligned_name_vectors: cupy.ndarray, name_counts: cudf.Series
) -> cupy.ndarray:
    """
    Calculates a centrality score for each unique name using canonical embeddings.

    This score is high for names that are both frequent and highly similar to
    other frequent names within a group. This version leverages
    pre-computed, canonical "name-only" embeddings for a faster, more accurate,
    and more consistent similarity measurement.

    The logic is unified and efficient:
    1.  It computes cosine similarity directly from the L2-normalized input vectors.
        This is a simple and fast matrix multiplication (`vectors @ vectors.T`).
    2.  It weights the resulting similarity matrix by the frequency of each name.
    3.  The final score reflects how similar a name is to other popular names in
        its group, based on the rich, multi-modal understanding captured in the
        canonical embeddings.

    Parameters
    ----------
    aligned_name_vectors : cupy.ndarray
        A dense CuPy array of shape (N, D) containing the "name-only" canonical
        embeddings. These vectors MUST be L2-normalized and must be perfectly
        aligned (row-for-row) with the `name_counts` Series.

    name_counts : cudf.Series
        A Series of length N containing the frequency count for each unique name,
        aligned with the `aligned_name_vectors`.

    Returns
    -------
    cupy.ndarray
        A 1D CuPy array of centrality scores of length N, one for each unique name.

    Usage Example (within your pipeline)
    -------------------------------------
    # Assume 'cluster_group' is a DataFrame for a single cluster, and
    # 'orchestrator' is your fitted EmbeddingOrchestrator instance.

    # Get the embeddings aligned to this specific cluster group
    aligned_embeddings = orchestrator.get_aligned_embeddings(cluster_group)
    name_vectors = aligned_embeddings['name']

    # Get the name counts for this group
    counts = cluster_group['name'].value_counts()

    # Calculate the score
    scores = calculate_centrality_score(name_vectors, counts)
    """
    n_unique = aligned_name_vectors.shape[0]
    logger.debug(f'Calculating centrality for {n_unique} unique names using canonical embeddings.')

    if n_unique == 0:
        return cupy.array([], dtype=cupy.float32)
    if n_unique == 1:
        # A single item is, by definition, 100% central to its group.
        return cupy.ones(1, dtype=cupy.float32)

    # 1. Calculate frequency weights (proportion of total items in the group).
    # These weights determine the "importance" of each name in the calculation.
    total_items = name_counts.sum()
    freq_weights = (name_counts.values / total_items).astype(cupy.float32)

    # 2. Calculate the cosine similarity matrix.
    # Because the input vectors are already L2-normalized, cosine similarity is
    # simply the dot product of the matrix with its transpose. This is extremely
    # fast on the GPU.
    # The resulting matrix has shape (N, N). similarity_matrix[i, j] is the
    # similarity between name i and name j.
    similarity_matrix = aligned_name_vectors @ aligned_name_vectors.T

    # 3. Calculate the centrality score.
    # This is the core of the algorithm. For each name (each row in the similarity
    # matrix), we calculate a weighted sum of its similarities to all other names.
    # The weight is the frequency of the other names. A high score means "I am
    # highly similar to other names that are very common in this group."
    # The '@' operator performs matrix-vector multiplication.
    centrality_score = similarity_matrix @ freq_weights

    logger.debug(f'Centrality score calculation complete for {n_unique} names.')
    return centrality_score


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
        return ''

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
    pandas_series = input_series.fillna('').astype('str').to_pandas()

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
