# entity_resolver/utils/embedding_streams.py
"""
Text Stream Preparation for Multi-Modal Entity Resolution Embeddings

This module prepares optimized text inputs for the multi-stream vectorization pipeline
used in entity resolution and deduplication. It transforms raw entity data (names and
addresses) into specialized text representations that feed different embedding models,
each capturing complementary signals for matching.

Overview
--------
Entity resolution requires matching records that may have typos, phonetic variations,
semantic equivalence, or structural differences. A single embedding approach can't
capture all these signals effectively. This module implements a multi-stream strategy:

1. **TF-IDF Stream**: Character n-gram analysis for exact lexical matching and typo detection
2. **Semantic Stream**: Transformer-based embeddings for meaning and context understanding  
3. **Phonetic Stream**: Sound-alike encoding for pronunciation-based matching (names only)

Each stream receives specially prepared text optimized for its vectorization approach.
After vectorization, the streams are normalized, balanced, concatenated, and reduced to
create canonical embedding matrices that preserve all three signal types.

Three Embedding Contexts
-------------------------
The module prepares text for three distinct embedding contexts:

**Combined Context** (Primary)
    Fuses entity name and address information. Used for general entity resolution,
    clustering, and initial duplicate detection. This is the main embedding space
    where most matching and grouping operations occur.

**Name-Only Context**
    Contains only entity name information. Used for name-specific similarity
    calculations, name-focused graph operations, and scenarios where location
    should not influence matching (e.g., finding all variations of "ACME Corp"
    regardless of location).

**Address-Only Context**  
    Contains only address information. Used for location-specific similarity,
    address-focused graph operations, and geographic clustering. Note: This
    context excludes the phonetic stream since phonetic encoding is meaningless
    for street numbers and ZIP codes.

Text Preparation Strategies
----------------------------
Different vectorization approaches require different text preparation:

**TF-IDF (Character N-grams)**
    - Uses repetition to weight important terms (higher term frequency = stronger signal)
    - Example: "acme acme acme 123 main st chicago il" 
    - Captures exact character sequences, making it excellent for typo detection

**Semantic (Transformer Models)**
    - Uses natural, well-structured language with contextual information
    - Example: "entity: acme corporation main street chicago illinois"
    - Understands meaning and semantic relationships between terms

**Phonetic (Sound-alike Encoding)**
    - Uses raw text that will be converted to phonetic codes (e.g., Double Metaphone)
    - Example: "acme" -> "AKM" phonetic code
    - Matches names that sound identical but spell differently (Smith/Smythe)

Pipeline Position
-----------------
This module represents the FIRST stage in the embedding pipeline:

    Raw Data → [TEXT PREPARATION] → Vectorization → SVD → Normalization → 
    Balance & Concatenate → Final SVD → Canonical Embeddings → UMAP → Clustering

The output text streams from this module feed directly into vectorizers that create
sparse (TF-IDF, phonetic) or dense (semantic) feature matrices.

Usage Example
-------------
```python
from entity_resolver.utils.embedding_streams import (
    prepare_text_streams,
    AllTextStreams,
    TextStreamSet
)
import cudf

# Prepare input DataFrame with required columns
gdf = cudf.DataFrame({
    'normalized_text': ['acme corp', 'contoso ltd', 'fabrikam inc'],
    'addr_street_number': ['123', '456', '789'],
    'addr_street_name': ['main', 'oak', 'elm'],
    'addr_city': ['chicago', 'boston', 'seattle'],
    'addr_state': ['illinois', 'massachusetts', 'washington'],
    'addr_zip': ['60601', '02101', '98101']
})

# Prepare all text streams
streams = prepare_text_streams(
    gdf=gdf,
    use_address_in_encoding=True
)

# Access specific streams for vectorization
combined_tfidf = streams.combined.tfidf      # For TF-IDF vectorizer
combined_semantic = streams.combined.semantic # For semantic model
combined_phonetic = streams.combined.phonetic # For phonetic encoder

name_only_tfidf = streams.name.tfidf         # Name-only TF-IDF
address_semantic = streams.address.semantic   # Address-only semantic
# Note: streams.address.phonetic is None (no phonetic for addresses)
```
Data Requirements
-----------------
Input DataFrame must contain:

    - 'normalized_text': Pre-normalized entity name (required)
    - 'addr_street_number': Numeric street address (optional)
    - 'addr_street_name': Street name (optional)
    - 'addr_city': City name (optional)
    - 'addr_state': Full state name, lowercase (optional)
    - 'addr_zip': 5-digit ZIP code (optional)

Missing address columns are handled gracefully with empty string fallbacks.

Notes
-----
All input text should already be lowercase and normalized before this stage
State abbreviations should already be expanded to full names
NFKC Unicode normalization is applied to all output streams
Address-only context intentionally excludes phonetic stream (set to None)
Downstream vectorizers must handle None phonetic streams appropriately

See Also
--------
entity_resolver.vectorizer : Vectorization logic that consumes these text streams
entity_resolver.utils.vector : Vector normalization and balancing operations
"""

import cudf
from typing import Optional
import logging
from dataclasses import dataclass

from .text import nfkc_normalize_series

# Set up a logger for this module
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _repeat_series_with_space(series: cudf.Series, repetitions: int) -> cudf.Series:
    """
    Repeat each string in a cuDF Series multiple times, separated by single spaces.
    
    This function provides a robust way to repeat strings in cuDF Series, avoiding
    potential issues with the * operator on Series objects. Each element is repeated
    the specified number of times with space delimiters.
    
    Parameters
    ----------
    series : cudf.Series
        Series of strings to repeat
    repetitions : int
        Number of times each string should appear in the output
        
    Returns
    -------
    cudf.Series
        Series where each element is repeated `repetitions` times with spaces
        
    Examples
    --------
    >>> s = cudf.Series(['acme', 'contoso'])
    >>> _repeat_series_with_space(s, 3)
    0      acme acme acme
    1    contoso contoso contoso
    dtype: object
    
    Notes
    -----
    If repetitions <= 1, returns the original series unchanged.
    cuDF can be finicky with Series * int operations, so this explicit
    concatenation approach is more reliable.
    """
    if repetitions <= 1:
        return series
    
    result = series.copy()
    for _ in range(repetitions - 1):
        result = result + ' ' + series
    
    return result

# ---------------------------------------------------------------------------
# Data containers with dot-notation access
# ---------------------------------------------------------------------------

@dataclass
class TextStreamSet:
    """
    Holds the prepared text series for a specific vectorization context.
    
    Each context (combined, name-only, address-only) produces its own embedding matrix
    by processing these three streams through different vectorization pipelines:
    - TF-IDF: Character n-gram sparse vectors → SVD reduction → dense vectors
    - Semantic: Sentence transformer (e.g., BGE) → dense vectors directly
    - Phonetic: Phonetic encoding (e.g., Double Metaphone) → sparse vectors → SVD reduction → dense vectors
    
    After individual stream processing, all three dense streams are L2-normalized,
    balanced by energy, concatenated, L2-normalized again, and optionally reduced
    via SVD to create the final canonical embedding matrix for that context.

    Attributes
    ----------
    tfidf : cudf.Series
        Text prepared for character-level TF-IDF vectorization. Typically includes
        weighted repetition of key terms to boost their importance in the sparse
        feature space. Optimized for capturing exact lexical patterns and typos.
        
    semantic : cudf.Series
        Text prepared for semantic transformer models (e.g., BAAI/bge-base-en-v1.5).
        Should be natural-language-like and well-structured. Semantic models understand
        meaning and context, so include relevant contextual information but avoid
        artificial repetition.
        
    phonetic : Optional[cudf.Series]
        Text prepared for phonetic encoding pipelines (e.g., Double Metaphone).
        This stream is ONLY meaningful for entity names where pronunciation matters
        (e.g., "Smith" vs "Smythe"). For address-only contexts, this should be None
        because phonetic encoding of street numbers, ZIP codes, and street suffixes
        provides no matching value.
    """
    tfidf: cudf.Series
    semantic: cudf.Series
    phonetic: Optional[cudf.Series] = None


@dataclass
class AllTextStreams:
    """
    Top-level container organizing all text stream sets by context.
    
    Three contexts are maintained:
    - combined: Entity name + address information fused (primary matching context)
    - name: Entity name only (for name-specific similarity and graph operations)
    - address: Address only (for address-specific similarity and graph operations)
    
    Each context produces its own canonical embedding matrix. Use the appropriate
    matrix for downstream operations:
    - Use combined embeddings for general entity resolution and clustering
    - Use name embeddings for name-specific similarity (e.g., "find similar company names")
    - Use address embeddings for address-specific similarity (e.g., "find nearby addresses")

    Access patterns
    ---------------
    streams.combined.tfidf     # Combined context TF-IDF text
    streams.combined.semantic  # Combined context semantic text
    streams.combined.phonetic  # Combined context phonetic text
    
    streams.name.tfidf         # Name-only context TF-IDF text
    streams.name.semantic      # Name-only context semantic text  
    streams.name.phonetic      # Name-only context phonetic text
    
    streams.address.tfidf      # Address-only context TF-IDF text
    streams.address.semantic   # Address-only context semantic text
    streams.address.phonetic   # Address-only phonetic (will be None)
    """
    combined: TextStreamSet
    name: TextStreamSet
    address: TextStreamSet


# ---------------------------------------------------------------------------
# Main method: prepares combined, name-only, and address-only text streams
# ---------------------------------------------------------------------------

def prepare_text_streams(
    gdf: cudf.DataFrame,
    use_address_in_encoding: bool = True
) -> AllTextStreams:
    """
    Prepares three distinct sets of text streams (combined, name-only, address-only)
    for downstream vectorization and embedding generation. Each stream set contains
    text optimized for different vectorization approaches (TF-IDF, semantic, phonetic).
    
    This method is the first stage in the embedding pipeline. The output streams are
    then processed through vectorizers to create dense embedding matrices that preserve
    the complementary signals from each stream.

    Design Principles by Context
    -----------------------------
    
    **COMBINED Context** (entity name + address):
    Primary context for entity resolution. Fuses name and location signals.
    
    - TF-IDF: Name repeated 3x (to boost name importance in character n-grams) + 
              full address (street_number, street_name, city, state, zip).
              Rationale: Character n-grams excel at catching typos and variations.
              Triple repetition ensures name has ~equal weight to full address.
              
    - Semantic: Name + semantic address parts (street_name, city, state only).
              Rationale: Semantic models understand spatial context. Exclude high-variance
              numerics (street_number, zip) which add noise to semantic space. The model
              learns that "Main St, Chicago, IL" is semantically meaningful.
              
    - Phonetic: Name only (no address).
              Rationale: Phonetic matching (Double Metaphone) is designed for pronounceable
              words. "smith"="smythe" phonetically. Address numbers/codes are not phonetic.
    
    
    **NAME-ONLY Context** (entity name):
    Used for name-specific similarity calculations and name-focused graph operations.
    
    - TF-IDF: Name repeated 2-3x to increase character density for n-gram analyzer.
              Rationale: Pure lexical matching. More repetitions = higher term frequencies
              for the name's character sequences, improving matching sensitivity.
              
    - Semantic: Name only with minimal stabilizer prefix.
              Rationale: Let the semantic model focus purely on name semantics without
              location context. Stabilizer prefix ("name: ") helps reduce embedding drift.
              
    - Phonetic: Name only.
              Rationale: Core use case for phonetic encoding. Captures pronunciation
              equivalence classes (e.g., "cathy"="kathy", "philip"="phillip").
    
    
    **ADDRESS-ONLY Context** (address):
    Used for address-specific similarity calculations and location-focused graph operations.
    
    - TF-IDF: Full address with street_name repeated 2x.
              Format: "street_number street_name street_name city state zip"
              Rationale: Street name is the most distinguishing address component and
              most prone to variations/typos. Weighting it 2x improves matching while
              keeping other components present for precision.
              
    - Semantic: Full natural address in USPS-like format with punctuation.
              Format: "address: 123 main st, chicago, illinois 60601"
              Rationale: Semantic models can handle numbers and understand complete
              addresses as meaningful geographic entities. Natural punctuation helps.
              
    - Phonetic: None (explicitly excluded).
              Rationale: Phonetic encoding is meaningless for addresses. "123" doesn't
              need phonetic matching. Street numbers, ZIP codes, and numeric components
              would just add noise. Only the name context benefits from phonetic signals.
    
    
    Text Preparation Best Practices
    --------------------------------
    1. **Repetition Strategy**: Used in TF-IDF to weight important terms. More repetitions
       = higher term frequency = stronger signal in the sparse feature space.
       
    2. **Stabilizer Prefixes**: Short prefixes ("entity:", "name:", "address:") added to
       semantic text to reduce embedding drift and give the model context about what
       type of entity is being encoded.
       
    3. **Normalization**: All output series undergo NFKC Unicode normalization to ensure
       consistent character representation across the entire pipeline.
       
    4. **Null Safety**: All address columns are safely fetched with fallback to empty
       strings.
       
    5. **Stream Independence**: Each context's streams are independently constructed.
       This allows flexible use (e.g., name-only matching, address-only matching) without
       reprocessing the entire dataset.

    Parameters
    ----------
    gdf : cudf.DataFrame
        Input DataFrame containing:
        - 'normalized_text' : str
            Pre-normalized entity name (required)
        - 'addr_street_number' : str, optional
            Numeric street address (e.g., "123", "4500")
        - 'addr_street_name' : str, optional  
            Street name (e.g., "main", "oak")
        - 'addr_city' : str, optional
            City name (e.g., "chicago", "naperville")
        - 'addr_state' : str, optional
            State name (e.g., "illinois", "california")
        - 'addr_zip' : str, optional
            5-digit ZIP code (e.g., "60540")
        - 'addr_normalized_key' : str, optional
            Pre-computed address key for logging purposes only

    Returns
    -------
    AllTextStreams
        Structured container with dot-notation access to all prepared text streams:
        - .combined : TextStreamSet (name + address streams)
        - .name : TextStreamSet (name-only streams)
        - .address : TextStreamSet (address-only streams, phonetic=None)
        
    Raises
    ------
    KeyError
        If required 'normalized_text' column is missing from input DataFrame

    Notes
    -----
    - The parameter 'use_address_in_encoding' controls whether address information
      is included in the COMBINED context. If False, combined context contains only name
      information (but name and address contexts are still separately created).
      
    - Address-only phonetic stream is intentionally set to None. Downstream vectorizer
      logic must handle None phonetic streams by excluding that stream from concatenation
      and adjusting stream balancing weights accordingly.
      
    - All three contexts will produce separate canonical embedding matrices after
      vectorization. Store all three matrices for flexible downstream operations.
    """

    logger.debug("Preparing optimized text streams for all vectorization contexts.")
    
    # Stabilizer prefixes help semantic models understand entity type and reduce drift
    SEMANTIC_COMBINED_PREFIX = 'entity: '
    SEMANTIC_NAME_PREFIX = 'name: '
    SEMANTIC_ADDRESS_PREFIX = 'address: '
    
    # Repetition factors for TF-IDF weighting
    NAME_REPETITIONS_COMBINED = 3  # Name appears 3x in combined TF-IDF
    NAME_REPETITIONS_NAME_ONLY = 3  # Name appears 3x in name-only TF-IDF  
    STREET_NAME_REPETITIONS_ADDRESS = 2  # Street name appears 2x in address TF-IDF

    # -----------------------------------------------------------------------
    # 0) Load core entity name column
    # -----------------------------------------------------------------------
    if 'normalized_text' not in gdf.columns:
        raise KeyError(
            "Required column 'normalized_text' not found in input DataFrame. "
            "This column must contain the pre-normalized entity name text."
        )

    entity_name_text = gdf['normalized_text'].fillna('').astype(str)
    logger.debug(f"Loaded entity names from 'normalized_text' column: {len(entity_name_text):,} records")

    # -----------------------------------------------------------------------
    # 1) Load address components safely (fabricate empty series if missing)
    # -----------------------------------------------------------------------
    # Configuration gate: determines if address info should be included in COMBINED context    
    if not use_address_in_encoding:
        logger.info(
            "Address usage disabled for combined context (use_address_in_encoding=False). "
            "Combined context will contain name-only information. "
            "Name and address contexts will still be created separately."
        )

    def _safe_column_fetch(column_name: str) -> cudf.Series:
        """
        Safely fetch a column from the DataFrame, returning an empty string series
        if the column doesn't exist. Maintains index alignment with input DataFrame.
        """
        if column_name in gdf.columns:
            return gdf[column_name].fillna('').astype(str)
        return cudf.Series([''] * len(gdf), dtype='str', index=gdf.index)

    street_number_series = _safe_column_fetch('addr_street_number')
    street_name_series = _safe_column_fetch('addr_street_name')
    city_series = _safe_column_fetch('addr_city')
    state_series = _safe_column_fetch('addr_state')
    zip_code_series = _safe_column_fetch('addr_zip')

    # Log how many records actually have address information
    if 'addr_normalized_key' in gdf.columns:
        records_with_address = (
            gdf['addr_normalized_key'].notna() & 
            (gdf['addr_normalized_key'] != '')
        ).sum()
        logger.debug(
            f"Address coverage: {records_with_address:,}/{len(gdf):,} records "
            f"({100 * records_with_address / len(gdf):.1f}%) have non-empty addresses"
        )

    # -----------------------------------------------------------------------
    # 2) Build COMBINED context streams (name + address)
    # -----------------------------------------------------------------------
    logger.debug("Building COMBINED context streams (name + address)...")
    
    if use_address_in_encoding:
        # Semantic address: exclude high-variance numerics (street_number, zip)
        # Keep only street_name, city, state for semantic understanding
        semantic_address_part = (
            (street_name_series + ' ') +
            (city_series + ' ') +
            state_series
        ).str.strip()

        # TF-IDF address: include ALL components for exhaustive character n-gram coverage
        tfidf_address_part = (
            (street_number_series + ' ') +
            (street_name_series + ' ') +
            (city_series + ' ') +
            (state_series + ' ') +
            zip_code_series
        ).str.strip()
    else:
        # Config disabled address usage: use empty strings
        semantic_address_part = cudf.Series([''] * len(gdf), dtype='str', index=gdf.index)
        tfidf_address_part = cudf.Series([''] * len(gdf), dtype='str', index=gdf.index)

    # Phonetic (combined): name only
    # Rationale: Phonetic encoding is only meaningful for pronounceable entity names
    phonetic_text_combined = entity_name_text.copy()

    # Semantic (combined): "entity: [name] [street_name city state]"
    # Rationale: Natural language format with location context, excluding noisy numerics
    semantic_text_combined = (
        SEMANTIC_COMBINED_PREFIX +
        entity_name_text + ' ' +
        semantic_address_part
    ).str.strip()

    # TF-IDF (combined): "[name name name] [full address]"
    # Rationale: Triple name repetition balances weight against full address in char n-grams
    weighted_name_combined = _repeat_series_with_space(entity_name_text, NAME_REPETITIONS_COMBINED)
    tfidf_text_combined = (
        weighted_name_combined + ' ' 
        + tfidf_address_part
    ).str.strip()

    # Apply NFKC normalization to ensure consistent Unicode representation
    phonetic_text_combined = nfkc_normalize_series(phonetic_text_combined)
    semantic_text_combined = nfkc_normalize_series(semantic_text_combined)
    tfidf_text_combined = nfkc_normalize_series(tfidf_text_combined)

    logger.info(
        "Created COMBINED streams: "
        f"phonetic(name only), "
        f"semantic(name + address context), "
        f"tfidf(name×{NAME_REPETITIONS_COMBINED} + full address)"
    )

    # -----------------------------------------------------------------------
    # 3) Build NAME-ONLY context streams
    # -----------------------------------------------------------------------
    logger.debug("Building NAME-ONLY context streams...")
    
    # TF-IDF (name-only): repeat name to increase character density for n-gram matching
    # Rationale: Higher term frequency = stronger signal in sparse feature space
    tfidf_text_name_only = _repeat_series_with_space(
        entity_name_text, 
        NAME_REPETITIONS_NAME_ONLY
    ).str.strip()

    # Phonetic (name-only): raw name text for phonetic encoding pipeline
    # Rationale: Core phonetic use case - match names by pronunciation
    phonetic_text_name_only = entity_name_text.copy()

    # Semantic (name-only): name with minimal stabilizer prefix
    # Rationale: Focus semantic model purely on name meaning, stabilizer reduces drift
    semantic_text_name_only = (
        SEMANTIC_NAME_PREFIX +
        entity_name_text
    ).str.strip()

    # Apply NFKC normalization
    tfidf_text_name_only = nfkc_normalize_series(tfidf_text_name_only)
    phonetic_text_name_only = nfkc_normalize_series(phonetic_text_name_only)
    semantic_text_name_only = nfkc_normalize_series(semantic_text_name_only)

    logger.info(
        "Created NAME-ONLY streams: "
        f"phonetic(name), "
        f"semantic(name), "
        f"tfidf(name×{NAME_REPETITIONS_NAME_ONLY})"
    )

    # -----------------------------------------------------------------------
    # 4) Build ADDRESS-ONLY context streams
    # -----------------------------------------------------------------------
    logger.debug("Building ADDRESS-ONLY context streams...")
    
    # TF-IDF (address-only): full address with street_name weighted 2x
    # Format: "street_number street_name street_name city state zip"
    # Rationale: Street name is most distinguishing and variation-prone component
    weighted_street_name = _repeat_series_with_space(street_name_series, STREET_NAME_REPETITIONS_ADDRESS)
    tfidf_text_address_only = (
        (street_number_series + ' ') +
        (weighted_street_name + ' ') +
        (city_series + ' ') +
        (state_series + ' ') +
        zip_code_series
    ).str.replace(r'\s+', ' ', regex=True).str.strip()

    # Semantic (address-only): natural USPS-style format with punctuation
    # Format: "address: 123 main st, chicago, illinois 60601"
    # Rationale: Semantic models understand full addresses as geographic entities
    semantic_text_address_only = (
        SEMANTIC_ADDRESS_PREFIX +
        (street_number_series + ' ') +
        (street_name_series + ', ') +
        (city_series + ', ') +
        (state_series + ' ') +
        zip_code_series
    ).str.replace(r'\s+', ' ', regex=True).str.replace(' ,', ',', regex=False).str.strip()

    # Phonetic (address-only): EXPLICITLY None
    # Rationale: Phonetic encoding provides no value for addresses. Street numbers,
    # ZIP codes, and directionals (N/S/E/W) are not pronounceable words. Including
    # phonetic signals for addresses would only add noise to the embedding space.
    # Downstream vectorizer must handle None by excluding this stream from concatenation.
    phonetic_text_address_only = None
    
    # Apply NFKC normalization to non-None streams
    tfidf_text_address_only = nfkc_normalize_series(tfidf_text_address_only)
    semantic_text_address_only = nfkc_normalize_series(semantic_text_address_only)

    logger.info(
        "Created ADDRESS-ONLY streams: "
        f"phonetic(EXCLUDED - None), "
        f"semantic(full natural address), "
        f"tfidf(street_name×{STREET_NAME_REPETITIONS_ADDRESS} + other components)"
    )
    logger.debug(
        "Note: Address-only context has no phonetic stream. Downstream vectorizer "
        "must detect None phonetic and exclude it from stream concatenation and balancing."
    )

    # -----------------------------------------------------------------------
    # 5) Package all streams into structured container
    # -----------------------------------------------------------------------
    combined_stream_set = TextStreamSet(
        tfidf=tfidf_text_combined,
        semantic=semantic_text_combined,
        phonetic=phonetic_text_combined
    )

    name_only_stream_set = TextStreamSet(
        tfidf=tfidf_text_name_only,
        semantic=semantic_text_name_only,
        phonetic=phonetic_text_name_only
    )

    address_only_stream_set = TextStreamSet(
        tfidf=tfidf_text_address_only,
        semantic=semantic_text_address_only,
        phonetic=phonetic_text_address_only  # Explicitly None - no phonetic stream for addresses
    )

    all_streams_container = AllTextStreams(
        combined=combined_stream_set,
        name=name_only_stream_set,
        address=address_only_stream_set
    )

    # -----------------------------------------------------------------------
    # 6) Validation and summary logging
    # -----------------------------------------------------------------------
    logger.debug("Validating stream set dimensions...")
    
    # Verify all series have correct length (except None)
    expected_length = len(gdf)
    
    def _validate_series_length(series: Optional[cudf.Series], context: str, stream: str):
        if series is not None and len(series) != expected_length:
            raise ValueError(
                f"{context}.{stream} series has incorrect length: "
                f"got {len(series)}, expected {expected_length}"
            )
    
    _validate_series_length(combined_stream_set.tfidf, "combined", "tfidf")
    _validate_series_length(combined_stream_set.semantic, "combined", "semantic")
    _validate_series_length(combined_stream_set.phonetic, "combined", "phonetic")
    
    _validate_series_length(name_only_stream_set.tfidf, "name", "tfidf")
    _validate_series_length(name_only_stream_set.semantic, "name", "semantic")
    _validate_series_length(name_only_stream_set.phonetic, "name", "phonetic")
    
    _validate_series_length(address_only_stream_set.tfidf, "address", "tfidf")
    _validate_series_length(address_only_stream_set.semantic, "address", "semantic")
    # phonetic is None for address - no validation needed
    
    logger.info(
        f"Text stream preparation complete. All contexts validated with {expected_length:,} records."
    )
    logger.debug(
        "Stream summary: "
        f"combined(tfidf={len(combined_stream_set.tfidf):,}, "
        f"semantic={len(combined_stream_set.semantic):,}, "
        f"phonetic={len(combined_stream_set.phonetic):,}); "
        f"name(tfidf={len(name_only_stream_set.tfidf):,}, "
        f"semantic={len(name_only_stream_set.semantic):,}, "
        f"phonetic={len(name_only_stream_set.phonetic):,}); "
        f"address(tfidf={len(address_only_stream_set.tfidf):,}, "
        f"semantic={len(address_only_stream_set.semantic):,}, "
        f"phonetic=None)"
    )

    return all_streams_container