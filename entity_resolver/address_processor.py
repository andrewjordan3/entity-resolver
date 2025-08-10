# entity_resolver/address_processor.py
"""
Address processing pipeline for parsing, cleaning, and consolidating addresses.

This module defines the `AddressProcessor` class which is responsible for:
  1) Combining multiple raw address columns into a single string (GPU where possible)
  2) Parsing the combined string to structured components using libpostal (CPU)
  3) Normalizing and generating an address key suitable for similarity work (GPU)
  4) (Training only) Consolidating near-duplicate addresses into canonical keys (GPU)

Design notes
------------
- Stateless utilities (tokenization, normalization, graph ops) live in `.utils`.
  They remain functions because they hold no mutable state.
- This class *does* hold config objects and orchestrates steps end‑to‑end, so
  it stays a class rather than a collection of free functions.
- The consolidation logic is kept in private methods here for cohesion, but if it
  grows (e.g., additional scorers/heuristics), consider extracting to
  `address_consolidator.py` with a small strategy interface.

Performance notes
-----------------
- cuDF is used for columnar ops. Libpostal parsing occurs on CPU; to reduce
  transfer overhead, we convert only the necessary Series to pandas.
- Vectorized string ops (`Series.str.*`) are preferred to Python loops.

Logging
-------
- `logger.info` describes major pipeline stages; `logger.debug` emits sizes and
  thresholds to aid troubleshooting without overly chatty logs by default.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, List

import cudf
import pandas as pd

# --- Local Package Imports ---
from .config import ValidationConfig, ColumnConfig, VectorizerConfig
from . import utils

# Set up a logger for this module
logger = logging.getLogger(__name__)


class AddressProcessor:
    """
    Orchestrates the parsing, cleaning, and consolidation of address data.

    Parameters
    ----------
    validation_config : ValidationConfig
        Thresholds and validation settings (e.g., fuzzy match ratio).
    column_config : ColumnConfig
        Column names for input address parts.
    vectorizer_config : VectorizerConfig
        Parameters for TF‑IDF and nearest‑neighbor similarity.
    """

    def __init__(
        self,
        validation_config: ValidationConfig,
        column_config: ColumnConfig,
        vectorizer_config: VectorizerConfig,
    ) -> None:
        self.validation_config = validation_config
        self.column_config = column_config
        self.vectorizer_config = vectorizer_config

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def process_addresses(self, gdf: cudf.DataFrame, is_training: bool) -> cudf.DataFrame:
        """Run the full address processing pipeline.

        Steps
        -----
        1. Combine configured address columns into a single string column
        2. Parse addresses on CPU using libpostal (via utils.safe_parse_address)
        3. Append parsed components to the GPU DataFrame and compute a normalized key
        4. Optionally consolidate near-duplicates into canonical keys (training only)

        Parameters
        ----------
        gdf : cudf.DataFrame
            Input GPU DataFrame containing the raw address columns specified by
            ``self.column_config.address_cols``.
        is_training : bool
            If True, perform address consolidation based on similarity.

        Returns
        -------
        cudf.DataFrame
            A new DataFrame with parsed components and (optionally) consolidated
            normalized keys.
        """
        if gdf is None or len(gdf) == 0:
            logger.info("Received empty DataFrame; returning as-is.")
            return gdf

        logger.info("Starting address processing pipeline…")
        logger.debug(f"Input shape: {(len(gdf), len(gdf.columns))}")

        combined_address_series = self._combine_address_columns(gdf)
        parsed_addresses_df = self._parse_address_series(combined_address_series)
        gdf_with_parsed = self._add_parsed_columns_to_gdf(gdf, parsed_addresses_df)

        if is_training:
            logger.info("Training mode enabled: consolidating similar addresses.")
            gdf_with_parsed = self._consolidate_similar_addresses(gdf_with_parsed)

        logger.info("Address processing complete.")
        logger.debug(f"Output shape: {(len(gdf_with_parsed), len(gdf_with_parsed.columns))}")
        return gdf_with_parsed

    def split_canonical_address(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split an existing ``canonical_address`` column into component columns.

        This is designed for final CPU output when a single string column needs to
        be exploded into parts for export or human review.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing a ``canonical_address`` column.

        Returns
        -------
        pandas.DataFrame
            Same DataFrame with additional columns: ``canonical_street_number``,
            ``canonical_street_name``, ``canonical_city``, ``canonical_state``,
            ``canonical_zip``. If the source column is missing, returns the input
            unchanged.
        """
        logger.info("Splitting canonical address into components for final output…")
        if "canonical_address" not in df.columns:
            logger.warning("'canonical_address' column not found; skipping split.")
            return df

        # Parse with the same libpostal-backed utility for consistency.
        parsed_dicts = df["canonical_address"].apply(utils.safe_parse_address)
        parsed_components_df = pd.json_normalize(parsed_dicts)

        addr_cols_map: Dict[str, str] = {
            "canonical_street_number": "address_line_1.street_number",
            "canonical_street_name": "address_line_1.street_name",
            "canonical_city": "city",
            "canonical_state": "state",
            "canonical_zip": "postal_code",
        }

        for out_col, src_col in addr_cols_map.items():
            df[out_col] = parsed_components_df[src_col] if src_col in parsed_components_df.columns else ""

        logger.debug(f"Added canonical component columns: {list(addr_cols_map.keys())}")
        return df

    # ---------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------
    def _combine_address_columns(self, gdf: cudf.DataFrame) -> cudf.Series:
        """Combine configured address columns into a single space‑separated Series.

        - Missing columns are logged and ignored.
        - ``None``/NaN values are treated as empty strings.
        - Excess whitespace is normalized and stripped.
        """
        address_cols: List[str] = list(self.column_config.address_cols or [])
        if not address_cols:
            logger.warning("No address columns specified in ColumnConfig; returning empty series.")
            return cudf.Series([""] * len(gdf), index=gdf.index)

        # Validate presence and collect parts. Missing columns are skipped with a warning.
        available_parts: List[cudf.Series] = []
        missing_cols: List[str] = []
        for col in address_cols:
            if col in gdf.columns:
                available_parts.append(gdf[col].fillna("").astype("str"))
            else:
                missing_cols.append(col)
        if missing_cols:
            logger.warning(f"Missing address columns (skipped): {missing_cols}")
        if not available_parts:
            logger.warning("No valid address columns available; returning empty series.")
            return cudf.Series([""] * len(gdf), index=gdf.index)

        # Vectorized GPU concat via str.cat is more efficient than Python loops.
        if len(available_parts) == 1:
            combined = available_parts[0]
        else:
            combined = available_parts[0].str.cat(others=available_parts[1:], sep=" ")

        combined = combined.str.normalize_spaces().str.strip()
        logger.debug(f"Combined address column constructed. Example (first 3): {combined.head(3).to_pandas().tolist()}")
        return combined

    def _parse_address_series(self, address_series: cudf.Series) -> cudf.DataFrame:
        """Parse addresses with libpostal on CPU and return a cuDF DataFrame.

        Notes
        -----
        - Conversion to pandas is scoped to the address Series only to minimize
          host/device transfer.
        - Failures from ``utils.safe_parse_address`` yield empty dicts; we log a
          count for observability and continue.
        """
        total = int(len(address_series))
        logger.info(f"Parsing {total} addresses on CPU using libpostal…")

        address_series_pd = address_series.to_pandas()
        parsed_dicts = address_series_pd.apply(utils.safe_parse_address)

        failures = sum(1 for d in parsed_dicts if not d)
        if failures:
            logger.warning(f"Could not parse {failures} of {total} addresses.")

        parsed_pdf = pd.json_normalize(parsed_dicts)
        parsed_gdf = cudf.DataFrame.from_pandas(parsed_pdf)
        logger.debug(f"Parsed components shape (rows, cols): {(len(parsed_gdf), len(parsed_gdf.columns))}")
        return parsed_gdf

    def _add_parsed_columns_to_gdf(self, gdf: cudf.DataFrame, parsed_df: cudf.DataFrame) -> cudf.DataFrame:
        """Merge parsed components into the original DataFrame and create a key.

        The resulting columns are:
            - ``addr_street_number``
            - ``addr_street_name``
            - ``addr_city``
            - ``addr_state``
            - ``addr_zip``
            - ``addr_normalized_key`` (from ``utils.create_address_key_gpu``)
        """
        rename_map: Dict[str, str] = {
            "address_line_1.street_number": "addr_street_number",
            "address_line_1.street_name": "addr_street_name",
            "city": "addr_city",
            "state": "addr_state",
            "postal_code": "addr_zip",
        }

        parsed_df_renamed = parsed_df.rename(columns=rename_map)

        # Concatenate along columns, ensuring index alignment.
        out_gdf = cudf.concat([gdf, parsed_df_renamed], axis=1)

        # Ensure columns exist and are string-typed, filling NA introduced by concat.
        for col in rename_map.values():
            if col not in out_gdf.columns:
                out_gdf[col] = ""
            else:
                out_gdf[col] = out_gdf[col].fillna("")

        logger.debug("Creating addr_normalized_key…")
        out_gdf["addr_normalized_key"] = utils.create_address_key_gpu(out_gdf)
        return out_gdf

    def _consolidate_similar_addresses(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """Find and consolidate groups of near-duplicate address keys.

        Implementation steps
        --------------------
        1) Work on a narrow view of relevant columns only
        2) Compute a similarity graph over unique keys
        3) Extract connected components and pick a canonical key per component
        4) Replace non-canonical keys with the component's canonical key
        """
        logger.info("Consolidating similar addresses on GPU…")

        cols_needed = [
            "addr_street_number",
            "addr_street_name",
            "addr_city",
            "addr_state",
            "addr_zip",
            "addr_normalized_key",
        ]
        missing = [c for c in cols_needed if c not in gdf.columns]
        if missing:
            logger.warning(f"Cannot consolidate addresses; missing columns: {missing}")
            return gdf

        address_components_gdf = gdf[cols_needed].copy()

        consolidation_map = self._get_consolidation_map(
            address_gdf=address_components_gdf,
            key_col="addr_normalized_key",
            fuzz_ratio=self.validation_config.address_fuzz_ratio,
        )

        if consolidation_map is not None and not consolidation_map.empty:
            logger.info(f"Consolidating {len(consolidation_map)} address variations.")
            # Replace keys in a copy to avoid unintended side-effects on caller-held refs
            out_gdf = gdf.copy()
            out_gdf["addr_normalized_key"] = out_gdf["addr_normalized_key"].replace(consolidation_map)
            return out_gdf

        logger.debug("No address consolidation applied (no matches found).")
        return gdf

    def _get_consolidation_map(
        self, address_gdf: cudf.DataFrame, key_col: str, fuzz_ratio: int
    ) -> Optional[cudf.Series]:
        """Build a mapping from each key to its canonical representative.

        Parameters
        ----------
        address_gdf : cudf.DataFrame
            Narrow data frame with key and component columns.
        key_col : str
            Column name containing the normalized address key.
        fuzz_ratio : int
            Fuzzy ratio in [0, 100]; higher means stricter match.

        Returns
        -------
        Optional[cudf.Series]
            A Series mapping ``original_key -> canonical_key`` or ``None`` if no
            consolidation is needed.
        """
        # Compute frequency across the *full* (non-deduped) set so it has signal.
        freq_map = address_gdf[key_col].value_counts()

        # Work over unique keys, reindexing to a dense RangeIndex so that graph
        # component outputs can align by integer position.
        unique_addresses = address_gdf.drop_duplicates(subset=[key_col]).reset_index(drop=True)
        if len(unique_addresses) < 2:
            logger.debug("Fewer than 2 unique addresses; skipping consolidation.")
            return None

        distance_threshold = 1.0 - (float(fuzz_ratio) / 100.0)
        logger.debug(f"Finding matches among {len(unique_addresses)} unique strings with threshold={distance_threshold:.3f}")

        matched_pairs = utils.find_similar_pairs(
            string_series=unique_addresses[key_col],
            vectorizer_params=self.vectorizer_config.similarity_tfidf,
            nn_params=self.vectorizer_config.similarity_nn,
            distance_threshold=distance_threshold,
        )
        if matched_pairs.empty:
            return None

        components = utils.find_graph_components(
            edge_list=matched_pairs,
            vertex_col_name="unique_addr_idx",
            component_col_name="component_id",
        )

        # Attach component IDs back to the unique address rows (by dense index)
        unique_addresses = unique_addresses.merge(
            components, left_index=True, right_on="unique_addr_idx"
        )

        # Carry over global frequency as a feature for canonical selection
        unique_addresses["freq"] = unique_addresses[key_col].map(freq_map)

        # Determine canonical representative per component
        canonical_map = self._determine_canonical_representatives(
            candidates_gdf=unique_addresses, key_col=key_col
        )

        # Build final mapping: for all members of a component, map key -> canonical_key
        final_map_df = unique_addresses[[key_col, "component_id"]].merge(
            canonical_map, on="component_id"
        )
        final_map_df = final_map_df[final_map_df[key_col] != final_map_df["canonical_key"]]

        if final_map_df.empty:
            return None

        return final_map_df.set_index(key_col)["canonical_key"]

    def _determine_canonical_representatives(
        self, candidates_gdf: cudf.DataFrame, key_col: str
    ) -> cudf.DataFrame:
        """Score and select the best representative per component.

        Heuristics used
        ---------------
        - ``completeness_score`` from ``utils.calculate_address_score_gpu`` ranks
          addresses with more complete component coverage higher.
        - ``freq`` ranks keys that appear more often in the dataset higher.
        - ``key_col`` (ascending) provides a final deterministic tiebreaker.

        Returns
        -------
        cudf.DataFrame
            Two-column frame: ``component_id`` and ``canonical_key``.
        """
        # Score completeness using available component columns
        candidates_gdf["completeness_score"] = utils.calculate_address_score_gpu(candidates_gdf)
        if "freq" not in candidates_gdf.columns:
            # Should be set upstream, but default to 1 for safety.
            candidates_gdf["freq"] = 1

        sorted_candidates = candidates_gdf.sort_values(
            ["completeness_score", "freq", key_col], ascending=[False, False, True]
        )

        # First row per component is the canonical representative
        canonical_reps = sorted_candidates.drop_duplicates(subset=["component_id"], keep="first")
        return canonical_reps[["component_id", key_col]].rename(columns={key_col: "canonical_key"})
