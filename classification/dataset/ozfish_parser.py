"""
ozfish_parser.py — Parse and index the raw OzFish dataset.

Responsibilities:
  - Scan the crops directory to build a filename → path index
  - Load and structurally validate crop_metadata.csv
  - Parse structured fields from OzFish filenames (video_id, camera, frame, bbox)
  - Compute derived fields: bbox dimensions, full species name, stereo pair ID
  - Detect and flag ambiguous / unidentifiable labels
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filename regex
# Pattern: {VideoID}_{Camera}.avi.{Frame}.{X1}.{Y1}.{X2}.{Y2}.png
# e.g.   : A000001_L.avi.5107.806.371.922.448.png
# ---------------------------------------------------------------------------
_FNAME_RE = re.compile(
    r"^(?P<video_id>[A-Z0-9]+)_(?P<camera>[LR])\.(?:avi|AVI|mp4|MP4|mov|MOV)\."
    r"(?P<frame>\d+)\."
    r"(?P<x1>-?\d+)\.(?P<y1>-?\d+)\."
    r"(?P<x2>-?\d+)\.(?P<y2>-?\d+)\.png$"
)

# Labels that indicate an unidentified / ambiguous specimen
_AMBIGUOUS_RE = re.compile(r"^sp\d*$|^spp$", re.IGNORECASE)


class OzFishParser:
    """
    Parse and index the OzFish crop dataset.

    Parameters
    ----------
    crops_dir : Path
        Directory containing the flat list of ~80K crop PNG files.
        Files follow the pattern: {base_name}-{asset_id}-{version}.png
    metadata_csv : Path
        Path to crop_metadata.csv with columns: uid, file_name, family, genus, species.
    """

    def __init__(self, crops_dir: Path, metadata_csv: Path) -> None:
        self.crops_dir = Path(crops_dir)
        self.metadata_csv = Path(metadata_csv)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self) -> pd.DataFrame:
        """
        Run the full parse pipeline.

        Returns
        -------
        pd.DataFrame with columns:
            uid, file_name, family, genus, species,
            video_id, camera, frame,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            bbox_width_px, bbox_height_px, bbox_area_px2,
            full_species, stereo_pair_id,
            file_path (str | NaN), file_exists (bool),
            is_ambiguous (bool), parse_ok (bool)
        """
        file_index = self._build_file_index()
        df = self._load_metadata()
        df = self._parse_filenames(df)
        df = self._compute_dimensions(df)
        df = self._map_file_paths(df, file_index)
        df = self._add_derived_columns(df)
        self._log_summary(df)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_file_index(self) -> dict[str, Path]:
        """
        Scan the crops directory and build {base_name: actual_path}.

        OzFish stores files as:  {base_name}-{asset_id}-{version}.png
        We strip the trailing   -\\d+-\\d+\\.png  suffix to recover base_name,
        which matches the file_name column in crop_metadata.csv.
        """
        logger.info("Scanning crops directory: %s", self.crops_dir)
        if not self.crops_dir.exists():
            raise FileNotFoundError(f"Crops directory not found: {self.crops_dir}")

        suffix_re = re.compile(r"-\d+-\d+\.png$")
        index: dict[str, Path] = {}
        total = 0

        for fpath in self.crops_dir.glob("*.png"):
            total += 1
            base = suffix_re.sub("", fpath.name)
            index[base] = fpath

        logger.info("Indexed %d PNG files → %d unique base names", total, len(index))
        return index

    def _load_metadata(self) -> pd.DataFrame:
        """Load and minimally validate crop_metadata.csv."""
        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {self.metadata_csv}")

        df = pd.read_csv(self.metadata_csv, dtype=str)
        logger.info("Loaded metadata CSV: %d rows, %d columns", len(df), df.shape[1])

        required = {"uid", "file_name", "family", "genus", "species"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        # Normalise: strip whitespace from all string columns
        for col in ("file_name", "family", "genus", "species"):
            df[col] = df[col].str.strip()

        return df

    def _parse_filenames(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract video_id, camera, frame, and bbox from file_name."""
        logger.info("Parsing %d filenames …", len(df))

        parsed_rows = df["file_name"].apply(self._parse_one)
        parsed_df = pd.DataFrame(list(parsed_rows), index=df.index)
        return pd.concat([df, parsed_df], axis=1)

    def _parse_one(self, filename: str) -> dict:
        m = _FNAME_RE.match(filename)
        if not m:
            return {
                "video_id": None, "camera": None, "frame": None,
                "bbox_x1": None, "bbox_y1": None,
                "bbox_x2": None, "bbox_y2": None,
                "parse_ok": False,
            }
        return {
            "video_id": m.group("video_id"),
            "camera":   m.group("camera"),
            "frame":    int(m.group("frame")),
            "bbox_x1":  int(m.group("x1")),
            "bbox_y1":  int(m.group("y1")),
            "bbox_x2":  int(m.group("x2")),
            "bbox_y2":  int(m.group("y2")),
            "parse_ok": True,
        }

    def _compute_dimensions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive bbox width, height, area from parsed coordinates."""
        df["bbox_width_px"]  = df["bbox_x2"] - df["bbox_x1"]
        df["bbox_height_px"] = df["bbox_y2"] - df["bbox_y1"]
        df["bbox_area_px2"]  = df["bbox_width_px"] * df["bbox_height_px"]
        return df

    def _map_file_paths(self, df: pd.DataFrame, index: dict[str, Path]) -> pd.DataFrame:
        """Map each CSV row's file_name to its actual path on disk."""
        df["file_path"]   = df["file_name"].map(lambda fn: str(index[fn]) if fn in index else None)
        df["file_exists"] = df["file_name"].map(lambda fn: fn in index)
        return df

    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add full_species, stereo_pair_id, is_ambiguous."""
        # Canonical scientific name: "Genus species"
        df["full_species"] = (
            df["genus"].str.capitalize() + " " + df["species"].str.lower()
        )

        # Stereo pair ID: same video + same frame number → same pair
        df["stereo_pair_id"] = df["video_id"].fillna("") + "_" + df["frame"].astype(str)

        # Flag ambiguous labels (sp, spp, sp1, sp10 …)
        df["is_ambiguous"] = (
            df["species"].str.match(_AMBIGUOUS_RE, na=False) |
            df["species"].isna() |
            df["family"].isna() |
            (df["species"].str.strip() == "")
        )

        return df

    def _log_summary(self, df: pd.DataFrame) -> None:
        n_total      = len(df)
        n_exists     = df["file_exists"].sum()
        n_missing    = n_total - n_exists
        n_parse_fail = (~df["parse_ok"]).sum()
        n_ambiguous  = df["is_ambiguous"].sum()
        n_species    = df.loc[~df["is_ambiguous"], "full_species"].nunique()

        logger.info("─" * 50)
        logger.info("OzFish Parse Summary")
        logger.info("  Total rows       : %d", n_total)
        logger.info("  Files found      : %d", n_exists)
        logger.info("  Files missing    : %d", n_missing)
        logger.info("  Parse failures   : %d", n_parse_fail)
        logger.info("  Ambiguous labels : %d", n_ambiguous)
        logger.info("  Identifiable sp. : %d", n_species)
        logger.info("─" * 50)
