"""
data_validator.py — Validate image and label quality in an OzFish DataFrame.

Responsibilities:
  - Verify files exist on disk (fast — uses parser output)
  - Optionally open images with PIL to confirm readability
  - Flag crops below minimum resolution thresholds
  - Detect negative / zero bounding boxes (annotation errors)
  - Produce a structured validation report dict
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validate image and label quality in the parsed OzFish DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of OzFishParser.parse().
    min_bbox_width_px : int
        Crops narrower than this are flagged.
    min_bbox_height_px : int
        Crops shorter than this are flagged.
    min_bbox_area_px2 : int
        Crops with smaller area are flagged.
    check_readability : bool
        If True, attempt PIL.Image.open on every (or sampled) image.
        This is disk-intensive and slow for 80 K files.
    readability_sample_frac : float
        Fraction of images to PIL-check when check_readability=True.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        min_bbox_width_px: int = 20,
        min_bbox_height_px: int = 20,
        min_bbox_area_px2: int = 400,
        check_readability: bool = False,
        readability_sample_frac: float = 0.05,
    ) -> None:
        self.df = df.copy()
        self.min_w = min_bbox_width_px
        self.min_h = min_bbox_height_px
        self.min_area = min_bbox_area_px2
        self.check_readability = check_readability
        self.readability_sample_frac = readability_sample_frac

    def validate(self) -> dict[str, Any]:
        """
        Run all validation checks.

        Returns
        -------
        dict with keys:
            summary, missing_files, bad_bbox, unreadable_images, flags_df
        """
        logger.info("Running data validation …")

        flags = pd.DataFrame(index=self.df.index)

        # ── 1. File existence ──────────────────────────────────────────
        flags["missing_file"]   = ~self.df["file_exists"]

        # ── 2. Bbox sanity ─────────────────────────────────────────────
        flags["bbox_too_narrow"] = self.df["bbox_width_px"]  < self.min_w
        flags["bbox_too_short"]  = self.df["bbox_height_px"] < self.min_h
        flags["bbox_too_small"]  = self.df["bbox_area_px2"]  < self.min_area
        flags["bbox_negative"]   = (
            (self.df["bbox_width_px"]  <= 0) |
            (self.df["bbox_height_px"] <= 0)
        )

        # ── 3. Ambiguous labels (from parser) ──────────────────────────
        flags["ambiguous_label"] = self.df["is_ambiguous"]

        # ── 4. Parse failures ──────────────────────────────────────────
        flags["parse_failure"]   = ~self.df["parse_ok"]

        # ── 5. Optional: image readability ────────────────────────────
        flags["unreadable"] = False
        if self.check_readability:
            flags["unreadable"] = self._check_readability()

        # ── Summary ───────────────────────────────────────────────────
        n = len(self.df)
        summary = {
            "total_rows":         n,
            "files_found":        int(self.df["file_exists"].sum()),
            "files_missing":      int(flags["missing_file"].sum()),
            "bbox_too_narrow":    int(flags["bbox_too_narrow"].sum()),
            "bbox_too_short":     int(flags["bbox_too_short"].sum()),
            "bbox_too_small":     int(flags["bbox_too_small"].sum()),
            "bbox_negative":      int(flags["bbox_negative"].sum()),
            "ambiguous_labels":   int(flags["ambiguous_label"].sum()),
            "parse_failures":     int(flags["parse_failure"].sum()),
            "unreadable_images":  int(flags["unreadable"].sum()),
        }

        # Rows that are completely usable
        any_critical = (
            flags["missing_file"] |
            flags["bbox_negative"] |
            flags["ambiguous_label"] |
            flags["parse_failure"] |
            flags["unreadable"]
        )
        summary["usable_rows"] = int((~any_critical).sum())
        summary["excluded_rows"] = int(any_critical.sum())

        self._log_summary(summary)

        return {
            "summary":          summary,
            "missing_files":    self.df.loc[flags["missing_file"], "file_name"].tolist(),
            "bad_bbox":         self.df.loc[flags["bbox_negative"], "file_name"].tolist(),
            "unreadable_images": self.df.loc[flags["unreadable"], "file_name"].tolist(),
            "flags_df":         flags,
        }

    def _check_readability(self) -> pd.Series:
        """PIL-open a sample of images; flag those that fail."""
        from PIL import Image, UnidentifiedImageError

        existing = self.df[self.df["file_exists"] & self.df["file_path"].notna()]
        k = max(1, int(len(existing) * self.readability_sample_frac))
        sample_idx = random.sample(list(existing.index), k)

        logger.info("PIL-checking %d/%d images for readability …", k, len(existing))
        unreadable = pd.Series(False, index=self.df.index)

        for idx in sample_idx:
            fpath = self.df.at[idx, "file_path"]
            try:
                with Image.open(fpath) as img:
                    img.verify()
            except (UnidentifiedImageError, Exception):
                unreadable.at[idx] = True
                logger.debug("Unreadable: %s", fpath)

        logger.info("Unreadable images in sample: %d", unreadable.sum())
        return unreadable

    def _log_summary(self, s: dict) -> None:
        logger.info("─" * 50)
        logger.info("Validation Summary")
        logger.info("  Total rows      : %d", s["total_rows"])
        logger.info("  Usable          : %d (%.1f%%)", s["usable_rows"],
                    100 * s["usable_rows"] / max(s["total_rows"], 1))
        logger.info("  Missing files   : %d", s["files_missing"])
        logger.info("  Negative bbox   : %d", s["bbox_negative"])
        logger.info("  Tiny bbox (<min): %d", s["bbox_too_small"])
        logger.info("  Ambiguous labels: %d", s["ambiguous_labels"])
        logger.info("  Parse failures  : %d", s["parse_failures"])
        logger.info("─" * 50)
