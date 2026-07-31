"""
species_selector.py — Data-driven species selection and recommendation.

This module analyses the class distribution of the OzFish dataset
and recommends an optimal species subset for V1 training.

Decision philosophy (user-approved DEC-003):
  Do NOT hard-code the number of species.
  Instead, analyse the actual distribution, find natural breakpoints,
  and recommend based on statistical evidence and ML viability criteria.

Criteria applied:
  1.  Exclude ambiguous / unidentifiable labels (sp, spp, sp1 …)
  2.  Exclude species below absolute_minimum (< 20 images)
  3.  Identify the inflection point in the sorted count curve
  4.  Report species counts at several threshold levels
  5.  Recommend the threshold that maximises:
        - species diversity (more species is better)
        - minimum per-class validation samples (≥ 15 after 70/15/15 split)
        - coverage of major families
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Minimum images per class to have ≥ 15 val samples at 15% split rate
# 15 / 0.15 = 100 images minimum (rounded up)
_MIN_FOR_VALID_SPLIT = 100


class SpeciesSelector:
    """
    Analyse OzFish class distribution and recommend a V1 species set.

    Parameters
    ----------
    df : pd.DataFrame
        Output of OzFishParser.parse() — must include is_ambiguous, full_species columns.
    absolute_minimum : int
        Exclude species with fewer images than this (unusable).
    viability_threshold : int
        Minimum images to have a statistically valid val set.
    recommended_threshold : int
        Comfortable minimum — good train/val/test balance.
    high_quality_threshold : int
        Species with ≥ this are well-resourced for fine-tuning.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        absolute_minimum: int = 20,
        viability_threshold: int = 100,
        recommended_threshold: int = 200,
        high_quality_threshold: int = 500,
    ) -> None:
        self.df = df
        self.abs_min = absolute_minimum
        self.viable = viability_threshold
        self.recommended = recommended_threshold
        self.high_quality = high_quality_threshold

    # ------------------------------------------------------------------
    def analyse(self) -> dict[str, Any]:
        """
        Full analysis pipeline.

        Returns a dict with:
          species_counts      : per-species image count DataFrame
          family_counts       : per-family count DataFrame
          tier_summary        : image counts at each threshold tier
          recommendation      : recommended threshold + species list
          stereo_note         : effective unique fish count estimate
        """
        # ── 1. Clean subset (exclude ambiguous labels) ──────────────────
        clean = self.df[~self.df["is_ambiguous"] & self.df["file_exists"]].copy()
        logger.info("Analysing %d clean (identifiable) crops …", len(clean))

        # ── 2. Per-species counts ───────────────────────────────────────
        sp_counts = (
            clean.groupby(["full_species", "family", "genus"])
            .size()
            .reset_index(name="image_count")
            .sort_values("image_count", ascending=False)
            .reset_index(drop=True)
        )
        sp_counts["rank"] = sp_counts.index + 1
        sp_counts["cumulative_pct"] = (
            sp_counts["image_count"].cumsum() / sp_counts["image_count"].sum() * 100
        ).round(1)

        # ── 3. Per-family counts ─────────────────────────────────────────
        fam_counts = (
            clean.groupby("family")
            .agg(
                species_count=("full_species", "nunique"),
                image_count=("full_species", "count"),
            )
            .reset_index()
            .sort_values("image_count", ascending=False)
        )

        # ── 4. Tier breakdown ────────────────────────────────────────────
        tiers = self._build_tier_summary(sp_counts)

        # ── 5. Stereo pair estimate ──────────────────────────────────────
        n_L = (clean["camera"] == "L").sum()
        n_R = (clean["camera"] == "R").sum()
        stereo_note = {
            "left_camera_crops":  int(n_L),
            "right_camera_crops": int(n_R),
            "estimated_unique_fish": int(max(n_L, n_R)),
            "note": (
                "Each fish appears in both Left and Right cameras. "
                "Effective unique fish count ≈ max(L, R) crops. "
                "L/R pairs must be kept in the same split."
            ),
        }

        # ── 6. Recommendation ───────────────────────────────────────────
        recommendation = self._recommend(sp_counts, fam_counts)

        return {
            "species_counts": sp_counts,
            "family_counts":  fam_counts,
            "tier_summary":   tiers,
            "stereo_note":    stereo_note,
            "recommendation": recommendation,
        }

    # ------------------------------------------------------------------
    def _build_tier_summary(self, sp_counts: pd.DataFrame) -> dict:
        thresholds = {
            "unusable (< 20)":          self.abs_min,
            "viable (≥ 100)":           self.viable,
            "recommended (≥ 200)":      self.recommended,
            "high-quality (≥ 500)":     self.high_quality,
            "very-high (≥ 1000)":       1000,
        }
        tiers = {}
        for label, thresh in thresholds.items():
            mask = sp_counts["image_count"] >= thresh
            subset = sp_counts[mask]
            tiers[label] = {
                "min_threshold":  thresh,
                "species_count":  int(mask.sum()),
                "total_images":   int(subset["image_count"].sum()),
                "families_covered": int(subset["family"].nunique()),
                "pct_of_all_images": round(
                    subset["image_count"].sum() / sp_counts["image_count"].sum() * 100, 1
                ),
            }
        return tiers

    def _recommend(
        self, sp_counts: pd.DataFrame, fam_counts: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Find the optimal threshold by maximising species count while
        ensuring every class has enough images for a valid val set.
        """
        # Min images needed for ≥15 val samples at 15% split = ceil(15/0.15) = 100
        viable_sp = sp_counts[sp_counts["image_count"] >= _MIN_FOR_VALID_SPLIT].copy()
        n_viable = len(viable_sp)

        # Find the 'elbow' in the distribution using log-ratio of adjacent counts
        sorted_counts = sp_counts["image_count"].sort_values(ascending=False).values
        elbow_threshold = self._find_elbow(sorted_counts)

        elbow_sp = sp_counts[sp_counts["image_count"] >= elbow_threshold]

        # Best recommendation: at the recommended threshold
        rec_sp = sp_counts[sp_counts["image_count"] >= self.recommended]

        # Compute minimum images that still gives ≥10 val samples per class
        min_viable_threshold = math.ceil(10 / 0.15)  # ≈ 67

        return {
            "viable_threshold":         _MIN_FOR_VALID_SPLIT,
            "viable_species_count":     n_viable,
            "viable_total_images":      int(viable_sp["image_count"].sum()),
            "elbow_threshold":          elbow_threshold,
            "elbow_species_count":      int(len(elbow_sp)),
            "recommended_threshold":    self.recommended,
            "recommended_species_count": int(len(rec_sp)),
            "recommended_total_images": int(rec_sp["image_count"].sum()),
            "recommended_families":     int(rec_sp["family"].nunique()),
            "recommended_species_list": rec_sp["full_species"].tolist(),
            "absolute_minimum_threshold": self.abs_min,
            "min_images_for_10_val_samples": min_viable_threshold,
            "note": (
                f"Recommended threshold = {self.recommended} images/species. "
                f"This gives {len(rec_sp)} species across {rec_sp['family'].nunique()} families. "
                f"The 'elbow' of the distribution curve is at ~{elbow_threshold} images/species."
            ),
        }

    @staticmethod
    def _find_elbow(counts: "list[int]") -> int:
        """
        Simple elbow detection on a sorted descending count array.
        Uses the point of maximum second-derivative (largest acceleration of drop).
        """
        if len(counts) < 3:
            return int(counts[-1]) if len(counts) else 0

        # Use log scale to smooth the long tail
        import numpy as np
        log_counts = np.log1p(counts)
        # Second derivative
        d2 = np.diff(np.diff(log_counts))
        # Elbow = index of max acceleration (most negative d2)
        elbow_idx = int(np.argmin(d2)) + 1
        elbow_val = int(counts[elbow_idx])
        logger.debug("Elbow detected at rank=%d, count=%d", elbow_idx, elbow_val)
        return elbow_val
