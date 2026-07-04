"""
data_splitter.py — Video-level stratified splitting for OzFish crops.

Strategy (approved DEC-004):
  - Split by VIDEO ID (not by image) to prevent data leakage.
  - L/R stereo pairs automatically stay together (same video + same frame).
  - Stratification ensures each species has proportional representation in
    every split.
  - Target ratios: 70% train / 15% validation / 15% test.

Usage:
  Called by scripts/split_dataset.py after species selection.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Split selected OzFish species into train/val/test directories.

    Parameters
    ----------
    df : pd.DataFrame
        Parsed + validated OzFish DataFrame (from OzFishParser).
    species_list : list[str]
        Canonical species names to include (e.g. "Lethrinus punctulatus").
    output_dir : Path
        Root output directory (datasets/processed/).
    train_ratio : float
    val_ratio : float
    test_ratio : float
    seed : int
    """

    def __init__(
        self,
        df: pd.DataFrame,
        species_list: list[str],
        output_dir: Path,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.df = df
        self.species_list = species_list
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    # ------------------------------------------------------------------
    def split_and_organize(self) -> dict[str, Any]:
        """
        Run the full split pipeline:
          1. Filter to selected species only
          2. Group images by video ID per species
          3. Assign videos to train/val/test (stratified)
          4. Copy images into species/ subdirectories
          5. Generate split manifest and report

        Returns
        -------
        dict with keys: summary, split_manifest, per_species_stats
        """
        random.seed(self.seed)

        # ── 1. Filter ──────────────────────────────────────────────────
        selected = self.df[
            self.df["full_species"].isin(self.species_list) &
            self.df["file_exists"] &
            ~self.df["is_ambiguous"] &
            self.df["file_path"].notna()
        ].copy()
        logger.info("Selected %d images across %d species",
                     len(selected), selected["full_species"].nunique())

        # ── 2. Group by species → video ────────────────────────────────
        # Each video may contain multiple species.
        # We assign at the species-video level to keep stratification clean.
        species_video_groups = self._group_by_species_video(selected)

        # ── 3. Assign splits ──────────────────────────────────────────
        assignments = self._assign_splits(species_video_groups)

        # ── 4. Tag each row ───────────────────────────────────────────
        selected["split"] = "unassigned"
        for species, video_map in assignments.items():
            for video_id, split_name in video_map.items():
                mask = (selected["full_species"] == species) & (selected["video_id"] == video_id)
                selected.loc[mask, "split"] = split_name

        unassigned = (selected["split"] == "unassigned").sum()
        if unassigned > 0:
            logger.warning("%d images unassigned (dropped)", unassigned)
            selected = selected[selected["split"] != "unassigned"]

        # ── 5. Copy files ─────────────────────────────────────────────
        self._copy_files(selected)

        # ── 6. Reports ────────────────────────────────────────────────
        return self._generate_reports(selected)

    # ------------------------------------------------------------------
    def _group_by_species_video(
        self, df: pd.DataFrame
    ) -> dict[str, dict[str, int]]:
        """
        Returns { species: { video_id: image_count } }
        """
        groups: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for _, row in df.iterrows():
            groups[row["full_species"]][row["video_id"]] += 1
        return dict(groups)

    def _assign_splits(
        self, species_video_groups: dict[str, dict[str, int]]
    ) -> dict[str, dict[str, str]]:
        """
        For each species, sort its videos by image count and greedily
        assign to train/val/test to match target ratios.

        Returns { species: { video_id: "train"|"val"|"test" } }
        """
        assignments: dict[str, dict[str, str]] = {}

        for species, video_counts in species_video_groups.items():
            total = sum(video_counts.values())
            target_val  = int(total * self.val_ratio)
            target_test = int(total * self.test_ratio)

            # Sort videos by count (descending) for greedy packing
            sorted_videos = sorted(video_counts.items(), key=lambda x: x[1], reverse=True)

            # Shuffle with seed for reproducibility (but keep sorted order within)
            rng = random.Random(self.seed + hash(species))
            rng.shuffle(sorted_videos)

            species_map: dict[str, str] = {}
            val_count = 0
            test_count = 0

            for video_id, count in sorted_videos:
                if test_count < target_test:
                    species_map[video_id] = "test"
                    test_count += count
                elif val_count < target_val:
                    species_map[video_id] = "val"
                    val_count += count
                else:
                    species_map[video_id] = "train"

            assignments[species] = species_map

        return assignments

    def _copy_files(self, df: pd.DataFrame) -> None:
        """Copy image files into output_dir/{split}/{species_safe_name}/."""
        logger.info("Copying %d files into split directories …", len(df))

        for split_name in ("train", "val", "test"):
            split_dir = self.output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for _, row in df.iterrows():
            src = Path(row["file_path"])
            # Safe directory name: replace spaces with underscores
            species_dir = row["full_species"].replace(" ", "_")
            dst_dir = self.output_dir / row["split"] / species_dir
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name
            if not dst.exists():
                shutil.copy2(str(src), str(dst))
            copied += 1

            if copied % 5000 == 0:
                logger.info("  Copied %d / %d …", copied, len(df))

        logger.info("  Copied %d files total", copied)

    def _generate_reports(self, df: pd.DataFrame) -> dict[str, Any]:
        """Generate split statistics and manifest."""
        # Per-split counts
        split_counts = df.groupby("split").size().to_dict()

        # Per-species per-split counts
        per_species = (
            df.groupby(["full_species", "split"])
            .size()
            .unstack(fill_value=0)
        )
        # Ensure all splits present
        for col in ("train", "val", "test"):
            if col not in per_species.columns:
                per_species[col] = 0
        per_species["total"] = per_species.sum(axis=1)
        per_species = per_species.sort_values("total", ascending=False)

        # Summary
        summary = {
            "total_images":    len(df),
            "total_species":   df["full_species"].nunique(),
            "total_families":  df["family"].nunique(),
            "split_counts":    split_counts,
            "train_pct":       round(split_counts.get("train", 0) / len(df) * 100, 1),
            "val_pct":         round(split_counts.get("val", 0) / len(df) * 100, 1),
            "test_pct":        round(split_counts.get("test", 0) / len(df) * 100, 1),
            "seed":            self.seed,
        }

        # Console output
        print("\n" + "─" * 70)
        print("  SPLIT SUMMARY")
        print("─" * 70)
        print(f"  Total images : {summary['total_images']:,}")
        print(f"  Species      : {summary['total_species']}")
        print(f"  Families     : {summary['total_families']}")
        print(f"  Train        : {split_counts.get('train', 0):,} ({summary['train_pct']}%)")
        print(f"  Validation   : {split_counts.get('val', 0):,} ({summary['val_pct']}%)")
        print(f"  Test         : {split_counts.get('test', 0):,} ({summary['test_pct']}%)")
        print()

        print("  PER-SPECIES BREAKDOWN (top 20)")
        print("─" * 70)
        print(f"  {'SPECIES':<40} {'TRAIN':>6} {'VAL':>6} {'TEST':>6} {'TOTAL':>7}")
        print("─" * 70)
        for species, row in per_species.head(20).iterrows():
            print(f"  {species:<40} {row['train']:>6} {row['val']:>6} "
                  f"{row['test']:>6} {row['total']:>7}")
        if len(per_species) > 20:
            print(f"  … {len(per_species) - 20} more species …")
        print()

        # Build split manifest
        manifest = {}
        for _, row in df.iterrows():
            manifest[row["file_name"]] = {
                "split":     row["split"],
                "species":   row["full_species"],
                "video_id":  row["video_id"],
                "file_path": row["file_path"],
            }

        return {
            "summary":          summary,
            "per_species_stats": per_species,
            "split_manifest":   manifest,
        }
