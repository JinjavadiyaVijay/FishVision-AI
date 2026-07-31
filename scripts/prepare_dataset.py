"""
prepare_dataset.py — Phase 2 entry point: OzFish dataset analysis and engineering.

Run from the project root:
    python scripts/prepare_dataset.py

Outputs (all saved to datasets/):
    datasets/
    ├── metadata/
    │   ├── master_species_catalog.json   ← canonical species catalog
    │   └── master_species_catalog.csv
    ├── reports/
    │   ├── data_quality_report.json      ← file integrity + label validation
    │   └── species_recommendation_report.json  ← data-driven species selection
    └── statistics/
        ├── species_distribution.csv      ← per-species image counts
        ├── family_distribution.csv       ← per-family image counts
        └── image_size_statistics.csv     ← resolution summary
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# ── Path bootstrap ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from classification.dataset.ozfish_parser   import OzFishParser
from classification.dataset.data_validator  import DataValidator
from classification.dataset.species_selector import SpeciesSelector
from classification.preprocessing.image_processor import ImageProcessor
from classification.utilities.taxonomy      import TaxonomyCatalogBuilder

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
CROPS_DIR    = PROJECT_ROOT / "OzFish" / "crops-zip" / "assets" / "projects" / "FDFML" / "crops"
METADATA_CSV = PROJECT_ROOT / "OzFish" / "crop_metadata.csv"
DATASETS_DIR = PROJECT_ROOT / "datasets"

# Output subdirectories
REPORTS_DIR    = DATASETS_DIR / "reports"
STATISTICS_DIR = DATASETS_DIR / "statistics"
METADATA_DIR   = DATASETS_DIR / "metadata"
CACHE_DIR      = DATASETS_DIR / "cache"


def save_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    logger.info("  → %s", path.relative_to(PROJECT_ROOT))


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("  → %s", path.relative_to(PROJECT_ROOT))


# ── Step helpers ───────────────────────────────────────────────────────────────

def step_parse() -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("STEP 1 — Parse OzFish metadata")
    logger.info("=" * 60)
    parser = OzFishParser(CROPS_DIR, METADATA_CSV)
    df = parser.parse()
    # Cache the parsed dataframe (parquet preferred; CSV fallback)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        cache_path = CACHE_DIR / "parsed_metadata.parquet"
        df.to_parquet(cache_path, index=False)
        logger.info("Cached → %s", cache_path.relative_to(PROJECT_ROOT))
    except Exception:
        cache_path = CACHE_DIR / "parsed_metadata.csv"
        df.to_csv(cache_path, index=False)
        logger.info("Cached (CSV fallback) → %s", cache_path.relative_to(PROJECT_ROOT))
    return df


def step_validate(df: pd.DataFrame) -> dict:
    logger.info("=" * 60)
    logger.info("STEP 2 — Validate data quality")
    logger.info("=" * 60)
    validator = DataValidator(df, check_readability=False)
    report = validator.validate()
    save_json(report["summary"], REPORTS_DIR / "data_quality_report.json")
    return report


def step_statistics(df: pd.DataFrame) -> dict:
    logger.info("=" * 60)
    logger.info("STEP 3 — Compute image statistics")
    logger.info("=" * 60)
    processor = ImageProcessor(df)
    stats = processor.compute_statistics()

    # Save CSV outputs
    save_csv(stats["size_buckets"], STATISTICS_DIR / "image_size_buckets.csv")
    save_csv(stats["per_species"],  STATISTICS_DIR / "image_size_per_species.csv")
    save_json(stats["overall"],     STATISTICS_DIR / "image_size_overall.json")

    # Log overall stats to console
    o = stats["overall"]
    logger.info("Crop width  : mean=%.0fpx  median=%.0fpx  p5=%.0f  p95=%.0f",
                o["width_mean"], o["width_median"], o["width_p5"], o["width_p95"])
    logger.info("Crop height : mean=%.0fpx  median=%.0fpx  p5=%.0f  p95=%.0f",
                o["height_mean"], o["height_median"], o["height_p5"], o["height_p95"])
    logger.info("Aspect ratio: mean=%.2f  median=%.2f",
                o["aspect_ratio_mean"], o["aspect_ratio_median"])
    return stats


def step_species_analysis(df: pd.DataFrame) -> dict:
    logger.info("=" * 60)
    logger.info("STEP 4 — Species distribution analysis")
    logger.info("=" * 60)
    selector = SpeciesSelector(
        df,
        absolute_minimum=20,
        viability_threshold=100,
        recommended_threshold=200,
        high_quality_threshold=500,
    )
    analysis = selector.analyse()

    sp_df  = analysis["species_counts"]
    fam_df = analysis["family_counts"]

    # ── Console table ──────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print(f"  {'FULL SPECIES':<40} {'COUNT':>7}  {'CUM%':>6}")
    print("─" * 65)
    for _, row in sp_df.head(40).iterrows():
        marker = ""
        if row["image_count"] >= 500:   marker = " ●"
        elif row["image_count"] >= 200: marker = " ○"
        elif row["image_count"] >= 100: marker = " ·"
        print(f"  {row['full_species']:<40} {row['image_count']:>7,}  "
              f"{row['cumulative_pct']:>5.1f}%{marker}")
    if len(sp_df) > 40:
        print(f"  … {len(sp_df) - 40} more species not shown …")
    print("─" * 65)
    print(f"  Legend:  ● ≥500 images  ○ ≥200 images  · ≥100 images")
    print()

    # ── Tier summary ───────────────────────────────────────────────────
    print("  TIER BREAKDOWN")
    print("─" * 65)
    for tier_name, tier_data in analysis["tier_summary"].items():
        print(f"  {tier_name:<30}  "
              f"{tier_data['species_count']:>4} species  "
              f"{tier_data['total_images']:>7,} images  "
              f"{tier_data['families_covered']:>3} families  "
              f"({tier_data['pct_of_all_images']}% of total)")
    print()

    # ── Stereo note ────────────────────────────────────────────────────
    sn = analysis["stereo_note"]
    print(f"  STEREO CAMERA NOTE")
    print(f"  Left camera crops : {sn['left_camera_crops']:,}")
    print(f"  Right camera crops: {sn['right_camera_crops']:,}")
    print(f"  Est. unique fish  : {sn['estimated_unique_fish']:,}")
    print()

    # ── Recommendation ─────────────────────────────────────────────────
    rec = analysis["recommendation"]
    print("  RECOMMENDATION")
    print("─" * 65)
    print(f"  Viable threshold (≥{rec['viable_threshold']} images/species):")
    print(f"    → {rec['viable_species_count']} species, "
          f"{rec['viable_total_images']:,} images")
    print(f"  Recommended threshold (≥{rec['recommended_threshold']} images/species):")
    print(f"    → {rec['recommended_species_count']} species across "
          f"{rec['recommended_families']} families, "
          f"{rec['recommended_total_images']:,} images")
    print(f"  Distribution elbow point: ~{rec['elbow_threshold']} images/species")
    print(f"    → {rec['elbow_species_count']} species at elbow threshold")
    print()

    # ── Save ───────────────────────────────────────────────────────────
    save_csv(sp_df,  STATISTICS_DIR / "species_distribution.csv")
    save_csv(fam_df, STATISTICS_DIR / "family_distribution.csv")

    rec_report = {
        "tier_summary":   analysis["tier_summary"],
        "stereo_note":    analysis["stereo_note"],
        "recommendation": rec,
    }
    save_json(rec_report, REPORTS_DIR / "species_recommendation_report.json")

    return analysis


def step_build_catalog(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("STEP 5 — Build master species catalog")
    logger.info("=" * 60)
    builder = TaxonomyCatalogBuilder(df)
    catalog = builder.build()
    builder.save(catalog, DATASETS_DIR)

    # Print catalog preview
    print("\n  MASTER SPECIES CATALOG (top 20 by image count)")
    print("─" * 90)
    print(f"  {'SCIENTIFIC NAME':<40} {'COMMON NAME':<30} {'IMAGES':>7}  {'FAMILY':<16}")
    print("─" * 90)
    for _, row in catalog.head(20).iterrows():
        common = row["common_name"] if pd.notna(row["common_name"]) else "(unknown)"
        print(f"  {row['scientific_name']:<40} {common:<30} "
              f"{row['image_count']:>7,}  {row['family']:<16}")
    print(f"\n  Total catalog entries: {len(catalog)}")
    known_cn = catalog["common_name"].notna().sum()
    print(f"  Common names resolved: {known_cn}/{len(catalog)} "
          f"({known_cn/len(catalog)*100:.0f}%)")
    print()

    return catalog


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.perf_counter()

    print("\n" + "=" * 60)
    print("  FishVision-AI — Phase 2: Dataset Analysis")
    print("=" * 60)
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Crops dir    : {CROPS_DIR.relative_to(PROJECT_ROOT)}")
    print(f"  Metadata CSV : {METADATA_CSV.relative_to(PROJECT_ROOT)}")
    print(f"  Output dir   : {DATASETS_DIR.relative_to(PROJECT_ROOT)}")
    print("=" * 60 + "\n")

    # Preflight
    for p in (CROPS_DIR, METADATA_CSV):
        if not p.exists():
            logger.error("Required path not found: %s", p)
            sys.exit(1)

    # Run pipeline
    df       = step_parse()
    _report  = step_validate(df)
    _stats   = step_statistics(df)
    analysis = step_species_analysis(df)
    catalog  = step_build_catalog(df)

    elapsed = time.perf_counter() - t0
    print("=" * 60)
    print(f"  Phase 2 complete in {elapsed:.1f}s")
    print(f"  All outputs saved under: datasets/")
    print("=" * 60 + "\n")

    print("  NEXT STEPS")
    print("  ─" * 30)
    rec = analysis["recommendation"]
    print(f"  1. Review datasets/reports/species_recommendation_report.json")
    print(f"  2. Choose minimum image threshold (recommended: {rec['recommended_threshold']})")
    print(f"  3. Confirm V1 species list (recommended threshold gives "
          f"{rec['recommended_species_count']} species)")
    print(f"  4. Approve → proceed to Phase 3 (dataset splitting)")
    print()


if __name__ == "__main__":
    main()
