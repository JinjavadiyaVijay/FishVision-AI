"""
split_dataset.py — Phase 3 entry point: species selection + dataset splitting.

Reads the Phase 2 recommendation report, applies the recommended threshold,
copies images into train/val/test/species/ directories, and generates manifests.

Run from the project root:
    python scripts/split_dataset.py
    python scripts/split_dataset.py --threshold 300   # override threshold
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# ── Path bootstrap ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from classification.dataset.data_splitter import DataSplitter

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASETS_DIR  = PROJECT_ROOT / "datasets"
CACHE_DIR     = DATASETS_DIR / "cache"
METADATA_DIR  = DATASETS_DIR / "metadata"
REPORTS_DIR   = DATASETS_DIR / "reports"
PROCESSED_DIR = DATASETS_DIR / "processed"
REC_REPORT    = REPORTS_DIR / "species_recommendation_report.json"


def load_cached_df() -> pd.DataFrame:
    """Load the cached parsed DataFrame from Phase 2."""
    parquet = CACHE_DIR / "parsed_metadata.parquet"
    csv     = CACHE_DIR / "parsed_metadata.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    elif csv.exists():
        return pd.read_csv(csv, dtype=str)
    else:
        logger.error("No cached DataFrame found. Run Phase 2 first: python scripts/prepare_dataset.py")
        sys.exit(1)


def load_recommendation() -> dict:
    """Load the species recommendation report from Phase 2."""
    if not REC_REPORT.exists():
        logger.error("Recommendation report not found. Run Phase 2 first.")
        sys.exit(1)
    with open(REC_REPORT, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    logger.info("  → %s", path.relative_to(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Split OzFish dataset")
    parser.add_argument(
        "--threshold", type=int, default=None,
        help="Override minimum image threshold per species (default: use recommendation)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    t0 = time.perf_counter()

    print("\n" + "=" * 60)
    print("  FishVision-AI — Phase 3: Dataset Splitting")
    print("=" * 60)

    # ── Load Phase 2 outputs ───────────────────────────────────────────
    df  = load_cached_df()
    rec = load_recommendation()

    # ── Determine threshold ────────────────────────────────────────────
    threshold = args.threshold or rec["recommendation"]["recommended_threshold"]
    species_list = [
        sp for sp in rec["recommendation"]["recommended_species_list"]
    ]

    # If custom threshold, recompute species list from the full catalog
    if args.threshold:
        catalog = pd.read_csv(METADATA_DIR / "master_species_catalog.csv")
        species_list = catalog[
            catalog["image_count"] >= args.threshold
        ]["scientific_name"].tolist()

    print(f"  Threshold    : ≥{threshold} images/species")
    print(f"  Species      : {len(species_list)}")
    print(f"  Output dir   : {PROCESSED_DIR.relative_to(PROJECT_ROOT)}")
    print(f"  Seed         : {args.seed}")
    print("=" * 60 + "\n")

    # ── Run splitter ───────────────────────────────────────────────────
    splitter = DataSplitter(
        df=df,
        species_list=species_list,
        output_dir=PROCESSED_DIR,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed,
    )
    results = splitter.split_and_organize()

    # ── Save reports ──────────────────────────────────────────────────
    save_json(results["summary"], REPORTS_DIR / "split_report.json")
    save_json(results["split_manifest"], METADATA_DIR / "split_manifest.json")

    # Per-species CSV
    stats_df = results["per_species_stats"]
    stats_path = DATASETS_DIR / "statistics" / "split_per_species.csv"
    stats_df.to_csv(stats_path)
    logger.info("  → %s", stats_path.relative_to(PROJECT_ROOT))

    elapsed = time.perf_counter() - t0
    print("=" * 60)
    print(f"  Phase 3 complete in {elapsed:.1f}s")
    print(f"  Dataset ready at: {PROCESSED_DIR.relative_to(PROJECT_ROOT)}/")
    print(f"    train/  val/  test/  — each with species subdirectories")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
