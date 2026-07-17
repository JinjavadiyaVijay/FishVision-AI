"""
cleanup_project.py — Safe, complete repository cleanup for FishVision-AI.

Performs:
  1. Removes __pycache__ and .pyc files
  2. Removes swap files (.crswap, .swp)
  3. Removes personal files (assets/*.pdf)
  4. Removes generated cache (datasets/cache/)
  5. Removes duplicate root yolov8n.pt
  6. Removes local venv/ directory
  7. Moves debug scripts to scripts/debug/
  8. Archives old experiments (keeps 2 latest)
  9. Removes empty plot/ directories from experiments
  10. Removes large training_state.pt from archived experiments (keeps best_accuracy.pt only)

Usage:
    python scripts/cleanup_project.py
    python scripts/cleanup_project.py --dry-run   # Preview only

Safety:
  - Never deletes datasets/processed/
  - Never deletes model weights in models/
  - Never deletes active classification/ code
  - Never deletes experiments/bioclip2_full_20260716_160143 (latest)
  - Never deletes experiments/bioclip2_full_20260715_235014 (second latest)
"""

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Experiments to KEEP (the two best/latest full runs)
KEEP_EXPERIMENTS = {
    "bioclip2_full_20260716_160143",  # Latest — best_accuracy=73.8%
    "bioclip2_full_20260715_235014",  # Second latest
}

# Experiment prefixes that are definitely safe to delete (no useful weights)
DISPOSABLE_PREFIXES = (
    "bioclip2_smoke_test_",
    "benchmark_w4_",
    "bioclip2_debug_",
    "gate_linear_smoke",
    "gate_linear_debug",
    "gate_lora_smoke",
    "gate_linear_smoke_v2",
)

deleted: list[str] = []
archived: list[str] = []
moved: list[str] = []
skipped: list[str] = []


def rm(path: Path, dry: bool) -> None:
    rel = str(path.relative_to(ROOT))
    if dry:
        print(f"  [DRY] DELETE  {rel}")
    else:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        print(f"  DELETE  {rel}")
    deleted.append(rel)


def mv(src: Path, dst: Path, dry: bool, is_archive=False) -> None:
    rel_src = str(src.relative_to(ROOT))
    rel_dst = str(dst.relative_to(ROOT))
    if dry:
        print(f"  [DRY] {'ARCHIVE' if is_archive else 'MOVE'}  {rel_src} → {rel_dst}")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        print(f"  {'ARCHIVE' if is_archive else 'MOVE'}  {rel_src} → {rel_dst}")
    if is_archive:
        archived.append(f"{rel_src} → {rel_dst}")
    else:
        moved.append(f"{rel_src} → {rel_dst}")


def main(dry: bool) -> None:
    print(f"\n{'=' * 60}")
    print(f"  FishVision-AI Repository Cleanup {'(DRY RUN)' if dry else ''}")
    print(f"{'=' * 60}\n")

    # ── 1. __pycache__ and .pyc ─────────────────────────────────
    print("[1/9] Removing __pycache__ and .pyc files...")
    for p in sorted(ROOT.rglob("__pycache__")):
        if "venv" not in p.parts and ".git" not in p.parts:
            rm(p, dry)
    for p in sorted(ROOT.rglob("*.pyc")):
        if "venv" not in p.parts:
            rm(p, dry)

    # ── 2. Swap / temp files ─────────────────────────────────────
    print("\n[2/9] Removing swap and temp files...")
    for pattern in ("*.crswap", "*.swp", "*.tmp"):
        for p in ROOT.rglob(pattern):
            if ".git" not in p.parts:
                rm(p, dry)

    # ── 3. Personal files from assets/ ──────────────────────────
    print("\n[3/9] Removing personal files from assets/...")
    for p in (ROOT / "assets").glob("*.pdf"):
        rm(p, dry)

    # ── 4. Generated dataset cache ───────────────────────────────
    print("\n[4/9] Removing generated dataset cache...")
    cache_parquet = ROOT / "datasets" / "cache" / "parsed_metadata.parquet"
    if cache_parquet.exists():
        rm(cache_parquet, dry)
    # If cache dir is now empty, remove it too
    cache_dir = ROOT / "datasets" / "cache"
    if cache_dir.exists() and not any(cache_dir.iterdir()):
        rm(cache_dir, dry)

    # ── 5. Duplicate root yolov8n.pt ────────────────────────────
    print("\n[5/9] Removing duplicate root yolov8n.pt...")
    root_yolo = ROOT / "yolov8n.pt"
    models_yolo = ROOT / "models" / "yolov8n.pt"
    if root_yolo.exists() and models_yolo.exists():
        if root_yolo.stat().st_size == models_yolo.stat().st_size:
            rm(root_yolo, dry)
        else:
            print(f"  SKIP: sizes differ — not a duplicate")
            skipped.append(str(root_yolo.relative_to(ROOT)))

    # ── 6. Local venv/ ───────────────────────────────────────────
    print("\n[6/9] Removing local venv/ (project uses Conda ai env)...")
    venv_dir = ROOT / "venv"
    if venv_dir.exists():
        rm(venv_dir, dry)

    # ── 7. Move debug scripts ────────────────────────────────────
    print("\n[7/9] Moving debug scripts to scripts/debug/...")
    debug_dir = ROOT / "scripts" / "debug"
    for fname in ("debug_filenames.py", "test_dataset.py"):
        src = ROOT / "scripts" / fname
        if src.exists():
            mv(src, debug_dir / fname, dry)
    # Move run_phase2.bat into scripts/
    bat = ROOT / "run_phase2.bat"
    if bat.exists():
        mv(bat, ROOT / "scripts" / "run_phase2.bat", dry)

    # ── 8. Archive old experiments ───────────────────────────────
    print("\n[8/9] Archiving old experiments...")
    exp_dir = ROOT / "experiments"
    archive_exp = ROOT / "archive" / "experiments"

    if exp_dir.exists():
        for d in sorted(exp_dir.iterdir()):
            if not d.is_dir():
                continue
            if d.name in KEEP_EXPERIMENTS:
                print(f"  KEEP    {d.name}")
                continue
            # Safe to delete smoke/debug/benchmark runs entirely
            if any(d.name.startswith(pfx) for pfx in DISPOSABLE_PREFIXES):
                rm(d, dry)
            else:
                # Other old full runs — archive, strip training_state.pt to save space
                dst = archive_exp / d.name
                mv(d, dst, dry, is_archive=True)
                # After moving, remove training_state.pt (huge, not needed for inference)
                ts = dst / "checkpoints" / "training_state.pt"
                if not dry and ts.exists():
                    ts.unlink()
                    print(f"  DELETE  {d.name}/checkpoints/training_state.pt (optimizer state, {ts.stat().st_size // 1e6:.0f} MB recovered)")

    # ── 9. Remove empty directories ──────────────────────────────
    print("\n[9/9] Removing empty directories...")
    for dirpath, dirnames, filenames in os.walk(str(ROOT), topdown=False):
        p = Path(dirpath)
        if p == ROOT:
            continue
        skip_dirs = {".git", "venv", "archive", "OzFish", "datasets", "experiments"}
        if any(s in p.parts for s in skip_dirs):
            continue
        try:
            if p.exists() and not list(p.iterdir()):
                rm(p, dry)
        except (PermissionError, OSError):
            pass

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  CLEANUP {'PREVIEW' if dry else 'COMPLETE'}")
    print(f"{'=' * 60}")
    print(f"\n  Deleted:  {len(deleted)}")
    print(f"  Archived: {len(archived)}")
    print(f"  Moved:    {len(moved)}")
    print(f"  Skipped:  {len(skipped)}")
    if dry:
        print(f"\n  Run without --dry-run to apply changes.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    args = parser.parse_args()
    main(dry=args.dry_run)
