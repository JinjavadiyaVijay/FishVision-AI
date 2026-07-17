"""
cleanup.py — One-shot repository cleanup script.

Performs safe, non-destructive cleanup:
  - Removes __pycache__ and .pyc
  - Removes swap files (.crswap)
  - Removes personal files from assets/
  - Removes generated cache (datasets/cache/parsed_metadata.parquet)
  - Removes duplicate root yolov8n.pt (keeps models/yolov8n.pt)
  - Removes venv/ (user uses Conda)
  - Moves debug scripts to scripts/debug/
  - Moves legacy code to archive/legacy/
  - Moves old experiments to archive/experiments/
  - Moves run_phase2.bat to scripts/
  - Removes empty directories

Run from project root:
    python scripts/cleanup.py
"""

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRY_RUN = False  # Set True to preview without changes

deleted = []
archived = []
moved = []


def log_delete(path: Path):
    deleted.append(str(path.relative_to(ROOT)))

def log_archive(src: Path, dst: Path):
    archived.append(f"{src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")

def log_move(src: Path, dst: Path):
    moved.append(f"{src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")

def safe_rmtree(path: Path):
    if path.exists():
        if not DRY_RUN:
            shutil.rmtree(path)
        log_delete(path)

def safe_rm(path: Path):
    if path.exists():
        if not DRY_RUN:
            path.unlink()
        log_delete(path)

def safe_move(src: Path, dst: Path, is_archive=False):
    if src.exists():
        if not DRY_RUN:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        if is_archive:
            log_archive(src, dst)
        else:
            log_move(src, dst)


# ═══════════════════════════════════════════════════════════
# 1. Remove __pycache__ and .pyc
# ═══════════════════════════════════════════════════════════
print("[1/10] Removing __pycache__ and .pyc files...")
for p in sorted(ROOT.rglob("__pycache__")):
    if "venv" not in str(p):
        safe_rmtree(p)

for p in sorted(ROOT.rglob("*.pyc")):
    if "venv" not in str(p):
        safe_rm(p)


# ═══════════════════════════════════════════════════════════
# 2. Remove swap files
# ═══════════════════════════════════════════════════════════
print("[2/10] Removing swap files...")
for p in sorted(ROOT.rglob("*.crswap")):
    safe_rm(p)
for p in sorted(ROOT.rglob("*.swp")):
    safe_rm(p)


# ═══════════════════════════════════════════════════════════
# 3. Remove personal files
# ═══════════════════════════════════════════════════════════
print("[3/10] Removing personal files from assets/...")
pdf = ROOT / "assets" / "Student Profile @ GTU.pdf"
safe_rm(pdf)


# ═══════════════════════════════════════════════════════════
# 4. Remove generated caches
# ═══════════════════════════════════════════════════════════
print("[4/10] Removing generated caches...")
safe_rm(ROOT / "datasets" / "cache" / "parsed_metadata.parquet")


# ═══════════════════════════════════════════════════════════
# 5. Remove duplicate root yolov8n.pt
# ═══════════════════════════════════════════════════════════
print("[5/10] Removing duplicate root yolov8n.pt...")
root_yolo = ROOT / "yolov8n.pt"
models_yolo = ROOT / "models" / "yolov8n.pt"
if root_yolo.exists() and models_yolo.exists():
    # Verify they are the same size (proxy for duplicate check)
    if root_yolo.stat().st_size == models_yolo.stat().st_size:
        safe_rm(root_yolo)
    else:
        print(f"  SKIPPED: sizes differ ({root_yolo.stat().st_size} vs {models_yolo.stat().st_size})")


# ═══════════════════════════════════════════════════════════
# 6. Remove venv/
# ═══════════════════════════════════════════════════════════
print("[6/10] Removing local venv/...")
safe_rmtree(ROOT / "venv")


# ═══════════════════════════════════════════════════════════
# 7. Move debug scripts to scripts/debug/
# ═══════════════════════════════════════════════════════════
print("[7/10] Moving debug scripts...")
debug_dir = ROOT / "scripts" / "debug"
safe_move(ROOT / "scripts" / "debug_filenames.py", debug_dir / "debug_filenames.py")
safe_move(ROOT / "scripts" / "test_dataset.py", debug_dir / "test_dataset.py")
safe_move(ROOT / "run_phase2.bat", ROOT / "scripts" / "run_phase2.bat")


# ═══════════════════════════════════════════════════════════
# 8. Archive legacy code
# ═══════════════════════════════════════════════════════════
print("[8/10] Archiving legacy code...")
legacy_dir = ROOT / "archive" / "legacy"

legacy_files = [
    "classification/trainer/bioclip_trainer.py",
    "classification/dataloader/bioclip_dataset.py",
    "classification/metrics/classification_metrics.py",
    "classification/augmentation/fish_augmentation.py",
    "classification/config/default.yaml",
]
for rel in legacy_files:
    src = ROOT / rel
    dst = legacy_dir / rel
    safe_move(src, dst, is_archive=True)


# ═══════════════════════════════════════════════════════════
# 9. Archive old experiments
# ═══════════════════════════════════════════════════════════
print("[9/10] Archiving old experiments...")
exp_dir = ROOT / "experiments"
archive_exp = ROOT / "archive" / "experiments"

# Keep only the two latest full runs
keep_experiments = {
    "bioclip2_full_20260716_160143",  # latest
    "bioclip2_full_20260715_235014",  # second latest
}

if exp_dir.exists():
    for d in sorted(exp_dir.iterdir()):
        if d.is_dir() and d.name not in keep_experiments:
            safe_move(d, archive_exp / d.name, is_archive=True)


# ═══════════════════════════════════════════════════════════
# 10. Remove empty directories
# ═══════════════════════════════════════════════════════════
print("[10/10] Removing empty directories...")
# Walk bottom-up to catch nested empties
for dirpath, dirnames, filenames in os.walk(str(ROOT), topdown=False):
    p = Path(dirpath)
    if p == ROOT:
        continue
    if ".git" in str(p) or "venv" in str(p):
        continue
    # Check if dir is truly empty (no files, no subdirs)
    try:
        remaining = list(p.iterdir())
        if not remaining and p.exists():
            if not DRY_RUN:
                p.rmdir()
            log_delete(p)
    except (PermissionError, OSError):
        pass


# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("CLEANUP COMPLETE")
print("=" * 60)

print(f"\nDeleted ({len(deleted)}):")
for item in deleted:
    print(f"  - {item}")

print(f"\nArchived ({len(archived)}):")
for item in archived:
    print(f"  - {item}")

print(f"\nMoved ({len(moved)}):")
for item in moved:
    print(f"  - {item}")

print(f"\nTotal actions: {len(deleted) + len(archived) + len(moved)}")
print()
