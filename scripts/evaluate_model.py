"""
evaluate_model.py — Full test-set evaluation of the trained BioCLIP 2 classifier.

Produces:
  - Top-1 Accuracy
  - Top-5 Accuracy
  - Macro Precision / Recall / F1
  - Weighted Precision / Recall / F1
  - Per-class accuracy, F1, precision, recall, support
  - Confusion matrix PNG (saved to experiment/plots/)
  - Full results JSON (saved to experiment/plots/)

Usage:
    # Evaluate on test split (default):
    python scripts/evaluate_model.py

    # Evaluate on val split:
    python scripts/evaluate_model.py --split val

    # Show top/bottom 15 species:
    python scripts/evaluate_model.py --top-n 15

    # Skip confusion matrix (faster):
    python scripts/evaluate_model.py --no-cm

    # Use a specific experiment:
    python scripts/evaluate_model.py --experiment experiments/bioclip2_full_20260716_160143
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the production inference module — no duplicate logic
from src.species_classifier import SpeciesClassifierInference, _find_best_checkpoint_dir
from classification.dataloader.fish_dataset import FishDatasetBuilder
from classification.utilities.device import get_device, log_gpu_memory
from classification.utilities.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(
    clf: SpeciesClassifierInference,
    loader,
    device: torch.device,
) -> dict:
    """
    Run batched evaluation over a DataLoader.

    Uses the loaded model directly for batch efficiency instead of
    calling clf.classify() per image.
    """
    from tqdm import tqdm
    from torch.amp import autocast

    all_preds: list[int] = []
    all_labels: list[int] = []
    top5_correct = 0
    total_images = 0
    total_time = 0.0

    clf._load_model() if clf._model is None else None  # ensure loaded
    model = clf._model
    model.eval()

    pbar = tqdm(loader, desc="Evaluating", unit="batch",
                bar_format="{l_bar}{bar:20}{r_bar}")

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels_dev = labels.to(device, non_blocking=True)

        t0 = time.perf_counter()
        with autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(images)                         # (B, C)
        total_time += time.perf_counter() - t0

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())

        if logits.shape[1] >= 5:
            _, top5 = logits.topk(5, dim=1)
            top5_correct += (top5 == labels_dev.unsqueeze(1)).any(dim=1).sum().item()

        total_images += labels.shape[0]

    pbar.close()

    return {
        "all_preds": np.array(all_preds),
        "all_labels": np.array(all_labels),
        "top5_correct": top5_correct,
        "total_images": total_images,
        "total_time_s": total_time,
    }


def compute_metrics(raw: dict, class_names: list[str]) -> dict:
    """Compute all classification metrics from raw predictions."""
    from sklearn.metrics import (
        accuracy_score, f1_score,
        precision_score, recall_score,
    )

    preds = raw["all_preds"]
    labels = raw["all_labels"]
    n = raw["total_images"]

    top1 = accuracy_score(labels, preds)
    top5 = raw["top5_correct"] / max(n, 1)
    ms_per_img = raw["total_time_s"] / max(n, 1) * 1000

    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    prec_macro = precision_score(labels, preds, average="macro", zero_division=0)
    prec_weighted = precision_score(labels, preds, average="weighted", zero_division=0)
    rec_macro = recall_score(labels, preds, average="macro", zero_division=0)
    rec_weighted = recall_score(labels, preds, average="weighted", zero_division=0)

    # Per-class
    f1_pc = f1_score(labels, preds, average=None, zero_division=0)
    prec_pc = precision_score(labels, preds, average=None, zero_division=0)
    rec_pc = recall_score(labels, preds, average=None, zero_division=0)
    support = np.bincount(labels, minlength=len(class_names))

    # Per-class accuracy = correct / support
    per_class_acc = np.zeros(len(class_names))
    for i in range(len(class_names)):
        mask = labels == i
        if mask.sum() > 0:
            per_class_acc[i] = (preds[mask] == i).mean()

    per_class = [
        {
            "class_idx": i,
            "species": class_names[i],
            "accuracy": round(float(per_class_acc[i]), 4),
            "f1": round(float(f1_pc[i]), 4),
            "precision": round(float(prec_pc[i]), 4),
            "recall": round(float(rec_pc[i]), 4),
            "support": int(support[i]),
        }
        for i in range(len(class_names))
    ]

    return {
        "top1_accuracy": round(top1, 4),
        "top5_accuracy": round(top5, 4),
        "precision_macro": round(prec_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_macro": round(rec_macro, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "ms_per_image": round(ms_per_img, 2),
        "total_images": n,
        "num_classes": len(class_names),
        "per_class": per_class,
    }


def save_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list[str],
    out_path: Path,
) -> None:
    """Save a normalized confusion matrix PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(labels, preds).astype(float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

        n = len(class_names)
        fig_size = max(22, n // 4)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title("Confusion Matrix — Normalized (row = true class)",
                     fontsize=14, pad=14)
        ax.set_xlabel("Predicted species", fontsize=10)
        ax.set_ylabel("True species", fontsize=10)

        # Use last word of species name for readability
        short = [c.split("_")[-1] for c in class_names]
        ticks = np.arange(n)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(short, rotation=90, fontsize=4)
        ax.set_yticklabels(short, fontsize=4)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Confusion matrix saved → %s", out_path)

    except Exception as exc:
        logger.warning("Confusion matrix skipped: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

def print_report(metrics: dict, split: str, top_n: int) -> None:
    W = 66
    print(f"\n{'═' * W}")
    print(f"  BioCLIP 2 Evaluation Report  ·  split={split.upper()}")
    print(f"{'═' * W}")
    print(f"\n  Total images  : {metrics['total_images']:,}")
    print(f"  Species       : {metrics['num_classes']}")
    print(f"  Inference     : {metrics['ms_per_image']:.1f} ms / image")

    print(f"\n  {'Metric':<32} {'Value':>10}")
    print(f"  {'-' * 44}")
    rows = [
        ("Top-1 Accuracy",        f"{metrics['top1_accuracy']:.1%}"),
        ("Top-5 Accuracy",        f"{metrics['top5_accuracy']:.1%}"),
        ("",                       ""),
        ("Precision  (macro)",    f"{metrics['precision_macro']:.4f}"),
        ("Precision  (weighted)", f"{metrics['precision_weighted']:.4f}"),
        ("Recall     (macro)",    f"{metrics['recall_macro']:.4f}"),
        ("Recall     (weighted)", f"{metrics['recall_weighted']:.4f}"),
        ("F1         (macro)",    f"{metrics['f1_macro']:.4f}"),
        ("F1         (weighted)", f"{metrics['f1_weighted']:.4f}"),
    ]
    for label, val in rows:
        if label:
            print(f"  {label:<32} {val:>10}")
        else:
            print()

    # Per-class: best and worst
    pc = sorted(metrics["per_class"], key=lambda x: x["f1"], reverse=True)

    def _row(r: dict) -> str:
        name = r["species"].replace("_", " ")
        return (f"  {name:<42} {r['accuracy']:>7.1%}"
                f" {r['f1']:>6.3f} {r['precision']:>6.3f}"
                f" {r['recall']:>6.3f} {r['support']:>5}")

    hdr = f"  {'Species':<42} {'Acc':>7} {'F1':>6} {'Prec':>6} {'Rec':>6} {'N':>5}"
    sep = f"  {'-' * 74}"

    print(f"\n  ┌─ Top {top_n} species by F1 {'─' * (W - 24)}┐")
    print(hdr)
    print(sep)
    for r in pc[:top_n]:
        print(_row(r))

    print(f"\n  ┌─ Bottom {top_n} species by F1 {'─' * (W - 27)}┐")
    print(hdr)
    print(sep)
    for r in pc[-top_n:]:
        print(_row(r))

    print(f"\n{'═' * W}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained BioCLIP 2 on test or val split"
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Path to experiment directory (auto-detected if omitted)",
    )
    parser.add_argument(
        "--split", choices=["val", "test"], default="test",
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for evaluation (default: 16)",
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Number of best/worst species to display (default: 10)",
    )
    parser.add_argument(
        "--no-cm", action="store_true",
        help="Skip confusion matrix generation",
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help="DataLoader num_workers (default: 2)",
    )
    args = parser.parse_args()

    set_seed(42, deterministic=False)
    device = get_device()
    log_gpu_memory("startup")

    # ── Resolve checkpoint dir ────────────────────────────────────────
    if args.experiment:
        exp_dir = Path(args.experiment)
        ckpt_dir = exp_dir / "checkpoints"
    else:
        ckpt_dir = _find_best_checkpoint_dir()
        # exp_dir is parent of checkpoints/
        exp_dir = ckpt_dir.parent if ckpt_dir.name == "checkpoints" else ckpt_dir.parent

    logger.info("Experiment : %s", exp_dir.name)
    logger.info("Checkpoint : %s", ckpt_dir)

    # ── Build inference object (lazy model load) ─────────────────────
    clf = SpeciesClassifierInference(checkpoint_dir=ckpt_dir, device="auto")
    clf._load_model()                           # explicit load for timing
    log_gpu_memory("after model load")

    # ── Build DataLoader ─────────────────────────────────────────────
    # CRITICAL: class indices in the DataLoader MUST match the training
    # class mapping exactly. We load class_mapping.json from the checkpoint
    # (written by the training engine) as the authoritative source of truth.
    # Never re-derive class ordering from disk — set.intersection() across
    # splits can produce a different order if any split is missing a species.
    import yaml

    # Load the training class mapping (species → index)
    class_mapping_path = (
        ckpt_dir.parent / "config" / "class_mapping.json"  # experiments/.../config/
    )
    if not class_mapping_path.exists():
        raise FileNotFoundError(
            f"class_mapping.json not found at {class_mapping_path}. "
            "This file is required to align DataLoader labels with model outputs."
        )

    with open(class_mapping_path) as f:
        name_to_idx: dict[str, int] = json.load(f)

    # Rebuild as a sorted list (index → species name) — matches training order
    training_class_names: list[str] = [
        name for name, _ in sorted(name_to_idx.items(), key=lambda x: x[1])
    ]

    # Validate that the checkpoint metadata agrees
    meta_class_names: list[str] = clf.class_names
    if training_class_names != meta_class_names:
        logger.error(
            "CLASS MAPPING MISMATCH: class_mapping.json has %d classes, "
            "checkpoint metadata has %d classes. First divergence at index %d.",
            len(training_class_names), len(meta_class_names),
            next(
                (i for i, (a, b) in enumerate(zip(training_class_names, meta_class_names)) if a != b),
                min(len(training_class_names), len(meta_class_names)),
            ),
        )
        raise RuntimeError(
            "class_mapping.json and best_accuracy_metadata.json disagree on "
            "class ordering. Evaluation cannot proceed safely."
        )

    # Load processed_dir path from dataset.yaml (clean YAML, no Python tags)
    processed_dir_rel = "datasets/processed"
    dataset_yaml = PROJECT_ROOT / "classification" / "config" / "dataset.yaml"
    if dataset_yaml.exists():
        try:
            with open(dataset_yaml) as f:
                ds_cfg = yaml.safe_load(f) or {}
            processed_dir_rel = (
                ds_cfg.get("dataset", {}).get("processed_dir", processed_dir_rel)
            )
        except yaml.YAMLError as exc:
            logger.warning("Could not parse dataset.yaml (%s); using default", exc)

    processed_dir = PROJECT_ROOT / processed_dir_rel

    # ── Diagnostic header (required) ─────────────────────────────────
    print("\n" + "─" * 66)
    print("  EVALUATION DIAGNOSTIC")
    print("─" * 66)
    print(f"  Checkpoint      : {ckpt_dir / 'best_accuracy.pt'}")
    print(f"  Num classes     : {len(training_class_names)}")
    print(f"  LoRA active     : {any('lora' in n for n, _ in clf._model.named_modules())}")
    print(f"  Preprocessor    : {clf._transform}")
    print(f"  Class mapping   : class_mapping.json ({class_mapping_path.name})")
    print(f"\n  First 10 class mappings (index → species):")
    for i, name in enumerate(training_class_names[:10]):
        print(f"    [{i:3d}] {name}")
    print("─" * 66 + "\n")

    # ── Build dataset with the EXACT training class ordering ──────────
    # We bypass FishDatasetBuilder's auto-ordering and build FishSpeciesDataset
    # directly, using training_class_names as the authoritative class list.
    from classification.dataloader.fish_dataset import FishSpeciesDataset

    split_dir = processed_dir / args.split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}"
        )

    _IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    image_paths: list = []
    labels: list = []
    missing_species = []

    for species_name in training_class_names:
        species_dir = split_dir / species_name
        if not species_dir.exists():
            missing_species.append(species_name)
            continue
        class_idx = name_to_idx[species_name]
        images = sorted([
            f for f in species_dir.iterdir()
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
        ])
        for img_path in images:
            image_paths.append(img_path)
            labels.append(class_idx)

    if missing_species:
        logger.warning(
            "%d species from training not found in %s split: %s",
            len(missing_species), args.split, missing_species,
        )

    val_transform = clf._transform          # same transform as inference
    dataset = FishSpeciesDataset(
        image_paths=image_paths,
        labels=labels,
        class_names=training_class_names,
        transform=val_transform,
    )

    from torch.utils.data import DataLoader as TorchDataLoader
    loader = TorchDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    logger.info(
        "Dataset: %d images, %d species, %d missing from %s split",
        len(image_paths), len(training_class_names) - len(missing_species),
        len(missing_species), args.split,
    )
    # ── Run evaluation ────────────────────────────────────────────────
    raw = run_evaluation(clf, loader, device)
    # Use training_class_names as the authoritative label list
    # (validated above to be identical to clf.class_names)
    metrics = compute_metrics(raw, training_class_names)
    log_gpu_memory("after evaluation")

    # ── Print report ─────────────────────────────────────────────────
    print_report(metrics, args.split, args.top_n)

    # ── Save results ──────────────────────────────────────────────────
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    results_path = plots_dir / f"evaluation_{args.split}.json"
    with open(results_path, "w") as f:
        save_data = {k: v for k, v in metrics.items()
                     if k != "per_class"}
        save_data["per_class"] = metrics["per_class"]
        save_data["experiment"] = exp_dir.name
        save_data["split"] = args.split
        json.dump(save_data, f, indent=2)
    logger.info("Results saved → %s", results_path)

    # ── Confusion matrix ──────────────────────────────────────────────
    if not args.no_cm:
        save_confusion_matrix(
            raw["all_labels"],
            raw["all_preds"],
            clf.class_names,
            plots_dir / f"confusion_matrix_{args.split}.png",
        )


if __name__ == "__main__":
    main()
