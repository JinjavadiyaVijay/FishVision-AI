"""
validate_model.py — Full evaluation of the trained BioCLIP 2 species classifier.

Loads the best checkpoint from the latest experiment and evaluates on the
held-out test set, reporting:
  - Top-1 and Top-5 accuracy
  - Macro and weighted F1
  - Per-class breakdown (sorted by F1)
  - Confusion matrix heatmap (saved to experiment/plots/)
  - Inference speed (ms/image)
  - GPU memory usage

Usage:
    python scripts/validate_model.py
    python scripts/validate_model.py --experiment experiments/bioclip2_full_20260716_160143
    python scripts/validate_model.py --split val     # validate on val set
    python scripts/validate_model.py --split test    # validate on test set (default)
    python scripts/validate_model.py --top-n 20      # show top/bottom N species
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from classification.models.backbone import create_backbone
from classification.models.classifier import ClassifierHead, SpeciesClassifier
from classification.dataloader.fish_dataset import FishDatasetBuilder
from classification.utilities.device import get_device, log_gpu_memory
from classification.utilities.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_best_experiment() -> Path:
    """Auto-detect the latest experiment with a best_accuracy.pt checkpoint."""
    exp_root = PROJECT_ROOT / "experiments"
    candidates = sorted(
        [d for d in exp_root.iterdir()
         if d.is_dir() and (d / "checkpoints" / "best_accuracy.pt").exists()],
        key=lambda d: d.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No experiment with best_accuracy.pt found in {exp_root}"
        )
    return candidates[0]


def load_model(checkpoint_dir: Path, device: torch.device):
    """Load SpeciesClassifier from a checkpoint directory."""
    metadata_path = checkpoint_dir / "best_accuracy_metadata.json"
    checkpoint_path = checkpoint_dir / "best_accuracy.pt"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with open(metadata_path) as f:
        meta = json.load(f)

    num_classes = meta["num_classes"]
    class_names = meta["class_names"]
    logger.info("Checkpoint: epoch=%d, val_accuracy=%.4f, num_classes=%d",
                meta["metrics"]["epoch"],
                meta["metrics"]["val_accuracy"],
                num_classes)

    # Load checkpoint state dict
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    return ckpt, class_names, num_classes, meta


def build_model_from_checkpoint(ckpt: dict, num_classes: int, config: dict, device: torch.device):
    """Rebuild model architecture and load weights."""
    from classification.config import load_configs
    backbone = create_backbone(config)

    cls_cfg = config.get("classifier", {})
    classifier = ClassifierHead(
        embed_dim=backbone.embed_dim,
        num_classes=num_classes,
        head_type=cls_cfg.get("type", "linear"),
        dropout=cls_cfg.get("dropout", 0.1),
    )

    model = SpeciesClassifier(backbone=backbone, classifier=classifier)

    # Load state dict
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, loader, device, class_names, use_amp=True):
    """Run evaluation and collect all predictions."""
    from torch.cuda.amp import autocast

    all_preds = []
    all_labels = []
    all_top5_correct = 0
    total = 0
    total_time = 0.0

    from tqdm import tqdm
    pbar = tqdm(loader, desc="Evaluating", unit="batch", bar_format="{l_bar}{bar:20}{r_bar}")

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        t0 = time.perf_counter()
        with autocast("cuda", enabled=use_amp and torch.cuda.is_available()):
            logits = model(images)
        total_time += time.perf_counter() - t0

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if logits.size(1) >= 5:
            _, top5 = logits.topk(5, dim=1)
            all_top5_correct += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)

    pbar.close()

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, classification_report
    )

    top1_acc = accuracy_score(all_labels, all_preds)
    top5_acc = all_top5_correct / max(total, 1)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    # Per-class F1
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    prec_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    rec_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    support = np.bincount(all_labels, minlength=len(class_names))

    per_class = [
        {
            "class": class_names[i],
            "f1": round(float(f1_per_class[i]), 4),
            "precision": round(float(prec_per_class[i]), 4),
            "recall": round(float(rec_per_class[i]), 4),
            "support": int(support[i]),
        }
        for i in range(len(class_names))
    ]

    ms_per_image = (total_time / max(total, 1)) * 1000

    return {
        "top1_accuracy": round(top1_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "total_images": total,
        "ms_per_image": round(ms_per_image, 2),
        "per_class": per_class,
        "all_preds": all_preds,
        "all_labels": all_labels,
    }


def save_confusion_matrix(all_labels, all_preds, class_names, out_path: Path):
    """Save confusion matrix as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(all_labels, all_preds)
        # Normalize per row
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

        n = len(class_names)
        fig_size = max(20, n // 4)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_title("Confusion Matrix (row-normalized)", fontsize=14, pad=12)
        ax.set_xlabel("Predicted species", fontsize=10)
        ax.set_ylabel("True species", fontsize=10)

        tick_marks = np.arange(n)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels([c.split("_")[-1] for c in class_names],
                            rotation=90, fontsize=4)
        ax.set_yticklabels([c.split("_")[-1] for c in class_names], fontsize=4)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Confusion matrix saved: %s", out_path)
        return True
    except Exception as e:
        logger.warning("Could not save confusion matrix: %s", e)
        return False


def print_results(results: dict, class_names: list, top_n: int, split: str) -> None:
    """Print a comprehensive human-readable results table."""
    print(f"\n{'=' * 65}")
    print(f"  BioCLIP 2 — Test Evaluation Results ({split.upper()} split)")
    print(f"{'=' * 65}")
    print(f"\n  Classes evaluated: {results['total_images']} images × {len(class_names)} species")
    print(f"\n  {'Metric':<30} {'Value':>10}")
    print(f"  {'-' * 42}")
    print(f"  {'Top-1 Accuracy':<30} {results['top1_accuracy']:>9.1%}")
    print(f"  {'Top-5 Accuracy':<30} {results['top5_accuracy']:>9.1%}")
    print(f"  {'F1 Macro':<30} {results['f1_macro']:>10.4f}")
    print(f"  {'F1 Weighted':<30} {results['f1_weighted']:>10.4f}")
    print(f"  {'Precision Macro':<30} {results['precision_macro']:>10.4f}")
    print(f"  {'Recall Macro':<30} {results['recall_macro']:>10.4f}")
    print(f"  {'Inference speed':<30} {results['ms_per_image']:>7.1f} ms/img")

    # Top-N best species
    per_class = sorted(results["per_class"], key=lambda x: x["f1"], reverse=True)
    print(f"\n  Top {top_n} Best Performing Species:")
    print(f"  {'Species':<45} {'F1':>6} {'Prec':>6} {'Rec':>6} {'N':>5}")
    print(f"  {'-' * 70}")
    for row in per_class[:top_n]:
        name = row["class"].replace("_", " ")
        print(f"  {name:<45} {row['f1']:>6.3f} {row['precision']:>6.3f} {row['recall']:>6.3f} {row['support']:>5}")

    # Bottom-N worst species
    print(f"\n  Bottom {top_n} Worst Performing Species:")
    print(f"  {'Species':<45} {'F1':>6} {'Prec':>6} {'Rec':>6} {'N':>5}")
    print(f"  {'-' * 70}")
    for row in per_class[-top_n:]:
        name = row["class"].replace("_", " ")
        print(f"  {name:<45} {row['f1']:>6.3f} {row['precision']:>6.3f} {row['recall']:>6.3f} {row['support']:>5}")

    print(f"\n{'=' * 65}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate trained BioCLIP 2 model on test/val set")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Path to experiment directory (auto-detected if not given)")
    parser.add_argument("--split", choices=["val", "test"], default="test",
                        help="Dataset split to evaluate on (default: test)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for evaluation (default: 16)")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of best/worst species to display (default: 10)")
    parser.add_argument("--no-confusion-matrix", action="store_true",
                        help="Skip confusion matrix generation")
    args = parser.parse_args()

    set_seed(42, deterministic=False)
    device = get_device()
    log_gpu_memory("startup")

    # ── Find experiment ───────────────────────────────────────────
    if args.experiment:
        exp_dir = Path(args.experiment)
    else:
        exp_dir = find_best_experiment()

    logger.info("Experiment: %s", exp_dir.name)
    checkpoint_dir = exp_dir / "checkpoints"

    # ── Load config ───────────────────────────────────────────────
    import yaml
    config = {}
    for cfg_name in ("model", "training", "augmentation", "dataset", "logging"):
        cfg_path = exp_dir / "config" / f"{cfg_name}.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                config.update(yaml.safe_load(f) or {})
        else:
            # Fall back to active config
            cfg_path = PROJECT_ROOT / "classification" / "config" / f"{cfg_name}.yaml"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    config.update(yaml.safe_load(f) or {})

    # ── Load checkpoint metadata + weights ───────────────────────
    logger.info("Loading best checkpoint...")
    ckpt, class_names, num_classes, meta = load_model(checkpoint_dir, device)
    log_gpu_memory("after checkpoint load")

    # ── Build model ───────────────────────────────────────────────
    logger.info("Building model (%d classes)...", num_classes)
    backbone = create_backbone(config)
    backbone.freeze()

    # Apply LoRA if checkpoint was trained with LoRA
    lora_cfg = config.get("lora", {})
    if lora_cfg.get("enabled", True):
        try:
            from peft import LoraConfig, get_peft_model
            target_modules = backbone.get_lora_target_modules()
            lora_config = LoraConfig(
                r=lora_cfg.get("rank", 16),
                lora_alpha=lora_cfg.get("alpha", 32),
                lora_dropout=0.0,  # No dropout at inference
                target_modules=target_modules,
                bias=lora_cfg.get("bias", "none"),
            )
            backbone.visual = get_peft_model(backbone.visual, lora_config)
            logger.info("LoRA applied (rank=%d)", lora_cfg.get("rank", 16))
        except Exception as e:
            logger.warning("LoRA not applied: %s", e)

    cls_cfg = config.get("classifier", {})
    classifier = ClassifierHead(
        embed_dim=backbone.embed_dim,
        num_classes=num_classes,
        head_type=cls_cfg.get("type", "linear"),
        dropout=0.0,  # No dropout at inference
    )
    model = SpeciesClassifier(backbone=backbone, classifier=classifier)

    # Load weights
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model = model.to(device)
    model.eval()
    log_gpu_memory("after model to device")

    # ── Build DataLoader ─────────────────────────────────────────
    logger.info("Building %s dataset...", args.split)
    processed_dir = PROJECT_ROOT / config.get("dataset", {}).get("processed_dir", "datasets/processed")

    builder = FishDatasetBuilder(
        processed_dir=processed_dir,
        exclude_species=["Lutjanus_erythropterus", "Sphyraena_qenie"],
    )

    val_tf = backbone.get_transforms(train=False)
    loader = builder.build_dataloader(
        args.split,
        transform=val_tf,
        batch_size=args.batch_size,
        num_workers=2,
        weighted_sampling=False,
    )

    logger.info("Evaluating on %s split: %d images", args.split, len(loader.dataset))
    log_gpu_memory("before evaluation")

    # ── Evaluate ─────────────────────────────────────────────────
    results = evaluate(model, loader, device, class_names)
    log_gpu_memory("after evaluation")

    # ── Print results ─────────────────────────────────────────────
    print_results(results, class_names, args.top_n, args.split)

    # ── Save JSON results ─────────────────────────────────────────
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    results_to_save = {k: v for k, v in results.items() if k not in ("all_preds", "all_labels")}
    results_path = plots_dir / f"test_results_{args.split}.json"
    with open(results_path, "w") as f:
        json.dump(results_to_save, f, indent=2)
    logger.info("Results saved: %s", results_path)

    # ── Confusion matrix ──────────────────────────────────────────
    if not args.no_confusion_matrix:
        logger.info("Generating confusion matrix...")
        save_confusion_matrix(
            results["all_labels"],
            results["all_preds"],
            class_names,
            plots_dir / f"confusion_matrix_{args.split}.png",
        )


if __name__ == "__main__":
    main()
