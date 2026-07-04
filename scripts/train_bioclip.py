"""
train_bioclip.py — Main training script for BioCLIP 2 species classifier.

Usage:
    python scripts/train_bioclip.py
    python scripts/train_bioclip.py --epochs 10 --batch-size 4
    python scripts/train_bioclip.py --resume experiments/bioclip2_lora_20260704/checkpoints/bioclip2_last
    python scripts/train_bioclip.py --mode linear_probe
    python scripts/train_bioclip.py --mode full_finetune

Modes:
    lora           — LoRA fine-tuning (default, recommended for RTX 3050)
    linear_probe   — Frozen backbone + trainable classifier head only
    full_finetune  — All parameters trainable (requires >6GB VRAM)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

# ── Path bootstrap ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from classification.dataloader.fish_dataset import FishDatasetBuilder
from classification.models.backbone import create_backbone
from classification.models.classifier import ClassifierHead, SpeciesClassifier
from classification.trainer.training_engine import TrainingEngine, create_experiment_dir
from classification.utilities.device import get_device, log_gpu_memory
from classification.utilities.seed import set_seed

CONFIG_DIR = PROJECT_ROOT / "classification" / "config"
logger = logging.getLogger(__name__)


def load_configs() -> dict:
    """Load and merge all YAML config files."""
    merged = {}
    for config_name in ("model", "training", "augmentation", "dataset", "logging"):
        config_path = CONFIG_DIR / f"{config_name}.yaml"
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
                merged.update(data)
    return merged


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply CLI argument overrides to config."""
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        config.setdefault("optimizer", {})["learning_rate"] = args.lr
    if args.workers is not None:
        config.setdefault("training", {})["num_workers"] = args.workers
    if args.seed is not None:
        config.setdefault("training", {})["seed"] = args.seed
    return config


def setup_model(config: dict, num_classes: int, mode: str, device: torch.device):
    """
    Build backbone + classifier with the specified training mode.

    Returns (model, experiment_base_name)
    """
    # Load backbone
    backbone = create_backbone(config)
    log_gpu_memory("after backbone load")

    # Classifier head
    cls_cfg = config.get("classifier", {})
    classifier = ClassifierHead(
        embed_dim=backbone.embed_dim,
        num_classes=num_classes,
        head_type=cls_cfg.get("type", "linear"),
        dropout=cls_cfg.get("dropout", 0.1),
    )

    if mode == "linear_probe":
        # Freeze backbone entirely
        backbone.freeze()
        base_name = "bioclip2_linear_probe"
        logger.info("Mode: LINEAR PROBE (backbone frozen)")

    elif mode == "lora":
        # Freeze backbone, then apply LoRA
        backbone.freeze()
        lora_cfg = config.get("lora", {})
        target_modules = backbone.get_lora_target_modules()

        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=lora_cfg.get("rank", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.1),
            target_modules=target_modules,
            bias=lora_cfg.get("bias", "none"),
        )
        backbone.visual = get_peft_model(backbone.visual, lora_config)
        base_name = f"bioclip2_lora_r{lora_cfg.get('rank', 16)}"
        logger.info("Mode: LoRA (rank=%d, alpha=%d)",
                     lora_cfg.get("rank", 16), lora_cfg.get("alpha", 32))

    elif mode == "full_finetune":
        # All parameters trainable
        backbone.unfreeze()
        base_name = "bioclip2_full_ft"
        logger.info("Mode: FULL FINE-TUNE (all parameters trainable)")
        logger.warning("Full fine-tuning requires >6GB VRAM. May OOM on RTX 3050.")

    else:
        raise ValueError(f"Unknown mode: '{mode}'. Use 'lora', 'linear_probe', or 'full_finetune'")

    model = SpeciesClassifier(backbone=backbone, classifier=classifier)

    # Log parameter summary
    summary = model.get_param_summary()
    logger.info("Model: %d total params, %d trainable (%.2f%%)",
                 summary["total_params"], summary["total_trainable"],
                 summary["trainable_pct"])

    return model, base_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train BioCLIP 2 species classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["lora", "linear_probe", "full_finetune"],
                        default="lora", help="Training mode")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--workers", type=int, default=None, help="Override num_workers")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint directory")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Override experiment base name")
    parser.add_argument("--no-weighted-sampler", action="store_true",
                        help="Disable weighted random sampling")
    args = parser.parse_args()

    # ── Setup logging ─────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Load config ───────────────────────────────────────────
    config = load_configs()
    config = apply_cli_overrides(config, args)

    train_cfg = config.get("training", {})
    seed = train_cfg.get("seed", 42)
    set_seed(seed, deterministic=train_cfg.get("deterministic", True))

    # ── Device ────────────────────────────────────────────────
    device = get_device()

    # ── Dataset ───────────────────────────────────────────────
    logger.info("Building dataset...")
    exclude_species = ["Lutjanus_erythropterus", "Sphyraena_qenie"]
    data_cfg = config.get("dataset", {})
    processed_dir = PROJECT_ROOT / data_cfg.get("processed_dir", "datasets/processed")

    builder = FishDatasetBuilder(
        processed_dir=processed_dir,
        exclude_species=exclude_species,
    )

    # Save class mapping
    experiment_base_name = args.experiment_name or "bioclip2"
    experiment_dir = create_experiment_dir(
        base_dir=PROJECT_ROOT / config.get("experiment", {}).get("base_dir", "experiments"),
        base_name=experiment_base_name,
    )
    builder.save_class_mapping(experiment_dir / "config" / "class_mapping.json")

    # Save split summary
    import json
    summary = builder.get_split_summary()
    with open(experiment_dir / "config" / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Model ─────────────────────────────────────────────────
    model, mode_name = setup_model(config, builder.num_classes, args.mode, device)

    # Update experiment name with mode
    if args.experiment_name is None:
        # Rename experiment dir with the mode-specific name
        new_dir = experiment_dir.parent / f"{mode_name}_{experiment_dir.name.split('_', 2)[-1]}"
        if not new_dir.exists():
            experiment_dir.rename(new_dir)
            experiment_dir = new_dir

    # ── Transforms ────────────────────────────────────────────
    train_tf = model.backbone.get_transforms(train=True)
    val_tf = model.backbone.get_transforms(train=False)

    # ── DataLoaders ───────────────────────────────────────────
    batch_size = train_cfg.get("batch_size", 8)
    num_workers = train_cfg.get("num_workers", 2)
    use_weighted = not args.no_weighted_sampler

    train_loader = builder.build_dataloader(
        "train", transform=train_tf, batch_size=batch_size,
        num_workers=num_workers, weighted_sampling=use_weighted,
    )
    val_loader = builder.build_dataloader(
        "val", transform=val_tf, batch_size=batch_size * 2,
        num_workers=num_workers, weighted_sampling=False,
    )

    logger.info("Train: %d batches (batch_size=%d)", len(train_loader), batch_size)
    logger.info("Val:   %d batches (batch_size=%d)", len(val_loader), batch_size * 2)

    # ── Training engine ───────────────────────────────────────
    engine = TrainingEngine(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        experiment_dir=experiment_dir,
        class_names=builder.class_names,
        device=device,
        class_weights=builder.class_weights if use_weighted else None,
    )

    # Resume if requested
    if args.resume:
        engine.resume_from(args.resume)

    # ── Train ─────────────────────────────────────────────────
    results = engine.train()

    # ── Final summary ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("  Experiment: %s", experiment_dir.name)
    logger.info("  Epochs: %d", results["total_epochs"])
    logger.info("  Time: %.1f minutes", results["total_time_minutes"])
    logger.info("  Best accuracy: %.4f", results["best_metrics"].get("val_accuracy", 0))
    logger.info("  Best F1 macro: %.4f", results["best_metrics"].get("val_f1_macro", 0))
    logger.info("  Checkpoints: %s", engine.checkpoint_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
