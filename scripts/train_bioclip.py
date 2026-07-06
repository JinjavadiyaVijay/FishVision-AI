"""
train_bioclip.py — Main training script for BioCLIP 2 species classifier.

Usage:
    python scripts/train_bioclip.py --smoke-test
    python scripts/train_bioclip.py --debug
    python scripts/train_bioclip.py --mode lora --epochs 30
    python scripts/train_bioclip.py --resume experiments/.../checkpoints/bioclip2_last

Training Modes:
    --smoke-test     — 30 batches, 1 epoch, validates full pipeline in <2 min
    --debug          — 500 batches, 1 epoch, quick dataset/model verification
    (default)        — Full training with all epochs

Model Modes:
    lora             — LoRA fine-tuning (default, recommended for RTX 3050)
    linear_probe     — Frozen backbone + trainable classifier head only
    full_finetune    — All parameters trainable (requires >6GB VRAM)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image

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


def log_reproducibility_state(seed: int, deterministic: bool) -> None:
    """Log deterministic settings that affect experiment reproducibility."""
    logger.info(
        "Reproducibility: seed=%d deterministic=%s cudnn.deterministic=%s cudnn.benchmark=%s",
        seed,
        deterministic,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )


def recommend_batch_size(mode: str, current_batch_size: int) -> int:
    """Return a conservative batch-size recommendation for the active GPU."""
    if not torch.cuda.is_available():
        logger.info("Batch-size recommendation: CPU mode, keep batch_size=%d", current_batch_size)
        return current_batch_size

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if mode == "full_finetune":
        recommended = 2 if total_gb <= 6.5 else 4
    elif mode == "lora":
        recommended = 8 if total_gb <= 6.5 else 16
    else:
        recommended = 8 if total_gb <= 6.5 else 16

    logger.info(
        "Batch-size recommendation: current=%d recommended=%d for mode=%s on %.1fGB VRAM",
        current_batch_size, recommended, mode, total_gb,
    )
    return recommended


def validate_dataset(builder: FishDatasetBuilder, config: dict) -> bool:
    """
    Pre-training dataset validation (Task 10).

    Checks class counts, split consistency, mapping consistency, and optional image readability.
    Returns True if validation passes, False otherwise.
    """
    logger.info("=" * 60)
    logger.info("DATASET VALIDATION")
    logger.info("-" * 60)

    issues = []
    summary = builder.get_split_summary()
    data_cfg = config.get("dataset", {})
    val_cfg = config.get("validation", {})

    # Check class count
    num_classes = summary["num_classes"]
    if num_classes == 0:
        issues.append("FATAL: No classes found in dataset")
    else:
        logger.info("  Classes: %d", num_classes)

    # Check each split
    for split_name in ("train", "val", "test"):
        if split_name not in summary:
            issues.append(f"FATAL: Missing '{split_name}' split")
            continue

        split_info = summary[split_name]
        logger.info("  %s: %d images, %d classes, min=%d/class, max=%d/class",
                     split_name, split_info["total_images"],
                     split_info["classes_with_images"],
                     split_info["min_per_class"],
                     split_info["max_per_class"])

        if split_info["total_images"] == 0:
            issues.append(f"FATAL: '{split_name}' split has 0 images")
        if split_info["classes_with_images"] != num_classes:
            issues.append(
                f"WARNING: '{split_name}' has {split_info['classes_with_images']}"
                f" classes, expected {num_classes}"
            )
        if split_info["min_per_class"] == 0:
            issues.append(f"WARNING: '{split_name}' has classes with 0 images")

    # Check class mapping against master catalog when present.
    catalog_path = PROJECT_ROOT / data_cfg.get("master_catalog", "datasets/metadata/master_species_catalog.json")
    if catalog_path.exists():
        with open(catalog_path) as f:
            catalog = json.load(f)
        if isinstance(catalog, dict):
            catalog_names = set(catalog.keys())
        else:
            catalog_names = {
                item.get("scientific_name") or item.get("species") or item.get("class_name") or item.get("name")
                for item in catalog
                if isinstance(item, dict)
            }
        catalog_names.discard(None)
        normalized_catalog = {str(name).replace(" ", "_") for name in catalog_names}
        missing_from_catalog = sorted(set(builder.class_names) - normalized_catalog)
        if missing_from_catalog:
            issues.append(
                f"FATAL: {len(missing_from_catalog)} training classes missing from master catalog"
            )
        else:
            logger.info("  Class mapping: consistent with %s", catalog_path)
    else:
        issues.append(f"WARNING: Master catalog not found: {catalog_path}")

    # Optional readability check. Defaults to config values so normal startup stays fast.
    check_readability = bool(val_cfg.get("check_readability", False))
    sample_frac = float(val_cfg.get("readability_sample_frac", 0.05))
    if check_readability:
        rng = random.Random(config.get("training", {}).get("seed", 42))
        all_paths = []
        for paths, _labels in builder._split_data.values():
            all_paths.extend(paths)
        sample_size = len(all_paths) if sample_frac >= 1.0 else max(1, int(len(all_paths) * sample_frac))
        sample_paths = rng.sample(all_paths, min(sample_size, len(all_paths)))
        logger.info("  Readability: checking %d/%d images", len(sample_paths), len(all_paths))
        corrupted = []
        for image_path in sample_paths:
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception as exc:
                corrupted.append((str(image_path), str(exc)))
                if len(corrupted) >= 10:
                    break
        if corrupted:
            issues.append(f"FATAL: Found unreadable/corrupted images, first={corrupted[0][0]}")
    else:
        logger.info("  Readability: skipped (validation.check_readability=false)")

    # Report
    if issues:
        for issue in issues:
            if issue.startswith("FATAL"):
                logger.error("  %s", issue)
            else:
                logger.warning("  %s", issue)

    fatal = any(i.startswith("FATAL") for i in issues)
    if fatal:
        logger.error("DATASET VALIDATION FAILED — cannot proceed")
    else:
        logger.info("  DATASET VALIDATION PASSED")
    logger.info("=" * 60)

    return not fatal


def setup_model(config: dict, num_classes: int, mode: str, device: torch.device):
    """
    Build backbone + classifier with the specified training mode.

    Returns (model, experiment_base_name)
    """
    backbone = create_backbone(config)
    log_gpu_memory("after backbone load")

    cls_cfg = config.get("classifier", {})
    classifier = ClassifierHead(
        embed_dim=backbone.embed_dim,
        num_classes=num_classes,
        head_type=cls_cfg.get("type", "linear"),
        dropout=cls_cfg.get("dropout", 0.1),
    )

    if mode == "linear_probe":
        backbone.freeze()
        base_name = "bioclip2_linear_probe"
        logger.info("Mode: LINEAR PROBE (backbone frozen)")

    elif mode == "lora":
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
        backbone.unfreeze()
        base_name = "bioclip2_full_ft"
        logger.info("Mode: FULL FINE-TUNE (all parameters trainable)")
        logger.warning("Full fine-tuning requires >6GB VRAM. May OOM on RTX 3050.")

    else:
        raise ValueError(f"Unknown mode: '{mode}'. Use 'lora', 'linear_probe', or 'full_finetune'")

    model = SpeciesClassifier(backbone=backbone, classifier=classifier)

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

    # Training run modes (Task 1 + Task 9)
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument("--smoke-test", action="store_true",
                          help="Quick pipeline validation: 30 batches, 1 epoch, <2 min")
    run_mode.add_argument("--debug", action="store_true",
                          help="Debug training: 500 batches, 1 epoch")
    run_mode.add_argument("--max-steps", type=int, default=None,
                          help="Custom max batches per epoch")

    args = parser.parse_args()

    # ── Setup logging ─────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Determine run mode and max_steps ──────────────────────
    max_steps = args.max_steps
    run_mode_name = "full"

    if args.smoke_test:
        max_steps = 30
        run_mode_name = "smoke_test"
        if args.epochs is None:
            args.epochs = 1
        logger.info("=" * 60)
        logger.info("SMOKE TEST MODE — 30 batches, 1 epoch")
        logger.info("=" * 60)

    elif args.debug:
        max_steps = 500
        run_mode_name = "debug"
        if args.epochs is None:
            args.epochs = 1
        logger.info("=" * 60)
        logger.info("DEBUG MODE — 500 batches, 1 epoch")
        logger.info("=" * 60)

    # ── Load config ───────────────────────────────────────────
    config = load_configs()
    config = apply_cli_overrides(config, args)

    train_cfg = config.get("training", {})
    seed = train_cfg.get("seed", 42)
    deterministic = train_cfg.get("deterministic", True)
    set_seed(seed, deterministic=deterministic)
    log_reproducibility_state(seed, deterministic)

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

    # ── Dataset validation (Task 10) ──────────────────────────
    if not validate_dataset(builder, config):
        logger.error("Aborting due to dataset validation failure.")
        sys.exit(1)

    # ── Experiment directory ──────────────────────────────────
    experiment_base_name = args.experiment_name or f"bioclip2_{run_mode_name}"
    experiment_dir = create_experiment_dir(
        base_dir=PROJECT_ROOT / config.get("experiment", {}).get("base_dir", "experiments"),
        base_name=experiment_base_name,
    )
    builder.save_class_mapping(experiment_dir / "config" / "class_mapping.json")

    # Save split summary
    summary = builder.get_split_summary()
    (experiment_dir / "config").mkdir(parents=True, exist_ok=True)
    with open(experiment_dir / "config" / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Model ─────────────────────────────────────────────────
    model, mode_name = setup_model(config, builder.num_classes, args.mode, device)

    # ── Transforms ────────────────────────────────────────────
    train_tf = model.backbone.get_transforms(train=True)
    val_tf = model.backbone.get_transforms(train=False)

    # ── DataLoaders (Task 3 — optimized settings) ─────────────
    batch_size = train_cfg.get("batch_size", 8)
    num_workers = train_cfg.get("num_workers", 2)
    use_weighted = not args.no_weighted_sampler
    recommend_batch_size(args.mode, batch_size)

    # Windows DataLoader constraints:
    # - persistent_workers requires num_workers > 0
    # - prefetch_factor requires num_workers > 0
    # - Too many workers on Windows causes handle exhaustion
    # Optimal for RTX 3050 Laptop + NVMe/SSD: 2-4 workers
    use_persistent = num_workers > 0
    use_prefetch = num_workers > 0

    logger.info("DataLoader: batch_size=%d, workers=%d, pin_memory=True, "
                "persistent_workers=%s, prefetch_factor=%s",
                batch_size, num_workers, use_persistent,
                2 if use_prefetch else "N/A")

    train_loader = builder.build_dataloader(
        "train", transform=train_tf, batch_size=batch_size,
        num_workers=num_workers, weighted_sampling=use_weighted,
        seed=seed,
    )
    val_loader = builder.build_dataloader(
        "val", transform=val_tf, batch_size=batch_size * 2,
        num_workers=num_workers, weighted_sampling=False,
        seed=seed,
    )

    logger.info("Train: %d batches (batch_size=%d)", len(train_loader), batch_size)
    logger.info("Val:   %d batches (batch_size=%d)", len(val_loader), batch_size * 2)

    if max_steps:
        logger.info("Max steps per epoch: %d (of %d total)", max_steps, len(train_loader))

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
        max_steps=max_steps,
    )

    # Resume if requested
    if args.resume:
        engine.resume_from(args.resume)

    # ── Train ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    results = engine.train()
    elapsed = time.perf_counter() - t0

    # ── Final summary ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("  Run mode:     %s", run_mode_name)
    logger.info("  Experiment:   %s", experiment_dir.name)
    logger.info("  Epochs:       %d", results["total_epochs"])
    logger.info("  Wall time:    %.1f seconds (%.1f min)", elapsed, elapsed / 60)
    logger.info("  Best accuracy: %.4f", results["best_metrics"].get("val_accuracy", 0))
    logger.info("  Best F1 macro: %.4f", results["best_metrics"].get("val_f1_macro", 0))
    logger.info("  Checkpoints:  %s", engine.checkpoint_dir)

    if "throughput" in results and results["throughput"]:
        tp = results["throughput"]
        logger.info("  Throughput:   %.1f img/sec", tp.get("images_per_sec", 0))
        logger.info("  Data loading: %.1f%% of time", tp.get("data_loading_pct", 0))

    logger.info("=" * 60)

    if args.smoke_test:
        if elapsed < 120:
            logger.info("SMOKE TEST PASSED (%.1fs < 2min target)", elapsed)
        else:
            logger.warning("SMOKE TEST SLOW (%.1fs > 2min target)", elapsed)


if __name__ == "__main__":
    main()
