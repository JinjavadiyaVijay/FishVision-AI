"""
Evaluate BioCLIP 2 zero-shot classification on the processed fish dataset.

This establishes the baseline that linear probe and LoRA runs should beat.
It does not train or modify checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.amp import autocast
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from classification.dataloader.fish_dataset import FishDatasetBuilder
from classification.trainer.training_engine import create_experiment_dir
from classification.utilities.device import get_device, log_gpu_memory
from classification.utilities.seed import set_seed

CONFIG_DIR = PROJECT_ROOT / "classification" / "config"
logger = logging.getLogger(__name__)


def load_configs() -> dict:
    merged = {}
    for config_name in ("model", "training", "dataset", "logging"):
        config_path = CONFIG_DIR / f"{config_name}.yaml"
        if config_path.exists():
            with open(config_path) as f:
                merged.update(yaml.safe_load(f) or {})
    return merged


def build_prompts(class_names: list[str]) -> list[str]:
    return [
        f"a photo of a {class_name.replace('_', ' ')} fish"
        for class_name in class_names
    ]


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    import open_clip

    config = load_configs()
    train_cfg = config.get("training", {})
    seed = args.seed if args.seed is not None else train_cfg.get("seed", 42)
    set_seed(seed, deterministic=train_cfg.get("deterministic", True))

    device = get_device()
    data_cfg = config.get("dataset", {})
    processed_dir = PROJECT_ROOT / data_cfg.get("processed_dir", "datasets/processed")
    builder = FishDatasetBuilder(
        processed_dir=processed_dir,
        exclude_species=["Lutjanus_erythropterus", "Sphyraena_qenie"],
    )

    backbone_cfg = config.get("backbone", {})
    model_name = backbone_cfg.get("name", "hf-hub:imageomics/bioclip-2")
    logger.info("Loading zero-shot OpenCLIP model: %s", model_name)
    model, _preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name,
        pretrained=backbone_cfg.get("pretrained", ""),
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    log_gpu_memory("zero-shot model loaded")

    prompts = build_prompts(builder.class_names)
    text_tokens = tokenizer(prompts).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    loader = builder.build_dataloader(
        args.split,
        transform=preprocess_val,
        batch_size=args.batch_size,
        num_workers=args.workers,
        weighted_sampling=False,
        seed=seed,
    )
    total_batches = len(loader)
    if args.max_batches is not None:
        total_batches = min(total_batches, args.max_batches)

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    t0 = time.perf_counter()

    pbar = tqdm(loader, total=total_batches, desc=f"Zero-shot {args.split}", unit="batch")
    for batch_idx, (images, labels) in enumerate(pbar):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast("cuda", enabled=device.type == "cuda"):
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ text_features.T

        pred1 = logits.argmax(dim=1)
        correct_top1 += (pred1 == labels).sum().item()
        if logits.size(1) >= 5:
            top5 = logits.topk(5, dim=1).indices
            correct_top5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.numel()
        pbar.set_postfix_str(f"top1={correct_top1 / max(total, 1):.4f}")

    elapsed = time.perf_counter() - t0
    metrics = {
        "split": args.split,
        "classes": builder.num_classes,
        "samples": total,
        "top1_accuracy": correct_top1 / max(total, 1),
        "top5_accuracy": correct_top5 / max(total, 1),
        "images_per_sec": total / elapsed if elapsed > 0 else 0.0,
        "elapsed_seconds": elapsed,
        "max_batches": args.max_batches,
        "timestamp": datetime.now().isoformat(),
    }

    experiment_dir = create_experiment_dir(
        base_dir=PROJECT_ROOT / config.get("experiment", {}).get("base_dir", "experiments"),
        base_name=args.experiment_name,
    )
    metrics_dir = experiment_dir / "metrics"
    config_dir = experiment_dir / "config"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "zeroshot_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(metrics_dir / "zeroshot_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    builder.save_class_mapping(config_dir / "class_mapping.json")

    logger.info("Zero-shot complete: top1=%.4f top5=%.4f images/sec=%.1f",
                metrics["top1_accuracy"], metrics["top5_accuracy"], metrics["images_per_sec"])
    logger.info("Baseline saved to %s", experiment_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate BioCLIP 2 zero-shot baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--experiment-name", type=str, default="bioclip2_zeroshot")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    evaluate(args)


if __name__ == "__main__":
    main()
