"""
train_bioclip.py — Phase 4 entry point: fine-tune BioCLIP 2 with LoRA.

Run from the project root:
    python scripts/train_bioclip.py
    python scripts/train_bioclip.py --epochs 10 --batch-size 4  # override

Prerequisites:
    pip install transformers peft torch torchvision tensorboard
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

# ── Path bootstrap ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from classification.augmentation.fish_augmentation import (
    get_train_transform,
    get_val_transform,
)
from classification.dataloader.bioclip_dataset import build_dataloaders
from classification.trainer.bioclip_trainer import (
    BioCLIPClassifierModel,
    BioCLIPTrainer,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
OUTPUT_DIR    = PROJECT_ROOT / "models" / "bioclip" / "v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Train BioCLIP 2 with LoRA")
    parser.add_argument("--model-name", default="imageomics/bioclip-2",
                        help="HuggingFace model ID")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    t0 = time.perf_counter()

    print("\n" + "=" * 60)
    print("  FishVision-AI — Phase 4: BioCLIP 2 Training")
    print("=" * 60)

    # ── Preflight ──────────────────────────────────────────────────────
    if not PROCESSED_DIR.exists():
        logger.error("Dataset not found at %s. Run Phase 3 first.", PROCESSED_DIR)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  GPU          : {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("  GPU          : NONE (CPU mode — will be very slow)")

    print(f"  Model        : {args.model_name}")
    print(f"  LoRA         : rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"  Batch size   : {args.batch_size} × {args.grad_accum} = "
          f"{args.batch_size * args.grad_accum} effective")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs       : {args.epochs} (patience={args.patience})")
    print(f"  Mixed prec.  : {'disabled' if args.no_amp else 'fp16'}")
    print(f"  Dataset      : {PROCESSED_DIR}")
    print(f"  Output       : {OUTPUT_DIR}")
    print("=" * 60 + "\n")

    # ── Build DataLoaders ──────────────────────────────────────────────
    print("  Loading dataset …")
    train_tf = get_train_transform(args.image_size)
    val_tf   = get_val_transform(args.image_size)

    data = build_dataloaders(
        processed_dir=PROCESSED_DIR,
        train_transform=train_tf,
        val_transform=val_tf,
        batch_size=args.batch_size,
        num_workers=args.workers,
        use_weighted_sampler=True,
    )

    print(f"  Classes      : {data['num_classes']}")
    print(f"  Train batches: {len(data['train_loader'])}")
    print(f"  Val batches  : {len(data['val_loader'])}")
    print()

    # ── Build Model ────────────────────────────────────────────────────
    print("  Building model …")
    model = BioCLIPClassifierModel(
        model_name=args.model_name,
        num_classes=data["num_classes"],
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
    )

    # ── Training Config ────────────────────────────────────────────────
    config = {
        "learning_rate":               args.lr,
        "weight_decay":                0.01,
        "epochs":                      args.epochs,
        "gradient_accumulation_steps": args.grad_accum,
        "label_smoothing":             args.label_smoothing,
        "mixed_precision":             not args.no_amp,
        "early_stopping_patience":     args.patience,
        "warmup_steps":                200,
        "lora_rank":                   args.lora_rank,
        "lora_alpha":                  args.lora_alpha,
    }

    # ── Train ──────────────────────────────────────────────────────────
    print("\n  Starting training …\n")
    print("─" * 80)

    trainer = BioCLIPTrainer(
        model=model,
        train_loader=data["train_loader"],
        val_loader=data["val_loader"],
        class_names=data["class_names"],
        class_weights=data["class_weights"],
        output_dir=OUTPUT_DIR,
        config=config,
    )

    results = trainer.train(epochs=args.epochs)

    print("─" * 80)
    elapsed = time.perf_counter() - t0
    print(f"\n  Training complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Best val accuracy: {results['best_val_accuracy']:.4f}")
    print(f"  Epochs trained:    {results['epochs_trained']}")
    print(f"  Model saved to:    {OUTPUT_DIR}")
    print(f"  TensorBoard logs:  {OUTPUT_DIR / 'tensorboard'}")
    print(f"\n  View training curves:")
    print(f"    tensorboard --logdir {OUTPUT_DIR / 'tensorboard'}")
    print()


if __name__ == "__main__":
    main()
