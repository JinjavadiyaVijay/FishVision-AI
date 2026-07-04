"""
bioclip_trainer.py — LoRA fine-tuning of BioCLIP 2 for fish species classification.

Architecture:
  BioCLIP 2 ViT-L/14 (frozen) + LoRA adapters (trainable) + Linear head (trainable)

Hardware target: RTX 3050 Laptop (6 GB VRAM)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from classification.metrics.classification_metrics import (
    compute_accuracy,
    compute_topk_accuracy,
    compute_per_class_metrics,
)

logger = logging.getLogger(__name__)


class BioCLIPClassifierModel(nn.Module):
    """
    BioCLIP 2 visual encoder + LoRA adapters + classification head.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID (e.g. "imageomics/bioclip-2").
    num_classes : int
        Number of output species.
    lora_rank : int
        LoRA rank (default 16).
    lora_alpha : int
        LoRA alpha (default 32).
    lora_dropout : float
        Dropout in LoRA layers.
    """

    def __init__(
        self,
        model_name: str = "imageomics/bioclip-2",
        num_classes: int = 159,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        # ── Load BioCLIP 2 visual encoder ──────────────────────────────
        logger.info("Loading BioCLIP 2 from: %s", model_name)
        from transformers import CLIPModel
        clip_model = CLIPModel.from_pretrained(model_name)
        self.vision_model = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection

        # Freeze all parameters
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.visual_projection.parameters():
            param.requires_grad = False

        # ── Apply LoRA ─────────────────────────────────────────────────
        logger.info("Applying LoRA: rank=%d, alpha=%d, dropout=%.2f",
                     lora_rank, lora_alpha, lora_dropout)
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            bias="none",
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)

        # Get embedding dimension
        embed_dim = self.visual_projection.out_features

        # ── Classification head ────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes),
        )

        # Log parameter counts
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("Total params:     %s", f"{total:,}")
        logger.info("Trainable params: %s (%.2f%%)",
                     f"{trainable:,}", 100 * trainable / total)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        pixel_values : (B, 3, 224, 224) tensor

        Returns
        -------
        logits : (B, num_classes) tensor
        """
        vision_out = self.vision_model(pixel_values=pixel_values)
        pooled = vision_out.pooler_output
        projected = self.visual_projection(pooled)
        logits = self.classifier(projected)
        return logits


class BioCLIPTrainer:
    """
    Training loop for BioCLIP fish species classifier.

    Parameters
    ----------
    model : BioCLIPClassifierModel
    train_loader, val_loader : DataLoader
    class_names : list[str]
    class_weights : torch.Tensor
    output_dir : Path
    config : dict
    """

    def __init__(
        self,
        model: BioCLIPClassifierModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_names: list[str],
        class_weights: Optional[torch.Tensor] = None,
        output_dir: Path = Path("models/bioclip/v1"),
        config: Optional[dict] = None,
    ) -> None:
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Loss ───────────────────────────────────────────────────────
        label_smoothing = self.config.get("label_smoothing", 0.1)
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

        # ── Optimizer ──────────────────────────────────────────────────
        lr = self.config.get("learning_rate", 2e-4)
        wd = self.config.get("weight_decay", 0.01)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)

        # ── Scheduler ──────────────────────────────────────────────────
        epochs = self.config.get("epochs", 30)
        grad_accum = self.config.get("gradient_accumulation_steps", 4)
        total_steps = len(train_loader) * epochs // grad_accum
        warmup_steps = self.config.get("warmup_steps", 200)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=max(total_steps, 1), eta_min=lr * 0.01,
        )

        # ── Mixed precision ───────────────────────────────────────────
        self.scaler = GradScaler(enabled=self.config.get("mixed_precision", True))
        self.grad_accum = grad_accum

        # ── TensorBoard ────────────────────────────────────────────────
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))

        # ── Best tracking ──────────────────────────────────────────────
        self.best_val_acc = 0.0
        self.patience = self.config.get("early_stopping_patience", 5)
        self.patience_counter = 0

    # ------------------------------------------------------------------
    def train(self, epochs: Optional[int] = None) -> dict:
        """Run the full training loop. Returns final metrics dict."""
        epochs = epochs or self.config.get("epochs", 30)

        logger.info("Starting training: %d epochs, device=%s", epochs, self.device)
        logger.info("Batch size: %d, Grad accum: %d, Effective batch: %d",
                     self.train_loader.batch_size,
                     self.grad_accum,
                     self.train_loader.batch_size * self.grad_accum)

        use_amp = self.config.get("mixed_precision", True)
        history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_top5": []}

        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()

            # ── Train ──────────────────────────────────────────────────
            train_loss = self._train_epoch(epoch, use_amp)
            history["train_loss"].append(train_loss)

            # ── Validate ───────────────────────────────────────────────
            val_metrics = self._validate(use_amp)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_top5"].append(val_metrics["top5_accuracy"])

            elapsed = time.perf_counter() - t0
            lr_now = self.optimizer.param_groups[0]["lr"]

            # ── Log ────────────────────────────────────────────────────
            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("val/loss", val_metrics["loss"], epoch)
            self.writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
            self.writer.add_scalar("val/top5_accuracy", val_metrics["top5_accuracy"], epoch)
            self.writer.add_scalar("train/lr", lr_now, epoch)

            print(f"  Epoch {epoch:>3}/{epochs}  │  "
                  f"train_loss={train_loss:.4f}  │  "
                  f"val_acc={val_metrics['accuracy']:.3f}  "
                  f"top5={val_metrics['top5_accuracy']:.3f}  │  "
                  f"lr={lr_now:.2e}  │  {elapsed:.0f}s")

            # ── Checkpoint ─────────────────────────────────────────────
            is_best = val_metrics["accuracy"] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics["accuracy"]
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"           ★ New best: {self.best_val_acc:.4f}")
            else:
                self.patience_counter += 1

            # Always save last
            self._save_checkpoint(epoch, val_metrics, is_best=False)

            # ── Early stopping ─────────────────────────────────────────
            if self.patience_counter >= self.patience:
                print(f"\n  ⏹ Early stopping at epoch {epoch} "
                      f"(no improvement for {self.patience} epochs)")
                break

        self.writer.close()

        # Save training history
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        return {
            "best_val_accuracy": self.best_val_acc,
            "epochs_trained": epoch,
            "history": history,
        }

    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int, use_amp: bool) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss = loss / self.grad_accum

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.grad_accum == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

            total_loss += loss.item() * self.grad_accum
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self, use_amp: bool) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            all_preds.append(logits.cpu())
            all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        return {
            "loss":           total_loss / max(len(self.val_loader), 1),
            "accuracy":       compute_accuracy(all_preds, all_labels),
            "top5_accuracy":  compute_topk_accuracy(all_preds, all_labels, k=5),
        }

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool) -> None:
        """Save LoRA weights + classifier head + metadata."""
        tag = "best" if is_best else "last"
        ckpt_dir = self.output_dir / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapter
        self.model.vision_model.save_pretrained(str(ckpt_dir / "lora_adapter"))

        # Save classifier head
        torch.save(
            self.model.classifier.state_dict(),
            ckpt_dir / "classifier_head.pt",
        )

        # Save metadata
        meta = {
            "epoch":        epoch,
            "val_accuracy":  metrics["accuracy"],
            "val_top5":     metrics["top5_accuracy"],
            "val_loss":     metrics["loss"],
            "num_classes":   self.model.num_classes,
            "class_names":   self.class_names,
            "model_name":    self.model.model_name,
            "config":        self.config,
        }
        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        if is_best:
            logger.info("Saved best checkpoint → %s", ckpt_dir)
