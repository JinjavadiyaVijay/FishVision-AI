"""
training_engine.py — Production training loop for species classification.

Model-agnostic: works with any BackboneWrapper + ClassifierHead combination.
Supports LoRA, mixed precision, gradient accumulation, checkpointing, early
stopping, and resume training.

Hardware target: RTX 3050 6GB VRAM.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from classification.models.classifier import SpeciesClassifier
from classification.utilities.device import log_gpu_memory, get_system_info

logger = logging.getLogger(__name__)


class TrainingEngine:
    """
    Production training loop for species classification.

    Parameters
    ----------
    model : SpeciesClassifier
        The backbone + classifier head model.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    config : dict
        Full merged configuration dict.
    experiment_dir : Path
        Directory for this experiment run (checkpoints, logs, metrics).
    class_names : list[str]
        Ordered species names for class index mapping.
    device : torch.device
        Target device.
    class_weights : torch.Tensor, optional
        Per-class loss weights for imbalance handling.
    """

    def __init__(
        self,
        model: SpeciesClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        experiment_dir: Path,
        class_names: list[str],
        device: torch.device,
        class_weights: Optional[torch.Tensor] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.experiment_dir = Path(experiment_dir)
        self.class_names = class_names
        self.device = device

        # Sub-directories
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.logs_dir = self.experiment_dir / "logs"
        self.metrics_dir = self.experiment_dir / "metrics"
        self.plots_dir = self.experiment_dir / "plots"
        self.config_dir = self.experiment_dir / "config"

        for d in (self.checkpoint_dir, self.logs_dir, self.metrics_dir,
                  self.plots_dir, self.config_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Training config
        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 30)
        self.grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 4)

        # Max steps (for smoke test / debug mode)
        self.max_steps = max_steps

        # Batch timing
        self.batch_times: list[float] = []
        self.data_times: list[float] = []
        self.transfer_times: list[float] = []
        self.forward_times: list[float] = []
        self.backward_times: list[float] = []
        self.optimizer_times: list[float] = []
        self.grad_norms: list[float] = []

        # Precision
        prec_cfg = config.get("precision", {})
        self.use_amp = prec_cfg.get("mixed_precision", True)
        self.grad_clip_norm = prec_cfg.get("gradient_clip_max_norm", 1.0)

        # Loss function
        loss_cfg = config.get("loss", {})
        label_smoothing = loss_cfg.get("label_smoothing", 0.1)
        if class_weights is not None and loss_cfg.get("use_class_weights", True):
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(device),
                label_smoothing=label_smoothing,
            )
            logger.info("Loss: CrossEntropy with label_smoothing=%.2f + class weights",
                         label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            logger.info("Loss: CrossEntropy with label_smoothing=%.2f", label_smoothing)

        # Optimizer
        opt_cfg = config.get("optimizer", {})
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=opt_cfg.get("learning_rate", 2e-4),
            weight_decay=opt_cfg.get("weight_decay", 0.01),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            eps=opt_cfg.get("eps", 1e-8),
        )
        logger.info(
            "Optimizer: AdamW, lr=%.1e, weight_decay=%.4f, %d param groups",
            opt_cfg.get("learning_rate", 2e-4),
            opt_cfg.get("weight_decay", 0.01),
            len(self.optimizer.param_groups),
        )

        # Scheduler
        sched_cfg = config.get("scheduler", {})
        batches_per_epoch = len(self.train_loader)
        if self.max_steps is not None:
            batches_per_epoch = min(batches_per_epoch, self.max_steps)
        optimizer_steps_per_epoch = max(math.ceil(batches_per_epoch / self.grad_accum_steps), 1)
        total_steps = max(self.epochs * optimizer_steps_per_epoch, 1)
        configured_warmup = sched_cfg.get("warmup_steps", 200)
        warmup_steps = min(configured_warmup, max(total_steps // 3, 1))
        min_lr_ratio = sched_cfg.get("min_lr_ratio", 0.01)

        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps - warmup_steps, 1),
            eta_min=opt_cfg.get("learning_rate", 2e-4) * min_lr_ratio,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
        logger.info(
            "Scheduler: CosineWithWarmup, warmup=%d/%d configured steps, total=%d optimizer steps",
            warmup_steps, configured_warmup, total_steps,
        )

        # AMP scaler
        self.scaler = GradScaler(enabled=self.use_amp)
        logger.info(
            "AMP: enabled=%s, scaler_enabled=%s, autocast_device=%s",
            self.use_amp, self.scaler.is_enabled(), "cuda" if torch.cuda.is_available() else "cpu",
        )
        logger.info("Gradient clipping: max_norm=%.3f", self.grad_clip_norm)

        # Early stopping
        es_cfg = config.get("early_stopping", {})
        self.early_stopping_enabled = es_cfg.get("enabled", True)
        self.patience = es_cfg.get("patience", 7)
        self.min_delta = es_cfg.get("min_delta", 0.001)
        self.monitor_metric = es_cfg.get("monitor", "val_accuracy")

        # State tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_metrics: dict[str, float] = {
            "val_accuracy": -1.0,
            "val_f1_macro": -1.0,
        }
        self.early_stopping_best = -1.0
        self.patience_counter = 0
        self.history: list[dict[str, Any]] = []

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.logs_dir))

        # CSV logger
        self.csv_path = self.metrics_dir / "training_history.csv"
        self.csv_initialized = False

        # Save config snapshot
        self._save_config_snapshot()

    def _save_config_snapshot(self) -> None:
        """Save all config files to the experiment directory."""
        config_snapshot = {
            **self.config,
            "system_info": get_system_info(),
            "experiment_dir": str(self.experiment_dir),
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.config_dir / "full_config.yaml", "w") as f:
            yaml.dump(config_snapshot, f, default_flow_style=False, sort_keys=False)
        logger.info("Config snapshot saved to %s", self.config_dir)

    def train(self) -> dict[str, Any]:
        """
        Run the full training loop.

        Returns
        -------
        dict with final metrics and best checkpoint paths.
        """
        logger.info("=" * 60)
        logger.info("Starting training: %d epochs, %d train batches/epoch",
                     self.epochs, len(self.train_loader))
        logger.info("Effective batch size: %d (batch=%d x accum=%d)",
                     self.train_loader.batch_size * self.grad_accum_steps,
                     self.train_loader.batch_size, self.grad_accum_steps)
        logger.info("=" * 60)

        log_gpu_memory("training start")
        total_t0 = time.perf_counter()

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch

            # ── Train one epoch ───────────────────────────────────
            train_metrics = self._train_epoch()

            # ── Validate ──────────────────────────────────────────
            val_metrics = self._validate()

            # ── Combine metrics ───────────────────────────────────
            epoch_metrics = {
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            self.history.append(epoch_metrics)

            # ── Log to TensorBoard ────────────────────────────────
            for key, value in epoch_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"epoch/{key}", value, epoch)

            # ── Log to CSV ────────────────────────────────────────
            self._log_csv(epoch_metrics)

            # ── Log to console ────────────────────────────────────
            logger.info(
                "Epoch %d/%d  |  train_loss=%.4f  val_loss=%.4f  "
                "val_acc=%.4f  val_top5=%.4f  val_f1=%.4f  lr=%.2e",
                epoch + 1, self.epochs,
                epoch_metrics["train_loss"],
                epoch_metrics["val_loss"],
                epoch_metrics["val_accuracy"],
                epoch_metrics.get("val_top5_accuracy", 0),
                epoch_metrics.get("val_f1_macro", 0),
                epoch_metrics["lr"],
            )

            # ── Checkpointing ─────────────────────────────────────
            self._checkpoint(epoch_metrics)

            # ── Early stopping ────────────────────────────────────
            if self._check_early_stopping(epoch_metrics):
                logger.info(
                    "Early stopping triggered at epoch %d (patience=%d)",
                    epoch + 1, self.patience,
                )
                break

            log_gpu_memory(f"epoch {epoch + 1}")

        total_time = time.perf_counter() - total_t0
        logger.info("Training complete: %.1f minutes", total_time / 60)

        # Throughput report (Task 4 — profiling)
        throughput = self.log_throughput_report()

        self.writer.close()

        return {
            "best_metrics": self.best_metrics,
            "total_epochs": self.current_epoch + 1,
            "total_time_minutes": round(total_time / 60, 1),
            "throughput": throughput,
            "history": self.history,
        }

    def _train_epoch(self) -> dict[str, float]:
        """Train for one epoch with tqdm progress. Returns metrics dict."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        running_loss = 0.0  # For smoothed display

        self.optimizer.zero_grad()

        # Determine total batches (may be limited by max_steps)
        total_batches = len(self.train_loader)
        if self.max_steps is not None:
            total_batches = min(total_batches, self.max_steps)

        pbar = tqdm(
            enumerate(self.train_loader),
            total=total_batches,
            desc=f"Epoch {self.current_epoch + 1}/{self.epochs}",
            unit="batch",
            leave=True,
            bar_format="{l_bar}{bar:20}{r_bar}",
        )

        data_start = time.perf_counter()
        processed_batches = 0

        for batch_idx, (images, labels) in pbar:
            processed_batches = batch_idx + 1
            cuda_timing = torch.cuda.is_available()
            # Data loading time
            data_time = time.perf_counter() - data_start
            self.data_times.append(data_time)

            batch_start = time.perf_counter()

            if cuda_timing:
                torch.cuda.synchronize()
            transfer_start = time.perf_counter()
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if cuda_timing:
                torch.cuda.synchronize()
            self.transfer_times.append(time.perf_counter() - transfer_start)

            # Forward pass with mixed precision
            forward_start = time.perf_counter()
            autocast_device = "cuda" if self.device.type == "cuda" else "cpu"
            with autocast(autocast_device, enabled=self.use_amp and self.device.type == "cuda"):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss = loss / self.grad_accum_steps
            if cuda_timing:
                torch.cuda.synchronize()
            self.forward_times.append(time.perf_counter() - forward_start)

            # Backward pass
            backward_start = time.perf_counter()
            self.scaler.scale(loss).backward()
            if cuda_timing:
                torch.cuda.synchronize()
            self.backward_times.append(time.perf_counter() - backward_start)

            # Accumulation step
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                optimizer_start = time.perf_counter()
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.grad_clip_norm,
                    )
                    self.grad_norms.append(float(grad_norm.detach().cpu()))

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                if cuda_timing:
                    torch.cuda.synchronize()
                self.optimizer_times.append(time.perf_counter() - optimizer_start)
                self.global_step += 1

                # TensorBoard: per-step loss
                tb_cfg = self.config.get("tensorboard", {})
                log_every = tb_cfg.get("log_train_loss_every_n_steps", 10)
                if self.global_step % log_every == 0:
                    self.writer.add_scalar(
                        "step/train_loss",
                        loss.item() * self.grad_accum_steps,
                        self.global_step,
                    )
                    self.writer.add_scalar(
                        "step/lr",
                        self.optimizer.param_groups[0]["lr"],
                        self.global_step,
                    )

            # Metrics
            batch_loss = loss.item() * self.grad_accum_steps
            total_loss += batch_loss * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Batch timing
            batch_time = time.perf_counter() - batch_start
            self.batch_times.append(batch_time)

            # Smoothed running loss for display
            running_loss = 0.9 * running_loss + 0.1 * batch_loss if running_loss > 0 else batch_loss

            # Update progress bar
            gpu_mem = ""
            if torch.cuda.is_available():
                gpu_gb = torch.cuda.memory_allocated(0) / 1e9
                gpu_mem = f"GPU:{gpu_gb:.1f}GB"

            lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix_str(
                f"loss={running_loss:.3f} lr={lr:.1e} {gpu_mem}",
                refresh=False,
            )

            # Max steps early exit (for smoke test / debug mode)
            if self.max_steps is not None and (batch_idx + 1) >= self.max_steps:
                break

            data_start = time.perf_counter()

        # Apply any remaining gradients when the epoch/step limit is not
        # divisible by gradient accumulation. Without this, smoke/debug runs
        # silently drop their final partial accumulation.
        if processed_batches > 0 and processed_batches % self.grad_accum_steps != 0:
            optimizer_start = time.perf_counter()
            if self.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.grad_clip_norm,
                )
                self.grad_norms.append(float(grad_norm.detach().cpu()))

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.optimizer_times.append(time.perf_counter() - optimizer_start)
            self.global_step += 1

        pbar.close()

        return {
            "loss": total_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
        }

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        """Validate on the validation set with tqdm progress. Returns metrics dict."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_top5_correct = 0
        total = 0

        # Limit validation batches proportionally in smoke/debug mode
        val_batches = len(self.val_loader)
        if self.max_steps is not None:
            val_batches = min(val_batches, max(self.max_steps // 2, 10))

        pbar = tqdm(
            enumerate(self.val_loader),
            total=val_batches,
            desc="  Validating",
            unit="batch",
            leave=False,
            bar_format="{l_bar}{bar:20}{r_bar}",
        )

        for batch_idx, (images, labels) in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            autocast_device = "cuda" if self.device.type == "cuda" else "cpu"
            with autocast(autocast_device, enabled=self.use_amp and self.device.type == "cuda"):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Top-5 accuracy
            if logits.size(1) >= 5:
                _, top5_indices = logits.topk(5, dim=1)
                top5_correct = (top5_indices == labels.unsqueeze(1)).any(dim=1)
                all_top5_correct += top5_correct.sum().item()
            total += labels.size(0)

            if self.max_steps is not None and (batch_idx + 1) >= val_batches:
                break

        pbar.close()

        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = (all_preds == all_labels).mean()
        top5_accuracy = all_top5_correct / max(total, 1)

        # Per-class F1 (macro)
        from sklearn.metrics import f1_score, precision_score, recall_score
        f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)

        return {
            "loss": total_loss / max(total, 1),
            "accuracy": float(accuracy),
            "top5_accuracy": float(top5_accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
        }

    def log_throughput_report(self) -> dict[str, float]:
        """Log profiling data: batch times, data loading times, throughput.
        
        Call this after at least one epoch to get meaningful data.
        """
        report = {}
        batch_size = self.train_loader.batch_size or 8

        if self.batch_times:
            avg_batch = sum(self.batch_times) / len(self.batch_times)
            avg_data = sum(self.data_times) / len(self.data_times) if self.data_times else 0
            avg_transfer = sum(self.transfer_times) / len(self.transfer_times) if self.transfer_times else 0
            avg_forward = sum(self.forward_times) / len(self.forward_times) if self.forward_times else 0
            avg_backward = sum(self.backward_times) / len(self.backward_times) if self.backward_times else 0
            avg_optimizer = sum(self.optimizer_times) / len(self.optimizer_times) if self.optimizer_times else 0
            avg_compute = avg_forward + avg_backward + avg_optimizer
            imgs_per_sec = batch_size / (avg_batch + avg_data) if (avg_batch + avg_data) > 0 else 0
            data_pct = 100 * avg_data / (avg_batch + avg_data) if (avg_batch + avg_data) > 0 else 0

            report = {
                "avg_batch_time_ms": round(avg_batch * 1000, 1),
                "avg_data_time_ms": round(avg_data * 1000, 1),
                "avg_transfer_time_ms": round(avg_transfer * 1000, 1),
                "avg_forward_time_ms": round(avg_forward * 1000, 1),
                "avg_backward_time_ms": round(avg_backward * 1000, 1),
                "avg_optimizer_time_ms": round(avg_optimizer * 1000, 1),
                "avg_compute_time_ms": round(avg_compute * 1000, 1),
                "images_per_sec": round(imgs_per_sec, 1),
                "data_loading_pct": round(data_pct, 1),
                "total_batches_profiled": len(self.batch_times),
            }
            if self.grad_norms:
                report["avg_grad_norm"] = round(sum(self.grad_norms) / len(self.grad_norms), 4)
                report["max_grad_norm"] = round(max(self.grad_norms), 4)

            logger.info("=" * 60)
            logger.info("THROUGHPUT REPORT")
            logger.info("-" * 60)
            logger.info("  Avg batch total:         %6.1f ms", report["avg_batch_time_ms"])
            logger.info("  Avg data wait:           %6.1f ms", report["avg_data_time_ms"])
            logger.info("  Avg H2D transfer:        %6.1f ms", report["avg_transfer_time_ms"])
            logger.info("  Avg forward:             %6.1f ms", report["avg_forward_time_ms"])
            logger.info("  Avg backward:            %6.1f ms", report["avg_backward_time_ms"])
            logger.info("  Avg optimizer/scheduler: %6.1f ms", report["avg_optimizer_time_ms"])
            logger.info("  Images/sec:              %6.1f", report["images_per_sec"])
            logger.info("  Data loading overhead:   %5.1f%%", report["data_loading_pct"])
            logger.info("  Batches profiled:        %6d", report["total_batches_profiled"])
            if "avg_grad_norm" in report:
                logger.info("  Grad norm avg/max:       %.4f / %.4f",
                            report["avg_grad_norm"], report["max_grad_norm"])

            # Bottleneck analysis
            compute_pct = 100 * avg_compute / (avg_batch + avg_data) if (avg_batch + avg_data) > 0 else 0
            if data_pct > 50:
                logger.warning("  BOTTLENECK: Data loading (%.0f%%). "
                             "Increase num_workers or enable pin_memory.", data_pct)
            elif data_pct > 30:
                logger.info("  Data loading is significant (%.0f%%). "
                          "Consider increasing num_workers.", data_pct)
            elif compute_pct > 60:
                logger.info("  Pipeline is compute-bound. Data loading is efficient.")
            else:
                logger.info("  No single dominant bottleneck from coarse timings.")

            # GPU memory summary
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated(0) / 1e9
                reserved = torch.cuda.memory_reserved(0) / 1e9
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info("  GPU memory: %.2f GB allocated, %.2f GB reserved, %.1f GB total",
                          alloc, reserved, total_mem)

            logger.info("=" * 60)

        return report

    def _checkpoint(self, epoch_metrics: dict[str, Any]) -> None:
        """Save checkpoints based on metric improvements."""
        ckpt_cfg = self.config.get("checkpointing", {})

        # Save best accuracy
        val_acc = epoch_metrics.get("val_accuracy", 0)
        if ckpt_cfg.get("save_best_accuracy", True):
            if val_acc > self.best_metrics.get("val_accuracy", 0) + self.min_delta:
                self.best_metrics["val_accuracy"] = val_acc
                self._save_checkpoint("best_accuracy.pt", epoch_metrics)
                logger.info("  New best accuracy: %.4f", val_acc)

        # Save best F1
        val_f1 = epoch_metrics.get("val_f1_macro", 0)
        if ckpt_cfg.get("save_best_f1", True):
            if val_f1 > self.best_metrics.get("val_f1_macro", 0) + self.min_delta:
                self.best_metrics["val_f1_macro"] = val_f1
                self._save_checkpoint("best_f1.pt", epoch_metrics)
                logger.info("  New best F1 (macro): %.4f", val_f1)

        # Always save canonical latest checkpoint and training state after
        # best-metric bookkeeping so resume sees the current best values.
        if ckpt_cfg.get("save_last", True):
            self._save_checkpoint(
                "latest.pt",
                epoch_metrics,
                save_optimizer=True,
            )

    def _trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only trainable model tensors for compact checkpoints."""
        trainable_names = {
            name for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        state = self.model.state_dict()
        return {
            name: tensor.detach().cpu()
            for name, tensor in state.items()
            if name in trainable_names
        }

    def _save_checkpoint(
        self,
        name: str,
        metrics: dict[str, Any],
        save_optimizer: bool = False,
    ) -> None:
        """Save a checkpoint with descriptive name."""
        ckpt_path = self.checkpoint_dir / name
        torch.save({
            "format_version": 2,
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "epoch": metrics["epoch"],
            "global_step": self.global_step,
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            "class_names": self.class_names,
            "num_classes": len(self.class_names),
            "trainable_model_state": self._trainable_state_dict(),
        }, ckpt_path)

        # Save optimizer state (for resume)
        if save_optimizer:
            torch.save({
                "format_version": 2,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "epoch": metrics["epoch"],
                "global_step": self.global_step,
                "best_metrics": self.best_metrics,
                "early_stopping_best": self.early_stopping_best,
                "patience_counter": self.patience_counter,
                "checkpoint": name,
            }, self.checkpoint_dir / "training_state.pt")

        # Save metadata
        metadata = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            "class_names": self.class_names,
            "num_classes": len(self.class_names),
        }
        with open(self.checkpoint_dir / f"{Path(name).stem}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def resume_from(self, checkpoint_dir: str | Path) -> None:
        """
        Resume training from a checkpoint.

        Parameters
        ----------
        checkpoint_dir : Path
            Path to a checkpoint directory containing training_state.pt,
            lora_adapter/ or backbone.pt, and classifier_head.pt.
        """
        checkpoint_path = Path(checkpoint_dir)
        logger.info("Resuming from: %s", checkpoint_path)

        if checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state = checkpoint.get("trainable_model_state", checkpoint.get("model_state", {}))
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            logger.info(
                "  Loaded trainable checkpoint tensors (missing=%d, unexpected=%d)",
                len(missing), len(unexpected),
            )

            state_path = checkpoint_path.parent / "training_state.pt"
            if state_path.exists():
                train_state = torch.load(state_path, map_location=self.device, weights_only=False)
                self.optimizer.load_state_dict(train_state["optimizer"])
                self.scheduler.load_state_dict(train_state["scheduler"])
                self.scaler.load_state_dict(train_state["scaler"])
                self.current_epoch = train_state["epoch"] + 1
                self.global_step = train_state["global_step"]
                self.best_metrics = train_state["best_metrics"]
                self.early_stopping_best = train_state.get(
                    "early_stopping_best",
                    self.best_metrics.get(self.monitor_metric, -1.0),
                )
                self.patience_counter = train_state["patience_counter"]
                logger.info("  Resumed optimizer state at epoch %d, step %d",
                            self.current_epoch, self.global_step)
            return

        checkpoint_dir = checkpoint_path

        # Load classifier head
        head_path = checkpoint_dir / "classifier_head.pt"
        if head_path.exists():
            self.model.classifier.load_state_dict(torch.load(head_path, weights_only=True))
            logger.info("  Loaded classifier head")

        # Load LoRA adapter
        lora_path = checkpoint_dir / "lora_adapter"
        backbone_path = checkpoint_dir / "backbone.pt"
        if lora_path.exists():
            from peft import PeftModel
            if isinstance(self.model.backbone.visual, PeftModel):
                self.model.backbone.visual.load_adapter(str(lora_path), adapter_name="resume", is_trainable=True)
                self.model.backbone.visual.set_adapter("resume")
                logger.info("  Loaded LoRA adapter into existing PEFT model")
            else:
                self.model.backbone.visual = PeftModel.from_pretrained(
                    self.model.backbone.visual, str(lora_path), is_trainable=True
                )
                logger.info("  Loaded LoRA adapter")
        elif backbone_path.exists():
            self.model.backbone.load_state_dict(
                torch.load(backbone_path, weights_only=True)
            )
            logger.info("  Loaded backbone weights")

        # Load training state
        state_path = checkpoint_dir / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, weights_only=True)
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.scaler.load_state_dict(state["scaler"])
            self.current_epoch = state["epoch"] + 1
            self.global_step = state["global_step"]
            self.best_metrics = state["best_metrics"]
            self.early_stopping_best = state.get(
                "early_stopping_best",
                self.best_metrics.get(self.monitor_metric, -1.0),
            )
            self.patience_counter = state["patience_counter"]
            logger.info("  Resumed at epoch %d, step %d",
                         self.current_epoch, self.global_step)

    def _check_early_stopping(self, epoch_metrics: dict[str, Any]) -> bool:
        """Check early stopping condition."""
        if not self.early_stopping_enabled:
            return False

        current_value = epoch_metrics.get(self.monitor_metric, 0)
        best_value = self.early_stopping_best

        if current_value > best_value + self.min_delta:
            self.early_stopping_best = current_value
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.patience

    def _log_csv(self, metrics: dict[str, Any]) -> None:
        """Append epoch metrics to CSV file."""
        numeric_metrics = {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        }

        if not self.csv_initialized:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(numeric_metrics.keys()))
                writer.writeheader()
                writer.writerow(numeric_metrics)
            self.csv_initialized = True
        else:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(numeric_metrics.keys()))
                writer.writerow(numeric_metrics)


def create_experiment_dir(
    base_dir: str | Path = "experiments",
    base_name: str = "bioclip2_lora",
) -> Path:
    """
    Create a timestamped experiment directory.

    Returns
    -------
    Path to the created experiment directory.
    Example: experiments/bioclip2_lora_20260704_185000/
    """
    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = base_dir / f"{base_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Experiment directory: %s", experiment_dir)
    return experiment_dir
