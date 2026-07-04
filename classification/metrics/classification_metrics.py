"""
classification_metrics.py — Metric computation for species classification.

Provides per-class precision/recall/F1, top-k accuracy, and confusion matrix.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy."""
    return (preds.argmax(dim=1) == labels).float().mean().item()


def compute_topk_accuracy(preds: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """Top-k accuracy."""
    _, topk_idx = preds.topk(k, dim=1)
    correct = topk_idx.eq(labels.unsqueeze(1).expand_as(topk_idx))
    return correct.any(dim=1).float().mean().item()


def compute_per_class_metrics(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """
    Compute per-class precision, recall, F1 and macro/weighted averages.

    Parameters
    ----------
    all_preds : array of predicted class indices
    all_labels : array of ground-truth class indices
    class_names : ordered list of class names

    Returns
    -------
    dict with per_class (list of dicts), macro_f1, weighted_f1, accuracy
    """
    num_classes = len(class_names)
    per_class = []

    for i in range(num_classes):
        tp = int(((all_preds == i) & (all_labels == i)).sum())
        fp = int(((all_preds == i) & (all_labels != i)).sum())
        fn = int(((all_preds != i) & (all_labels == i)).sum())
        support = int((all_labels == i).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class.append({
            "class_name": class_names[i],
            "precision":  round(precision, 4),
            "recall":     round(recall, 4),
            "f1":         round(f1, 4),
            "support":    support,
        })

    # Macro F1 (unweighted mean)
    f1_scores = [c["f1"] for c in per_class if c["support"] > 0]
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    # Weighted F1
    total = sum(c["support"] for c in per_class)
    weighted_f1 = sum(c["f1"] * c["support"] for c in per_class) / total if total > 0 else 0.0

    accuracy = (all_preds == all_labels).mean()

    return {
        "per_class":   per_class,
        "macro_f1":    round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "accuracy":    round(float(accuracy), 4),
        "num_classes":  num_classes,
        "total_samples": int(total),
    }


def build_confusion_matrix(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Build a num_classes × num_classes confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, label in zip(all_preds, all_labels):
        cm[label, pred] += 1
    return cm
