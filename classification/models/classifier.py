"""
classifier.py — Classification head for species prediction.

Sits on top of any BackboneWrapper and maps embeddings → class logits.
Completely backbone-agnostic: only needs the embedding dimension.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ClassifierHead(nn.Module):
    """
    Linear or MLP classification head.

    Parameters
    ----------
    embed_dim : int
        Input embedding dimensionality (from backbone).
    num_classes : int
        Number of output classes.
    head_type : str
        "linear" — LayerNorm + Dropout + Linear
        "mlp"    — LayerNorm + Linear(D, D//2) + GELU + Dropout + Linear(D//2, N)
    dropout : float
        Dropout rate before the final linear layer.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        head_type: str = "linear",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.head_type = head_type

        if head_type == "linear":
            self.head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes),
            )
        elif head_type == "mlp":
            hidden = embed_dim // 2
            self.head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )
        else:
            raise ValueError(f"Unknown head_type: '{head_type}'. Use 'linear' or 'mlp'.")

        # Initialize weights
        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "ClassifierHead(%s): embed_dim=%d → num_classes=%d (%d params)",
            head_type, embed_dim, num_classes, total_params,
        )

    def _init_weights(self) -> None:
        """Initialize linear layers with small weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Map embeddings to class logits.

        Parameters
        ----------
        embeddings : (B, embed_dim) tensor from backbone.

        Returns
        -------
        logits : (B, num_classes) tensor.
        """
        return self.head(embeddings)


class SpeciesClassifier(nn.Module):
    """
    Complete species classifier: backbone + classification head.

    This module composes a BackboneWrapper and a ClassifierHead into a
    single nn.Module for end-to-end forward passes.

    Parameters
    ----------
    backbone : nn.Module
        Vision encoder (any BackboneWrapper subclass).
    classifier : ClassifierHead
        Classification head.
    """

    def __init__(
        self,
        backbone: nn.Module,
        classifier: ClassifierHead,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        End-to-end: images → logits.

        Parameters
        ----------
        images : (B, 3, H, W) preprocessed tensor.

        Returns
        -------
        logits : (B, num_classes) tensor.
        """
        embeddings = self.backbone(images)
        logits = self.classifier(embeddings)
        return logits

    def get_param_summary(self) -> dict[str, Any]:
        """Detailed parameter breakdown."""
        backbone_total = sum(p.numel() for p in self.backbone.parameters())
        backbone_train = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        head_total = sum(p.numel() for p in self.classifier.parameters())
        head_train = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        total = backbone_total + head_total
        trainable = backbone_train + head_train

        return {
            "backbone_params": backbone_total,
            "backbone_trainable": backbone_train,
            "head_params": head_total,
            "head_trainable": head_train,
            "total_params": total,
            "total_trainable": trainable,
            "trainable_pct": round(100 * trainable / max(total, 1), 2),
        }
