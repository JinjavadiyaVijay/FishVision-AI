"""
bioclip_dataset.py — PyTorch Dataset for OzFish species classification.

Auto-discovers species from the directory structure:
    datasets/processed/{train,val,test}/{Species_name}/image.png

Computes balanced class weights for handling imbalanced classes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

logger = logging.getLogger(__name__)


class FishSpeciesDataset(Dataset):
    """
    ImageFolder-style dataset for fish species classification.

    Parameters
    ----------
    root_dir : Path
        Split directory (e.g. datasets/processed/train/).
    transform : transforms.Compose
        Image transform pipeline.
    """

    def __init__(
        self,
        root_dir: Path,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Discover species (sorted for deterministic class_id mapping)
        self.classes = sorted([
            d.name for d in self.root_dir.iterdir() if d.is_dir()
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        # Build sample list
        self.samples: list[tuple[Path, int]] = []
        self.class_counts: dict[int, int] = {}

        for cls_name in self.classes:
            cls_idx = self.class_to_idx[cls_name]
            cls_dir = self.root_dir / cls_name
            images = list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg"))
            self.class_counts[cls_idx] = len(images)
            for img_path in images:
                self.samples.append((img_path, cls_idx))

        logger.info(
            "%s: %d images, %d classes",
            self.root_dir.name, len(self.samples), self.num_classes,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for balanced loss."""
        counts = torch.tensor([self.class_counts[i] for i in range(self.num_classes)],
                              dtype=torch.float)
        weights = 1.0 / counts
        weights = weights / weights.sum() * self.num_classes  # normalize
        return weights

    def get_sampler(self) -> WeightedRandomSampler:
        """Weighted sampler that oversamples minority classes."""
        sample_weights = []
        total = sum(self.class_counts.values())
        class_weight = {
            cls_idx: total / count
            for cls_idx, count in self.class_counts.items()
        }
        for _, label in self.samples:
            sample_weights.append(class_weight[label])
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )


def build_dataloaders(
    processed_dir: Path,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
    batch_size: int = 8,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
) -> dict:
    """
    Build train/val/test DataLoaders from the processed directory.

    Returns dict with: train_loader, val_loader, test_loader,
                       num_classes, class_names, class_weights
    """
    train_ds = FishSpeciesDataset(processed_dir / "train", train_transform)
    val_ds   = FishSpeciesDataset(processed_dir / "val",   val_transform)
    test_ds  = FishSpeciesDataset(processed_dir / "test",  val_transform)

    # Verify class consistency
    assert train_ds.classes == val_ds.classes == test_ds.classes, (
        "Class mismatch between splits! "
        f"train={len(train_ds.classes)}, val={len(val_ds.classes)}, test={len(test_ds.classes)}"
    )

    train_sampler = train_ds.get_sampler() if use_weighted_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(not use_weighted_sampler),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,   # No grad → can use larger batch
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {
        "train_loader":  train_loader,
        "val_loader":    val_loader,
        "test_loader":   test_loader,
        "num_classes":   train_ds.num_classes,
        "class_names":   train_ds.classes,
        "class_weights": train_ds.get_class_weights(),
    }
