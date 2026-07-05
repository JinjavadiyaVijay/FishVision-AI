"""
fish_dataset.py — Model-agnostic dataset for species classification.

Loads images from the ImageFolder-style directory structure produced
by Phase 2 data splitting. Class mapping is built from the intersection
of species present in ALL splits, ensuring train/val/test consistency.

Usage:
    from classification.dataloader.fish_dataset import FishDatasetBuilder

    builder = FishDatasetBuilder(
        processed_dir="datasets/processed",
        exclude_species=["Lutjanus_erythropterus", "Sphyraena_qenie"],
    )
    train_ds = builder.build_split("train", transform=train_tf)
    val_ds   = builder.build_split("val", transform=val_tf)
    class_names = builder.class_names      # sorted list
    class_weights = builder.class_weights  # inverse-frequency weights
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

logger = logging.getLogger(__name__)

# Supported image extensions
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


class FishSpeciesDataset(Dataset):
    """
    PyTorch Dataset for fish species classification.

    Parameters
    ----------
    image_paths : list[Path]
        Absolute paths to image files.
    labels : list[int]
        Integer class labels corresponding to each image.
    class_names : list[str]
        Ordered list of class names (index = label).
    transform : callable, optional
        Image transform (preprocessing + augmentation).
    """

    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        class_names: list[str],
        transform: Optional[Callable] = None,
    ) -> None:
        assert len(image_paths) == len(labels), (
            f"Mismatch: {len(image_paths)} images vs {len(labels)} labels"
        )
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


class FishDatasetBuilder:
    """
    Builds consistent train/val/test datasets with a shared class mapping.

    The class mapping is derived from the intersection of species present
    in ALL splits, ensuring that every class has at least one sample in
    every split. Species can also be manually excluded.

    Parameters
    ----------
    processed_dir : str or Path
        Root of the processed dataset (contains train/, val/, test/).
    exclude_species : list[str], optional
        Species names to exclude from all splits.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        exclude_species: Optional[list[str]] = None,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.exclude_species = set(exclude_species or [])

        # Discover species in each split
        self._split_dirs = {}
        self._split_species: dict[str, set[str]] = {}
        for split_name in ("train", "val", "test"):
            split_dir = self.processed_dir / split_name
            if split_dir.exists():
                self._split_dirs[split_name] = split_dir
                species = {
                    d.name for d in split_dir.iterdir()
                    if d.is_dir() and d.name not in self.exclude_species
                }
                self._split_species[split_name] = species

        if not self._split_dirs:
            raise FileNotFoundError(
                f"No split directories found in {self.processed_dir}"
            )

        # Class mapping = intersection of all available splits
        common_species = set.intersection(*self._split_species.values())
        self.class_names = sorted(common_species)
        self._name_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Log summary
        for split_name, species in self._split_species.items():
            excluded = species - common_species
            if excluded:
                logger.warning(
                    "Split '%s': excluding %d species not in all splits: %s",
                    split_name, len(excluded), sorted(excluded),
                )

        logger.info(
            "Dataset: %d species common across %d splits (excluded %d manually)",
            len(self.class_names), len(self._split_dirs), len(self.exclude_species),
        )

        # Pre-scan all splits to cache image paths and compute class weights
        self._split_data: dict[str, tuple[list[Path], list[int]]] = {}
        for split_name in self._split_dirs:
            paths, labels = self._scan_split(split_name)
            self._split_data[split_name] = (paths, labels)

        # Compute class weights from training set
        if "train" in self._split_data:
            self._class_weights = self._compute_class_weights(
                self._split_data["train"][1]
            )
        else:
            self._class_weights = torch.ones(len(self.class_names))

    def _scan_split(self, split_name: str) -> tuple[list[Path], list[int]]:
        """Scan a split directory and return (image_paths, labels)."""
        split_dir = self._split_dirs[split_name]
        paths: list[Path] = []
        labels: list[int] = []

        for species_name in self.class_names:
            species_dir = split_dir / species_name
            if not species_dir.exists():
                logger.warning(
                    "Species '%s' missing from %s split", species_name, split_name
                )
                continue

            images = sorted([
                f for f in species_dir.iterdir()
                if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
            ])

            for img_path in images:
                paths.append(img_path)
                labels.append(self._name_to_idx[species_name])

        logger.info(
            "Split '%s': %d images across %d species",
            split_name, len(paths), len(set(labels)),
        )
        return paths, labels

    def _compute_class_weights(self, labels: list[int]) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for loss balancing.

        weight_c = N_total / (N_classes * N_c)

        This gives higher weight to under-represented species.
        """
        counts = Counter(labels)
        n_total = len(labels)
        n_classes = len(self.class_names)

        weights = torch.zeros(n_classes)
        for cls_idx in range(n_classes):
            n_c = counts.get(cls_idx, 1)  # Avoid division by zero
            weights[cls_idx] = n_total / (n_classes * n_c)

        # Normalize so mean weight = 1.0
        weights = weights / weights.mean()

        logger.info(
            "Class weights: min=%.3f, max=%.3f, mean=%.3f",
            weights.min().item(), weights.max().item(), weights.mean().item(),
        )
        return weights

    @property
    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights tensor (for loss function)."""
        return self._class_weights

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def build_split(
        self,
        split_name: str,
        transform: Optional[Callable] = None,
    ) -> FishSpeciesDataset:
        """
        Build a FishSpeciesDataset for the given split.

        Parameters
        ----------
        split_name : str
            One of "train", "val", "test".
        transform : callable, optional
            Image transform pipeline.

        Returns
        -------
        FishSpeciesDataset
        """
        if split_name not in self._split_data:
            raise ValueError(
                f"Split '{split_name}' not found. "
                f"Available: {list(self._split_data.keys())}"
            )

        paths, labels = self._split_data[split_name]
        return FishSpeciesDataset(
            image_paths=paths,
            labels=labels,
            class_names=self.class_names,
            transform=transform,
        )

    def build_weighted_sampler(self, split_name: str = "train") -> WeightedRandomSampler:
        """
        Build a WeightedRandomSampler for the given split.

        Each sample's weight is the inverse of its class frequency,
        so rare species are sampled more frequently.
        """
        _, labels = self._split_data[split_name]
        counts = Counter(labels)
        sample_weights = [1.0 / counts[label] for label in labels]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(labels),
            replacement=True,
        )

    def build_dataloader(
        self,
        split_name: str,
        transform: Optional[Callable] = None,
        batch_size: int = 8,
        num_workers: int = 2,
        weighted_sampling: bool = False,
    ) -> DataLoader:
        """
        Build a DataLoader with optional weighted sampling.

        Parameters
        ----------
        split_name : str
            One of "train", "val", "test".
        transform : callable, optional
            Image transform pipeline.
        batch_size : int
            Batch size.
        num_workers : int
            Number of data loading workers.
        weighted_sampling : bool
            If True, use WeightedRandomSampler (for training only).
        """
        dataset = self.build_split(split_name, transform=transform)

        sampler = None
        shuffle = (split_name == "train")

        if weighted_sampling and split_name == "train":
            sampler = self.build_weighted_sampler(split_name)
            shuffle = False  # Sampler handles ordering

        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "sampler": sampler,
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
            "drop_last": (split_name == "train"),
        }

        # Workers-dependent optimizations
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2

        return DataLoader(dataset, **loader_kwargs)

    def get_split_summary(self) -> dict[str, Any]:
        """Return a summary dict of all splits for logging."""
        summary = {
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "excluded_species": sorted(self.exclude_species),
        }
        for split_name, (paths, labels) in self._split_data.items():
            counts = Counter(labels)
            summary[split_name] = {
                "total_images": len(paths),
                "classes_with_images": len(counts),
                "min_per_class": min(counts.values()) if counts else 0,
                "max_per_class": max(counts.values()) if counts else 0,
                "median_per_class": int(np.median(list(counts.values()))) if counts else 0,
            }
        return summary

    def save_class_mapping(self, path: str | Path) -> None:
        """Save the class name -> index mapping as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._name_to_idx, f, indent=2)
        logger.info("Class mapping saved to %s", path)
