"""
fish_augmentation.py — Biologically valid augmentation transforms.

Carefully designed to preserve species-diagnostic features (color patterns,
fin shape, body proportions) while introducing realistic underwater variation.
"""

from __future__ import annotations

from torchvision import transforms

# BioCLIP / CLIP normalization constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def get_train_transform(image_size: int = 224) -> transforms.Compose:
    """Training augmentation pipeline — biologically valid transforms only."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),      # Slightly larger
        transforms.RandomCrop(image_size),                # Random crop
        transforms.RandomHorizontalFlip(p=0.5),           # Fish swim both ways
        transforms.RandomRotation(degrees=15),             # Natural tilt
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.15, hue=0.05,                    # Conservative color
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.0))
        ], p=0.3),                                        # Underwater turbidity
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.2, scale=(0.02, 0.10), ratio=(0.3, 3.3), # Small occlusions
        ),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def get_val_transform(image_size: int = 224) -> transforms.Compose:
    """Validation/test transform — deterministic, no augmentation."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
