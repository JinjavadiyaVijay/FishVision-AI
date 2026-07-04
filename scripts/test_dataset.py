"""Quick test of the FishDatasetBuilder."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classification.dataloader.fish_dataset import FishDatasetBuilder

builder = FishDatasetBuilder(
    "datasets/processed",
    exclude_species=["Lutjanus_erythropterus", "Sphyraena_qenie"],
)

summary = builder.get_split_summary()
print(f"\nClasses: {summary['num_classes']}")
print(f"Excluded: {summary['excluded_species']}")

for split in ("train", "val", "test"):
    s = summary[split]
    print(f"  {split}: {s['total_images']} images, "
          f"{s['classes_with_images']} classes, "
          f"min={s['min_per_class']}, max={s['max_per_class']}, "
          f"median={s['median_per_class']}")

# Verify all splits have the same class count
train_ds = builder.build_split("train")
val_ds = builder.build_split("val")
test_ds = builder.build_split("test")
assert train_ds.num_classes == val_ds.num_classes == test_ds.num_classes, "Class mismatch!"
print(f"\n  All splits have {train_ds.num_classes} classes")

# Test a batch
from classification.models.backbone import OpenCLIPBackbone
backbone = OpenCLIPBackbone()
val_tf = backbone.get_transforms(train=False)
val_ds = builder.build_split("val", transform=val_tf)

import torch
img, label = val_ds[0]
print(f"\n  Sample: shape={list(img.shape)}, label={label}, species={builder.class_names[label]}")
print(f"  Class weights shape: {builder.class_weights.shape}")
print(f"  Class weights range: [{builder.class_weights.min():.3f}, {builder.class_weights.max():.3f}]")

print("\n  ALL DATASET CHECKS PASSED")
