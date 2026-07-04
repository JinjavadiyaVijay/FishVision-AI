# FishVision-AI — Architecture Document

> **Status:** Phase 1 — Architecture Review  
> **Last Updated:** 2026-07-04  
> **Author:** AI Technical Lead (Opus)  

---

## 1. System Overview

FishVision-AI is a real-time fish detection, tracking, and biometric measurement system. The current pipeline uses a single YOLO model for both detection and species classification. This upgrade introduces a **two-stage pipeline** that decouples detection from classification using BioCLIP 2.

### Current Pipeline (v1)

```
Camera/Image → YOLOv8n → Detection + Classification (13 species) → Tracker → Biometrics → CSV → Dashboard
```

### Proposed Pipeline (v2)

```
Camera/Image → YOLOv8n → Detection Only (fish/no-fish) → Crop Fish Regions → BioCLIP 2 (ViT-L/14) → Species Classification (N species) → Tracker → Biometrics → CSV → Dashboard
```

### Architectural Rationale

| Decision | Reasoning |
|----------|-----------|
| **Keep YOLO for detection** | YOLO excels at real-time object localization. Retraining it for 497 species is impractical — OzFish doesn't provide YOLO-format bounding boxes for all species. |
| **BioCLIP 2 for classification** | Pre-trained on TreeOfLife-200M (biological images). ViT-L/14 architecture. Strong zero-shot and fine-tuned performance on species classification. |
| **Two-stage over single-stage** | Decoupling allows independent improvement of each stage. Detection model stays fast; classification model can be more accurate on crops. |
| **BioCLIP 2 over 2.5** | ViT-L/14 (304M params) vs ViT-H/14 (632M params). RTX 3050 6GB VRAM. BioCLIP 2 fits comfortably; 2.5 risks OOM during training and slows inference. |

---

## 2. Current Project Analysis

### Source Code Modules

| File | Lines | Responsibility | BioCLIP Impact |
|------|-------|---------------|----------------|
| `detector.py` | 68 | YOLO model loading & inference | Unchanged |
| `predictor.py` | 108 | YOLO results → DataFrame | Minor update (accept species override) |
| `tracker.py` | 225 | IoU-based fish tracker | Unchanged (uses string species) |
| `biometrics.py` | 78 | Weight/maturity estimation | Extended (add OzFish species) |
| `veto_gate.py` | 164 | COCO-based false positive gate | Unchanged |
| `oak_detector.py` | ~170 | OAK-D hardware abstraction | Unchanged |
| `oak_runner.py` | 400 | Live detection loop | Modified (add BioCLIP step) |
| `visualization.py` | ~80 | Streamlit rendering | Unchanged |
| `utils.py` | ~50 | Path resolution | Unchanged |
| `app.py` | 136 | Streamlit dashboard | Modified (add BioCLIP step) |

### Key Design Observations

1. **Clean module separation** — Each `src/` file has a single responsibility
2. **Tracker is species-aware** — Matches detections by species AND IoU (tracker.py line 167)
3. **Three-layer false-positive defence** — Veto gate → size filter → bbox filter
4. **Biometrics are species-specific** — Weight coefficients and maturity thresholds are per-species dicts
5. **Current species names are informal** — "GoldFish", "ClownFish", etc. OzFish uses scientific taxonomy

---

## 3. Proposed Project Structure (v2)

```
fish_detection/
├── app.py                          # Streamlit dashboard (MINOR UPDATE)
├── data.yaml                       # YOLO training config (UNCHANGED)
├── requirements.txt                # Updated with BioCLIP deps
│
├── src/                            # EXISTING — MOSTLY UNCHANGED
│   ├── biometrics.py               # Extended with OzFish species
│   ├── detector.py                 # UNCHANGED
│   ├── oak_detector.py             # UNCHANGED
│   ├── oak_runner.py               # MODIFIED — integrates BioCLIP
│   ├── predictor.py                # MINOR UPDATE
│   ├── tracker.py                  # UNCHANGED
│   ├── utils.py                    # UNCHANGED
│   ├── veto_gate.py                # UNCHANGED
│   └── visualization.py            # UNCHANGED
│
├── classification/                  # NEW — BioCLIP species classification
│   ├── __init__.py
│   ├── config/
│   │   ├── default.yaml             # Default hyperparameters
│   │   └── rtx3050.yaml             # GPU-specific overrides
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── ozfish_parser.py         # Parse OzFish metadata & annotations
│   │   ├── data_validator.py        # Image integrity, label consistency
│   │   ├── data_splitter.py         # Stratified train/val/test splitting
│   │   └── species_selector.py      # V1 species selection logic
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── image_processor.py       # Resize, normalize, quality checks
│   │   └── deduplicator.py          # Perceptual hash duplicate detection
│   ├── augmentation/
│   │   ├── __init__.py
│   │   └── fish_augmentation.py     # Biologically valid augmentations
│   ├── dataloader/
│   │   ├── __init__.py
│   │   └── bioclip_dataset.py       # PyTorch Dataset + DataLoader
│   ├── trainer/
│   │   ├── __init__.py
│   │   ├── bioclip_trainer.py       # Training loop with LoRA
│   │   └── callbacks.py             # Checkpointing, early stopping, logging
│   ├── evaluator/
│   │   ├── __init__.py
│   │   └── evaluator.py             # Per-class metrics, confusion matrix
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── classifier.py            # BioCLIP inference wrapper
│   │   └── batch_classifier.py      # Batched inference for real-time
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── classification_metrics.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py                 # Training curves, confusion matrices
│   └── utilities/
│       ├── __init__.py
│       ├── taxonomy.py              # Scientific ↔ common name mapping
│       └── device.py                # GPU memory checks, device selection
│
├── datasets/                        # NEW — Organized data directory
│   ├── raw/
│   │   └── OzFish/                  # Symlink or copy from OzFish/
│   ├── processed/
│   │   ├── train/                   # species_name/image.png
│   │   ├── validation/
│   │   └── test/
│   ├── metadata/
│   │   ├── species_mapping.json
│   │   ├── label_statistics.csv
│   │   └── split_manifest.json
│   ├── reports/
│   ├── statistics/
│   ├── cache/
│   └── temporary/
│
├── models/
│   ├── best.pt                      # YOLO (existing)
│   ├── last.pt
│   ├── yolo26n.pt
│   ├── yolov8n.pt
│   └── bioclip/                     # NEW
│       ├── v1/
│       └── production/
│
├── scripts/                         # NEW — CLI entry points
│   ├── prepare_dataset.py
│   ├── train_bioclip.py
│   ├── evaluate_bioclip.py
│   └── benchmark_inference.py
│
├── docs/                            # NEW — Design documentation
│   ├── Architecture.md
│   ├── Roadmap.md
│   ├── Dataset_Plan.md
│   ├── Training_Plan.md
│   ├── Integration_Plan.md
│   └── Decisions.md
│
├── tests/                           # NEW — Test suite
│   ├── test_dataset.py
│   ├── test_classifier.py
│   └── test_integration.py
│
└── configs/
    └── logging.yaml
```

### Module Responsibility Matrix

| Module | Responsibility | Why Separate? |
|--------|---------------|---------------|
| `classification/dataset/` | Parse, validate, split OzFish data | Data engineering is independent of model choice |
| `classification/preprocessing/` | Image quality checks, dedup | Reusable across future datasets |
| `classification/augmentation/` | Training-time transforms | Augmentation strategy is domain-specific |
| `classification/dataloader/` | PyTorch Dataset/DataLoader | Framework coupling isolated |
| `classification/trainer/` | Training loop, LoRA setup | Model-specific training logic |
| `classification/evaluator/` | Metrics computation | Decoupled from training for standalone eval |
| `classification/inference/` | Production inference wrapper | Different requirements from training (no grad, batching) |
| `classification/metrics/` | Metric definitions | Reusable across trainer and evaluator |
| `classification/visualization/` | Plots and figures | Optional dependency (matplotlib) |
| `classification/utilities/` | Taxonomy, device utils | Cross-cutting concerns |
| `classification/config/` | YAML hyperparameters | Reproducibility |

---

## 4. Integration Points

The BioCLIP classifier integrates into the existing pipeline at **exactly two points**:

### Integration Point 1: `oak_runner.py` (Real-time Pipeline)

**Current** (line 253): YOLO returns species directly.
```python
species = model.names[cls_idx]
```

**Proposed**: YOLO detects "fish" generically; BioCLIP classifies species.
```python
# After YOLO detection loop collects raw_detections:
crops = [frame[d['y1']:d['y2'], d['x1']:d['x2']] for d in raw_detections]
species_predictions = bioclip_classifier.predict_batch(crops)
for det, species in zip(raw_detections, species_predictions):
    det['species'] = species
```

### Integration Point 2: `app.py` (Streamlit Dashboard)

**Current**: YOLO result contains species.
```python
df = detections_to_dataframe(result, cm_per_pixel, adult_threshold_cm)
```

**Proposed**: BioCLIP classifies after YOLO detects.
```python
crops = extract_crops(image, result)
species = classifier.predict_batch(crops)
df = detections_to_dataframe(result, cm_per_pixel, adult_threshold_cm, species_override=species)
```

### What Does NOT Change

| Component | Status | Notes |
|-----------|--------|-------|
| `detector.py` | ✅ Unchanged | |
| `tracker.py` | ✅ Unchanged | Receives species from BioCLIP instead of YOLO |
| `veto_gate.py` | ✅ Unchanged | |
| `oak_detector.py` | ✅ Unchanged | |
| `visualization.py` | ✅ Unchanged | |
| `biometrics.py` | ⚠️ Extended | Add OzFish species to weight/maturity tables |
| `predictor.py` | ⚠️ Minor update | Accept `species_override` parameter |

---

## 5. Technology Stack

| Component | Choice | Justification |
|-----------|--------|---------------|
| **Detection** | YOLOv8n (existing) | Already trained, proven, fast |
| **Classification** | BioCLIP 2 (ViT-L/14) | Best accuracy/VRAM tradeoff for RTX 3050 6GB |
| **Fine-tuning** | LoRA (via PEFT) | 10-100× fewer trainable parameters; fits in 6GB |
| **Framework** | PyTorch + HuggingFace transformers + PEFT | Industry standard; BioCLIP published on HF |
| **Dataset** | OzFish crops (80,823 images, 497 species) | Pre-cropped; directly usable for classification |
| **Config** | YAML files | Reproducible experiments |
| **Logging** | Python logging + CSV | Consistent with existing project |
| **Experiment tracking** | TensorBoard | Lightweight, no external service required |

---

## 6. Hardware Constraints (RTX 3050 Laptop — 6GB VRAM)

### VRAM Budget: Training

| Component | Estimate |
|-----------|----------|
| BioCLIP 2 frozen backbone | ~1.2 GB |
| LoRA adapters | ~50 MB |
| Optimizer states | ~100 MB |
| Activations (batch_size=8) | ~2.0 GB |
| Gradients | ~0.5 GB |
| **Total** | **~3.85 GB** |
| Safety margin | ~2.15 GB |

### VRAM Budget: Inference (Real-time)

| Component | Estimate |
|-----------|----------|
| YOLO fish detector | ~20 MB |
| YOLO veto gate | ~20 MB |
| BioCLIP 2 classifier | ~1.2 GB |
| Image batch (4 crops) | ~50 MB |
| **Total** | **~1.3 GB** |
| Remaining for OS/display | ~4.7 GB ✅ |

---

## 7. Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|-----------|
| BioCLIP 2 too slow for real-time | High | Low | Batch crops, cache embeddings, async inference |
| OzFish label noise | Medium | Medium | Automated validation pipeline, manual review |
| Class imbalance (2–6,095/species) | High | Certain | Strategic species selection; oversampling; weighted loss |
| VRAM OOM during training | High | Low | LoRA, gradient accumulation, mixed precision |
| Taxonomy inconsistency (sp, spp) | Medium | Certain | Auto-clean labels; exclude unidentifiable |
| YOLO species field in tracker | Low | None | Tracker uses string matching — direct replacement |

---

## 8. Open Questions (Require Approval)

1. **YOLO Mode**: Should we retrain YOLO as a single-class "fish" detector, or keep the current 13-species model and just ignore its species predictions?
   - **Recommendation**: Keep as-is initially. Use YOLO for detection only; discard its species label. Retrain as single-class later if desired.

2. **Dataset Location**: Should processed data live inside `fish_detection/datasets/` or in a separate directory outside the repo?
   - **Recommendation**: Inside the project at `datasets/`. Add to `.gitignore`.

3. **Species Set for V1**: How many species should the first version target?
   - **Recommendation**: 20-30 species with ≥100 crops each. Full analysis in Dataset_Plan.md.

4. **Naming Convention**: The existing system uses common names ("GoldFish"). OzFish uses scientific names ("Lethrinus punctulatus"). Which should be primary?
   - **Recommendation**: Use scientific names internally; provide a mapping table for display.
