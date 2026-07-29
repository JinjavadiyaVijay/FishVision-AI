# FishVision-AI

**Real-time underwater fish detection and fine-grained species classification** using YOLOv8 + BioCLIP 2 with LoRA fine-tuning.

Detects fish in images or live video, crops each detection, and classifies it into one of **157 marine species** with **72.6% top-1 accuracy** and **91.7% top-5 accuracy**.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Model Performance](#model-performance)
- [System Architecture](#system-architecture)
- [Image Processing Pipeline](#image-processing-pipeline)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Module Reference](#module-reference)
- [Training Pipeline](#training-pipeline)
- [Dataset](#dataset)
- [Deployment Modes](#deployment-modes)
- [Configuration](#configuration)
- [Hardware Requirements](#hardware-requirements)

---

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the web app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Upload any underwater image — YOLO detects fish, BioCLIP identifies the species.

---

## Model Performance

### Test Set Evaluation (18,296 images · 157 species)

| Metric              | Score   |
|---------------------|---------|
| **Top-1 Accuracy**  | 72.64%  |
| **Top-5 Accuracy**  | 91.72%  |
| **Precision (Macro)** | 0.8443 |
| **Recall (Macro)**    | 0.7504 |
| **F1 Macro**          | 0.7723 |
| **F1 Weighted**       | 0.7479 |
| **Inference Speed**   | 5.5 ms/image (RTX 3050) |

### Accuracy Distribution Across 157 Species

| Accuracy Range | Species Count | Percentage |
|---------------|--------------|------------|
| ≥ 90%         | 55           | 35%        |
| 70–89%        | 52           | 33%        |
| 50–69%        | 30           | 19%        |
| < 50%         | 20           | 13%        |

Full per-species accuracy table is available in the Streamlit app (📊 All Species Accuracy tab) and in `experiments/bioclip2_full_20260716_160143/plots/evaluation_test.json`.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FishVision-AI                            │
├─────────────┬───────────────────────────────────────────────────┤
│  Frontend   │  Streamlit Web App (app.py)                      │
│             │  • Image upload → detection + classification     │
│             │  • Per-species accuracy browser (157 species)     │
│             │  • CSV export of detections                      │
├─────────────┼───────────────────────────────────────────────────┤
│  Pipeline   │  FishPipeline (src/pipeline.py)                  │
│             │  • Unified entry point for all consumers         │
│             │  • Lazy model loading, session caching            │
│             │  • Accepts PIL images or BGR numpy frames         │
├─────────────┼────────────────────┬──────────────────────────────┤
│  Detection  │  YOLOv8n           │  BioCLIP 2 ViT-L/14 + LoRA  │
│  Layer      │  (src/detector.py) │  (src/species_classifier.py) │
│             │  • 13-class fish   │  • 157-species classifier    │
│             │    localisation    │  • LoRA rank=16, alpha=32    │
│             │  • NMS + conf gate │  • Softmax → top-K ranked    │
├─────────────┼────────────────────┴──────────────────────────────┤
│  Edge       │  OAK-D Pro Live Detection (src/oak_runner.py)    │
│  Deployment │  • Stereo depth → real-world measurements        │
│             │  • Multi-object tracking (IoU + distance)         │
│             │  • Veto gate (blocks humans/objects)              │
│             │  • Biometric estimation (length, weight, maturity)│
│             │  • CSV logging + frame capture                   │
└─────────────┴───────────────────────────────────────────────────┘
```

---

## Image Processing Pipeline

This is the exact sequence that happens when you upload an image:

### Step 1: YOLO Fish Detection
```
Input Image (any size, RGB)
    │
    ▼
YOLOv8n (models/best.pt)
    │  • Resizes to 640×640 internally
    │  • Confidence threshold (default 0.25)
    │  • Non-Maximum Suppression (IoU = 0.45)
    │  • Outputs: list of bounding boxes (x1, y1, x2, y2) + class + confidence
    ▼
Detection DataFrame (one row per fish)
    │  • fish_id, class, confidence, bbox coordinates
    │  • Optional: estimated_length_cm, life_stage (if cm/pixel set)
    ▼
Annotated Frame (BGR numpy, boxes drawn on original image)
```

### Step 2: BioCLIP Species Classification (per detected fish)
```
For each detection:
    │
    ▼
Crop bounding box from original image
    │  • 10% margin added on all sides for context
    │  • Clipped to image boundaries
    ▼
BioCLIP 2 Preprocessing
    │  • Resize to 224×224 (bicubic interpolation, shortest-edge mode)
    │  • Center crop to 224×224
    │  • Normalize: mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758]
    │    (OpenAI CLIP standard normalisation)
    │  • Convert to float32 tensor
    ▼
BioCLIP 2 ViT-L/14 Backbone (frozen, from Hugging Face)
    │  • Vision Transformer with 24 layers, 1024-dim, 16 attention heads
    │  • LoRA adapters injected into attention + MLP layers (rank=16)
    │  • Outputs: 768-dimensional embedding vector
    ▼
Classification Head
    │  • LayerNorm(768) → Dropout(0.1) → Linear(768 → 157)
    │  • Outputs: 157 raw logits
    ▼
Softmax → Top-K ranking
    │  • Converts logits to probabilities
    │  • Returns top-K species with confidence scores
    ▼
Structured Result: FishDetection dataclass
    • fish_id, yolo_class, yolo_confidence, bbox
    • species (top-1), species_confidence
    • top_k_species: [{rank, species, confidence}, ...]
```

### Step 3: Results Rendering (Streamlit)
```
PipelineResult
    │
    ├── Metric cards (fish count, species ID'd, avg confidence, timing)
    ├── Annotated image with YOLO bounding boxes
    ├── BioCLIP species cards with confidence bars
    ├── Expandable per-fish detail (YOLO + BioCLIP top-K table)
    └── CSV download of all detections
```

---

## Tech Stack

### Core Frameworks

| Component           | Technology                | Version    | Purpose                           |
|---------------------|---------------------------|------------|-----------------------------------|
| Fish Detection      | **Ultralytics YOLOv8n**   | ≥ 8.2.0    | Real-time object detection        |
| Species Backbone    | **BioCLIP 2 ViT-L/14**   | via OpenCLIP | Biology-aware vision transformer |
| Fine-Tuning         | **LoRA (PEFT)**           | latest     | Parameter-efficient adaptation    |
| Deep Learning       | **PyTorch**               | ≥ 2.0      | Model training and inference      |
| Web Interface       | **Streamlit**             | ≥ 1.35.0   | Interactive UI                    |
| Image Processing    | **Pillow (PIL)**          | ≥ 10.0     | Image loading and cropping        |
| Data Processing     | **Pandas / NumPy**        | latest     | DataFrames and array ops          |
| Edge Hardware       | **DepthAI**               | v3.6.x     | OAK-D Pro stereo camera           |

### ML Architecture Details

| Parameter              | Value                                    |
|------------------------|------------------------------------------|
| Backbone               | `hf-hub:imageomics/bioclip-2` (ViT-L/14) |
| Backbone parameters    | ~304M (frozen during fine-tuning)         |
| LoRA rank              | 16                                        |
| LoRA alpha             | 32                                        |
| LoRA target modules    | `attn.out_proj`, `mlp.c_fc`, `mlp.c_proj` |
| Trainable parameters   | ~2.4M (LoRA adapters + classifier head)   |
| Classifier head        | LayerNorm → Dropout(0.1) → Linear(768→157)|
| Input resolution       | 224 × 224                                 |
| Normalisation          | OpenAI CLIP (mean/std)                    |
| Number of classes      | 157 species                               |
| Training epochs        | 30                                        |
| Best validation acc    | 73.8% (epoch 29)                          |

### YOLO Detection Model

| Parameter        | Value                          |
|------------------|--------------------------------|
| Architecture     | YOLOv8n (nano)                 |
| Classes          | 13 fish species (localisation) |
| Input resolution | 640 × 640 (auto-resized)       |
| Weights file     | `models/best.pt` (6.2 MB)      |

---

## Project Structure

```
fish_detection/
│
├── app.py                              # Streamlit web app (main entry point)
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── LICENSE                             # MIT License
├── .gitignore                          # Git exclusions
│
├── src/                                # Runtime modules (inference + edge)
│   ├── __init__.py
│   ├── pipeline.py                     # Unified YOLO + BioCLIP pipeline
│   ├── species_classifier.py           # BioCLIP inference bridge
│   ├── detector.py                     # YOLO model loading + inference
│   ├── predictor.py                    # YOLO results → DataFrame
│   ├── utils.py                        # Path resolution, helpers
│   ├── oak_runner.py                   # OAK-D Pro live detection loop
│   ├── oak_detector.py                 # OAK-D pipeline builder
│   ├── tracker.py                      # Multi-object fish tracker
│   ├── veto_gate.py                    # False-positive suppression
│   └── biometrics.py                   # Length/weight estimation
│
├── classification/                     # Model architecture + training code
│   ├── __init__.py
│   ├── config/                         # YAML configuration files
│   │   ├── model.yaml                  #   Backbone + LoRA + head config
│   │   ├── dataset.yaml                #   Dataset paths + split ratios
│   │   ├── training.yaml               #   Optimiser, scheduler, epochs
│   │   ├── augmentation.yaml           #   Data augmentation settings
│   │   ├── default.yaml                #   Global defaults
│   │   └── logging.yaml                #   Logging configuration
│   ├── models/                         # Neural network definitions
│   │   ├── backbone.py                 #   OpenCLIP backbone wrapper
│   │   └── classifier.py              #   Classification head (Linear/MLP)
│   ├── dataloader/                     # Dataset loading
│   │   └── fish_dataset.py             #   FishDatasetBuilder + FishSpeciesDataset
│   ├── dataset/                        # Data preparation (Phase 2)
│   │   ├── ozfish_parser.py            #   OzFish CSV → structured metadata
│   │   ├── data_validator.py           #   Image quality validation
│   │   ├── species_selector.py         #   Species filtering + selection
│   │   └── data_splitter.py            #   Train/val/test splitting
│   ├── preprocessing/                  # Image preprocessing
│   │   └── image_processor.py          #   Crop extraction + resizing
│   ├── trainer/                        # Training engine
│   │   └── training_engine.py          #   Full training loop + checkpointing
│   └── utilities/                      # Shared utilities
│       ├── device.py                   #   GPU detection + memory logging
│       ├── seed.py                     #   Reproducibility (seed management)
│       └── taxonomy.py                 #   Species catalog builder
│
├── scripts/                            # Executable scripts
│   ├── evaluate_model.py               # Test set evaluation (all metrics)
│   ├── train_bioclip.py                # BioCLIP LoRA fine-tuning
│   ├── prepare_dataset.py              # OzFish → processed crops
│   └── split_dataset.py                # Train/val/test splitting
│
├── models/                             # Model weights
│   ├── best.pt                         # YOLOv8n fish detector (production)
│   └── bioclip_production/             # BioCLIP production pointer
│       └── model_manifest.json         #   Points to active experiment checkpoint
│
├── experiments/                        # Training experiment outputs
│   └── bioclip2_full_20260716_160143/  # Active production experiment
│       ├── checkpoints/                #   best_accuracy.pt + metadata
│       ├── config/                     #   Frozen config + class_mapping.json
│       ├── logs/                       #   Training logs
│       ├── metrics/                    #   Per-epoch metrics
│       └── plots/                      #   evaluation_test.json + confusion matrix
│
├── datasets/                           # Processed dataset (ImageFolder layout)
│   ├── processed/                      #   train/ val/ test/ (157 species dirs)
│   ├── metadata/                       #   master_species_catalog.json
│   ├── reports/                        #   Dataset statistics
│   └── statistics/                     #   Split summaries
│
├── OzFish/                             # Raw source data (OzFish dataset)
│   ├── crop_metadata.csv               #   Crop-level annotations
│   ├── crops-zip/                      #   Original fish crop images
│   └── ...
│
└── logs/                               # OAK-D runtime logs
    ├── oak_detections.csv              #   Live detection log
    └── captures/                       #   Saved frame captures
```

---

## Module Reference

### `src/pipeline.py` — FishPipeline

The unified entry point. Both the Streamlit app and OAK-D runner use this.

```python
from src.pipeline import FishPipeline

pipeline = FishPipeline()               # auto-detects model paths
result = pipeline.run(pil_image)        # PIL → PipelineResult
result = pipeline.run_frame(bgr_frame)  # numpy BGR → PipelineResult

for det in result.detections:
    print(det.species, det.species_confidence, det.bbox)
```

**Key classes:**
- `FishPipeline` — orchestrates YOLO + BioCLIP with lazy loading
- `FishDetection` — structured result per detected fish
- `PipelineResult` — all detections + annotated frame + timing

### `src/species_classifier.py` — SpeciesClassifierInference

Low-level BioCLIP inference. Used internally by `FishPipeline`.

```python
from src.species_classifier import SpeciesClassifierInference

clf = SpeciesClassifierInference()     # auto-detects best checkpoint
preds = clf.classify(pil_crop)         # list of {rank, species, confidence}
preds = clf.classify_crop(image, bbox) # crop + classify in one call
```

### `src/detector.py` — YOLO Wrapper

```python
from src.detector import load_model, run_inference

model = load_model("models/best.pt")
result = run_inference(model, pil_image, conf=0.25, iou=0.45)
```

### `src/predictor.py` — Result Parser

```python
from src.predictor import detections_to_dataframe, summary_stats

df = detections_to_dataframe(yolo_result, cm_per_pixel=0.0)
stats = summary_stats(df)  # {total, species, avg_confidence}
```

---

## Training Pipeline

Training is fully automated via `scripts/train_bioclip.py`:

```bash
python scripts/train_bioclip.py
```

### Training Flow

```
1. Load config from classification/config/*.yaml
2. Build FishDatasetBuilder → train/val/test splits
3. Create OpenCLIPBackbone (BioCLIP 2 ViT-L/14)
4. Freeze backbone → apply LoRA adapters (rank=16)
5. Add ClassifierHead (768 → 157)
6. Train for 30 epochs:
   • AdamW optimiser (lr=1e-4, weight_decay=0.01)
   • Cosine annealing LR schedule (warmup: 2 epochs)
   • Mixed precision (FP16) for RTX 3050 6GB VRAM
   • Weighted random sampling for class imbalance
   • Save best_accuracy.pt on validation improvement
7. Export experiment to experiments/<name>/
```

### Training Configuration

All training parameters are in `classification/config/`:

- `model.yaml` — backbone name, LoRA rank/alpha, classifier type
- `training.yaml` — epochs, learning rate, batch size, scheduler
- `dataset.yaml` — data paths, split ratios, class filtering
- `augmentation.yaml` — data augmentation transforms

### Evaluation

```bash
python scripts/evaluate_model.py
```

Produces: Top-1/Top-5 accuracy, macro/weighted precision/recall/F1, per-class breakdown, confusion matrix PNG, and JSON results saved to `experiments/.../plots/`.

---

## Dataset

### Source: OzFish

The [OzFish dataset](https://github.com/open-AIMS/ozfish) contains annotated underwater imagery from Australian reefs.

### Processing Pipeline

```
OzFish/crops-zip/ (raw crops)
    │
    ▼
scripts/prepare_dataset.py
    │  • Parse crop_metadata.csv
    │  • Validate image quality (min bbox size, readability)
    │  • Build master_species_catalog.json
    │  • Filter species (min 100 images, exclude spp./empty labels)
    ▼
scripts/split_dataset.py
    │  • Video-level stratified splitting (70/15/15)
    │  • Ensures species appear in ALL splits
    │  • Random seed: 42
    ▼
datasets/processed/
    ├── train/   (85,391 images)
    │   ├── Abalistes_stellatus/
    │   ├── Abudefduf_bengalensis/
    │   └── ... (157 species)
    ├── val/     (18,295 images)
    └── test/    (18,296 images)
```

### Species Coverage

- **157 species** across 67 genera and 35 families
- **121,982 total images** (train + val + test)
- Species range from 15 to 1,675 images per species in test set
- 2 species excluded from training: `Lutjanus_erythropterus`, `Sphyraena_qenie` (insufficient data)

---

## Deployment Modes

### 1. Streamlit Web App (Primary)

```bash
streamlit run app.py
```

**Features:**
- Upload image → YOLO detection + BioCLIP classification
- Per-fish species cards with confidence bars
- Top-K species ranking per detection
- Full 157-species accuracy browser with search/sort/filter
- CSV download of results

### 2. OAK-D Pro Live Detection

```bash
python -m src.oak_runner
```

**Features:**
- Real-time stereo depth camera feed
- Multi-object fish tracking (IoU + distance matching)
- 3-layer false-positive defence:
  1. **Veto Gate** — COCO-pretrained YOLOv8n blocks humans/objects
  2. **Size Filter** — species-specific max-length rejection
  3. **Bbox Filter** — min area, max aspect ratio, frame coverage
- Biometric estimation (length, width, weight, maturity)
- BioCLIP species overlay on video feed
- CSV logging + automatic frame capture

### 3. Python API

```python
from src.pipeline import FishPipeline
from PIL import Image

pipeline = FishPipeline(bioclip_top_k=5)
result = pipeline.run(Image.open("underwater.jpg"))

for det in result.detections:
    print(f"Fish #{det.fish_id}: {det.display_species} ({det.display_confidence:.1%})")
    print(f"  YOLO: {det.yolo_class} ({det.yolo_confidence:.1%})")
    print(f"  BBox: {det.bbox}")
    for pred in det.top_k_species:
        print(f"    #{pred['rank']} {pred['species']} ({pred['confidence']:.1%})")
```

---

## Configuration

### classification/config/model.yaml

```yaml
backbone:
  library: "open_clip"
  name: "hf-hub:imageomics/bioclip-2"
  image_size: 224

classifier:
  type: "linear"
  dropout: 0.1

lora:
  enabled: true
  rank: 16
  alpha: 32
  dropout: 0.1
  bias: "none"
```

### classification/config/dataset.yaml

```yaml
dataset:
  crops_dir: "OzFish/crops-zip/assets/projects/FDFML/crops"
  metadata_csv: "OzFish/crop_metadata.csv"
  processed_dir: "datasets/processed"
  class_mapping_source: "catalog"
  master_catalog: "datasets/metadata/master_species_catalog.json"

split:
  ratios: [0.70, 0.15, 0.15]
  random_seed: 42
  min_images_per_species: 100
```

---

## Hardware Requirements

### Minimum (Inference)

| Component | Requirement |
|-----------|-------------|
| GPU       | NVIDIA GPU with ≥ 4GB VRAM (or CPU fallback) |
| RAM       | 8 GB |
| Disk      | ~2 GB (models + dependencies) |

### Recommended (Training)

| Component | Requirement |
|-----------|-------------|
| GPU       | NVIDIA RTX 3050 6GB or better |
| RAM       | 16 GB |
| Disk      | ~50 GB (dataset + experiments) |

### Tested On

- **GPU:** NVIDIA GeForce RTX 3050 6GB Laptop GPU
- **OS:** Windows 10/11
- **Python:** 3.10+
- **CUDA:** 12.x

---

## License

MIT License. See [LICENSE](LICENSE).
