# FishVision-AI — Training Plan

> **Status:** Phase 1 — Architecture Review  
> **Last Updated:** 2026-07-04  
> **Author:** AI Technical Lead (Opus)  

---

## 1. Model Selection: BioCLIP 2

### 1.1 Why BioCLIP 2 (ViT-L/14)?

| Alternative | Pros | Cons | Verdict |
|------------|------|------|---------|
| **BioCLIP 2 (ViT-L/14)** | Pre-trained on TreeOfLife-200M; strong bio-domain transfer; 304M params; fits RTX 3050 | Requires fine-tuning for OzFish species | ✅ **Selected** |
| **BioCLIP 2.5 (ViT-H/14)** | Higher accuracy ceiling; more capacity | 632M params; risks OOM on 6GB VRAM; slower inference | ❌ Too large |
| **ResNet-50 ImageNet** | Small, fast, well-understood | No biological pre-training; lower fish accuracy | ❌ Worse transfer |
| **EfficientNet** | Good accuracy/size tradeoff | No biological pre-training | ❌ Worse transfer |
| **Standard CLIP** | Good zero-shot | Not biology-specific | ❌ Worse domain fit |

### 1.2 BioCLIP 2 Architecture Details

```
Model:          BioCLIP 2 (imageomics/bioclip-2)
Base:           ViT-L/14 (Vision Transformer, Large, 14×14 patches)
Parameters:     ~304M (visual encoder)
Input:          224×224 RGB images
Pre-training:   TreeOfLife-200M (biological images with taxonomic labels)
Initialize:     From CLIP (LAION-2B) → BioCLIP training
Output:         768-dimensional embedding
```

---

## 2. Fine-Tuning Strategy: LoRA

### 2.1 LoRA vs Full Fine-Tuning

| Approach | Trainable Params | VRAM (batch=8) | Time per Epoch | Risk |
|----------|-----------------|----------------|----------------|------|
| **Full Fine-Tuning** | ~304M | >8 GB ❌ | ~15 min | Catastrophic forgetting |
| **LoRA (r=8)** | ~2.4M (0.8%) | ~3.8 GB ✅ | ~5 min | Minimal |
| **LoRA (r=16)** | ~4.8M (1.6%) | ~4.0 GB ✅ | ~5 min | Minimal |
| **Linear Probe** | ~24K | ~1.5 GB ✅ | ~2 min | Limited capacity |

### 2.2 Recommendation: LoRA (r=16)

**Reasoning:**
1. **VRAM**: Full fine-tuning doesn't fit on RTX 3050 6GB. LoRA fits comfortably.
2. **Catastrophic Forgetting**: LoRA preserves BioCLIP's pre-trained biological knowledge.
3. **Data Efficiency**: With 30 species and ~38K images, we don't have enough data to justify training 304M params from scratch.
4. **LoRA rank 16**: Provides more capacity than r=8 while still fitting in VRAM. Good for 30-class fish classification where inter-species differences can be subtle.
5. **Target Modules**: Apply LoRA to all attention layers (`q_proj`, `k_proj`, `v_proj`, `out_proj`) in the visual encoder.

### 2.3 LoRA Configuration

```yaml
lora:
  rank: 16
  alpha: 32            # alpha/rank = 2.0 (standard scaling)
  dropout: 0.1
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - out_proj
  bias: none
  task_type: FEATURE_EXTRACTION
```

---

## 3. Training Hyperparameters

### 3.1 Recommended Configuration for RTX 3050

```yaml
training:
  # Core
  epochs: 30
  batch_size: 8                    # Max safe for 6GB VRAM
  gradient_accumulation_steps: 4   # Effective batch = 32
  
  # Optimizer
  optimizer: AdamW
  learning_rate: 2e-4              # Higher than full FT; LoRA can handle it
  weight_decay: 0.01
  betas: [0.9, 0.999]
  
  # Scheduler
  scheduler: cosine_with_warmup
  warmup_steps: 200                # ~5% of total steps
  min_lr_ratio: 0.01               # Final LR = 2e-6
  
  # Loss
  loss: cross_entropy
  label_smoothing: 0.1             # Prevents overconfident predictions
  class_weights: balanced          # Computed from training set
  
  # Precision
  mixed_precision: fp16            # Halves activation memory
  
  # Regularization
  dropout: 0.1                     # In LoRA layers
  
  # Checkpointing
  save_strategy: epoch
  save_best_only: true
  monitor: val_accuracy
  
  # Early stopping
  early_stopping:
    patience: 5                    # Stop after 5 epochs without improvement
    min_delta: 0.001
  
  # Reproducibility
  seed: 42
  deterministic: true
```

### 3.2 Hyperparameter Justification

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| **batch_size=8** | 8 | Maximum that fits in 6GB VRAM with LoRA + mixed precision |
| **grad_accum=4** | 4 | Effective batch=32 for stable gradients without extra VRAM |
| **lr=2e-4** | 2e-4 | LoRA adapters train faster than full model; 2e-4 is standard for PEFT |
| **epochs=30** | 30 | With early stopping, actual training is likely 15-20 epochs |
| **AdamW** | — | Standard for transformers; decoupled weight decay |
| **Cosine warmup** | — | Smooth LR decay; warmup prevents early gradient explosions |
| **label_smoothing=0.1** | 0.1 | Fish species can be visually ambiguous; soft labels improve calibration |
| **class_weights=balanced** | — | Counters class imbalance (punctulatus: 6K vs others: 500) |
| **fp16** | — | Halves memory for activations; safe for ViT training |

---

## 4. Augmentation Strategy

### 4.1 Biologically Meaningful Augmentations

| Augmentation | Use? | Reasoning |
|-------------|------|-----------|
| **Horizontal flip** | ✅ Yes | Fish swim in both directions; preserves all morphological features |
| **Random rotation (±15°)** | ✅ Yes | Fish tilt slightly; mimics natural swimming angle |
| **Color jitter (brightness, contrast)** | ✅ Yes | Underwater lighting varies with depth, turbidity, time of day |
| **Hue shift (slight, ±10)** | ⚠️ Careful | Water color varies but too much distorts diagnostic coloration |
| **Random crop (85-100%)** | ✅ Yes | Simulates imperfect bounding box crops |
| **Gaussian blur (σ=0.5-1.0)** | ✅ Yes | Underwater turbidity and motion blur |
| **Random erasing (5-15%)** | ✅ Yes | Simulates partial occlusion by coral, other fish |

### 4.2 Augmentations to AVOID

| Augmentation | Avoid? | Reasoning |
|-------------|--------|-----------|
| **Vertical flip** | ❌ Never | Fish don't swim upside down; creates unrealistic samples |
| **Large rotation (>30°)** | ❌ Never | Artificial; fish rarely rotate beyond 15-20° while swimming |
| **Heavy color distortion** | ❌ Never | Destroys species-diagnostic color patterns (stripes, spots) |
| **Cutout/CutMix** | ⚠️ Avoid | Could remove fins, tail, or head — key diagnostic features |
| **Style transfer** | ❌ Never | Changes texture which is species-diagnostic |
| **Extreme scale** | ❌ Never | ViT input is fixed 224×224; extreme scale creates artifacts |

### 4.3 Recommended Augmentation Pipeline

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize(256),                      # Slightly larger than model input
    transforms.RandomCrop(224),                   # Random crop to 224×224
    transforms.RandomHorizontalFlip(p=0.5),       # Fish swim both ways
    transforms.RandomRotation(degrees=15),         # Natural swimming tilt
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2,
        saturation=0.15, hue=0.05                 # Conservative: preserve species color
    ),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.0))
    ], p=0.3),                                    # Underwater turbidity
    transforms.RandomErasing(
        p=0.2, scale=(0.02, 0.10),
        ratio=(0.3, 3.3)                          # Small occlusions
    ),
    transforms.ToTensor(),
    transforms.Normalize(                          # BioCLIP/CLIP normalization
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    ),
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    ),
])
```

---

## 5. Training Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Training Pipeline                          │
│                                                                 │
│  Config (YAML) ──► DataLoader ──► BioCLIP 2 (frozen) + LoRA    │
│                        │              │                          │
│                   Augmentation    Classification Head (linear)   │
│                        │              │                          │
│                        ▼              ▼                          │
│                    Forward Pass ──► CE Loss (label-smoothed)     │
│                        │              │                          │
│                        ▼              ▼                          │
│                    Backward (LoRA params only)                   │
│                        │                                         │
│                        ▼                                         │
│                    AdamW Step ──► Cosine LR Schedule             │
│                        │                                         │
│                        ▼                                         │
│                    Validation ──► Metrics ──► Checkpoint         │
│                        │              │                          │
│                        ▼              ▼                          │
│                    Early Stop     TensorBoard                    │
└─────────────────────────────────────────────────────────────────┘
```

### 5.1 Classification Head Design

Two options for adapting BioCLIP 2 to N-class classification:

**Option A: Linear probe on frozen embeddings + LoRA**
```python
# Extract 768-dim embedding from ViT-L/14
# Add: nn.Linear(768, num_classes)
```

**Option B: Cosine similarity with text embeddings**
```python
# Use BioCLIP's text encoder to generate species embeddings
# Classify by nearest-neighbor in embedding space
```

**Recommendation: Option A** — Direct linear classifier on top of LoRA-adapted visual features.
- Simpler training loop
- Faster inference (no text encoding needed)
- Text encoder embeddings for species names may not be meaningful for scientific Latin names
- Can add text-embedding similarity as an auxiliary loss later

---

## 6. Evaluation Metrics

### 6.1 Primary Metrics

| Metric | Why | Formula |
|--------|-----|---------|
| **Top-1 Accuracy** | Overall performance | Correct / Total |
| **Top-5 Accuracy** | Practical (shows if correct species is in top 5) | Correct-in-top-5 / Total |
| **Macro F1-Score** | Class-balanced performance | Mean of per-class F1 |
| **Weighted F1-Score** | Overall weighted by class frequency | Weighted mean of per-class F1 |
| **Per-class Precision/Recall** | Identifies weak classes | Per species |

### 6.2 Secondary Metrics

| Metric | Why |
|--------|-----|
| **Confusion Matrix** | Identifies systematic misclassifications (e.g., Lethrinus species confusion) |
| **Per-family Accuracy** | Tests generalization across fish families |
| **Confidence Calibration** | Ensures model confidence ≈ actual accuracy |
| **Inference Latency** | Must be < 50ms per crop for real-time |

### 6.3 Target Performance (V1)

| Metric | Target | Stretch Goal |
|--------|--------|-------------|
| Top-1 Accuracy | > 80% | > 90% |
| Top-5 Accuracy | > 95% | > 98% |
| Macro F1 | > 0.75 | > 0.85 |
| Inference (per crop) | < 50ms | < 30ms |
| Inference (4-crop batch) | < 80ms | < 50ms |

---

## 7. Training Monitoring

### 7.1 TensorBoard Logging

```
Per Step:
  - train/loss
  - train/learning_rate

Per Epoch:
  - train/accuracy
  - val/accuracy
  - val/loss
  - val/top5_accuracy
  - val/macro_f1
  - val/per_class_accuracy (histogram)
```

### 7.2 Checkpointing Strategy

```
Save:
  - Best model (by val_accuracy) → models/bioclip/v1/best.pt
  - Last model → models/bioclip/v1/last.pt
  - Every 5 epochs → models/bioclip/v1/epoch_{n}.pt (kept for analysis)

Save content:
  - LoRA adapter weights (tiny, ~10MB)
  - Classification head weights
  - Optimizer state (for resume)
  - Training config
  - Species mapping
  - Best metrics
```
