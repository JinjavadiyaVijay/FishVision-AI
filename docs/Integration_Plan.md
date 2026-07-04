# FishVision-AI — Integration Plan

> **Status:** Phase 1 — Architecture Review  
> **Last Updated:** 2026-07-04  
> **Author:** AI Technical Lead (Opus)  

---

## 1. Inference Pipeline Design

### 1.1 Real-time Flow (OAK-D)

```
Frame arrives (30 FPS)
    │
    ├── YOLO inference (~8ms, 640px)
    │       └── Returns N bounding boxes (generic "fish")
    │
    ├── Veto Gate check (~5ms, 320px)
    │       └── Blocks person/object overlapping fish boxes
    │
    ├── Crop extraction (~1ms)
    │       └── Extract N fish crops from frame
    │
    ├── BioCLIP batch inference (~25ms for 4 crops)
    │       └── Returns species + confidence for each crop
    │
    ├── Merge species into detections (~0ms)
    │
    ├── Tracker update (~0ms)
    │       └── Assigns persistent track IDs
    │
    ├── Biometrics (~0ms)
    │       └── Weight estimation, maturity classification
    │
    └── CSV log + display (~1ms)
```

**Total per-frame budget: ~40ms = 25 FPS** ← Acceptable for monitoring

### 1.2 BioCLIP Inference Wrapper

```python
class BioCLIPClassifier:
    """Production inference wrapper for fine-tuned BioCLIP 2."""
    
    def __init__(self, model_dir: Path, device: str = "cuda"):
        # Load base BioCLIP 2 model
        # Load LoRA adapter weights
        # Load classification head
        # Load species mapping
        # Set to eval mode
        # Warm up with dummy inference
    
    def predict_single(self, crop: np.ndarray) -> tuple[str, float]:
        """Classify a single fish crop.
        Returns: (species_name, confidence)
        """
    
    def predict_batch(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Classify multiple crops in a single forward pass.
        Returns: list of (species_name, confidence)
        """
    
    @property
    def species_names(self) -> list[str]:
        """Ordered list of species this model can classify."""
```

### 1.3 Batching Strategy

For real-time inference:
1. Collect all fish crops from current frame
2. Batch them into a single tensor (pad batch if needed)
3. Run ONE forward pass through BioCLIP
4. ViT inference scales well with batch size on GPU
5. Typical scene: 1-8 fish → batch of 1-8 crops

For dashboard (single image):
1. Same batching approach
2. No latency constraint → can use larger batch

### 1.4 Caching Strategy

```python
# Cache 1: Model stays in GPU memory
# - BioCLIP model loaded once at startup
# - Never moves between CPU/GPU

# Cache 2: Species embedding cache (if using text-based approach)
# - Pre-compute species text embeddings at startup
# - Reuse for every frame

# Cache 3: Crop resize cache
# - Pre-allocate reusable tensor for batch
# - Avoids repeated memory allocation
```

---

## 2. Integration into oak_runner.py

### 2.1 Changes Required

**New imports (top of file):**
```python
from classification.inference.classifier import BioCLIPClassifier
```

**New initialization (in run() function, after veto gate):**
```python
# Load BioCLIP species classifier
bioclip_model_dir = Path(__file__).resolve().parent.parent / "models" / "bioclip" / "production"
classifier = BioCLIPClassifier(bioclip_model_dir, device="cuda")
print(f"  BioCLIP classifier: {len(classifier.species_names)} species")
```

**Modified detection loop (after veto filtering, before tracker):**
```python
# After building raw_detections list (existing code)
# Add BioCLIP species classification:
if raw_detections:
    crops = []
    for det in raw_detections:
        crop = frame[det['y1']:det['y2'], det['x1']:det['x2']]
        crops.append(crop)
    
    species_predictions = classifier.predict_batch(crops)
    
    for det, (species, species_conf) in zip(raw_detections, species_predictions):
        det['species'] = species
        det['species_conf'] = species_conf
```

### 2.2 Backward Compatibility

The integration is designed to be **opt-in**:

```python
# If BioCLIP model directory doesn't exist, fall back to YOLO species
USE_BIOCLIP = bioclip_model_dir.exists()

if USE_BIOCLIP:
    classifier = BioCLIPClassifier(bioclip_model_dir)
else:
    classifier = None
    print("  [WARN] BioCLIP model not found — using YOLO species labels")

# In detection loop:
if classifier and raw_detections:
    crops = [frame[d['y1']:d['y2'], d['x1']:d['x2']] for d in raw_detections]
    predictions = classifier.predict_batch(crops)
    for det, (species, _) in zip(raw_detections, predictions):
        det['species'] = species
# else: species already set from YOLO model.names[cls_idx]
```

### 2.3 Lines Changed in oak_runner.py

| Line Range | Change | Impact |
|-----------|--------|--------|
| 1-36 (imports) | Add 1 import | Minimal |
| 181-188 (initialization) | Add 4 lines | Minimal |
| 305-315 (after detection list) | Add 8 lines | Core integration |
| — | No existing lines deleted | Full backward compatibility |

---

## 3. Integration into app.py

### 3.1 Changes Required

**New imports:**
```python
from classification.inference.classifier import BioCLIPClassifier
```

**After model loading, add classifier loading:**
```python
bioclip_path = Path("models/bioclip/production")
if bioclip_path.exists():
    classifier = st.cache_resource(BioCLIPClassifier)(bioclip_path, device="cuda")
else:
    classifier = None
```

**After YOLO inference, add classification:**
```python
if classifier and result.boxes is not None and len(result.boxes) > 0:
    crops = []
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    img_array = np.array(image)
    for box in boxes:
        x1, y1, x2, y2 = box
        crop = img_array[y1:y2, x1:x2]
        crops.append(crop)
    
    species_predictions = classifier.predict_batch(crops)
    species_override = [sp for sp, _ in species_predictions]
else:
    species_override = None

df = detections_to_dataframe(result, cm_per_pixel, adult_threshold_cm,
                              species_override=species_override)
```

### 3.2 predictor.py Update

Add `species_override` parameter to `detections_to_dataframe()`:

```python
def detections_to_dataframe(
    result,
    cm_per_pixel: float = 0.0,
    adult_threshold_cm: float = 10.0,
    species_override: list[str] | None = None,  # NEW
) -> pd.DataFrame:
    # ... existing code ...
    for idx, (box, class_id, conf) in enumerate(...):
        # Replace species name if override is provided
        if species_override and idx < len(species_override):
            species_name = species_override[idx]
        else:
            species_name = names.get(class_id, str(class_id))
        
        rows.append({
            "class": species_name,  # was: names.get(class_id, str(class_id))
            # ... rest unchanged
        })
```

---

## 4. biometrics.py Extension

### 4.1 Strategy

The current `biometrics.py` has weight coefficients and maturity thresholds for 13 common-name species. OzFish uses scientific names with hundreds of species.

**Approach:**
1. Keep existing common-name entries (backward compatibility)
2. Add a `species_biometrics.json` config file with scientific name entries
3. Load at runtime; merge with existing dicts
4. Use FishBase allometric coefficients where available
5. For unknown species, use family-level average coefficients

### 4.2 Family-Level Defaults

```python
FAMILY_WEIGHT_DEFAULTS = {
    "Lethrinidae":    {"a": 0.0150, "b": 3.00},  # Emperors
    "Labridae":       {"a": 0.0120, "b": 3.05},  # Wrasses
    "Lutjanidae":     {"a": 0.0160, "b": 2.98},  # Snappers
    "Acanthuridae":   {"a": 0.0175, "b": 3.00},  # Surgeonfishes
    "Serranidae":     {"a": 0.0180, "b": 2.95},  # Groupers
    "Pomacentridae":  {"a": 0.0100, "b": 3.10},  # Damselfishes
    "Carangidae":     {"a": 0.0200, "b": 2.90},  # Jacks
    "Balistidae":     {"a": 0.0280, "b": 2.85},  # Triggerfishes
    # ... etc
}
```

---

## 5. Confidence Calibration

### 5.1 Problem

BioCLIP fine-tuned models often produce overconfident predictions (softmax probabilities > 0.95 even when wrong). This is problematic for a monitoring system where confidence is displayed.

### 5.2 Solution: Temperature Scaling

After training, calibrate predictions using temperature scaling on the validation set:

```python
class CalibratedClassifier(BioCLIPClassifier):
    def __init__(self, *args, temperature: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
    
    def predict_batch(self, crops):
        logits = self._get_logits(crops)
        calibrated = logits / self.temperature
        probs = torch.softmax(calibrated, dim=-1)
        # ... return predictions
```

Temperature is optimized on the validation set using NLL minimization (typically T ∈ [1.2, 2.5]).

---

## 6. Graceful Degradation

### 6.1 Fallback Chain

```
1. BioCLIP available + species recognized → Use BioCLIP species
2. BioCLIP available + low confidence (< 0.3) → Use "Unknown Fish"
3. BioCLIP model not found → Fall back to YOLO species
4. YOLO species not in biometrics → Use family-level defaults
5. Family unknown → Use generic fish defaults
```

### 6.2 Unknown Species Handling

If BioCLIP confidence is below a threshold, report it transparently:

```python
MIN_SPECIES_CONF = 0.30

species, conf = classifier.predict_single(crop)
if conf < MIN_SPECIES_CONF:
    species = f"Unknown ({species}?)"  # Show best guess but flag uncertainty
```

---

## 7. Performance Optimization

### 7.1 GPU Memory During Inference

```python
# Use torch.inference_mode() instead of torch.no_grad()
# - Slightly faster
# - Prevents accidental gradient tracking

@torch.inference_mode()
def predict_batch(self, crops):
    # ...
```

### 7.2 Model Compilation (PyTorch 2.x)

```python
# Optional: Compile model for ~20% speedup
# Only if PyTorch >= 2.0
if hasattr(torch, 'compile'):
    self.model = torch.compile(self.model, mode="reduce-overhead")
```

### 7.3 Input Pre-processing Optimization

```python
# Pre-allocate batch tensor (avoid repeated allocation)
self._batch_buffer = torch.zeros(MAX_BATCH, 3, 224, 224, device=device)

# Use CUDA streams for async preprocessing
# Resize crops on GPU using torchvision.transforms.functional
```
