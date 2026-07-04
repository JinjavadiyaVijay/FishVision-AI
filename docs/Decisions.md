# FishVision-AI — Decisions Log

> **Status:** Phase 2 — Dataset Analysis & Engineering  
> **Last Updated:** 2026-07-04  
> **Author:** AI Technical Lead (Opus)  

---

## Decision Format

Each decision follows this template:
- **Context**: What problem are we solving?
- **Options Considered**: What alternatives exist?
- **Decision**: What did we choose?
- **Rationale**: Why?
- **Status**: Proposed / Approved / Implemented

---

## DEC-001: Classification Model Selection

**Context:** Need a species classifier that runs on RTX 3050 6GB alongside YOLO.

**Options:**
| Option | Params | VRAM (inference) | Bio Pre-training | Accuracy Ceiling |
|--------|--------|-----------------|-----------------|-----------------|
| BioCLIP 2 (ViT-L/14) | 304M | ~1.2 GB | ✅ TreeOfLife-200M | High |
| BioCLIP 2.5 (ViT-H/14) | 632M | ~2.5 GB | ✅ TreeOfLife-200M | Higher |
| ResNet-50 ImageNet | 25M | ~0.2 GB | ❌ | Medium |
| EfficientNet-B4 | 19M | ~0.15 GB | ❌ | Medium |

**Decision:** BioCLIP 2 (ViT-L/14)

**Rationale:** Best accuracy/VRAM tradeoff. BioCLIP 2.5 is too large for 6GB VRAM during training. Generic ImageNet models lack biological pre-training. BioCLIP 2 was specifically trained on 200M biological images.

**Status:** ✅ APPROVED (2026-07-04)

---

## DEC-002: Fine-Tuning Approach

**Context:** How to adapt BioCLIP 2 to OzFish species.

**Options:**
| Option | Trainable Params | VRAM (train) | Forgetting Risk |
|--------|-----------------|-------------|-----------------|
| Full fine-tuning | 304M | >8 GB ❌ | High |
| LoRA (r=16) | 4.8M | ~4 GB ✅ | Low |
| LoRA (r=8) | 2.4M | ~3.8 GB ✅ | Low |
| Linear probe only | 24K | ~1.5 GB ✅ | None |

**Decision:** LoRA with rank=16

**Rationale:** Full FT exceeds VRAM. Linear probe may be too limited for distinguishing similar species (multiple Lethrinus, Lutjanus). LoRA r=16 gives enough capacity while fitting in 6GB with mixed precision.

**Status:** ✅ APPROVED (2026-07-04)

---

## DEC-003: V1 Species Set Size

**Context:** OzFish has 497 species. How many for V1?

**Options:**
| Option | Species | Images | Risk |
|--------|---------|--------|------|
| All 497 | 497 | 80K | Extreme imbalance, noisy labels |
| Top 100 (≥50 samples) | 100 | ~73K | Moderate imbalance |
| Top 30 (≥500 samples) | 30 | ~38K | Well-balanced |
| Top 10 | 10 | ~27K | Too easy |

**Decision:** Data-driven species selection — analyze full OzFish distribution first, then recommend optimal subset and minimum image threshold based on statistical evidence.

**Rationale (user-modified):** Do not hard-code 30 species. Run complete distribution analysis first. Recommend based on actual data shape, inflection points in the class-count curve, and validation set viability per class.

**Status:** ✅ APPROVED with modification (2026-07-04)

---

## DEC-004: Data Splitting Strategy

**Context:** How to split data into train/val/test without data leakage.

**Options:**
| Option | Leakage Risk | Data Utilization |
|--------|-------------|-----------------|
| Random image split | HIGH (L/R pairs leak) | Maximum |
| Video-level split | None | High |
| Location-level split | None | Medium |

**Decision:** Video-level split (70/15/15)

**Rationale:** Each video shows different fish in different locations. Splitting by video ensures no individual fish appears in multiple splits. L/R stereo pairs are always kept together.

**Status:** ✅ APPROVED (2026-07-04)

---

## DEC-005: Taxonomy Naming Convention

**Context:** Existing system uses common names ("GoldFish"). OzFish uses scientific names ("Lethrinus punctulatus").

**Options:**
| Option | Pro | Con |
|--------|-----|-----|
| Scientific names only | Unambiguous, standard | Less user-friendly |
| Common names only | User-friendly | Ambiguous, many species lack common names |
| Scientific primary + common display | Best of both | More code complexity |

**Decision:** Scientific names as canonical identifiers internally; common names for display only. Generate a master species catalog with: scientific_name, common_name, family, genus, class_id, image_count, dataset_source.

**Rationale (user-confirmed):** Scientific names are unambiguous, standard in ichthyology, and used in OzFish metadata directly. The master catalog is the single source of truth for the entire pipeline.

**Status:** ✅ APPROVED with catalog requirement (2026-07-04)

---

## DEC-006: YOLO Mode During Transition

**Context:** Should we retrain YOLO as single-class "fish" or keep 13-species model?

**Options:**
| Option | Effort | Risk |
|--------|--------|------|
| Keep 13-species, ignore species label | Zero | YOLO may miss species not in its 13 classes |
| Retrain as single-class | Medium | Requires retraining setup |
| Use pre-trained COCO "fish" class | Zero | COCO doesn't have fish class |

**Decision:** Keep existing YOLO model unchanged. Use ONLY for fish bounding boxes. All species labels from YOLO are discarded. BioCLIP 2 is the sole species classifier.

**Rationale (user-confirmed):** Zero disruption to existing pipeline. YOLO's species vocabulary (13 aquarium fish) is irrelevant to OzFish wild species. Clean separation of concerns.

**Status:** ✅ APPROVED (2026-07-04)

---

## DEC-007: Classification Head Design

**Context:** How to classify species from BioCLIP embeddings.

**Options:**
| Option | Inference Speed | Extensibility |
|--------|----------------|---------------|
| Linear classifier (768 → N) | Fastest | Must retrain to add species |
| Cosine similarity to text embeddings | Slower | Zero-shot extensible |
| MLP (768 → 512 → N) | Fast | Must retrain |

**Decision:** Linear classifier with LoRA-adapted features

**Rationale:** Fastest inference. Text encoder may not produce meaningful embeddings for Latin species names. Adding species later is fine — LoRA retraining is fast (~30 min).

**Status:** ✅ APPROVED (2026-07-04)

---

## DEC-008: Beyond-BioCLIP Improvements

**Context:** Opportunities identified during architecture review.

### Tracking Enhancement
- Current tracker matches by species + IoU
- BioCLIP species may change between frames (brief misclassification)
- **Recommendation:** Use majority-vote species across a track's lifetime instead of per-frame species

### Weight Estimation
- Current coefficients are for 13 aquarium species
- OzFish species are wild reef fish with different allometric relationships
- **Recommendation:** Source FishBase length-weight data for V1 species; provide family defaults

### Adult/Juvenile Prediction
- Current uses simple length threshold per species
- **Recommendation:** Source FishBase Lm (length at maturity) for V1 species

### Confidence Calibration
- Fine-tuned models are typically overconfident
- **Recommendation:** Post-hoc temperature scaling on validation set

### Dataset Quality
- frame_metadata.csv appears corrupted (empty/swap file)
- Some crops may be extremely small (< 32px)
- **Recommendation:** Full quality audit in Phase 2

### Deployment
- BioCLIP 2 (ViT-L/14) is too large for OAK-D on-device inference
- **Recommendation:** Keep on host GPU; consider distillation for edge deployment in V2

**Status:** Documented — To be prioritized after V1
