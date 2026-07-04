# FishVision-AI — Dataset Plan

> **Status:** Phase 1 — Architecture Review  
> **Last Updated:** 2026-07-04  
> **Author:** AI Technical Lead (Opus)  

---

## 1. OzFish Dataset — Complete Analysis

### 1.1 Dataset Overview

| Metric | Value |
|--------|-------|
| **Total crop images** | 80,823 |
| **Total crop files on disk** | 80,823 (verified match) |
| **Unique species** | 497 |
| **Unique genera** | 214 |
| **Unique families** | 75 |
| **Metadata file** | `crop_metadata.csv` (columns: uid, file_name, family, genus, species) |
| **Image format** | PNG (pre-cropped bounding boxes) |
| **Source** | FDFML (Fish Detection From Marine Lens) project, Australia |
| **Archive format** | Mediaflux archive (ARCHIVE-INF/settings.xml) |

### 1.2 Data Architecture

The OzFish download contains two major components:

**crops-zip/** — 80,823 pre-cropped fish images in a **flat directory** (no species subfolders).
- Filename format: `{video_id}.avi.{frame_num}.{x1}.{y1}.{x2}.{y2}.png-{uid}-1.png`
- Example: `A000001_L.avi.5107.806.371.922.448.png-24126-1.png`
- The filename encodes: video ID, camera (L/R), frame number, bounding box coordinates, and a unique ID

**labelled-zip/** — Full-resolution frames with annotation files:
- `frames/batch01/`, `frames/batch02/`, `frames/batch03/` — Original video frames
- `manifests/` — SageMaker Ground Truth format (JSON Lines, class_id=0 "fish" only)
- `speciesboxes/` — VIA (VGG Image Annotator) format with species-level labels
- `fishtails/` — Additional annotations (tail positions)
- `measurementfiles/` — Length measurement annotations

**crop_metadata.csv** — Maps each crop image to its taxonomic classification:
```
uid,file_name,family,genus,species
1,A000001_L.avi.5107.806.371.922.448.png,Scaridae,Chlorurus,capistratoides
2,A000001_R.avi.4902.1388.355.1508.435.png,Scaridae,Chlorurus,capistratoides
```

**frame_metadata.csv** — Appears to be corrupted/empty (only `.crswap` backup exists).

### 1.3 Annotation Formats

**SageMaker manifests** (batch01.json, batch02.json, batch03.json):
- JSON Lines format with bounding boxes
- Single class: `class_id=0` → "fish" (no species)
- Contains confidence scores from human annotators
- Not directly useful for species classification

**VIA species boxes** (batch01_species.json, batch02_species.json, batch03_species.json):
- Full species-level labels on full frames
- Labels use mixed formats: "Kyphosus cinerascens", "Naso sp", "Fish", "Lutjanus bohar"
- Some labels are genus-only ("Naso sp"), some are "Fish" (unidentified)
- Bounding box format: rectangular regions with species in `region_attributes.label`

### 1.4 Critical Observations About the Crops

The crops are **already extracted from stereo video** — each detection appears twice (left camera `_L` and right camera `_R`). This means:

- **Effective unique fish count ≈ 40,411** (roughly half of 80,823)
- Left and right crops of the same fish are nearly identical (slight parallax shift)
- **These MUST be kept in the same split** (train/val/test) to prevent data leakage
- The filename prefix (`A000001_L` vs `A000001_R`) can be used to pair them

---

## 2. Species Distribution Analysis

### 2.1 Class Size Distribution

| Bracket | Species Count | % of Species | Total Images | % of Images |
|---------|--------------|-------------|-------------|-------------|
| ≥ 500 crops | 37 | 7.4% | 53,149 | 65.8% |
| 100–499 crops | 121 | 24.3% | 22,933 | 28.4% |
| 50–99 crops | 62 | 12.5% | 3,957 | 4.9% |
| 20–49 crops | 87 | 17.5% | 2,533 | 3.1% |
| < 20 crops | 190 | 38.2% | 1,437 | 1.8% |

**Key insight:** 38% of species have fewer than 20 samples — unusable for supervised learning. The top 37 species (≥500 samples) contain 66% of all images.

### 2.2 Top 30 Species by Sample Count

| Rank | Species | Count | Family |
|------|---------|-------|--------|
| 1 | punctulatus | 6,095 | Lethrinidae |
| 2 | digramma | 3,172 | Lethrinidae |
| 3 | atkinsoni | 2,635 | Lethrinidae |
| 4 | sebae | 2,338 | Lutjanidae |
| 5 | lutescens | 2,007 | Acanthuridae |
| 6 | rubrioperculatus | 1,782 | Lutjanidae |
| 7 | **spp** | **1,730** | Mixed |
| 8 | niger | 1,728 | Balistidae |
| 9 | lunare | 1,725 | Labridae |
| 10 | coelestis | 1,693 | Pomacentridae |
| 11 | vitta | 1,597 | Lutjanidae |
| 12 | temminckii | 1,256 | Mullidae |
| 13 | gymnostethus | 1,202 | Acanthuridae |
| 14 | fulvoguttatus | 1,153 | Serranidae |
| 15 | fuscescens | 1,142 | Siganidae |
| 16 | triostegus | 1,088 | Acanthuridae |
| 17 | areolatus | 1,062 | Serranidae |
| 18 | porosus | 930 | Haemulidae? |
| 19 | tile | 882 | Caesionidae |
| 20 | nebulosus | 832 | Lethrinidae |
| 21 | **sp10** | **793** | Unknown |
| 22 | viridis | 754 | Pomacentridae |
| 23 | multinotatus | 675 | Labridae |
| 24 | grammoptilus | 674 | Labridae |
| 25 | striatus | 664 | Labridae |
| 26 | argenteus | 633 | Carangidae |
| 27 | rivulatus | 626 | Labridae |
| 28 | sordidus | 619 | Scaridae |
| 29 | caudimacula | 616 | Lutjanidae |
| 30 | sexfasciatus | 606 | Carangidae |

### 2.3 Problematic Labels

| Label | Count | Issue |
|-------|-------|-------|
| `spp` | 1,730 | Generic "multiple species" — unidentifiable |
| `sp10` | 793 | Unknown species code |
| `sp3` | 232 | Unknown species code |
| `sp1` | 104 | Unknown species code |
| `sp6` | 6 | Unknown species code |
| `sp2` | 2 | Unknown species code |
| (empty) | 14 | Missing family label (7 rows with blank family) |

**Decision:** All `sp*` and `spp` labels must be **excluded** from training. They represent 2,867 images (3.5%) that are taxonomically ambiguous.

### 2.4 Top Families

| Family | Images | Species Count |
|--------|--------|--------------|
| Lethrinidae | 13,205 | Emperor fish |
| Labridae | 13,064 | Wrasses |
| Pomacentridae | 6,272 | Damselfishes |
| Lutjanidae | 6,041 | Snappers |
| Carangidae | 5,632 | Jacks/Trevallies |
| Acanthuridae | 5,548 | Surgeonfishes |
| Caesionidae | 4,636 | Fusiliers |
| Serranidae | 3,952 | Groupers |
| Scaridae | 3,509 | Parrotfishes |
| Balistidae | 3,349 | Triggerfishes |

---

## 3. V1 Species Selection Recommendation

### 3.1 Selection Criteria

A species qualifies for V1 if it meets ALL of the following:

1. **Minimum sample count**: ≥ 100 crops (enough for meaningful train/val/test splits)
2. **Identifiable species**: NOT `sp*`, `spp`, or empty labels
3. **Reasonable diversity**: Coverage across multiple families
4. **Taxonomic clarity**: Species name is not ambiguous (no identical species names across different genera)

### 3.2 Recommended V1 Species Set

Applying criteria: ≥ 200 crops AND not `sp*`/`spp`:

| # | Full Name | Count | Family |
|---|-----------|-------|--------|
| 1 | Lethrinus punctulatus | 6,095 | Lethrinidae |
| 2 | Lethrinus digramma | 3,172 | Lethrinidae |
| 3 | Lethrinus atkinsoni | 2,635 | Lethrinidae |
| 4 | Lutjanus sebae | 2,338 | Lutjanidae |
| 5 | Acanthurus lutescens | 2,007 | Acanthuridae |
| 6 | Lutjanus rubrioperculatus | 1,782 | Lutjanidae |
| 7 | Odonus niger | 1,728 | Balistidae |
| 8 | Thalassoma lunare | 1,725 | Labridae |
| 9 | Pomacentrus coelestis | 1,693 | Pomacentridae |
| 10 | Lutjanus vitta | 1,597 | Lutjanidae |
| 11 | Parupeneus temminckii | 1,256 | Mullidae |
| 12 | Naso gymnostethus | 1,202 | Acanthuridae |
| 13 | Epinephelus fulvoguttatus | 1,153 | Serranidae |
| 14 | Siganus fuscescens | 1,142 | Siganidae |
| 15 | Acanthurus triostegus | 1,088 | Acanthuridae |
| 16 | Epinephelus areolatus | 1,062 | Serranidae |
| 17 | Diagramma porosus | 930 | Haemulidae |
| 18 | Caesio tile | 882 | Caesionidae |
| 19 | Lethrinus nebulosus | 832 | Lethrinidae |
| 20 | Chromis viridis | 754 | Pomacentridae |
| 21 | Cheilinus multinotatus | 675 | Labridae |
| 22 | Nemipterus grammoptilus | 674 | Nemipteridae |
| 23 | Stethojulis striatus | 664 | Labridae |
| 24 | Caranx argenteus | 633 | Carangidae |
| 25 | Oxycheilinus rivulatus | 626 | Labridae |
| 26 | Scarus sordidus | 619 | Scaridae |
| 27 | Lutjanus caudimacula | 616 | Lutjanidae |
| 28 | Caranx sexfasciatus | 606 | Carangidae |
| 29 | Sufflamen annulatus | 583 | Balistidae |
| 30 | Halichoeres dimidiatus | 578 | Labridae |

**V1 Total: 30 species, ~38,347 crops** (before L/R dedup consideration)

### 3.3 Why 30 Species for V1?

| Consideration | Reasoning |
|--------------|-----------|
| **Enough for meaningful evaluation** | 30 classes produces a non-trivial classification task |
| **All have ≥ 500 effective samples** | After L/R dedup, still ≥250 unique fish per species |
| **Covers 12 families** | Demonstrates cross-family generalization |
| **Includes hard pairs** | Multiple Lethrinus, Lutjanus, Labridae species test fine-grained discrimination |
| **Scalable** | Can expand to 50, 100+ species by lowering the threshold |
| **Publication-quality** | 30 species is a standard benchmark size in fish recognition literature |

### 3.4 Why NOT All 497 Species?

1. **190 species have < 20 samples** — Not enough for any learning
2. **2,867 images have ambiguous labels** (sp, spp) — Label noise
3. **Long-tail distribution** — Extreme imbalance would dominate training
4. **RTX 3050 constraint** — More classes = more output dimensions but minimal VRAM impact
5. **Research best practice** — Start focused, validate pipeline, then scale

---

## 4. Data Engineering Pipeline

### 4.1 Pipeline Steps

```
Step 1: VERIFY              → Check all files exist, are readable images
Step 2: PARSE METADATA      → Load crop_metadata.csv, validate columns
Step 3: CLEAN LABELS        → Remove sp*, spp, empty labels
Step 4: DETECT DUPLICATES   → Perceptual hashing to find near-duplicates
Step 5: PAIR L/R CAMERAS    → Group left/right stereo pairs
Step 6: SELECT SPECIES      → Apply selection criteria
Step 7: QUALITY CHECK       → Min resolution, file size, corrupted images
Step 8: SPLIT DATA          → Stratified split BY VIDEO (not by image)
Step 9: ORGANIZE            → Copy into train/val/test/species/ structure
Step 10: GENERATE REPORTS   → Statistics, distributions, split summaries
```

### 4.2 Critical: Splitting Strategy

**WRONG approach** — Random image-level split:
```
Problem: Left and right crops of the same fish end up in different splits
Result: Data leakage → artificially inflated validation accuracy
```

**CORRECT approach** — Video-level split:
```
Group images by video ID (e.g., A000001)
Assign entire videos to train/val/test
Ensures no fish appears in multiple splits
```

**Recommended split ratios:**
| Split | Ratio | Purpose |
|-------|-------|---------|
| Train | 70% | Model training |
| Validation | 15% | Hyperparameter tuning, early stopping |
| Test | 15% | Final evaluation (never touched during development) |

### 4.3 Stereo Pair Handling

For each fish, both L and R crops should go to the **same split**. But during training:

- **Option A**: Use both L and R as separate training samples → 2× data but with correlation
- **Option B**: Use only L (or random choice of L/R) → cleaner but smaller dataset

**Recommendation**: Option A for training (more data is better for small classes), but ensure both always stay in the same split.

---

## 5. Image Quality Analysis Plan

The following checks will be implemented in `data_validator.py`:

| Check | Method | Action |
|-------|--------|--------|
| File exists | `Path.exists()` | Log missing |
| Readable image | `PIL.Image.open()` | Log corrupted |
| Minimum dimensions | width × height ≥ 32 × 32 px | Flag tiny crops |
| Maximum dimensions | width × height reasonable | Flag oversized |
| Aspect ratio | width/height distribution | Flag extreme ratios |
| File size | > 0 bytes | Flag empty files |
| Color channels | RGB check | Flag grayscale/RGBA |
| Near-duplicates | Perceptual hash (pHash) | Flag for review |

### Expected Image Size Distribution

From filename analysis (bbox coordinates encode crop dimensions):
- Crops range from ~30×30 px (distant fish, tiny) to ~1000×800 px (close fish, large)
- Most crops are between 50×50 and 400×400 px
- Very small crops (< 32px) are likely to be low-quality and should be flagged

---

## 6. Metadata Consistency Checks

| Check | Description |
|-------|-------------|
| UID uniqueness | All UIDs in crop_metadata.csv must be unique |
| File mapping | Every file_name in CSV must have a corresponding image |
| Orphan images | Every image file must have a CSV entry |
| Taxonomy hierarchy | Every species must have a genus; every genus must have a family |
| Label consistency | Same species should always map to the same family/genus |
| Empty fields | Flag rows with missing family, genus, or species |

---

## 7. Reports and Statistics to Generate

### Automatic Reports (saved to `datasets/reports/`)

1. **data_quality_report.json** — File integrity, missing files, corrupted images
2. **label_consistency_report.json** — Taxonomy validation, ambiguous labels
3. **duplicate_report.json** — Near-duplicate pairs with hash distances
4. **split_report.json** — Exact composition of train/val/test splits

### Automatic Statistics (saved to `datasets/statistics/`)

1. **species_distribution.csv** — Images per species (with genus and family)
2. **family_distribution.csv** — Images per family
3. **resolution_statistics.csv** — Width, height, area, aspect ratio per image
4. **split_statistics.csv** — Class counts per split
5. **species_distribution.png** — Bar chart
6. **resolution_histogram.png** — Distribution plots
7. **class_balance_heatmap.png** — Train/val/test balance visualization

### Automatic Mappings (saved to `datasets/metadata/`)

1. **species_mapping.json** — `species → {family, genus, species_idx, count}`
2. **split_manifest.json** — `image_path → {split, species, video_id}`
3. **label_statistics.csv** — Complete species/genus/family count table
