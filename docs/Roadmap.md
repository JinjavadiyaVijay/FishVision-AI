# FishVision-AI — Roadmap

> **Status:** Phase 1 — Architecture Review  
> **Last Updated:** 2026-07-04  
> **Author:** AI Technical Lead (Opus)  

---

## Implementation Phases

### Phase 1: Architecture Review ← CURRENT
- [x] Deep analysis of existing project structure
- [x] Complete OzFish dataset inspection
- [x] BioCLIP 2 architecture research
- [x] Hardware constraint analysis (RTX 3050 6GB)
- [x] Design documentation suite
- [ ] **User approval of architecture decisions**

### Phase 2: Dataset Analysis & Engineering
- [ ] Build `classification/dataset/ozfish_parser.py`
- [ ] Build `classification/dataset/data_validator.py`
- [ ] Build `classification/preprocessing/deduplicator.py`
- [ ] Build `classification/preprocessing/image_processor.py`
- [ ] Run full dataset quality report
- [ ] Generate species distribution statistics and plots
- [ ] Identify and log corrupted/unreadable images
- [ ] Validate taxonomy consistency
- [ ] Verify L/R stereo pair mapping
- [ ] **User review of data quality report**

### Phase 3: Dataset Organization & Splitting
- [ ] Build `classification/dataset/species_selector.py`
- [ ] Build `classification/dataset/data_splitter.py`
- [ ] Create `datasets/` directory structure
- [ ] Select V1 species set (30 species, ≥200 per species)
- [ ] Video-level stratified splitting (70/15/15)
- [ ] Organize into `train/val/test/species/` structure
- [ ] Generate split reports and balance statistics
- [ ] Create species mapping JSON
- [ ] **User approval of V1 species set and splits**

### Phase 4: BioCLIP Training Pipeline
- [ ] Build `classification/config/` with YAML configs
- [ ] Build `classification/augmentation/fish_augmentation.py`
- [ ] Build `classification/dataloader/bioclip_dataset.py`
- [ ] Build `classification/trainer/bioclip_trainer.py`
- [ ] Build `classification/trainer/callbacks.py`
- [ ] Build `classification/metrics/classification_metrics.py`
- [ ] Set up TensorBoard logging
- [ ] Run training with LoRA on RTX 3050
- [ ] Monitor for VRAM issues
- [ ] **User review of training curves and initial metrics**

### Phase 5: Evaluation & Inference Pipeline
- [ ] Build `classification/evaluator/evaluator.py`
- [ ] Build `classification/inference/classifier.py`
- [ ] Build `classification/inference/batch_classifier.py`
- [ ] Build `classification/visualization/plots.py`
- [ ] Run comprehensive evaluation (confusion matrix, per-class metrics)
- [ ] Temperature scaling calibration
- [ ] Inference latency benchmarking
- [ ] **User review of evaluation results**

### Phase 6: Integration
- [ ] Modify `oak_runner.py` (add BioCLIP inference step)
- [ ] Modify `app.py` (add BioCLIP to Streamlit dashboard)
- [ ] Update `predictor.py` (species_override parameter)
- [ ] Extend `biometrics.py` (OzFish species)
- [ ] Add backward compatibility (graceful fallback)
- [ ] Update `requirements.txt`
- [ ] **User integration testing on OAK-D**

### Phase 7: Testing
- [ ] Unit tests for data pipeline
- [ ] Unit tests for classifier
- [ ] Integration tests (YOLO + BioCLIP + tracker)
- [ ] End-to-end test with sample video
- [ ] Edge case testing (no fish, many fish, tiny fish, occluded fish)
- [ ] **User acceptance testing**

### Phase 8: Optimization & Documentation
- [ ] Profile inference latency
- [ ] Optimize batch sizing
- [ ] Consider `torch.compile()` for speedup
- [ ] Update README.md
- [ ] Add BioCLIP setup instructions
- [ ] Add model download script
- [ ] Final `.gitignore` update
- [ ] **Release-ready review**

---

## Milestone Targets

| Milestone | Target | Success Criteria |
|-----------|--------|-----------------|
| Data pipeline complete | End of Phase 3 | Clean dataset, verified splits, no data leakage |
| First training run | End of Phase 4 | Model trains without OOM, loss decreases |
| V1 model ready | End of Phase 5 | Top-1 accuracy > 80% on test set |
| Integrated pipeline | End of Phase 6 | OAK-D shows BioCLIP species names |
| Production ready | End of Phase 8 | All tests pass, docs complete |

---

## Future Enhancements (Post-V1)

| Enhancement | Description | Priority |
|-------------|-------------|----------|
| **More species** | Expand from 30 → 100+ species | High |
| **Single-class YOLO** | Retrain YOLO as pure fish detector | Medium |
| **Confidence display** | Show BioCLIP confidence in UI | High |
| **OzFish full frames** | Use full-frame annotations for additional training | Medium |
| **Active learning** | Flag low-confidence detections for human review | Medium |
| **Model distillation** | Distill LoRA model into smaller architecture | Low |
| **Edge deployment** | Convert BioCLIP to ONNX/TensorRT for OAK-D on-device | Low |
| **Multi-dataset** | Train on OzFish + FishNet + iNaturalist | High |
