# FishVision-AI

Production-grade fish recognition and measurement pipeline.

## Architecture

FishVision-AI combines:

- YOLO detection for fish localization
- Fish crop extraction
- BioCLIP 2 / OpenCLIP species classification
- Length and weight estimation
- Adult/juvenile classification
- OAK-D Pro deployment path
- Streamlit dashboard for inspection and demos

Current Phase 3 work focuses on BioCLIP 2 classification for the processed OzFish dataset.

## Current Dataset

- Source: OzFish V1
- Processed layout: `datasets/processed/{train,val,test}/<species>/`
- Current classifier classes: 157 species after exclusions
- Metadata: `datasets/metadata/master_species_catalog.json`

The training entrypoint validates split consistency and class mapping before training starts.

## Environment

Use the existing Conda environment. Do not create a project `.venv`.

```powershell
C:\ProgramData\miniconda3\envs\ai\python.exe --version
```

Install any missing research dependencies into that environment, not into `venv/`.

## Training Gates

Full LoRA training should not start until these gates pass:

```powershell
# 1. Zero-shot BioCLIP baseline
C:\ProgramData\miniconda3\envs\ai\python.exe scripts\evaluate_zeroshot.py --split val --max-batches 50

# 2. Linear-probe smoke test
C:\ProgramData\miniconda3\envs\ai\python.exe scripts\train_bioclip.py --smoke-test --mode linear_probe

# 3. Linear-probe debug run
C:\ProgramData\miniconda3\envs\ai\python.exe scripts\train_bioclip.py --debug --mode linear_probe

# 4. LoRA smoke test
C:\ProgramData\miniconda3\envs\ai\python.exe scripts\train_bioclip.py --smoke-test --mode lora

# 5. LoRA resume verification
C:\ProgramData\miniconda3\envs\ai\python.exe scripts\train_bioclip.py --mode lora --epochs 2 --max-steps 10 --resume <experiment>\checkpoints\latest.pt
```

Only after those pass should a full LoRA run be started.

## Training Modes

```powershell
# True smoke test: 30 train batches, limited validation
C:\ProgramData\miniconda3\envs\ai\python.exe scripts\train_bioclip.py --smoke-test --mode linear_probe

# Debug run: 500 train batches
C:\ProgramData\miniconda3\envs\ai\python.exe scripts\train_bioclip.py --debug --mode linear_probe

# Full run, after validation gates
C:\ProgramData\miniconda3\envs\ai\python.exe scripts\train_bioclip.py --mode lora --epochs 30
```

Supported model modes:

- `linear_probe`: frozen BioCLIP 2 visual backbone, train classifier head only
- `lora`: frozen backbone with LoRA adapters on OpenCLIP visual modules
- `full_finetune`: all parameters trainable, not recommended for 6 GB VRAM

## Experiment Outputs

Experiments are written under `experiments/<name>_<timestamp>/`.

Current outputs include:

- `config/full_config.yaml`
- `config/class_mapping.json`
- `config/dataset_summary.json`
- `metrics/training_history.csv`
- `logs/events.out.tfevents...`
- `checkpoints/latest.pt`
- `checkpoints/best_accuracy.pt`
- `checkpoints/best_f1.pt`
- `checkpoints/training_state.pt`

Zero-shot baselines write:

- `metrics/zeroshot_metrics.json`
- `metrics/zeroshot_metrics.csv`

## Hardware Notes

Target machine:

- Windows
- RTX 3050 Laptop GPU
- 6 GB VRAM
- Conda env: `ai`

Measured DataLoader setting:

- `num_workers=2`
- `pin_memory=True`
- `persistent_workers=True`
- `prefetch_factor=2`

On this Windows laptop, `num_workers=4` was slower than `num_workers=2`.

## Repository Layout

```text
classification/
  config/          YAML config
  dataloader/      processed split loader
  models/          BioCLIP/OpenCLIP backbone and classifier head
  trainer/         training loop, checkpointing, profiling
  utilities/       device and seed helpers
scripts/
  train_bioclip.py
  evaluate_zeroshot.py
  verify_backbone.py
  test_dataset.py
docs/
experiments/
src/               existing YOLO/OAK-D application modules
```

## Status

Phase 3 is active. The project should continue incrementally: verify, profile, then train. Avoid full LoRA training until smoke, debug, resume, and zero-shot baseline gates are complete.
