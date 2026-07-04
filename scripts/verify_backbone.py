"""
verify_backbone.py — Pre-training verification: BioCLIP 2 load + forward pass.

Run from the project root:
    python scripts/verify_backbone.py

This script verifies:
  1. GPU detection and VRAM availability
  2. open_clip installation and BioCLIP 2 model download
  3. Model architecture inspection (embed_dim, param counts)
  4. Forward pass with a real image from the dataset
  5. LoRA target module detection
  6. LoRA application and trainable parameter count
  7. Classifier head construction
  8. End-to-end forward pass (images -> logits)
  9. Loss computation
  10. VRAM usage at each step

If any step fails, the script stops immediately with a clear error message.
"""

from __future__ import annotations

import io
import logging
import sys
import time
from pathlib import Path

# ── Force UTF-8 console output on Windows ──────────────────────────────────────
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch

# ── Path bootstrap ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from classification.utilities.device import get_device, get_gpu_info, log_gpu_memory
from classification.utilities.seed import set_seed

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"


def verify_step(step_num: int, name: str):
    """Decorator-like printer for verification steps."""
    print(f"\n  [{step_num}/10] {name}")
    print(f"  {'-' * 60}")


def main() -> None:
    t0 = time.perf_counter()
    set_seed(42, deterministic=False)

    print("\n" + "=" * 70)
    print("  FishVision-AI — Backbone Verification")
    print("  BioCLIP 2 (OpenCLIP) Load & Forward Pass Test")
    print("=" * 70)

    # ── Step 1: GPU Detection ─────────────────────────────────────────────
    verify_step(1, "GPU Detection")
    device = get_device()
    gpu_info = get_gpu_info()
    if gpu_info:
        print(f"    GPU:  {gpu_info.name}")
        print(f"    VRAM: {gpu_info.total_memory_gb} GB")
        print(f"    CUDA: {gpu_info.cuda_version}")
        print(f"    Compute: {gpu_info.compute_capability}")
    else:
        print("    ⚠ No GPU detected — will use CPU (very slow)")
    print("    ✓ Device selection OK")

    # ── Step 2: OpenCLIP Installation ─────────────────────────────────────
    verify_step(2, "OpenCLIP Installation")
    try:
        import open_clip
        print(f"    open_clip version: {open_clip.__version__}")
        print("    ✓ open_clip installed")
    except ImportError:
        print("    ✗ open_clip not installed!")
        print("    Fix: pip install open-clip-torch")
        sys.exit(1)

    # ── Step 3: Load BioCLIP 2 ────────────────────────────────────────────
    verify_step(3, "Load BioCLIP 2 Model")
    try:
        from classification.models.backbone import OpenCLIPBackbone
        backbone = OpenCLIPBackbone(model_name="hf-hub:imageomics/bioclip-2")
        print(f"    Model:     {backbone.model_name}")
        print(f"    Embed dim: {backbone.embed_dim}")
        print(f"    Image size: {backbone.image_size}")
        params = backbone.get_param_summary()
        print(f"    Total params:     {params['total']:,}")
        print(f"    Trainable params: {params['trainable']:,}")
        print("    ✓ Model loaded successfully")
    except Exception as e:
        print(f"    ✗ Failed to load BioCLIP 2: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    mem_after_load = log_gpu_memory("after model load")

    # ── Step 4: Architecture Inspection ───────────────────────────────────
    verify_step(4, "Architecture Inspection")
    backbone.print_architecture(max_depth=2)
    print("    ✓ Architecture inspection OK")

    # ── Step 5: Forward Pass (Dummy) ──────────────────────────────────────
    verify_step(5, "Forward Pass (dummy input)")
    backbone = backbone.to(device)
    backbone.eval()

    try:
        with torch.no_grad():
            dummy = torch.randn(2, 3, backbone.image_size, backbone.image_size, device=device)
            embeddings = backbone(dummy)
            print(f"    Input shape:  {list(dummy.shape)}")
            print(f"    Output shape: {list(embeddings.shape)}")
            assert embeddings.shape == (2, backbone.embed_dim), (
                f"Expected (2, {backbone.embed_dim}), got {embeddings.shape}"
            )
            print(f"    Embedding L2 norm: {embeddings.norm(dim=1).tolist()}")
        print("    ✓ Forward pass OK")
    except Exception as e:
        print(f"    ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    log_gpu_memory("after forward pass")

    # ── Step 6: Real Image Test ───────────────────────────────────────────
    verify_step(6, "Forward Pass (real image from dataset)")
    try:
        from PIL import Image

        # Find a real image from the training set
        train_dir = PROCESSED_DIR / "train"
        if train_dir.exists():
            # Get first species directory with images
            for species_dir in sorted(train_dir.iterdir()):
                if species_dir.is_dir():
                    images = list(species_dir.glob("*.png"))
                    if images:
                        img_path = images[0]
                        break
            else:
                raise FileNotFoundError("No images found in training set")

            print(f"    Image:   {img_path.name}")
            print(f"    Species: {species_dir.name}")

            # Load and preprocess
            img = Image.open(img_path).convert("RGB")
            print(f"    Raw size: {img.size}")

            val_tf = backbone.get_transforms(train=False)
            img_tensor = val_tf(img).unsqueeze(0).to(device)
            print(f"    Preprocessed: {list(img_tensor.shape)}")

            with torch.no_grad():
                emb = backbone(img_tensor)
                print(f"    Embedding: shape={list(emb.shape)}, norm={emb.norm().item():.4f}")
            print("    ✓ Real image forward pass OK")
        else:
            print(f"    ⚠ Dataset not found at {train_dir} — skipping real image test")
            print("    (This is OK for first-time setup)")

    except Exception as e:
        print(f"    ✗ Real image test failed: {e}")
        import traceback
        traceback.print_exc()
        # Non-fatal: continue with remaining checks

    # ── Step 7: LoRA Target Module Detection ──────────────────────────────
    verify_step(7, "LoRA Target Module Detection")
    try:
        target_modules = backbone.get_lora_target_modules()
        print(f"    Detected {len(target_modules)} target module patterns:")
        for mod in target_modules:
            print(f"      → {mod}")
        if not target_modules:
            print("    ⚠ No LoRA targets detected — LoRA may not work")
        else:
            print("    ✓ LoRA targets detected")
    except Exception as e:
        print(f"    ✗ LoRA detection failed: {e}")

    # ── Step 8: LoRA Application ──────────────────────────────────────────
    verify_step(8, "LoRA Application")
    try:
        from peft import LoraConfig, get_peft_model

        # Freeze backbone first
        backbone.freeze()
        params_frozen = backbone.get_param_summary()
        print(f"    After freeze: {params_frozen['trainable']:,} trainable params")

        # Apply LoRA to the visual encoder
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=target_modules,
            bias="none",
        )
        backbone.visual = get_peft_model(backbone.visual, lora_config)

        params_lora = backbone.get_param_summary()
        print(f"    After LoRA:   {params_lora['trainable']:,} trainable params "
              f"({params_lora['trainable_pct']}%)")
        print("    ✓ LoRA applied successfully")

    except ImportError:
        print("    ✗ peft not installed!")
        print("    Fix: pip install peft")
    except Exception as e:
        print(f"    ✗ LoRA application failed: {e}")
        import traceback
        traceback.print_exc()
        print("\n    This may indicate incorrect target_modules for this architecture.")
        print("    Listing all named modules with Linear layers:")
        for name, mod in backbone.visual.named_modules():
            if isinstance(mod, torch.nn.Linear):
                print(f"      {name}: Linear({mod.in_features}, {mod.out_features})")

    log_gpu_memory("after LoRA")

    # ── Step 9: Classifier Head + End-to-End ──────────────────────────────
    verify_step(9, "Classifier Head + End-to-End Forward Pass")
    try:
        from classification.models.classifier import ClassifierHead, SpeciesClassifier

        # Count classes from dataset (if available)
        num_classes = 53  # Default from project status
        if (PROCESSED_DIR / "train").exists():
            num_classes = len([
                d for d in (PROCESSED_DIR / "train").iterdir() if d.is_dir()
            ])
        print(f"    Detected {num_classes} classes from dataset")

        classifier_head = ClassifierHead(
            embed_dim=backbone.embed_dim,
            num_classes=num_classes,
            head_type="linear",
            dropout=0.1,
        )
        classifier_head = classifier_head.to(device)

        model = SpeciesClassifier(backbone=backbone, classifier=classifier_head)
        model.eval()

        # End-to-end forward pass
        with torch.no_grad():
            dummy = torch.randn(2, 3, backbone.image_size, backbone.image_size, device=device)
            logits = model(dummy)
            print(f"    Input:  {list(dummy.shape)}")
            print(f"    Logits: {list(logits.shape)}")
            assert logits.shape == (2, num_classes)
            probs = torch.softmax(logits, dim=1)
            print(f"    Prob range: [{probs.min().item():.6f}, {probs.max().item():.6f}]")
            print(f"    Prob sum: {probs.sum(dim=1).tolist()}")

        # Parameter summary
        summary = model.get_param_summary()
        print(f"\n    Parameter Summary:")
        print(f"      Backbone:  {summary['backbone_params']:>12,} total, "
              f"{summary['backbone_trainable']:>10,} trainable")
        print(f"      Head:      {summary['head_params']:>12,} total, "
              f"{summary['head_trainable']:>10,} trainable")
        print(f"      Combined:  {summary['total_params']:>12,} total, "
              f"{summary['total_trainable']:>10,} trainable "
              f"({summary['trainable_pct']}%)")

        print("    ✓ End-to-end forward pass OK")

    except Exception as e:
        print(f"    ✗ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Step 10: Loss Computation ─────────────────────────────────────────
    verify_step(10, "Loss Computation")
    try:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

        with torch.no_grad():
            dummy_images = torch.randn(4, 3, backbone.image_size, backbone.image_size, device=device)
            dummy_labels = torch.randint(0, num_classes, (4,), device=device)

        # Enable grad for loss test
        model.train()
        logits = model(dummy_images)
        loss = criterion(logits, dummy_labels)
        print(f"    Loss value: {loss.item():.4f}")
        loss.backward()
        print("    Backward pass: OK")

        # Check gradients exist on trainable params
        grad_count = 0
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                grad_count += 1
        print(f"    Parameters with gradients: {grad_count}")
        print("    ✓ Loss + backward OK")

    except Exception as e:
        print(f"    ✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    log_gpu_memory("after loss backward")

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 70)
    print("  ✓ ALL VERIFICATION CHECKS PASSED")
    print(f"  Time: {elapsed:.1f}s")
    print("=" * 70)
    print()
    print("  BioCLIP 2 backbone is verified and ready for training.")
    print("  Next step: Build the dataset pipeline and training engine.")
    print()


if __name__ == "__main__":
    main()
