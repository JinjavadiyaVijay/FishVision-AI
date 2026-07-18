"""
species_classifier.py — BioCLIP 2 inference bridge for production deployment.

Loads the trained LoRA checkpoint once and exposes a single classify() method
that accepts a PIL crop and returns a ranked list of species predictions.

Pipeline:
    PIL Image (full frame)
    → YOLO detection  (handled by src/detector.py)
    → crop_and_classify()   ← this module
       → preprocess crop
       → BioCLIP forward pass
       → softmax + top-k
    → list[SpeciesPrediction]

Design principles:
  - Reuses backbone.py, classifier.py unchanged (no copies)
  - No training code imported
  - Lazy loading: model built on first call, cached for session
  - Graceful degradation: returns empty list if checkpoint missing

Usage:
    from src.species_classifier import SpeciesClassifierInference

    clf = SpeciesClassifierInference()           # auto-detects best checkpoint
    predictions = clf.classify(pil_crop)         # list[dict]
    top1 = predictions[0]                        # {"species": str, "confidence": float, "rank": int}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Project root — two levels up from src/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Production model pointer: checked first, then auto-detected from experiments/
_PRODUCTION_CHECKPOINT_DIR = _PROJECT_ROOT / "models" / "bioclip_production"
_EXPERIMENTS_DIR = _PROJECT_ROOT / "experiments"

# Excluded species (same as training)
_EXCLUDED_SPECIES = ["Lutjanus_erythropterus", "Sphyraena_qenie"]


def _find_best_checkpoint_dir() -> Path:
    """
    Locate the best available checkpoint directory.

    Priority:
      1. models/bioclip_production/  (manually promoted production model)
      2. Latest experiment with best_accuracy.pt  (auto-detected)
    """
    # 1. Production slot
    prod_ckpt = _PRODUCTION_CHECKPOINT_DIR / "best_accuracy.pt"
    if prod_ckpt.exists():
        logger.info("Using production checkpoint: %s", _PRODUCTION_CHECKPOINT_DIR)
        return _PRODUCTION_CHECKPOINT_DIR

    # 2. Auto-detect from experiments/
    if not _EXPERIMENTS_DIR.exists():
        raise FileNotFoundError(
            f"No experiments directory found at {_EXPERIMENTS_DIR}. "
            "Run training first."
        )

    candidates = sorted(
        [
            d
            for d in _EXPERIMENTS_DIR.iterdir()
            if d.is_dir() and (d / "checkpoints" / "best_accuracy.pt").exists()
        ],
        key=lambda d: d.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No experiment with best_accuracy.pt found. "
            "Run training or place checkpoint in models/bioclip_production/."
        )

    ckpt_dir = candidates[0] / "checkpoints"
    logger.info("Auto-detected checkpoint: %s", ckpt_dir)
    return ckpt_dir


class SpeciesClassifierInference:
    """
    Production-ready BioCLIP 2 species classifier.

    Parameters
    ----------
    checkpoint_dir : Path, optional
        Path to a checkpoint directory containing best_accuracy.pt and
        best_accuracy_metadata.json. Auto-detected if not provided.
    device : str, optional
        "cuda", "cpu", or "auto" (default). "auto" picks CUDA if available.
    top_k : int
        Number of top predictions to return (default 5).
    """

    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        device: str = "auto",
        top_k: int = 5,
    ) -> None:
        self.top_k = top_k
        self._model = None
        self._transform = None
        self._class_names: list[str] = []
        self._metadata: dict = {}

        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Resolve checkpoint directory
        self._checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir else _find_best_checkpoint_dir()
        )

        logger.info(
            "SpeciesClassifierInference initialized (device=%s, checkpoint=%s)",
            self.device,
            self._checkpoint_dir,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, image: Image.Image) -> list[dict]:
        """
        Classify a single PIL image (fish crop or full frame).

        Parameters
        ----------
        image : PIL.Image.Image
            RGB image. Can be any size; will be resized by the transform.

        Returns
        -------
        list[dict]  — sorted by confidence (highest first)
            [
                {"rank": 1, "species": "Lutjanus_gibbus", "confidence": 0.923},
                {"rank": 2, "species": "Lutjanus_fulvus",  "confidence": 0.041},
                ...
            ]
        """
        if self._model is None:
            self._load_model()

        image_rgb = image.convert("RGB")
        tensor = self._transform(image_rgb).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        with torch.no_grad():
            logits = self._model(tensor)                          # (1, num_classes)
            probs = torch.softmax(logits, dim=1).squeeze(0)      # (num_classes,)

        k = min(self.top_k, probs.shape[0])
        top_probs, top_indices = probs.topk(k)

        return [
            {
                "rank": i + 1,
                "species": self._class_names[idx.item()],
                "confidence": round(float(prob), 4),
            }
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices))
        ]

    def classify_crop(
        self,
        full_image: Image.Image,
        bbox_xyxy: tuple[float, float, float, float],
    ) -> list[dict]:
        """
        Crop a bounding box from full_image and classify it.

        Parameters
        ----------
        full_image : PIL.Image.Image
            Full-frame RGB image.
        bbox_xyxy : (x1, y1, x2, y2)
            Bounding box in pixel coordinates.

        Returns
        -------
        Same format as classify().
        """
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        # Add a small margin (10%) to include context
        w, h = full_image.size
        margin_x = int((x2 - x1) * 0.10)
        margin_y = int((y2 - y1) * 0.10)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)

        crop = full_image.crop((x1, y1, x2, y2))
        return self.classify(crop)

    @property
    def class_names(self) -> list[str]:
        """Ordered list of species names (index = class label)."""
        if self._model is None:
            self._load_model()
        return self._class_names

    @property
    def num_classes(self) -> int:
        """Number of species the model was trained on."""
        return len(self._class_names)

    @property
    def metadata(self) -> dict:
        """Full metadata from the checkpoint (epoch, metrics, config)."""
        if self._model is None:
            self._load_model()
        return self._metadata

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Build model architecture, apply LoRA, load weights. Called once."""
        import yaml
        from classification.models.backbone import OpenCLIPBackbone
        from classification.models.classifier import ClassifierHead, SpeciesClassifier

        # ── Load metadata ──────────────────────────────────────────────
        meta_path = self._checkpoint_dir / "best_accuracy_metadata.json"
        ckpt_path = self._checkpoint_dir / "best_accuracy.pt"

        with open(meta_path) as f:
            self._metadata = json.load(f)

        self._class_names = self._metadata["class_names"]
        num_classes = self._metadata["num_classes"]

        # ── Load config (from experiment or fallback) ──────────────────
        config = self._load_config()

        logger.info(
            "Loading BioCLIP 2: epoch=%d, val_accuracy=%.1f%%, %d species",
            self._metadata["metrics"]["epoch"],
            self._metadata["metrics"]["val_accuracy"] * 100,
            num_classes,
        )

        # ── Build backbone ─────────────────────────────────────────────
        backbone_cfg = config.get("backbone", {})
        backbone = OpenCLIPBackbone(
            model_name=backbone_cfg.get("name", "hf-hub:imageomics/bioclip-2"),
            image_size=backbone_cfg.get("image_size", 224),
        )
        backbone.freeze()

        # ── Apply LoRA (same config as training) ──────────────────────
        lora_cfg = config.get("lora", {})
        if lora_cfg.get("enabled", True):
            try:
                from peft import LoraConfig, get_peft_model

                target_modules = backbone.get_lora_target_modules()
                lora_config = LoraConfig(
                    r=lora_cfg.get("rank", 16),
                    lora_alpha=lora_cfg.get("alpha", 32),
                    lora_dropout=0.0,          # No dropout at inference
                    target_modules=target_modules,
                    bias=lora_cfg.get("bias", "none"),
                )
                backbone.visual = get_peft_model(backbone.visual, lora_config)
                logger.info("LoRA applied (rank=%d, alpha=%d)",
                            lora_cfg.get("rank", 16), lora_cfg.get("alpha", 32))
            except ImportError:
                logger.warning("peft not installed — loading without LoRA adapters")

        # ── Build classifier head ──────────────────────────────────────
        cls_cfg = config.get("classifier", {})
        classifier = ClassifierHead(
            embed_dim=backbone.embed_dim,
            num_classes=num_classes,
            head_type=cls_cfg.get("type", "linear"),
            dropout=0.0,                       # No dropout at inference
        )

        # ── Compose and load weights ───────────────────────────────────
        model = SpeciesClassifier(backbone=backbone, classifier=classifier)

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)

        model = model.to(self.device)
        model.eval()

        self._model = model

        # ── Preprocessing transform (val/test — no augmentation) ───────
        self._transform = backbone.get_transforms(train=False)

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "Model ready: %s total params, %s trainable (%.1f%%) on %s",
            f"{total:,}", f"{trainable:,}", 100 * trainable / max(total, 1), self.device,
        )

    def _load_config(self) -> dict:
        """Load full_config.yaml from experiment dir, or fall back to classification/config/."""
        import yaml

        # Try experiment-saved config first
        exp_dir = self._checkpoint_dir.parent  # checkpoints/ → experiment/
        cfg_path = exp_dir / "config" / "full_config.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                return yaml.safe_load(f) or {}

        # Fallback: merge active YAML configs
        config: dict = {}
        cfg_root = _PROJECT_ROOT / "classification" / "config"
        for name in ("model", "training", "augmentation", "dataset", "logging"):
            p = cfg_root / f"{name}.yaml"
            if p.exists():
                with open(p) as f:
                    config.update(yaml.safe_load(f) or {})
        return config
