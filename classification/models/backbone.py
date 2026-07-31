"""
backbone.py — Model-agnostic backbone abstraction for vision encoders.

Supports:
  - OpenCLIP models (BioCLIP 2, SigLIP, CLIP variants)
  - timm models (DINOv2, EVA, EfficientNet, etc.)  [future]
  - HuggingFace transformers models [future]

Usage:
    backbone = create_backbone(config)
    embeddings = backbone(images)          # (B, embed_dim)
    dim = backbone.embed_dim               # e.g. 768
    train_tf = backbone.get_transforms(train=True)
    val_tf   = backbone.get_transforms(train=False)
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Optional

import torch
import torch.nn as nn
from torchvision import transforms as T

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base class
# ──────────────────────────────────────────────────────────────────────────────

class BackboneWrapper(abc.ABC, nn.Module):
    """
    Abstract wrapper around a vision backbone.

    Every backbone implementation must provide:
      - embed_dim       : output embedding dimensionality
      - image_size      : expected input resolution
      - forward()       : images → embeddings
      - get_transforms(): return appropriate preprocessing
      - freeze()        : freeze all parameters
      - get_lora_target_modules(): list of module names for LoRA
    """

    def __init__(self, model_name: str, image_size: int = 224) -> None:
        super().__init__()
        self.model_name = model_name
        self.image_size = image_size
        self.embed_dim: int = 0  # Set by subclass

    @abc.abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from images.

        Parameters
        ----------
        images : (B, 3, H, W) tensor, preprocessed.

        Returns
        -------
        (B, embed_dim) tensor of L2-normalized (or raw) embeddings.
        """
        ...

    @abc.abstractmethod
    def get_transforms(self, train: bool = False) -> T.Compose:
        """
        Return the appropriate image transforms.

        Parameters
        ----------
        train : bool
            If True, include training augmentations.
            If False, return deterministic val/test transforms.
        """
        ...

    @abc.abstractmethod
    def get_lora_target_modules(self) -> list[str]:
        """
        Return module name patterns suitable for PEFT LoRA.

        Returns
        -------
        List of module name patterns (e.g. ["attn.out_proj", "mlp.c_fc"]).
        These are passed to peft.LoraConfig(target_modules=...).
        """
        ...

    def freeze(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen: all %d parameters set to requires_grad=False",
                     sum(1 for _ in self.parameters()))

    def unfreeze(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen: all parameters set to requires_grad=True")

    def get_param_summary(self) -> dict[str, int]:
        """Return parameter count summary."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "trainable_pct": round(100 * trainable / max(total, 1), 2),
        }


# ──────────────────────────────────────────────────────────────────────────────
# OpenCLIP backbone (BioCLIP 2, SigLIP, etc.)
# ──────────────────────────────────────────────────────────────────────────────

class OpenCLIPBackbone(BackboneWrapper):
    """
    Backbone wrapper for OpenCLIP models.

    Loads only the visual encoder from an OpenCLIP model and exposes it
    as a standard PyTorch module producing (B, embed_dim) embeddings.

    Parameters
    ----------
    model_name : str
        OpenCLIP model identifier, e.g. "hf-hub:imageomics/bioclip-2".
    image_size : int
        Expected input resolution.
    pretrained : str
        Pretrained weights tag (used for non-HF models). For HF hub models,
        this is auto-detected.
    """

    def __init__(
        self,
        model_name: str = "hf-hub:imageomics/bioclip-2",
        image_size: int = 224,
        pretrained: str = "",
    ) -> None:
        super().__init__(model_name=model_name, image_size=image_size)

        import open_clip

        logger.info("Loading OpenCLIP model: %s", model_name)

        # Load full CLIP model + transforms
        clip_model, self._preprocess_train, self._preprocess_val = (
            open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        )

        # Extract visual encoder only (discard text encoder to save memory)
        self.visual = clip_model.visual
        self.embed_dim = self._detect_embed_dim(clip_model)

        # Update image_size from model config if available
        try:
            model_cfg = open_clip.get_model_config(model_name)
            if model_cfg and "vision_cfg" in model_cfg:
                self.image_size = model_cfg["vision_cfg"].get("image_size", image_size)
        except Exception:
            pass  # Use provided image_size

        # Free text encoder memory
        del clip_model

        logger.info("OpenCLIP backbone loaded: embed_dim=%d, image_size=%d",
                     self.embed_dim, self.image_size)

    def _detect_embed_dim(self, clip_model: Any) -> int:
        """Auto-detect the embedding dimension from the CLIP model."""
        # Try multiple attributes that different OpenCLIP versions use
        for attr in ("embed_dim", "output_dim", "visual.output_dim"):
            obj = clip_model
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                if isinstance(obj, int) and obj > 0:
                    return obj
            except AttributeError:
                continue

        # Fallback: run a dummy forward pass
        logger.warning("Could not detect embed_dim from model attributes; running probe")
        with torch.no_grad():
            dummy = torch.randn(1, 3, self.image_size, self.image_size)
            out = clip_model.visual(dummy)
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out.shape[-1]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract visual embeddings.

        Parameters
        ----------
        images : (B, 3, H, W) preprocessed tensor.

        Returns
        -------
        (B, embed_dim) embedding tensor.
        """
        features = self.visual(images)
        # Some OpenCLIP models return tuple (features, extra); extract features
        if isinstance(features, (tuple, list)):
            features = features[0]
        return features

    def get_transforms(self, train: bool = False) -> T.Compose:
        """
        Return OpenCLIP's native preprocessing transforms.

        These are the transforms the model was trained with, ensuring
        correct normalization and resolution.
        """
        return self._preprocess_train if train else self._preprocess_val

    def get_lora_target_modules(self) -> list[str]:
        """
        Detect LoRA-compatible nn.Linear modules in the visual encoder.

        Scans all named modules and extracts repeating name patterns.
        For OpenCLIP ViT-L/14 this discovers:
          - attn.out_proj  : attention output projection  (24 layers)
          - mlp.c_fc       : MLP first projection         (24 layers)
          - mlp.c_proj     : MLP second projection        (24 layers)

        Note: open_clip uses nn.MultiheadAttention with a fused QKV
        *parameter* (in_proj_weight), not an nn.Linear module, so LoRA
        cannot target QKV directly. Only out_proj is a Linear submodule.

        Returns
        -------
        List of unique module name patterns (e.g. ["attn.out_proj", "mlp.c_fc"]).
        """
        seen_patterns: dict[str, int] = {}  # pattern -> count

        for name, module in self.visual.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            # Extract the last two dot-separated components as the pattern
            # e.g. "transformer.resblocks.0.attn.out_proj" -> "attn.out_proj"
            #      "transformer.resblocks.0.mlp.c_fc"      -> "mlp.c_fc"
            parts = name.split(".")
            if len(parts) >= 2:
                pattern = ".".join(parts[-2:])
            else:
                pattern = name

            seen_patterns[pattern] = seen_patterns.get(pattern, 0) + 1

        target_modules = list(seen_patterns.keys())

        if target_modules:
            logger.info("Detected LoRA target modules:")
            for pattern, count in seen_patterns.items():
                logger.info("  %s (x%d)", pattern, count)
        else:
            logger.warning("No nn.Linear modules found in visual encoder")

        return target_modules

    def print_architecture(self, max_depth: int = 3) -> None:
        """Print a summary of the visual encoder architecture."""
        print(f"\n{'─' * 70}")
        print(f"  OpenCLIP Visual Encoder: {self.model_name}")
        print(f"  Embedding dim: {self.embed_dim}")
        print(f"  Image size:    {self.image_size}")
        print(f"{'─' * 70}")

        for name, module in self.visual.named_modules():
            depth = name.count(".")
            if depth <= max_depth and name:
                indent = "  " * (depth + 1)
                cls_name = module.__class__.__name__
                # Show param count for leaf modules
                params = sum(p.numel() for p in module.parameters(recurse=False))
                if params > 0:
                    print(f"{indent}{name}: {cls_name} ({params:,} params)")
                elif depth <= 1:
                    print(f"{indent}{name}: {cls_name}")


# ──────────────────────────────────────────────────────────────────────────────
# Timm backbone (DINOv2, EVA, etc.) — stub for future use
# ──────────────────────────────────────────────────────────────────────────────

class TimmBackbone(BackboneWrapper):
    """
    Backbone wrapper for timm models (DINOv2, EVA, EfficientNet, etc.).

    Not yet implemented. Placeholder for future model-agnostic expansion.
    """

    def __init__(self, model_name: str, image_size: int = 224, **kwargs: Any) -> None:
        super().__init__(model_name=model_name, image_size=image_size)
        raise NotImplementedError(
            "TimmBackbone is planned for future use. "
            "Currently only OpenCLIPBackbone is implemented."
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_transforms(self, train: bool = False) -> T.Compose:
        raise NotImplementedError

    def get_lora_target_modules(self) -> list[str]:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

_BACKBONE_REGISTRY: dict[str, type[BackboneWrapper]] = {
    "open_clip": OpenCLIPBackbone,
    "timm": TimmBackbone,
}


def create_backbone(config: dict[str, Any]) -> BackboneWrapper:
    """
    Create a backbone from configuration.

    Parameters
    ----------
    config : dict
        Must contain at minimum:
            backbone.library : str — "open_clip", "timm", or "transformers"
            backbone.name    : str — model identifier
            backbone.image_size : int — input resolution

    Returns
    -------
    BackboneWrapper subclass instance.
    """
    backbone_cfg = config.get("backbone", {})
    library = backbone_cfg.get("library", "open_clip")
    name = backbone_cfg.get("name", "hf-hub:imageomics/bioclip-2")
    image_size = backbone_cfg.get("image_size", 224)

    if library not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone library: '{library}'. "
            f"Available: {list(_BACKBONE_REGISTRY.keys())}"
        )

    cls = _BACKBONE_REGISTRY[library]
    logger.info("Creating backbone: library=%s, name=%s", library, name)

    return cls(model_name=name, image_size=image_size)
