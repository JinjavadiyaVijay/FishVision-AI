"""
pipeline.py — Unified YOLO + BioCLIP inference pipeline.

This is the single reusable entry point for both Streamlit (app.py)
and the OAK-D live loop (oak_runner.py).

Pipeline:
    PIL Image  (or numpy BGR frame from OAK-D)
    → YOLO Detection          src/detector.py     (unchanged)
    → bbox parsing            src/predictor.py    (unchanged)
    → crop each detection
    → BioCLIP Classification  src/species_classifier.py  (unchanged)
    → FishDetection dataclass (structured result per fish)
    → PipelineResult          (all detections + annotated frame)

Usage — batch / Streamlit:
    from src.pipeline import FishPipeline

    pipeline = FishPipeline()                         # loads models once
    result   = pipeline.run(pil_image)
    for det in result.detections:
        print(det.yolo_class, det.species, det.species_confidence)

Usage — OAK-D (numpy BGR frame):
    result = pipeline.run_frame(bgr_frame, conf=0.60)

Design constraints:
    - Zero modifications to detector.py, predictor.py, species_classifier.py
    - Zero duplicate inference logic
    - Optional BioCLIP (pipeline works without it if checkpoint missing)
    - Thread-safe model reuse (models loaded once, shared across calls)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FishDetection:
    """Structured result for a single detected fish."""

    # YOLO outputs
    fish_id: int                        # 1-based index within this frame
    yolo_class: str                     # YOLO class name (e.g. "Lutjanus_gibbus")
    yolo_confidence: float              # YOLO detection confidence  [0–1]
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels

    # BioCLIP outputs (None if BioCLIP disabled or unavailable)
    species: Optional[str] = None           # Top-1 predicted species
    species_confidence: float = 0.0         # Top-1 confidence [0–1]
    top_k_species: list[dict] = field(default_factory=list)
    # Each element: {"rank": int, "species": str, "confidence": float}

    # Optional biometrics (populated by caller if depth available)
    estimated_length_cm: Optional[float] = None
    life_stage: Optional[str] = None

    # ── Convenience ──────────────────────────────────────────────────────────

    @property
    def display_species(self) -> str:
        """Best available species name (BioCLIP > YOLO), underscores → spaces."""
        name = self.species or self.yolo_class
        return name.replace("_", " ")

    @property
    def display_confidence(self) -> float:
        """Confidence for display: BioCLIP confidence if available, else YOLO."""
        return self.species_confidence if self.species else self.yolo_confidence

    def as_dict(self) -> dict:
        """Flat dict suitable for pandas / CSV logging."""
        return {
            "fish_id": self.fish_id,
            "yolo_class": self.yolo_class,
            "yolo_confidence": self.yolo_confidence,
            "x1": self.bbox[0],
            "y1": self.bbox[1],
            "x2": self.bbox[2],
            "y2": self.bbox[3],
            "species": self.species,
            "species_confidence": self.species_confidence,
            "estimated_length_cm": self.estimated_length_cm,
            "life_stage": self.life_stage,
        }


@dataclass
class PipelineResult:
    """All detections for a single image / frame."""

    detections: list[FishDetection]
    annotated_frame: Optional[np.ndarray] = None   # BGR numpy (from result.plot())
    elapsed_yolo_ms: float = 0.0
    elapsed_bioclip_ms: float = 0.0

    @property
    def total_fish(self) -> int:
        return len(self.detections)

    @property
    def species_identified(self) -> int:
        return sum(1 for d in self.detections if d.species is not None)

    def as_summary(self) -> dict:
        return {
            "total_fish": self.total_fish,
            "species_identified": self.species_identified,
            "elapsed_yolo_ms": round(self.elapsed_yolo_ms, 1),
            "elapsed_bioclip_ms": round(self.elapsed_bioclip_ms, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class FishPipeline:
    """
    Reusable YOLO + BioCLIP inference pipeline.

    Models are loaded once on construction (or lazily on first call)
    and reused across all subsequent calls.

    Parameters
    ----------
    yolo_model_path : str or Path, optional
        Path to YOLOv8 .pt weights. Auto-detected from models/ if omitted.
    bioclip_enabled : bool
        Enable BioCLIP species classification (default True).
        If the checkpoint is missing, BioCLIP silently degrades to disabled.
    bioclip_top_k : int
        Number of top-K species predictions to return per detection (default 5).
    bioclip_checkpoint_dir : Path, optional
        Explicit BioCLIP checkpoint directory. Auto-detected if omitted.
    device : str
        "auto" (default), "cuda", or "cpu".
    """

    def __init__(
        self,
        yolo_model_path: Optional[str | Path] = None,
        bioclip_enabled: bool = True,
        bioclip_top_k: int = 5,
        bioclip_checkpoint_dir: Optional[Path] = None,
        device: str = "auto",
    ) -> None:
        self._yolo_model_path = self._resolve_yolo_path(yolo_model_path)
        self._bioclip_enabled = bioclip_enabled
        self._bioclip_top_k = bioclip_top_k
        self._bioclip_checkpoint_dir = bioclip_checkpoint_dir
        self._device = device

        self._yolo = None           # loaded lazily
        self._clf = None            # loaded lazily
        self._bioclip_available = False

        logger.info(
            "FishPipeline created (yolo=%s, bioclip=%s, top_k=%d, device=%s)",
            self._yolo_model_path.name, bioclip_enabled, bioclip_top_k, device,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        image: Image.Image,
        conf: float = 0.25,
        iou: float = 0.45,
        cm_per_pixel: float = 0.0,
        adult_threshold_cm: float = 10.0,
    ) -> PipelineResult:
        """
        Run the full pipeline on a PIL image.

        Parameters
        ----------
        image : PIL.Image.Image
            RGB input image (any size).
        conf : float
            YOLO confidence threshold.
        iou : float
            YOLO IoU threshold for NMS.
        cm_per_pixel : float
            Set > 0 to populate estimated_length_cm and life_stage.
        adult_threshold_cm : float
            Length threshold for life_stage = "Adult".

        Returns
        -------
        PipelineResult
        """
        self._ensure_loaded()

        # ── Step 1: YOLO detection ────────────────────────────────────────────
        from src.detector import run_inference
        from src.predictor import detections_to_dataframe

        t0 = time.perf_counter()
        yolo_result = run_inference(self._yolo, image, conf=conf, iou=iou)
        elapsed_yolo = (time.perf_counter() - t0) * 1000

        df = detections_to_dataframe(yolo_result, cm_per_pixel, adult_threshold_cm)
        annotated = yolo_result.plot()   # BGR numpy

        if df.empty:
            return PipelineResult(
                detections=[],
                annotated_frame=annotated,
                elapsed_yolo_ms=elapsed_yolo,
                elapsed_bioclip_ms=0.0,
            )

        # ── Step 2: BioCLIP classification (per-crop) ─────────────────────────
        t1 = time.perf_counter()
        detections = self._classify_dataframe(image, df)
        elapsed_bioclip = (time.perf_counter() - t1) * 1000

        return PipelineResult(
            detections=detections,
            annotated_frame=annotated,
            elapsed_yolo_ms=elapsed_yolo,
            elapsed_bioclip_ms=elapsed_bioclip,
        )

    def run_frame(
        self,
        bgr_frame: np.ndarray,
        conf: float = 0.60,
        iou: float = 0.45,
    ) -> PipelineResult:
        """
        Run the pipeline on a raw BGR numpy frame (OAK-D / OpenCV source).

        Parameters
        ----------
        bgr_frame : np.ndarray
            BGR uint8 frame from OAK-D or cv2.VideoCapture.
        conf : float
            YOLO confidence threshold (default 0.60 for live video).
        iou : float
            YOLO IoU threshold.

        Returns
        -------
        PipelineResult
        """
        # Convert BGR → RGB PIL (detector.run_inference accepts PIL)
        rgb = bgr_frame[:, :, ::-1]  # BGR → RGB (zero-copy view)
        pil_image = Image.fromarray(rgb.astype(np.uint8))
        return self.run(pil_image, conf=conf, iou=iou)

    def preload(self) -> None:
        """
        Explicitly load both models now rather than on first call.
        Call this at startup to avoid latency on the first inference.
        """
        self._ensure_loaded()

    @property
    def bioclip_available(self) -> bool:
        """True if BioCLIP loaded successfully."""
        return self._bioclip_available

    @property
    def num_bioclip_species(self) -> int:
        """Number of species in the BioCLIP classifier (0 if not loaded)."""
        if self._clf is None:
            return 0
        return self._clf.num_classes

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_yolo_path(self, path: Optional[str | Path]) -> Path:
        """Find the YOLO weights file."""
        if path is not None:
            return Path(path)
        # Search in models/ — prefer best.pt, then yolov8n.pt
        for candidate in ("models/best.pt", "models/yolov8n.pt"):
            p = _PROJECT_ROOT / candidate
            if p.exists():
                return p
        raise FileNotFoundError(
            "No YOLO weights found in models/. "
            "Place best.pt there or pass yolo_model_path explicitly."
        )

    def _ensure_loaded(self) -> None:
        """Load YOLO and BioCLIP once; no-op on subsequent calls."""
        if self._yolo is None:
            self._load_yolo()
        if self._bioclip_enabled and self._clf is None:
            self._load_bioclip()

    def _load_yolo(self) -> None:
        from src.detector import load_model
        logger.info("Loading YOLO: %s", self._yolo_model_path)
        self._yolo = load_model(self._yolo_model_path)
        logger.info("YOLO loaded: %d classes", len(self._yolo.names))

    def _load_bioclip(self) -> None:
        try:
            from src.species_classifier import SpeciesClassifierInference
            self._clf = SpeciesClassifierInference(
                checkpoint_dir=self._bioclip_checkpoint_dir,
                device=self._device,
                top_k=self._bioclip_top_k,
            )
            self._clf._load_model()   # explicit eager load
            self._bioclip_available = True
            logger.info(
                "BioCLIP loaded: %d species, device=%s",
                self._clf.num_classes, self._clf.device,
            )
        except Exception as exc:
            logger.warning("BioCLIP unavailable (%s) — running YOLO-only mode", exc)
            self._bioclip_available = False
            self._clf = None

    def _classify_dataframe(self, image: Image.Image, df) -> list[FishDetection]:
        """Build FishDetection list, adding BioCLIP results if available."""
        detections: list[FishDetection] = []

        for _, row in df.iterrows():
            bbox = (row["x1"], row["y1"], row["x2"], row["y2"])

            # BioCLIP classification
            top_k: list[dict] = []
            species: Optional[str] = None
            species_conf: float = 0.0

            if self._bioclip_available and self._clf is not None:
                try:
                    top_k = self._clf.classify_crop(image, bbox)
                    if top_k:
                        species = top_k[0]["species"]
                        species_conf = top_k[0]["confidence"]
                except Exception as exc:
                    logger.warning("BioCLIP classify_crop failed for fish_id=%s: %s",
                                   row["fish_id"], exc)

            # life_stage / length (from predictor, already in df)
            length_cm = (
                float(row["estimated_length_cm"])
                if "estimated_length_cm" in row and not np.isnan(row.get("estimated_length_cm", float("nan")))
                else None
            )
            life_stage = row.get("life_stage") or None

            detections.append(FishDetection(
                fish_id=int(row["fish_id"]),
                yolo_class=str(row["class"]),
                yolo_confidence=float(row["confidence"]),
                bbox=bbox,
                species=species,
                species_confidence=species_conf,
                top_k_species=top_k,
                estimated_length_cm=length_cm,
                life_stage=life_stage,
            ))

        return detections
