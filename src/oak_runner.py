# src/oak_runner.py — FishVision-AI OAK-D Pro live detection (depthai v3.6.x)
#
# Run from project root:
#   python -m src.oak_runner
#
# Press Q or Esc to quit.  Detections are auto-saved to logs/oak_detections.csv
# Captured frames are saved to logs/captures/
#
# False-positive prevention (3-layer defence):
#   1. Veto Gate     — COCO model blocks person / object overlaps
#   2. Size Filter   — species-specific max length rejects impossible sizes
#   3. Bbox Filter   — min area, max aspect ratio, max frame coverage

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── project imports ────────────────────────────────────────────────────────────
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.biometrics import MAX_LENGTH_CM, classify_maturity, predict_weight
from src.oak_detector import (
    IMAGE_WIDTH_PX,
    build_pipeline,
    connect_device,
    model,
    pixel_to_real_mm,
)
from src.tracker import FishTracker
from src.veto_gate import VetoGate
from src.pipeline import FishPipeline

# ── constants ──────────────────────────────────────────────────────────────────
MIN_CONF          = 0.60          # confidence threshold
MIN_DEPTH_MM      = 100           # < 10 cm → skip (sensor noise)
MAX_DEPTH_MM      = 5_000         # > 5 m  → skip (unreliable range)
MAX_FRAME_COVER   = 0.40          # bbox covering >40% of frame → reject
LOG_DIR           = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE          = LOG_DIR / "oak_detections.csv"
CAPTURES_DIR      = LOG_DIR / "captures"
CSV_HEADER        = ["timestamp", "track_id", "species", "confidence",
                     "length_cm", "width_cm", "depth_mm",
                     "weight_g", "maturity",
                     "x1", "y1", "x2", "y2", "frame_file"]

# ── tracker tuning ─────────────────────────────────────────────────────────────
TRACKER_KWARGS = dict(
    iou_threshold=0.25,
    max_distance_px=120,
    max_age_s=3.0,
    min_bbox_area_px=1500,
    max_aspect_ratio=5.0,
)

# ── colour palette per class (BGR) ────────────────────────────────────────────
_PALETTE = [
    (0, 200, 100), (255, 100,   0), (0, 100, 255), (200, 200,   0),
    (180,   0, 180), (0, 220, 220), (255, 140,   0), (50, 205,  50),
    (0, 165, 255), (138,  43, 226), (255,  20, 147), (64, 224, 208),
    (255, 215,   0),
]


def _box_colour(cls_idx: int):
    return _PALETTE[cls_idx % len(_PALETTE)]


# ── depth sampling ─────────────────────────────────────────────────────────────
def get_depth_at_box(depth_frame: np.ndarray,
                     x1: int, y1: int, x2: int, y2: int) -> float | None:
    """Return median depth (mm) sampled from central 30 % of the bounding box."""
    cx1 = int(x1 + (x2 - x1) * 0.35)
    cx2 = int(x1 + (x2 - x1) * 0.65)
    cy1 = int(y1 + (y2 - y1) * 0.35)
    cy2 = int(y1 + (y2 - y1) * 0.65)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    roi   = depth_frame[cy1:cy2, cx1:cx2]
    valid = roi[(roi > MIN_DEPTH_MM) & (roi < MAX_DEPTH_MM)]
    return float(np.median(valid)) if len(valid) > 0 else None


# ── annotation ────────────────────────────────────────────────────────────────
def _draw_detection(frame, x1, y1, x2, y2, colour, lines, track_id=None):
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness  = 2
    line_h     = 18

    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    sq = 8
    for px, py in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        cv2.rectangle(frame,
                      (px - sq // 2, py - sq // 2),
                      (px + sq // 2, py + sq // 2),
                      colour, -1)

    # Show track ID in bottom-right corner of bbox
    if track_id is not None:
        id_text = f"#{track_id}"
        (tw, th), _ = cv2.getTextSize(id_text, font, 0.45, 1)
        cv2.rectangle(frame,
                      (x2 - tw - 6, y2 - th - 6),
                      (x2, y2), colour, -1)
        cv2.putText(frame, id_text, (x2 - tw - 3, y2 - 4),
                    font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    for i, line in enumerate(lines):
        y_pos = y1 - 10 - (len(lines) - 1 - i) * line_h
        if y_pos < 15:
            y_pos = y1 + 15 + i * line_h
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        cv2.rectangle(frame,
                      (x1 - 1, y_pos - th - 3),
                      (x1 + tw + 2, y_pos + 3),
                      (0, 0, 0), -1)
        cv2.putText(frame, line, (x1, y_pos),
                    font, font_scale, colour, thickness, cv2.LINE_AA)


def _draw_veto_overlay(frame, veto_boxes):
    """Draw translucent red overlay on vetoed regions (persons, objects)."""
    for (x1, y1, x2, y2, cls_name) in veto_boxes:
        # Semi-transparent red rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        # Red border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 1)
        # Label
        cv2.putText(frame, f"BLOCKED: {cls_name}",
                    (x1 + 4, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 220), 1)


def _draw_hud(frame, fps: float, n_fish_now: int, total_unique: int,
              n_vetoed: int):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (400, 60), (0, 0, 0), -1)
    cv2.putText(frame, "FishVision-AI  OAK-D Pro",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 120), 2)
    info = f"FPS:{fps:4.1f}  Now:{n_fish_now}  Total:{total_unique}"
    if n_vetoed:
        info += f"  Blocked:{n_vetoed}"
    cv2.putText(frame, info,
                (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, "Press Q to quit",
                (w - 160, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (120, 120, 120), 1)


# ── frame capture ─────────────────────────────────────────────────────────────
def _save_frame(frame: np.ndarray, track_id: int, species: str, ts: str) -> str:
    """Save a frame capture for a newly detected fish. Returns the filename."""
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    safe_ts = ts.replace(":", "-").replace("T", "_")
    filename = f"fish_{track_id:04d}_{species}_{safe_ts}.jpg"
    filepath = CAPTURES_DIR / filename
    cv2.imwrite(str(filepath), frame)
    return filename


# ── CSV logger ────────────────────────────────────────────────────────────────
def _open_csv():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    already_exists = LOG_FILE.exists()
    fh = LOG_FILE.open("a", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    if not already_exists:
        writer.writerow(CSV_HEADER)
    return fh, writer


# ── main ──────────────────────────────────────────────────────────────────────
def run():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    csv_fh, csv_writer = _open_csv()

    # Initialise tracker + veto gate
    tracker = FishTracker(**TRACKER_KWARGS)
    veto = VetoGate()

    print("=" * 60)
    print("  FishVision-AI  —  OAK-D Pro Live Detection")
    print("=" * 60)
    print(f"  Classes        : {len(model.names)}")
    print(f"  Min conf       : {MIN_CONF:.0%}")
    print(f"  Max frame cover: {MAX_FRAME_COVER:.0%}")
    print(f"  Depth range    : {MIN_DEPTH_MM}–{MAX_DEPTH_MM} mm")
    print(f"  CSV log        : {LOG_FILE}")
    print(f"  Captures       : {CAPTURES_DIR}")
    print(f"  Tracker        : IoU≥{TRACKER_KWARGS['iou_threshold']}, "
          f"min_area={TRACKER_KWARGS['min_bbox_area_px']}px², "
          f"max_aspect={TRACKER_KWARGS['max_aspect_ratio']}")
    print("  Press Q or Esc to stop.")
    print("=" * 60)

    # ── BioCLIP pipeline (optional — degrades gracefully if unavailable) ──────
    _fish_pipeline = FishPipeline(
        bioclip_enabled=True,
        bioclip_top_k=3,
    )
    _fish_pipeline.preload(yolo=True, bioclip=True)   # eager load; avoids latency on first frame
    if _fish_pipeline.bioclip_available:
        print(f"  BioCLIP        : {_fish_pipeline.num_bioclip_species} species")
    else:
        print("  BioCLIP        : unavailable (YOLO-only mode)")
    print("=" * 60)

    # ── Step 1: Connect device ────────────────────────────────────────────────────
    print("\n  Scanning for OAK-D Pro ...")
    try:
        device = connect_device()
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        csv_fh.close()
        sys.exit(1)

    # ── Step 2: Build pipeline on device ──────────────────────────────────────
    print("  Building pipeline ...")
    try:
        pipeline, q_rgb, q_depth = build_pipeline(device)
    except Exception as e:
        print(f"\n[ERROR] Pipeline build failed: {e}")
        device.close()
        csv_fh.close()
        sys.exit(1)

    print("  Pipeline running. Detection loop started.\n")

    fps_timer   = time.perf_counter()
    frame_count = 0
    fps         = 0.0

    try:
        while True:
            # ── grab frames ───────────────────────────────────────────────────
            in_rgb   = q_rgb.get()
            in_depth = q_depth.get()

            frame       = in_rgb.getCvFrame()
            depth_frame = in_depth.getFrame()    # uint16, mm

            # ensure depth map matches RGB frame size
            fh, fw = frame.shape[:2]
            dh, dw = depth_frame.shape[:2]
            if (dh, dw) != (fh, fw):
                depth_frame = cv2.resize(
                    depth_frame, (fw, fh), interpolation=cv2.INTER_NEAREST
                )

            frame_area = fh * fw

            # ── LAYER 1: Veto Gate — detect persons & objects ─────────────────
            veto_boxes = veto.detect_veto_objects(frame)

            # ── YOLO fish inference ───────────────────────────────────────────
            results = model(frame, imgsz=640, verbose=False)[0]
            ts      = datetime.now().isoformat(timespec="seconds")

            # ── BioCLIP: classify the whole frame in one pipeline call ─────────
            # run_frame() reuses the YOLO result implicitly through PIL conversion;
            # we only use the species labels here, not the second YOLO pass,
            # because the OAK-D YOLO (oak_detector.model) runs on-device.
            # So we call classify_crop directly per bbox after building raw_detections.
            # Species lookup dict: (x1,y1,x2,y2) → top-k list
            _frame_rgb = frame[:, :, ::-1]  # BGR→RGB view
            _pil_frame = None              # lazy PIL conversion

            def _bioclip_for_box(x1, y1, x2, y2):
                """Return BioCLIP top-1 (species, conf) or (None, 0.0)."""
                nonlocal _pil_frame
                if not _fish_pipeline.bioclip_available:
                    return None, 0.0
                try:
                    if _pil_frame is None:
                        from PIL import Image as _PILImage
                        _pil_frame = _PILImage.fromarray(_frame_rgb.astype(np.uint8))
                    preds = _fish_pipeline._clf.classify_crop(_pil_frame, (x1, y1, x2, y2))
                    if preds:
                        return preds[0]["species"], preds[0]["confidence"]
                except Exception:
                    pass
                return None, 0.0

            # ── Build detection list (with 3-layer filtering) ─────────────────
            raw_detections: list[dict] = []
            n_vetoed = 0

            for box in results.boxes:
                conf = float(box.conf[0])
                if conf < MIN_CONF:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_idx = int(box.cls[0])
                species = model.names[cls_idx]

                # ── LAYER 3a: Bbox shape sanity ───────────────────────────────
                if not tracker.is_valid_detection(x1, y1, x2, y2):
                    continue

                # ── LAYER 3b: Frame coverage — reject if >40% of frame ───────
                box_area = (x2 - x1) * (y2 - y1)
                if box_area / frame_area > MAX_FRAME_COVER:
                    n_vetoed += 1
                    continue

                # ── LAYER 1: Veto Gate — check overlap with person/objects ────
                veto_reason = veto.is_vetoed((x1, y1, x2, y2), veto_boxes)
                if veto_reason:
                    n_vetoed += 1
                    continue

                # ── depth ─────────────────────────────────────────────────────
                Z = get_depth_at_box(depth_frame, x1, y1, x2, y2)
                length_cm = width_cm = weight_g = None
                maturity  = "—"

                if Z is not None:
                    length_mm = pixel_to_real_mm(x2 - x1, Z, image_width_px=fw)
                    width_mm  = pixel_to_real_mm(y2 - y1, Z, image_width_px=fw)
                    length_cm = round(length_mm / 10, 1)
                    width_cm  = round(width_mm  / 10, 1)

                    # ── LAYER 2: Size filter — reject impossible sizes ────────
                    max_len = MAX_LENGTH_CM.get(species, 80)
                    if length_cm > max_len:
                        n_vetoed += 1
                        continue

                    weight_g  = predict_weight(species, length_cm)
                    maturity  = classify_maturity(species, length_cm)

                raw_detections.append({
                    "species": species,
                    "conf": conf,
                    "cls_idx": cls_idx,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "length_cm": length_cm,
                    "width_cm": width_cm,
                    "depth_mm": Z,
                    "weight_g": weight_g,
                    "maturity": maturity,
                    # BioCLIP fine-grained species (added non-destructively)
                    "bioclip_species": None,
                    "bioclip_conf": 0.0,
                })

            # ── BioCLIP classification (batched after filtering) ──────────────
            for det in raw_detections:
                bc_species, bc_conf = _bioclip_for_box(
                    det["x1"], det["y1"], det["x2"], det["y2"]
                )
                det["bioclip_species"] = bc_species
                det["bioclip_conf"]    = bc_conf

            # ── Update tracker ────────────────────────────────────────────────
            tracked = tracker.update(raw_detections)
            n_fish_now = len(tracked)

            # Draw veto overlay (translucent red) on blocked regions
            if veto_boxes:
                _draw_veto_overlay(frame, veto_boxes)

            for trk, is_new in tracked:
                x1, y1, x2, y2 = trk.bbox
                det = next(
                    (d for d in raw_detections
                     if d["x1"] == x1 and d["y1"] == y1),
                    None,
                )
                cls_idx = det["cls_idx"] if det else 0
                colour  = _box_colour(cls_idx)

                length_cm = trk.length_cm
                width_cm  = trk.width_cm
                Z         = trk.depth_mm
                weight_g  = trk.weight_g
                maturity  = trk.maturity

                depth_label = f"D:{Z/10:.0f}cm" if Z else "no depth"

                # ── annotation ────────────────────────────────────────────────
                if length_cm is not None:
                    lines = [
                        f"{trk.species}  {trk.best_conf:.0%}",
                        f"L:{length_cm}cm  W:{width_cm}cm  {depth_label}",
                        f"{weight_g}g  {maturity}" if weight_g else maturity,
                    ]
                else:
                    # Include BioCLIP species in the OSD if available
                    bc_sp   = det["bioclip_species"] if det else None
                    bc_conf = det["bioclip_conf"]    if det else 0.0
                    bc_line = (f"BioCLIP: {bc_sp.replace('_',' ')}  {bc_conf:.0%}"
                               if bc_sp else "")
                    lines = [
                        f"{trk.species}  {trk.best_conf:.0%}",
                        depth_label,
                    ]
                    if bc_line:
                        lines.append(bc_line)

                _draw_detection(frame, x1, y1, x2, y2, colour, lines,
                                track_id=trk.track_id)

                # ── Log + save frame ONLY for NEW fish ────────────────────────
                if is_new:
                    frame_file = _save_frame(frame, trk.track_id, trk.species, ts)
                    csv_writer.writerow([
                        ts, trk.track_id, trk.species, f"{trk.best_conf:.3f}",
                        length_cm, width_cm, Z,
                        weight_g, maturity,
                        x1, y1, x2, y2,
                        frame_file,
                    ])
                    print(f"  [NEW] #{trk.track_id}  {trk.species}  "
                          f"conf={trk.best_conf:.0%}  "
                          f"L={length_cm}cm  → {frame_file}")

            csv_fh.flush()

            # ── FPS ───────────────────────────────────────────────────────────
            frame_count += 1
            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 1.0:
                fps        = frame_count / elapsed
                frame_count = 0
                fps_timer   = time.perf_counter()

            _draw_hud(frame, fps, n_fish_now, tracker.total_unique, n_vetoed)
            cv2.imshow("FishVision-AI  |  OAK-D Pro", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        csv_fh.close()
        cv2.destroyAllWindows()
        device.close()
        print(f"\n[INFO] Unique fish detected : {tracker.total_unique}")
        print(f"[INFO] Detections saved to  : {LOG_FILE}")
        print(f"[INFO] Frames captured in   : {CAPTURES_DIR}")
        print("[INFO] Done.")


if __name__ == "__main__":
    run()