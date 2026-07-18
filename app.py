"""
app.py — Fish Species Detection System (Streamlit front-end)

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from src.detector import load_model, run_inference
from src.predictor import detections_to_dataframe, summary_stats
from src.utils import resolve_model_path
from src.visualization import (
    render_annotated_image,
    render_class_counts,
    render_detection_table,
    render_metrics,
)

# BioCLIP 2 fine-grained species classifier (loaded lazily)
try:
    from src.species_classifier import SpeciesClassifierInference
    _BIOCLIP_AVAILABLE = True
except Exception:
    _BIOCLIP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fish Detection System",
    page_icon="🐠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_MODEL_PATH = resolve_model_path("models/best.pt")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    st.subheader("Model")
    model_path_str = st.text_input(
        "Weights path",
        value=str(DEFAULT_MODEL_PATH),
        help="Path to a YOLOv8 .pt file. The default model lives in models/best.pt.",
    )
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("IoU threshold (NMS)", 0.10, 0.95, 0.45, 0.05)

    st.divider()

    st.subheader("Measurement (optional)")
    cm_per_pixel = st.number_input(
        "Centimetres per pixel",
        min_value=0.0,
        value=0.0,
        step=0.001,
        format="%.4f",
        help=(
            "Set this once you have depth / camera calibration data. "
            "Leave at 0 to skip length and life-stage estimation."
        ),
    )
    adult_threshold_cm = st.number_input(
        "Adult length threshold (cm)",
        min_value=0.0,
        value=10.0,
        step=0.5,
        help="Fish whose estimated length exceeds this value are labelled 'Adult'.",
    )

    st.divider()

    # BioCLIP toggle
    if _BIOCLIP_AVAILABLE:
        st.subheader("BioCLIP 2 Classification")
        use_bioclip = st.toggle(
            "Enable species identification",
            value=False,
            help=(
                "Run BioCLIP 2 (157 species, 73.8% top-1 accuracy) on each "
                "detected fish crop for fine-grained species identification."
            ),
        )
        bioclip_top_k = st.slider("Top-K predictions", 1, 10, 3) if use_bioclip else 3
    else:
        use_bioclip = False
        bioclip_top_k = 3

    st.divider()
    st.caption("Fish Detection System · YOLOv8n · 13 species · BioCLIP 2 · 157 species")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("🐠 Fish Species Detection")
st.caption("Upload a photo and the model will identify and count fish species.")

# Model loading — cached so it only runs once per session
model_file = Path(model_path_str)
if not model_file.exists():
    st.error(
        f"**Model not found:** `{model_file}`\n\n"
        "Make sure `best.pt` is inside the `models/` directory, "
        "or update the path in the sidebar."
    )
    st.stop()

try:
    model = st.cache_resource(load_model)(str(model_file))
except Exception as exc:
    st.error(f"**Could not load model:** {exc}")
    st.stop()

# Image upload
uploaded_file = st.file_uploader(
    "Upload a fish image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Supported formats: JPG, JPEG, PNG, BMP, WebP",
)

if uploaded_file is None:
    st.info("👆 Upload an image to get started.")
    st.stop()

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
image = Image.open(uploaded_file).convert("RGB")

with st.spinner("Running detection…"):
    result = run_inference(model, image, conf=confidence, iou=iou)

df = detections_to_dataframe(result, cm_per_pixel, adult_threshold_cm)
stats = summary_stats(df)
annotated_image = result.plot()

# ── BioCLIP 2 classification (optional) ─────────────────────────────────────
bioclip_results: dict[int, list[dict]] = {}  # fish_id → predictions
if use_bioclip and _BIOCLIP_AVAILABLE and not df.empty:
    with st.spinner("Running BioCLIP species identification…"):
        try:
            clf = st.cache_resource(SpeciesClassifierInference)(top_k=bioclip_top_k)
            for _, row in df.iterrows():
                bbox = (row["x1"], row["y1"], row["x2"], row["y2"])
                preds = clf.classify_crop(image, bbox)
                bioclip_results[int(row["fish_id"])] = preds
        except Exception as exc:
            st.warning(f"BioCLIP classification failed: {exc}")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
render_metrics(stats["total"], stats["species"], stats["avg_confidence"])

st.divider()

img_col, summary_col = st.columns([2, 1], gap="large")
with img_col:
    render_annotated_image(annotated_image)
with summary_col:
    render_class_counts(df)

render_detection_table(df, cm_per_pixel)

# ── BioCLIP 2 species results ────────────────────────────────────────────────
if bioclip_results:
    st.divider()
    st.subheader("🔬 BioCLIP 2 — Fine-Grained Species Identification")
    st.caption(
        f"Model: BioCLIP 2 ViT-L/14 + LoRA · 157 species · "
        f"Val accuracy 73.8% · Top-5 accuracy 92.3%"
    )
    for fish_id, preds in bioclip_results.items():
        if not preds:
            continue
        top1 = preds[0]
        with st.expander(
            f"Fish #{fish_id} — {top1['species'].replace('_', ' ')} "
            f"({top1['confidence']:.1%} confidence)",
            expanded=True,
        ):
            import pandas as pd
            df_preds = pd.DataFrame(preds)
            df_preds["species"] = df_preds["species"].str.replace("_", " ")
            df_preds["confidence"] = df_preds["confidence"].map("{:.1%}".format)
            df_preds.columns = ["Rank", "Species", "Confidence"]
            st.dataframe(df_preds, use_container_width=True, hide_index=True)
