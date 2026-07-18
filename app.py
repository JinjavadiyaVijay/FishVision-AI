"""
app.py — FishVision-AI  (BioCLIP 2 Primary Classifier + YOLO Detection)

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.pipeline import FishPipeline
from src.utils import resolve_model_path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FishVision-AI",
    page_icon="🐠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project paths ─────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
_EVAL_JSON = (
    _ROOT / "experiments/bioclip2_full_20260716_160143/plots/evaluation_test.json"
)
_DEFAULT_YOLO = resolve_model_path("models/best.pt")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark background */
.stApp { background: #0d1117; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #161b27 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-val  { font-size: 2rem; font-weight: 700; color: #58a6ff; }
.metric-lbl  { font-size: 0.78rem; color: #8b949e; text-transform: uppercase;
               letter-spacing: 0.08em; margin-top: 0.2rem; }

/* Section headers */
.section-header {
    font-size: 1.1rem; font-weight: 600; color: #e6edf3;
    border-left: 3px solid #238636; padding-left: 0.7rem; margin: 1.2rem 0 0.8rem;
}

/* Species cards */
.species-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.15s;
}
.species-card:hover { border-color: #58a6ff; }
.species-name { font-size: 1rem; font-weight: 600; color: #e6edf3; }
.species-sci  { font-size: 0.8rem; color: #8b949e; margin-top: 0.15rem; }
.conf-bar     { height: 6px; border-radius: 3px; margin-top: 0.6rem;
                background: linear-gradient(90deg, #238636, #58a6ff); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}

/* Buttons / tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: #161b22; border-radius: 8px;
    color: #8b949e; padding: 6px 16px;
    border: 1px solid #30363d;
}
.stTabs [aria-selected="true"] {
    background: #1f6feb !important; color: #fff !important;
    border-color: #1f6feb !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load evaluation results (species accuracy) ────────────────────────────────
@st.cache_data
def load_eval_data() -> dict | None:
    if _EVAL_JSON.exists():
        return json.loads(_EVAL_JSON.read_text(encoding="utf-8"))
    return None


# ── Load pipeline (cached across sessions) ───────────────────────────────────
@st.cache_resource
def get_pipeline(yolo_path: str, top_k: int) -> FishPipeline:
    p = FishPipeline(
        yolo_model_path=yolo_path,
        bioclip_enabled=True,
        bioclip_top_k=top_k,
    )
    p.preload()
    return p


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🐠 FishVision-AI")
    st.caption("BioCLIP 2 · ViT-L/14 · LoRA · 157 species")
    st.divider()

    st.markdown("### 🎛 Detection Settings")
    yolo_path = st.text_input("YOLO weights", value=str(_DEFAULT_YOLO),
                               help="YOLOv8 .pt file for fish localisation")
    conf_thresh = st.slider("YOLO confidence", 0.10, 0.90, 0.25, 0.05)
    iou_thresh  = st.slider("YOLO IoU (NMS)",  0.10, 0.90, 0.45, 0.05)

    st.divider()
    st.markdown("### 🔬 BioCLIP Settings")
    top_k = st.slider("Top-K species", 1, 10, 5)
    cm_per_px = st.number_input("cm / pixel (0 = skip)", 0.0, step=0.001,
                                 format="%.4f")
    adult_cm  = st.number_input("Adult threshold (cm)", 0.0, value=10.0, step=0.5)

    st.divider()
    eval_data = load_eval_data()
    if eval_data:
        st.markdown("### 📊 Model Performance")
        st.markdown(f"**Top-1** &nbsp; `{eval_data['top1_accuracy']:.1%}`")
        st.markdown(f"**Top-5** &nbsp; `{eval_data['top5_accuracy']:.1%}`")
        st.markdown(f"**F1** &nbsp;&nbsp;&nbsp;&nbsp; `{eval_data['f1_macro']:.4f}`")
        st.caption(f"{eval_data['total_images']:,} test images · {eval_data['num_classes']} species")


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🐠 FishVision-AI")
st.markdown(
    "**BioCLIP 2** fine-grained fish species classifier · "
    "YOLO-detected crops → 157-species identification"
)

# ── Load pipeline ─────────────────────────────────────────────────────────────
if not Path(yolo_path).exists():
    st.error(f"YOLO weights not found: `{yolo_path}`. Check models/ directory.")
    st.stop()

try:
    pipeline = get_pipeline(yolo_path, top_k)
except Exception as exc:
    st.error(f"Failed to load pipeline: {exc}")
    st.stop()

bioclip_ok = pipeline.bioclip_available
if bioclip_ok:
    st.success(
        f"✅ BioCLIP 2 loaded · {pipeline.num_bioclip_species} species · "
        f"72.6% Top-1 · 91.7% Top-5",
        icon=None,
    )
else:
    st.warning("⚠️ BioCLIP checkpoint not found — YOLO-only mode")


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_infer, tab_accuracy = st.tabs(["🔍 Identify Fish", "📊 All Species Accuracy"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_infer:
    uploaded = st.file_uploader(
        "Upload a fish image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded is None:
        st.info("👆 Upload an underwater image — BioCLIP will identify every fish detected by YOLO.")
        st.stop()

    image = Image.open(uploaded).convert("RGB")

    with st.spinner("Running YOLO + BioCLIP…"):
        result = pipeline.run(
            image,
            conf=conf_thresh,
            iou=iou_thresh,
            cm_per_pixel=cm_per_px,
            adult_threshold_cm=adult_cm,
        )

    # ── Top metrics row ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{result.total_fish}</div>
            <div class="metric-lbl">Fish Detected</div></div>""",
            unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{result.species_identified}</div>
            <div class="metric-lbl">Species ID'd</div></div>""",
            unsafe_allow_html=True)
    with c3:
        avg_conf = (
            sum(d.species_confidence for d in result.detections if d.species)
            / max(result.species_identified, 1)
        )
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{avg_conf:.0%}</div>
            <div class="metric-lbl">Avg Confidence</div></div>""",
            unsafe_allow_html=True)
    with c4:
        total_ms = result.elapsed_yolo_ms + result.elapsed_bioclip_ms
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{total_ms:.0f}ms</div>
            <div class="metric-lbl">Inference Time</div></div>""",
            unsafe_allow_html=True)

    st.markdown("")

    # ── Image + species panel ─────────────────────────────────────────────────
    if result.total_fish == 0:
        st.image(image, caption="No fish detected — try lowering YOLO confidence", use_container_width=True)
        st.stop()

    img_col, cls_col = st.columns([2, 1], gap="large")

    with img_col:
        st.markdown('<div class="section-header">Annotated Frame</div>', unsafe_allow_html=True)
        import numpy as np
        st.image(result.annotated_frame[:, :, ::-1], use_container_width=True)

    with cls_col:
        st.markdown('<div class="section-header">BioCLIP Predictions</div>', unsafe_allow_html=True)
        for det in result.detections:
            top1_sp   = det.species or det.yolo_class
            top1_conf = det.display_confidence
            bar_w = int(top1_conf * 100)
            st.markdown(f"""
            <div class="species-card">
              <div class="species-name">Fish #{det.fish_id}</div>
              <div class="species-sci">{top1_sp.replace('_', ' ')}</div>
              <div style="display:flex;align-items:center;gap:8px;margin-top:6px;">
                <div style="flex:1;background:#21262d;border-radius:3px;height:6px;">
                  <div class="conf-bar" style="width:{bar_w}%"></div>
                </div>
                <span style="color:#58a6ff;font-size:0.85rem;font-weight:600">
                  {top1_conf:.1%}
                </span>
              </div>
            </div>""", unsafe_allow_html=True)

    # ── Detailed per-fish results ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Detailed Results</div>', unsafe_allow_html=True)

    for det in result.detections:
        label = (det.species or det.yolo_class).replace("_", " ")
        conf_str = f"{det.display_confidence:.1%}"
        with st.expander(f"Fish #{det.fish_id} — {label} ({conf_str})", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**YOLO Detection**")
                st.markdown(f"- Class: `{det.yolo_class.replace('_', ' ')}`")
                st.markdown(f"- Confidence: `{det.yolo_confidence:.3f}`")
                st.markdown(f"- BBox: `({det.bbox[0]:.0f}, {det.bbox[1]:.0f}, {det.bbox[2]:.0f}, {det.bbox[3]:.0f})`")
                if det.estimated_length_cm:
                    st.markdown(f"- Est. length: `{det.estimated_length_cm} cm` ({det.life_stage})")

            with col_b:
                st.markdown("**BioCLIP Top-K Species**")
                if det.top_k_species:
                    df_k = pd.DataFrame(det.top_k_species)
                    df_k["species"] = df_k["species"].str.replace("_", " ")
                    df_k["confidence"] = df_k["confidence"].map("{:.2%}".format)
                    df_k.columns = ["Rank", "Species", "Confidence"]
                    st.dataframe(df_k, hide_index=True, use_container_width=True)
                else:
                    st.caption("BioCLIP not available")

    # ── Raw DataFrame ─────────────────────────────────────────────────────────
    with st.expander("📋 Raw detection data (CSV)"):
        df_raw = pd.DataFrame([d.as_dict() for d in result.detections])
        df_raw["species"] = df_raw["species"].fillna("").str.replace("_", " ")
        st.dataframe(df_raw, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Download CSV",
            df_raw.to_csv(index=False),
            "detections.csv",
            "text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: ALL SPECIES ACCURACY
# ══════════════════════════════════════════════════════════════════════════════
with tab_accuracy:
    st.markdown("### 📊 BioCLIP 2 — Per-Species Accuracy (Test Set)")

    eval_data = load_eval_data()
    if eval_data is None:
        st.warning("Evaluation results not found. Run `python scripts/evaluate_model.py` first.")
        st.stop()

    # ── Global metrics ─────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    global_metrics = [
        (m1, "Top-1 Acc",       f"{eval_data['top1_accuracy']:.2%}"),
        (m2, "Top-5 Acc",       f"{eval_data['top5_accuracy']:.2%}"),
        (m3, "Precision (M)",   f"{eval_data['precision_macro']:.4f}"),
        (m4, "Recall (M)",      f"{eval_data['recall_macro']:.4f}"),
        (m5, "F1 Macro",        f"{eval_data['f1_macro']:.4f}"),
        (m6, "F1 Weighted",     f"{eval_data['f1_weighted']:.4f}"),
    ]
    for col, lbl, val in global_metrics:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-val" style="font-size:1.4rem">{val}</div>
                <div class="metric-lbl">{lbl}</div></div>""",
                unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin:1rem 0;padding:0.8rem 1rem;background:#161b22;border:1px solid #21262d;
    border-radius:8px;color:#8b949e;font-size:0.85rem">
    📁 Test set · <b style="color:#e6edf3">{eval_data['total_images']:,} images</b> ·
    <b style="color:#e6edf3">{eval_data['num_classes']} species</b> ·
    <b style="color:#e6edf3">{eval_data['ms_per_image']:.1f} ms/image</b> ·
    Experiment: bioclip2_full_20260716_160143
    </div>""", unsafe_allow_html=True)

    # ── Build full species DataFrame ───────────────────────────────────────────
    pc = eval_data["per_class"]
    df_acc = pd.DataFrame(pc)
    df_acc["Species"] = df_acc["species"].str.replace("_", " ")
    df_acc["Accuracy"] = df_acc["accuracy"].map("{:.1%}".format)
    df_acc["F1"] = df_acc["f1"].map("{:.4f}".format)
    df_acc["Precision"] = df_acc["precision"].map("{:.4f}".format)
    df_acc["Recall"] = df_acc["recall"].map("{:.4f}".format)
    df_acc["Images"] = df_acc["support"]
    df_acc["acc_raw"] = df_acc["accuracy"]   # for sorting / coloring

    # ── Filters ───────────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1:
        search = st.text_input("🔍 Search species", placeholder="e.g. Lutjanus")
    with fc2:
        sort_by = st.selectbox("Sort by", ["F1 ↓", "Accuracy ↓", "Accuracy ↑", "Species A–Z", "Images ↓"])
    with fc3:
        min_acc = st.slider("Min accuracy", 0.0, 1.0, 0.0, 0.05)

    # Apply filters
    mask = df_acc["acc_raw"] >= min_acc
    if search:
        mask &= df_acc["Species"].str.contains(search, case=False)
    df_show = df_acc[mask].copy()

    sort_map = {
        "F1 ↓":       ("f1", False),
        "Accuracy ↓": ("accuracy", False),
        "Accuracy ↑": ("accuracy", True),
        "Species A–Z":("species", True),
        "Images ↓":   ("support", False),
    }
    scol, sasc = sort_map[sort_by]
    df_show = df_show.sort_values(scol, ascending=sasc)

    st.caption(f"Showing {len(df_show)} of {len(df_acc)} species")

    # ── Display table ─────────────────────────────────────────────────────────
    display_df = df_show[["Species", "Accuracy", "F1", "Precision", "Recall", "Images"]].reset_index(drop=True)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=min(60 + len(display_df) * 35, 650),
        column_config={
            "Species":   st.column_config.TextColumn("Species", width="large"),
            "Accuracy":  st.column_config.TextColumn("Top-1 Acc", width="small"),
            "F1":        st.column_config.TextColumn("F1", width="small"),
            "Precision": st.column_config.TextColumn("Precision", width="small"),
            "Recall":    st.column_config.TextColumn("Recall", width="small"),
            "Images":    st.column_config.NumberColumn("Test Images", width="small"),
        },
    )

    # ── Download ──────────────────────────────────────────────────────────────
    st.download_button(
        "⬇ Download full accuracy table (CSV)",
        df_show[["Species", "Accuracy", "F1", "Precision", "Recall", "Images"]].to_csv(index=False),
        "bioclip_species_accuracy.csv",
        "text/csv",
    )

    # ── Quick summary ─────────────────────────────────────────────────────────
    with st.expander("📈 Distribution summary"):
        bins = {"≥ 90%": 0, "70–89%": 0, "50–69%": 0, "< 50%": 0}
        for r in pc:
            a = r["accuracy"]
            if a >= 0.90:   bins["≥ 90%"] += 1
            elif a >= 0.70: bins["70–89%"] += 1
            elif a >= 0.50: bins["50–69%"] += 1
            else:           bins["< 50%"] += 1
        b1, b2, b3, b4 = st.columns(4)
        for col, (label, count) in zip([b1, b2, b3, b4], bins.items()):
            with col:
                pct = count / len(pc) * 100
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-val" style="font-size:1.5rem">{count}</div>
                    <div class="metric-lbl">{label} ({pct:.0f}%)</div></div>""",
                    unsafe_allow_html=True)
