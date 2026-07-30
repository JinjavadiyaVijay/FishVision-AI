"""
app.py — FishVision-AI  (BioCLIP 2 Primary Classifier + YOLO Detection)

Orchestrator only. Every pipeline call, cached loader, and result field is
unchanged from the original app — presentation now lives entirely in ui/
(css, svg, components, layout, inference, performance).

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.pipeline import FishPipeline
from src.utils import resolve_model_path

from ui import css, components as ui, layout, inference, performance

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FishVision-AI",
    page_icon="🐠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Project paths ─────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
_EVAL_JSON = (
    _ROOT / "experiments/bioclip2_full_20260716_160143/plots/evaluation_test.json"
)
_DEFAULT_YOLO = resolve_model_path("models/best.pt")

# ── Theme (presentation-only state) ───────────────────────────────────────────
if "fv_theme" not in st.session_state:
    st.session_state.fv_theme = "light"

st.markdown(css.get_css(st.session_state.fv_theme), unsafe_allow_html=True)
st.markdown('<div class="fv-watermark"></div>', unsafe_allow_html=True)


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
# HEADER + FLOATING SETTINGS PANEL  (replaces st.sidebar entirely)
# ─────────────────────────────────────────────────────────────────────────────
eval_data = load_eval_data()
settings = layout.render_page_header(str(_DEFAULT_YOLO), eval_data)

yolo_path = settings["yolo_path"]
conf_thresh = settings["conf_thresh"]
iou_thresh = settings["iou_thresh"]
top_k = settings["top_k"]
cm_per_px = settings["cm_per_px"]
adult_cm = settings["adult_cm"]

# ── Load pipeline ─────────────────────────────────────────────────────────────
if not Path(yolo_path).exists():
    st.markdown(
        ui.banner("warn", "alert", f"YOLO weights not found at <code>{yolo_path}</code>. Check the models/ directory."),
        unsafe_allow_html=True,
    )
    st.stop()

try:
    pipeline = get_pipeline(yolo_path, top_k)
except Exception as exc:
    st.markdown(ui.banner("warn", "alert", f"Failed to load pipeline: {exc}"), unsafe_allow_html=True)
    st.stop()

bioclip_ok = pipeline.bioclip_available
if bioclip_ok:
    st.markdown(
        ui.banner(
            "ok",
            "check",
            f"BioCLIP 2 loaded · {pipeline.num_bioclip_species} species · "
            f"72.6% Top-1 · 91.7% Top-5",
        ),
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        ui.banner("warn", "alert", "BioCLIP checkpoint not found — running in YOLO-only mode"),
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_infer, tab_accuracy = st.tabs(["Identify Fish", "Species Accuracy"])

with tab_infer:
    inference.render_inference_tab(pipeline, conf_thresh, iou_thresh, cm_per_px, adult_cm)

with tab_accuracy:
    performance.render_performance_tab(eval_data)

layout.render_footer()
