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
_FAVICON = Path(__file__).resolve().parent / "assets" / "favicon.svg"
st.set_page_config(
    page_title="FishVision-AI",
    page_icon=str(_FAVICON) if _FAVICON.exists() else "🐠",
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

st.html(css.get_css(st.session_state.fv_theme))
st.html('<div class="fv-watermark"></div>')


# ── Load evaluation results (species accuracy) ────────────────────────────────
@st.cache_data
def load_eval_data() -> dict | None:
    if _EVAL_JSON.exists():
        return json.loads(_EVAL_JSON.read_text(encoding="utf-8"))
    return None


# ── Load pipeline (cached across sessions) ───────────────────────────────────
@st.cache_resource
def get_pipeline(yolo_path: str, top_k: int, preload_yolo: bool) -> FishPipeline:
    p = FishPipeline(
        yolo_model_path=yolo_path if preload_yolo else None,
        bioclip_enabled=True,
        bioclip_top_k=top_k,
    )
    p.preload(yolo=preload_yolo, bioclip=False)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# HEADER + FLOATING SETTINGS PANEL  (replaces st.sidebar entirely)
# ─────────────────────────────────────────────────────────────────────────────
eval_data = load_eval_data()
settings = layout.render_page_header(str(_DEFAULT_YOLO), eval_data)

yolo_path = settings["yolo_path"]
use_yolo = settings["use_yolo"]
conf_thresh = settings["conf_thresh"]
iou_thresh = settings["iou_thresh"]
top_k = settings["top_k"]
cm_per_px = settings["cm_per_px"]
adult_cm = settings["adult_cm"]

# ── Load pipeline ─────────────────────────────────────────────────────────────
if use_yolo and not Path(yolo_path).exists():
    st.markdown(
        ui.banner("warn", "alert", f"YOLO weights not found at <code>{yolo_path}</code>. Check the models/ directory."),
        unsafe_allow_html=True,
    )
    st.stop()

try:
    pipeline = get_pipeline(yolo_path, top_k, preload_yolo=use_yolo)
except Exception as exc:
    st.markdown(ui.banner("warn", "alert", f"Failed to load pipeline: {exc}"), unsafe_allow_html=True)
    st.stop()

if not use_yolo and not pipeline.bioclip_checkpoint_available:
    st.markdown(
        ui.banner("warn", "alert", "BioCLIP-only mode requires a BioCLIP checkpoint. Train or place weights in models/bioclip_production/."),
        unsafe_allow_html=True,
    )
    st.stop()

bioclip_ok = pipeline.bioclip_available
bioclip_ready = pipeline.bioclip_checkpoint_available
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
elif bioclip_ready:
    st.markdown(
        ui.banner(
            "ok",
            "check",
            "BioCLIP 2 ready · loads automatically when fish are detected · "
            "157 species · 72.6% Top-1",
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
    inference.render_inference_tab(
        pipeline, conf_thresh, iou_thresh, cm_per_px, adult_cm, use_yolo=use_yolo,
    )

with tab_accuracy:
    performance.render_performance_tab(eval_data)

layout.render_footer()
