"""
ui/inference.py — the "Identify Fish" tab.

The only pipeline-facing line in this file is `pipeline.run(...)`, called
with the exact same keyword arguments as the original app.py. Every
`result.*` / `det.*` field accessed below matches the original field names.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from PIL import Image

from . import svg, components as comp
from .css import tokens


def render_inference_tab(
    pipeline,
    conf_thresh: float,
    iou_thresh: float,
    cm_per_px: float,
    adult_cm: float,
    *,
    use_yolo: bool = True,
) -> None:
    st.markdown(
        f'<div class="fv-upload-label">{svg.icon("fish", 18)}<span>Drop your underwater image here</span></div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Upload a fish image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        video_uri = svg.get_media_uri("animation_1.mp4")
        st.markdown(
            '<div class="fv-empty-video-container">'
            '<div class="fv-empty-title">Ready for Analysis</div>'
            '<div class="fv-empty-subtitle">Upload an image to detect marine species</div>'
            f'<video class="fv-empty-video" autoplay muted loop playsinline preload="auto" src="{video_uri}">'
            '</video>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    image = Image.open(uploaded).convert("RGB")

    spinner_label = "Running YOLO + BioCLIP…" if use_yolo else "Running BioCLIP (full-frame)…"
    with st.spinner(spinner_label):
        result = pipeline.run(
            image,
            conf=conf_thresh,
            iou=iou_thresh,
            cm_per_pixel=cm_per_px,
            adult_threshold_cm=adult_cm,
            use_yolo=use_yolo,
        )

    _render_top_metrics(result, use_yolo=use_yolo)

    if result.total_fish == 0:
        no_l, no_r = st.columns([2, 1], vertical_alignment="center")
        with no_l:
            if use_yolo:
                st.image(image, caption="No fish detected — try lowering YOLO confidence", width="stretch")
                st.caption(
                    "Tip: underwater photos often need confidence **≤ 0.10**, or turn off "
                    "**Use YOLO detection** in settings for BioCLIP-only species ID."
                )
            else:
                st.image(image, caption="BioCLIP could not classify this image", width="stretch")
                st.caption("Check that the BioCLIP checkpoint is installed and the image shows a clear fish.")
        with no_r:
            st.image("assets/image.png", use_container_width=True)
        return

    _render_image_and_species(result, use_yolo=use_yolo)
    _render_timeline_and_chart(result)
    _render_detailed_results(result, use_yolo=use_yolo)
    _render_raw_data(result)


def _render_top_metrics(result, *, use_yolo: bool = True) -> None:
    avg_conf = (
        sum(d.species_confidence for d in result.detections if d.species)
        / max(result.species_identified, 1)
    )
    total_ms = result.elapsed_yolo_ms + result.elapsed_bioclip_ms
    fish_label = "Fish Detected" if use_yolo else "Frame Classified"

    st.markdown(
        comp.stat_strip([
            comp.metric_card("target", str(result.total_fish), fish_label),
            comp.metric_card("fish", str(result.species_identified), "Species ID'd"),
            comp.metric_card("gauge", f"{avg_conf:.0%}", "Avg Confidence"),
            comp.metric_card("timer", f"{total_ms:.0f}ms", "Inference Time"),
        ]),
        unsafe_allow_html=True,
    )


def _render_image_and_species(result, *, use_yolo: bool = True) -> None:
    img_col, cls_col = st.columns([2, 1], gap="large")
    frame_label = "Annotated Frame" if use_yolo else "Uploaded Frame (BioCLIP-only)"

    with img_col:
        st.markdown(comp.section_label("layers", frame_label), unsafe_allow_html=True)
        st.image(result.annotated_frame[:, :, ::-1], width="stretch")

    with cls_col:
        st.markdown(comp.section_label("fish", "BioCLIP Predictions"), unsafe_allow_html=True)
        for det in result.detections:
            top1_sp = (det.species or det.yolo_class).replace("_", " ")
            st.markdown(
                comp.species_card(det.fish_id, top1_sp, det.life_stage, det.display_confidence),
                unsafe_allow_html=True,
            )


def _render_timeline_and_chart(result) -> None:
    tl_col, cg_col = st.columns([1, 1], gap="large")

    with tl_col:
        st.markdown(comp.section_label("clock", "Detection Timeline"), unsafe_allow_html=True)
        items = [
            comp.timeline_item(
                f"Fish #{d.fish_id} — {(d.species or d.yolo_class).replace('_', ' ')}",
                f"{d.display_confidence:.0%} confidence · {d.life_stage or 'stage n/a'}",
            )
            for d in result.detections
        ]
        st.markdown(comp.timeline(items), unsafe_allow_html=True)

    with cg_col:
        st.markdown(comp.section_label("gauge", "Confidence by Fish"), unsafe_allow_html=True)
        conf_df = pd.DataFrame({
            "Fish": [f"#{d.fish_id}" for d in result.detections],
            "Confidence": [d.display_confidence for d in result.detections],
        }).set_index("Fish")
        chart_color = tokens(st.session_state.get("fv_theme", "light"))["CORAL"]
        st.bar_chart(conf_df, color=chart_color, height=220)


def _render_detailed_results(result, *, use_yolo: bool = True) -> None:
    st.markdown(comp.section_label("ruler", "Detailed Results"), unsafe_allow_html=True)

    for det in result.detections:
        label = (det.species or det.yolo_class).replace("_", " ")
        conf_str = f"{det.display_confidence:.1%}"
        with st.expander(f"Fish #{det.fish_id} — {label} ({conf_str})", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                if use_yolo:
                    st.markdown("**YOLO Detection**")
                    st.markdown(f"- Class: `{det.yolo_class.replace('_', ' ')}`")
                    st.markdown(f"- Confidence: `{det.yolo_confidence:.3f}`")
                    st.markdown(
                        f"- BBox: `({det.bbox[0]:.0f}, {det.bbox[1]:.0f}, "
                        f"{det.bbox[2]:.0f}, {det.bbox[3]:.0f})`"
                    )
                    if det.estimated_length_cm:
                        st.markdown(f"- Est. length: `{det.estimated_length_cm} cm` ({det.life_stage})")
                else:
                    st.markdown("**Detection mode**")
                    st.markdown("- YOLO: disabled")
                    st.markdown("- BioCLIP classified the full uploaded image")

            with col_b:
                st.markdown("**BioCLIP Top-K Species**")
                if det.top_k_species:
                    df_k = pd.DataFrame(det.top_k_species)
                    df_k["species"] = df_k["species"].str.replace("_", " ")
                    df_k["confidence"] = df_k["confidence"].map("{:.2%}".format)
                    df_k.columns = ["Rank", "Species", "Confidence"]
                    st.dataframe(df_k, hide_index=True, width="stretch")
                else:
                    st.caption("BioCLIP not available")


def _render_raw_data(result) -> None:
    with st.expander("Raw detection data (CSV)"):
        df_raw = pd.DataFrame([d.as_dict() for d in result.detections])
        df_raw["species"] = df_raw["species"].fillna("").str.replace("_", " ")
        st.dataframe(df_raw, width="stretch", hide_index=True)
        st.download_button(
            "Download CSV",
            df_raw.to_csv(index=False),
            "detections.csv",
            "text/csv",
            icon=":material/download:",
        )
