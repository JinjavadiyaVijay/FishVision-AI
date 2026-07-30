"""
ui/layout.py — page chrome: top nav, hero, the floating settings panel
(st.popover, standing in for the old sidebar), and the about panel.

Every widget default/range below is copied verbatim from the original
sidebar in app.py — only the container and styling changed.
"""
from __future__ import annotations

import streamlit as st

from . import svg, components as comp


def render_header() -> tuple:
    """Render the brand + icon-button row. Returns (settings_popover, about_popover)
    context managers; callers fill them in afterwards."""
    nav_l, nav_r = st.columns([3, 2], vertical_alignment="center")

    with nav_l:
        st.markdown(
            f"""
            <div class="fv-nav-brand">
              {svg.logo_mark()}
              <div>
                <div class="fv-nav-title">FishVision-AI</div>
                <div class="fv-nav-sub">Intelligent Marine Species Analysis</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with nav_r:
        b_settings, b_about, b_github, b_theme = st.columns(4)

        with b_settings:
            settings_pop = st.popover("", icon=":material/tune:", width="stretch")
        with b_about:
            about_pop = st.popover("", icon=":material/info:", width="stretch")
        with b_github:
            # Placeholder URL — no repository link was supplied for this project;
            # point it at the real FishVision-AI repo.
            st.link_button(
                "", "https://github.com",
                icon=":material/code:", width="stretch",
            )
        with b_theme:
            theme = st.session_state.get("fv_theme", "light")
            theme_icon = ":material/dark_mode:" if theme == "light" else ":material/light_mode:"
            if st.button("", icon=theme_icon, width="stretch", key="theme_toggle"):
                st.session_state.fv_theme = "dark" if theme == "light" else "light"
                st.rerun()

    return settings_pop, about_pop


def render_page_header(default_yolo_path: str, eval_data: dict | None) -> dict:
    """Render nav, popovers, hero, and return pipeline settings from the panel."""
    settings_pop, about_pop = render_header()
    render_about(about_pop)
    settings = render_settings_panel(settings_pop, default_yolo_path, eval_data)
    render_hero()
    return settings


def render_footer() -> None:
    st.markdown(
        """
        <div class="fv-footer">
          <span>FishVision-AI · YOLO + BioCLIP 2 · Marine species identification</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_about(about_pop) -> None:
    with about_pop:
        st.markdown(f"""
        <div class="fv-panel-heading">{svg.icon('fish', 16)}<span>About FishVision-AI</span></div>
        <p style="font-size:.86rem;color:var(--muted);line-height:1.55;">
        FishVision-AI pairs a YOLO detector with a BioCLIP 2 fine-grained classifier
        to localize, identify, and biometrically measure fish from a single frame —
        no dedicated training dataset required for length or weight estimation.
        </p>
        """, unsafe_allow_html=True)
        st.markdown('<hr class="fv-panel-divider"/>', unsafe_allow_html=True)
        st.markdown(
            "**Pipeline** &nbsp;YOLOv8 → BioCLIP 2 (ViT-L/14, LoRA)  \n"
            "**Coverage** &nbsp;157 species"
        )


def render_hero() -> None:
    hero_l, hero_r = st.columns([3, 2], vertical_alignment="center")
    with hero_l:
        st.markdown(
            f"""
            <div>
              <span class="fv-hero-eyebrow">{svg.icon('layers', 13)} Computer Vision · Marine Research</span>
              <h1>See what's beneath the surface.</h1>
              <p>Upload an underwater frame — FishVision-AI detects every fish,
              identifies its species, and estimates length, weight, and life
              stage in a single pass.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hero_r:
        st.markdown(
            f'<div class="fv-hero-illustration">'
            f'<img src="{svg.data_uri("World_Oceans_Day-amico.svg")}" alt="Ocean research illustration"/>'
            f"</div>",
            unsafe_allow_html=True,
        )


def render_settings_panel(settings_pop, default_yolo_path: str, eval_data: dict | None) -> dict:
    """Fills the floating control panel and returns every value the pipeline needs.
    Widget names/defaults/ranges are unchanged from the original sidebar."""
    with settings_pop:
        st.markdown(
            f'<div class="fv-panel-heading">{svg.icon("target", 15)}<span>Detection</span></div>',
            unsafe_allow_html=True,
        )
        yolo_path = st.text_input(
            "YOLO weights", value=default_yolo_path,
            help="YOLOv8 .pt file for fish localisation",
        )
        conf_thresh = st.slider("Confidence", 0.10, 0.90, 0.25, 0.05,
                                 help="Minimum YOLO detection confidence")
        iou_thresh = st.slider("IoU (NMS)", 0.10, 0.90, 0.45, 0.05,
                                help="Non-max suppression overlap threshold")

        st.markdown('<hr class="fv-panel-divider"/>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="fv-panel-heading">{svg.icon("layers", 15)}<span>Classification</span></div>',
            unsafe_allow_html=True,
        )
        top_k = st.slider("Top-K species", 1, 10, 5)
        cm_per_px = st.number_input("cm / pixel (0 = skip)", 0.0, step=0.001, format="%.4f")
        adult_cm = st.number_input("Adult threshold (cm)", 0.0, value=10.0, step=0.5)

        if eval_data:
            st.markdown('<hr class="fv-panel-divider"/>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="fv-panel-heading">{svg.icon("gauge", 15)}<span>Model Performance</span></div>',
                unsafe_allow_html=True,
            )
            pc1, pc2, pc3 = st.columns(3)
            pc1.markdown(f"**Top-1**  \n`{eval_data['top1_accuracy']:.1%}`")
            pc2.markdown(f"**Top-5**  \n`{eval_data['top5_accuracy']:.1%}`")
            pc3.markdown(f"**F1**  \n`{eval_data['f1_macro']:.4f}`")
            st.caption(
                f"{eval_data['total_images']:,} test images · "
                f"{eval_data['num_classes']} species"
            )

    return dict(
        yolo_path=yolo_path,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        top_k=top_k,
        cm_per_px=cm_per_px,
        adult_cm=adult_cm,
    )
