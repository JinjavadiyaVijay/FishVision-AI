"""
ui/layout.py — page chrome: top nav, hero, the floating settings panel
(st.popover, standing in for the old sidebar), and the about panel.

Widget defaults/ranges are unchanged — only presentation and structure changed.
"""
from __future__ import annotations

import streamlit as st

from . import svg, components as comp


def render_header() -> tuple:
    """Render the brand + icon-button row. Returns (settings_popover, about_popover)."""
    theme = st.session_state.get("fv_theme", "light")
    nav_l, nav_r = st.columns([3, 2], vertical_alignment="center")

    with nav_l:
        st.markdown(
            f"""
            <div class="fv-nav-brand">
              {svg.logo_mark(theme)}
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
            st.link_button(
                "", "https://github.com",
                icon=":material/code:", width="stretch",
            )
        with b_theme:
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
        st.markdown('<div class="fv-popover-content">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="fv-settings-header" style="position:static;border:none;margin:0 0 0.85rem 0;padding:0 0 0.75rem 0;">
              <div class="fv-settings-header-top">{svg.icon('fish', 16)}<span>About FishVision-AI</span></div>
            </div>
            <p class="fv-about-text">
              FishVision-AI pairs a YOLO detector with a BioCLIP 2 fine-grained classifier
              to localize, identify, and biometrically measure fish from a single frame —
              no dedicated training dataset required for length or weight estimation.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<hr class="fv-panel-divider"/>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="fv-about-meta">
              <strong>Pipeline</strong>&nbsp; YOLOv8 → BioCLIP 2 (ViT-L/14, LoRA)<br/>
              <strong>Coverage</strong>&nbsp; 157 species
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_hero() -> None:
    hero_l, hero_r = st.columns([3, 2], vertical_alignment="center")
    with hero_l:
        st.markdown(
            f"""
            <div class="fv-hero">
              <span class="fv-hero-eyebrow">{svg.icon('layers', 13)} Computer Vision · Marine Research</span>
              <h1>See what's beneath the surface.</h1>
              <p class="fv-hero-copy">Upload an underwater frame — FishVision-AI detects every fish,
              identifies its species, and estimates length, weight, and life
              stage in a single pass.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hero_r:
        st.markdown(
            f'<div class="fv-hero-illustration">'
            f'<img src="{svg.get_media_uri("image.png")}" alt="Ocean research illustration"/>'
            f"</div>",
            unsafe_allow_html=True,
        )


def _slider_with_value(label: str, fmt: str, **kwargs) -> float:
    """Slider with plain-text value (no orange thumb box)."""
    result = st.slider(label, **kwargs)
    st.markdown(comp.slider_value(fmt.format(result)), unsafe_allow_html=True)
    return result


def render_settings_panel(settings_pop, default_yolo_path: str, eval_data: dict | None) -> dict:
    """Fills the floating control panel and returns every value the pipeline needs."""
    with settings_pop:
        st.markdown(
            f"""
            <div class="fv-settings-header">
              <div class="fv-settings-header-top">{svg.icon("settings", 17)}<span>Settings</span></div>
              <p class="fv-settings-sub">Detection &amp; classification parameters</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="fv-panel-heading">{svg.icon("target", 14)}<span>Detection</span></div>',
            unsafe_allow_html=True,
        )
        use_yolo = st.toggle(
            "Use YOLO detection",
            value=True,
            help="Off = skip YOLO and classify the whole image with BioCLIP only (best for single-fish photos)",
        )


        yolo_path = st.text_input(
            "YOLO weights",
            value=default_yolo_path,
            disabled=not use_yolo,
            help="YOLOv8 .pt file for fish localisation",
        )
        conf_thresh = _slider_with_value(
            "Confidence",
            "{:.2f}",
            min_value=0.01, max_value=0.90, value=0.10, step=0.01,
            disabled=not use_yolo,
            help="Minimum YOLO detection confidence (default 0.10 for underwater photos)",
        )
        iou_thresh = _slider_with_value(
            "IoU (NMS)",
            "{:.2f}",
            min_value=0.10, max_value=0.90, value=0.45, step=0.05,
            disabled=not use_yolo,
            help="Non-max suppression overlap threshold",
        )

        note_hidden = "" if not use_yolo else " fv-field-note--hidden"
        st.markdown(
            f'<p class="fv-field-note{note_hidden}">YOLO is off — BioCLIP will identify species from the '
            'full uploaded image. No bounding boxes or multi-fish localisation.</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<hr class="fv-panel-divider"/>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="fv-panel-heading">{svg.icon("layers", 14)}<span>Classification</span></div>',
            unsafe_allow_html=True,
        )
        top_k = _slider_with_value(
            "Top-K species",
            "{:.0f}",
            min_value=1, max_value=10, value=5, step=1,
        )
        cm_per_px = st.number_input("cm / pixel (0 = skip)", 0.0, step=0.001, format="%.4f")
        adult_cm = st.number_input("Adult threshold (cm)", 0.0, value=10.0, step=0.5)

        if eval_data:
            st.markdown('<hr class="fv-panel-divider"/>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="fv-panel-heading">{svg.icon("gauge", 14)}<span>Model Performance</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="fv-perf-grid">
                  <div class="fv-perf-stat"><strong>Top-1</strong><span>{eval_data['top1_accuracy']:.1%}</span></div>
                  <div class="fv-perf-stat"><strong>Top-5</strong><span>{eval_data['top5_accuracy']:.1%}</span></div>
                  <div class="fv-perf-stat"><strong>F1</strong><span>{eval_data['f1_macro']:.4f}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                f"{eval_data['total_images']:,} test images · "
                f"{eval_data['num_classes']} species"
            )

    return dict(
        use_yolo=use_yolo,
        yolo_path=yolo_path if use_yolo else default_yolo_path,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        top_k=int(top_k),
        cm_per_px=cm_per_px,
        adult_cm=adult_cm,
    )
