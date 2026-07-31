"""
ui/css.py — design tokens and the full stylesheet.

Two token sets (light / dark) get swapped into one CSS template.
Nothing here touches Streamlit widgets directly — app.py calls get_css(theme).
"""
from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

_TOKENS = {
    "light": {
        "FOREST": "#1A3D34",
        "OCEAN": "#2F4F46",
        "CHARCOAL": "#2A2A28",
        "SURFACE": "#F6F5F0",
        "SURFACE_2": "#FFFFFF",
        "SURFACE_3": "#EEEBE3",
        "CORAL": "#C4715A",
        "MOSS": "#6B8260",
        "SLATE": "#5C6B66",
        "TEXT": "#1E1E1C",
        "MUTED": "#6B6B66",
        "BORDER": "#DDD8CC",
        "BORDER_SUBTLE": "#E8E4DB",
        "SHADOW": "rgba(26, 61, 52, 0.08)",
        "BANNER_OK_BG": "rgba(107, 130, 96, 0.12)",
        "BANNER_OK_BORDER": "rgba(107, 130, 96, 0.32)",
        "BANNER_OK_TEXT": "#3D5233",
        "BANNER_WARN_BG": "rgba(196, 113, 90, 0.10)",
        "BANNER_WARN_BORDER": "rgba(196, 113, 90, 0.28)",
        "BANNER_WARN_TEXT": "#7A4030",
        "INPUT_BG": "#FFFFFF",
        "POPOVER_BG": "#FFFFFF",
        "ICON_BG": "rgba(26, 61, 52, 0.07)",
        "HERO_BADGE_BG": "rgba(107, 130, 96, 0.12)",
        "TRACK": "#DDD8CC",
        "THUMB": "#C4715A",
        "DISABLED": "rgba(107, 107, 102, 0.45)",
    },
    "dark": {
        "FOREST": "#E8E4DA",
        "OCEAN": "#9AADA6",
        "CHARCOAL": "#0F1412",
        "SURFACE": "#141916",
        "SURFACE_2": "#1C2420",
        "SURFACE_3": "#243028",
        "CORAL": "#D4846E",
        "MOSS": "#8FA885",
        "SLATE": "#7A8A84",
        "TEXT": "#ECEAE4",
        "MUTED": "#9AA8A2",
        "BORDER": "#2E3D36",
        "BORDER_SUBTLE": "#243028",
        "SHADOW": "rgba(0, 0, 0, 0.45)",
        "BANNER_OK_BG": "rgba(143, 168, 133, 0.14)",
        "BANNER_OK_BORDER": "rgba(143, 168, 133, 0.30)",
        "BANNER_OK_TEXT": "#B8CEB0",
        "BANNER_WARN_BG": "rgba(212, 132, 110, 0.14)",
        "BANNER_WARN_BORDER": "rgba(212, 132, 110, 0.30)",
        "BANNER_WARN_TEXT": "#E8B5A4",
        "INPUT_BG": "#1C2420",
        "POPOVER_BG": "#1C2420",
        "ICON_BG": "rgba(232, 228, 218, 0.08)",
        "HERO_BADGE_BG": "rgba(143, 168, 133, 0.14)",
        "TRACK": "#2E3D36",
        "THUMB": "#D4846E",
        "DISABLED": "rgba(154, 168, 162, 0.40)",
    },
}

_TEMPLATE_PATH = Path(__file__).resolve().parent / "style.css"


def tokens(theme: str) -> dict:
    return _TOKENS.get(theme, _TOKENS["light"])


@st.cache_data
def _build(theme: str) -> str:
    css = _TEMPLATE_PATH.read_text(encoding="utf-8")
    for key, val in tokens(theme).items():
        pattern = r'\{\s*\{\s*' + key + r'\s*\}\s*\}'
        css = re.sub(pattern, val, css)
    return css


def get_css(theme: str = "light") -> str:
    """Return the full '<style>...</style>' markup for the active theme."""
    css = _build(theme)
    return (
        f"<style>{css}</style>"
        f'<script>document.documentElement.setAttribute("data-fv-theme","{theme}");</script>'
    )


def inject(theme: str = "light") -> None:
    st.markdown(get_css(theme), unsafe_allow_html=True)
