"""
ui/css.py — design tokens and the full stylesheet.

Two token sets (light "field-notebook" / dark "deep ocean") get swapped into
one CSS template. Nothing here touches Streamlit widgets directly — app.py
just calls inject(theme) once per run.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

_TOKENS = {
    "light": {
        "FOREST": "#163A32", "OCEAN": "#254441", "SURFACE": "#F8F7F3",
        "SURFACE_2": "#FFFFFF", "CORAL": "#D97B5F", "MOSS": "#758E67",
        "TEXT": "#202124", "MUTED": "#666666", "BORDER": "#E7E3D8",
        "SHADOW": "rgba(22,58,50,0.10)",
    },
    "dark": {
        "FOREST": "#EDEAE0", "OCEAN": "#9FB8B0", "SURFACE": "#132A25",
        "SURFACE_2": "#1B3A33", "CORAL": "#E2916F", "MOSS": "#93AD84",
        "TEXT": "#F1EFE8", "MUTED": "#9FB0AB", "BORDER": "#2C4941",
        "SHADOW": "rgba(0,0,0,0.35)",
    },
}

_TEMPLATE_PATH = Path(__file__).resolve().parent / "style.css"


def tokens(theme: str) -> dict:
    return _TOKENS.get(theme, _TOKENS["light"])


@st.cache_data
def _build(theme: str) -> str:
    """Read style.css and substitute tokens — cached so this only runs once per
    theme, not on every rerun (every slider move / upload previously re-read
    the file from disk and re-did string substitution for nothing)."""
    css = _TEMPLATE_PATH.read_text(encoding="utf-8")
    for key, val in tokens(theme).items():
        css = css.replace(f"{{{{{key}}}}}", val)
    return css


def get_css(theme: str = "light") -> str:
    """Return the full '<style>...</style>' markup for the active theme —
    call sites do st.markdown(css.get_css(theme), unsafe_allow_html=True)."""
    return f"<style>{_build(theme)}</style>"


def inject(theme: str = "light") -> None:
    """Convenience wrapper: injects the stylesheet directly (equivalent to
    st.markdown(get_css(theme), unsafe_allow_html=True))."""
    st.markdown(get_css(theme), unsafe_allow_html=True)