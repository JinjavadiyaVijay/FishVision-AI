"""
ui/svg.py — inline SVG marks: the logo, a small Lucide-style icon set,
illustration loading, and the circular accuracy/confidence ring.
"""
from __future__ import annotations

import base64
import math
from pathlib import Path

import streamlit as st

from .css import tokens

_ASSETS = Path(__file__).resolve().parent.parent / "assets"
_ILLUSTRATIONS = _ASSETS / "illustrations"


def logo_mark(theme: str = "light", size: int = 36) -> str:
    """
    Minimal mark: viewfinder corners (computer vision) + fish silhouette + AI dot.
    Colors come from the active theme tokens.
    """
    t = tokens(theme)
    forest = t["FOREST"]
    ivory = t["SURFACE"] if theme == "light" else t["TEXT"]
    coral = t["CORAL"]
    moss = t["MOSS"]
    r = size * 0.2
    return f"""
    <svg class="fv-logo" width="{size}" height="{size}" viewBox="0 0 40 40"
         fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <rect width="40" height="40" rx="{r}" fill="{forest}"/>
      <path stroke="{ivory}" stroke-width="1.3" stroke-linecap="round"
            d="M10 12h3.2M10 12v3.2M30 12h-3.2M30 12v3.2M10 28h3.2M10 28v-3.2M30 28h-3.2M30 28v-3.2"/>
      <path d="M11 20.5c3-5.2 8.8-7.6 13.8-4.8 1.8 1 3.1 2.8 3.9 4.8-.8 2-2.1 3.8-3.9 4.8-5 2.8-10.8.4-13.8-4.8Z"
            fill="{ivory}"/>
      <circle cx="16.5" cy="19" r="1.3" fill="{forest}"/>
      <path d="M27.5 20.5 31 18l-1 2.5 1 2.5-3.5-2.5Z" fill="{coral}"/>
      <circle cx="20" cy="10" r="1" fill="{moss}"/>
    </svg>"""


@st.cache_data
def data_uri(filename: str) -> str:
    """Base64 data-uri for any SVG in assets/ or assets/illustrations/."""
    p = _ASSETS / filename
    if not p.exists():
        p = _ILLUSTRATIONS / filename
    if not p.exists():
        placeholder = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 160">'
            '<rect width="200" height="160" rx="12" fill="#EEEBE3"/>'
            '<path d="M40 85c18-28 52-42 82-30 10 4 18 12 24 22-8 14-20 24-36 28-28 8-58-2-70-20Z" fill="#2F4F46"/>'
            '<circle cx="58" cy="78" r="4" fill="#F6F5F0"/>'
            '</svg>'
        )
        encoded = base64.b64encode(placeholder.encode()).decode()
        return f"data:image/svg+xml;base64,{encoded}"
    return f"data:image/svg+xml;base64,{base64.b64encode(p.read_bytes()).decode()}"


@st.cache_data
def get_media_uri(filename: str) -> str:
    """Base64 data-uri for any media in assets/ or assets/illustrations/."""
    p = _ASSETS / filename
    if not p.exists():
        p = _ILLUSTRATIONS / filename
    if not p.exists():
        return ""
    
    ext = p.suffix.lower()
    if ext == ".mp4":
        mime = "video/mp4"
    elif ext == ".png":
        mime = "image/png"
    elif ext == ".svg":
        mime = "image/svg+xml"
    else:
        mime = "application/octet-stream"
        
    encoded = base64.b64encode(p.read_bytes()).decode()
    return f"data:{mime};base64,{encoded}"


_ICON_PATHS = {
    "settings": '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>',
    "info": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    "fish": '<path d="M6.5 12c.94-3.46 4.94-6 8.5-6 3.56 0 6.06 2.54 7 6-1 3.46-3.44 6-7 6-3.56 0-7.56-2.54-8.5-6Z"/><path d="M18 9l3-2.5-1 2.5 1 2.5L18 9Z"/><circle cx="9.5" cy="10.6" r=".6" fill="currentColor" stroke="none"/><path d="M6.5 12c-1.5.7-2.7 1.9-3.5 3M6.5 12c-1.5-.7-2.7-1.9-3.5-3"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "gauge": '<path d="M12 20a8 8 0 1 0-8-8"/><path d="M12 12 8 8"/><path d="M4 20h16"/>',
    "timer": '<circle cx="12" cy="13" r="8"/><path d="M12 9v4l2 2M9 2h6"/>',
    "layers": '<path d="m12 2 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5"/><path d="m3 17 9 5 9-5"/>',
    "ruler": '<path d="M21.3 8.7 8.7 21.3a1 1 0 0 1-1.4 0l-4.6-4.6a1 1 0 0 1 0-1.4L15.3 2.7a1 1 0 0 1 1.4 0l4.6 4.6a1 1 0 0 1 0 1.4Z"/><path d="m7.5 10.5 2 2M10.5 7.5l2 2M13.5 4.5l2 2"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "alert": '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
}


def icon(name: str, size: int = 18, color: str = "currentColor", stroke: float = 2.0) -> str:
    body = _ICON_PATHS.get(name, _ICON_PATHS["info"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    )


def confidence_ring(
    pct: float,
    size: int = 84,
    stroke: int = 7,
    accent: str = "coral",
    label: str = "",
) -> str:
    """Circular progress ring — colors via CSS accent class."""
    pct = max(0.0, min(1.0, pct))
    r = (size - stroke) / 2
    c = 2 * math.pi * r
    dash = c * pct
    return f"""
    <div class="fv-ring-wrap fv-ring-{accent}">
      <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
        <circle class="fv-ring-track" cx="{size/2}" cy="{size/2}" r="{r}"
                fill="none" stroke-width="{stroke}"/>
        <circle class="fv-ring-fill" cx="{size/2}" cy="{size/2}" r="{r}"
                fill="none" stroke-width="{stroke}" stroke-linecap="round"
                stroke-dasharray="{dash:.2f} {c:.2f}"
                transform="rotate(-90 {size/2} {size/2})"/>
        <text class="fv-ring-text" x="50%" y="52%" text-anchor="middle"
              dominant-baseline="middle" font-family="Sora, Inter, sans-serif"
              font-weight="600" font-size="{size*0.22}">{label}</text>
      </svg>
    </div>"""
