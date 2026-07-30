"""
ui/components.py — small reusable render functions. Every function returns an
HTML string; app-level modules decide when/where to st.markdown() it. No
widgets, no session_state, no pipeline calls live here.
"""
from __future__ import annotations

from . import svg


def banner(kind: str, icon_name: str, text: str) -> str:
    """kind: 'ok' | 'warn'."""
    return f'<div class="fv-banner {kind}">{svg.icon(icon_name, 16)}<span>{text}</span></div>'


def section_label(icon_name: str, text: str) -> str:
    return f'<div class="fv-section-title">{svg.icon(icon_name, 16)}<span>{text}</span></div>'


def metric_card(icon_name: str, value: str, label: str) -> str:
    return f"""
    <div class="fv-metric-card">
      <div class="fv-metric-icon">{svg.icon(icon_name, 19)}</div>
      <div>
        <div class="fv-metric-val">{value}</div>
        <div class="fv-metric-lbl">{label}</div>
      </div>
    </div>"""


def stat_strip(cards_html: list[str], columns: int = 4) -> str:
    """Wrap a list of metric_card() strings in the responsive grid row."""
    style = f'style="grid-template-columns:repeat({columns},1fr);"' if columns != 4 else ""
    return f'<div class="fv-metric-row" {style}>{"".join(cards_html)}</div>'


def species_card(fish_id, species_label: str, life_stage: str | None, conf_pct: float) -> str:
    bar_w = int(conf_pct * 100)
    return f"""
    <div class="fv-species-card">
      <div class="fv-species-top">
        <div class="fv-species-name">Fish #{fish_id}</div>
        <div class="fv-species-tag">{life_stage or '—'}</div>
      </div>
      <div class="fv-species-sci">{species_label}</div>
      <div class="fv-conf-row">
        <div class="fv-conf-track"><div class="fv-conf-fill" style="width:{bar_w}%"></div></div>
        <span class="fv-conf-num">{conf_pct:.0%}</span>
      </div>
    </div>"""


def timeline_item(title: str, meta: str) -> str:
    return f'<div class="fv-tl-item"><div class="fv-tl-title">{title}</div><div class="fv-tl-meta">{meta}</div></div>'


def timeline(items_html: list[str]) -> str:
    return f'<div class="fv-timeline">{"".join(items_html)}</div>'


def ring_with_label(pct: float, label_text: str, color: str, track: str = "#E7E3D8") -> str:
    return (
        f'<div class="fv-ring-wrap">{svg.confidence_ring(pct, 84, 8, color, track, f"{pct:.0%}")}</div>'
        f'<div class="fv-ring-label">{label_text}</div>'
    )
