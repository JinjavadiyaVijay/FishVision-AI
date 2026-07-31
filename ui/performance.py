"""
ui/performance.py — the "Species Accuracy" tab. Every key read from
`eval_data` below (top1_accuracy, per_class, support, ...) matches the
evaluation_test.json schema used by the original app.py unchanged.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from . import components as comp


def render_performance_tab(eval_data: dict | None) -> None:
    st.markdown(comp.section_label("gauge", "BioCLIP 2 — Per-Species Accuracy (Test Set)"), unsafe_allow_html=True)

    if eval_data is None:
        st.markdown(
            comp.banner("warn", "Evaluation results not found. Run `python scripts/evaluate_model.py` first."),
            unsafe_allow_html=True,
        )
        return

    _render_global_metrics(eval_data)
    _render_summary_banner(eval_data)
    df_acc = _build_species_df(eval_data)
    _render_table(df_acc, eval_data)
    _render_distribution(eval_data)


def _render_global_metrics(eval_data: dict) -> None:
    ring_col, cards_col = st.columns([1, 2], gap="large")
    with ring_col:
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(
                comp.ring_with_label(eval_data["top1_accuracy"], "Top-1 Accuracy", "#D97B5F"),
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                comp.ring_with_label(eval_data["top5_accuracy"], "Top-5 Accuracy", "#758E67"),
                unsafe_allow_html=True,
            )
    with cards_col:
        st.markdown(
            comp.stat_strip([
                comp.metric_card("target", f"{eval_data['precision_macro']:.4f}", "Precision (M)"),
                comp.metric_card("layers", f"{eval_data['recall_macro']:.4f}", "Recall (M)"),
                comp.metric_card("gauge", f"{eval_data['f1_macro']:.4f}", "F1 Macro"),
                comp.metric_card("gauge", f"{eval_data['f1_weighted']:.4f}", "F1 Weighted"),
            ]),
            unsafe_allow_html=True,
        )


def _render_summary_banner(eval_data: dict) -> None:
    st.markdown(
        comp.banner(
            "ok",
            "check",
            f"Test set — <b>{eval_data['total_images']:,} images</b> · "
            f"<b>{eval_data['num_classes']} species</b> · "
            f"<b>{eval_data['ms_per_image']:.1f} ms/image</b> · "
            f"Experiment: bioclip2_full_20260716_160143",
        ),
        unsafe_allow_html=True,
    )


def _build_species_df(eval_data: dict) -> pd.DataFrame:
    pc = eval_data["per_class"]
    df_acc = pd.DataFrame(pc)
    df_acc["Species"] = df_acc["species"].str.replace("_", " ")
    df_acc["Accuracy"] = df_acc["accuracy"].map("{:.1%}".format)
    df_acc["F1"] = df_acc["f1"].map("{:.4f}".format)
    df_acc["Precision"] = df_acc["precision"].map("{:.4f}".format)
    df_acc["Recall"] = df_acc["recall"].map("{:.4f}".format)
    df_acc["Images"] = df_acc["support"]
    df_acc["acc_raw"] = df_acc["accuracy"]
    return df_acc


def _render_table(df_acc: pd.DataFrame, eval_data: dict) -> None:
    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1:
        search = st.text_input("Search species", placeholder="e.g. Lutjanus", icon=":material/search:")
    with fc2:
        sort_by = st.selectbox("Sort by", ["F1 ↓", "Accuracy ↓", "Accuracy ↑", "Species A–Z", "Images ↓"])
    with fc3:
        min_acc = st.slider("Min accuracy", 0.0, 1.0, 0.0, 0.05)

    mask = df_acc["acc_raw"] >= min_acc
    if search:
        mask &= df_acc["Species"].str.contains(search, case=False)
    df_show = df_acc[mask].copy()

    sort_map = {
        "F1 ↓": ("f1", False),
        "Accuracy ↓": ("accuracy", False),
        "Accuracy ↑": ("accuracy", True),
        "Species A–Z": ("species", True),
        "Images ↓": ("support", False),
    }
    scol, sasc = sort_map[sort_by]
    df_show = df_show.sort_values(scol, ascending=sasc)

    st.caption(f"Showing {len(df_show)} of {len(df_acc)} species")

    display_df = df_show[["Species", "Accuracy", "F1", "Precision", "Recall", "Images"]].reset_index(drop=True)
    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        height=min(60 + len(display_df) * 35, 650),
        column_config={
            "Species": st.column_config.TextColumn("Species", width="large"),
            "Accuracy": st.column_config.TextColumn("Top-1 Acc", width="small"),
            "F1": st.column_config.TextColumn("F1", width="small"),
            "Precision": st.column_config.TextColumn("Precision", width="small"),
            "Recall": st.column_config.TextColumn("Recall", width="small"),
            "Images": st.column_config.NumberColumn("Test Images", width="small"),
        },
    )

    st.download_button(
        "Download full accuracy table (CSV)",
        df_show[["Species", "Accuracy", "F1", "Precision", "Recall", "Images"]].to_csv(index=False),
        "bioclip_species_accuracy.csv",
        "text/csv",
        icon=":material/download:",
    )


def _render_distribution(eval_data: dict) -> None:
    with st.expander("Distribution summary"):
        pc = eval_data["per_class"]
        bins = {"≥ 90%": 0, "70–89%": 0, "50–69%": 0, "< 50%": 0}
        for r in pc:
            a = r["accuracy"]
            if a >= 0.90:
                bins["≥ 90%"] += 1
            elif a >= 0.70:
                bins["70–89%"] += 1
            elif a >= 0.50:
                bins["50–69%"] += 1
            else:
                bins["< 50%"] += 1
        st.markdown(
            comp.stat_strip([
                comp.metric_card("layers", str(count), f"{label} ({count/len(pc)*100:.0f}%)")
                for label, count in bins.items()
            ]),
            unsafe_allow_html=True,
        )