"""Seaborn plotting helpers for lastfm_bench.ipynb."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from metrics import BASELINE_COL, EMBEDDING_COL


def show_theme(
    metrics_df: pd.DataFrame,
    theme: str,
    variant: str | None = None,
    n: int | list[int] | None = None,
) -> pd.DataFrame:
    mask = metrics_df["theme"] == theme
    if variant is not None:
        mask &= metrics_df["variant"] == variant
    if n is not None:
        if isinstance(n, list):
            mask &= metrics_df["n"].isin(n)
        else:
            mask &= metrics_df["n"] == n
    return metrics_df.loc[mask].sort_values(["variant", "metric", "n"])


def _long_recommenders(df: pd.DataFrame, id_vars: list[str]) -> pd.DataFrame:
    return df.melt(
        id_vars=id_vars,
        value_vars=[EMBEDDING_COL, BASELINE_COL],
        var_name="recommender",
        value_name="value",
    )


def legend_recommenders(ax, **kwargs):
    palette = sns.color_palette(n_colors=2)
    handles = [
        Patch(facecolor=palette[i], label=label)
        for i, label in enumerate([EMBEDDING_COL, BASELINE_COL])
    ]
    ax.legend(handles=handles, **kwargs)


def plot_metric_sweep(
    ax,
    metrics_df: pd.DataFrame,
    theme: str,
    variant: str,
    metric: str,
    *,
    title: str | None = None,
    legend: bool = True,
):
    sub = show_theme(metrics_df, theme, variant)
    sub = sub[sub["metric"] == metric].sort_values("n")
    long = _long_recommenders(sub, id_vars=["n"])
    sns.lineplot(data=long, x="n", y="value", hue="recommender", marker="o", ax=ax, legend=legend)
    ax.set(xlabel="n", ylabel=metric, title=title or metric)
    ax.set_xticks(sorted(sub["n"].dropna().unique()))


def plot_variant_bars(ax, chunk: pd.DataFrame, *, title: str, ylim: tuple[float, float] | None = (0, 1)):
    long = _long_recommenders(chunk.reset_index(), id_vars=["variant"])
    sns.barplot(data=long, x="variant", y="value", hue="recommender", ax=ax, legend=False)
    ax.set_title(title)
    ax.set(xlabel=None, ylabel=None)
    if ylim is not None:
        ax.set_ylim(*ylim)


def plot_recommender_bars(
    ax,
    row: pd.Series,
    *,
    title: str,
    ylim: tuple[float, float] | None = None,
    hline: float | None = None,
):
    long = pd.DataFrame(
        {"recommender": [EMBEDDING_COL, BASELINE_COL], "value": [row[EMBEDDING_COL], row[BASELINE_COL]]}
    )
    sns.barplot(data=long, x="recommender", y="value", ax=ax, legend=False)
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    if hline is not None:
        ax.axhline(hline, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    if ylim is not None:
        ax.set_ylim(*ylim)


def plot_symmetric_sweep(
    ax,
    metrics_df: pd.DataFrame,
    lines: list[tuple[str, str, str]],
    *,
    top_n: int,
    title: str,
    ylabel: str = "Jaccard",
):
    parts = []
    for variant, metric, label in lines:
        sub = show_theme(metrics_df, "disagreement", variant)
        sub = sub[sub["metric"] == metric].sort_values("n")[["n", EMBEDDING_COL]].copy()
        sub["series"] = label
        parts.append(sub.rename(columns={EMBEDDING_COL: "value"}))
    long = pd.concat(parts, ignore_index=True)
    sns.lineplot(data=long, x="n", y="value", hue="series", marker="o", ax=ax)
    ax.set(xlabel="n", ylabel=ylabel, title=title)
    ax.set_xticks(range(1, top_n + 1))


def _theme_metric(
    metrics_df: pd.DataFrame,
    theme: str,
    variant: str,
    metric: str,
    n: int,
    *,
    col: str = EMBEDDING_COL,
) -> float:
    row = metrics_df.loc[
        (metrics_df["theme"] == theme)
        & (metrics_df["variant"] == variant)
        & (metrics_df["metric"] == metric)
        & (metrics_df["n"] == n),
        col,
    ]
    return float(row.iloc[0]) if len(row) else float("nan")


def _disagreement_metric(
    metrics_df: pd.DataFrame,
    variant: str,
    metric: str,
    n: int,
) -> float:
    return _theme_metric(metrics_df, "disagreement", variant, metric, n)


def _agreement_metric(metrics_df: pd.DataFrame, variant: str, metric: str, n: int) -> float:
    return _theme_metric(metrics_df, "agreement", variant, metric, n)


def _section_two_line_title(ax, heading: str, subtitle: str) -> None:
    ax.set_title("")
    ax.text(
        0.5,
        1.10,
        heading,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="semibold",
    )
    ax.text(
        0.5,
        1,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )


def _disagreement_two_line_title(ax, subtitle: str) -> None:
    _section_two_line_title(ax, "Disagreement", subtitle)


def _style_pct_axes(ax, *, ylabel: str | None = None) -> None:
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=9, labelpad=2)
    ax.tick_params(axis="both", labelsize=8, pad=2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))


def _style_disagreement_axes(ax, *, ylabel: str | None = None) -> None:
    _style_pct_axes(ax, ylabel=ylabel)


def _compact_section_layout(fig, *, legend: bool = False) -> None:
    right = 0.86 if legend else 0.98
    fig.subplots_adjust(left=0.13, right=right, bottom=0.16, top=0.78)


def _pct_text(ax, x: float, y: float, value: float, *, va: str = "center") -> None:
    if not np.isfinite(value) or value <= 0:
        return
    ax.text(x, y, f"{value:.1%}", ha="center", va=va, fontsize=8)


def plot_disagreement_jaccard_bars(
    metrics_df: pd.DataFrame,
    *,
    n: int,
    figsize: tuple[float, float] = (5, 2.75),
):
    """Mean Jaccard at fixed n for albums, artists, and comparable tags."""
    specs = [
        ("albums", "mean_jaccard", "Albums"),
        ("artists", "mean_jaccard", "Artists"),
        ("tags_comparable", "mean_jaccard_comparable", "Tags\n(comparable)"),
    ]
    labels = [label for _, _, label in specs]
    values = [_disagreement_metric(metrics_df, variant, metric, n) for variant, metric, _ in specs]

    fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette(n_colors=len(labels))
    bars = ax.bar(labels, values, color=palette)
    _disagreement_two_line_title(ax, f"Mean Jaccard @ n = {n}")
    _style_disagreement_axes(ax, ylabel="Mean Jaccard")
    ax.set_ylim(0, 1)
    ax.set_xlabel(None)
    ax.tick_params(axis="x", labelsize=8)
    for bar, value in zip(bars, values):
        _pct_text(
            ax,
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            value,
            va="bottom",
        )
    plt.tight_layout(pad=0.3)
    _compact_section_layout(fig)
    plt.show()


def plot_disagreement_overlap_stacked(
    metrics_df: pd.DataFrame,
    *,
    n: int,
    figsize: tuple[float, float] = (5, 2.75),
):
    """Stacked zero-overlap vs any-overlap shares at fixed n."""
    specs = [
        ("albums", "zero_overlap_share", "any_overlap_share", "Albums"),
        ("artists", "zero_overlap_share", "any_overlap_share", "Artists"),
        (
            "tags_comparable",
            "zero_overlap_comparable_share",
            "any_overlap_comparable_share",
            "Tags\n(comparable)",
        ),
    ]
    labels = [label for *_, label in specs]
    zero = [_disagreement_metric(metrics_df, variant, z_metric, n) for variant, z_metric, _, _ in specs]
    any_overlap = [_disagreement_metric(metrics_df, variant, a_metric, n) for variant, _, a_metric, _ in specs]

    colors = sns.color_palette(n_colors=2)

    fig, ax = plt.subplots(figsize=figsize)
    bars_zero = ax.bar(labels, zero, label="Zero overlap", color=colors[0])
    bars_any = ax.bar(labels, any_overlap, bottom=zero, label="Any overlap", color=colors[1])
    ax.tick_params(axis="x", labelsize=8)
    _disagreement_two_line_title(ax, f"Overlap @ n = {n}")
    _style_disagreement_axes(ax, ylabel="Share of shared queries")
    ax.set_ylim(0, 1)
    for bz, ba, z, a in zip(bars_zero, bars_any, zero, any_overlap):
        if z >= 0.04:
            _pct_text(ax, bz.get_x() + bz.get_width() / 2, z / 2, z)
        if a >= 0.04:
            _pct_text(ax, ba.get_x() + ba.get_width() / 2, z + a / 2, a)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=9)
    plt.tight_layout(pad=0.3)
    _compact_section_layout(fig, legend=True)
    plt.show()


def _plot_baseline_recovery_metric_bars(
    metrics_df: pd.DataFrame,
    *,
    n: int,
    metric: str,
    subtitle: str,
    ylabel: str,
    figsize: tuple[float, float] = (4, 2.75),
) -> None:
    specs = [("full", "Full"), ("in_corpus", "In-corpus")]
    labels = [label for _, label in specs]
    values = [_agreement_metric(metrics_df, variant, metric, n) for variant, _ in specs]

    fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette(n_colors=len(labels))
    bars = ax.bar(labels, values, color=palette)
    _section_two_line_title(ax, "Baseline recovery", subtitle)
    _style_pct_axes(ax, ylabel=ylabel)
    ax.set_ylim(0, 1)
    ax.set_xlabel(None)
    ax.tick_params(axis="x", labelsize=8)
    for bar, value in zip(bars, values):
        _pct_text(
            ax,
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            value,
            va="bottom",
        )
    plt.tight_layout(pad=0.3)
    _compact_section_layout(fig)
    plt.show()


def plot_baseline_recovery_hit_rate_bars(
    metrics_df: pd.DataFrame,
    *,
    n: int,
    figsize: tuple[float, float] = (4, 2.75),
) -> None:
    """Hit rate at fixed n for full vs in-corpus queries."""
    _plot_baseline_recovery_metric_bars(
        metrics_df,
        n=n,
        metric="hit_rate",
        subtitle=f"Hit rate @ n = {n}",
        ylabel="Hit rate",
        figsize=figsize,
    )


def plot_baseline_recovery_precision_bars(
    metrics_df: pd.DataFrame,
    *,
    n: int,
    figsize: tuple[float, float] = (4, 2.75),
) -> None:
    """Precision at fixed n for full vs in-corpus queries."""
    _plot_baseline_recovery_metric_bars(
        metrics_df,
        n=n,
        metric="precision",
        subtitle=f"Precision @ n = {n}",
        ylabel="Precision",
        figsize=figsize,
    )


def plot_repetition_sweeps(
    metrics_df: pd.DataFrame,
    variant: str,
    metrics: tuple[str, ...] = ("unique_share", "hub_slot_share"),
    *,
    figsize: tuple[float, float] = (8, 3),
):
    titles = {
        "unique_share": "Coverage",
        "hub_slot_share": "Hub slot share",
        "gini": "Gini",
        "top1pct_concentration": "Top 1%",
    }
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        plot_metric_sweep(
            ax,
            metrics_df,
            "repetition",
            variant,
            metric,
            title=titles.get(metric, metric),
            legend=False,
        )
        ax.set_ylim(0, 1)
    legend_recommenders(axes[0])
    plt.tight_layout()
    plt.show()
