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


def _section_two_line_title(
    ax,
    heading: str,
    subtitle: str = "",
    *,
    align: str = "center",
) -> None:
    x = 0.0 if align == "left" else 0.5
    ha = "left" if align == "left" else "center"
    ax.set_title("")
    ax.text(
        x,
        1.10,
        heading,
        transform=ax.transAxes,
        ha=ha,
        va="bottom",
        fontsize=14,
        fontweight="semibold",
    )
    if subtitle:
        ax.text(
            x,
            1,
            subtitle,
            transform=ax.transAxes,
            ha=ha,
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


def _popularity_metric(
    metrics_df: pd.DataFrame,
    variant: str,
    metric: str,
    n: int,
    *,
    col: str = EMBEDDING_COL,
) -> float:
    return _theme_metric(metrics_df, "popularity", variant, metric, n, col=col)


def _style_ratio_axes(ax, *, ylabel: str | None = None) -> None:
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=9, labelpad=2)
    ax.tick_params(axis="both", labelsize=8, pad=2)


def _metric_bar_label(value: float, *, as_pct: bool) -> str:
    return f"{value:.1%}" if as_pct else f"{value:.2f}"


def plot_popularity_bars(
    metrics_df: pd.DataFrame,
    *,
    n: int,
    figsize: tuple[float, float] = (10, 2.75),
) -> None:
    """Popularity bias: review ratio, listener/catalog ratio, and upscale-pick share."""
    specs = [
        ("reviews", "mean_reviews_ratio", "(a) Mean Reviews*", False, 1.0),
        ("listeners", "mean_rec_catalog_ratio", "(b) Mean Listeners*", False, 1.0),
        ("listeners", "rec_more_popular_share", "(c) More popular than seed (listeners)", True, None),
    ]
    fig, axes = plt.subplots(1, len(specs), figsize=figsize)
    axes = np.atleast_1d(axes)
    palette = sns.color_palette(n_colors=2)
    x = np.arange(2)

    for ax, (variant, metric, panel_title, as_pct, hline) in zip(axes, specs):
        values = [
            _popularity_metric(metrics_df, variant, metric, n),
            _popularity_metric(metrics_df, variant, metric, n, col=BASELINE_COL),
        ]
        bars = ax.bar(x, values, color=palette)
        ax.set_title(panel_title, fontsize=9)
        ax.set_xlabel(None)
        ax.tick_params(axis="x", labelbottom=False)

        if as_pct:
            _style_pct_axes(ax)
            ax.set_ylim(0, 1)
        else:
            _style_ratio_axes(ax)
            finite = [v for v in values if np.isfinite(v)]
            ymax = max(finite + [hline or 0]) * 1.15 if finite else 1.2
            ax.set_ylim(0, ymax)
            if hline is not None:
                ax.axhline(hline, color="gray", linestyle="--", linewidth=1, alpha=0.6)

        for bar, value in zip(bars, values):
            if not np.isfinite(value):
                continue
            pad = 0.02 if as_pct else max(v for v in values if np.isfinite(v)) * 0.02
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + pad,
                _metric_bar_label(value, as_pct=as_pct),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.text(0.13, 0.97, "Popularity bias", ha="left", va="top", fontsize=14, fontweight="semibold")
    legend_recommenders(axes[-1], loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=9)
    fig.text(0.13, 0.07, "* adjusted to catalog mean", ha="left", va="bottom", fontsize=8)
    plt.tight_layout(pad=0.3)
    fig.subplots_adjust(left=0.13, right=0.86, bottom=0.14, top=0.80)
    plt.show()


def plot_diversity_bars(
    metrics_df: pd.DataFrame,
    *,
    n: int,
    variant: str = "artists",
    figsize: tuple[float, float] = (7, 2.75),
) -> None:
    """Within-list diversity: mean ratio and fully diverse list share."""
    specs = [
        ("mean_diversity_ratio", "(a) Mean diversity ratio", True, 1.0),
        ("all_distinct_share", "(b) Fully diverse lists", True, None),
    ]
    fig, axes = plt.subplots(1, len(specs), figsize=figsize)
    axes = np.atleast_1d(axes)
    palette = sns.color_palette(n_colors=2)
    x = np.arange(2)

    for ax, (metric, panel_title, as_pct, hline) in zip(axes, specs):
        values = [
            _theme_metric(metrics_df, "diversity", variant, metric, n),
            _theme_metric(metrics_df, "diversity", variant, metric, n, col=BASELINE_COL),
        ]
        bars = ax.bar(x, values, color=palette)
        ax.set_title(panel_title, fontsize=9)
        ax.set_xlabel(None)
        ax.tick_params(axis="x", labelbottom=False)
        _style_pct_axes(ax)
        ax.set_ylim(0, 1)
        if hline is not None:
            ax.axhline(hline, color="gray", linestyle="--", linewidth=1, alpha=0.6)

        for bar, value in zip(bars, values):
            if not np.isfinite(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                _metric_bar_label(value, as_pct=as_pct),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.text(
        0.13,
        0.97,
        f"Diversity ({variant})",
        ha="left",
        va="top",
        fontsize=14,
        fontweight="semibold",
    )
    legend_recommenders(axes[-1], loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=9)
    plt.tight_layout(pad=0.3)
    fig.subplots_adjust(left=0.13, right=0.86, bottom=0.14, top=0.80)
    plt.show()


def _plot_theme_bars(
    metrics_df: pd.DataFrame,
    *,
    theme: str,
    n: int,
    section_title: str,
    specs: list[tuple[str, str, str, bool, float | None]],
    figsize: tuple[float, float] = (7, 2.75),
    footnote: str | None = None,
) -> None:
    """Multi-panel embedding vs baseline bars for a theme at fixed n."""
    fig, axes = plt.subplots(1, len(specs), figsize=figsize)
    axes = np.atleast_1d(axes)
    palette = sns.color_palette(n_colors=2)
    x = np.arange(2)

    for ax, (variant, metric, panel_title, as_pct, hline) in zip(axes, specs):
        values = [
            _theme_metric(metrics_df, theme, variant, metric, n),
            _theme_metric(metrics_df, theme, variant, metric, n, col=BASELINE_COL),
        ]
        bars = ax.bar(x, values, color=palette)
        ax.set_title(panel_title, fontsize=9)
        ax.set_xlabel(None)
        ax.tick_params(axis="x", labelbottom=False)

        if as_pct:
            _style_pct_axes(ax)
            ax.set_ylim(0, 1)
        else:
            _style_ratio_axes(ax)
            finite = [v for v in values if np.isfinite(v)]
            ymax = max(finite + [hline or 0]) * 1.15 if finite else 1.2
            ax.set_ylim(0, ymax)

        if hline is not None:
            ax.axhline(hline, color="gray", linestyle="--", linewidth=1, alpha=0.6)

        for bar, value in zip(bars, values):
            if not np.isfinite(value):
                continue
            pad = 0.02 if as_pct else max(v for v in values if np.isfinite(v)) * 0.02
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + pad,
                _metric_bar_label(value, as_pct=as_pct),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.text(0.13, 0.97, section_title, ha="left", va="top", fontsize=14, fontweight="semibold")
    legend_recommenders(axes[-1], loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=9)
    if footnote:
        fig.text(0.13, 0.07, footnote, ha="left", va="bottom", fontsize=8)
    plt.tight_layout(pad=0.3)
    bottom = 0.14 if footnote else 0.14
    fig.subplots_adjust(left=0.13, right=0.86, bottom=bottom, top=0.80)
    plt.show()


def plot_repetition_bars(
    metrics_df: pd.DataFrame,
    *,
    n: int,
    variant: str,
    figsize: tuple[float, float] = (10, 2.75),
) -> None:
    """Hubness at fixed n: coverage, slot recycling, and head concentration."""
    _plot_theme_bars(
        metrics_df,
        theme="repetition",
        n=n,
        section_title=f"Repetition & hubness ({variant})",
        specs=[
            (variant, "unique_share", "(a) Global coverage", True, None),
            (variant, "hub_slot_share", "(b) Slot-level recycling", True, None),
            (variant, "top1pct_concentration", "(c) Head concentration", True, None),
        ],
        figsize=figsize,
    )


def plot_reciprocity_bars(
    metrics_df: pd.DataFrame,
    *,
    n: int,
    figsize: tuple[float, float] = (7, 2.75),
) -> None:
    """Reciprocal edge rate for albums and artists."""
    _plot_theme_bars(
        metrics_df,
        theme="reciprocity",
        n=n,
        section_title="Reciprocity",
        specs=[
            ("albums", "reciprocal_rate", "(a) Albums", True, None),
            ("artists", "reciprocal_rate", "(b) Artists", True, None),
        ],
        figsize=figsize,
    )


def plot_novelty_bars(
    metrics_df: pd.DataFrame,
    *,
    n: int,
    variant: str,
    figsize: tuple[float, float] = (7, 2.75),
) -> None:
    """Cross-list novelty at fixed n."""
    _plot_theme_bars(
        metrics_df,
        theme="novelty",
        n=n,
        section_title=f"Novelty ({variant})",
        specs=[
            (variant, "singleton_share", "(a) One-list recs", True, None),
            (variant, "mean_novelty", "(b) Normalized surprise", True, None),
        ],
        figsize=figsize,
    )


def plot_serendipity_bars(
    metrics_df: pd.DataFrame,
    *,
    n: int,
    figsize: tuple[float, float] = (10, 2.75),
) -> None:
    """Serendipity proxy: in-corpus, listener-comparable, and coverage rates."""
    _plot_theme_bars(
        metrics_df,
        theme="serendipity",
        n=n,
        section_title="Serendipity",
        specs=[
            ("reviews", "serendipitous_in_corpus_share", "(a) Serendipitous (in-corpus)", True, None),
            ("listeners", "serendipitous_listener_share", "(b) Serendipitous (listeners)", True, None),
            ("reviews", "in_corpus_share", "(c) In-corpus coverage", True, None),
        ],
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
