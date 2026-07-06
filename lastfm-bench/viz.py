"""Seaborn plotting helpers for lastfm_bench.ipynb."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

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
