"""Benchmark metric calculations for lastfm_bench.ipynb."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from utils import (
    album_sets,
    album_target,
    artist_target,
    explode_rec_tags,
    filter_comparable_slots,
    hubness_from_counts,
    norm,
    query_key,
    rec_key,
)

EMBEDDING_COL = "embedding recs"
BASELINE_COL = "last.fm baseline"
MIN_COMPARABLE_TAG_SLOTS = 3

SKIP_METRICS = {
    "n_repeated",
    "repeated_share",
    "in_corpus_only",
    "min_in_corpus",
    "max_hub_degree",
    "median_hub_degree",
    "top10_concentration",
}


@dataclass
class BenchInputs:
    top_n: int
    catalog_reviews: pd.Series
    catalog_ids: pd.Series
    catalog_listeners: pd.Series
    recs_by_n: dict[int, pd.DataFrame]
    baseline_by_n: dict[int, pd.DataFrame]
    recs_tags_by_n: dict[int, pd.DataFrame]
    baseline_tags_by_n: dict[int, pd.DataFrame]
    tag_slots_by_n: dict[int, set]
    recs_listeners_by_n: dict[int, pd.DataFrame]
    baseline_listeners_by_n: dict[int, pd.DataFrame]
    listener_slots_by_n: dict[int, set]


def pivot_recommender(
    theme: str,
    emb: dict,
    base: dict,
    *,
    variant: str | None = None,
    n: int | None = None,
) -> pd.DataFrame:
    """One row per metric; embedding and baseline as columns."""
    keys = sorted((set(emb) | set(base)) - SKIP_METRICS)
    rows = []
    for metric in keys:
        if metric in {"metric", "recommender", "theme", "variant"}:
            continue
        row = {
            "theme": theme,
            "variant": variant,
            "metric": metric,
            EMBEDDING_COL: emb.get(metric),
            BASELINE_COL: base.get(metric),
        }
        if n is not None:
            row["n"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def pivot_agreement(stats: dict, *, variant: str) -> pd.DataFrame:
    n = stats["n"]
    rows = [
        {
            "theme": "agreement",
            "variant": variant,
            "metric": "precision",
            "n": n,
            EMBEDDING_COL: stats["precision"],
            BASELINE_COL: float("nan"),
        },
        {
            "theme": "agreement",
            "variant": variant,
            "metric": "recall",
            "n": n,
            EMBEDDING_COL: float("nan"),
            BASELINE_COL: stats["recall"],
        },
        {
            "theme": "agreement",
            "variant": variant,
            "metric": "hit_rate",
            "n": n,
            EMBEDDING_COL: stats["hit_rate"],
            BASELINE_COL: stats["hit_rate"],
        },
        {
            "theme": "agreement",
            "variant": variant,
            "metric": "mean_intersection",
            "n": n,
            EMBEDDING_COL: stats["mean_intersection"],
            BASELINE_COL: stats["mean_intersection"],
        },
        {
            "theme": "agreement",
            "variant": variant,
            "metric": "n_shared_queries",
            "n": n,
            EMBEDDING_COL: stats["n_shared_queries"],
            BASELINE_COL: stats["n_shared_queries"],
        },
    ]
    return pd.DataFrame(rows)


def pivot_pairwise(theme: str, stats: dict, *, variant: str, metrics: list[str]) -> pd.DataFrame:
    """Symmetric comparison metrics (same value in both model columns)."""
    rows = []
    for metric in metrics:
        if metric not in stats:
            continue
        value = stats[metric]
        rows.append(
            {
                "theme": theme,
                "variant": variant,
                "metric": metric,
                "n": stats.get("n"),
                EMBEDDING_COL: value,
                BASELINE_COL: value,
            }
        )
    return pd.DataFrame(rows)


# --- repetition / hubness ---


def repeated_stats(df: pd.DataFrame, target: Callable[[pd.DataFrame], pd.Series]) -> dict:
    counts = target(df).value_counts()
    stats = hubness_from_counts(counts)
    stats["n_total"] = len(df)
    stats["n_queries"] = df.groupby(["query_artist", "query_album"], observed=True).ngroups
    return stats


def repetition_metrics(
    recs_by_n: dict[int, pd.DataFrame],
    baseline_by_n: dict[int, pd.DataFrame],
    target_fn: Callable[[pd.DataFrame], pd.Series],
    *,
    variant: str,
) -> pd.DataFrame:
    parts = []
    for n in sorted(recs_by_n):
        emb = repeated_stats(recs_by_n[n], target_fn)
        base = repeated_stats(baseline_by_n[n], target_fn)
        parts.append(pivot_recommender("repetition", emb, base, variant=variant, n=n))
    return pd.concat(parts, ignore_index=True)


# --- stuckness ---


def stuckness_stats(df: pd.DataFrame) -> dict:
    q_art = norm(df["query_artist"])
    r_art = norm(df["rec_artist"])
    return {
        "n_slots": len(df),
        "n_queries": df.groupby(["query_artist", "query_album"], observed=True).ngroups,
        "same_artist_share": (q_art == r_art).mean(),
        "self_rec_share": (rec_key(df) == query_key(df)).mean(),
    }


def stuckness_metrics(recs_n: pd.DataFrame, baseline_n: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    return pivot_recommender(
        "stuckness",
        stuckness_stats(recs_n),
        stuckness_stats(baseline_n),
        n=top_n,
    )


# --- agreement ---


def agreement_stats(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    n: int,
    catalog_ids: pd.Series,
    in_corpus_only: bool = False,
) -> dict:
    sets_a = album_sets(df_a, catalog_ids, in_corpus_only=in_corpus_only)
    sets_b = album_sets(df_b, catalog_ids, in_corpus_only=in_corpus_only)
    shared = sorted(set(sets_a) & set(sets_b))
    if not shared:
        return {
            "n": n,
            "n_shared_queries": 0,
            "precision": float("nan"),
            "recall": float("nan"),
            "hit_rate": float("nan"),
            "mean_intersection": float("nan"),
        }
    prec, rec, hits, inter = [], [], [], []
    for qk in shared:
        a, b = sets_a[qk], sets_b[qk]
        inter.append(len(a & b))
        prec.append(len(a & b) / len(a) if a else float("nan"))
        rec.append(len(a & b) / len(b) if b else float("nan"))
        hits.append(1.0 if a & b else 0.0)
    return {
        "n": n,
        "n_shared_queries": len(shared),
        "precision": float(pd.Series(prec).mean()),
        "recall": float(pd.Series(rec).mean()),
        "hit_rate": float(pd.Series(hits).mean()),
        "mean_intersection": float(pd.Series(inter).mean()),
    }


def agreement_metrics(
    recs_by_n: dict[int, pd.DataFrame],
    baseline_by_n: dict[int, pd.DataFrame],
    catalog_ids: pd.Series,
) -> pd.DataFrame:
    parts = []
    for in_corpus, variant in ((False, "full"), (True, "in_corpus")):
        for n in sorted(recs_by_n):
            stats = agreement_stats(
                recs_by_n[n],
                baseline_by_n[n],
                n,
                catalog_ids,
                in_corpus_only=in_corpus,
            )
            parts.append(pivot_agreement(stats, variant=variant))
    return pd.concat(parts, ignore_index=True)


# --- diversity ---


def list_diversity_stats(df: pd.DataFrame, kind: str) -> dict:
    keys = ["query_artist", "query_album"]
    if kind == "artist":
        target = norm(df["rec_artist"])
    elif kind == "album":
        target = rec_key(df)
    else:
        raise ValueError(f"unknown kind: {kind}")
    stats = (
        df.assign(_target=target)
        .groupby(keys, observed=True)["_target"]
        .agg(nunique="nunique", count="count")
    )
    ratio = stats["nunique"] / stats["count"]
    return {
        "n_queries": len(stats),
        "mean_unique": float(stats["nunique"].mean()),
        "mean_diversity_ratio": float(ratio.mean()),
        "all_distinct_share": float((stats["nunique"] == stats["count"]).mean()),
        "single_target_share": float((stats["nunique"] == 1).mean()),
    }


def tag_diversity_stats(df: pd.DataFrame, n_comparable_slots: int) -> dict:
    keys = ["query_artist", "query_album"]
    exploded = explode_rec_tags(df)
    n_queries = df.groupby(keys, observed=True).ngroups
    if exploded.empty:
        return {
            "n_comparable_slots": n_comparable_slots,
            "n_queries": n_queries,
            "queries_with_tags": 0,
            "queries_with_tags_share": 0.0,
            "mean_unique": float("nan"),
            "mean_diversity_ratio": float("nan"),
            "all_distinct_share": float("nan"),
            "single_target_share": float("nan"),
        }
    stats = exploded.groupby(keys, observed=True).agg(
        n_mentions=("rec_tag", "size"),
        n_unique=("rec_tag", "nunique"),
    )
    stats["ratio"] = stats["n_unique"] / stats["n_mentions"]
    return {
        "n_comparable_slots": n_comparable_slots,
        "n_queries": n_queries,
        "queries_with_tags": len(stats),
        "queries_with_tags_share": len(stats) / n_queries if n_queries else float("nan"),
        "mean_unique": float(stats["n_unique"].mean()),
        "mean_diversity_ratio": float(stats["ratio"].mean()),
        "all_distinct_share": float((stats["n_unique"] == stats["n_mentions"]).mean()),
        "single_target_share": float((stats["n_unique"] == 1).mean()),
    }


def diversity_metrics(
    recs_n: pd.DataFrame,
    baseline_n: pd.DataFrame,
    recs_tags_n: pd.DataFrame,
    baseline_tags_n: pd.DataFrame,
    *,
    top_n: int,
    tag_slots_n: set,
) -> pd.DataFrame:
    artist = pivot_recommender(
        "diversity",
        list_diversity_stats(recs_n, "artist"),
        list_diversity_stats(baseline_n, "artist"),
        variant="artists",
        n=top_n,
    )
    tags = pivot_recommender(
        "diversity",
        tag_diversity_stats(recs_tags_n, len(tag_slots_n)),
        tag_diversity_stats(baseline_tags_n, len(tag_slots_n)),
        variant="tags",
        n=top_n,
    )
    return pd.concat([artist, tags], ignore_index=True)


# --- reciprocity ---


def reciprocity_stats(edges: set) -> dict:
    reciprocal = sum(1 for a, b in edges if (b, a) in edges)
    n_edges = len(edges)
    return {
        "n_edges": n_edges,
        "n_reciprocal": reciprocal,
        "reciprocal_rate": reciprocal / n_edges if n_edges else float("nan"),
    }


def album_reciprocity_stats(df: pd.DataFrame) -> dict:
    edges = set(
        zip(
            norm(df["query_artist"]) + "::" + norm(df["query_album"]),
            norm(df["rec_artist"]) + "::" + norm(df["rec_album"]),
        )
    )
    return reciprocity_stats(edges)


def artist_reciprocity_stats(df: pd.DataFrame) -> dict:
    edges = set(zip(norm(df["query_artist"]), norm(df["rec_artist"])))
    return reciprocity_stats(edges)


def reciprocity_metrics(recs_n: pd.DataFrame, baseline_n: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    albums = pivot_recommender(
        "reciprocity",
        album_reciprocity_stats(recs_n),
        album_reciprocity_stats(baseline_n),
        variant="albums",
        n=top_n,
    )
    artists = pivot_recommender(
        "reciprocity",
        artist_reciprocity_stats(recs_n),
        artist_reciprocity_stats(baseline_n),
        variant="artists",
        n=top_n,
    )
    return pd.concat([albums, artists], ignore_index=True)


# --- novelty ---


def novelty_stats(df: pd.DataFrame, target: Callable[[pd.DataFrame], pd.Series] = rec_key) -> dict:
    n_lists = df.groupby(["query_artist", "query_album"], observed=True).ngroups
    key = target(df)
    reach = key.map(key.value_counts())
    # Self-information per slot: log2(n_lists / reach) bits.
    # reach=1 (singleton) -> log2(n_lists); reach=n_lists (every list) -> 0.
    info = np.log2(n_lists / reach)
    log_n = np.log2(n_lists)
    # Normalize to 0-1 by dividing by the singleton ceiling log2(n_lists).
    # Same ranking as raw self-information, but comparable across runs with different n_lists.
    norm_info = info / log_n if log_n else reach * 0
    counts = key.value_counts()
    p = counts / counts.sum()
    return {
        "n_lists": n_lists,
        "mean_reach": reach.mean(),
        "median_reach": reach.median(),
        "singleton_share": (reach == 1).mean(),
        "mean_self_information": info.mean(),
        "median_self_information": info.median(),
        "mean_novelty": norm_info.mean(),
        "median_novelty": norm_info.median(),
        "effective_catalog": 2 ** (-(p * np.log2(p)).sum()),
    }


def novelty_metrics(recs_n: pd.DataFrame, baseline_n: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    albums = pivot_recommender(
        "novelty",
        novelty_stats(recs_n, album_target),
        novelty_stats(baseline_n, album_target),
        variant="albums",
        n=top_n,
    )
    artists = pivot_recommender(
        "novelty",
        novelty_stats(recs_n, artist_target),
        novelty_stats(baseline_n, artist_target),
        variant="artists",
        n=top_n,
    )
    return pd.concat([albums, artists], ignore_index=True)


# --- popularity bias ---


def review_count_bias_stats(df: pd.DataFrame, catalog_reviews: pd.Series) -> dict:
    rec_reviews = rec_key(df).map(catalog_reviews)
    in_corpus = rec_reviews.notna()
    rec_reviews = rec_reviews[in_corpus]
    catalog_mean = catalog_reviews.mean()
    return {
        "in_corpus_rec_share": in_corpus.mean(),
        "mean_reviews_recs": rec_reviews.mean(),
        "mean_reviews_catalog": catalog_mean,
        "mean_reviews_ratio": rec_reviews.mean() / catalog_mean,
        "recs_3plus_share": (rec_reviews >= 3).mean(),
        "catalog_3plus_share": (catalog_reviews >= 3).mean(),
    }


def listener_popularity_stats(
    df: pd.DataFrame,
    n_comparable_slots: int,
    catalog_listeners: pd.Series,
) -> dict:
    catalog_mean = catalog_listeners.dropna().mean()
    catalog_coverage = catalog_listeners.notna().mean()
    sub = df.dropna(subset=["rec_listeners", "query_listeners"])
    if sub.empty:
        return {
            "n_comparable_slots": n_comparable_slots,
            "n_rows_with_data": 0,
            "mean_listeners_catalog": catalog_mean,
            "catalog_listener_coverage": catalog_coverage,
            "mean_rec_listeners": float("nan"),
            "mean_query_listeners": float("nan"),
            "mean_rec_query_ratio": float("nan"),
            "mean_rec_catalog_ratio": float("nan"),
            "rec_more_popular_share": float("nan"),
        }
    ratio = (sub["rec_listeners"] + 1) / (sub["query_listeners"] + 1)
    mean_rec = sub["rec_listeners"].mean()
    return {
        "n_comparable_slots": n_comparable_slots,
        "n_rows_with_data": len(sub),
        "mean_listeners_catalog": catalog_mean,
        "catalog_listener_coverage": catalog_coverage,
        "mean_rec_listeners": mean_rec,
        "mean_query_listeners": sub["query_listeners"].mean(),
        "mean_rec_query_ratio": ratio.mean(),
        "mean_rec_catalog_ratio": mean_rec / catalog_mean if catalog_mean else float("nan"),
        "rec_more_popular_share": (ratio > 1).mean(),
    }


def popularity_metrics(
    recs_n: pd.DataFrame,
    baseline_n: pd.DataFrame,
    recs_listeners_n: pd.DataFrame,
    baseline_listeners_n: pd.DataFrame,
    catalog_reviews: pd.Series,
    catalog_listeners: pd.Series,
    *,
    top_n: int,
    listener_slots_n: set,
) -> pd.DataFrame:
    reviews = pivot_recommender(
        "popularity",
        review_count_bias_stats(recs_n, catalog_reviews),
        review_count_bias_stats(baseline_n, catalog_reviews),
        variant="reviews",
        n=top_n,
    )
    listeners = pivot_recommender(
        "popularity",
        listener_popularity_stats(recs_listeners_n, len(listener_slots_n), catalog_listeners),
        listener_popularity_stats(baseline_listeners_n, len(listener_slots_n), catalog_listeners),
        variant="listeners",
        n=top_n,
    )
    return pd.concat([reviews, listeners], ignore_index=True)


# --- disagreement ---


def target_lists(df: pd.DataFrame, target: Callable[[pd.DataFrame], pd.Series]) -> pd.Series:
    frame = pd.DataFrame({"query": query_key(df), "rec": target(df)})
    return frame.groupby("query", observed=True)["rec"].agg(set)


def rec_lists(df: pd.DataFrame) -> pd.Series:
    return target_lists(df, rec_key)


def artist_lists(df: pd.DataFrame) -> pd.Series:
    return target_lists(df, artist_target)


def full_jaccard_disagreement_stats(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    n: int,
    list_fn: Callable[[pd.DataFrame], pd.Series],
) -> dict:
    lists_a, lists_b = list_fn(df_a), list_fn(df_b)
    shared = lists_a.index.intersection(lists_b.index)
    nan_metrics = {
        "mean_jaccard": float("nan"),
        "zero_overlap_share": float("nan"),
        "any_overlap_share": float("nan"),
    }
    if shared.empty:
        return {"n": n, "n_shared_queries": 0, **nan_metrics}

    j_full = jaccard(lists_a, lists_b, shared)
    return {
        "n": n,
        "n_shared_queries": len(shared),
        "mean_jaccard": j_full.mean(),
        "zero_overlap_share": (j_full == 0).mean(),
        "any_overlap_share": (j_full > 0).mean(),
    }


def jaccard(lists_a: pd.Series, lists_b: pd.Series, queries) -> pd.Series:
    values = []
    for q in queries:
        union = lists_a[q] | lists_b[q]
        values.append(len(lists_a[q] & lists_b[q]) / len(union) if union else float("nan"))
    return pd.Series(values, index=queries)


def disagreement_stats(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    n: int,
    catalog_reviews: pd.Series,
    min_in_corpus: int | None = None,
) -> dict:
    if min_in_corpus is None:
        min_in_corpus = n
    lists_a, lists_b = rec_lists(df_a), rec_lists(df_b)
    shared = lists_a.index.intersection(lists_b.index)
    nan_metrics = {
        "mean_jaccard": float("nan"),
        "zero_overlap_share": float("nan"),
        "any_overlap_share": float("nan"),
        "n_comparable_queries": 0,
        "comparable_query_share": float("nan"),
        "mean_jaccard_in_corpus": float("nan"),
        "zero_overlap_in_corpus_share": float("nan"),
        "any_overlap_in_corpus_share": float("nan"),
    }
    if shared.empty:
        return {"n": n, "n_shared_queries": 0, **nan_metrics}

    j_full = jaccard(lists_a, lists_b, shared)
    catalog_keys = set(catalog_reviews.index)
    in_a = lists_a.map(lambda s: s & catalog_keys)
    in_b = lists_b.map(lambda s: s & catalog_keys)
    comparable = [q for q in shared if len(in_a[q]) >= min_in_corpus and len(in_b[q]) >= min_in_corpus]
    out = {
        "n": n,
        "n_shared_queries": len(shared),
        "mean_jaccard": j_full.mean(),
        "zero_overlap_share": (j_full == 0).mean(),
        "any_overlap_share": (j_full > 0).mean(),
        "n_comparable_queries": len(comparable),
        "comparable_query_share": len(comparable) / len(shared),
    }
    if comparable:
        j_inc = jaccard(in_a, in_b, comparable)
        out.update(
            {
                "mean_jaccard_in_corpus": j_inc.mean(),
                "zero_overlap_in_corpus_share": (j_inc == 0).mean(),
                "any_overlap_in_corpus_share": (j_inc > 0).mean(),
            }
        )
    else:
        out.update(
            {
                "mean_jaccard_in_corpus": float("nan"),
                "zero_overlap_in_corpus_share": float("nan"),
                "any_overlap_in_corpus_share": float("nan"),
            }
        )
    return out


def tag_lists(df: pd.DataFrame) -> pd.Series:
    exploded = explode_rec_tags(df)
    if exploded.empty:
        return pd.Series(dtype=object)
    frame = exploded.assign(query=query_key(exploded))
    return frame.groupby("query")["rec_tag"].agg(set)


def comparable_tag_queries(
    recs_n,
    baseline_n,
    tag_slots,
    *,
    min_slots: int = MIN_COMPARABLE_TAG_SLOTS,
):
    comp_a = filter_comparable_slots(recs_n, tag_slots)
    comp_b = filter_comparable_slots(baseline_n, tag_slots)
    counts_a = comp_a.assign(query=query_key(comp_a)).groupby("query", observed=True).size()
    counts_b = comp_b.assign(query=query_key(comp_b)).groupby("query", observed=True).size()
    shared = counts_a.index.intersection(counts_b.index)
    return [q for q in shared if counts_a[q] >= min_slots and counts_b[q] >= min_slots]


def tag_disagreement_stats(recs_n, baseline_n, tag_slots, n) -> dict:
    comp_a = filter_comparable_slots(recs_n, tag_slots)
    comp_b = filter_comparable_slots(baseline_n, tag_slots)
    lists_a, lists_b = tag_lists(recs_n), tag_lists(baseline_n)
    shared = lists_a.index.intersection(lists_b.index)
    nan_metrics = {
        "mean_jaccard": float("nan"),
        "zero_overlap_share": float("nan"),
        "any_overlap_share": float("nan"),
        "n_comparable_queries": 0,
        "comparable_query_share": float("nan"),
        "mean_jaccard_comparable": float("nan"),
        "zero_overlap_comparable_share": float("nan"),
        "any_overlap_comparable_share": float("nan"),
    }
    if shared.empty:
        return {"n": n, "n_shared_queries": 0, **nan_metrics}

    j_full = jaccard(lists_a, lists_b, shared).dropna()
    comparable = comparable_tag_queries(recs_n, baseline_n, tag_slots)
    comp_lists_a, comp_lists_b = tag_lists(comp_a), tag_lists(comp_b)
    out = {
        "n": n,
        "n_shared_queries": len(shared),
        "mean_jaccard": j_full.mean() if len(j_full) else float("nan"),
        "zero_overlap_share": (j_full == 0).mean() if len(j_full) else float("nan"),
        "any_overlap_share": (j_full > 0).mean() if len(j_full) else float("nan"),
        "n_comparable_queries": len(comparable),
        "comparable_query_share": len(comparable) / len(shared) if len(shared) else float("nan"),
    }
    if comparable:
        j_comp = jaccard(comp_lists_a, comp_lists_b, comparable).dropna()
        out.update(
            {
                "mean_jaccard_comparable": j_comp.mean() if len(j_comp) else float("nan"),
                "zero_overlap_comparable_share": (j_comp == 0).mean() if len(j_comp) else float("nan"),
                "any_overlap_comparable_share": (j_comp > 0).mean() if len(j_comp) else float("nan"),
            }
        )
    else:
        out.update(
            {
                "mean_jaccard_comparable": float("nan"),
                "zero_overlap_comparable_share": float("nan"),
                "any_overlap_comparable_share": float("nan"),
            }
        )
    return out


def disagreement_metrics(
    recs_by_n: dict[int, pd.DataFrame],
    baseline_by_n: dict[int, pd.DataFrame],
    catalog_reviews: pd.Series,
) -> pd.DataFrame:
    full_metrics = [
        "n_shared_queries",
        "mean_jaccard",
        "zero_overlap_share",
        "any_overlap_share",
        "n_comparable_queries",
        "comparable_query_share",
    ]
    inc_metrics = [
        "mean_jaccard_in_corpus",
        "zero_overlap_in_corpus_share",
        "any_overlap_in_corpus_share",
    ]
    artist_metrics = [
        "n_shared_queries",
        "mean_jaccard",
        "zero_overlap_share",
        "any_overlap_share",
    ]
    parts = []
    for n in sorted(recs_by_n):
        recs_n = recs_by_n[n]
        baseline_n = baseline_by_n[n]
        stats = disagreement_stats(recs_n, baseline_n, n, catalog_reviews)
        parts.append(pivot_pairwise("disagreement", stats, variant="albums", metrics=full_metrics))
        parts.append(pivot_pairwise("disagreement", stats, variant="albums_in_corpus", metrics=inc_metrics))
        artist_stats = full_jaccard_disagreement_stats(recs_n, baseline_n, n, artist_lists)
        parts.append(pivot_pairwise("disagreement", artist_stats, variant="artists", metrics=artist_metrics))
    return pd.concat(parts, ignore_index=True)


def tag_disagreement_metrics(
    recs_by_n: dict[int, pd.DataFrame],
    baseline_by_n: dict[int, pd.DataFrame],
    tag_slots_by_n: dict[int, set],
) -> pd.DataFrame:
    full_metrics = [
        "n_shared_queries",
        "mean_jaccard",
        "zero_overlap_share",
        "any_overlap_share",
        "n_comparable_queries",
        "comparable_query_share",
    ]
    comp_metrics = [
        "mean_jaccard_comparable",
        "zero_overlap_comparable_share",
        "any_overlap_comparable_share",
    ]
    parts = []
    for n in sorted(recs_by_n):
        stats = tag_disagreement_stats(recs_by_n[n], baseline_by_n[n], tag_slots_by_n[n], n)
        parts.append(pivot_pairwise("disagreement", stats, variant="tags", metrics=full_metrics))
        parts.append(pivot_pairwise("disagreement", stats, variant="tags_comparable", metrics=comp_metrics))
    return pd.concat(parts, ignore_index=True)


# --- serendipity ---


def serendipity_lite_stats(df: pd.DataFrame, catalog_reviews: pd.Series) -> dict:
    same_artist = norm(df["rec_artist"]) == norm(df["query_artist"])
    rec_reviews = rec_key(df).map(catalog_reviews)
    median_reviews = catalog_reviews.median()
    in_corpus = rec_reviews.notna()
    obscure = rec_reviews <= median_reviews
    serendipitous = ~same_artist & obscure
    return {
        "catalog_median_review_count": median_reviews,
        "in_corpus_share": in_corpus.mean(),
        "serendipitous_share": serendipitous.mean(),
        "serendipitous_in_corpus_share": serendipitous[in_corpus].mean() if in_corpus.any() else float("nan"),
    }


def serendipity_listener_stats(df: pd.DataFrame, n_comparable_slots: int, catalog_listeners: pd.Series) -> dict:
    same_artist = norm(df["rec_artist"]) == norm(df["query_artist"])
    median_listeners = catalog_listeners.median()
    obscure = df["rec_listeners"] <= median_listeners
    serendipitous = ~same_artist & obscure
    return {
        "catalog_median_listeners": median_listeners,
        "n_comparable_slots": n_comparable_slots,
        "serendipitous_listener_share": serendipitous.mean(),
    }


def serendipity_metrics(
    recs_n: pd.DataFrame,
    baseline_n: pd.DataFrame,
    recs_listeners_n: pd.DataFrame,
    baseline_listeners_n: pd.DataFrame,
    catalog_reviews: pd.Series,
    catalog_listeners: pd.Series,
    *,
    top_n: int,
    listener_slots_n: set,
) -> pd.DataFrame:
    reviews = pivot_recommender(
        "serendipity",
        serendipity_lite_stats(recs_n, catalog_reviews),
        serendipity_lite_stats(baseline_n, catalog_reviews),
        variant="reviews",
        n=top_n,
    )
    listeners = pivot_recommender(
        "serendipity",
        serendipity_listener_stats(recs_listeners_n, len(listener_slots_n), catalog_listeners),
        serendipity_listener_stats(baseline_listeners_n, len(listener_slots_n), catalog_listeners),
        variant="listeners",
        n=top_n,
    )
    return pd.concat([reviews, listeners], ignore_index=True)


# --- aggregate ---


def compute_benchmark_metrics(inputs: BenchInputs) -> pd.DataFrame:
    top_n = inputs.top_n
    recs_n = inputs.recs_by_n[top_n]
    baseline_n = inputs.baseline_by_n[top_n]

    parts = [
        repetition_metrics(inputs.recs_by_n, inputs.baseline_by_n, album_target, variant="albums"),
        repetition_metrics(inputs.recs_by_n, inputs.baseline_by_n, artist_target, variant="artists"),
        _repetition_tag_metrics(inputs),
        stuckness_metrics(recs_n, baseline_n, top_n=top_n),
        agreement_metrics(inputs.recs_by_n, inputs.baseline_by_n, inputs.catalog_ids),
        diversity_metrics(
            recs_n,
            baseline_n,
            inputs.recs_tags_by_n[top_n],
            inputs.baseline_tags_by_n[top_n],
            top_n=top_n,
            tag_slots_n=inputs.tag_slots_by_n[top_n],
        ),
        reciprocity_metrics(recs_n, baseline_n, top_n=top_n),
        novelty_metrics(recs_n, baseline_n, top_n=top_n),
        popularity_metrics(
            recs_n,
            baseline_n,
            inputs.recs_listeners_by_n[top_n],
            inputs.baseline_listeners_by_n[top_n],
            inputs.catalog_reviews,
            inputs.catalog_listeners,
            top_n=top_n,
            listener_slots_n=inputs.listener_slots_by_n[top_n],
        ),
        disagreement_metrics(inputs.recs_by_n, inputs.baseline_by_n, inputs.catalog_reviews),
        tag_disagreement_metrics(inputs.recs_by_n, inputs.baseline_by_n, inputs.tag_slots_by_n),
        serendipity_metrics(
            recs_n,
            baseline_n,
            inputs.recs_listeners_by_n[top_n],
            inputs.baseline_listeners_by_n[top_n],
            inputs.catalog_reviews,
            inputs.catalog_listeners,
            top_n=top_n,
            listener_slots_n=inputs.listener_slots_by_n[top_n],
        ),
    ]

    out = pd.concat(parts, ignore_index=True)
    return out[["theme", "variant", "metric", "n", EMBEDDING_COL, BASELINE_COL]]


def _repeated_tag_stats(df: pd.DataFrame, n_comparable_slots: int) -> dict:
    exploded = explode_rec_tags(df)
    n_queries = df.groupby(["query_artist", "query_album"], observed=True).ngroups
    if exploded.empty:
        return {
            "n_comparable_slots": n_comparable_slots,
            "n_total": 0,
            "n_queries": n_queries,
            "n_unique": 0,
            "n_hubs": 0,
            "unique_share": float("nan"),
            "hub_share": float("nan"),
            "hub_slot_share": float("nan"),
            "gini": float("nan"),
        }
    counts = exploded["rec_tag"].value_counts()
    stats = hubness_from_counts(counts)
    stats["n_comparable_slots"] = n_comparable_slots
    stats["n_total"] = int(counts.sum())
    stats["n_queries"] = n_queries
    return stats


def _repetition_tag_metrics(inputs: BenchInputs) -> pd.DataFrame:
    parts = []
    for n in sorted(inputs.recs_tags_by_n):
        emb = _repeated_tag_stats(inputs.recs_tags_by_n[n], len(inputs.tag_slots_by_n[n]))
        base = _repeated_tag_stats(inputs.baseline_tags_by_n[n], len(inputs.tag_slots_by_n[n]))
        parts.append(pivot_recommender("repetition", emb, base, variant="tags", n=n))
    return pd.concat(parts, ignore_index=True)
