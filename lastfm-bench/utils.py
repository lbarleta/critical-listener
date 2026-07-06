"""Shared helpers for lastfm_bench.ipynb."""

from __future__ import annotations

import re
import unicodedata
from typing import Callable

import numpy as np
import pandas as pd

QUERY_KEYS = ["query_artist", "query_album"]


def top_n_subset(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Keep ranks 1..top_n and drop query albums with fewer than top_n recs."""
    sub = df[df["rank"] <= top_n].copy()
    n_recs = sub.groupby(QUERY_KEYS, observed=True)["rank"].transform("count")
    return sub[n_recs >= top_n].reset_index(drop=True)


def keep_top_n(df: pd.DataFrame, top_n: int, verbose: bool = True) -> pd.DataFrame:
    n_queries_before = df[df["rank"] <= top_n].groupby(QUERY_KEYS, observed=True).ngroups
    out = top_n_subset(df, top_n)
    if verbose:
        n_queries_after = out.groupby(QUERY_KEYS, observed=True).ngroups
        print(f"Kept {n_queries_after:,} / {n_queries_before:,} query albums with >={top_n} recs")
    return out


def norm_str(s) -> str:
    """Same normalization as ingestion/albums.ipynb (quotes, accents, whitespace)."""
    s = str(s).strip().lower()
    for ch in ("\u2019", "\u2018", "\u201b", "\u2032", "`", "\u00b4"):
        s = s.replace(ch, "'")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s)


def norm(s: pd.Series) -> pd.Series:
    return s.fillna("").map(norm_str)


def query_key(df: pd.DataFrame) -> pd.Series:
    return norm(df["query_artist"]) + "::" + norm(df["query_album"])


def rec_key(df: pd.DataFrame) -> pd.Series:
    return norm(df["rec_artist"]) + "::" + norm(df["rec_album"])


def album_target(df: pd.DataFrame) -> pd.Series:
    return rec_key(df)


def artist_target(df: pd.DataFrame) -> pd.Series:
    return norm(df["rec_artist"])


def has_listener_data(df: pd.DataFrame) -> pd.Series:
    return df["rec_listeners"].notna() & df["query_listeners"].notna()


def has_rec_tags(df: pd.DataFrame) -> pd.Series:
    tags = df["rec_tags"].astype("string")
    return tags.notna() & tags.str.strip().ne("")


def _slot_set(df: pd.DataFrame, mask: Callable[[pd.DataFrame], pd.Series]) -> set:
    sub = df.loc[mask(df)]
    if sub.empty:
        return set()
    return set(zip(query_key(sub), sub["rank"]))


def comparable_slots(
    recs_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    mask_fn: Callable[[pd.DataFrame], pd.Series],
) -> set:
    return _slot_set(recs_df, mask_fn) & _slot_set(baseline_df, mask_fn)


def filter_comparable_slots(df: pd.DataFrame, slots: set) -> pd.DataFrame:
    if not slots:
        return df.iloc[0:0].copy()
    idx = pd.MultiIndex.from_arrays([query_key(df), df["rank"]])
    return df.loc[idx.isin(slots)].copy()


def filter_to_baseline(df: pd.DataFrame, baseline_query_keys: set[str]) -> pd.DataFrame:
    out = df.loc[query_key(df).isin(baseline_query_keys)].copy()
    out["_query_key"] = query_key(out)
    canonical = (
        out.groupby("_query_key", observed=True)[["query_artist", "query_album"]]
        .first()
        .rename(columns={"query_artist": "_canon_artist", "query_album": "_canon_album"})
    )
    out = out.join(canonical, on="_query_key")
    out = out[
        (out["query_artist"] == out["_canon_artist"])
        & (out["query_album"] == out["_canon_album"])
    ].drop(columns=["_query_key", "_canon_artist", "_canon_album"]).reset_index(drop=True)
    n_queries = out.groupby(QUERY_KEYS, observed=True).ngroups
    print(f"{len(out):,} rows, {n_queries:,} queries")
    return out


def build_subsets(recs_matched: pd.DataFrame, baseline: pd.DataFrame, top_n: int) -> dict:
    """Precompute top-n views and comparable listener/tag subsets for n = 1..top_n."""
    recs_by_n: dict[int, pd.DataFrame] = {}
    baseline_by_n: dict[int, pd.DataFrame] = {}
    listener_slots_by_n: dict[int, set] = {}
    tag_slots_by_n: dict[int, set] = {}
    recs_listeners_by_n: dict[int, pd.DataFrame] = {}
    baseline_listeners_by_n: dict[int, pd.DataFrame] = {}
    recs_tags_by_n: dict[int, pd.DataFrame] = {}
    baseline_tags_by_n: dict[int, pd.DataFrame] = {}
    rec_sources_by_n: dict[int, list] = {}
    rec_sources_listeners_by_n: dict[int, list] = {}
    rec_sources_tags_by_n: dict[int, list] = {}

    for n in range(1, top_n + 1):
        recs_n = keep_top_n(recs_matched, n, verbose=False)
        base_n = keep_top_n(baseline, n, verbose=False)
        recs_by_n[n] = recs_n
        baseline_by_n[n] = base_n

        listener_slots = comparable_slots(recs_n, base_n, has_listener_data)
        tag_slots = comparable_slots(recs_n, base_n, has_rec_tags)
        listener_slots_by_n[n] = listener_slots
        tag_slots_by_n[n] = tag_slots

        recs_listeners_by_n[n] = filter_comparable_slots(recs_n, listener_slots)
        baseline_listeners_by_n[n] = filter_comparable_slots(base_n, listener_slots)
        recs_tags_by_n[n] = filter_comparable_slots(recs_n, tag_slots)
        baseline_tags_by_n[n] = filter_comparable_slots(base_n, tag_slots)

        rec_sources_by_n[n] = [("embedding recs", recs_n), ("last.fm baseline", base_n)]
        rec_sources_listeners_by_n[n] = [
            ("embedding recs", recs_listeners_by_n[n]),
            ("last.fm baseline", baseline_listeners_by_n[n]),
        ]
        rec_sources_tags_by_n[n] = [
            ("embedding recs", recs_tags_by_n[n]),
            ("last.fm baseline", baseline_tags_by_n[n]),
        ]

    emb_n = recs_by_n[top_n]
    print(f"Subsets for n = 1..{top_n}:")
    print(f"  Embedding @ n={top_n}: {len(emb_n):,} rows, {emb_n.groupby(QUERY_KEYS).ngroups:,} queries")
    print(
        f"  Baseline @ n={top_n}:  {len(baseline_by_n[top_n]):,} rows, "
        f"{baseline_by_n[top_n].groupby(QUERY_KEYS).ngroups:,} queries"
    )
    print(
        f"  Comparable listener slots @ n={top_n}: {len(listener_slots_by_n[top_n]):,} "
        f"({len(listener_slots_by_n[top_n]) / len(emb_n):.1%} of emb rows)"
    )
    print(
        f"  Comparable tag slots @ n={top_n}:      {len(tag_slots_by_n[top_n]):,} "
        f"({len(tag_slots_by_n[top_n]) / len(emb_n):.1%} of emb rows)"
    )

    return {
        "recs_by_n": recs_by_n,
        "baseline_by_n": baseline_by_n,
        "listener_slots_by_n": listener_slots_by_n,
        "tag_slots_by_n": tag_slots_by_n,
        "recs_listeners_by_n": recs_listeners_by_n,
        "baseline_listeners_by_n": baseline_listeners_by_n,
        "recs_tags_by_n": recs_tags_by_n,
        "baseline_tags_by_n": baseline_tags_by_n,
        "rec_sources_by_n": rec_sources_by_n,
        "rec_sources_listeners_by_n": rec_sources_listeners_by_n,
        "rec_sources_tags_by_n": rec_sources_tags_by_n,
    }


def parse_tag_string(val) -> list[str]:
    if pd.isna(val):
        return []
    return [tag.strip().lower() for tag in str(val).split(";") if tag.strip()]


def explode_rec_tags(df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    base_cols = [c for c in df.columns if c != "rec_tag"]
    for row in df[base_cols].itertuples(index=False):
        row_dict = dict(zip(base_cols, row))
        for tag in parse_tag_string(row_dict.get("rec_tags")):
            pieces.append({**row_dict, "rec_tag": tag})
    if not pieces:
        return pd.DataFrame(columns=[*base_cols, "rec_tag"])
    return pd.DataFrame(pieces)


def gini(counts) -> float:
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    cumx = np.cumsum(x)
    return float((n + 1 - 2 * (cumx / cumx[-1]).sum()) / n)


def hubness_from_counts(counts: pd.Series) -> dict:
    n_total = int(counts.sum())
    n_unique = len(counts)
    hub_counts = counts[counts > 1]
    n_hubs = len(hub_counts)
    hub_slots = int(hub_counts.sum()) if n_hubs else 0
    sorted_counts = counts.sort_values(ascending=False)
    top1pct_n = max(1, int(np.ceil(n_unique * 0.01)))
    top1pct_slots = int(sorted_counts.head(top1pct_n).sum())
    return {
        "n_unique": n_unique,
        "n_hubs": n_hubs,
        "hub_share": n_hubs / n_unique if n_unique else float("nan"),
        "hub_slot_share": hub_slots / n_total if n_total else float("nan"),
        "unique_share": n_unique / n_total if n_total else float("nan"),
        "gini": gini(counts.values),
        "top1pct_concentration": top1pct_slots / n_total if n_total else float("nan"),
    }


def album_sets(df: pd.DataFrame, catalog_ids: pd.Series, in_corpus_only: bool = False) -> dict:
    sub = df
    if in_corpus_only:
        sub = sub[rec_key(sub).map(catalog_ids).notna()]
    out = {}
    qkeys = query_key(sub)
    for qk, grp in sub.groupby(qkeys, observed=True):
        out[qk] = set(rec_key(grp))
    return out


def top_hubs(df: pd.DataFrame, target: Callable[[pd.DataFrame], pd.Series], k: int = 10) -> pd.Series:
    return target(df).value_counts().head(k).rename("query_count")
