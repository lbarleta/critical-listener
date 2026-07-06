"""Data loading helpers for lastfm_bench.ipynb."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import norm

BASELINE_COLUMNS = [
    "album_id",
    "query_artist",
    "query_album",
    "rec_artist",
    "rec_album",
    "score",
    "rank",
    "query_listeners",
    "rec_listeners",
    "rec_tags",
]


def get_catalog(path: Path | str) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Load albums.csv and return (catalog, reviews, ids, listeners) indexed by normalized key."""
    catalog = pd.read_csv(path)
    catalog["key"] = norm(catalog["artist"]) + "::" + norm(catalog["album"])
    keyed = catalog.set_index("key")
    catalog_reviews = keyed["review_count"]
    catalog_ids = keyed["album_id"]
    catalog_listeners = (
        keyed["lastfm_listeners"]
        if "lastfm_listeners" in catalog.columns
        else pd.Series(dtype="Float64")
    )
    return catalog, catalog_reviews, catalog_ids, catalog_listeners


def get_recs(path: Path | str) -> pd.DataFrame:
    """Load embedding recommendations (with Last.fm metadata joined in ingestion)."""
    recs = pd.read_parquet(path)
    recs = recs.drop(columns=["rec_length_flag"], errors="ignore")
    return recs.rename(
        columns={
            "query_lastfm_listeners": "query_listeners",
            "rec_lastfm_listeners": "rec_listeners",
            "query_lastfm_tags": "query_tags",
            "rec_lastfm_tags": "rec_tags",
        }
    )


def get_baseline(path: Path | str, verbose: bool = False) -> pd.DataFrame:
    """Load and normalize the Last.fm baseline recommendation export."""
    path = Path(path)
    df = pd.read_csv(path)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    for col in ("seed_listeners", "rec_listeners"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["status"] == "ok"].reset_index(drop=True)

    if verbose:
        print(f"File: {path.name}")
        print(
            f"Strategy: {df['strategy'].iloc[0]}  |  "
            f"Albums: {df['album_id'].nunique():,}  |  Rec rows: {len(df):,}"
        )

    baseline = (
        df.rename(
            columns={
                "artist": "query_artist",
                "album": "query_album",
                "seed_listeners": "query_listeners",
            }
        )
        .reindex(columns=BASELINE_COLUMNS)
        .copy()
    )
    baseline["rec_tags"] = baseline["rec_tags"].astype("string")
    return baseline
