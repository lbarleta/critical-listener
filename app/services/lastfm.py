"""Thin wrapper around lastfm-recommender/lastfm_albums.py."""

from __future__ import annotations

import sys
from pathlib import Path

_LASTFM_DIR = Path(__file__).resolve().parents[2] / "lastfm-recommender"
if str(_LASTFM_DIR) not in sys.path:
    sys.path.insert(0, str(_LASTFM_DIR))

from lastfm_albums import recommend_album  # noqa: E402


def recommend(artist: str, album: str, k: int = 5) -> tuple[dict, list[dict]]:
    """
    Call Last.fm and return (seed, recommendations).

    Each recommendation: artist, album, rank, score (similarity when available).
    """
    seed, _seed_track, frame, _used_fallback = recommend_album(
        album=album,
        artist=artist,
        n_recs=k,
        track_selection="top_n_tracks",
        clear_cache=False,
    )

    recs: list[dict] = []
    if not frame.empty:
        for i, row in enumerate(frame.itertuples(index=False), start=1):
            score = getattr(row, "similarity_score", None)
            recs.append(
                {
                    "artist": row.artist,
                    "album": row.album,
                    "rank": i,
                    "score": float(score) if score is not None else None,
                }
            )

    return seed, recs
