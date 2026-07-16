"""Precomputed embedding recommendations (JSON lookup for now)."""

from __future__ import annotations

import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "embedding_recs.json"

# key: "artist::album" (already lowercased / normalized in the export)
_recs: dict[str, list[dict]] = {}


def album_key(artist: str, album: str) -> str:
    return f"{artist.strip().lower()}::{album.strip().lower()}"


def load(path: Path | None = None) -> int:
    """Load recommendation JSON into memory. Returns number of seed albums."""
    global _recs
    data_path = path or DATA_PATH
    with data_path.open(encoding="utf-8") as f:
        _recs = json.load(f)
    return len(_recs)


def list_albums() -> list[dict[str, str]]:
    albums = []
    for key in sorted(_recs):
        artist, album = key.split("::", 1)
        albums.append({"artist": artist, "album": album})
    return albums


def status() -> dict:
    """Stats for the embedding-recs store (JSON file for now)."""
    return {
        "backend": "json",
        "path": DATA_PATH.name,
        "size_bytes": DATA_PATH.stat().st_size if DATA_PATH.exists() else 0,
        "seed_albums": len(_recs),
        "recommendation_edges": sum(len(rows) for rows in _recs.values()),
    }


def recommend(artist: str, album: str, k: int = 5) -> list[dict] | None:
    """Exact key match. Returns None if the seed is missing."""
    key = album_key(artist, album)
    rows = _recs.get(key)
    if rows is None:
        return None
    return rows[:k]
