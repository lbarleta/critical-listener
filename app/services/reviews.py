"""Album review lookup (sample JSON for now; SQLite later)."""

from __future__ import annotations

import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "album_reviews.json"

# key: "artist::album" -> {artist, album, review_id, text}
_reviews: dict[str, dict[str, str]] = {}


def album_key(artist: str, album: str) -> str:
    return f"{artist.strip().lower()}::{album.strip().lower()}"


def load(path: Path | None = None) -> int:
    """Load review JSON into memory. Returns number of albums with a review."""
    global _reviews
    data_path = path or DATA_PATH
    with data_path.open(encoding="utf-8") as f:
        _reviews = json.load(f)
    return len(_reviews)


def get(artist: str, album: str) -> dict[str, str] | None:
    """Return one review for the album, or None if missing."""
    return _reviews.get(album_key(artist, album))


def status() -> dict:
    return {
        "backend": "json",
        "path": DATA_PATH.name,
        "size_bytes": DATA_PATH.stat().st_size if DATA_PATH.exists() else 0,
        "albums": len(_reviews),
    }
