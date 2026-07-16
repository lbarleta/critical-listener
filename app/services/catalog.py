"""Album catalog search (in-memory for now; SQLite later)."""

from __future__ import annotations

# Populated at startup from the embedding store (later: SQLite).
_albums: list[dict[str, str]] = []


def load(albums: list[dict[str, str]]) -> int:
    """Replace the in-memory catalog. Returns number of albums loaded."""
    global _albums
    _albums = list(albums)
    return len(_albums)


def search(q: str, limit: int = 15) -> list[dict[str, str]]:
    """
    Return up to ``limit`` albums matching ``q``.

    Simple substring match on artist, album, or "artist album".
    Empty query returns no rows (autocomplete should send a prefix).
    """
    needle = q.strip().lower()
    if not needle or limit <= 0:
        return []

    hits: list[dict[str, str]] = []
    for album in _albums:
        artist = album["artist"]
        title = album["album"]
        haystack = f"{artist} {title}"
        if needle in artist or needle in title or needle in haystack:
            hits.append({"artist": artist, "album": title})
            if len(hits) >= limit:
                break
    return hits
