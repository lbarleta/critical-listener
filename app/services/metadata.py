"""Album metadata lookup (CSV catalog when available; sample JSON fallback)."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = Path(__file__).resolve().parent.parent / "data" / "album_meta.json"
CSV_PATH = REPO_ROOT / "datasets" / "albums.csv"

# key: "artist::album" -> metadata dict
_meta: dict[str, dict] = {}
# folded key for accent/apostrophe mismatches
_meta_fold: dict[str, dict] = {}


def album_key(artist: str, album: str) -> str:
    return f"{artist.strip().lower()}::{album.strip().lower()}"


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("…", "...")
    )
    return re.sub(r"\s+", " ", text).strip().lower()


def _fold_key(artist: str, album: str) -> str:
    return f"{_fold(artist)}::{_fold(album)}"


def _parse_tags(val: object) -> list[str]:
    if val is None:
        return []
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return []
    for sep in (";", "|", ","):
        if sep in s:
            return [p.strip() for p in s.split(sep) if p.strip()]
    return [s]


def _to_int(val: object) -> int | None:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _row_to_meta(artist: str, album: str, row: dict[str, str]) -> dict:
    genres: list[str] = []
    for col in ("genre_pitchfork", "mb_genre"):
        raw = row.get(col)
        if raw and str(raw).strip() and str(raw).lower() != "nan":
            genres.append(str(raw).strip())
    tags = _parse_tags(row.get("lastfm_tags")) or _parse_tags(row.get("mb_tags"))
    genres = _dedupe(genres)[:4]
    tags = _dedupe(tags)[:6]
    if not genres and tags:
        genres = [
            t.title() if t.islower() else t
            for t in tags
            if not re.fullmatch(r"\d{4}", t)
        ][:2]
    return {
        "artist": artist,
        "album": album,
        "genres": genres,
        "tags": tags,
        "review_count": _to_int(row.get("review_count")),
        "listeners": _to_int(row.get("lastfm_listeners")),
    }


def _store(entry: dict) -> None:
    key = album_key(entry["artist"], entry["album"])
    _meta[key] = entry
    _meta_fold[_fold_key(entry["artist"], entry["album"])] = entry


def _load_json(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    for entry in data.values():
        _store(
            {
                "artist": entry.get("artist", ""),
                "album": entry.get("album", ""),
                "genres": list(entry.get("genres") or []),
                "tags": list(entry.get("tags") or []),
                "review_count": entry.get("review_count"),
                "listeners": entry.get("listeners"),
            }
        )
    return len(data)


def _load_csv(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            artist = (row.get("artist_norm") or row.get("artist") or "").strip().lower()
            album = (row.get("album_norm") or row.get("album") or "").strip().lower()
            if not artist or not album:
                continue
            _store(_row_to_meta(artist, album, row))
            count += 1
    return count


def load() -> int:
    """Load metadata. Prefer full albums.csv; always merge sample JSON."""
    global _meta, _meta_fold
    _meta = {}
    _meta_fold = {}

    if CSV_PATH.exists():
        _load_csv(CSV_PATH)
    if JSON_PATH.exists():
        _load_json(JSON_PATH)
    return len(_meta)


def get(artist: str, album: str) -> dict:
    """Return metadata for an album; empty fields if missing."""
    key = album_key(artist, album)
    hit = _meta.get(key) or _meta_fold.get(_fold_key(artist, album))
    if hit is None:
        return {
            "artist": artist.strip().lower(),
            "album": album.strip().lower(),
            "genres": [],
            "tags": [],
            "review_count": None,
            "listeners": None,
        }
    return {
        "artist": hit.get("artist", artist),
        "album": hit.get("album", album),
        "genres": list(hit.get("genres") or []),
        "tags": list(hit.get("tags") or []),
        "review_count": hit.get("review_count"),
        "listeners": hit.get("listeners"),
    }


def enrich(rec: dict) -> dict:
    """Attach metadata fields onto a recommendation dict."""
    meta = get(rec["artist"], rec["album"])
    return {
        **rec,
        "genres": meta["genres"],
        "tags": meta["tags"],
        "review_count": meta["review_count"],
        "listeners": meta["listeners"],
    }
