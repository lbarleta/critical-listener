"""Fetch album recommendations from the Last.fm API."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import requests
from dotenv import load_dotenv

LASTFM_API_URL = "https://ws.audioscrobbler.com/2.0/"
ENV_PATH = Path(__file__).resolve().parent / ".env"
DEFAULT_FETCH_FLOOR = 10

_api_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
_request_delay_sec: float = 1.0
_last_request_at: float = 0.0


def set_request_delay(seconds: float) -> None:
    """Minimum seconds between uncached Last.fm API requests."""
    global _request_delay_sec
    _request_delay_sec = max(0.0, seconds)


def _throttle_request() -> None:
    global _last_request_at
    if _request_delay_sec <= 0:
        return
    elapsed = time.monotonic() - _last_request_at
    if elapsed < _request_delay_sec:
        time.sleep(_request_delay_sec - elapsed)
    _last_request_at = time.monotonic()


def get_api_key() -> str:
    load_dotenv(ENV_PATH)
    api_key = os.environ.get("LASTFM_API_KEY")
    if not api_key:
        raise ValueError(
            "LASTFM_API_KEY is not set. Add it to bench/.env or export it in your shell."
        )
    return api_key


def clear_api_cache() -> None:
    _api_cache.clear()


def album_key(artist: str, album: str) -> str:
    return f"{artist.strip().lower()}::{album.strip().lower()}"


def _as_list(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    if isinstance(value, list):
        return value
    return [value]


def lastfm_get(method: str, **params: Any) -> dict[str, Any]:
    cache_key = (method, tuple(sorted((key, str(value)) for key, value in params.items())))
    if cache_key in _api_cache:
        return _api_cache[cache_key]

    _throttle_request()
    response = requests.get(
        LASTFM_API_URL,
        params={
            "method": method,
            "api_key": get_api_key(),
            "format": "json",
            **params,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(f"Last.fm API error {data['error']}: {data.get('message')}")
    _api_cache[cache_key] = data
    return data


def album_to_record(album: dict[str, Any]) -> dict[str, Any]:
    artist = album.get("artist")
    if isinstance(artist, dict):
        artist_name = artist.get("name") or artist.get("#text", "Unknown Artist")
    else:
        artist_name = str(artist or "Unknown Artist")

    return {
        "key": album_key(artist_name, album.get("name", "")),
        "artist": artist_name,
        "album": album.get("name", ""),
        "mbid": album.get("mbid") or None,
        "playcount": int(album.get("playcount", 0) or 0),
        "listeners": int(album.get("listeners", 0) or 0),
        "url": album.get("url"),
    }


def _parse_tracks(album_data: dict[str, Any]) -> list[dict[str, str]]:
    tracks = []
    for track in _as_list(album_data.get("tracks", {}).get("track")):
        name = track.get("name")
        if name:
            tracks.append({"name": name})
    return tracks


def get_album_detail(artist: str, album: str) -> dict[str, Any]:
    """One album.getinfo call: album metadata plus tracklist."""
    data = lastfm_get("album.getinfo", artist=artist, album=album)
    album_data = data["album"]
    record = album_to_record(album_data)
    record["tracks"] = _parse_tracks(album_data)
    return record


def get_album_info(artist: str, album: str) -> dict[str, Any]:
    return get_album_detail(artist, album)


def search_albums(query: str, limit: int = 10) -> list[dict[str, Any]]:
    data = lastfm_get("album.search", album=query, limit=limit)
    matches = _as_list(data.get("results", {}).get("albummatches", {}).get("album"))
    return [album_to_record(match) for match in matches if match.get("name")]


def resolve_album(album: str, artist: str | None = None) -> dict[str, Any]:
    if artist:
        return get_album_detail(artist, album)

    matches = search_albums(album, limit=10)
    if not matches:
        raise ValueError(f"Could not find Last.fm album: {album!r}")

    exact = [row for row in matches if row["album"].lower() == album.lower()]
    if len(exact) == 1:
        return get_album_detail(exact[0]["artist"], exact[0]["album"])

    if len(matches) == 1:
        row = matches[0]
        return get_album_detail(row["artist"], row["album"])

    options = ", ".join(f"{row['album']} — {row['artist']}" for row in matches[:5])
    raise ValueError(
        f"Multiple albums match {album!r}. Pass artist=... as well. Options: {options}"
    )


def get_track_listeners(artist: str, track: str) -> int:
    data = lastfm_get("track.getinfo", artist=artist, track=track)
    return int(data["track"].get("listeners", 0) or 0)


def pick_top_listener_tracks(seed: dict[str, Any], top_n: int = 3) -> list[dict[str, Any]]:
    """Return up to top_n album tracks ranked by Last.fm listener count."""
    if not seed.get("tracks"):
        raise ValueError(f"No tracks found for album {seed['album']!r}")

    ranked: list[dict[str, Any]] = []
    for track in seed["tracks"]:
        listeners = get_track_listeners(seed["artist"], track["name"])
        ranked.append({"name": track["name"], "listeners": listeners})
    ranked.sort(key=lambda item: item["listeners"], reverse=True)
    return ranked[: max(1, top_n)]


def pick_top_listener_track(seed: dict[str, Any]) -> dict[str, Any]:
    """Pick the album track with the highest Last.fm listener count."""
    return pick_top_listener_tracks(seed, top_n=1)[0]


def pick_random_track(
    seed: dict[str, Any],
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Pick a random track from the album tracklist."""
    if not seed.get("tracks"):
        raise ValueError(f"No tracks found for album {seed['album']!r}")

    rng = random.Random(random_seed)
    track = rng.choice(seed["tracks"])
    return {"name": track["name"]}


def get_similar_tracks(artist: str, track: str, limit: int = 10) -> list[dict[str, Any]]:
    data = lastfm_get("track.getsimilar", artist=artist, track=track, limit=limit)
    tracks = _as_list(data.get("similartracks", {}).get("track"))
    results = []
    for item in tracks:
        track_artist = item.get("artist")
        if isinstance(track_artist, dict):
            artist_name = track_artist.get("name", "")
        else:
            artist_name = str(track_artist or "")
        results.append(
            {
                "track": item.get("name", ""),
                "artist": artist_name,
                "match": float(item.get("match", 0) or 0),
                "url": item.get("url"),
            }
        )
    return results


def get_track_album(artist: str, track: str) -> dict[str, str]:
    """Resolve the parent album for a track (one track.getinfo call)."""
    data = lastfm_get("track.getinfo", artist=artist, track=track)
    track_data = data["track"]
    album = track_data.get("album", {})
    return {
        "artist": track_data.get("artist", {}).get("name", artist),
        "album": album.get("title", ""),
        "track": track_data.get("name", track),
        "url": album.get("url") or track_data.get("url"),
    }


def _is_seed_artist(artist: str, seed_artist: str) -> bool:
    return artist.strip().lower() == seed_artist.strip().lower()


def _finalize_recommendations(
    rows: list[dict[str, Any]],
    seed: dict[str, Any],
    n: int,
    score_column: str,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    frame = frame[frame["key"] != seed["key"]]
    frame = frame[~frame["artist"].str.lower().eq(seed["artist"].lower())]
    frame = frame.sort_values(score_column, ascending=False)
    frame = frame.drop_duplicates(subset="key", keep="first")
    return frame.head(n).reset_index(drop=True)


def _collect_album_recommendations_for_track(
    seed: dict[str, Any],
    seed_track: dict[str, Any],
    similar_per_track: int = 10,
    track_lookup_limit: int = 12,
) -> list[dict[str, Any]]:
    """
    track.getSimilar -> track.getInfo for one seed track; returns raw album rows.

    Fetches up to similar_per_track similar tracks, drops same-artist matches,
    keeps the top track_lookup_limit by similarity score, then resolves each to
    a parent album via track.getinfo.
    """
    similar_tracks = get_similar_tracks(
        seed["artist"],
        seed_track["name"],
        limit=similar_per_track,
    )

    pooled: dict[str, dict[str, Any]] = {}
    for similar in similar_tracks:
        if _is_seed_artist(similar["artist"], seed["artist"]):
            continue
        lookup_key = f"{similar['artist'].lower()}::{similar['track'].lower()}"
        existing = pooled.get(lookup_key)
        if existing is None or similar["match"] > existing["match"]:
            pooled[lookup_key] = similar

    rows: list[dict[str, Any]] = []
    for similar in sorted(pooled.values(), key=lambda item: item["match"], reverse=True)[
        :track_lookup_limit
    ]:
        parent = get_track_album(similar["artist"], similar["track"])
        if not parent["album"]:
            continue
        if _is_seed_artist(parent["artist"], seed["artist"]):
            continue

        key = album_key(parent["artist"], parent["album"])
        row = {
            "key": key,
            "artist": parent["artist"],
            "album": parent["album"],
            "url": parent["url"],
            "similarity_score": similar["match"],
            "seed_track": seed_track["name"],
            "matched_via": (
                f"track '{similar['track']}' similar to '{seed_track['name']}'"
            ),
        }
        existing = next((item for item in rows if item["key"] == key), None)
        if existing is None:
            rows.append(row)
        elif similar["match"] > existing["similarity_score"]:
            existing.update(row)

    return rows


def recommend_via_similar_tracks(
    seed: dict[str, Any],
    seed_track: dict[str, Any],
    n_recs: int = 5,
    fetch_floor: int = DEFAULT_FETCH_FLOOR,
    min_lookup: int = 5,
) -> pd.DataFrame:
    """
    track.getSimilar -> track.getInfo (parent album) from one seed track.

    Resolves at least min_lookup candidates (default 5) regardless of n_recs,
    so low n_recs values (e.g. 1) still have enough tracks to resolve before
    filtering drops them all.
    """
    rows = _collect_album_recommendations_for_track(
        seed,
        seed_track,
        similar_per_track=fetch_floor,
        track_lookup_limit=max(n_recs, min_lookup),
    )
    return _finalize_recommendations(rows, seed, n_recs, "similarity_score")


def _aggregate_track_recommendations(
    seed: dict[str, Any],
    seed_tracks: list[dict[str, Any]],
    n_recs: int,
    fetch_floor: int,
) -> pd.DataFrame:
    """Merge album recommendations from multiple seed tracks."""
    album_votes: dict[str, dict[str, Any]] = {}
    for seed_track in seed_tracks:
        for row in _collect_album_recommendations_for_track(
            seed,
            seed_track,
            similar_per_track=fetch_floor,
            track_lookup_limit=fetch_floor,
        ):
            entry = album_votes.get(row["key"])
            if entry is None:
                album_votes[row["key"]] = {
                    **row,
                    "vote_count": 1,
                    "seed_tracks": [seed_track["name"]],
                }
                continue

            entry["vote_count"] += 1
            entry["seed_tracks"].append(seed_track["name"])
            if row["similarity_score"] > entry["similarity_score"]:
                entry["similarity_score"] = row["similarity_score"]
                entry["matched_via"] = row["matched_via"]

    if not album_votes:
        return pd.DataFrame()

    seed_track_label = ", ".join(track["name"] for track in seed_tracks)
    rows = []
    for row in album_votes.values():
        votes = ", ".join(dict.fromkeys(row["seed_tracks"]))
        rows.append(
            {
                **row,
                "seed_track": seed_track_label,
                "matched_via": (
                    f"from {row['vote_count']} of {len(seed_tracks)} top tracks"
                    f" ({votes})"
                ),
            }
        )

    frame = pd.DataFrame(rows)
    frame = frame[frame["key"] != seed["key"]]
    frame = frame[~frame["artist"].str.lower().eq(seed["artist"].lower())]
    frame = frame.sort_values(["vote_count", "similarity_score"], ascending=False)
    frame = frame.drop_duplicates(subset="key", keep="first")
    return frame.head(n_recs).reset_index(drop=True)


def recommend_via_top_tracks(
    seed: dict[str, Any],
    n_recs: int = 5,
    fetch_floor: int = DEFAULT_FETCH_FLOOR,
    top_n: int = 3,
) -> tuple[pd.DataFrame, list[dict[str, Any]], bool]:
    """
    Use the top_n album tracks by listener count as seed tracks.

    Albums are ranked by how many of those tracks surface them, then by
    similarity score. Falls back to a single top-listener track if nothing
    is found.

    Returns (recommendations, seed_tracks, used_fallback).
    """
    seed_tracks = pick_top_listener_tracks(seed, top_n=top_n)
    recommendations = _aggregate_track_recommendations(
        seed, seed_tracks, n_recs=n_recs, fetch_floor=fetch_floor
    )
    if recommendations.empty:
        fallback_track = seed_tracks[0]
        recommendations = recommend_via_similar_tracks(
            seed, fallback_track, n_recs=n_recs, fetch_floor=fetch_floor
        )
        return recommendations, [fallback_track], True
    return recommendations, seed_tracks, False


def compare_recommendations(
    album: str,
    artist: str | None = None,
    n_recs: int = 5,
    fetch_floor: int = DEFAULT_FETCH_FLOOR,
    random_seed: int | None = 42,
    top_n: int = 3,
) -> dict[str, Any]:
    """
    Compare three seed-track strategies on the same album:

    (a) track with the most listeners on the album
    (b) a random track from the album
    (c) top_n tracks by listener count, aggregated by vote count
    """
    clear_api_cache()
    seed = resolve_album(album, artist=artist)

    top_listener_track = pick_top_listener_track(seed)
    random_track = pick_random_track(seed, random_seed=random_seed)

    top_listener_recs = recommend_via_similar_tracks(
        seed, top_listener_track, n_recs=n_recs, fetch_floor=fetch_floor
    )
    random_track_recs = recommend_via_similar_tracks(
        seed, random_track, n_recs=n_recs, fetch_floor=fetch_floor
    )
    top_n_tracks_recs, top_n_seed_tracks, top_n_used_fallback = recommend_via_top_tracks(
        seed, n_recs=n_recs, fetch_floor=fetch_floor, top_n=top_n
    )

    pairwise_overlap = pd.DataFrame()
    if not top_listener_recs.empty and not random_track_recs.empty:
        pairwise_overlap = top_listener_recs.merge(
            random_track_recs,
            on="key",
            suffixes=("_top_listener", "_random"),
        )[
            [
                "album_top_listener",
                "artist_top_listener",
                "seed_track_top_listener",
                "similarity_score_top_listener",
                "similarity_score_random",
            ]
        ]

    return {
        "seed": seed,
        "top_listener_track": top_listener_track,
        "random_track": random_track,
        "top_n_seed_tracks": top_n_seed_tracks,
        "top_listener_recs": top_listener_recs,
        "random_track_recs": random_track_recs,
        "top_n_tracks_recs": top_n_tracks_recs,
        "top_n_used_fallback": top_n_used_fallback,
        "pairwise_overlap": pairwise_overlap,
        "api_calls": len(_api_cache),
    }


def recommend_album(
    album: str,
    artist: str | None = None,
    n_recs: int = 5,
    fetch_floor: int = DEFAULT_FETCH_FLOOR,
    track_selection: Literal["top_listener", "random", "top_n_tracks"] = "top_listener",
    top_n: int = 3,
    random_seed: int | None = None,
    clear_cache: bool = True,
    *,
    n: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None, pd.DataFrame, bool]:
    """Convenience wrapper for a single seed-track strategy."""
    if n is not None:
        n_recs = n
    if clear_cache:
        clear_api_cache()
    seed = resolve_album(album, artist=artist)
    if track_selection == "top_n_tracks":
        recommendations, seed_tracks, used_fallback = recommend_via_top_tracks(
            seed, n_recs=n_recs, fetch_floor=fetch_floor, top_n=top_n
        )
        seed_track = {
            "name": ", ".join(track["name"] for track in seed_tracks),
            "tracks": seed_tracks,
        }
        return seed, seed_track, recommendations, used_fallback

    if track_selection == "top_listener":
        seed_track = pick_top_listener_track(seed)
    else:
        seed_track = pick_random_track(seed, random_seed=random_seed)
    recommendations = recommend_via_similar_tracks(
        seed, seed_track, n_recs=n_recs, fetch_floor=fetch_floor
    )
    return seed, seed_track, recommendations, False
