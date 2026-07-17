# Last.fm Album Recommender

Given an album (artist + title), this tool suggests other albums using the public [Last.fm API](https://www.last.fm/api). Recommendations are derived from **track similarity**, because Last.fm does not expose a native “similar album” endpoint.

The pipeline is:

**seed album → seed track(s) → similar track → parent album**

## How it works

1. Look up the seed album (`album.getinfo`) and its tracklist.
2. Pick one or more seed tracks (see matching strategies below).
3. For each seed track, call `track.getsimilar` to fetch similar tracks from other artists.
4. Resolve each similar track to its parent album (`track.getinfo`).
5. Return album recommendations, excluding albums by the seed artist.

## Matching strategies

| Strategy | Seed track selection | Output |
|----------|----------------------|--------|
| **(a) Top listener** | The track on the album with the most Last.fm listeners | Top N albums from that track’s similarity graph |
| **(b) Random** | One random track from the album tracklist | Top N albums from that track’s similarity graph |
| **(c) Top-N tracks** | The `top_n` tracks (default 3) with the most listeners | Albums ranked by vote count across those tracks, then similarity score; falls back to **(a)** if nothing is found |

Responses are cached within a run (`lastfm_get`); pass `clear_cache=False` on later `recommend_album` calls in the same session to avoid redundant requests.

## Getting Started

1. Make sure you have all dependencies (`requirements.txt`) installed.
2. Prefer the repo-root `.env` (see `.env.example`), or copy `lastfm/.env.example` to `lastfm/.env`:

```
LASTFM_API_KEY=your_key_here
```

3. From the repo root (with `PYTHONPATH=.`), run:

```python
from lastfm.recommender import recommend_album

seed, seed_track, recs, used_fallback = recommend_album(
    "OK Computer",
    artist="Radiohead",
    track_selection="top_n_tracks",  # top_listener | random | top_n_tracks
    top_n=3,
    n_recs=5,
    fetch_floor=10,
    clear_cache=False,  # keep cache warm after the first call in a notebook session
)
```

For batch runs, throttle requests with `set_request_delay(seconds)` (see `lastfm_batch.ipynb`).
