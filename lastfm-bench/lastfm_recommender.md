# Last.fm Album Recommender

Given an album (artist + title), this tool suggests other albums using the public [Last.fm API](https://www.last.fm/api). Recommendations are derived from **track similarity**, because Last.fm does not include a native “similar album” endpoint.

The pipeline is:

**seed album → seed track(s) → similar track → parent album**

## How it works

1. Look up the seed album (`album.getinfo`) and its tracklist.
2. Pick one or more seed tracks (see matching strategies below).
3. For each seed track, call `track.getSimilar` to fetch similar tracks from other artists.
4. Resolve each similar track to its parent album (`track.getinfo`).
5. Return album recommendations, excluding albums by the seed artist.

## Matching strategies

| Strategy | Seed track selection | Output |
|----------|----------------------|--------|
| **(a) Top listener** | The track on the album with the most Last.fm listeners | Top N albums from that track’s similarity graph |
| **(b) Random** | One random track from the album tracklist | Top N albums from that track’s similarity graph |
| **(c) Top-N tracks** | The `top_n` tracks (default 3) with the most listeners | Albums ranked by vote count across those tracks, then similarity score; falls back to **(a)** if nothing is found |

For strategies **(a)** and **(b)** with `n_recs` recommendations, expect about **`2 + n_recs` API calls per album** (`album.getinfo`, `track.getSimilar`, then `n_recs`× `track.getInfo`). `track.getSimilar` still requests `fetch_floor` candidates (default 10); same-artist tracks are dropped, the top `n_recs` by similarity are kept, and only those are resolved via `track.getInfo`. Strategy **(a)** also calls `track.getInfo` once per song on the album to pick the top-listener seed track. Strategy **(c)** calls `track.getInfo` once per album track to rank listeners, then `track.getSimilar` + up to `fetch_floor`× `track.getInfo` per seed track.

## Project files

| File | Purpose |
|------|---------|
| `lastfm_albums.py` | API client and recommendation logic |
| `lastfm_recs.ipynb` | Interactive demo comparing all three strategies |
| `lastfm_batch.ipynb` | Batch run over `albums.csv` subsets; writes to `lastfm_recommendations_<subset>_<strategy>.csv` |
| `.env.example` | Template for API key configuration |


## Basic Usage

```python
from lastfm_albums import compare_recommendations

results = compare_recommendations("OK Computer", artist="Radiohead", n_recs=5, fetch_floor=10, top_n=3)

results["top_listener_recs"]   # strategy (a)
results["random_track_recs"]   # strategy (b)
results["top_n_tracks_recs"]     # strategy (c)
```

```python
from lastfm_albums import recommend_album

seed, seed_track, recs, used_fallback = recommend_album(
    "OK Computer",
    artist="Radiohead",
    track_selection="top_n_tracks",
    top_n=3,
    n_recs=5,
)
```

API responses are cached within a single run to avoid duplicate requests.

## Notes

- Results reflect Last.fm’s crowd-sourced listening data and can vary depending on which seed track(s) are used.
- Always pass `artist=` when possible; album search alone can be ambiguous.
- This recommender does not use local ratings, review text, or collaborative filtering from our datasets.
