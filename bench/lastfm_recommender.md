# Last.fm Album Recommender

Given an album (artist + title), this tool suggests other albums using the public [Last.fm API](https://www.last.fm/api). Recommendations are derived from **track similarity**, because Last.fm does not expose a native “similar album” endpoint.

The pipeline is:

**seed album → seed track → similar track → parent album**

## How it works

1. Look up the seed album (`album.getinfo`) and its tracklist.
2. Pick one or more seed tracks (see matching strategies below).
3. For each seed track, call `track.getSimilar` to fetch similar tracks from other artists.
4. Resolve each similar track to its parent album (`track.getinfo`).
5. Return ranked album recommendations, **excluding albums by the seed artist**.

## Matching strategies

| Strategy | Seed track selection | Output |
|----------|----------------------|--------|
| **(a) Top listener** | The track on the album with the most Last.fm listeners | Top N albums from that track’s similarity graph |
| **(b) Random** | One random track from the album tracklist | Top N albums from that track’s similarity graph |
| **(c) All-tracks overlap** | Every track on the album | Albums surfaced by **≥ 2** seed tracks, ranked by vote count; falls back to **(a)** if no overlap is found |

Strategy **(c)** is the most robust but also the most expensive: it typically requires ~40–50 API calls for a standard album.

## Project files

| File | Purpose |
|------|---------|
| `lastfm_albums.py` | API client and recommendation logic |
| `lastfm_recs.ipynb` | Interactive demo comparing all three strategies |
| `.env.example` | Template for API key configuration |

## Setup

1. Copy the environment template:

   ```bash
   cp bench/.env.example bench/.env
   ```

2. Add your Last.fm API key to `bench/.env`:

   ```
   LASTFM_API_KEY=your_key_here
   ```

   Keys are available from [Last.fm API account settings](https://www.last.fm/api/account/create). Ask a teammate if you need a shared project key.

3. Open and run `lastfm_recs.ipynb`.

## Usage

```python
from lastfm_albums import compare_recommendations

results = compare_recommendations("OK Computer", artist="Radiohead", n=5)

results["top_listener_recs"]   # strategy (a)
results["random_track_recs"]   # strategy (b)
results["all_tracks_recs"]     # strategy (c)
```

API responses are cached within a single run to avoid duplicate requests.

## Notes

- Results reflect Last.fm’s crowd-sourced listening data and can vary depending on which seed track(s) are used.
- Always pass `artist=` when possible; album search alone can be ambiguous.
- This recommender does not use local ratings, review text, or collaborative filtering from our datasets.
