"""Thin wrapper around explainer/explainer.py."""

from __future__ import annotations

import sys
from pathlib import Path

_EXPLAINER_DIR = Path(__file__).resolve().parents[2] / "explainer"
if str(_EXPLAINER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLAINER_DIR))

from explainer import Explainer  # noqa: E402

_explainer = Explainer()


def explain(
    query_artist: str,
    query_album: str,
    rec_artist: str,
    rec_album: str,
    n: int = 3,
) -> list[dict]:
    qualities = _explainer.explain(
        query_artist=query_artist,
        query_album=query_album,
        rec_artist=rec_artist,
        rec_album=rec_album,
        n=n,
    )
    return [
        {
            "quality": q.quality,
            "seed_quote": q.seed_quote,
            "rec_quote": q.rec_quote,
        }
        for q in qualities
    ]
