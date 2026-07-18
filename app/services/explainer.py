"""Thin wrapper around the explainer package."""

from __future__ import annotations

from explainer import Explainer

_explainer: Explainer | None = None


def _get_explainer() -> Explainer:
    global _explainer
    if _explainer is None:
        _explainer = Explainer()
    return _explainer


def explain(
    seed_review_text: str,
    rec_review_text: str,
    *,
    seed_review_id: str | None = None,
    rec_review_id: str | None = None,
    n: int = 3,
) -> dict:
    result = _get_explainer().explain(
        seed_review_text,
        rec_review_text,
        seed_review_id=seed_review_id,
        rec_review_id=rec_review_id,
        n=n,
    )
    return {
        "qualities": [
            {
                "quality": q.quality,
                "seed_quote": q.seed_quote,
                "rec_quote": q.rec_quote,
            }
            for q in result.qualities
        ],
        "raw_text": result.raw_text,
        "seed_review_id": result.seed_review_id,
        "rec_review_id": result.rec_review_id,
    }
