"""LLM explainer for recommendation pairs (placeholder for now)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SharedQuality:
    """One grounded connection between a seed album and a recommendation."""

    quality: str
    seed_quote: str
    rec_quote: str


class Explainer:
    """
    Explain why a recommended album relates to a seed album.

    Real implementation (see model_selection/recommender+explainer_in_action.ipynb)
    loads reviews for both albums and asks an LLM for up to ``n`` shared qualities,
    each backed by verbatim quotes from the two reviews.
    """

    def explain(
        self,
        query_artist: str,
        query_album: str,
        rec_artist: str,
        rec_album: str,
        n: int = 3,
    ) -> list[SharedQuality]:
        """
        Return up to ``n`` shared qualities for the seed → recommendation pair.

        Parameters mirror ``explain_pair_simple`` in the explainer notebook:
        query_* is the seed album; rec_* is the recommended album.
        """
        seed_label = f"{query_artist} — {query_album}"
        rec_label = f"{rec_artist} — {rec_album}"
        # Bogus placeholder until the real LLM + review lookup lands.
        return [
            SharedQuality(
                quality=f"Placeholder shared quality {i} ({seed_label} ↔ {rec_label})",
                seed_quote=f"[placeholder quote from review of {seed_label}]",
                rec_quote=f"[placeholder quote from review of {rec_label}]",
            )
            for i in range(1, n + 1)
        ]
