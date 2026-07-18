"""LLM explainer for recommendation pairs."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_ROOT_ENV = Path(__file__).resolve().parents[1] / ".env"
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_MAX_REVIEW_CHARS = 2000


@dataclass
class SharedQuality:
    """One grounded connection between a seed album and a recommendation."""

    quality: str
    seed_quote: str
    rec_quote: str


@dataclass
class ExplainResult:
    """Structured qualities plus the raw model response."""

    qualities: list[SharedQuality]
    raw_text: str
    seed_review_id: str | None = None
    rec_review_id: str | None = None


class Explainer:
    """
    Explain why two albums relate, using their review texts.

    Review lookup is the caller's job. Pass cleaned review text (and optional
    review ids for future use). Requires ``ANTHROPIC_API_KEY`` in the environment
    or repo-root ``.env``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        max_review_chars: int = _MAX_REVIEW_CHARS,
    ) -> None:
        load_dotenv(_ROOT_ENV)
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Add it to the repo-root .env "
                "or pass api_key= to Explainer()."
            )

        import anthropic

        self._client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.max_review_chars = max_review_chars

    def explain(
        self,
        seed_review_text: str,
        rec_review_text: str,
        *,
        seed_review_id: str | None = None,
        rec_review_id: str | None = None,
        n: int = 3,
    ) -> ExplainResult:
        """
        Return shared qualities for a seed → recommendation review pair.

        Parameters
        ----------
        seed_review_text, rec_review_text
            Review bodies (typically ``cleaned_text``).
        seed_review_id, rec_review_id
            Optional ids reserved for future caching / audit trails.
        n
            Maximum number of shared qualities to request.
        """
        prompt = self._build_prompt(seed_review_text, rec_review_text, n=n)
        response = self._client.messages.create(
            model=self.model,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.content[0].text.strip()
        qualities = _parse_qualities(raw_text)

        return ExplainResult(
            qualities=qualities,
            raw_text=raw_text,
            seed_review_id=seed_review_id,
            rec_review_id=rec_review_id,
        )

    def _build_prompt(self, seed_text: str, rec_text: str, n: int) -> str:
        seed_clip = seed_text[: self.max_review_chars]
        rec_clip = rec_text[: self.max_review_chars]
        return f"""You are analyzing two music reviews to find their most meaningful shared qualities.

Identify up to {n} significant qualities that both albums genuinely share, based ONLY on what is actually written in the reviews.
If fewer than {n} genuine connections exist, list only those that are real — do not pad with weak or generic observations.
If there are no meaningful connections, return an empty list.

For each shared quality:
- State it in one clear phrase
- Quote the specific language from each review that supports it (verbatim)

Review A:
{seed_clip}

Review B:
{rec_clip}

Respond with JSON only (no markdown fences, no intro/outro), using this shape:
{{
  "qualities": [
    {{
      "quality": "short phrase",
      "seed_quote": "verbatim quote from Review A",
      "rec_quote": "verbatim quote from Review B"
    }}
  ]
}}"""


def _parse_qualities(raw_text: str) -> list[SharedQuality]:
    """Parse model JSON into SharedQuality rows; empty list on failure."""
    payload = _extract_json_object(raw_text)
    if not isinstance(payload, dict):
        return []

    rows = payload.get("qualities")
    if not isinstance(rows, list):
        return []

    qualities: list[SharedQuality] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        quality = str(row.get("quality", "")).strip()
        seed_quote = str(row.get("seed_quote", "")).strip()
        rec_quote = str(row.get("rec_quote", "")).strip()
        if quality and seed_quote and rec_quote:
            qualities.append(
                SharedQuality(
                    quality=quality,
                    seed_quote=seed_quote,
                    rec_quote=rec_quote,
                )
            )
    return qualities


def _extract_json_object(text: str) -> dict | list | None:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        cleaned = cleaned[start : end + 1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None
