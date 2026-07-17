"""Rerank step - Gemini picks the pages that actually answer the question.

Qdrant MaxSim retrieval is recall-oriented and returns RETRIEVE_K candidates. This
narrows them to RERANK_K pages before the answer step, so the answer/citation call
reasons over only the relevant pages (sharper citations, less distraction). The
candidate pages are sent as downscaled thumbnails so the triage call stays cheap;
the winners are re-sent at full resolution by the answer step.
"""

import io
import json
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel

from src.answerer import image_part
from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    RERANK_K,
    RERANK_THUMBNAIL_EDGE,
)


class Rerank(BaseModel):
    """The 1-based PAGE indices judged most relevant, best first."""

    page_indices: list[int]


_PROMPT = (
    """You are given {n} pages from a document as images, each labeled "PAGE <n>".
    Decide which pages actually contain information that helps answer the question
    below - including charts, tables, and scanned text.

    Return the {k} most relevant pages as a list of their PAGE numbers in
    `page_indices`, ordered most relevant first. Use only the integer labels shown
    (1 to {n}). Do not invent numbers. If fewer than {k} pages are relevant, return
    only the relevant ones.

    Question: {question}""")


def _thumb_part(image_path: Path, max_edge: int) -> types.Part:
    """Load a page as a downscaled JPEG part - a cheap image for the rerank triage."""
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_edge, max_edge))  # in place, preserves aspect ratio
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=80)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


def _candidate_part(image_path: Path) -> types.Part:
    """Full-res PNG when thumbnails are disabled, otherwise a downscaled JPEG."""
    if RERANK_THUMBNAIL_EDGE is None:
        return image_part(image_path)
    return _thumb_part(image_path, RERANK_THUMBNAIL_EDGE)


def _valid_order(raw: list, n: int, k: int) -> list[int]:
    """Clean a model's raw page_indices into exactly min(k, n) valid 1-based indices.

    Keeps in-range [1, n] ints in the model's order (rejecting bools/non-ints),
    de-duplicates, then tops up from Qdrant score order (1..n) until it holds
    min(k, n) entries. An empty `raw` therefore yields the pure Qdrant top-k, which
    is the fallback path for a failed or empty response.
    """
    order: list[int] = []
    seen: set[int] = set()
    for idx in raw or []:
        if isinstance(idx, int) and not isinstance(idx, bool) and 1 <= idx <= n and idx not in seen:
            order.append(idx)
            seen.add(idx)
            if len(order) == k:
                break
    for idx in range(1, n + 1):  # top-up / fallback in best-score order
        if len(order) >= k:
            break
        if idx not in seen:
            order.append(idx)
            seen.add(idx)
    return order[:k]


def rerank(question: str, pages: list[dict], k: int = RERANK_K) -> list[dict]:
    """Return the k most relevant pages (subset of `pages`, reordered best-first).

    `pages` is the Qdrant result, already best-score-first. On any failure or a
    degenerate request the result falls back to the Qdrant top-k, so this never
    raises into the graph.
    """
    n = len(pages)
    if n == 0 or k >= n:  # nothing to trim -> skip the Gemini call
        return pages

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        contents: list = []
        for i, page in enumerate(pages, start=1):
            contents.append(f"PAGE {i} ({page['pdf']} p.{page['page_number']}):")
            contents.append(_candidate_part(Path(page["image_path"])))
        contents.append(_PROMPT.format(n=n, k=k, question=question))
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Rerank,
            ),
        )
        parsed = response.parsed
        raw = (parsed.page_indices if parsed is not None
               else json.loads(response.text).get("page_indices", []))
    except Exception:  # network / quota / parse -> Qdrant top-k
        raw = []

    order = _valid_order(raw, n, k)
    return [pages[i - 1] for i in order]
