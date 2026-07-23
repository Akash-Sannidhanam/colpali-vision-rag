"""Vision answer step - Gemini 3.5 Flash reads the retrieved page images."""

import json
from pathlib import Path
from typing import Literal

from google.genai import types
from pydantic import BaseModel

from src.config import GEMINI_MODEL
from src.gemini_client import generate
from src.logging_setup import get_logger

log = get_logger("answerer")

Confidence = Literal["high", "medium", "low"]

# Cap on regions the model may cite, so one answer can't fan out into a wall of crops.
MAX_REGIONS = 3


class Region(BaseModel):
    """One place the answer was read from: a box on a specific retrieved page."""

    source_page: int     # 1-based PAGE number from the labels below
    box: list[int]       # [ymin, xmin, ymax, xmax], integers 0-1000


class Citation(BaseModel):
    """Structured answer plus every region of the pages it was read from."""

    answer: str          # natural-language answer, using only the pages
    found: bool          # whether the answer was actually located in the pages
    # Cited regions, most important first ([] if not found). The first region is the
    # "primary" one the back-compat source_page/box fields are derived from.
    regions: list[Region]
    # The model's own self-reported confidence in the answer. Optional with a neutral
    # default so a valid answer that omits it still comes through rather than degrading
    # to not-found. Miscalibrated by nature - surface it as the model's word, next to
    # the deterministic retrieval confidence, never as ground truth.
    confidence: Confidence = "medium"


_PROMPT = (
    """You are given one or more pages from a document as images, each labeled
    "PAGE <n>". Answer the question using only what is visible in these pages -
    including charts, tables, and scanned text. If the answer is a number from a
    chart or table, state it exactly.

    Also report every distinct region you read the answer from, most important
    first (at most 3 - e.g. two cells being compared, or a value and its label):
    - regions: a list where each item has:
        - source_page: the PAGE <n> number that region came from.
        - box: a tight bounding box around that chart/table/text region, as four
          integers [ymin, xmin, ymax, xmax] normalized to a 0-1000 scale (top-left
          origin) relative to that page.
    - found: true if the pages contain the answer, otherwise false.
    - confidence: how confident you are that this answer is correct given only
      these pages - exactly one of "high", "medium", or "low".

    If the pages do not contain the answer, say so in `answer`, set found=false,
    and regions=[].

    Question: {question}""")

def image_part(image_path: Path) -> types.Part:
    """Load a saved PNG as an inline image part for the model."""
    data = Path(image_path).read_bytes()
    return types.Part.from_bytes(data=data, mime_type = "image/png")

# Returned when the answer step fails or the model response can't be read as a
# Citation. highlight_node's guards (graph.py) skip a not-found citation cleanly.
_NOT_FOUND = {
    "answer": "Couldn't read the retrieved pages.",
    "found": False,
    "regions": [],
    "source_page": 0,
    "box": [],
    "confidence": "low",
}


def _with_primary(citation: Citation) -> dict:
    """Flatten a parsed Citation to the pipeline dict, adding back-compat primary fields.

    The multi-region `regions` list is authoritative, but downstream consumers (graph
    highlight, server, CLI, eval/scoring) still read a single 1-based `source_page` +
    `box`. Derive those from the first region (capped at MAX_REGIONS), and normalize a
    not-found answer to no regions (and "low" confidence) so highlight/eval skip cleanly.
    """
    d = citation.model_dump()
    if d["found"]:
        regions = d["regions"][:MAX_REGIONS]
    else:
        # A model may claim found=false yet report high confidence; pin the not-found
        # invariant so a not-found answer is always "low" with no regions.
        regions = []
        d["confidence"] = "low"
    d["regions"] = regions
    primary = regions[0] if regions else {"source_page": 0, "box": []}
    d["source_page"] = primary["source_page"]
    d["box"] = primary["box"]
    return d


def answer(question: str, pages: list[dict]) -> dict:
    """Ask Gemini the question against the retrieved page images.

    Returns a dict with keys answer, found, source_page (1-based index into
    `pages`), and box ([ymin, xmin, ymax, xmax] on a 0-1000 scale). On any failure
    - a transient API error that outlived the client's retries, or a malformed /
    wrong-shape response - degrades to a well-formed not-found citation so the
    graph's highlight step skips cleanly instead of crashing.
    """
    contents: list = []
    for i, page in enumerate(pages, start=1):
        contents.append(f"PAGE {i} ({page['pdf']} p.{page['page_number']}):")
        contents.append(image_part(Path(page["image_path"])))
    contents.append(_PROMPT.format(question = question))
    try:
        response = generate(
            model=GEMINI_MODEL,
            contents=contents,
            response_schema=Citation,
            purpose="answer",
        )
        parsed = response.parsed
        if parsed is not None:
            return _with_primary(parsed)
        # No SDK-parsed object: validate the raw JSON text through the schema, so a
        # valid-JSON-but-wrong-shape response also degrades to not-found instead of
        # KeyError-ing downstream where answer_node reads result["answer"].
        return _with_primary(Citation(**json.loads(response.text)))
    except Exception:
        log.warning(
            "answer step failed; returning not-found citation",
            exc_info=True,
            extra={"degraded": True, "stage": "answer"},
        )
        return dict(_NOT_FOUND)
