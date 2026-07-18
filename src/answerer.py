"""Vision answer step - Gemini 3.5 Flash reads the retrieved page images."""

import json
from pathlib import Path
from google.genai import types
from pydantic import BaseModel

from src.config import GEMINI_MODEL
from src.gemini_client import generate
from src.logging_setup import get_logger

log = get_logger("answerer")


class Citation(BaseModel):
    """Structured answer plus the region of the page it was read from."""

    answer: str          # natural-language answer, using only the pages
    found: bool          # whether the answer was actually located in the pages
    source_page: int     # 1-based PAGE number from the labels below (0 if not found)
    box: list[int]       # [ymin, xmin, ymax, xmax], integers 0-1000; [] if not found


_PROMPT = (
    """You are given one or more pages from a document as images, each labeled
    "PAGE <n>". Answer the question using only what is visible in these pages -
    including charts, tables, and scanned text. If the answer is a number from a
    chart or table, state it exactly.

    Also report exactly where you read the answer from:
    - source_page: the PAGE <n> number the answer came from.
    - box: a tight bounding box around that chart/table/text region, as four
      integers [ymin, xmin, ymax, xmax] normalized to a 0-1000 scale (top-left
      origin) relative to that page.
    - found: true if the pages contain the answer, otherwise false.

    If the pages do not contain the answer, say so in `answer`, set found=false,
    source_page=0, and box=[].

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
    "source_page": 0,
    "box": [],
}


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
            return parsed.model_dump()
        # No SDK-parsed object: validate the raw JSON text through the schema, so a
        # valid-JSON-but-wrong-shape response also degrades to not-found instead of
        # KeyError-ing downstream where answer_node reads result["answer"].
        return Citation(**json.loads(response.text)).model_dump()
    except Exception:
        log.warning("answer step failed; returning not-found citation", exc_info=True)
        return dict(_NOT_FOUND)
