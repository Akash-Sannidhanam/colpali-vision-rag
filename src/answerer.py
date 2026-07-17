"""Vision answer step - Gemini 3.5 Flash reads the retrieved page images."""

import json
from pathlib import Path
from google import genai
from google.genai import types
from pydantic import BaseModel

from src.config import GEMINI_API_KEY, GEMINI_MODEL


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

def _image_part(image_path: Path) -> types.Part:
    """Load a saved PNG as an inline image part for the model."""
    data = Path(image_path).read_bytes()
    return types.Part.from_bytes(data=data, mime_type = "image/png")

def answer(question: str, pages: list[dict]) -> dict:
    """Ask Gemini the question against the retrieved page images.

    Returns a dict with keys answer, found, source_page (1-based index into
    `pages`), and box ([ymin, xmin, ymax, xmax] on a 0-1000 scale).
    """
    client = genai.Client(api_key = GEMINI_API_KEY)
    contents: list = []
    for i, page in enumerate(pages, start=1):
        contents.append(f"PAGE {i} ({page['pdf']} p.{page['page_number']}):")
        contents.append(_image_part(Path(page["image_path"])))
    contents.append(_PROMPT.format(question = question))
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Citation,
        ),
    )
    parsed = response.parsed
    if parsed is not None:
        return parsed.model_dump()
    return json.loads(response.text)
