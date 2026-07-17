"""Query ClI: ask a question, get an answer read off the page images."""

import subprocess
import sys
from pathlib import Path

from src.graph import build_graph
from src.vector_store import close_client

def _open_file(path: str) -> None:
    """Best-effort open a saved image in the OS viewer (macOS only, non-fatal)."""
    if sys.platform != "darwin" or not Path(path).exists():
        return
    try:
        subprocess.run(["open", path], check=False)
    except OSError:
        pass

def run(question: str) -> None:
    """Run one question through the retrieve -> answer -> highlight graph and print it."""
    graph = build_graph()

    try:
        result = graph.invoke({"question": question})
    finally:
        close_client()

    print("\n" + "=" * 60 + "\nRETRIEVED PAGES\n" + "=" * 60)
    for hit in result["retrieved"]:
        print(f"{hit['pdf']}- page {hit['page_number']} (score {hit['score']})")

    print("\n" + "=" * 60 + "\nANSWER\n" + "=" * 60)
    print(result["answer"] + "\n" + "=" * 60)

    print("\n" + "=" * 60 + "\nSOURCE REGION\n" + "=" * 60)
    crop_path = result.get("crop_path")
    if crop_path:
        citation = result["citation"]
        hit = result["retrieved"][citation["source_page"] - 1]
        print(f"From {hit['pdf']} - page {hit['page_number']}")
        print(f"crop:      {crop_path}")
        print(f"annotated: {result.get('annotated_path')}")
        print("=" * 60)
        _open_file(crop_path)
    else:
        print("No region located for this answer.\n" + "=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: PYTHONPATH=. uv run python src/main.py "your question"')
        sys.exit(1)

    run(" ".join(sys.argv[1:]))
