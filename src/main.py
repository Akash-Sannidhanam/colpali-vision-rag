"""Query ClI: ask a question, get an answer read off the page images."""

import subprocess
import sys
import time
from pathlib import Path

from src import request_context
from src.config import validate
from src.graph import build_graph
from src.logging_setup import get_logger
from src.vector_store import close_client

log = get_logger("query")

def _open_file(path: str) -> None:
    """Best-effort open a saved image in the OS viewer (macOS only, non-fatal)."""
    if sys.platform != "darwin" or not Path(path).exists():
        return
    try:
        subprocess.run(["open", path], check=False)
    except OSError:
        pass

def run_query(question: str) -> dict:
    """Run one question through the graph and return the structured result dict.

    Pure: no printing, no file-opening, no client teardown - the reusable seam
    shared by the CLI, a service, and the eval harness. The caller owns the
    Qdrant client lifecycle (the CLI closes it; a warm server keeps it open).

    Binds a per-query `request_id` (stamped onto every log line) and emits a single
    `"query complete"` summary line with total latency and aggregated token/cost. The
    id is also passed to LangGraph's invoke config so opt-in LangSmith traces are
    searchable by it.
    """
    scope = request_context.begin_request()
    start = time.perf_counter()
    try:
        graph = build_graph()
        return graph.invoke(
            {"question": question},
            config={
                "run_name": "rag_query",
                "metadata": {"request_id": request_context.current_request_id()},
            },
        )
    finally:
        log.info(
            "query complete",
            extra={
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                **request_context.usage_totals(),
            },
        )
        request_context.end_request(scope)


def run(question: str) -> None:
    """CLI: run one question, print the result, and open the crop (macOS)."""
    validate()
    try:
        result = run_query(question)
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
