"""Query ClI: ask a question, get an answer read off the page images."""

import subprocess
import sys
import time
from pathlib import Path

from src import request_context
from src.confidence import retrieval_confidence
from src.config import validate
from src.graph import get_graph
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

    Binds a per-query `request_id` (stamped onto every log line), emits a single
    `"query complete"` summary line, and folds a `meta` block - `request_id`, total
    `latency_ms`, aggregated token/cost totals, and the per-stage breakdown - into the
    returned dict, so any caller (HTTP response, eval report) gets the observability
    surface without re-reading the torn-down request scope. The id is also passed to
    LangGraph's invoke config so opt-in LangSmith traces are searchable by it.
    """
    scope = request_context.begin_request()
    request_id = request_context.current_request_id()
    start = time.perf_counter()
    try:
        result = get_graph().invoke(
            {"question": question},
            config={"run_name": "rag_query", "metadata": {"request_id": request_id}},
        )
    finally:
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        usage = request_context.usage_totals()
        stages = request_context.stage_breakdown()
        log.info("query complete", extra={"latency_ms": latency_ms, **usage})
        request_context.end_request(scope)

    # Deterministic retrieval-decisiveness confidence: how sharply MaxSim preferred
    # the cited page over the full pre-rerank candidate set. None when nothing was
    # cited (a not-found answer). This is about retrieval, not answer correctness -
    # the model's own self-report rides on citation.confidence.
    citation = result.get("citation") or {}
    retrieved = result.get("retrieved", [])
    source_page = citation.get("source_page", 0)
    cited = retrieved[source_page - 1] if 1 <= source_page <= len(retrieved) else None

    result["meta"] = {
        "request_id": request_id,
        "latency_ms": latency_ms,
        **usage,
        "retrieval_confidence": retrieval_confidence(result.get("candidates", []), cited),
        "stages": stages,
    }
    return result


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
