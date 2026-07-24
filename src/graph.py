"""Langraph flow: question -> retrieve -> rerank -> answer -> highlight."""

import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src import request_context
from src.answerer import answer as gemini_answer
from src.embedder import embed_query
from src.highlight import annotate_page, crop_region
from src.logging_setup import get_logger
from src.reranker import rerank
from src.vector_store import search

log = get_logger("graph")

class RAGState(TypedDict):
    """The state threaded through retrieve -> rerank -> answer -> highlight.

    Each node returns a partial dict that LangGraph merges in, so a key is only
    populated once the node that owns it has run. `retrieved` is overwritten by the
    rerank step (which is why `candidates` keeps the untrimmed top-k for the eval).
    """

    question: str
    retrieved: list[dict]
    candidates: list[dict]
    answer: str
    citation: dict | None
    crop_path: str | None            # primary (first) region's crop - CLI/server back-compat
    annotated_path: str | None       # primary region's page, annotated
    cited_regions: list[dict]        # [{source_page, box, crop_path}] for every valid region
    annotated_paths: list[str]       # one annotated page image per distinct cited page

def retrieve_node(state: RAGState) -> dict:
    """Embed the quesiton visually and pull the top matching pages.

    `candidates` keeps the full pre-rerank top-k (rerank overwrites `retrieved`),
    so eval recall@k and a "retrieved N, used K" UI can see what retrieval produced.
    """
    hits = search(embed_query(state["question"]))
    return {"retrieved": hits, "candidates": hits}

def rerank_node(state: RAGState) -> dict:
    """Ask Gemini which retrieved pages are actually relevant; keep only those.

    Overwrites `retrieved` so the 1-based source_page from the answer step stays
    aligned for highlight/printing.
    """
    return {"retrieved": rerank(state["question"], state["retrieved"])}

def answer_node(state: RAGState) -> dict:
    """Send the retrieved page images to Gemini and capture the answer + citation."""
    result = gemini_answer(state["question"], state["retrieved"])
    return {"answer": result["answer"], "citation": result}

def highlight_node(state: RAGState) -> dict:
    """Crop every cited region out of its page and draw an annotated copy per page."""
    citation = state.get("citation")
    retrieved = state["retrieved"]
    regions = (citation or {}).get("regions") or []
    empty: dict = {"crop_path": None, "annotated_path": None, "cited_regions": [], "annotated_paths": []}
    if not citation or not citation.get("found") or not regions:
        return empty

    # Crop each region whose source_page + box are valid (skip the rest - graceful
    # degradation); collect the boxes per page for one annotated image apiece.
    cited_regions: list[dict] = []
    boxes_by_page: dict[int, list[list[int]]] = {}
    for region in regions:
        source_page = region.get("source_page", 0)
        box = region.get("box") or []
        if not (1 <= source_page <= len(retrieved)) or len(box) != 4:
            continue
        image_path = retrieved[source_page - 1]["image_path"]
        crop_path = str(crop_region(image_path, box, index=len(cited_regions)))
        cited_regions.append({"source_page": source_page, "box": box, "crop_path": crop_path})
        boxes_by_page.setdefault(source_page, []).append(box)

    if not cited_regions:
        return empty

    annotated_by_page = {
        page: str(annotate_page(retrieved[page - 1]["image_path"], boxes))
        for page, boxes in boxes_by_page.items()
    }
    return {
        "crop_path": cited_regions[0]["crop_path"],
        "annotated_path": annotated_by_page[cited_regions[0]["source_page"]],
        "cited_regions": cited_regions,
        "annotated_paths": list(annotated_by_page.values()),
    }

def _timed(name: str, fn):
    """Wrap a graph node to log its start/end + latency_ms, leaving the node fn pure.

    Applied at registration (not as a decorator) so the raw node functions can still
    be called directly in tests without emitting timing logs. Every line carries the
    query's `request_id` via the root handler's filter.
    """

    def wrapped(state: RAGState) -> dict:
        """Run the wrapped node inside a timed, stage-scoped logging block."""
        request_context.enter_stage(name)
        log.info("node start", extra={"node": name})
        start = time.perf_counter()
        try:
            return fn(state)
        finally:
            latency_ms = round((time.perf_counter() - start) * 1000, 1)
            request_context.exit_stage(name, latency_ms)
            log.info("node end", extra={"node": name, "latency_ms": latency_ms})

    return wrapped


def build_graph():
    """Compile the retrieve -> rerank -> answer -> highlight graph."""
    builder = StateGraph(RAGState)
    builder.add_node("retrieve", _timed("retrieve", retrieve_node))
    builder.add_node("rerank", _timed("rerank", rerank_node))
    builder.add_node("answer", _timed("answer", answer_node))
    builder.add_node("highlight", _timed("highlight", highlight_node))
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "answer")
    builder.add_edge("answer", "highlight")
    builder.add_edge("highlight", END)
    return builder.compile()


_graph = None


def get_graph():
    """Compile the graph once and reuse it - a warm server pays compilation at boot,
    not per query. `build_graph` stays public so tests can call the raw nodes directly.
    """
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
