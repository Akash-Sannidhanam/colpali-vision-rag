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
    question: str
    retrieved: list[dict]
    answer: str
    citation: dict | None
    crop_path: str | None
    annotated_path: str | None

def retrieve_node(state: RAGState) -> dict:
    """Embed the quesiton visually and pull the top matching pages."""
    query_vec = embed_query(state["question"])
    return {"retrieved": search(query_vec)}

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
    """Crop the cited region out of its page and draw an annotated copy."""
    citation = state.get("citation")
    retrieved = state["retrieved"]
    if not citation or not citation.get("found") or not citation.get("box"):
        return {"crop_path": None, "annotated_path": None}

    source_page = citation.get("source_page", 0)
    if not (1 <= source_page <= len(retrieved)):
        return {"crop_path": None, "annotated_path": None}

    image_path = retrieved[source_page - 1]["image_path"]
    box = citation["box"]
    return {
        "crop_path": str(crop_region(image_path, box)),
        "annotated_path": str(annotate_page(image_path, box)),
    }

def _timed(name: str, fn):
    """Wrap a graph node to log its start/end + latency_ms, leaving the node fn pure.

    Applied at registration (not as a decorator) so the raw node functions can still
    be called directly in tests without emitting timing logs. Every line carries the
    query's `request_id` via the root handler's filter.
    """

    def wrapped(state: RAGState) -> dict:
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
