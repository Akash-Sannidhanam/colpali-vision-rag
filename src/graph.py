"""Langraph flow: question -> retrieve -> answer -> highlight."""

from typing import TypedDict
from langgraph.graph import END, START, StateGraph

from src.answerer import answer as gemini_answer
from src.embedder import embed_query
from src.highlight import annotate_page, crop_region
from src.vector_store import search

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

def build_graph():
    """Compile the retrieve -> answer -> highlight graph."""
    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("answer", answer_node)
    builder.add_node("highlight", highlight_node)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", "highlight")
    builder.add_edge("highlight", END)
    return builder.compile()
