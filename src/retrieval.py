"""The question -> candidate pages seam, shared by the graph and the eval harness.

This is one function, and it exists because it was previously *two*: `retrieve_node`
and `eval.run_eval.run_retrieval_only` each embedded the question and searched on
their own. That duplication was harmless while retrieval was a single embed + search,
and stopped being harmless the moment query decomposition made "how a question becomes
candidates" a policy decision - turning the knob on moved the pipeline while the eval
went on measuring the old path, so the first treatment arm silently reproduced its own
control. Anything that changes retrieval policy belongs here, where both callers see it.

Kept free of langgraph and Gemini imports on purpose: `--retrieval-only` runs with no
API key, and that stays true only if this module's dependencies stay narrow.
"""

from src.config import DECOMPOSE_ORIGINAL_WEIGHT, QUERY_DECOMPOSE
from src.embedder import embed_query
from src.logging_setup import get_logger
from src.query_decompose import decompose
from src.vector_store import search_multi

log = get_logger("retrieval")


def retrieve(question: str) -> list[dict]:
    """Embed `question` visually and return the candidate pages, best first.

    With QUERY_DECOMPOSE on, a two-part question is embedded as its halves and their
    rankings are fused, so a document the whole question never ranks can still reach
    the slate. DECOMPOSE_ORIGINAL_WEIGHT decides how much the whole question still
    counts for; at 0 it is dropped from the fusion entirely and its embed is skipped.

    The split costs one local embed and one Qdrant query per part - milliseconds - and
    no extra Gemini tokens, because the slate stays RETRIEVE_K wide however many
    sub-queries fed it.
    """
    queries = decompose(question) if QUERY_DECOMPOSE else [question]
    weights = None
    if len(queries) > 1:
        # queries[0] is always the whole question; the rest are its halves.
        if DECOMPOSE_ORIGINAL_WEIGHT <= 0:
            queries = queries[1:]           # dropped outright, so its embed is skipped too
        else:
            weights = [DECOMPOSE_ORIGINAL_WEIGHT] + [1.0] * (len(queries) - 1)
        # Logged so a live split is auditable after the fact, the same way search's
        # dropped hits are - a splitter that fires on the wrong question is otherwise
        # invisible until it surfaces as a coverage number nobody can explain.
        log.info("decomposed query", extra={
            "subquery_count": len(queries),
            "subquery_lengths": [len(q) for q in queries]
        })
    return search_multi([embed_query(q) for q in queries], weights=weights)
