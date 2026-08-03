"""Tests for the question -> candidate pages seam (src.retrieval.retrieve).

This module exists because the seam did not: `graph.retrieve_node` and
`eval.run_eval.run_retrieval_only` each embedded and searched on their own, so
turning decomposition on moved the pipeline and left the eval measuring the old
behaviour - a treatment arm that silently reproduced its own control.

Collaborators (embed_query, search_multi) are stubbed, so nothing here loads
ColQwen2 or touches Qdrant.
"""

import logging

from src import retrieval

TWO_PART = ("How many layers are in the Transformer's encoder stack, "
            "and how many layers does BERT-base have?")


def _stub(monkeypatch, hits=()):
    """Stub embed_query + search_multi; returns the list embed_query was called with."""
    embedded: list[str] = []

    def embed_query(text):
        """Record the query and hand back a stand-in multivector."""
        embedded.append(text)
        return [[float(len(embedded))]]

    monkeypatch.setattr(retrieval, "embed_query", embed_query)
    monkeypatch.setattr(retrieval, "search_multi",
                        lambda mvs, weights=None: list(hits))
    return embedded


def test_embeds_only_the_question_when_decomposition_is_off(monkeypatch):
    """The default path must not pay for embeds it was not asked for."""
    monkeypatch.setattr(retrieval, "QUERY_DECOMPOSE", False)
    embedded = _stub(monkeypatch)

    retrieval.retrieve(TWO_PART)

    assert embedded == [TWO_PART]


def test_embeds_only_the_halves_at_the_shipped_weight(monkeypatch):
    """At DECOMPOSE_ORIGINAL_WEIGHT=0 the whole question is dropped, embed included.

    This is the shipped configuration. Keeping the whole question in the fusion is what
    let it out-vote the half that could find the second document - RRF rewards
    agreement, so the pages the whole question already liked got counted twice.
    """
    monkeypatch.setattr(retrieval, "QUERY_DECOMPOSE", True)
    monkeypatch.setattr(retrieval, "DECOMPOSE_ORIGINAL_WEIGHT", 0.0)
    embedded = _stub(monkeypatch)

    retrieval.retrieve(TWO_PART)

    assert TWO_PART not in embedded
    assert len(embedded) == 2


def test_keeps_the_whole_question_at_a_positive_weight(monkeypatch):
    """The rejected equal-weight design stays reproducible from the knob."""
    monkeypatch.setattr(retrieval, "QUERY_DECOMPOSE", True)
    monkeypatch.setattr(retrieval, "DECOMPOSE_ORIGINAL_WEIGHT", 1.0)
    embedded = _stub(monkeypatch)

    retrieval.retrieve(TWO_PART)

    assert embedded[0] == TWO_PART            # the whole question leads when it is kept
    assert len(embedded) == 3


def test_a_positive_weight_reaches_the_fusion(monkeypatch):
    """The weight has to arrive at fuse_rrf, or the knob is decorative."""
    monkeypatch.setattr(retrieval, "QUERY_DECOMPOSE", True)
    monkeypatch.setattr(retrieval, "DECOMPOSE_ORIGINAL_WEIGHT", 0.25)
    seen: list = []
    monkeypatch.setattr(retrieval, "embed_query", lambda t: [[float(len(t))]])
    monkeypatch.setattr(retrieval, "search_multi",
                        lambda mvs, weights=None: seen.append(weights) or [])

    retrieval.retrieve(TWO_PART)

    assert seen[0] == [0.25, 1.0, 1.0]


def test_leaves_a_single_part_question_alone_with_decomposition_on(monkeypatch):
    """63 of the 83 eval questions are single-document; the knob must not touch them."""
    monkeypatch.setattr(retrieval, "QUERY_DECOMPOSE", True)
    embedded = _stub(monkeypatch)

    retrieval.retrieve("What was Q4 revenue?")

    assert embedded == ["What was Q4 revenue?"]


def test_passes_one_multivector_per_query_to_search(monkeypatch):
    """search_multi receives the embeddings, not the strings."""
    monkeypatch.setattr(retrieval, "QUERY_DECOMPOSE", True)
    seen: list = []
    monkeypatch.setattr(retrieval, "embed_query", lambda t: [[float(len(t))]])
    monkeypatch.setattr(retrieval, "search_multi",
                        lambda mvs, weights=None: seen.append(mvs) or [])

    retrieval.retrieve(TWO_PART)

    assert len(seen[0]) == 2


def _capture(monkeypatch, level):
    """Run retrieve(TWO_PART) at `level` and return the records the retrieval logger emitted."""
    monkeypatch.setattr(retrieval, "QUERY_DECOMPOSE", True)
    _stub(monkeypatch)
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger = logging.getLogger("retrieval")
    logger.addHandler(handler)
    prev = logger.level
    logger.setLevel(level)
    try:
        retrieval.retrieve(TWO_PART)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev)
    return records


def test_logs_the_shape_of_the_split_at_info(monkeypatch):
    """A live split has to be visible in a production log, like dropped hits are."""
    records = _capture(monkeypatch, logging.INFO)

    shape = [r for r in records if getattr(r, "subquery_count", None)]
    assert shape and shape[0].subquery_count == 2
    assert shape[0].subquery_lengths == [54, 36]


def test_does_not_log_question_text_at_info(monkeypatch):
    """No other log line in this codebase emits question text; this one must not either."""
    records = _capture(monkeypatch, logging.INFO)

    assert not [r for r in records if getattr(r, "subqueries", None)]


def test_logs_the_split_itself_at_debug(monkeypatch):
    """Shape alone cannot tell a sensible split from a garbage one - count=2 with
    plausible lengths is what a bad split looks like too. The actual split stays
    available, one log level down."""
    records = _capture(monkeypatch, logging.DEBUG)

    split = [r for r in records if getattr(r, "subqueries", None)]
    assert split and split[0].subqueries == [
        "How many layers are in the Transformer's encoder stack",
        "how many layers does BERT-base have?",
    ]


def test_returns_the_hits_search_produced(monkeypatch):
    """The seam is a pass-through for results; it must not reshape them."""
    monkeypatch.setattr(retrieval, "QUERY_DECOMPOSE", True)
    hits = [{"pdf": "a.pdf", "page_number": 1}]
    _stub(monkeypatch, hits)

    assert retrieval.retrieve(TWO_PART) == hits
