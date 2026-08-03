"""Tests for the pure query-decomposition helpers in src.query_decompose.

Both functions are pure - strings and plain dicts in, strings and plain dicts out -
so nothing here touches Qdrant, ColQwen2 or Gemini.

The splitter is deliberately conservative, and most of these tests pin the cases it
must *refuse*. 63 of the 83 eval questions are single-document, so a splitter that
fires too eagerly costs more than one that fires too rarely.
"""

from src.query_decompose import decompose, fuse_rrf


def _hit(pdf, page, score=0.0):
    """One retrieval hit, shaped like the dicts vector_store.search yields."""
    return {"pdf": pdf, "page_number": page, "image_path": f"{pdf}-{page}.png", "score": score}


# --- decompose: when it should split ---

def test_splits_a_two_clause_question_on_comma_and():
    """The dataset's cross-document shape: '<A>, and <B>?' becomes the whole plus both halves."""
    q = ("How many layers are in the Transformer's encoder stack, "
         "and how many layers does BERT-base have?")
    assert decompose(q) == [
        q,
        "How many layers are in the Transformer's encoder stack",
        "how many layers does BERT-base have?",
    ]


def test_splits_on_a_sentence_boundary():
    """Two separate questions split without needing a conjunction at all."""
    q = "What dimension does DPR encode a passage into? What about ColBERT's per-token size?"
    assert decompose(q) == [
        q,
        "What dimension does DPR encode a passage into?",
        "What about ColBERT's per-token size?",
    ]


def test_splits_on_versus():
    """'versus' joins two comparable halves the same way a conjunction does."""
    q = "ColBERT's per-token embedding dimension versus the dense vector size DPR encodes"
    assert decompose(q) == [
        q,
        "ColBERT's per-token embedding dimension",
        "the dense vector size DPR encodes",
    ]


# --- decompose: when it must refuse ---

def test_keeps_a_single_part_question_whole():
    """No split marker at all - the caller must get exactly one query back."""
    q = "What was Q4 revenue?"
    assert decompose(q) == [q]


def test_refuses_to_split_a_noun_phrase_and():
    """'and' inside a noun phrase is not clause coordination.

    This is the case a bare ' and ' rule mangles: splitting here would embed
    'question answering' alone, which names no document in the corpus. Requiring a
    comma or sentence boundary before the conjunction is what rules it out.
    """
    q = "What is the ColPali architecture for document retrieval and question answering?"
    assert decompose(q) == [q]


def test_refuses_to_split_when_a_half_is_too_short():
    """A two-token tail is not a searchable query; keep the question whole."""
    q = "What dimension does DPR encode a passage into, and why?"
    assert decompose(q) == [q]


def test_refuses_to_split_an_empty_question():
    """Degenerate input must not raise or produce empty sub-queries."""
    assert decompose("") == [""]
    assert decompose("   ") == ["   "]


# --- decompose: bounding the fanout ---

def test_caps_the_number_of_sub_queries():
    """A listy question yields at most MAX_SUBQUERIES parts beyond the original.

    Dropping the tail clause is safe rather than arbitrary: the full question is
    always in the fusion set, so an uncapped clause is no worse off than it is today.
    """
    q = ("How many layers does BERT-base have, and how many parameters does ViT-Base have, "
         "and what dimension does DPR encode a passage into?")
    parts = decompose(q)
    assert parts[0] == q
    assert len(parts) == 3  # the original + 2 halves


# --- fuse_rrf ---

def test_single_ranking_is_returned_in_its_original_order():
    """The identity property the untouched one-query path depends on.

    search() delegates to search_multi() with one ranking, so if fusion ever
    reordered a lone ranking every non-decomposed query would silently change.
    """
    ranking = [_hit("a.pdf", 1), _hit("b.pdf", 2), _hit("c.pdf", 3)]
    fused = fuse_rrf([ranking])
    assert [(h["pdf"], h["page_number"]) for h in fused] == [
        ("a.pdf", 1), ("b.pdf", 2), ("c.pdf", 3)
    ]


def test_a_page_in_both_rankings_appears_once():
    """Dedup is on (pdf, page_number), not object identity."""
    r1 = [_hit("a.pdf", 1), _hit("b.pdf", 2)]
    r2 = [_hit("a.pdf", 1), _hit("c.pdf", 3)]
    fused = fuse_rrf([r1, r2])
    keys = [(h["pdf"], h["page_number"]) for h in fused]
    assert keys.count(("a.pdf", 1)) == 1
    assert set(keys) == {("a.pdf", 1), ("b.pdf", 2), ("c.pdf", 3)}


def test_a_page_ranked_well_by_both_queries_outranks_one_ranked_well_by_either():
    """Agreement across sub-queries is what fusion rewards."""
    r1 = [_hit("solo.pdf", 1), _hit("both.pdf", 9)]
    r2 = [_hit("other.pdf", 1), _hit("both.pdf", 9)]
    fused = fuse_rrf([r1, r2])
    assert (fused[0]["pdf"], fused[0]["page_number"]) == ("both.pdf", 9)


def test_fusion_ignores_score_magnitude():
    """The property score-based merging would break.

    MaxSim sums over query tokens, so a longer sub-query's scores are systematically
    larger. Here the second ranking's scores are two orders of magnitude smaller; its
    top page must still beat a page sitting fifth in the high-scoring ranking.
    """
    big = [_hit("long.pdf", i, score=90.0 - i) for i in range(1, 6)]
    small = [_hit("short.pdf", 1, score=0.4)]
    fused = fuse_rrf([big, small])
    keys = [(h["pdf"], h["page_number"]) for h in fused]
    assert keys.index(("short.pdf", 1)) < keys.index(("long.pdf", 5))


def test_fused_scores_replace_raw_maxsim_when_rankings_are_merged():
    """Downstream consumers mean-scale `score` (src/confidence.py), so a fused slate
    must not carry scores from two different query scales side by side.

    The equality is the point, not an accident: each page is ranked first by its own
    sub-query, so fusion must call them equally preferred however far apart their raw
    MaxSim scores are.
    """
    r1 = [_hit("a.pdf", 1, score=88.0)]
    r2 = [_hit("b.pdf", 2, score=0.5)]
    fused = fuse_rrf([r1, r2])
    assert [h["score"] for h in fused] != [88.0, 0.5]
    assert fused[0]["score"] == fused[1]["score"]


def test_single_ranking_keeps_its_raw_scores():
    """With nothing to fuse there is no scale conflict, so the untouched path keeps
    the real MaxSim score retrieval_confidence was calibrated on."""
    ranking = [_hit("a.pdf", 1, score=88.0), _hit("b.pdf", 2, score=42.0)]
    fused = fuse_rrf([ranking])
    assert [h["score"] for h in fused] == [88.0, 42.0]


def test_empty_rankings_fuse_to_nothing():
    """No candidates anywhere must not raise."""
    assert fuse_rrf([]) == []
    assert fuse_rrf([[], []]) == []


# --- weighted fusion ---

def test_a_down_weighted_ranking_contributes_less():
    """Weights scale a ranking's vote without changing anyone's position in it."""
    r1 = [_hit("a.pdf", 1)]
    r2 = [_hit("b.pdf", 1)]
    equal = fuse_rrf([r1, r2])
    assert equal[0]["score"] == equal[1]["score"]

    tilted = fuse_rrf([r1, r2], weights=[0.2, 1.0])
    assert (tilted[0]["pdf"], tilted[0]["page_number"]) == ("b.pdf", 1)


def test_agreement_with_a_down_weighted_ranking_stops_burying_a_solo_hit():
    """The failure this parameter exists for.

    `solo` is found only by the second ranking but ranked first there; `pair` is found
    by both, but far down the second. With equal weights the agreement bonus puts
    `pair` first - which is how fusing the whole question alongside its halves buried
    the exact page decomposition was meant to surface. Down-weighting the first
    ranking restores the solo hit.

    The rank gap has to be this wide because `_RRF_K` is 60: at that damping, ranks 1
    and 4 of a 12-deep slate differ by ~5%, so a page merely *appearing* in a second
    ranking outweighs almost any rank advantage. That flatness is itself a finding -
    60 is tuned for fusing TREC lists thousands deep, not slates of a dozen.
    """
    whole = [_hit("pair.pdf", 1)]
    half = ([_hit("solo.pdf", 9)]
            + [_hit(f"filler{i}.pdf", 1) for i in range(18)]
            + [_hit("pair.pdf", 1)])

    equal = fuse_rrf([whole, half])
    assert (equal[0]["pdf"], equal[0]["page_number"]) == ("pair.pdf", 1)

    tilted = fuse_rrf([whole, half], weights=[0.2, 1.0])
    assert (tilted[0]["pdf"], tilted[0]["page_number"]) == ("solo.pdf", 9)


def test_weights_default_to_equal():
    """Omitting weights must behave exactly as passing all ones."""
    r1 = [_hit("a.pdf", 1), _hit("b.pdf", 2)]
    r2 = [_hit("b.pdf", 2), _hit("c.pdf", 3)]
    assert fuse_rrf([r1, r2]) == fuse_rrf([r1, r2], weights=[1.0, 1.0])
