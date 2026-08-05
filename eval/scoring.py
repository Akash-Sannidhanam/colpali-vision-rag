"""Pure scoring logic for the eval harness - plain dicts in, plain values out.

No imports from `src.` and no I/O, so the whole module is unit-testable without
models, Qdrant, or an API key (tests/test_eval_scoring.py). run_eval.py owns the
orchestration: it runs the pipeline, builds one row dict per question from these
scorers, then aggregates and renders.

Row shape consumed by `aggregate` / `format_table` (keys absent or None = N/A,
excluded from that metric's denominator):
    id, tags, gold_rank, rerank_hit, citation_correct, candidate_doc_coverage,
    gold_doc_coverage, substring_match, abstention_correct, retrieval_confidence,
    top1_decisiveness, self_confidence, judge {correct, score}, latency_ms

Every metric is computed over applicable rows only, which is what lets one dataset
hold questions of different kinds: an unanswerable row carries `abstention_correct`
and no `gold_rank`, so it scores the hallucination rate without polluting recall.
"""

import json
import re


def load_dataset(lines) -> list[dict]:
    """Parse dataset.jsonl lines into validated row dicts.

    Each non-blank line must be a JSON object with a unique `id`, a non-empty
    `question`, and a non-empty `gold` list of {pdf, page>=1}; `answer_contains`
    (any-of substrings), `answer_contains_all` (all-of substrings) and `tags` are
    optional. Raises ValueError naming the 1-based line number of the first bad row.

    Setting `unanswerable: true` inverts the gold rules: the question has no answer
    anywhere in the corpus, so `gold` must be absent or empty and `answer_contains`
    must be absent - the only correct behaviour is to decline. The flag is explicit
    rather than inferred from an empty `gold` so that a row which *meant* to name a
    gold page but lost it stays a validation error instead of silently becoming a
    negative question.
    """
    rows: list[dict] = []
    seen_ids: set[str] = set()
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"dataset line {lineno}: not valid JSON ({exc})") from exc

        def bad(reason: str):
            """A ValueError naming the offending dataset line."""
            return ValueError(f"dataset line {lineno}: {reason}")

        if not isinstance(row, dict):
            raise bad("row must be a JSON object")
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id.strip():
            raise bad("missing or empty `id`")
        if row_id in seen_ids:
            raise bad(f"duplicate id {row_id!r}")
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise bad("missing or empty `question`")
        unanswerable = row.get("unanswerable", False)
        if not isinstance(unanswerable, bool):
            raise bad("`unanswerable` must be a boolean")
        gold = row.get("gold", [])
        if not isinstance(gold, list):
            raise bad("`gold` must be a non-empty list of {pdf, page}")
        if unanswerable and gold:
            raise bad("an `unanswerable` row must not name any `gold` page")
        if not unanswerable and not gold:
            raise bad("`gold` must be a non-empty list of {pdf, page}")
        for g in gold:
            if not isinstance(g, dict) or not isinstance(g.get("pdf"), str):
                raise bad("each gold entry needs a string `pdf`")
            page = g.get("page")
            if not isinstance(page, int) or isinstance(page, bool) or page < 1:
                raise bad("each gold entry needs an integer `page` >= 1")
        expected = row.get("answer_contains")
        expected_all = row.get("answer_contains_all")
        for field, value in (("answer_contains", expected), ("answer_contains_all", expected_all)):
            if value is not None and (
                not isinstance(value, list)
                or not all(isinstance(s, str) and s for s in value)
            ):
                raise bad(f"`{field}` must be a list of non-empty strings")
            if unanswerable and value:
                raise bad(f"an `unanswerable` row must not set `{field}`")
        tags = row.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            raise bad("`tags` must be a list of strings")

        seen_ids.add(row_id)
        rows.append({
            "id": row_id,
            "question": question,
            "gold": gold,
            "answer_contains": expected or None,
            "answer_contains_all": expected_all or None,
            "tags": tags,
            "unanswerable": unanswerable,
        })
    return rows


def _is_gold(hit: dict, gold: list[dict]) -> bool:
    """True when a retrieved hit is one of the gold (pdf, page) pairs."""
    return any(
        hit.get("pdf") == g["pdf"] and hit.get("page_number") == g["page"] for g in gold
    )


def gold_rank(hits: list[dict], gold: list[dict]) -> int | None:
    """1-based rank of the first hit matching any gold {pdf, page}; None if absent."""
    for rank, hit in enumerate(hits, start=1):
        if _is_gold(hit, gold):
            return rank
    return None


def citation_correct(citation: dict | None, reranked: list[dict], gold: list[dict]) -> bool:
    """Did the answer's citation land on a gold page?

    `source_page` is a 1-based index into the RERANKED page list (the pages the
    answer step actually saw). Not-found / 0 / out-of-range all score False.
    """
    if not citation or not citation.get("found"):
        return False
    source_page = citation.get("source_page", 0)
    if not isinstance(source_page, int) or not (1 <= source_page <= len(reranked)):
        return False
    return _is_gold(reranked[source_page - 1], gold)


def abstention_correct(citation: dict | None) -> bool:
    """Did the system correctly decline to answer a question the corpus can't answer?

    True only when the citation reports `found` false - which includes the degraded
    not-found shape `answerer` falls back to. Scored over `unanswerable` rows only, so
    its complement is the hallucination rate: how often the pipeline invented an answer
    (and a bounding box) for a fact that is nowhere in the index.
    """
    return not (citation and citation.get("found"))


def gold_doc_coverage(hits: list[dict], gold: list[dict]) -> float | None:
    """Fraction of the distinct gold *documents* `hits` reached a gold page in.

    None (N/A) unless gold spans more than one pdf, so single-document questions drop
    out of the metric for free and a cross-document question needs no schema flag - the
    gold list already says whether it is one. This is the metric that puts RERANK_K
    under pressure: with gold in two pdfs and RERANK_K=2, scoring 1.0 means the
    reranker spent one of its two slots on each document rather than both on one.

    Deliberately stage-agnostic in `hits`, because run_eval scores it **twice** - once
    on the reranked set (`gold_doc_coverage`) and once on the untrimmed Qdrant
    candidates (`candidate_doc_coverage`). The pair is what attributes a coverage miss
    to a stage, which neither number can do alone:

        candidates 1.0, reranked <1.0     -> rerank lost it; the page was on the table
        candidates == reranked AND <1.0   -> retrieval-only loss; rerank was blameless
        candidates <1.0, reranked <cand   -> both stages failed (retrieval + rerank)

    Any decrease from candidate coverage to reranked coverage must be attributed to
    reranking (potentially alongside retrieval, not retrieval alone). Before this pair
    existed, every 0.5 read as one undifferentiated failure and there was no way to
    tell which stage to fix.
    """
    gold_pdfs = {g["pdf"] for g in gold}
    if len(gold_pdfs) < 2:
        return None
    covered = {hit.get("pdf") for hit in hits if _is_gold(hit, gold)}
    return round(len(covered) / len(gold_pdfs), 4)


# A comma grouping thousands, e.g. the one in "37,000". Removed from both sides of a
# substring check so a purely typographic difference isn't scored as a wrong answer:
# the model answered "37,000" where the label said "37000", which is the same fact.
_THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def _comparable(text: str) -> str:
    """Lowercase and drop thousands separators, so "37,000" and "37000" compare equal."""
    return _THOUSANDS_SEPARATOR.sub("", (text or "").lower())


def substring_match(
    answer: str, expected: list[str] | None, expected_all: list[str] | None = None
) -> bool | None:
    """Substring scoring: `expected` is any-of, `expected_all` is all-of; both must hold.

    None (N/A) when neither is given. Digit grouping is normalized away on both sides
    (see `_comparable`).

    The all-of form exists because any-of cannot score a question with two required
    facts. A cross-document row labelled `["128"]` passed on an answer that gave
    ColPali's dimension and then said it could not determine ColBERT's - so
    substring_accuracy read 1.0 on precisely the question type added to de-saturate it.
    Where both halves share a value and no substring can tell a whole answer from a
    half one, the row carries no expectation at all and scores N/A rather than a
    misleading pass; the judge is what covers it.
    """
    if not expected and not expected_all:
        return None
    lowered = _comparable(answer)
    if expected and not any(_comparable(s) in lowered for s in expected):
        return False
    return all(_comparable(s) in lowered for s in (expected_all or []))


MIN_CALIBRATION_N = 5
"""Rows a calibration slice needs on each side before its comparison is reported.

Not a statistical threshold, a defence against one specific way this eval lied. Every
calibration figure here is a *difference between two groups*, and the pipeline getting
better shrinks one of the groups toward zero - so the metric degrades exactly when the
system it measures improves. `baseline_decomposed.json` reported
`confidence_separation: -0.0062` to four decimals off **one** wrong citation in 73 rows,
and `baseline_diverse` / `baseline_swept` reported the same quantity with the opposite
sign off two and three. All three were noise being read as a finding.

`_rate` and `_mean` already withhold a figure whose denominator is empty; this is the
same rule at a threshold that is not zero. It applies only to the *comparisons* - the
per-group means keep reporting, because a mean over 1 row is still that row's value.
"""


def _rate(values: list) -> float | None:
    """Fraction of truthy values; None when no row was applicable."""
    if not values:
        return None
    return round(sum(1 for v in values if v) / len(values), 4)


def _floored_rate(values: list) -> float | None:
    """`_rate`, withheld below MIN_CALIBRATION_N (see that constant)."""
    return _rate(values) if len(values) >= MIN_CALIBRATION_N else None


def _separation(high: list[float], low: list[float]) -> float | None:
    """mean(high) - mean(low), withheld unless both sides clear MIN_CALIBRATION_N.

    Positive means the signal is informative, ~0 means it is noise, negative means it
    is actively misleading - but only once both groups are big enough to say so.
    """
    if len(high) < MIN_CALIBRATION_N or len(low) < MIN_CALIBRATION_N:
        return None
    hi, lo = _mean(high), _mean(low)
    return None if hi is None or lo is None else round(hi - lo, 4)


def _mean(values: list[float]) -> float | None:
    """Arithmetic mean; None when no row was applicable."""
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _calibration(rows: list[dict], out: dict) -> None:
    """Add the confidence-calibration metrics for one slice of rows, in place.

    A confidence signal is useful only if it *separates*: the confident cases should
    come out better than the unconfident ones. Report that separation directly rather
    than an invented calibration error. Every figure here is withheld below
    MIN_CALIBRATION_N, and every one ships with the counts it was computed from, so it
    can never again be quoted without its denominator.

    **Two pairings, against different labels and at very different sample sizes.**

    `confidence_separation` asks whether retrieval confidence on the *cited* page
    predicts whether that citation was right. Well-posed, and starved: its negative
    class is the pipeline's own mistakes, so it shrinks as the pipeline improves and
    costs a judged run to refresh. On the pinned baseline it is one row.

    `decisiveness_separation` asks the cheaper, higher-N version of the same question
    about the same formula - does a decisive slate mean retrieval's *top* page was
    right? The negative class is recall@1 misses, which are deterministic, need no API
    key, and number 24 where wrong citations number 1. This is the one that can
    actually adjudicate whether `src/confidence.py` carries information.

    `self_conf_*_acc` buckets citation accuracy by the model's own self-report. Read
    `self_conf_low_acc` with care even above the floor: `answerer._normalize` pins
    "low" onto every not-found answer, and a declined answerable row scores
    citation_correct=False, so that bucket is partly a tautology rather than a
    measurement of the model's calibration.
    """
    scored = [r for r in rows
              if r.get("citation_correct") is not None and r.get("retrieval_confidence") is not None]
    right = [r["retrieval_confidence"] for r in scored if r["citation_correct"]]
    wrong = [r["retrieval_confidence"] for r in scored if not r["citation_correct"]]
    # The per-group means keep reporting unfloored: a mean over one row is still that
    # row's value, and hiding it would hide the evidence for why the difference is gone.
    out["retrieval_conf_correct_avg"] = _mean(right)
    out["retrieval_conf_wrong_avg"] = _mean(wrong)
    out["n_conf_correct"] = len(right)
    out["n_conf_wrong"] = len(wrong)
    out["confidence_separation"] = _separation(right, wrong)

    # `gold_rank` is absent on unanswerable rows by construction (run_full omits it so a
    # correct refusal can never read as a retrieval miss), which also keeps them out of
    # this denominator - they have no gold page whose rank could be hit or missed.
    #
    # Answerable rows with `gold_rank: None` (key present, value None) are retrieval
    # failures and MUST count as misses, not be excluded. Only rows where the key is
    # completely absent (unanswerable rows) should be excluded.
    #
    # Fused (query-decomposed, RRF scores) vs unfused (single-query, raw MaxSim scores)
    # use incompatible score scales, so their decisiveness values must be tracked
    # separately. Infer fusion from the top candidate's score: RRF scores are < 1.0
    # (typically 0.02-0.03), MaxSim scores are >= 1.0 (typically 8-22).
    decisive = [r for r in rows
                if "gold_rank" in r and r.get("top1_decisiveness") is not None]

    def _is_fused(row: dict) -> bool:
        """True when the row used query decomposition (RRF scores), False for raw MaxSim."""
        pages = row.get("candidate_pages") or row.get("reranked_pages", [])
        if not pages or not pages[0].get("score"):
            return False
        return pages[0]["score"] < 1.0

    fused = [r for r in decisive if _is_fused(r)]
    unfused = [r for r in decisive if not _is_fused(r)]

    fused_hit = [r["top1_decisiveness"] for r in fused if r["gold_rank"] == 1]
    fused_miss = [r["top1_decisiveness"] for r in fused if r["gold_rank"] != 1]
    unfused_hit = [r["top1_decisiveness"] for r in unfused if r["gold_rank"] == 1]
    unfused_miss = [r["top1_decisiveness"] for r in unfused if r["gold_rank"] != 1]

    # Legacy fields: average across both populations (kept for compatibility, but mixing scales)
    hit = [r["top1_decisiveness"] for r in decisive if r["gold_rank"] == 1]
    miss = [r["top1_decisiveness"] for r in decisive if r["gold_rank"] != 1]
    out["decisiveness_hit_avg"] = _mean(hit)
    out["decisiveness_miss_avg"] = _mean(miss)
    out["n_decisive_hit"] = len(hit)
    out["n_decisive_miss"] = len(miss)
    out["decisiveness_separation"] = _separation(hit, miss)

    # Separated by fusion status: fused (RRF) vs unfused (MaxSim)
    out["decisiveness_fused_hit_avg"] = _mean(fused_hit)
    out["decisiveness_fused_miss_avg"] = _mean(fused_miss)
    out["n_decisive_fused_hit"] = len(fused_hit)
    out["n_decisive_fused_miss"] = len(fused_miss)
    out["decisiveness_fused_separation"] = _separation(fused_hit, fused_miss)

    out["decisiveness_unfused_hit_avg"] = _mean(unfused_hit)
    out["decisiveness_unfused_miss_avg"] = _mean(unfused_miss)
    out["n_decisive_unfused_hit"] = len(unfused_hit)
    out["n_decisive_unfused_miss"] = len(unfused_miss)
    out["decisiveness_unfused_separation"] = _separation(unfused_hit, unfused_miss)

    for level in ("high", "medium", "low"):
        bucket = [r["citation_correct"] for r in rows
                  if r.get("self_confidence") == level and r.get("citation_correct") is not None]
        out[f"self_conf_{level}_acc"] = _floored_rate(bucket)
        out[f"n_self_conf_{level}"] = len(bucket)


def _metrics(rows: list[dict], ks: tuple) -> dict:
    """Aggregate one slice of rows into rates, computed over applicable rows only."""
    ranked = [r["gold_rank"] for r in rows if "gold_rank" in r]
    judges = [r["judge"] for r in rows if r.get("judge") is not None]
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    coverage = [r["gold_doc_coverage"] for r in rows if r.get("gold_doc_coverage") is not None]
    cand_coverage = [r["candidate_doc_coverage"] for r in rows
                     if r.get("candidate_doc_coverage") is not None]
    out: dict = {"n": len(rows)}
    for k in ks:
        out[f"recall@{k}"] = _rate([rank is not None and rank <= k for rank in ranked]) if ranked else None
    for key, metric in (
        ("rerank_hit", "rerank_recall"),
        ("citation_correct", "citation_accuracy"),
        ("substring_match", "substring_accuracy"),
        ("abstention_correct", "abstention_accuracy"),
    ):
        out[metric] = _rate([r[key] for r in rows if r.get(key) is not None])
    out["gold_coverage_avg"] = _mean(coverage)
    # Its retrieval-stage twin. The gap between the two is the share of coverage the
    # rerank step throws away: equal means retrieval is the ceiling, and no rerank
    # change can move `gold_coverage_avg` at all.
    out["candidate_coverage_avg"] = _mean(cand_coverage)
    out["judge_accuracy"] = _rate([j["correct"] for j in judges])
    out["judge_score_avg"] = round(sum(j["score"] for j in judges) / len(judges), 2) if judges else None
    _calibration(rows, out)
    out["avg_latency_ms"] = round(sum(latencies) / len(latencies), 1) if latencies else None
    return out


def aggregate(rows: list[dict], ks: tuple = (1, 3, 10)) -> dict:
    """Summary rates over applicable rows only, plus the same metrics per tag."""
    tags = sorted({t for r in rows for t in r.get("tags", [])})
    return {
        **_metrics(rows, ks),
        "per_tag": {t: _metrics([r for r in rows if t in r.get("tags", [])], ks) for t in tags},
    }


def _cell(value) -> str:
    """Render one table cell: None as '-', bools as Y/N, everything else as str."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "Y" if value else "N"
    return str(value)


def format_table(rows: list[dict], summary: dict) -> str:
    """Plain-text report: one line per question, then summary + per-tag blocks."""
    # cand_cov sits immediately left of cov so the attribution pair reads together:
    # a row showing `1.0  0.5` lost its second document at rerank, `0.5  0.5` at retrieval.
    headers = ["id", "gold_rank", "rerank", "cite", "cand_cov", "cov",
               "substr", "abst", "judge", "latency_ms"]
    body = []
    for r in rows:
        judge = r.get("judge")
        body.append([
            r["id"],
            _cell(r.get("gold_rank")),
            _cell(r.get("rerank_hit")),
            _cell(r.get("citation_correct")),
            _cell(r.get("candidate_doc_coverage")),
            _cell(r.get("gold_doc_coverage")),
            _cell(r.get("substring_match")),
            _cell(r.get("abstention_correct")),
            _cell(None if judge is None else judge.get("correct")),
            _cell(r.get("latency_ms")),
        ])
    widths = [max(len(h), *(len(row[i]) for row in body)) if body else len(h)
              for i, h in enumerate(headers)]
    lines = [
        "  ".join(h.ljust(w) for h, w in zip(headers, widths)),
        "  ".join("-" * w for w in widths),
    ]
    lines += ["  ".join(cell.ljust(w) for cell, w in zip(row, widths)) for row in body]

    def block(title: str, metrics: dict) -> list[str]:
        """One `label: k=v  k=v` summary line, excluding the nested per-tag block."""
        pairs = [f"{k}={_cell(v)}" for k, v in metrics.items() if k != "per_tag"]
        return [f"{title}: " + "  ".join(pairs)]

    lines += [""] + block("summary", summary)
    for tag, metrics in summary.get("per_tag", {}).items():
        lines += block(f"  tag:{tag}", metrics)
    return "\n".join(lines)
