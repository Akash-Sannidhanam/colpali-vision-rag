"""Pure scoring logic for the eval harness - plain dicts in, plain values out.

No imports from `src.` and no I/O, so the whole module is unit-testable without
models, Qdrant, or an API key (tests/test_eval_scoring.py). run_eval.py owns the
orchestration: it runs the pipeline, builds one row dict per question from these
scorers, then aggregates and renders.

Row shape consumed by `aggregate` / `format_table` (keys absent or None = N/A,
excluded from that metric's denominator):
    id, tags, gold_rank, rerank_hit, citation_correct, gold_doc_coverage,
    substring_match, abstention_correct, retrieval_confidence, self_confidence,
    judge {correct, score}, latency_ms

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


def gold_doc_coverage(reranked: list[dict], gold: list[dict]) -> float | None:
    """Fraction of the distinct gold *documents* the reranked set reached a gold page in.

    None (N/A) unless gold spans more than one pdf, so single-document questions drop
    out of the metric for free and a cross-document question needs no schema flag - the
    gold list already says whether it is one. This is the metric that puts RERANK_K
    under pressure: with gold in two pdfs and RERANK_K=2, scoring 1.0 means the
    reranker spent one of its two slots on each document rather than both on one.
    """
    gold_pdfs = {g["pdf"] for g in gold}
    if len(gold_pdfs) < 2:
        return None
    covered = {hit.get("pdf") for hit in reranked if _is_gold(hit, gold)}
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


def _rate(values: list) -> float | None:
    """Fraction of truthy values; None when no row was applicable."""
    if not values:
        return None
    return round(sum(1 for v in values if v) / len(values), 4)


def _mean(values: list[float]) -> float | None:
    """Arithmetic mean; None when no row was applicable."""
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _calibration(rows: list[dict], out: dict) -> None:
    """Add the confidence-calibration metrics for one slice of rows, in place.

    A confidence signal is useful only if it *separates*: correct citations should
    carry higher confidence than wrong ones. Report that separation directly rather
    than an invented calibration error, for both signals the pipeline produces - the
    deterministic retrieval decisiveness (`src/confidence.py`) and the model's own
    high/medium/low self-report. `confidence_separation` is the headline: positive
    means the signal is informative, ~0 means it is noise, negative means it is
    actively misleading.
    """
    scored = [r for r in rows
              if r.get("citation_correct") is not None and r.get("retrieval_confidence") is not None]
    right = [r["retrieval_confidence"] for r in scored if r["citation_correct"]]
    wrong = [r["retrieval_confidence"] for r in scored if not r["citation_correct"]]
    correct_avg, wrong_avg = _mean(right), _mean(wrong)
    out["retrieval_conf_correct_avg"] = correct_avg
    out["retrieval_conf_wrong_avg"] = wrong_avg
    # Needs both sides to mean anything - a slice with no wrong citations has no
    # separation to report, which is exactly the saturated case this eval exists to avoid.
    out["confidence_separation"] = (
        round(correct_avg - wrong_avg, 4)
        if correct_avg is not None and wrong_avg is not None else None
    )
    for level in ("high", "medium", "low"):
        out[f"self_conf_{level}_acc"] = _rate(
            [r["citation_correct"] for r in rows
             if r.get("self_confidence") == level and r.get("citation_correct") is not None]
        )


def _metrics(rows: list[dict], ks: tuple) -> dict:
    """Aggregate one slice of rows into rates, computed over applicable rows only."""
    ranked = [r["gold_rank"] for r in rows if "gold_rank" in r]
    judges = [r["judge"] for r in rows if r.get("judge") is not None]
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    coverage = [r["gold_doc_coverage"] for r in rows if r.get("gold_doc_coverage") is not None]
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
    headers = ["id", "gold_rank", "rerank", "cite", "cov", "substr", "abst", "judge", "latency_ms"]
    body = []
    for r in rows:
        judge = r.get("judge")
        body.append([
            r["id"],
            _cell(r.get("gold_rank")),
            _cell(r.get("rerank_hit")),
            _cell(r.get("citation_correct")),
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
