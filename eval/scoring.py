"""Pure scoring logic for the eval harness - plain dicts in, plain values out.

No imports from `src.` and no I/O, so the whole module is unit-testable without
models, Qdrant, or an API key (tests/test_eval_scoring.py). run_eval.py owns the
orchestration: it runs the pipeline, builds one row dict per question from these
scorers, then aggregates and renders.

Row shape consumed by `aggregate` / `format_table` (keys absent or None = N/A,
excluded from that metric's denominator):
    id, tags, gold_rank, rerank_hit, citation_correct, substring_match,
    judge {correct, score}, latency_ms
"""

import json


def load_dataset(lines) -> list[dict]:
    """Parse dataset.jsonl lines into validated row dicts.

    Each non-blank line must be a JSON object with a unique `id`, a non-empty
    `question`, and a non-empty `gold` list of {pdf, page>=1}; `answer_contains`
    (list of substrings) and `tags` are optional. Raises ValueError naming the
    1-based line number of the first bad row.
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
        gold = row.get("gold")
        if not isinstance(gold, list) or not gold:
            raise bad("`gold` must be a non-empty list of {pdf, page}")
        for g in gold:
            if not isinstance(g, dict) or not isinstance(g.get("pdf"), str):
                raise bad("each gold entry needs a string `pdf`")
            page = g.get("page")
            if not isinstance(page, int) or isinstance(page, bool) or page < 1:
                raise bad("each gold entry needs an integer `page` >= 1")
        expected = row.get("answer_contains")
        if expected is not None and (
            not isinstance(expected, list)
            or not all(isinstance(s, str) and s for s in expected)
        ):
            raise bad("`answer_contains` must be a list of non-empty strings")
        tags = row.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            raise bad("`tags` must be a list of strings")

        seen_ids.add(row_id)
        rows.append({
            "id": row_id,
            "question": question,
            "gold": gold,
            "answer_contains": expected or None,
            "tags": tags,
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


def substring_match(answer: str, expected: list[str] | None) -> bool | None:
    """Case-insensitive any-of substring check; None (N/A) when nothing is expected."""
    if not expected:
        return None
    lowered = (answer or "").lower()
    return any(s.lower() in lowered for s in expected)


def _rate(values: list) -> float | None:
    """Fraction of truthy values; None when no row was applicable."""
    if not values:
        return None
    return round(sum(1 for v in values if v) / len(values), 4)


def _metrics(rows: list[dict], ks: tuple) -> dict:
    """Aggregate one slice of rows into rates, computed over applicable rows only."""
    ranked = [r["gold_rank"] for r in rows if "gold_rank" in r]
    judges = [r["judge"] for r in rows if r.get("judge") is not None]
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    out: dict = {"n": len(rows)}
    for k in ks:
        out[f"recall@{k}"] = _rate([rank is not None and rank <= k for rank in ranked]) if ranked else None
    for key, metric in (
        ("rerank_hit", "rerank_recall"),
        ("citation_correct", "citation_accuracy"),
        ("substring_match", "substring_accuracy"),
    ):
        out[metric] = _rate([r[key] for r in rows if r.get(key) is not None])
    out["judge_accuracy"] = _rate([j["correct"] for j in judges])
    out["judge_score_avg"] = round(sum(j["score"] for j in judges) / len(judges), 2) if judges else None
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
    headers = ["id", "gold_rank", "rerank", "cite", "substr", "judge", "latency_ms"]
    body = []
    for r in rows:
        judge = r.get("judge")
        body.append([
            r["id"],
            _cell(r.get("gold_rank")),
            _cell(r.get("rerank_hit")),
            _cell(r.get("citation_correct")),
            _cell(r.get("substring_match")),
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
