"""Eval harness CLI - the regression guard (PRODUCTION_HARDENING.md Phase 4).

Runs every question in eval/dataset.jsonl and scores retrieval recall@k,
rerank recall, citation correctness, and answer quality, then writes a JSON
report whose `config` snapshot makes before/after diffs meaningful. Re-run after
changing DPI, RERANK_K, or a model to *prove* no regression.

    PYTHONPATH=. uv run python eval/run_eval.py                    # full pipeline
    PYTHONPATH=. uv run python eval/run_eval.py --retrieval-only   # no GEMINI_API_KEY
    PYTHONPATH=. uv run python eval/run_eval.py --judge            # + LLM-as-judge
    PYTHONPATH=. uv run python eval/run_eval.py --fail-under-recall 0.9

Exit codes: 0 = ran and report written; 1 = a --gate / --fail-under-recall threshold
breached; 2 = setup error (bad dataset, corpus not ingested, Qdrant down, missing key)
*or* a degraded run - too many Gemini calls fell through to graceful degradation for
the numbers to mean anything (see --max-degraded-frac and `degradation_summary`).
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from eval.scoring import (
    abstention_correct,
    aggregate,
    citation_correct,
    format_table,
    gold_doc_coverage,
    gold_rank,
    load_dataset,
    substring_match,
)
from src import config
from src.config import EVAL_JUDGE_MODEL, RERANK_K, RETRIEVE_K
from src.gemini_client import generate
from src.logging_setup import get_logger
from src.vector_store import close_client, list_documents, ping

log = get_logger("eval")

DEFAULT_DATASET = Path(__file__).parent / "dataset.jsonl"
DEFAULT_REPORT_DIR = Path(__file__).parent / "reports"


class EvalSetupError(Exception):
    """Anything that should stop the run before scoring begins (exit code 2)."""


class JudgeVerdict(BaseModel):
    """LLM-as-judge grade for one answer against the reference facts."""

    correct: bool    # does the answer assert the reference fact?
    score: int       # 1-5 overall quality (completeness / clarity)
    reasoning: str   # one sentence


_JUDGE_PROMPT = (
    """You are grading an answer produced by a document QA system.

    Question: {question}
    Reference facts the answer must assert: {reference}
    System answer: {answer}

    Grade strictly: `correct` is true only if the answer states the reference
    fact (paraphrase and unit formatting are fine; a different number or entity
    is not). `score` rates overall quality 1-5 (completeness, clarity), and
    `reasoning` is one short sentence.""")


def check_corpus(dataset: list[dict]) -> None:
    """Fail fast if any gold page can't exist in the live index."""
    docs = {d["pdf"]: d["page_count"] for d in list_documents()}
    for row in dataset:
        for g in row["gold"]:
            if g["pdf"] not in docs:
                raise EvalSetupError(
                    f"gold pdf {g['pdf']!r} (question {row['id']!r}) is not in the index; "
                    f"ingest it first: PYTHONPATH=. uv run python src/ingest.py"
                )
            if g["page"] > docs[g["pdf"]]:
                raise EvalSetupError(
                    f"question {row['id']!r} labels page {g['page']} of {g['pdf']!r}, "
                    f"but the index holds only {docs[g['pdf']]} pages"
                )


def judge_answer(question: str, answer: str, expected: list[str] | None, gold: list[dict]) -> dict | None:
    """Score one answer with the judge model; None (N/A) on any failure."""
    reference = "; ".join(expected or []) or "none given - judge only faithfulness to the question"
    gold_sources = ', '.join(f"{g['pdf']} p.{g['page']}" for g in gold)
    reference += f" (read from {gold_sources})"
    try:
        response = generate(
            model=EVAL_JUDGE_MODEL,
            contents=[_JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)],
            response_schema=JudgeVerdict,
            purpose="judge",
        )
        parsed = response.parsed
        if parsed is None:
            parsed = JudgeVerdict(**json.loads(response.text))
        return parsed.model_dump()
    except Exception:
        log.warning(
            "judge call failed; marking judge N/A",
            exc_info=True,
            extra={"degraded": True, "stage": "judge"},
        )
        return None


def _pages_of(hits: list[dict]) -> list[dict]:
    """The `[{pdf, page}, ...]` identity of a hit list, in order, without the payload.

    Stored per row so a later change to the gold labels is re-scorable **offline**:
    `gold_rank`, `rerank_hit`, `gold_doc_coverage` and `candidate_doc_coverage` are all
    pure functions of (pages, gold), and `cited` / `answer` are already kept. Without
    these two lists, re-labelling one question costs a full ~25-minute judged re-run to
    learn what the pipeline had already done. Costs ~45 KB on a 192 KB report.
    """
    return [{"pdf": h["pdf"], "page": h["page_number"]} for h in hits]


def run_retrieval_only(dataset: list[dict]) -> list[dict]:
    """Retrieve per question - no Gemini modules touched, no API key needed.

    Goes through the same `src.retrieval.retrieve` seam the graph's retrieve node
    uses. Embedding and searching here directly is what once let a retrieval-policy
    change (query decomposition) move the pipeline while this mode went on measuring
    the old behaviour.
    """
    from src.retrieval import retrieve

    rows = []
    for item in dataset:
        if item["unanswerable"]:
            # Nothing here is scorable without the answer step: there is no gold page to
            # rank, and abstention is a property of the citation. Keep the row so `n`
            # still matches the dataset, but omit `gold_rank` - which is what excludes it
            # from every recall denominator. Scoring it as rank=None would count a
            # correctly-unanswerable question as a retrieval miss.
            rows.append({"id": item["id"], "tags": item["tags"], "unanswerable": True})
            continue
        hits = retrieve(item["question"])
        rank = gold_rank(hits, item["gold"])
        # The retrieval top-1 makes a recall@1 miss auditable: it shows *which* page
        # out-ranked the gold page (kept in the report row, not the summary table).
        top1 = {"pdf": hits[0]["pdf"], "page": hits[0]["page_number"]} if hits else None
        row = {
            "id": item["id"], "tags": item["tags"], "gold_rank": rank, "top1": top1,
            # Scored here too, not only in run_full, because retrieval is the half of
            # the attribution that costs nothing: this mode needs no API key and no
            # Gemini spend, and retrieval is deterministic, so the number carries none
            # of the rerank/answer run-to-run variance the full mode does.
            "candidate_doc_coverage": gold_doc_coverage(hits, item["gold"]),
            "candidate_pages": _pages_of(hits),
        }
        rows.append(row)
        log.info("eval question scored",
                 extra={"eval_id": item["id"], "gold_rank": rank, "top1": top1})
    return rows


def run_full(dataset: list[dict], use_judge: bool) -> list[dict]:
    """Full pipeline per question via the run_query seam; scores all metric families.

    Answerable and unanswerable questions produce deliberately different row shapes.
    An unanswerable row carries only `abstention_correct`, omitting gold_rank /
    rerank_hit / citation_correct / substring_match entirely, so it can never enter a
    denominator where "no gold page was retrieved" would read as a failure.
    """
    from src.main import run_query

    rows = []
    for item in dataset:
        result = run_query(item["question"])
        reranked = result.get("retrieved", [])
        citation = result.get("citation")
        answer = result.get("answer", "")
        meta = result.get("meta", {})
        row = {
            "id": item["id"],
            "tags": item["tags"],
            "found": bool(citation and citation.get("found")),
            "latency_ms": meta.get("latency_ms"),
        }
        if item["unanswerable"]:
            # The only correct behaviour is declining; the judge is skipped because its
            # prompt assumes a reference fact exists.
            row["abstention_correct"] = abstention_correct(citation)
        else:
            source_page = (citation or {}).get("source_page", 0)
            cited = (
                {"pdf": reranked[source_page - 1]["pdf"],
                 "page": reranked[source_page - 1]["page_number"]}
                if 1 <= source_page <= len(reranked) else None
            )
            candidates = result.get("candidates", [])
            row |= {
                "gold_rank": gold_rank(candidates, item["gold"]),
                "rerank_hit": gold_rank(reranked, item["gold"]) is not None,
                "citation_correct": citation_correct(citation, reranked, item["gold"]),
                # The attribution pair. Scored on the same gold with the same function,
                # one stage apart: candidates 1.0 with coverage <1.0 means the second
                # document was in the top-k and the reranker dropped it; equal coverage
                # with both <1.0 means retrieval never offered it and rerank was blameless;
                # any decrease from candidate to reranked coverage is attributed to reranking
                # (potentially alongside retrieval, not retrieval alone).
                "candidate_doc_coverage": gold_doc_coverage(candidates, item["gold"]),
                "gold_doc_coverage": gold_doc_coverage(reranked, item["gold"]),
                "cited": cited,  # which page was actually cited - makes gold-label gaps auditable
                "candidate_pages": _pages_of(candidates),
                "reranked_pages": _pages_of(reranked),
                "substring_match": substring_match(
                    answer, item["answer_contains"], item["answer_contains_all"]
                ),
                # Both confidence signals ride along on the result already (main.run_query),
                # so calibration costs no extra call - only the scoring in eval.scoring.
                "retrieval_confidence": meta.get("retrieval_confidence"),
                "self_confidence": (citation or {}).get("confidence"),
            }
            if use_judge:
                # Both label kinds are reference facts. Passing only `answer_contains`
                # handed the judge an empty reference for every row labelled purely
                # all-of, silently downgrading it to a faithfulness-only grade on
                # exactly the cross-document rows that need the strictest one.
                row["judge"] = judge_answer(
                    item["question"], answer,
                    (item["answer_contains"] or []) + (item["answer_contains_all"] or []),
                    item["gold"],
                )
        # The report keeps the answer + full meta for debugging; the table doesn't show them.
        rows.append({**row, "answer": answer, "meta": meta})
        log.info("eval question scored", extra={"eval_id": item["id"],
                                                "gold_rank": row.get("gold_rank"),
                                                "citation_correct": row.get("citation_correct"),
                                                "abstention_correct": row.get("abstention_correct")})
    return rows


def _repo_relative(path: str) -> str:
    """Path relative to the repo root when it lives inside it, else unchanged.

    The default dataset path is absolute, so storing it verbatim baked the operator's
    home directory into the *committed* baseline report and made snapshots differ per
    machine for no reason.
    """
    resolved = Path(path).resolve()
    root = Path(__file__).resolve().parent.parent
    return str(resolved.relative_to(root)) if resolved.is_relative_to(root) else str(resolved)


def _config_snapshot(mode: str, dataset_path: str, use_judge: bool) -> dict:
    """The knob values this run used, embedded in the report so runs diff cleanly."""
    return {
        "mode": mode,
        "dataset": _repo_relative(dataset_path),
        "retrieve_k": RETRIEVE_K,
        "rerank_k": RERANK_K,
        "rerank_adaptive": config.RERANK_ADAPTIVE,
        "max_pages_per_doc": config.MAX_PAGES_PER_DOC,
        "candidate_fanout": config.CANDIDATE_FANOUT,
        # Both here for the reason the slate knobs are: without them diff_reports.py
        # prints "config changes: none" on precisely the comparison this pass exists
        # to make.
        "query_decompose": config.QUERY_DECOMPOSE,
        "max_subqueries": config.MAX_SUBQUERIES,
        "decompose_original_weight": config.DECOMPOSE_ORIGINAL_WEIGHT,
        "rescore_oversampling": config.RESCORE_OVERSAMPLING,
        "gemini_model": config.GEMINI_MODEL,
        "rerank_model": config.RERANK_MODEL,
        "eval_judge_model": EVAL_JUDGE_MODEL if use_judge else None,
        "render_dpi": config.RENDER_DPI,
        "colpali_model": config.COLPALI_MODEL,
        # Unlike EMBED_BATCH_SIZE (which is vector-identical and so deliberately absent),
        # this changes every stored vector and therefore what retrieval can find. It also
        # needs its own index, so `collection` records which one the arm actually read -
        # the one config difference that could otherwise make two arms look identical.
        "embed_visual_tokens": config.EMBED_VISUAL_TOKENS,
        "collection": config.COLLECTION_NAME,
        "qdrant_mode": "server" if config.QDRANT_URL else "embedded",
    }


def degradation_summary(rows: list[dict], use_judge: bool) -> dict:
    """Count the Gemini calls in this run that fell through to graceful degradation.

    Two sources, both already recorded: `meta.degraded_calls` (threaded up from
    `request_context.record_degraded` in the rerank/answer nodes) and, when judging, a
    `judge` of None - `judge_answer` returns None only on failure, so a missing verdict
    *is* a failed call and needs no separate counter. Unanswerable rows are never
    judged by design, so their absent verdict is not degradation.

    `degraded_calls` counts calls, `rows_with_degraded` counts questions: one question
    can fail twice, and it is the share of *questions* that says whether the run is
    still measuring anything. Retrieval-only rows carry no `meta` and touch no Gemini,
    so they contribute zero without a special case.

    Kept pure so the guard is testable without an API key (tests/test_run_eval.py).
    """
    degraded_calls = 0
    judge_na = 0
    affected = 0
    for row in rows:
        row_calls = (row.get("meta") or {}).get("degraded_calls", 0) or 0
        row_judge_na = (
            use_judge
            and not row.get("unanswerable")
            and "abstention_correct" not in row
            and row.get("judge", ...) is None
        )
        degraded_calls += row_calls
        judge_na += int(row_judge_na)
        affected += int(bool(row_calls) or row_judge_na)
    return {
        "degraded_calls": degraded_calls,
        "rows_with_degraded": affected,
        "judge_na": judge_na,
        "degraded_row_frac": round(affected / len(rows), 4) if rows else 0.0,
    }


def degradation_breached(degradation: dict, limit: float) -> bool:
    """True when too many rows degraded for the report to be worth pinning.

    Strictly above the limit, so the default tolerates the odd transient 429 that
    outlived its retries while still catching a real outage.
    """
    return degradation["degraded_row_frac"] > limit


def gate_status(summary: dict, metric: str, threshold: float | None) -> tuple[bool, object]:
    """Evaluate one gate against a chosen summary metric.

    Returns (failed, value). `failed` is False when no threshold is set. A metric
    absent from the summary (unknown name, or N/A for the run) has value None and
    fails the gate, so a typo can't silently pass. Kept pure so it's unit-testable
    without running the pipeline (tests/test_run_eval.py).
    """
    if threshold is None:
        return False, None
    value = summary.get(metric)
    failed = not isinstance(value, (int, float)) or value < threshold
    return failed, value


def parse_gates(
    specs: list[str] | None, fail_metric: str, fail_under: float | None
) -> list[tuple[str, float]]:
    """Build the (metric, minimum) list from repeated --gate plus the legacy flag pair.

    One run costs minutes of GPU and a few hundred model calls, so a guard that can
    only watch a single metric per run isn't much of a guard - the metrics with real
    headroom (gold_coverage_avg, recall@1, citation_accuracy) need watching together.
    Raises ValueError naming the bad spec, so a typo fails at parse time rather than
    after the whole eval has run.
    """
    gates: list[tuple[str, float]] = []
    for spec in specs or []:
        metric, sep, raw = spec.rpartition(":")
        if not sep or not metric.strip():
            raise ValueError(f"--gate {spec!r} must look like METRIC:MIN, e.g. recall@1:0.70")
        try:
            minimum = float(raw)
        except ValueError as exc:
            raise ValueError(f"--gate {spec!r}: {raw!r} is not a number") from exc
        # float() happily parses "nan", and `value < nan` is always False - a gate that
        # can never fire, which is worse than no gate at all given this one is the whole
        # point of the fail-closed design.
        if not math.isfinite(minimum):
            raise ValueError(f"--gate {spec!r}: {raw!r} is not a finite threshold")
        gates.append((metric.strip(), minimum))
    if fail_under is not None:
        # argparse's type=float parses "nan" just as happily, and this path skipped the
        # check above - guarding one of two entry points to the same list is no guard.
        if not math.isfinite(fail_under):
            raise ValueError(f"--fail-under-recall {fail_under!r} is not a finite threshold")
        gates.append((fail_metric, fail_under))
    return gates


def main(argv: list[str] | None = None) -> int:
    """Parse args, run the requested eval mode, write the report; return the exit code."""
    parser = argparse.ArgumentParser(description="Score the RAG pipeline against the labeled dataset.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=None, help="report path (default eval/reports/eval_<utc>.json)")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="recall@k only - no Gemini calls, runs without GEMINI_API_KEY")
    parser.add_argument("--judge", action="store_true",
                        help="also score answers with the LLM judge (EVAL_JUDGE_MODEL)")
    parser.add_argument("--gate", action="append", metavar="METRIC:MIN",
                        help="exit 1 if METRIC falls below MIN; repeatable, e.g. "
                             "--gate recall@1:0.70 --gate gold_coverage_avg:0.60")
    parser.add_argument("--fail-under-recall", type=float, default=None, metavar="RATE",
                        help="exit 1 if the gate metric (see --fail-metric) falls below RATE")
    parser.add_argument("--fail-metric", default=f"recall@{RETRIEVE_K}", metavar="METRIC",
                        help="summary metric the --fail-under-recall threshold applies to, "
                             "e.g. recall@1, recall@3, citation_accuracy "
                             f"(default recall@{RETRIEVE_K})")
    parser.add_argument("--max-degraded-frac", type=float, default=0.02, metavar="FRAC",
                        help="exit 2 without a pinnable report when more than this share of "
                             "rows had a Gemini call degrade (default 0.02, ~1-2 rows in 83)")
    args = parser.parse_args(argv)
    if args.retrieval_only and args.judge:
        parser.error("--judge needs the full pipeline; drop --retrieval-only")
    try:
        gates = parse_gates(args.gate, args.fail_metric, args.fail_under_recall)
    except ValueError as exc:
        parser.error(str(exc))  # bad spec now, not after a multi-minute run

    mode = "retrieval-only" if args.retrieval_only else "full"
    try:
        dataset = load_dataset(Path(args.dataset).read_text().splitlines())
        if not dataset:
            raise EvalSetupError(f"dataset {args.dataset} has no rows")
        if not args.retrieval_only:
            config.validate()  # fail fast on a missing GEMINI_API_KEY
        ping()
        check_corpus(dataset)
        log.info("eval starting", extra={"mode": mode, "questions": len(dataset)})
        rows = run_retrieval_only(dataset) if args.retrieval_only else run_full(dataset, args.judge)
    except (EvalSetupError, OSError, ValueError, RuntimeError) as exc:
        print(f"eval setup error: {exc}", file=sys.stderr)
        return 2
    finally:
        close_client()

    ks = tuple(sorted({1, 3, RETRIEVE_K}))
    summary = aggregate(rows, ks=ks)
    per_tag = summary.pop("per_tag")
    degradation = degradation_summary(rows, args.judge)
    degraded_run = degradation_breached(degradation, args.max_degraded_frac)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": _config_snapshot(mode, args.dataset, args.judge),
        # Always present, zeros and all: a report that doesn't say how much of the run
        # reached the model can't be told apart from one written before this existed.
        "degradation": degradation,
        "degraded_run": degraded_run,
        "summary": summary,
        "per_tag": per_tag,
        "rows": rows,
    }
    # A degraded run is written but deliberately hard to mistake for a baseline: a
    # depleted quota degrades every answer to a not-found citation, which scores
    # abstention_accuracy 1.0 while citation and substring read 0.0. The stamp and the
    # filename are the difference between an artifact for diagnosis and a pinnable one.
    stem = "degraded" if degraded_run else "eval"
    output = Path(args.output) if args.output else (
        DEFAULT_REPORT_DIR / f"{stem}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")

    print(format_table(rows, {**summary, "per_tag": per_tag}))
    print(f"\nreport: {output}")

    if degraded_run:
        print(
            f"DEGRADED RUN: {degradation['rows_with_degraded']}/{len(rows)} rows "
            f"({degradation['degraded_row_frac']:.1%}) had a Gemini call degrade "
            f"[{degradation['degraded_calls']} pipeline calls, {degradation['judge_na']} judge]. "
            f"Above --max-degraded-frac {args.max_degraded_frac}. Do NOT pin this as a baseline; "
            f"these numbers measure the outage, not the pipeline.",
            file=sys.stderr,
        )
        # Gates are skipped, not evaluated-and-passed: thresholds applied to a run that
        # never reached the model would report a regression that isn't one - or worse,
        # pass, because the metrics an outage flatters are the ones gates watch.
        return 2

    # Report every breached gate, not just the first: one run is expensive enough that
    # finding out about the second regression should not cost another one.
    breaches = [(metric, threshold, gate_status(summary, metric, threshold)[1])
                for metric, threshold in gates
                if gate_status(summary, metric, threshold)[0]]
    for metric, threshold, value in breaches:
        print(f"FAIL: {metric}={value} below threshold {threshold}", file=sys.stderr)
    return 1 if breaches else 0


if __name__ == "__main__":
    sys.exit(main())
