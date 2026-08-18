"""Re-score a stored eval report against changed labels, without running the pipeline.

A judged run costs ~25 minutes and a few hundred Gemini calls. Most of what it produces
is not a function of the model at all: `gold_rank`, `rerank_hit`, `citation_correct`,
both document coverages and `substring_match` are pure functions of (what was retrieved,
what the label says), and every answerable row already stores what was retrieved -
`candidate_pages` and `reranked_pages`, with the cited page resolved into `cited`.

So a gold-label fix is re-scorable offline, for free. That property was claimed in the
docs from the moment `candidate_pages` was added; this is the tool that makes it real.
It exists because auditing labels is the single most common reason to want new numbers
here, and under-labeled gold has caused more apparent regressions than the pipeline has.

**What this cannot do.** The answer text is fixed, so anything that would change it -
a reworded question, a knob, a model - needs a live run. Two consequences worth stating
because they are easy to get wrong:

  * A **judge verdict cannot be recomputed** (it is an API call, and its prompt embeds
    the gold labels). Verdicts are carried over unchanged, and every row whose labels
    moved *and* which carries a verdict is listed as stale, because `judge_accuracy`
    then mixes two label sets. Rescoring is honest about a stale judge; it does not
    pretend to fix one.
  * The output is stamped `rescored` and never written to the default report name, so a
    derived report cannot quietly become a baseline.

Usage:

    PYTHONPATH=. uv run python eval/rescore.py eval/reports/baseline_decomposed.json \\
        --output eval/reports/relabeled.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from eval.scoring import (
    aggregate,
    citation_correct,
    gold_doc_coverage,
    gold_rank,
    load_dataset,
    substring_match,
)

DEFAULT_DATASET = Path(__file__).resolve().parent / "dataset.jsonl"

# The metrics recomputed here. Named so the summary of changes can be read without
# cross-referencing the code, and so a row-level diff can iterate them.
RESCORED_KEYS = (
    "gold_rank",
    "rerank_hit",
    "citation_correct",
    "candidate_doc_coverage",
    "gold_doc_coverage",
    "substring_match",
)


def _as_hits(pages: list[dict] | None) -> list[dict]:
    """Stored page dicts -> the hit shape `eval.scoring` matches gold against.

    The stored key is `page` (diff_reports.py and every committed report depend on it);
    `_is_gold` reads `page_number`, the pipeline's key. This is the one adapter between
    them - `score` is dropped because no scorer here reads it.
    """
    return [{"pdf": p["pdf"], "page_number": p["page"]} for p in pages or []]


def _cited_is_gold(row: dict, gold: list[dict]) -> bool:
    """`citation_correct` recomputed from the stored `cited` page.

    `citation_correct` normally resolves a 1-based `source_page` against the reranked
    list. That index is not stored, but its *result* is: `cited` is exactly
    `reranked[source_page - 1]` when the index was in range and None otherwise. So this
    reuses the real scorer by handing it a one-page list and index 1, rather than
    reimplementing `_is_gold` and drifting from it.
    """
    cited = row.get("cited")
    if not row.get("found") or not cited:
        return False
    hit = {"pdf": cited["pdf"], "page_number": cited["page"]}
    return citation_correct({"found": True, "source_page": 1}, [hit], gold)


def _ks(report: dict) -> tuple[int, ...]:
    """The recall@k set the report was aggregated with.

    Taken from the stored `retrieve_k` so a rescored report carries the same metric
    *names* as its source - `recall@12` staying `recall@12` is what lets diff_reports
    join the two. Falls back to reading the existing summary keys for a report written
    before the config snapshot carried the knob.
    """
    retrieve_k = (report.get("config") or {}).get("retrieve_k")
    if isinstance(retrieve_k, int):
        return tuple(sorted({1, 3, retrieve_k}))
    found = {
        int(key.removeprefix("recall@"))
        for key in report.get("summary", {})
        if key.startswith("recall@") and key.removeprefix("recall@").isdigit()
    }
    return tuple(sorted(found or {1, 3, 10}))


def rescore_rows(
    rows: list[dict], dataset: list[dict]
) -> tuple[list[dict], list[dict], list[str]]:
    """Recompute the label-derived metrics on every answerable row.

    Returns (new rows, per-row change records, ids present in the report but not the
    dataset). Rows are copied, never mutated in place, so a caller can diff against the
    originals. An unanswerable row is returned untouched: `abstention_correct` is a
    function of the citation alone and no label change can move it.
    """
    by_id = {item["id"]: item for item in dataset}
    out: list[dict] = []
    changes: list[dict] = []
    orphans: list[str] = []

    for row in rows:
        item = by_id.get(row["id"])
        if item is None:
            orphans.append(row["id"])
            out.append(row)
            continue
        # Unanswerable rows carry `abstention_correct` and none of the keys below; the
        # shape difference is deliberate in run_eval and is preserved here.
        if item["unanswerable"] or "abstention_correct" in row:
            out.append(row)
            continue

        gold = item["gold"]
        candidates = _as_hits(row.get("candidate_pages"))
        reranked = _as_hits(row.get("reranked_pages"))

        new = dict(row)
        new["gold_rank"] = gold_rank(candidates, gold)
        new["candidate_doc_coverage"] = gold_doc_coverage(candidates, gold)
        # A retrieval-only report has no rerank stage at all. Recomputing its absent
        # keys would invent a rerank that never ran, so they are only touched when the
        # source report actually carried them.
        if "reranked_pages" in row:
            new["rerank_hit"] = gold_rank(reranked, gold) is not None
            new["gold_doc_coverage"] = gold_doc_coverage(reranked, gold)
            new["citation_correct"] = _cited_is_gold(row, gold)
        if "substring_match" in row:
            new["substring_match"] = substring_match(
                row.get("answer", ""), item["answer_contains"], item["answer_contains_all"]
            )

        moved = {k: (row.get(k), new[k]) for k in RESCORED_KEYS if k in new and row.get(k) != new[k]}
        if moved:
            changes.append({"id": row["id"], "moved": moved, "judged": "judge" in row})
        out.append(new)

    return out, changes, orphans


def build_report(source: dict, path: Path, rows: list[dict], changes: list[dict]) -> dict:
    """Assemble the rescored report: the source's shape, re-aggregated, plus provenance."""
    summary = aggregate(rows, ks=_ks(source))
    per_tag = summary.pop("per_tag")
    stale = [c["id"] for c in changes if c["judged"]]
    return {
        **source,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Provenance, so a rescored report can never be mistaken for a measured one.
        # `stale_judge_rows` is the honest caveat: those verdicts were produced against
        # the labels this rescore replaced, so judge_accuracy mixes two label sets.
        "rescored": {
            "from": str(path),
            "source_generated_at": source.get("generated_at"),
            "rows_changed": len(changes),
            "changed_ids": [c["id"] for c in changes],
            "stale_judge_rows": stale,
        },
        "summary": summary,
        "per_tag": per_tag,
        "rows": rows,
    }


def _diff_summary(before: dict, after: dict) -> list[tuple[str, object, object]]:
    """Summary keys whose value moved, for the console report."""
    keys = sorted(set(before) | set(after))
    return [(k, before.get(k), after.get(k)) for k in keys if before.get(k) != after.get(k)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-score a stored eval report against changed labels (no pipeline run)."
    )
    parser.add_argument("report", help="the stored report to re-score")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", required=True, help="where to write the rescored report")
    args = parser.parse_args(argv)

    path = Path(args.report)
    source = json.loads(path.read_text())
    rows = source.get("rows", [])

    if source.get("degraded_run"):
        print(f"REFUSING: {path} is a degraded_run - its answers did not all reach the model.")
        return 2

    # The hard precondition. Reports written before `candidate_pages` existed carry no
    # retrieved pages at all, so every scorer here would see an empty list and quietly
    # write zeros - a rescore that looks like a catastrophic regression. Refuse instead.
    answerable = [r for r in rows if "abstention_correct" not in r and not r.get("unanswerable")]
    if not answerable:
        print(f"REFUSING: {path} has no answerable rows to re-score.")
        return 2
    without = [r["id"] for r in answerable if "candidate_pages" not in r]
    if without:
        print(
            f"REFUSING: {len(without)} of {len(answerable)} answerable rows have no "
            f"`candidate_pages` (e.g. {without[0]}).\n"
            "  That report predates stored retrieval and can only be re-measured by a live run."
        )
        return 2

    dataset = load_dataset(Path(args.dataset).read_text().splitlines())
    new_rows, changes, orphans = rescore_rows(rows, dataset)

    missing = sorted({item["id"] for item in dataset} - {r["id"] for r in rows})
    for label, ids in (("in the report but not the dataset", orphans),
                       ("in the dataset but not the report", missing)):
        if ids:
            print(f"note: {len(ids)} row(s) {label}, left alone: {', '.join(ids[:5])}"
                  + (" …" if len(ids) > 5 else ""))

    before = source.get("summary", {})
    report = build_report(source, path, new_rows, changes)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\nre-scored {len(new_rows)} rows from {path} -> {out}")
    print(f"rows changed: {len(changes)}")
    for change in changes:
        moved = ", ".join(f"{k}: {old} -> {new}" for k, (old, new) in change["moved"].items())
        print(f"  {change['id']}: {moved}")

    diff = _diff_summary(before, report["summary"])
    print(f"\nsummary metrics moved: {len(diff)}")
    for key, old, new in diff:
        print(f"  {key}: {old} -> {new}")

    stale = report["rescored"]["stale_judge_rows"]
    if stale:
        print(
            f"\nWARNING: {len(stale)} changed row(s) carry a judge verdict scored against the "
            f"old labels:\n  {', '.join(stale)}\n"
            "  `judge_accuracy` in this report mixes two label sets. Re-run with --judge "
            "before pinning it."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
