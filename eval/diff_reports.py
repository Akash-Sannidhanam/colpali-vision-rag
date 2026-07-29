"""Compare two eval reports question by question - the tool that decides a knob change.

A summary average is the wrong evidence for a knob like RERANK_K. `gold_coverage_avg`
runs over the cross-document rows alone, so one question moves it by 1/n, and runs of
*identical* code have produced 0.667 and 0.75 on the same corpus. A delta of 0.08
between two arms is therefore unreadable on its own.

Paired per-question comparison is readable: the same questions on both sides, and a
named list of which ones flipped. Five questions gaining coverage and none losing it is
a result; the same +0.08 made of four gains and three losses is noise wearing a result's
clothes. The flip list is also what makes the older lesson enforceable - audit every
changed row against its `cited`/`top1` before calling it a regression, because four of
six apparent regressions on the last pass turned out to be under-labeled gold.

    PYTHONPATH=. uv run python eval/diff_reports.py BEFORE.json AFTER.json
    PYTHONPATH=. uv run python eval/diff_reports.py before.json after.json --metric citation_correct
    PYTHONPATH=. uv run python eval/diff_reports.py before.json after.json --metric judge.correct

Stdlib only, so it runs with or without PYTHONPATH=. and needs no key, model, or Qdrant.

Exit codes: 0 = compared; 2 = unreadable report or a metric no row carries.
"""

import argparse
import json
import sys
from pathlib import Path

# Which way is better. Everything scored here rises with quality except gold_rank,
# which is a position: rank 1 beats rank 5. Unlisted metrics default to higher-is-better.
METRIC_DIRECTION = {
    "gold_rank": -1,
    "latency_ms": -1,
}

DEFAULT_METRIC = "gold_doc_coverage"


def row_metric(row: dict, metric: str):
    """Read `metric` out of a scored row, or None when the row doesn't carry it.

    Supports one dotted level (`judge.correct`) because the judge verdict is the only
    nested block. None means N/A - the overwhelmingly common case, since most rows are
    N/A for most metrics (single-document rows have no gold_doc_coverage, unanswerable
    rows have no citation_correct), so it must not raise.
    """
    head, _, tail = metric.partition(".")
    value = row.get(head)
    if tail:
        value = (value or {}).get(tail) if isinstance(value, (dict, type(None))) else None
    return value


def _score(value) -> float | None:
    """Order a metric value for improved/regressed counting; None when unorderable.

    Booleans sort False < True, which is what makes citation_correct flipping True an
    improvement rather than an incomparable change.
    """
    if isinstance(value, bool):
        return float(value)
    return float(value) if isinstance(value, (int, float)) else None


def _mean(values: list[float]) -> float | None:
    """Rounded mean, or None over an empty list."""
    return round(sum(values) / len(values), 4) if values else None


def config_changes(before: dict, after: dict) -> dict:
    """The knob values that differ between the two runs, as {key: (before, after)}.

    An arm comparison is only meaningful when one knob moved; printing which one did
    catches the two silent mistakes - comparing a run against itself, and comparing
    across different datasets or embedding models.
    """
    b, a = before.get("config", {}), after.get("config", {})
    return {k: (b.get(k), a.get(k)) for k in sorted(set(b) | set(a)) if b.get(k) != a.get(k)}


def _degraded_sides(before: dict, after: dict) -> list[str]:
    """Which side, if either, was written by a run that degraded past its limit."""
    return [name for name, report in (("before", before), ("after", after))
            if report.get("degraded_run")]


def compare(before: dict, after: dict, metric: str = DEFAULT_METRIC) -> dict:
    """Join two reports on question id and compare `metric` row by row.

    Only rows present on *both* sides and applicable (non-None) on at least one enter
    the arithmetic, so a dataset that grew between runs can't masquerade as a metric
    change - the unpaired ids are reported separately instead.

    Raises ValueError when no row on either side carries the metric, so a typo or a
    summary-only name (`gold_coverage_avg` is the summary's key; the row's is
    `gold_doc_coverage`) fails loudly rather than reporting a serene zero flips.
    """
    b_rows = {r["id"]: r for r in before.get("rows", [])}
    a_rows = {r["id"]: r for r in after.get("rows", [])}
    shared = [rid for rid in b_rows if rid in a_rows]

    if not any(row_metric(r, metric) is not None
               for r in list(b_rows.values()) + list(a_rows.values())):
        raise ValueError(
            f"no row in either report carries {metric!r} - check the name "
            f"(row metrics include gold_doc_coverage, citation_correct, gold_rank, "
            f"substring_match, rerank_hit, abstention_correct, judge.correct)"
        )

    flips, b_values, a_values = [], [], []
    improved = regressed = unchanged = applicable = 0
    for rid in shared:
        b_val, a_val = row_metric(b_rows[rid], metric), row_metric(a_rows[rid], metric)
        if b_val is None and a_val is None:
            continue                                  # N/A both sides - not this metric's row
        applicable += 1
        b_score, a_score = _score(b_val), _score(a_val)
        if b_score is not None:
            b_values.append(b_score)
        if a_score is not None:
            a_values.append(a_score)
        if b_val == a_val:
            unchanged += 1
            continue
        flips.append({"id": rid, "before": b_val, "after": a_val,
                      "tags": a_rows[rid].get("tags", [])})
        if b_score is None or a_score is None:
            continue                                  # became (or stopped being) N/A - unorderable
        direction = METRIC_DIRECTION.get(metric, 1)
        improved += int((a_score - b_score) * direction > 0)
        regressed += int((a_score - b_score) * direction < 0)

    b_summary, a_summary = before.get("summary", {}), after.get("summary", {})
    summary_deltas = {
        k: {"before": b_summary[k], "after": a_summary[k],
            "delta": round(a_summary[k] - b_summary[k], 4)}
        for k in b_summary
        # Skip anything N/A on either side: confidence_separation is None whenever the
        # pipeline made no wrong citation, and None - 0.03 is a TypeError, not a delta.
        if isinstance(b_summary.get(k), (int, float)) and isinstance(a_summary.get(k), (int, float))
        and not isinstance(b_summary.get(k), bool)
    }

    return {
        "metric": metric,
        "applicable": applicable,
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
        "flips": flips,
        "before_avg": _mean(b_values),
        "after_avg": _mean(a_values),
        "summary_deltas": summary_deltas,
        "only_in_before": sorted(set(b_rows) - set(a_rows)),
        "only_in_after": sorted(set(a_rows) - set(b_rows)),
        "config_changes": config_changes(before, after),
        "degraded_sides": _degraded_sides(before, after),
    }


def format_diff(result: dict, metric: str) -> str:
    """Render a comparison for a terminal."""
    lines = []
    if result["degraded_sides"]:
        lines += [
            f"!! DEGRADED: the {', '.join(result['degraded_sides'])} report was written by a run "
            f"that degraded past its limit. These numbers measure an outage, not a change.",
            "",
        ]
    if result["config_changes"]:
        lines.append("config changes")
        lines += [f"  {k}: {b} -> {a}" for k, (b, a) in result["config_changes"].items()]
    else:
        lines.append("config changes: none (same knobs on both sides - a variance check)")
    lines.append("")

    lines.append(f"{metric} over {result['applicable']} paired applicable rows")
    lines.append(f"  avg {result['before_avg']} -> {result['after_avg']}")
    lines.append(
        f"  improved {result['improved']}  regressed {result['regressed']}  "
        f"unchanged {result['unchanged']}"
    )
    lines.append("")

    if result["flips"]:
        lines.append(f"flipped rows ({len(result['flips'])})")
        width = max(len(f["id"]) for f in result["flips"])
        for f in result["flips"]:
            tags = f" [{','.join(f['tags'])}]" if f["tags"] else ""
            lines.append(f"  {f['id']:<{width}}  {f['before']} -> {f['after']}{tags}")
        lines += [
            "",
            "Audit each flipped row against its `cited` / `top1` in the report before "
            "calling it a win or a regression - under-labeled gold has produced more "
            "apparent regressions on this dataset than the pipeline has.",
        ]
    else:
        lines.append("no rows flipped")

    for side, ids in (("before", result["only_in_before"]), ("after", result["only_in_after"])):
        if ids:
            lines += ["", f"only in {side} ({len(ids)}): {', '.join(ids)}",
                      "  (excluded from the paired arithmetic - the datasets differ)"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Parse args, compare the two reports, print the diff; return the exit code."""
    parser = argparse.ArgumentParser(
        description="Compare two eval reports question by question."
    )
    parser.add_argument("before", help="the baseline report JSON")
    parser.add_argument("after", help="the report to compare against it")
    parser.add_argument("--metric", default=DEFAULT_METRIC,
                        help=f"row metric to pair on (default {DEFAULT_METRIC}); also "
                             f"citation_correct, gold_rank, substring_match, rerank_hit, "
                             f"abstention_correct, judge.correct")
    parser.add_argument("--json", action="store_true", help="emit the comparison as JSON")
    args = parser.parse_args(argv)

    try:
        before = json.loads(Path(args.before).read_text())
        after = json.loads(Path(args.after).read_text())
        result = compare(before, after, args.metric)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"diff error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2) if args.json else format_diff(result, args.metric))
    return 0


if __name__ == "__main__":
    sys.exit(main())
