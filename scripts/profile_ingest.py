"""Profile where ingest's wall-clock actually goes, stage by stage.

Ingest is the slowest thing this project does - a corpus indexes at single-digit
pages/min on a laptop - but until this script nobody had measured *which part* is slow.
Four candidates, and the answer picks the optimisation:

- **render**  `pdf_render.pdf_to_images` (poppler), one call per document
- **save**    `pdf_render.save_page_image`, one PNG per page
- **embed**   the ColQwen2 forward pass, split into preprocess / forward / decode
- **upsert**  `vector_store.upsert_pages`, one round-trip per UPSERT_BATCH_SIZE pages

    PYTHONPATH=. uv run python scripts/profile_ingest.py                       # default doc
    PYTHONPATH=. uv run python scripts/profile_ingest.py pdfs/colpali.pdf --pages 8
    PYTHONPATH=. uv run python scripts/profile_ingest.py --no-store            # skip Qdrant
    PYTHONPATH=. uv run python scripts/profile_ingest.py --batch-sizes 1,2,4,8 # the sweep
    PYTHONPATH=. uv run python scripts/profile_ingest.py --verify-equivalence  # batching is safe?
    PYTHONPATH=. uv run python scripts/profile_ingest.py --output bench/reports/ingest.json

**embed is three stages, not one**, and they respond to different fixes: `preprocess` is CPU
(resize/patchify), `forward` is the GPU, `decode` is the GPU->CPU copy. Only `forward` gets
faster from a bigger batch or a smaller image; the other two get faster only by running while
the GPU is busy with something else. Timing them as one number is why an earlier pass
concluded "batch the forward pass" and stopped there.

The numbers are **machine-specific and contention-sensitive** - device, unified-memory
pressure and thermal state all move them. Compare runs on the same box, and run them on a
*quiet* box: measured here, running the test suite alongside a profile inflated the embed
figure 4x (9.8 s/page -> 41.5 s/page) on the same eight pages.

Two deliberate choices in the method:

- **Warm-up before timing.** The first forward pass pays the model load and, on MPS, a
  kernel compile for that input shape. Timing it would make the first document look 10x
  slower than the second and hide the steady-state cost, which is the one that matters.
  A new batch size is a new shape, so the sweep warms up once per arm, not once per run.
- **The upsert is measured by default** (`--no-store` opts out). It needs a live Qdrant, and
  it is safe: point ids are `uuid5(pdf, page)`, so re-storing an indexed document overwrites
  in place with identical vectors. It was opt-in originally, which is exactly how it stayed
  unmeasured through a whole ingest-optimisation pass while sitting inline in the embed loop.
  An unreachable Qdrant degrades to a skip with a recorded reason rather than aborting the
  run, so a missing server never costs you the embed numbers.

Unlike the other scripts here this one imports `src.`, so it needs `PYTHONPATH=.`.
"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

from src.config import (
    COLPALI_MODEL,
    EMBED_BATCH_SIZE,
    EMBED_VERSION,
    EMBED_VISUAL_TOKENS,
    PDFS_DIR,
    RENDER_DPI,
    UPSERT_BATCH_SIZE,
)
from src.embedder import embed_image, embed_images, iter_embedded, load_model
from src.ingest import _fingerprint
from src.pdf_render import pdf_to_images, save_page_image
from src.vector_store import build_point, close_client, live_collection, ping, upsert_pages

# bfloat16 quantises to ~3 significant digits and a batched matmul reduces in a different
# order than a single-row one, so the arms differ in the last bits. Anything beyond this is
# not float noise - it is padding being stored, or the wrong rows being kept.
EQUIVALENCE_TOLERANCE = 1e-2


def _default_pdf() -> Path | None:
    """A reasonable document to profile when none is named: a tracked demo paper."""
    preferred = PDFS_DIR / "attention.pdf"
    if preferred.exists():
        return preferred
    candidates = sorted(PDFS_DIR.glob("*.pdf"))
    return candidates[0] if candidates else None


def _stats(samples: list[float]) -> dict:
    """Per-page timing summary for a stage that runs once per page."""
    return {
        "total_s": round(sum(samples), 3),
        "mean_s": round(statistics.mean(samples), 3),
        "median_s": round(statistics.median(samples), 3),
        "n": len(samples),
    }


def _time_embed(pages: list, batch_size: int) -> tuple[list[float], list, list[int], dict]:
    """Embed `pages` at one batch size, timing each forward pass.

    Returns `(samples, vectors, observed_batch_sizes, stats)`. The third is what the arm
    *actually* ran at, which is not always what was asked for: `iter_embedded` refuses to
    batch on a backend that miscomputes it, and halves on an OOM. An arm that reports
    `batch_size: 8` while every pass ran at 1 would be a fabricated speedup number, so the
    report carries both.

    The fourth is `embedder`'s own sub-stage accounting for the arm - preprocess (CPU) vs
    forward (GPU) vs decode (GPU->CPU) - which is what says whether the answer to a slow
    ingest is a bigger batch, a smaller image, or getting the CPU off the GPU's path. It is
    collected from the same `_embed_batch` ingest runs rather than a parallel copy of it,
    so the split cannot drift from the thing being measured.

    A sample is one *batch*, charged to the pages it covered, so the per-page figures stay
    comparable across arms. The arm is warmed up first: on MPS a batch size is an input
    shape, and the first pass at a new shape pays a kernel compile that is not part of the
    steady-state cost being measured.

    Goes through `iter_embedded` over the whole page list rather than pre-chunking, for the
    same reason ingest does: an arm too large for the box halves once and stays there, so
    the arm reports the size it actually settled at instead of re-paying the failed pass on
    every window. A batch smaller than `batch_size` in the samples is that backoff showing
    up - which is a finding about the arm, not noise to average away.
    """
    embed_images(pages[:batch_size], batch_size=batch_size)  # warm up this shape
    samples: list[float] = []
    vectors: list = []
    observed: list[int] = []
    stats: dict = {}
    t = time.perf_counter()
    for _, batch in iter_embedded(pages, batch_size=batch_size, stats=stats):
        samples.extend([(time.perf_counter() - t) / len(batch)] * len(batch))
        vectors.extend(batch)
        observed.append(len(batch))
        t = time.perf_counter()   # restart after the yield: don't charge the caller's work
    stats.pop("_t", None)  # bookkeeping cursor, not a measurement
    return samples, vectors, observed, stats


def profile_document(pdf: Path, page_cap: int | None, store: bool, batch_sizes: list[int]) -> dict:
    """Time every ingest stage over one document; return its raw per-stage samples.

    The whole document is rendered even when `--pages` caps the rest of the work, so the
    render figure is a true per-page cost rather than poppler's fixed startup amortised
    over a handful of pages. Render, save and upsert are batch-size-independent and so are
    measured once; only embed is swept.
    """
    name = pdf.name
    collection = live_collection() if store else ""
    # When profiling a capped subset of pages, use a non-matching content_hash to prevent
    # marking the document as current with only partial coverage. The upsert is still
    # measured for the profiled pages, but the next sync will correctly re-embed the full
    # document. When no page cap is requested, use the real fingerprint so overwrites
    # happen in place with identical vectors.
    content_hash = _fingerprint(pdf) if (store and page_cap is None) else ""

    t0 = time.perf_counter()
    pages = pdf_to_images(pdf)
    render_total = time.perf_counter() - t0
    rendered = len(pages)

    if page_cap is not None:
        pages = pages[:page_cap]

    save_samples: list[float] = []
    image_paths: list[Path] = []
    for offset, page in enumerate(pages):
        t = time.perf_counter()
        image_paths.append(save_page_image(page, name, offset + 1))
        save_samples.append(time.perf_counter() - t)

    embed_samples: dict[int, list[float]] = {}
    embed_observed: dict[int, list[int]] = {}
    embed_stats: dict[int, dict] = {}
    vectors: list = []
    for size in batch_sizes:
        embed_samples[size], vectors, embed_observed[size], embed_stats[size] = _time_embed(
            pages, size,
        )

    upsert_samples: list[float] = []
    if store:
        # Stores the last arm's vectors; every arm produces the same ones (see
        # --verify-equivalence), so which arm wrote them does not matter.
        batch: list = []
        for offset, multivector in enumerate(vectors):
            batch.append(build_point(
                multivector, name, offset + 1, image_paths[offset], content_hash, EMBED_VERSION,
            ))
            if len(batch) >= UPSERT_BATCH_SIZE:
                t = time.perf_counter()
                upsert_pages(batch, collection_name=collection)
                upsert_samples.append(time.perf_counter() - t)
                batch = []
        if batch:
            t = time.perf_counter()
            upsert_pages(batch, collection_name=collection)
            upsert_samples.append(time.perf_counter() - t)

    return {
        "pdf": name,
        "pages_rendered": rendered,
        "pages_profiled": len(pages),
        "render_total_s": render_total,
        "save": save_samples,
        "embed": embed_samples,
        "embed_observed": embed_observed,
        "embed_stats": embed_stats,
        "upsert": upsert_samples,
    }


def verify_equivalence(pdf: Path, page_cap: int | None, batch_size: int) -> tuple[str, dict]:
    """Check that batching leaves the stored vectors unchanged. Returns (verdict, detail).

    This is what licenses keeping EMBED_BATCH_SIZE out of EMBED_VERSION. On `"corrupt"`, the
    mask trim in embedder._embed_batch is wrong and the index is being silently corrupted -
    fix the trim; do not paper over it by bumping the fingerprint.

    bfloat16 reductions are order-sensitive, so the arms are compared against a tolerance
    rather than for bit-equality. Patch *counts* must match exactly - a count mismatch means
    padding is being stored, which is the specific failure this guards.

    **Three verdicts, not two.** "Did batching change the vectors?" and "did batching run at
    all?" are separate questions, and collapsing them makes the correct configuration look
    like the broken one: `embedder._batching_is_supported` disables batching on MPS by
    design, so on Apple Silicon nothing is ever batched and the comparison is vacuously
    exact. Reporting that as failure cries wolf on every Mac; reporting it as success
    overclaims a check that never ran. Hence `"not_applicable"`, which is a pass (the stored
    vectors are right) that refuses to claim batching was verified.
    """
    pages = pdf_to_images(pdf)
    if page_cap is not None:
        pages = pages[:page_cap]

    single = [embed_image(p) for p in pages]
    batched: list = []
    observed: list[int] = []
    for _, chunk in iter_embedded(pages, batch_size=batch_size):
        batched.extend(chunk)
        observed.append(len(chunk))

    # A document shorter than the batch cannot reach it, so compare against what was possible.
    batching_ran = max(observed, default=1) > 1
    counts = [(len(a), len(b)) for a, b in zip(single, batched)]
    mismatched = [i + 1 for i, (a, b) in enumerate(counts) if a != b]
    max_delta = 0.0
    for a, b in zip(single, batched):
        for row_a, row_b in zip(a, b):
            for x, y in zip(row_a, row_b):
                max_delta = max(max_delta, abs(x - y))

    vectors_agree = (len(single) == len(batched) and not mismatched
                     and max_delta <= EQUIVALENCE_TOLERANCE)
    if not vectors_agree:
        verdict = "corrupt"
    elif not batching_ran:
        verdict = "not_applicable"
    else:
        verdict = "equivalent"

    return verdict, {
        "pages": len(pages),
        "batch_size": batch_size,
        "batching_ran": batching_ran,
        "effective_batch_sizes": observed,
        "output_length_mismatch": len(single) != len(batched),
        "patch_counts": [a for a, _ in counts],
        "pages_with_mismatched_patch_counts": mismatched,
        "max_abs_delta": max_delta,
        "tolerance": EQUIVALENCE_TOLERANCE,
        "verdict": verdict,
    }


def build_report(
    docs: list[dict], store: bool, batch_sizes: list[int],
    model_load_s: float, device: str, dtype: str, upsert_skipped: str = "",
) -> dict:
    """Fold the per-document samples into the pinned report shape.

    Render, save and upsert are fixed per-page costs; embed gets one `arms` entry per
    batch size, each carrying what a whole page costs at that arm and the resulting
    pages/min. `speedup_vs_batch_1` is the headline when batch 1 is in the sweep.
    """
    pages = sum(d["pages_profiled"] for d in docs)
    render_total = sum(d["render_total_s"] for d in docs)
    rendered = sum(d["pages_rendered"] for d in docs)
    save = [s for d in docs for s in d["save"]]
    upsert = [s for d in docs for s in d["upsert"]]

    render_per_page = render_total / rendered if rendered else 0.0
    save_per_page = statistics.mean(save) if save else 0.0
    # Upsert is charged per UPSERT_BATCH_SIZE pages, so spread it over the pages it stored.
    upsert_per_page = (sum(upsert) / pages) if (upsert and pages) else 0.0
    fixed_per_page = render_per_page + save_per_page + upsert_per_page

    stages = {
        "render": {
            "total_s": round(render_total, 3),
            "per_page_s": round(render_per_page, 3),
            "pages_rendered": rendered,
        },
        "save": {**_stats(save), "per_page_s": round(save_per_page, 3)} if save else {"n": 0},
        "upsert": {
            "measured": store,
            **(_stats(upsert) if upsert else {"n": 0}),
            "per_page_s": round(upsert_per_page, 3),
        },
    }

    arms: list[dict] = []
    for size in batch_sizes:
        samples = [s for d in docs for s in d["embed"][size]]
        observed = [n for d in docs for n in d["embed_observed"][size]]
        embed_per_page = statistics.mean(samples) if samples else 0.0
        page_s = fixed_per_page + embed_per_page
        effective = max(observed) if observed else 0
        # The largest document determines the maximum possible final batch
        largest_doc_pages = max((d["pages_profiled"] for d in docs), default=0)
        expected_max = min(size, largest_doc_pages)
        # Sub-stage split, summed across documents and charged per page. `patches` is the
        # visual-token count the model actually saw, which is the number the embed cost
        # tracks - not RENDER_DPI, since the processor downscales to its own pixel budget
        # long before the model sees a page (see EMBED_VISUAL_TOKENS in src/config.py).
        stats = [d["embed_stats"][size] for d in docs]
        totals = {
            key: sum(s.get(key, 0.0) for s in stats)
            for key in ("preprocess_s", "forward_s", "decode_s", "patches")
        }
        arms.append({
            "batch_size": size,
            # What the arm actually ran at. Below `batch_size` means iter_embedded refused
            # the requested size - an untrustworthy backend, or an OOM backoff. When these
            # differ, the arm is NOT a measurement of `batch_size`.
            "effective_batch_size": effective,
            "requested_size_honoured": effective >= expected_max,
            "embed": {**_stats(samples), "per_page_s": round(embed_per_page, 3)},
            "embed_stages": {
                stage: round(totals[f"{stage}_s"] / pages, 3) if pages else 0.0
                for stage in ("preprocess", "forward", "decode")
            },
            "visual_tokens_per_page": round(totals["patches"] / pages) if pages else 0,
            "embed_share_of_page_s": round(embed_per_page / page_s, 4) if page_s else 0.0,
            "page_s": round(page_s, 3),
            "pages_per_min": round(60.0 / page_s, 2) if page_s else 0.0,
        })
    # Find the batch-1 arm for speedup calculations (it may not be first)
    batch_1_arm = next((a for a in arms if a["batch_size"] == 1), None)
    if batch_1_arm:
        baseline_page_s = batch_1_arm["page_s"]
        for arm in arms:
            arm["speedup_vs_batch_1"] = round(baseline_page_s / arm["page_s"], 2) if arm["page_s"] else 0.0

    return {
        "config": {
            "colpali_model": COLPALI_MODEL,
            "render_dpi": RENDER_DPI,
            # None = the checkpoint's own budget. Read it together with each arm's
            # `visual_tokens_per_page`, which is what the model actually saw.
            "embed_visual_tokens": EMBED_VISUAL_TOKENS,
            "batch_sizes": batch_sizes,
            "upsert_batch_size": UPSERT_BATCH_SIZE,
            "device": device,
            "dtype": dtype,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "upsert_measured": store,
            # Why, when it wasn't. An unexplained `false` is what let the upsert stay
            # invisible through a whole optimisation pass.
            "upsert_skipped": upsert_skipped,
        },
        "documents": [
            {"pdf": d["pdf"], "pages_rendered": d["pages_rendered"],
             "pages_profiled": d["pages_profiled"]}
            for d in docs
        ],
        "model_load_s": round(model_load_s, 2),
        "pages": pages,
        "stages": stages,
        "arms": arms,
    }


def _print_report(report: dict) -> None:
    """Human-readable summary; the JSON is the artifact, this is the read-at-a-glance."""
    cfg = report["config"]
    print(f"\n{cfg['colpali_model']} on {cfg['device']} ({cfg['dtype']}), {cfg['render_dpi']} DPI")
    print(f"model load: {report['model_load_s']}s (excluded from the timings below)")
    print(f"{report['pages']} pages profiled\n")

    upsert_note = "" if cfg["upsert_measured"] else f"   (not measured: {cfg['upsert_skipped']})"
    print("fixed per-page cost")
    for stage in ("render", "save", "upsert"):
        note = upsert_note if stage == "upsert" else ""
        print(f"  {stage:<8}{report['stages'][stage]['per_page_s']:>9.3f}s{note}")

    print(f"\n{'batch':<8}{'actual':>8}{'embed/page':>12}{'page':>10}{'pages/min':>12}{'speedup':>10}")
    for arm in report["arms"]:
        speedup = arm.get("speedup_vs_batch_1")
        cell = f"{speedup:.2f}x" if speedup else "-"
        actual = str(arm["effective_batch_size"]) + ("" if arm["requested_size_honoured"] else "!")
        print(f"{arm['batch_size']:<8}{actual:>8}{arm['embed']['per_page_s']:>11.3f}s"
              f"{arm['page_s']:>9.3f}s{arm['pages_per_min']:>12}{cell:>10}")
    if any(not a["requested_size_honoured"] for a in report["arms"]):
        print("\n! = the backend refused the requested batch size (untrusted batching, or an\n"
              "    OOM backoff). Those arms measure the size under 'actual', not 'batch'.")

    # The split inside embed: what a bigger batch can help (forward) vs what only getting the
    # CPU off the GPU's path can (preprocess, decode).
    print(f"\n{'batch':<8}{'tokens':>8}{'preprocess':>12}{'forward':>10}{'decode':>9}")
    for arm in report["arms"]:
        st = arm["embed_stages"]
        print(f"{arm['batch_size']:<8}{arm['visual_tokens_per_page']:>8}"
              f"{st['preprocess']:>11.3f}s{st['forward']:>9.3f}s{st['decode']:>8.3f}s")
    # Stated because it has already been misread once as evidence of a contended box.
    print("\npreprocess is measured on the prefetch thread, overlapped with the GPU, so its\n"
          "wall time is stretched by GIL contention (~3-5x its serial cost) and these\n"
          "columns legitimately sum to more than 'page'. That gap IS the overlap. Compare it\n"
          "only against another pipelined run, never against a serial measurement.")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdfs", nargs="*", type=Path, help="PDFs to profile (default: a demo document)")
    parser.add_argument("--pages", type=int, default=None, help="profile at most N pages per document")
    parser.add_argument("--no-store", action="store_true",
                        help="skip the Qdrant upsert timing (it is measured by default)")
    parser.add_argument("--batch-sizes", default=str(EMBED_BATCH_SIZE),
                        help="comma-separated embed batch sizes to sweep (default: EMBED_BATCH_SIZE)")
    parser.add_argument("--verify-equivalence", action="store_true",
                        help="check batched vectors match batch-1 ones, then exit (1 if they differ)")
    parser.add_argument("--output", type=Path, default=None, help="write the JSON report here")
    args = parser.parse_args(argv)

    pdfs = args.pdfs or [p for p in [_default_pdf()] if p]
    missing = [p for p in pdfs if not p.exists()]
    if not pdfs or missing:
        print(f"No PDF to profile: {missing or PDFS_DIR}", file=sys.stderr)
        return 1
    if args.pages is not None and args.pages < 1:
        print("--pages must be at least 1", file=sys.stderr)
        return 1
    try:
        batch_sizes = [int(s) for s in args.batch_sizes.split(",") if s.strip()]
    except ValueError:
        print(f"--batch-sizes must be comma-separated integers, got {args.batch_sizes!r}", file=sys.stderr)
        return 1
    if not batch_sizes or any(s < 1 for s in batch_sizes):
        print("--batch-sizes entries must all be at least 1", file=sys.stderr)
        return 1

    # Measured by default: the upsert sits inline in ingest's embed loop, so leaving it
    # opt-in is how it stayed unmeasured through an entire ingest-optimisation pass. But an
    # unreachable Qdrant must not cost you the embed numbers you actually came for, so this
    # degrades to an explained skip rather than the fail-fast the opt-in version could afford.
    store = not args.no_store
    upsert_skipped = ""
    if store:
        try:
            ping()
        except Exception as exc:
            store = False
            upsert_skipped = f"Qdrant unreachable ({type(exc).__name__}: {exc})"
            print(f"upsert not measured: {upsert_skipped}", file=sys.stderr)
    else:
        upsert_skipped = "--no-store"

    t = time.perf_counter()
    model, _ = load_model()
    model_load_s = time.perf_counter() - t

    if args.verify_equivalence:
        verdict, detail = verify_equivalence(pdfs[0], args.pages, max(batch_sizes))
        print(json.dumps(detail, indent=2))
        # "not_applicable" exits 0 - the stored vectors are right - but must never read as a
        # green light on batching, which is why it does not say "equivalent".
        print({
            "equivalent": "\nbatching is vector-equivalent",
            "not_applicable": "\nNOT VERIFIED: this backend never batched (see "
                              "embedder._batching_is_supported), so there was nothing to "
                              "compare. The stored vectors are correct; batching is untested.",
            "corrupt": "\nBATCHING CHANGED THE VECTORS",
        }[verdict])
        return 1 if verdict == "corrupt" else 0

    # Warm up outside the timers: the first forward pass pays a one-off kernel compile.
    embed_image(pdf_to_images(pdfs[0])[0])

    try:
        docs = [profile_document(pdf, args.pages, store, batch_sizes) for pdf in pdfs]
    finally:
        if store:
            close_client()

    report = build_report(
        docs, store, batch_sizes, model_load_s, str(model.device), str(model.dtype),
        upsert_skipped,
    )
    _print_report(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
