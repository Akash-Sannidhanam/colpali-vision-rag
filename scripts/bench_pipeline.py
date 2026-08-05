"""A/B the ingest pipeline against a serial baseline, interleaved, in one process.

Exists to answer one open question: **`ingest._StoreWorker` shipped on probation.** It moves
the PNG save and the Qdrant upsert off the GPU's thread, and on the box it was written on it
bought *nothing* - 0 of 3 rounds - because `build_point` is pure-Python and the worker
thread's GIL traffic costs the main thread about what moving the work off it saves. It was
kept only because its share grows as the forward pass shrinks. This script is how you make
that call rather than guessing; delete the worker if it is still flat.

    PYTHONPATH=. COLLECTION_NAME=pdf_pages_bench uv run python scripts/bench_pipeline.py 6 3

**Set COLLECTION_NAME.** This runs a real ingest and upserts real points; without an override
it writes into the live index. Drop the arm's collection afterwards.

Three method choices, each of which a naive version of this got wrong:

- **Interleaved, in one process.** Wall-clock s/page spreads ~50% *within* a single arm here
  (thermal drift over a 20-minute run), so A-then-B measures the drift, not the arms.
  Alternating with one model load means every arm sees the same thermal state.
- **GPU-busy fraction is the metric, not s/page.** Every arm runs the identical forward pass,
  so `forward_s / wall` isolates how much of the loop was *not* the GPU - and drift scales
  numerator and denominator together, so the ratio survives what the absolute numbers do not.
- **The serial baseline is reconstructed here, not checked out.** Both arms then run the same
  code for everything except the two things under test, and no arm needs a dirty tree. The
  cost is that `serial_iter_embedded` and `SerialStore` must be kept honest by hand if
  `embedder.iter_embedded` or `ingest._StoreWorker` change shape.

Reaches into private names (`embedder._embed_batch`, `ingest._StoreWorker`) on purpose: the
point is to swap the internals, not the public seam. Needs the real model, so it cannot be a
test. Run it on a quiet box - see `scripts/profile_ingest.py` for how badly this one drifts.
"""
import sys
import time
from pathlib import Path

from src import embedder, ingest
from src.config import EMBED_VERSION, UPSERT_BATCH_SIZE
from src.pdf_render import save_page_image
from src.vector_store import build_point, close_client, live_collection, upsert_pages

PDF = Path("pdfs/attention.pdf")


def serial_iter_embedded(images, batch_size=None, stats=None):
    """iter_embedded as it was: preprocess and forward inline, no lookahead."""
    size = max(1, batch_size or 1)
    if size > 1 and not embedder._batching_is_supported():
        size = 1
    start = 0
    while start < len(images):
        chunk = images[start:start + size]
        yield start, embedder._embed_batch(chunk, stats)
        start += len(chunk)


class SerialStore:
    """_StoreWorker as it was: PNG save, build_point and upsert inline on the GPU's thread."""

    def __init__(self, name, collection_name, content_hash, total, progress):
        self._name, self._collection = name, collection_name
        self._content_hash, self._total, self._progress = content_hash, total, progress
        self._batch: list = []
        self.stored = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._flush()
        return False

    def submit(self, page_number, image, multivector):
        path = save_page_image(image, self._name, page_number)
        self._batch.append(build_point(
            multivector, self._name, page_number, path, self._content_hash, EMBED_VERSION))
        if len(self._batch) >= UPSERT_BATCH_SIZE:
            self._flush()

    def _flush(self):
        if not self._batch:
            return
        upsert_pages(self._batch, collection_name=self._collection)
        self.stored += len(self._batch)
        self._batch = []


def run(collection, pages_cap, iter_fn):
    """One timed ingest of the first `pages_cap` pages.

    Returns (s/page, gpu_busy_fraction). The fraction is the headline: both arms run the
    *identical* forward pass, so it isolates how much of the loop was something other than
    the GPU. Thermal drift scales the forward time and the wall clock together and cancels,
    which raw s/page does not - measured here, s/page spread 50% within one arm.
    """
    stats: dict = {}
    real_render = ingest.pdf_to_images
    ingest.pdf_to_images = lambda p: real_render(p)[:pages_cap]
    ingest.iter_embedded = lambda imgs, batch_size=None: iter_fn(imgs, batch_size, stats)
    try:
        t = time.perf_counter()
        ingest.ingest_pdf(PDF, collection, lambda evt: None, "bench")
        wall = time.perf_counter() - t
    finally:
        ingest.pdf_to_images = real_render
    return wall / pages_cap, stats.get("forward_s", 0.0) / wall, stats


def main():
    pages_cap = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    collection = live_collection()
    embedder.load_model()
    worker_cls = ingest._StoreWorker
    run(collection, 2, embedder.iter_embedded)  # warm up kernels outside the timers

    # Three arms, to attribute the win (or the GIL loss) to each half separately.
    arms = {"serial": (serial_iter_embedded, SerialStore),
            "store-only": (serial_iter_embedded, worker_cls),
            "full": (embedder.iter_embedded, worker_cls)}
    results: dict[str, list[tuple[float, float]]] = {a: [] for a in arms}
    last: dict[str, dict] = {}
    for i in range(rounds):
        for arm, (iter_fn, store) in arms.items():
            ingest._StoreWorker = store
            per_page, busy, stats = run(collection, pages_cap, iter_fn)
            results[arm].append((per_page, busy))
            last[arm] = stats
            print(f"round {i + 1} {arm:>10s}: {per_page:6.3f} s/page | "
                  f"GPU busy {busy:6.1%}", flush=True)

    print()
    med = {}
    for arm, xs in results.items():
        pp = sorted(x[0] for x in xs)[len(xs) // 2]
        busy = sorted(x[1] for x in xs)[len(xs) // 2]
        med[arm] = (pp, busy)
        s = last[arm]
        print(f"{arm:>10s}: median {pp:6.3f} s/page | GPU busy {busy:6.1%} | "
              f"per page: preprocess {s.get('preprocess_s', 0) / pages_cap:.3f}s "
              f"forward {s.get('forward_s', 0) / pages_cap:.3f}s "
              f"decode {s.get('decode_s', 0) / pages_cap:.3f}s")
    print('\nGPU idle: ' + '  '.join(f'{a} {1 - m[1]:.1%}' for a, m in med.items()))
    close_client()


if __name__ == "__main__":
    main()
