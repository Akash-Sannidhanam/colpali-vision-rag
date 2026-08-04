"""Ingest CLI: PDF pages -> page PNGSs -> ColPali multivectors -> Qdrant.

Two modes, both driven from `run_ingest`:

- **Sync** (default) - upsert into the live collection, embedding only the documents
  whose bytes or embedding config changed. Adding a document costs its own pages, and
  re-running over an unchanged corpus is a no-op.
- **Rebuild** (`rebuild=True`, `--rebuild`) - the original atomic wholesale build:
  everything is re-embedded into a fresh collection that is published only on success.
  The escape hatch for a genuine wipe, and the way to reclaim space after documents
  have been deleted.

Sync never prunes: a document that is indexed but no longer on disk is left alone.
Removal is always explicit (`vector_store.delete_document`, `DELETE /corpus/{pdf}`) or
wholesale (`--rebuild`), never inferred from a file's absence.
"""

import hashlib
import queue
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from src.config import EMBED_VERSION, PDFS_DIR, UPSERT_BATCH_SIZE
from src.embedder import iter_embedded
from src.pdf_render import pdf_to_images, save_page_image
from src.vector_store import (
    abort_ingest,
    begin_ingest,
    build_point,
    close_client,
    delete_document,
    document_index,
    finish_ingest,
    live_collection,
    ping,
    upsert_pages,
)

# A progress event is a small dict: {"phase": render|pages|embed|stored|skip, ...}. The
# callback is invoked from whatever thread runs the ingest, so a consumer that touches
# an event loop (the server's SSE endpoint) must hop back to it.
Progress = Callable[[dict], None]


def _print_progress(evt: dict) -> None:
    """Default progress sink: the human-readable lines the CLI used to print inline."""
    phase = evt["phase"]
    if phase == "render":
        print(f"\nRendering {evt['pdf']} ...")
    elif phase == "pages":
        print(f"{evt['total']} pages")
    elif phase == "embed":
        print(f"embedded page {evt['page']}")
    elif phase == "stored":
        print(f"stored {evt['count']} pages")
    elif phase == "skip":
        print(f"\n{evt['pdf']} unchanged ({evt['total']} pages) - skipping")


def _fingerprint(pdf_path: Path) -> str:
    """A content hash of the PDF's bytes, read in chunks so a big file isn't slurped."""
    digest = hashlib.sha256()
    with pdf_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _StoreWorker:
    """Writes embedded pages to disk and Qdrant on a worker thread.

    Everything *after* the forward pass - the PNG save, `build_point`, and the Qdrant
    upsert - is CPU and network work that used to run inline in the embed loop, so the GPU
    idled through all of it: 0.044 s (save) + 0.264 s (upsert) per page.

    **On this box that reclaims nothing, and the reason is worth knowing.** Attributed over
    three interleaved rounds, GPU idle went 9.7% serial -> 10.0% with this worker alone ->
    7.0% with it plus the preprocess lookahead in `embedder.iter_embedded`. The worker won
    0 of 3 rounds on its own: `build_point` is pure-Python, so the thread's GIL traffic
    costs the main thread roughly what moving the work off it saves (visible as the main
    thread's own preprocess inflating 0.177 s -> 0.313 s when only this changed).

    Kept because that balance is a function of how big the forward pass is - 0.31 s against
    a 5.45 s forward is 6%, against a 2.7 s one it is 11% - and because a remote Qdrant or a
    slower disk moves it the same way. It is the piece to re-measure and delete if a future
    profile still shows it flat.

    One worker draining a FIFO queue, so pages are stored in page order and the `stored`
    events keep the sequence the SSE consumer has always seen. The queue is bounded: a
    page's multivector is several MB of Python floats and the GPU can produce them faster
    than Qdrant accepts them, so an unbounded one would trade GPU stalls for memory growth.

    The `embed` events stay on the caller's thread, fired the moment a page comes off the
    GPU, because that is what drives the UI's progress bar - routing them through here would
    make the bar lag a whole upsert batch behind the work it reports.
    """

    def __init__(self, name: str, collection_name: str, content_hash: str,
                 total: int, progress: Progress):
        self._name = name
        self._collection = collection_name
        self._content_hash = content_hash
        self._total = total
        self._progress = progress
        # Two upsert batches: one being uploaded while the next fills, so a batched forward
        # pass never blocks on the previous flush.
        self._queue: queue.Queue = queue.Queue(maxsize=UPSERT_BATCH_SIZE * 2)
        self._error: BaseException | None = None
        self.stored = 0
        self._thread = threading.Thread(target=self._run, name="ingest-store", daemon=True)

    def __enter__(self) -> "_StoreWorker":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> Literal[False]:
        self._queue.put(None)   # sentinel: finish the pending batch, then stop
        self._thread.join()
        # Only surface a worker failure when the body succeeded; otherwise the body's own
        # exception (an OOM, say) is the real story and must not be masked by this one.
        if exc_type is None:
            self._raise_if_failed()
        return False

    def submit(self, page_number: int, image, multivector) -> None:
        """Hand one embedded page to the worker, blocking while the queue is full."""
        self._raise_if_failed()  # fail fast rather than embed a whole document into a corpse
        self._queue.put((page_number, image, multivector))

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error

    def _flush(self, batch: list) -> None:
        if not batch:
            return
        upsert_pages(batch, collection_name=self._collection)
        self.stored += len(batch)
        self._progress({"phase": "stored", "pdf": self._name,
                        "count": self.stored, "total": self._total})

    def _run(self) -> None:
        batch: list = []
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                page_number, image, multivector = item
                image_path = save_page_image(image, self._name, page_number)
                batch.append(build_point(
                    multivector, self._name, page_number, image_path,
                    self._content_hash, EMBED_VERSION,
                ))
                if len(batch) >= UPSERT_BATCH_SIZE:
                    self._flush(batch)
                    batch = []
            self._flush(batch)  # the remainder
        except BaseException as exc:
            self._error = exc
            # Keep draining to the sentinel. A producer blocked on a full queue would
            # otherwise wait forever for a worker that has already died - an ingest that
            # hangs is worse than one that crashes, because `document_index` fingerprints a
            # document from its first indexed page, so a killed run strands a truncated
            # document that the next sync considers current.
            while self._queue.get() is not None:
                pass


def ingest_pdf(
    pdf_path: Path, collection_name: str, progress: Progress, content_hash: str = "",
) -> int:
    """Render, embed, and store every page of one PDF, flushing in batches; return the
    page count.

    Upserts target `collection_name` explicitly: during a rebuild that is the freshly-built
    physical collection, during a sync it is the live one. Point ids are derived from
    (pdf name, page number) inside `build_point`, so a re-ingest overwrites pages in place
    rather than duplicating them. Emits a progress event at each render/embed/store step so
    a caller can stream per-page updates.

    Pages are embedded up to EMBED_BATCH_SIZE at a time - the forward pass is the dominant
    cost and the only stage that batches. The `embed` events still fire once per page, in
    page order; they just arrive in bursts, because a batch's pages all finish together. On a
    backend that does not batch at all (MPS) the effective size is 1 and the events simply
    stop bursting, which is why nothing here reads EMBED_BATCH_SIZE directly.

    The whole page list goes to `iter_embedded` in one call, deliberately: it is what keeps
    an out-of-memory backoff for the rest of the document instead of re-paying the failed
    forward pass on every batch (see embedder.iter_embedded).

    **The GPU is the only serial resource, so nothing else runs on its thread.**
    `iter_embedded` preprocesses the next batch ahead of the current forward pass, and
    `_StoreWorker` takes the PNG save and the Qdrant upsert behind it. This loop does the
    forward pass and nothing more. Measured: GPU idle 9.7% -> 7.0%, the lookahead earning
    all of it (see `_StoreWorker` for the per-half attribution).
    """
    name = pdf_path.name
    progress({"phase": "render", "pdf": name})
    pages = pdf_to_images(pdf_path)
    total = len(pages)
    progress({"phase": "pages", "pdf": name, "total": total})

    with _StoreWorker(name, collection_name, content_hash, total, progress) as store:
        for start, multivectors in iter_embedded(pages):
            for i, multivector in enumerate(multivectors):
                page_number = start + i + 1
                store.submit(page_number, pages[start + i], multivector)
                progress({"phase": "embed", "pdf": name, "page": page_number, "total": total})

    return total


def _rebuild(pdfs: list[Path], emit: Progress) -> int:
    """Atomic wholesale build: everything into a fresh collection, published on success."""
    target = begin_ingest()
    total = 0
    try:
        for pdf in pdfs:
            if pdf.exists():
                total += ingest_pdf(pdf, target, emit, _fingerprint(pdf))
        finish_ingest(target)  # atomic alias swap (server) / no-op (embedded)
    except BaseException:  # incl. KeyboardInterrupt: drop the partial, keep old index
        abort_ingest(target)
        raise
    return total


def _sync(pdfs: list[Path], emit: Progress) -> int:
    """Embed only what changed, upserting into the live collection.

    A document is current when both its content hash and the embedding config that
    produced it still match, in which case it is skipped outright. Otherwise its pages
    are deleted first and re-embedded: point ids are stable per (pdf, page), so an
    upsert alone would overwrite pages 1..n but strand pages n+1.. of a longer previous
    revision.

    No rollback: existing documents are never touched, so an interrupted run leaves the
    rest of the index serving and the offending document is completed on the next pass.
    """
    target = live_collection()
    indexed = document_index()
    total = 0
    for pdf in pdfs:
        if not pdf.exists():
            continue
        name = pdf.name
        fingerprint = _fingerprint(pdf)
        current = indexed.get(name)
        if (current
                and current["content_hash"] == fingerprint
                and current["embed_version"] == EMBED_VERSION):
            emit({"phase": "skip", "pdf": name, "total": current["page_count"]})
            continue
        if current:
            delete_document(name)
        total += ingest_pdf(pdf, target, emit, fingerprint)
    return total


def run_ingest(pdfs: list[Path], progress: Progress | None = None, *, rebuild: bool = False) -> int:
    """Index `pdfs` and return the number of pages embedded *this run*.

    The client-teardown-free core seam (mirrors `main.run_query` vs `run`): a warm
    server calls this and keeps the shared Qdrant client open, while the CLI `main()`
    wrapper owns closing it.

    Defaults to the incremental sync; `rebuild=True` takes the atomic wholesale path,
    where an interrupted or failed build drops the partial and leaves the previous index
    serving. A skipped (unchanged) document contributes 0 to the return value, so a
    no-op re-run returns 0.

    `progress` receives one event dict per render/embed/store/skip step; it defaults to
    printing the same lines the CLI always has, so the CLI is unchanged.
    """
    emit = progress or _print_progress
    ping()  # fail fast with a clear message before the expensive render/embed loop
    return _rebuild(pdfs, emit) if rebuild else _sync(pdfs, emit)


def main(argv: list[str]) -> None:
    """Resolve PDFs and index them (CLI wrapper; owns Qdrant client teardown)."""
    rebuild = "--rebuild" in argv
    pdf_args = [a for a in argv if a != "--rebuild"]
    pdfs = [Path(p) for p in pdf_args] if pdf_args else sorted(PDFS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFS found. Put PDF in {PDFS_DIR} or pass a path.")
        sys.exit(1)

    skipped = 0

    def progress(evt: dict) -> None:
        """The default printer, plus a tally for the closing summary."""
        nonlocal skipped
        if evt["phase"] == "skip":
            skipped += 1
        _print_progress(evt)

    try:
        indexed = run_ingest(pdfs, progress, rebuild=rebuild)
    finally:
        close_client()

    if rebuild:
        print(f"\nDone. Rebuilt the index with {indexed} pages.")
    else:
        tail = f", {skipped} document{'' if skipped == 1 else 's'} already up to date" if skipped else ""
        print(f"\nDone. Embedded {indexed} pages{tail}.")

if __name__ == "__main__":
    main(sys.argv[1:])
