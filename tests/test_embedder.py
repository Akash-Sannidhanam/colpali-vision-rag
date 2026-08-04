"""Tests for the batched embed path (src.embedder).

The model is stubbed at `load_model`, so these need no checkpoint, GPU, or network -
they exercise the two things batching adds that a batch of one never exercised:

1. **Padding is trimmed.** `process_images` left-pads a batch to its longest member and
   ColQwen2 zeroes those positions without removing them. Storing them would change
   MaxSim, so the mask trim is the check that keeps EMBED_BATCH_SIZE out of
   EMBED_VERSION - and it is invisible at batch 1, where nothing is ever padded.
2. **Out-of-memory backs off instead of dying.** An ingest killed mid-document leaves a
   truncated document that a later sync considers current, so the retry matters.

`_FakeModel` returns rows whose every value is the row index, so a trimmed result says
exactly which positions survived.
"""

import threading
import time

import pytest
import torch

from src import embedder

DIM = 128


class _FakeInputs(dict):
    """A processor batch: unpacks as **kwargs and answers .to(device) like a BatchFeature."""

    def to(self, device):
        return self


class _FakeProcessor:
    """Records the batch sizes it is handed and left-pads to the longest member."""

    def __init__(self, patch_counts: dict[str, int]):
        self.patch_counts = patch_counts
        self.calls: list[list[str]] = []

    def process_images(self, images):
        self.calls.append(list(images))
        counts = [self.patch_counts.get(img, 2) for img in images]
        width = max(counts)
        mask = torch.zeros(len(images), width, dtype=torch.long)
        for i, n in enumerate(counts):
            mask[i, width - n:] = 1  # left padding, as ColQwen2Processor does
        return _FakeInputs(attention_mask=mask)


class _FakeModel:
    """Emits (batch, width, DIM) where every value in row j is j."""

    device = "cpu"
    dtype = "float32"

    def __init__(self, oom_above: int | None = None, error: Exception | None = None):
        self.oom_above = oom_above
        self.error = error
        self.batch_sizes: list[int] = []

    def __call__(self, **kwargs):
        mask = kwargs["attention_mask"]
        batch, width = mask.shape
        self.batch_sizes.append(batch)
        if self.error is not None:
            raise self.error
        if self.oom_above is not None and batch > self.oom_above:
            raise RuntimeError("MPS backend out of memory (MPS allocated 9.00 GB)")
        rows = torch.arange(width, dtype=torch.float32).unsqueeze(-1).expand(width, DIM)
        return rows.unsqueeze(0).expand(batch, width, DIM).clone()


@pytest.fixture(autouse=True)
def _clear_batching_cache():
    """`_batching_is_supported` is functools.cache'd, so a verdict would otherwise leak from
    one test into the next (same reasoning as the vector_store._client note)."""
    embedder._batching_is_supported.cache_clear()
    yield
    embedder._batching_is_supported.cache_clear()


def _install(monkeypatch, model, processor, device: str = "cpu"):
    """Install a stub model and pin the device batching support is decided from.

    Defaults to a device that batches, because most tests here are about chunking and OOM
    backoff; the ones about the MPS exclusion pass `device="mps"`.
    """
    monkeypatch.setattr(embedder, "load_model", lambda: (model, processor))
    monkeypatch.setattr(embedder, "_device_and_dtype", lambda: (device, torch.float32))


def test_padding_is_trimmed_off_each_page(monkeypatch):
    """A short page in a batch keeps its own patches, not the batch's padded width."""
    processor = _FakeProcessor({"short": 2, "long": 5})
    _install(monkeypatch, _FakeModel(), processor)

    out = embedder.embed_images(["short", "long"], batch_size=2)

    # Padded width is 5; the short page must come back with 2 vectors, not 5.
    assert [len(v) for v in out] == [2, 5]
    # Left padding means the short page's real rows are the LAST two of the padded five.
    assert out[0] == [[3.0] * DIM, [4.0] * DIM]
    assert out[1] == [[float(i)] * DIM for i in range(5)]


def test_an_unpadded_batch_is_untouched(monkeypatch):
    """Equal-length pages have an all-ones mask, so nothing is dropped."""
    processor = _FakeProcessor({"a": 3, "b": 3})
    _install(monkeypatch, _FakeModel(), processor)

    out = embedder.embed_images(["a", "b"], batch_size=2)

    assert [len(v) for v in out] == [3, 3]


def test_pages_are_chunked_by_batch_size(monkeypatch):
    """7 pages at batch 3 is three forward passes of 3, 3, 1 - in page order."""
    processor = _FakeProcessor({})
    model = _FakeModel()
    _install(monkeypatch, model, processor)

    pages = [f"p{i}" for i in range(7)]
    out = embedder.embed_images(pages, batch_size=3)

    assert len(out) == 7
    assert model.batch_sizes == [3, 3, 1]
    assert processor.calls == [pages[0:3], pages[3:6], pages[6:7]]


def test_out_of_memory_halves_the_batch_and_retries_the_same_pages(monkeypatch):
    """An OOM at 4 retries those pages at 2 - and the size stays shrunk afterwards."""
    processor = _FakeProcessor({})
    model = _FakeModel(oom_above=2)
    _install(monkeypatch, model, processor)

    pages = [f"p{i}" for i in range(8)]
    out = embedder.embed_images(pages, batch_size=4)

    assert len(out) == 8                       # every page still embedded, in order
    # 4 fails, halves to 2, then stays at 2: one wasted pass, not one per batch.
    assert model.batch_sizes == [4, 2, 2, 2, 2]
    assert processor.calls[-1] == pages[6:8]


def test_the_shrink_persists_for_the_rest_of_the_document(monkeypatch):
    """The backoff is per-generator, not per-batch - a long document pays the OOM once.

    This is why ingest hands `iter_embedded` the whole page list instead of pre-chunking it
    and calling once per chunk: each call starts back at EMBED_BATCH_SIZE, so chunking would
    re-pay the failed forward pass on every chunk. At 20 pages that is 5 wasted passes
    rather than 1, on exactly the memory-tight box the backoff exists for.
    """
    model = _FakeModel(oom_above=2)
    _install(monkeypatch, model, _FakeProcessor({}))

    pages = [f"p{i}" for i in range(20)]
    out = [v for _, batch in embedder.iter_embedded(pages, batch_size=4) for v in batch]

    assert len(out) == 20
    # One failed pass at 4, then ten good ones at 2 - not a failed 4 every fifth batch.
    assert model.batch_sizes == [4] + [2] * 10
    assert model.batch_sizes.count(4) == 1


def test_iter_embedded_reports_each_batch_start_index(monkeypatch):
    """Callers derive page numbers from the yielded start, so it must track the real offset
    even after a backoff changes the batch size mid-document."""
    _install(monkeypatch, _FakeModel(oom_above=2), _FakeProcessor({}))

    seen = [(start, len(batch)) for start, batch in embedder.iter_embedded(
        [f"p{i}" for i in range(7)], batch_size=4)]

    # 4 OOMs and never yields; the run is 2+2+2+1 from offset 0, with no gaps or repeats.
    assert seen == [(0, 2), (2, 2), (4, 2), (6, 1)]


def test_a_single_page_that_cannot_fit_raises(monkeypatch):
    """Backing off below one page is impossible, so the error surfaces."""
    _install(monkeypatch, _FakeModel(oom_above=0), _FakeProcessor({}))

    with pytest.raises(RuntimeError, match="out of memory"):
        embedder.embed_images(["p0"], batch_size=1)


def test_a_non_oom_error_is_not_retried(monkeypatch):
    """A real bug must not be laundered into a slow ingest by the backoff path."""
    model = _FakeModel(error=RuntimeError("boom"))
    _install(monkeypatch, model, _FakeProcessor({}))

    with pytest.raises(RuntimeError, match="boom"):
        embedder.embed_images(["a", "b", "c", "d"], batch_size=4)

    assert model.batch_sizes == [4]            # tried once, not halved


def test_embed_image_matches_the_batched_path(monkeypatch):
    """The single-page helper is the batch path, so the two cannot drift."""
    processor = _FakeProcessor({"solo": 4})
    _install(monkeypatch, _FakeModel(), processor)

    assert embedder.embed_image("solo") == embedder.embed_images(["solo"])[0]


def test_no_pages_means_no_forward_pass(monkeypatch):
    """An empty document costs nothing (and must not index an empty batch)."""
    model = _FakeModel()
    _install(monkeypatch, model, _FakeProcessor({}))

    assert embedder.embed_images([]) == []
    assert model.batch_sizes == []


# --- the MPS batching exclusion (see embedder._batching_is_supported) ---

def test_batching_is_disabled_on_mps(monkeypatch):
    """MPS miscomputes slot 0 of a batched pass, so pages must go one at a time there.

    The failure it avoids is silent - right patch counts, no error, and an ingest that
    records the document as current - so nothing downstream would ever notice it.
    """
    model = _FakeModel()
    _install(monkeypatch, model, _FakeProcessor({}), device="mps")

    out = embedder.embed_images([f"p{i}" for i in range(5)], batch_size=4)

    assert len(out) == 5                        # every page still embedded, in order
    assert model.batch_sizes == [1, 1, 1, 1, 1]  # but never more than one per forward pass
    assert embedder._batching_is_supported() is False


def test_batching_is_kept_on_other_backends(monkeypatch):
    """The exclusion is MPS-only: cpu and cuda must still get the batched path."""
    for device in ("cpu", "cuda"):
        model = _FakeModel()
        _install(monkeypatch, model, _FakeProcessor({}), device=device)
        embedder._batching_is_supported.cache_clear()  # or the 2nd arm reuses the 1st verdict

        embedder.embed_images([f"p{i}" for i in range(5)], batch_size=4)

        assert model.batch_sizes == [4, 1], device
        assert embedder._batching_is_supported() is True, device


def test_the_device_is_only_inspected_once(monkeypatch):
    """The verdict is cached, so a long ingest does not re-derive it per batch."""
    calls = []
    model = _FakeModel()
    _install(monkeypatch, model, _FakeProcessor({}), device="mps")
    monkeypatch.setattr(embedder, "_device_and_dtype",
                        lambda: calls.append(1) or ("mps", torch.float32))

    embedder.embed_images([f"p{i}" for i in range(6)], batch_size=3)
    embedder.embed_images([f"q{i}" for i in range(6)], batch_size=3)

    assert len(calls) == 1


def test_single_page_work_skips_the_check_entirely(monkeypatch):
    """embed_image/embed_query batch nothing, so they must not even ask."""
    calls = []
    _install(monkeypatch, _FakeModel(), _FakeProcessor({"solo": 4}))
    monkeypatch.setattr(embedder, "_device_and_dtype",
                        lambda: calls.append(1) or ("cpu", torch.float32))

    embedder.embed_image("solo")

    assert calls == []


@pytest.mark.parametrize("message", ["CUDA out of memory", "can't allocate memory"])
def test_allocation_failures_are_recognised_across_backends(message):
    """CUDA, MPS and CPU word it differently; all three must count as OOM."""
    assert embedder._is_oom(RuntimeError(message))
    assert embedder._is_oom(MemoryError())
    assert not embedder._is_oom(RuntimeError("shape mismatch"))


# --- the preprocess/GPU overlap ---

def test_the_next_batch_is_preprocessed_while_the_gpu_runs_the_current_one(monkeypatch):
    """The overlap this pipelining exists for, asserted rather than assumed.

    `preprocess` is pure CPU and measured at 0.30 s/page against a 5.45 s forward, so
    running it inline idled the GPU for ~5% of every page - and much more once the forward
    pass shrinks. The fake model blocks until the processor has been called for a *later*
    batch, which can only happen if preprocessing runs ahead on another thread. Serially,
    nothing would ever set the event and the wait would time out.
    """
    started_second = threading.Event()

    class _AheadProcessor(_FakeProcessor):
        def process_images(self, images):
            out = super().process_images(images)
            if len(self.calls) >= 2:
                started_second.set()
            return out

    class _WaitingModel(_FakeModel):
        def __call__(self, **kwargs):
            # Called on the main thread; the setter runs on the prefetch thread.
            assert started_second.wait(timeout=10), "the next batch was not preprocessed ahead"
            return super().__call__(**kwargs)

    processor = _AheadProcessor({})
    _install(monkeypatch, _WaitingModel(), processor)

    out = embedder.embed_images([f"p{i}" for i in range(4)], batch_size=2)

    assert len(out) == 4
    assert processor.calls == [["p0", "p1"], ["p2", "p3"]]


def test_a_preprocess_failure_propagates_instead_of_hanging(monkeypatch):
    """A raise on the prefetch thread must surface, not deadlock the generator.

    An ingest that hangs is worse than one that crashes: `document_index` fingerprints a
    document from its first indexed page, so a killed run strands a truncated document that
    the next sync considers current.
    """
    class _BrokenProcessor(_FakeProcessor):
        def process_images(self, images):
            raise ValueError("processor exploded")

    _install(monkeypatch, _FakeModel(), _BrokenProcessor({}))

    with pytest.raises(ValueError, match="processor exploded"):
        embedder.embed_images(["a", "b", "c"], batch_size=2)


def test_the_prefetch_thread_does_not_outlive_an_abandoned_generator(monkeypatch):
    """Closing the iterator early must not leak the pool - ingest streams these per document."""
    before = threading.active_count()
    _install(monkeypatch, _FakeModel(), _FakeProcessor({}))

    it = embedder.iter_embedded([f"p{i}" for i in range(8)], batch_size=2)
    next(it)          # start the pool, then walk away
    it.close()

    for _ in range(100):                      # the pool shuts down asynchronously
        if threading.active_count() <= before:
            break
        time.sleep(0.01)
    assert threading.active_count() <= before
