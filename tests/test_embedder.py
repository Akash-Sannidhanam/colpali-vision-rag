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


def _install(monkeypatch, model, processor, verified: bool | None = True):
    """Install a stub model, and say what the backend's batching has already been found to be.

    `verified` seeds the cached self-check verdict (a module global, so it is patched rather
    than left to leak between tests - same reasoning as the vector_store._client note).
    It defaults to True because most tests here are about chunking and OOM backoff, not about
    the check, and an unverified backend would insert a probe forward pass into every one of
    their expected batch sequences. The tests that exercise the check pass `verified=None`.
    """
    monkeypatch.setattr(embedder, "load_model", lambda: (model, processor))
    monkeypatch.setattr(embedder, "_batching_verified", verified)


class _SlotZeroCorruptingModel(_FakeModel):
    """Reproduces the MPS+bf16 failure: the first sequence of a multi-page batch is wrong.

    Slots 1..n-1 come back correct, patch counts are right, and nothing raises - which is
    exactly why the corruption is invisible without an explicit check.
    """

    def __call__(self, **kwargs):
        out = super().__call__(**kwargs)
        if out.shape[0] > 1:
            out = out.clone()
            out[0] += 0.4     # ~ the observed per-component drift on unit-norm vectors
        return out


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


# --- the batching self-check (see embedder._batching_is_trustworthy) ---

def test_a_corrupting_backend_never_yields_a_poisoned_vector(monkeypatch):
    """The whole point: a backend that miscomputes batch slot 0 must not reach the index.

    Every page must come back matching what a single-page pass produces, because that is
    what actually gets stored - a vector that is silently 0.4 off per component retrieves
    wrongly forever and the ingest still records the document as current.
    """
    model = _SlotZeroCorruptingModel()
    _install(monkeypatch, model, _FakeProcessor({}), verified=None)

    pages = [f"p{i}" for i in range(6)]
    got = [v for _, batch in embedder.iter_embedded(pages, batch_size=3) for v in batch]
    reference = [[[float(i)] * DIM for i in range(2)]] * 6   # what a batch of one produces

    assert got == reference          # nothing corrupted made it out
    assert embedder._batching_verified is False


def test_the_check_pins_the_process_to_single_pages_after_it_fails(monkeypatch):
    """One failed check, not one per batch or per document - the verdict is cached."""
    model = _SlotZeroCorruptingModel()
    _install(monkeypatch, model, _FakeProcessor({}), verified=None)

    list(embedder.iter_embedded([f"p{i}" for i in range(4)], batch_size=4))
    first_run = list(model.batch_sizes)
    model.batch_sizes.clear()
    list(embedder.iter_embedded([f"q{i}" for i in range(4)], batch_size=4))

    # First run: the batch of 4, the single-page probe, then 4 single passes.
    assert first_run == [4, 1, 1, 1, 1, 1]
    # Second run: already knows the backend is bad, so no batch and no re-probe.
    assert model.batch_sizes == [1, 1, 1, 1]


def test_a_healthy_backend_is_checked_once_and_keeps_batching(monkeypatch):
    """The guard must not cost a probe per batch on a backend that is fine."""
    model = _FakeModel()
    _install(monkeypatch, model, _FakeProcessor({}), verified=None)

    out = embedder.embed_images([f"p{i}" for i in range(9)], batch_size=3)

    assert len(out) == 9
    # 3 batches of 3, plus exactly one single-page probe on the first of them.
    assert model.batch_sizes == [3, 1, 3, 3]
    assert embedder._batching_verified is True


def test_single_page_work_is_never_probed(monkeypatch):
    """embed_image and embed_query paths batch nothing, so they must pay nothing."""
    model = _FakeModel()
    _install(monkeypatch, model, _FakeProcessor({"solo": 4}), verified=None)

    embedder.embed_image("solo")

    assert model.batch_sizes == [1]              # no probe forward pass
    assert embedder._batching_verified is None   # and no verdict claimed either way


def test_vectors_agree_rejects_a_patch_count_mismatch():
    """A different number of patches is a padding bug, not drift - never tolerate it."""
    assert not embedder._vectors_agree([[0.0] * DIM], [[0.0] * DIM, [0.0] * DIM])
    assert embedder._vectors_agree([[0.0] * DIM], [[0.0] * DIM])
    # Float noise passes, the observed corruption does not.
    assert embedder._vectors_agree([[0.0] * DIM], [[1e-6] * DIM])
    assert not embedder._vectors_agree([[0.0] * DIM], [[0.4] * DIM])


@pytest.mark.parametrize("message", ["CUDA out of memory", "can't allocate memory"])
def test_allocation_failures_are_recognised_across_backends(message):
    """CUDA, MPS and CPU word it differently; all three must count as OOM."""
    assert embedder._is_oom(RuntimeError(message))
    assert embedder._is_oom(MemoryError())
    assert not embedder._is_oom(RuntimeError("shape mismatch"))
