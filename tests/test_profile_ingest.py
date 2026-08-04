"""Tests for the batching equivalence gate (scripts/profile_ingest.verify_equivalence).

Pure logic over stubbed embed calls: no model, no poppler, no Qdrant. The gate is what
licenses keeping EMBED_BATCH_SIZE out of EMBED_VERSION, so its *verdict* is the thing worth
pinning - and specifically its refusal to collapse two different questions into one boolean:

- "did batching change the vectors?" is the correctness question the gate exists for
- "did batching run at all?" is a coverage question about whether the check meant anything

`embedder._batching_is_supported` disables batching on MPS by design, so on Apple Silicon the
second is legitimately false while the first is definitively no-corruption. A gate that ANDs
them reports the correct configuration as corruption on every Mac - which is exactly the
regression this file exists to prevent recurring.
"""

import pytest

from scripts import profile_ingest

DIM = 8


def _page(seed: float, patches: int = 3):
    """A stand-in multivector; `seed` shifts every component so pages differ."""
    return [[seed + i] * DIM for i in range(patches)]


def _install(monkeypatch, *, single, batched, batch_shape):
    """Stub the two embed paths the gate compares, plus rendering.

    `batch_shape` is the list of batch sizes iter_embedded reports yielding, which is how the
    gate learns whether batching actually happened.
    """
    monkeypatch.setattr(profile_ingest, "pdf_to_images", lambda pdf: list(range(len(single))))
    monkeypatch.setattr(profile_ingest, "embed_image", lambda page: single[page])

    def _iter_embedded(pages, batch_size=None):
        start = 0
        for n in batch_shape:
            yield start, [batched[p] for p in pages[start:start + n]]
            start += n

    monkeypatch.setattr(profile_ingest, "iter_embedded", _iter_embedded)


def test_matching_vectors_from_a_real_batch_are_equivalent(monkeypatch):
    """The happy path: batching ran, and it produced what single-page passes produce."""
    vectors = [_page(0.0), _page(1.0), _page(2.0), _page(3.0)]
    _install(monkeypatch, single=vectors, batched=vectors, batch_shape=[4])

    verdict, detail = profile_ingest.verify_equivalence(None, None, 4)

    assert verdict == "equivalent"
    assert detail["batching_ran"] is True
    assert detail["max_abs_delta"] == 0.0


def test_a_backend_that_never_batched_is_not_applicable_not_a_failure(monkeypatch):
    """The MPS case: nothing was batched, so there is nothing to compare.

    This must NOT read as corruption - the stored vectors are right, and calling it a
    failure cries wolf on every Apple Silicon machine. It must also not read as a pass on
    batching, which was never exercised.
    """
    vectors = [_page(0.0), _page(1.0), _page(2.0)]
    _install(monkeypatch, single=vectors, batched=vectors, batch_shape=[1, 1, 1])

    verdict, detail = profile_ingest.verify_equivalence(None, None, 4)

    assert verdict == "not_applicable"
    assert detail["batching_ran"] is False
    assert detail["max_abs_delta"] == 0.0        # vacuously exact, because nothing batched
    assert detail["effective_batch_sizes"] == [1, 1, 1]


def test_drifted_vectors_are_corrupt(monkeypatch):
    """The failure the gate exists for: batching ran and changed what gets stored."""
    single = [_page(0.0), _page(1.0)]
    corrupted = [_page(0.4), _page(1.0)]         # ~the observed MPS+bf16 slot-0 drift
    _install(monkeypatch, single=single, batched=corrupted, batch_shape=[2])

    verdict, detail = profile_ingest.verify_equivalence(None, None, 2)

    assert verdict == "corrupt"
    assert detail["max_abs_delta"] == pytest.approx(0.4)


def test_a_patch_count_mismatch_is_corrupt_regardless_of_delta(monkeypatch):
    """A different number of patches means padding is being stored - never tolerate it."""
    single = [_page(0.0, patches=3)]
    padded = [_page(0.0, patches=5)]
    _install(monkeypatch, single=single, batched=padded, batch_shape=[1, 1])

    verdict, detail = profile_ingest.verify_equivalence(None, None, 2)

    assert verdict == "corrupt"                  # not "not_applicable", despite batch_shape
    assert detail["pages_with_mismatched_patch_counts"] == [1]


def test_a_document_shorter_than_the_batch_still_counts_as_batched(monkeypatch):
    """2 pages at batch 8 can only ever yield 2, and that is a genuine batched pass."""
    vectors = [_page(0.0), _page(1.0)]
    _install(monkeypatch, single=vectors, batched=vectors, batch_shape=[2])

    verdict, _ = profile_ingest.verify_equivalence(None, None, 8)

    assert verdict == "equivalent"


@pytest.mark.parametrize("verdict,expected_exit", [
    ("equivalent", 0), ("not_applicable", 0), ("corrupt", 1),
])
def test_only_corruption_exits_nonzero(monkeypatch, capsys, verdict, expected_exit):
    """A CI job gating on this must fail on corruption and only on corruption."""
    monkeypatch.setattr(profile_ingest, "verify_equivalence",
                        lambda *a, **k: (verdict, {"verdict": verdict}))
    monkeypatch.setattr(profile_ingest, "load_model", lambda: (type("M", (), {"device": "cpu"})(), None))
    monkeypatch.setattr(profile_ingest, "_default_pdf", lambda: __import__("pathlib").Path(__file__))

    assert profile_ingest.main(["--verify-equivalence"]) == expected_exit
    out = capsys.readouterr().out
    if verdict == "not_applicable":
        assert "NOT VERIFIED" in out and "equivalent" not in out.split("NOT VERIFIED")[0]
