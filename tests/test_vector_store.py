"""Tests for the atomic alias-swap ingest logic and health check in
src.vector_store.

A small recording FakeClient stands in for QdrantClient, so these run with no
server. Two monkeypatch details matter: patch `vector_store.get_client` (not the
`_client` global, which would leak between tests) and patch the by-value
`vector_store.QDRANT_URL` (not `src.config.QDRANT_URL`).
"""

from types import SimpleNamespace

import pytest
from qdrant_client import models as qm

from src import vector_store

ALIAS = vector_store.COLLECTION_NAME  # "pdf_pages"


class FakeClient:
    """Records collection/alias operations for assertions; no real Qdrant."""

    def __init__(self, collections=(), alias_target=None, get_collections_error=None):
        self._names = list(collections)
        self._alias_target = alias_target
        self._get_collections_error = get_collections_error
        self.calls: list[tuple] = []

    def get_collections(self):
        if self._get_collections_error is not None:
            raise self._get_collections_error
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in self._names])

    def get_aliases(self):
        aliases = ([SimpleNamespace(alias_name=ALIAS, collection_name=self._alias_target)]
                   if self._alias_target else [])
        return SimpleNamespace(aliases=aliases)

    def collection_exists(self, name):
        return name in self._names

    def create_collection(self, collection_name, **kwargs):
        self.calls.append(("create", collection_name))
        self._names.append(collection_name)

    def delete_collection(self, name):
        self.calls.append(("delete", name))
        if name in self._names:
            self._names.remove(name)

    def update_collection_aliases(self, change_aliases_operations):
        self.calls.append(("swap", change_aliases_operations))
        # Reflect the create-alias so a later _current_alias_target() is realistic.
        for op in change_aliases_operations:
            create = getattr(op, "create_alias", None)
            if create is not None and create.alias_name == ALIAS:
                self._alias_target = create.collection_name

    def upsert(self, collection_name, points):
        self.calls.append(("upsert", collection_name))


def _use(monkeypatch, fake, *, qdrant_url="http://x"):
    """Point vector_store at `fake` and select server (default) or embedded mode."""
    monkeypatch.setattr(vector_store, "get_client", lambda: fake)
    monkeypatch.setattr(vector_store, "QDRANT_URL", qdrant_url)


# --- versioned naming ---

def test_next_physical_name_increments_and_ignores_non_numeric():
    fake = FakeClient(collections=["pdf_pages_1", "pdf_pages_3", "pdf_pages_x", "other"])
    assert vector_store._next_physical_name(fake) == "pdf_pages_4"


def test_next_physical_name_bootstraps_to_one():
    assert vector_store._next_physical_name(FakeClient(collections=[])) == "pdf_pages_1"


# --- atomic swap ---

def test_promote_swaps_atomically_then_deletes_old(monkeypatch):
    fake = FakeClient(collections=["pdf_pages_2", "pdf_pages_3"], alias_target="pdf_pages_2")
    _use(monkeypatch, fake)

    vector_store.promote_collection_version("pdf_pages_3")

    swaps = [c for c in fake.calls if c[0] == "swap"]
    assert len(swaps) == 1                       # one atomic call
    assert swaps[0][1] == [                       # delete-before-create, exact payload
        qm.DeleteAliasOperation(delete_alias=qm.DeleteAlias(alias_name="pdf_pages")),
        qm.CreateAliasOperation(create_alias=qm.CreateAlias(
            collection_name="pdf_pages_3", alias_name="pdf_pages")),
    ]
    # the old physical is dropped, and only AFTER the swap
    assert fake.calls.index(("delete", "pdf_pages_2")) > fake.calls.index(swaps[0])


def test_first_ingest_creates_alias_without_delete_op(monkeypatch):
    fake = FakeClient(collections=["pdf_pages_1"], alias_target=None)
    _use(monkeypatch, fake)

    vector_store.promote_collection_version("pdf_pages_1")

    swaps = [c for c in fake.calls if c[0] == "swap"]
    assert len(swaps) == 1
    assert swaps[0][1] == [qm.CreateAliasOperation(create_alias=qm.CreateAlias(
        collection_name="pdf_pages_1", alias_name="pdf_pages"))]   # create only, no delete op
    assert not any(c[0] == "delete" for c in fake.calls)           # nothing to clean up


def test_promote_sweeps_orphans_from_earlier_crashes(monkeypatch):
    # pdf_pages_4 is a stray partial left by a hard-killed prior ingest; promoting a
    # new version must both drop the old alias target AND sweep the orphan.
    fake = FakeClient(collections=["pdf_pages_3", "pdf_pages_4", "pdf_pages_5"],
                      alias_target="pdf_pages_3")
    _use(monkeypatch, fake)

    vector_store.promote_collection_version("pdf_pages_5")

    assert ("delete", "pdf_pages_3") in fake.calls   # old alias target dropped
    assert ("delete", "pdf_pages_4") in fake.calls   # orphaned partial swept
    # the freshly-promoted collection survives
    assert not any(c == ("delete", "pdf_pages_5") for c in fake.calls)


def test_promote_migrates_legacy_real_collection(monkeypatch):
    # A real (non-alias) "pdf_pages" from the old wipe path must be freed first.
    fake = FakeClient(collections=["pdf_pages"], alias_target=None)
    _use(monkeypatch, fake)

    vector_store.promote_collection_version("pdf_pages_1")

    assert ("delete", "pdf_pages") in fake.calls
    swaps = [c for c in fake.calls if c[0] == "swap"]
    assert swaps[0][1] == [qm.CreateAliasOperation(create_alias=qm.CreateAlias(
        collection_name="pdf_pages_1", alias_name="pdf_pages"))]
    # the legacy collection is freed before the alias is created
    assert fake.calls.index(("delete", "pdf_pages")) < fake.calls.index(swaps[0])


# --- mode-hiding orchestration ---

def test_begin_ingest_embedded_resets_in_place(monkeypatch):
    fake = FakeClient(collections=["pdf_pages"])
    _use(monkeypatch, fake, qdrant_url=None)

    assert vector_store.begin_ingest() == "pdf_pages"
    assert ("delete", "pdf_pages") in fake.calls and ("create", "pdf_pages") in fake.calls
    assert not any(c[0] == "swap" for c in fake.calls)   # embedded never aliases


def test_begin_ingest_server_creates_version_without_touching_alias(monkeypatch):
    fake = FakeClient(collections=[])
    _use(monkeypatch, fake, qdrant_url="http://x")

    assert vector_store.begin_ingest() == "pdf_pages_1"
    assert ("create", "pdf_pages_1") in fake.calls
    assert not any(c[0] in ("swap", "delete") for c in fake.calls)  # alias untouched during build


# --- upsert targeting ---

def test_upsert_targets_given_collection_else_alias(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(vector_store, "get_client", lambda: fake)

    vector_store.upsert_pages([object()], collection_name="pdf_pages_3")  # build target
    vector_store.upsert_pages([object()])                                 # default -> alias

    assert ("upsert", "pdf_pages_3") in fake.calls
    assert ("upsert", "pdf_pages") in fake.calls


def test_upsert_skips_empty_batch(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(vector_store, "get_client", lambda: fake)
    vector_store.upsert_pages([])
    assert fake.calls == []


# --- abort ---

def test_abort_ingest_drops_partial(monkeypatch):
    fake = FakeClient(collections=["pdf_pages_2", "pdf_pages_3"], alias_target="pdf_pages_2")
    _use(monkeypatch, fake, qdrant_url="http://x")
    vector_store.abort_ingest("pdf_pages_3")            # partial, not the live alias
    assert ("delete", "pdf_pages_3") in fake.calls


def test_abort_ingest_never_drops_live_target(monkeypatch):
    fake = FakeClient(collections=["pdf_pages_3"], alias_target="pdf_pages_3")
    _use(monkeypatch, fake, qdrant_url="http://x")
    vector_store.abort_ingest("pdf_pages_3")            # already live -> no-op
    assert not any(c[0] == "delete" for c in fake.calls)


def test_abort_ingest_noop_on_embedded(monkeypatch):
    fake = FakeClient(collections=["pdf_pages"])
    _use(monkeypatch, fake, qdrant_url=None)
    vector_store.abort_ingest("pdf_pages")
    assert fake.calls == []


# --- health check ---

def test_ping_raises_clear_error_when_unreachable(monkeypatch):
    fake = FakeClient(get_collections_error=ConnectionError("refused"))
    monkeypatch.setattr(vector_store, "get_client", lambda: fake)
    with pytest.raises(RuntimeError, match="Cannot reach Qdrant"):
        vector_store.ping()


def test_ping_ok_when_reachable(monkeypatch):
    fake = FakeClient(collections=["pdf_pages"])
    monkeypatch.setattr(vector_store, "get_client", lambda: fake)
    assert vector_store.ping() is None


# --- search hit filtering ---

def _search_client(points):
    """A stand-in client whose query_points returns the given fake points."""
    return SimpleNamespace(query_points=lambda **kw: SimpleNamespace(points=points))


def test_search_keeps_valid_hits_and_drops_invalid(monkeypatch, tmp_path):
    img = tmp_path / "page1.png"
    img.write_bytes(b"x")  # a page image that exists on disk
    points = [
        SimpleNamespace(id=1, score=0.9912,
                        payload={"pdf": "a.pdf", "page_number": 1, "image_path": str(img)}),
        SimpleNamespace(id=2, score=0.98,  # image file no longer on disk (stale index)
                        payload={"pdf": "a.pdf", "page_number": 2,
                                 "image_path": str(tmp_path / "gone.png")}),
        SimpleNamespace(id=3, score=0.97,  # missing image_path field
                        payload={"pdf": "a.pdf", "page_number": 3}),
        SimpleNamespace(id=4, score=0.96, payload=None),  # no payload at all
    ]
    monkeypatch.setattr(vector_store, "get_client", lambda: _search_client(points))

    hits = vector_store.search([[0.0] * 128])

    assert len(hits) == 1  # only the fully-valid, on-disk hit survives
    assert hits[0] == {"pdf": "a.pdf", "page_number": 1,
                       "image_path": str(img), "score": 0.9912}


def test_search_passes_configured_oversampling(monkeypatch, tmp_path):
    # The RESCORE_OVERSAMPLING knob must reach Qdrant's QuantizationSearchParams so
    # the recall/I-O trade-off is actually tunable (patch the by-value module global).
    img = tmp_path / "p.png"
    img.write_bytes(b"x")
    captured: dict = {}

    def query_points(**kw):
        captured.update(kw)
        return SimpleNamespace(points=[SimpleNamespace(
            id=1, score=0.9,
            payload={"pdf": "a.pdf", "page_number": 1, "image_path": str(img)})])

    monkeypatch.setattr(vector_store, "get_client",
                        lambda: SimpleNamespace(query_points=query_points))
    monkeypatch.setattr(vector_store, "RESCORE_OVERSAMPLING", 3.5)

    vector_store.search([[0.0] * 128])

    quant = captured["search_params"].quantization
    assert quant.rescore is True
    assert quant.oversampling == 3.5


def test_search_returns_all_when_every_hit_is_valid(monkeypatch, tmp_path):
    imgs = [tmp_path / f"p{n}.png" for n in (1, 2)]
    for p in imgs:
        p.write_bytes(b"x")
    points = [
        SimpleNamespace(id=n, score=1.0 - n / 100,
                        payload={"pdf": "a.pdf", "page_number": n, "image_path": str(imgs[n - 1])})
        for n in (1, 2)
    ]
    monkeypatch.setattr(vector_store, "get_client", lambda: _search_client(points))

    hits = vector_store.search([[0.0] * 128])

    assert [h["page_number"] for h in hits] == [1, 2]
