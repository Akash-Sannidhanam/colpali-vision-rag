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

    def __init__(self, collections=(), alias_target=None, get_collections_error=None,
                 payloads=(), scroll_page=256):
        self._names = list(collections)
        self._alias_target = alias_target
        self._get_collections_error = get_collections_error
        self._payloads = list(payloads)   # one dict per stored point, for scroll()
        self._scroll_page = scroll_page   # small values force multi-page scrolls
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

    def create_payload_index(self, collection_name, field_name, field_schema):
        self.calls.append(("index", collection_name, field_name))

    def upsert(self, collection_name, points):
        self.calls.append(("upsert", collection_name))

    def scroll(self, collection_name, with_payload, with_vectors, limit, offset):
        """Paginate self._payloads, using the list index as the scroll offset."""
        self.calls.append(("scroll", collection_name))
        start = offset or 0
        page = self._payloads[start:start + min(limit, self._scroll_page)]
        nxt = start + len(page)
        points = [SimpleNamespace(id=start + i, payload=p) for i, p in enumerate(page)]
        return points, (nxt if nxt < len(self._payloads) else None)

    def delete(self, collection_name, points_selector):
        self.calls.append(("delete_points", collection_name, points_selector))


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


def test_create_collection_indexes_the_pdf_field(monkeypatch):
    # delete_document and the fingerprint lookup both filter on `pdf`; without the
    # payload index those degrade to a full scan as the corpus grows.
    fake = FakeClient(collections=[])
    _use(monkeypatch, fake)
    vector_store.begin_ingest()
    assert ("index", "pdf_pages_1", "pdf") in fake.calls


# --- incremental path: live_collection ---

def test_live_collection_returns_alias_when_one_exists(monkeypatch):
    fake = FakeClient(collections=["pdf_pages_2"], alias_target="pdf_pages_2")
    _use(monkeypatch, fake, qdrant_url="http://x")

    assert vector_store.live_collection() == ALIAS
    # an existing index is reused as-is: nothing created, nothing swapped
    assert not any(c[0] in ("create", "swap") for c in fake.calls)


def test_live_collection_bootstraps_on_a_cold_server(monkeypatch):
    fake = FakeClient(collections=[], alias_target=None)
    _use(monkeypatch, fake, qdrant_url="http://x")

    assert vector_store.live_collection() == ALIAS
    assert ("create", "pdf_pages_1") in fake.calls
    assert any(c[0] == "swap" for c in fake.calls)      # promoted so the alias resolves


def test_live_collection_embedded_creates_without_wiping(monkeypatch):
    fake = FakeClient(collections=["pdf_pages"])
    _use(monkeypatch, fake, qdrant_url=None)

    assert vector_store.live_collection() == ALIAS
    # the whole point of the incremental path: existing pages survive
    assert not any(c[0] == "delete" for c in fake.calls)


# --- incremental path: document_index / delete_document ---

def _page(pdf, content_hash="h1", embed_version="m@150"):
    return {"pdf": pdf, "content_hash": content_hash, "embed_version": embed_version}


def test_document_index_aggregates_counts_and_fingerprints(monkeypatch):
    fake = FakeClient(payloads=[_page("a.pdf"), _page("b.pdf", "h2"), _page("a.pdf")])
    monkeypatch.setattr(vector_store, "get_client", lambda: fake)

    index = vector_store.document_index()

    assert list(index) == ["a.pdf", "b.pdf"]                  # sorted by name
    assert index["a.pdf"] == {"page_count": 2, "content_hash": "h1", "embed_version": "m@150"}
    assert index["b.pdf"]["content_hash"] == "h2"


def test_document_index_pages_through_a_long_scroll(monkeypatch):
    fake = FakeClient(payloads=[_page("a.pdf")] * 5, scroll_page=2)
    monkeypatch.setattr(vector_store, "get_client", lambda: fake)

    assert vector_store.document_index()["a.pdf"]["page_count"] == 5
    assert len([c for c in fake.calls if c[0] == "scroll"]) == 3   # 2 + 2 + 1


def test_document_index_defaults_missing_fingerprints_to_empty(monkeypatch):
    # Points written before fingerprinting existed: "" never equals a real sha256, so
    # the next sync re-embeds them once rather than trusting a stale vector.
    fake = FakeClient(payloads=[{"pdf": "old.pdf", "page_number": 1}])
    monkeypatch.setattr(vector_store, "get_client", lambda: fake)

    assert vector_store.document_index()["old.pdf"] == {
        "page_count": 1, "content_hash": "", "embed_version": "",
    }


def test_list_documents_derives_from_the_index(monkeypatch):
    fake = FakeClient(payloads=[_page("b.pdf"), _page("a.pdf"), _page("a.pdf")])
    monkeypatch.setattr(vector_store, "get_client", lambda: fake)

    assert vector_store.list_documents() == [
        {"pdf": "a.pdf", "page_count": 2}, {"pdf": "b.pdf", "page_count": 1},
    ]


def test_delete_document_filters_on_pdf_and_returns_page_count(monkeypatch):
    fake = FakeClient(payloads=[_page("a.pdf"), _page("a.pdf"), _page("b.pdf")])
    monkeypatch.setattr(vector_store, "get_client", lambda: fake)

    assert vector_store.delete_document("a.pdf") == 2

    deletes = [c for c in fake.calls if c[0] == "delete_points"]
    assert len(deletes) == 1
    assert deletes[0][1] == ALIAS                              # targets the live alias
    assert deletes[0][2] == qm.FilterSelector(filter=qm.Filter(
        must=[qm.FieldCondition(key="pdf", match=qm.MatchValue(value="a.pdf"))]))


def test_delete_document_is_a_noop_for_an_unknown_pdf(monkeypatch):
    fake = FakeClient(payloads=[_page("a.pdf")])
    monkeypatch.setattr(vector_store, "get_client", lambda: fake)

    assert vector_store.delete_document("ghost.pdf") == 0
    assert not any(c[0] == "delete_points" for c in fake.calls)


# --- deterministic point ids ---

def test_point_id_is_stable_per_page_and_distinct_across_pages():
    # Stability is what makes an incremental re-ingest overwrite in place instead of
    # duplicating; distinctness is what stops pages from clobbering each other.
    assert vector_store.point_id("a.pdf", 1) == vector_store.point_id("a.pdf", 1)
    assert vector_store.point_id("a.pdf", 1) != vector_store.point_id("a.pdf", 2)
    assert vector_store.point_id("a.pdf", 1) != vector_store.point_id("b.pdf", 1)


def test_build_point_carries_the_fingerprint_payload():
    point = vector_store.build_point([[0.0] * 128], "a.pdf", 3, "/img/a_page_3.png",
                                     "deadbeef", "model@150")
    assert point.id == vector_store.point_id("a.pdf", 3)
    assert point.payload == {
        "pdf": "a.pdf", "page_number": 3, "image_path": "/img/a_page_3.png",
        "content_hash": "deadbeef", "embed_version": "model@150",
    }


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
