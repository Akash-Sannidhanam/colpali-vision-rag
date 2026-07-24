"""Qdrant multivector store - one multivector per page, ranked by MaxSim.

Two write paths, both reached through the mode-hiding seam at the bottom so
`ingest.py` never branches on deployment mode:

- **Incremental** (the default) - upsert straight into the live collection via
  `live_collection()`, touching only the documents that changed, and remove one
  document with `delete_document()`. Adding a document is O(its own pages), and it
  inherits the crash-safety that matters for free: existing points are never
  touched, so an interrupted add leaves every other document intact.
- **Rebuild** (`begin_ingest`/`finish_ingest`/`abort_ingest`) - the atomic
  wholesale path. Server deployments build a fresh versioned physical collection
  (`pdf_pages_<n>`) and swap the read alias `COLLECTION_NAME` onto it in a single
  atomic operation only once the build completes, so a mid-ingest crash leaves the
  previous index serving. The embedded on-disk fallback (no QDRANT_URL) keeps the
  simpler wipe-and-rebuild path, since QdrantLocal does not support aliases.

`search()`/`upsert_pages()` reference the alias, which Qdrant resolves to the live
physical collection transparently.
"""

import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from src.config import (
    COLLECTION_NAME,
    QDRANT_API_KEY,
    QDRANT_PATH,
    QDRANT_URL,
    RESCORE_OVERSAMPLING,
    RETRIEVE_K,
    VECTOR_DIM,
)
from src.logging_setup import get_logger

log = get_logger("qdrant")

_client: QdrantClient | None = None

# Physical collections are named "pdf_pages_<n>"; COLLECTION_NAME ("pdf_pages") is
# the alias that resolves to the live one.
_PHYSICAL_PREFIX = f"{COLLECTION_NAME}_"

# Namespace for deriving a stable point id from (pdf name, page number), so an
# incremental re-ingest of a page overwrites it in place instead of duplicating it.
_POINT_NAMESPACE = uuid.UUID("6f9e6a5c-1f1e-4c8a-9b3d-0a1c2d3e4f50")


def get_client() -> QdrantClient:
    """Connect to the Qdrant server (QDRANT_URL) once and reuse it.

    Falls back to the embedded on-disk store at QDRANT_PATH when QDRANT_URL is
    unset, so quick prototyping/tests work without a container running.
    """
    global _client
    if _client is None:
        if QDRANT_URL:
            _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            _client = QdrantClient(path=str(QDRANT_PATH))  # local prototyping fallback
    return _client

def close_client() -> None:
    """Close the client connection (a server socket, or the on-disk file lock)."""
    global _client
    if _client is not None:
        _client.close()
        _client = None

def ping() -> None:
    """Verify Qdrant is reachable; raise a clear RuntimeError if not.

    A reusable startup/health probe (Phase 3's /health calls this too) that
    surfaces connectivity problems here, with an actionable message, instead of
    deep inside search().
    """
    try:
        get_client().get_collections()
    except Exception as exc:
        target = QDRANT_URL or f"embedded store at {QDRANT_PATH}"
        raise RuntimeError(
            f"Cannot reach Qdrant ({target}): {exc}. Is the server running? "
            f"Start it with docker-compose, or check QDRANT_URL in .env (see README)."
        ) from exc


def _create_collection(client: QdrantClient, name: str) -> None:
    """Create one empty multivector collection with the project's vector config."""
    client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(
            size=VECTOR_DIM,
            distance=qm.Distance.COSINE,
            on_disk=True,  # full-precision vectors live on disk (read only during rescore)
            multivector_config = qm.MultiVectorConfig(
                comparator = qm.MultiVectorComparator.MAX_SIM),
            ),
        # Binary quantization: each 128-d vector -> 128 bits (32x smaller), kept
        # in RAM for a fast first pass; the on-disk originals rescore the top hits.
        quantization_config=qm.BinaryQuantization(
            binary=qm.BinaryQuantizationConfig(always_ram=True),
        ),
    )
    # Every per-document operation (fingerprint lookup, delete_document) filters on
    # `pdf`; indexing it keeps those off a full payload scan as the corpus grows.
    client.create_payload_index(
        collection_name=name, field_name="pdf", field_schema=qm.PayloadSchemaType.KEYWORD,
    )

def ensure_collection(reset: bool = False) -> None:
    """Create the collection named COLLECTION_NAME if it is missing (embedded path).

    The simple wipe-and-rebuild path used in embedded mode; `reset=True` drops and
    recreates in place. Server ingest uses the atomic begin_ingest/finish_ingest
    path instead.
    """
    client = get_client()
    if reset and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    if not client.collection_exists(COLLECTION_NAME):
        _create_collection(client, COLLECTION_NAME)


# --- Versioned physical collections + atomic alias swap (server path) ---

def _physical_name(version: int) -> str:
    """The physical collection name for a version number (`pdf_pages_<n>`)."""
    return f"{_PHYSICAL_PREFIX}{version}"

def _list_physical_versions(client: QdrantClient) -> list[int]:
    """Sorted integer suffixes of existing pdf_pages_<n> collections."""
    versions: list[int] = []
    for c in client.get_collections().collections:
        suffix = c.name[len(_PHYSICAL_PREFIX):] if c.name.startswith(_PHYSICAL_PREFIX) else ""
        if suffix.isdigit():
            versions.append(int(suffix))
    return sorted(versions)

def _next_physical_name(client: QdrantClient) -> str:
    """The next free pdf_pages_<n> name (max existing + 1, or pdf_pages_1)."""
    versions = _list_physical_versions(client)
    return _physical_name(versions[-1] + 1 if versions else 1)

def _current_alias_target(client: QdrantClient) -> str | None:
    """Physical collection COLLECTION_NAME resolves to now, or None if no alias."""
    for a in client.get_aliases().aliases:
        if a.alias_name == COLLECTION_NAME:
            return a.collection_name
    return None

def _drop_quietly(client: QdrantClient, name: str) -> None:
    """Best-effort delete; never let cleanup mask the primary outcome."""
    try:
        if client.collection_exists(name):
            client.delete_collection(name)
    except Exception:
        log.warning("failed to drop collection during cleanup",
                    extra={"collection": name}, exc_info=True)

def _sweep_orphans(client: QdrantClient, keep: str) -> None:
    """Drop stray pdf_pages_<n> collections left by earlier crashes (best-effort)."""
    current = _current_alias_target(client)
    for version in _list_physical_versions(client):
        name = _physical_name(version)
        if name != keep and name != current:
            _drop_quietly(client, name)

def create_collection_version() -> str:
    """Create a fresh empty physical collection and return its name (server path)."""
    client = get_client()
    name = _next_physical_name(client)
    _create_collection(client, name)
    return name

def promote_collection_version(new_physical: str) -> None:
    """Atomically point COLLECTION_NAME at new_physical, then drop the old physical.

    The delete-old-alias + create-new-alias operations are applied by Qdrant as a
    single atomic batch, so readers never see a missing or ambiguous alias.
    """
    client = get_client()
    old = _current_alias_target(client)  # None on the first ingest

    # A leftover real collection literally named COLLECTION_NAME (from the old
    # wipe-before-ingest path) would occupy the alias name; free it first. Gated on
    # `old is None` because an existing alias also reports collection_exists True.
    if old is None and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    ops: list = []
    if old is not None:
        ops.append(qm.DeleteAliasOperation(
            delete_alias=qm.DeleteAlias(alias_name=COLLECTION_NAME)))
    ops.append(qm.CreateAliasOperation(
        create_alias=qm.CreateAlias(collection_name=new_physical, alias_name=COLLECTION_NAME)))
    client.update_collection_aliases(change_aliases_operations=ops)

    if old is not None and old != new_physical:
        _drop_quietly(client, old)   # swap already succeeded; this is best-effort
    _sweep_orphans(client, keep=new_physical)

def abort_ingest(target: str) -> None:
    """Drop a partially-built physical collection after a failed ingest (server path).

    Guarded so it never deletes the live alias target - if a promote already
    succeeded, `target` is live and this is a no-op. Embedded mode has nothing
    versioned to abort.
    """
    if not QDRANT_URL:
        return
    client = get_client()
    if target != _current_alias_target(client):
        _drop_quietly(client, target)


# --- Mode-hiding ingest orchestration (ingest.py stays mode-agnostic) ---

def begin_ingest() -> str:
    """Start an ingest; return the collection name the build should upsert into.

    Server: a fresh versioned physical collection (the live alias keeps serving the
    previous index until finish_ingest). Embedded: wipe-and-rebuild COLLECTION_NAME
    in place (QdrantLocal has no alias support).
    """
    if QDRANT_URL:
        return create_collection_version()
    ensure_collection(reset=True)
    return COLLECTION_NAME

def finish_ingest(target: str) -> None:
    """Publish a completed ingest.

    Server: atomically swap the alias onto `target` and drop the old physical.
    Embedded: the data is already in COLLECTION_NAME - nothing to do.
    """
    if QDRANT_URL:
        promote_collection_version(target)

def live_collection() -> str:
    """The collection an incremental ingest writes into, creating it if this is a cold start.

    The counterpart to begin_ingest for the non-rebuild path: instead of building a new
    version off to the side, it returns the collection that is already serving reads, so
    upserts land in the live index immediately. Server mode returns the alias (creating
    and promoting a first physical collection when no alias exists yet); embedded mode
    creates COLLECTION_NAME if missing - `reset=False`, so existing pages survive.
    """
    if QDRANT_URL:
        if _current_alias_target(get_client()) is None:
            promote_collection_version(create_collection_version())
        return COLLECTION_NAME
    ensure_collection(reset=False)
    return COLLECTION_NAME

def delete_document(pdf_name: str) -> int:
    """Remove every page of one PDF from the live collection; return the pages removed.

    Filters on the indexed `pdf` payload field, so it is independent of point ids and
    works identically in both modes (QdrantLocal implements filtered delete too). The
    count comes from document_index() beforehand because Qdrant's UpdateResult reports
    an operation status, not a row count.
    """
    removed = document_index().get(pdf_name, {}).get("page_count", 0)
    if not removed:
        return 0
    get_client().delete(
        collection_name=COLLECTION_NAME,
        points_selector=qm.FilterSelector(filter=_pdf_filter(pdf_name)),
    )
    log.info("deleted document from index", extra={"pdf": pdf_name, "pages": removed})
    return removed

def _pdf_filter(pdf_name: str) -> qm.Filter:
    """Match every point belonging to one PDF."""
    return qm.Filter(must=[qm.FieldCondition(key="pdf", match=qm.MatchValue(value=pdf_name))])

def point_id(pdf_name: str, page_number: int) -> str:
    """The stable point id for one page: a uuid5 of (pdf name, page number).

    Deriving the id rather than assigning it sequentially is what makes incremental
    ingest safe - re-ingesting a page overwrites it in place, and adding a document to
    a live collection cannot collide with ids handed out by an earlier run.
    """
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{pdf_name}:{page_number}"))

def build_point(
    multivector, pdf_name, page_number, image_path, content_hash="", embed_version="",
) -> qm.PointStruct:
    """Build one page's point: its multivector plus source metadata and fingerprint.

    `content_hash` + `embed_version` are what let a later ingest decide whether this
    page is still current (see `ingest.run_ingest`) - the PDF's bytes and the embedding
    config that produced the vector, respectively.
    """
    return qm.PointStruct(
        id = point_id(pdf_name, page_number),
        vector = multivector,
        payload = {
            "pdf": pdf_name,
            "page_number": page_number,
            "image_path": str(image_path),
            "content_hash": content_hash,
            "embed_version": embed_version,
        },
    )

def upsert_pages(points: list[qm.PointStruct], collection_name: str = COLLECTION_NAME) -> None:
    """Store a batch of page points in a single round-trip (skips empty batches).

    During an atomic server ingest, `collection_name` is the new physical collection
    being built - the alias still points at the previous index until finish_ingest.
    """
    if not points:
        return
    client = get_client()
    client.upsert(collection_name=collection_name, points=points)

def search(query_multivector: list[list[float]], top_k: int = RETRIEVE_K) -> list[dict]:
    """Return the top_k pages for a query multivector, best score first.

    Drops points whose payload is missing a required field (`pdf`, `page_number`,
    `image_path`) or whose page image is no longer on disk - which happens when the
    persisted index outlives a wiped `page_images/`. Each drop is logged at WARNING so
    a stale index stays visible rather than silently answering off a shrunken set;
    downstream (rerank/answer/highlight) can then assume every hit resolves to a page.
    """
    client = get_client()
    response = client.query_points(
        collection_name = COLLECTION_NAME,
        query = query_multivector,
        limit = top_k,
        with_payload = True,
        # Binary quantization is lossy: pull extra candidates on the fast quantized
        # pass, then rescore them against the full-precision vectors to keep recall.
        # Oversampling depth is tunable (RESCORE_OVERSAMPLING) to trade recall vs I/O.
        search_params = qm.SearchParams(
            quantization = qm.QuantizationSearchParams(
                rescore=True, oversampling=RESCORE_OVERSAMPLING),
        ),
    )

    hits = []
    for p in response.points:
        payload = p.payload or {}
        if not all(field in payload for field in ("pdf", "page_number", "image_path")):
            log.warning("dropped hit with incomplete payload", extra={"point_id": p.id})
            continue
        # Validate payload types before using them
        if not isinstance(payload["image_path"], str) or not payload["image_path"]:
            log.warning("dropped hit with invalid image_path", extra={"point_id": p.id})
            continue
        if not isinstance(payload["pdf"], str):
            log.warning("dropped hit with invalid pdf", extra={"point_id": p.id})
            continue
        if not isinstance(payload["page_number"], int):
            log.warning("dropped hit with invalid page_number", extra={"point_id": p.id})
            continue
        if not Path(payload["image_path"]).exists():
            log.warning("dropped hit with missing image file",
                        extra={"point_id": p.id, "image_path": payload["image_path"]})
            continue
        hits.append({**payload, "score": round(p.score, 4)})
    return hits

def document_index() -> dict[str, dict]:
    """Every indexed PDF -> {page_count, content_hash, embed_version}.

    The one place the collection is scrolled, shared by `list_documents` (the /corpus
    rail), `delete_document` (page counts), and `ingest` (the skip-unchanged decision).
    Scrolls in pages and pulls three payload fields with no vectors, so it stays cheap
    as the corpus grows past the sample set.

    A document's fingerprint is taken from the first page seen for it; all of its pages
    are written by one ingest, so they agree. Pages indexed before fingerprinting
    existed report `""`, which never matches a real hash - so they re-embed once.
    """
    client = get_client()
    index: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=["pdf", "content_hash", "embed_version"],
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        for p in points:
            payload = p.payload or {}
            pdf = payload.get("pdf")
            if not pdf:
                continue
            entry = index.setdefault(pdf, {
                "page_count": 0,
                "content_hash": payload.get("content_hash") or "",
                "embed_version": payload.get("embed_version") or "",
            })
            entry["page_count"] += 1
        if offset is None:
            break
    return dict(sorted(index.items()))

def list_documents() -> list[dict]:
    """List the indexed PDFs and their page counts, by name (powers GET /corpus)."""
    return [{"pdf": pdf, "page_count": e["page_count"]} for pdf, e in document_index().items()]
