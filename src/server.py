"""Warm FastAPI serving layer for the vision RAG pipeline.

A single-worker HTTP surface over the LangGraph pipeline. The ~2B ColQwen2 model is
loaded once at boot (`lifespan`), so the cold start is paid at startup, not per query.
All heavy work runs in a threadpool behind `gpu_arbiter.GpuArbiter`, so the one
GPU-resident model is never asked to run two forward passes at once and `/health` stays
responsive. Anything touching the model goes through `_gpu.run_exclusive` rather than
taking a lock itself - that is what makes shedding (503 + `Retry-After` past
`GPU_WAIT_TIMEOUT_S`), disconnect-safety, and an ingest that yields between page batches
properties of the server rather than of each endpoint that remembered them.

Endpoints:
  POST /query   {question}          -> answer + visual citation + used pages + meta
  POST /heatmap {question,pdf,page_number} -> per-patch MaxSim heatmap grid for one page (on-demand)
  GET  /health                      -> model-loaded + Qdrant reachability + corpus integrity
  GET  /corpus                      -> indexed documents + page counts (for the UI rail)
  GET  /corpus/{pdf}/pages          -> every page of one document + image URLs (the viewer)
  GET  /corpus/{pdf}/file           -> download that document's original PDF
  POST /ingest  (multipart PDF)     -> render/embed/index a PDF (blocking, holds the lock)
  DELETE /corpus/{pdf}              -> drop a document's vectors, page images, crops, PDF
  /images/...                       -> static page/crop/annotated PNGs
  /...                              -> the built UI (ui/dist), when it has been built

Everything above except `/health`, `/images` and the UI lives on the `api` router, which
carries the `require_api_key` + `rate_limit` dependencies. Add new endpoints to that
router, not to `app`, so they are gated by default - see src/auth.py for why those two
are exempt.

The pipeline seam is `main.run_query` (never the CLI `run()`, which closes the Qdrant
client); the ingest seam is `ingest.run_ingest` (never `main()`, same reason). The server
owns the client lifecycle: opened lazily, closed once on shutdown.
"""

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from urllib.parse import urljoin

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.answerer import Confidence, cited_hit
from src.auth import auth_enabled, require_api_key
from src.config import (
    CORS_ALLOW_ORIGINS,
    MAX_UPLOAD_MB,
    PAGE_IMAGES_DIR,
    PDFS_DIR,
    RETRIEVE_K,
    SERVER_HOST,
    SERVER_PORT,
    UI_DIST_DIR,
    validate,
)
from src.embedder import is_loaded, load_model
from src.gpu_arbiter import GpuArbiter, GpuBusy
from src.graph import get_graph
from src.heatmap import page_similarity
from src.ingest import run_ingest
from src.logging_setup import get_logger
from src.main import run_query
from src.pdf_render import crop_images_for, page_image_path, page_images_for
from src.ratelimit import rate_limit, rate_limit_ingest
from src.vector_store import (
    close_client,
    delete_document,
    document_index,
    index_health,
    list_documents,
    ping,
)

log = get_logger("server")


# --- Response / request models (the contract the UI is built against) ---

class QueryRequest(BaseModel):
    """One question to answer; length-bounded so a runaway prompt can't reach Gemini."""

    question: str = Field(min_length=1, max_length=2000)


class ImageRef(BaseModel):
    """One image, delivered as a static URL and/or an inline base64 data-URI.

    `url` is always set; `data_uri` is populated only when the request asks for it
    (`?inline=true`), so the same shape serves a normal web UI and a sandboxed one.
    """

    url: str | None = None
    data_uri: str | None = None


class PageHit(BaseModel):
    """One retrieved page: its 1-based rank, source, MaxSim score, and page image."""

    index: int          # 1-based; matches citation.source_page
    pdf: str
    page_number: int
    score: float
    image: ImageRef


class RegionOut(BaseModel):
    """One cited region: its box on a retrieved page, plus that region's own crop."""

    source_page: int    # 1-based index into pages[]
    box: list[int]      # [ymin, xmin, ymax, xmax] on a 0-1000 scale
    pdf: str | None = None          # enriched from pages[source_page-1]
    page_number: int | None = None
    crop: ImageRef | None = None


class CitationOut(BaseModel):
    """Where the answer was read: the primary region plus every cited region."""

    found: bool
    source_page: int    # 1-based index into pages[]; 0 when not found (primary region)
    box: list[int]      # [ymin, xmin, ymax, xmax] on a 0-1000 scale; [] when not found
    pdf: str | None = None          # enriched from pages[source_page-1]
    page_number: int | None = None
    confidence: Confidence = "low"  # the model's self-reported answer confidence
    regions: list[RegionOut] = []   # every cited region (primary first); [] when not found


class StageMeta(BaseModel):
    """One pipeline node's latency and Gemini spend (retrieve/rerank/answer/highlight)."""

    node: str
    latency_ms: float
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    est_cost_usd: float = 0.0
    gemini_calls: int = 0


class QueryMeta(BaseModel):
    """Per-request observability: ids, latency, token/cost totals, and a stage breakdown."""

    request_id: str
    latency_ms: float
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    est_cost_usd: float = 0.0
    gemini_calls: int = 0
    retrieve_k: int = 0             # configured candidate count (for "retrieved N" display)
    # Deterministic retrieval-decisiveness (softmax share of the cited page over the
    # candidate MaxSim scores). None when nothing was cited. See src/confidence.py.
    retrieval_confidence: float | None = None
    stages: list[StageMeta] = []


class QueryResponse(BaseModel):
    """The full /query contract: answer, visual citation, candidates, and meta."""

    question: str
    answer: str
    citation: CitationOut
    pages: list[PageHit]
    crop: ImageRef | None = None        # null when citation.found is false
    annotated: ImageRef | None = None
    meta: QueryMeta


class DocumentInfo(BaseModel):
    """One indexed document and how many pages of it are in the index."""

    pdf: str
    page_count: int


class CorpusResponse(BaseModel):
    """The corpus rail's view: indexed documents, total pages, Qdrant status."""

    documents: list[DocumentInfo]
    total_pages: int
    qdrant: str


class DocumentPage(BaseModel):
    """One page of an indexed document: its 1-based number and its rendered page image.

    `image` is null when the page is in the index but its PNG is not on disk. The page is
    still listed, and that is two invariants at once. It keeps the vectors-without-page-
    images split visible - the same reason `index_health` reports it rather than leaving
    it to be inferred - and it keeps `pages[n-1]` being page n. Filtering would renumber
    the list, so the viewer would draw page 5's citation box over page 6's image.
    """

    page_number: int                # 1-based; matches PageHit.page_number
    image: ImageRef | None = None   # null when this page's PNG is missing from disk


class DocumentPagesResponse(BaseModel):
    """Every page of one indexed document - the full-screen viewer's manifest.

    `page_count` is the *indexed* count, read from the Qdrant payloads, so it is always
    len(pages) even when some of those pages have no image, and even when a leftover PNG
    from a longer previous revision is sitting in PAGE_IMAGES_DIR (`ingest._sync` deletes
    a changed document's vectors but not its old page images - see
    `pdf_render.missing_page_numbers` for why that case is real).

    `has_original` is whether the source PDF is still on disk. Indexed-with-no-source is a
    genuine state - DELETE removes the PDF, and a restore can bring back the vectors and
    the page images without pdfs/ - so the UI disables the download rather than offering a
    link that 404s. Not `has_source`: `source_page` already means "1-based index into
    pages[]" in RegionOut/CitationOut, and reusing the word here would read as that.
    """

    pdf: str
    page_count: int
    has_original: bool
    pages: list[DocumentPage]


class HealthResponse(BaseModel):
    """Liveness: whether the model is warm, Qdrant is reachable, and the corpus is whole.

    `corpus` reports the vectors-without-page-images split (see `vector_store.index_health`),
    which is otherwise invisible: the corpus still lists, and every query just answers
    "not found". It does not affect `status` - the server is serving, some documents
    just cannot be read from - so an orchestrator's liveness probe is unchanged.
    """

    status: str
    model_loaded: bool
    qdrant: str
    corpus: str = "ok"


class IngestResponse(BaseModel):
    """Result of an ingest: the document and how many pages were embedded (0 if unchanged)."""

    pdf: str
    indexed_pages: int


class DeleteResponse(BaseModel):
    """Result of a delete: the document and how many indexed pages were removed."""

    pdf: str
    removed_pages: int


class HeatmapRequest(BaseModel):
    """A question plus the specific indexed page to compute patch similarities for."""

    question: str = Field(min_length=1, max_length=2000)
    pdf: str = Field(min_length=1)
    page_number: int = Field(ge=1)


class HeatmapResponse(BaseModel):
    """A per-patch MaxSim heatmap for one page: `grid[y][x]` in [0, 1] over an n_x x n_y
    patch grid (the query's match strength at each ColQwen2 patch). Small enough to send
    as JSON; the UI paints it onto a canvas stretched over the page image."""

    pdf: str
    page_number: int
    n_x: int
    n_y: int
    grid: list[list[float]]


# --- Lifespan warmup: pay the cold start once, at boot ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the model + Qdrant + graph once at startup; fail fast if any step fails.

    A server that can't answer is worse than an obvious boot failure, so a bad key,
    an unloadable model, or an unreachable Qdrant propagates and uvicorn refuses to
    start. `/health`'s degraded path is for transient blips after a good boot.
    """
    validate()                                   # bad/empty GEMINI_API_KEY -> raise, abort boot
    load_model()                                 # pay the ~2B cold start here, once
    ping()                                        # lazily opens + verifies the Qdrant client
    get_graph()                                  # compile the LangGraph once
    # Surface the auth posture at boot: an operator should never have to guess whether
    # the deployment they just started is open to the world.
    if not auth_enabled():
        log.warning("API_KEY is unset - every endpoint is open, including DELETE /corpus")
    # Same reasoning for the corpus. A restore that brought back the Qdrant volume but
    # not PAGE_IMAGES_DIR boots perfectly and then answers "not found" to everything, so
    # it is stated here rather than left to be inferred from a WARNING per dropped hit.
    # ERROR, not a refusal to boot: config errors fail fast, recoverable data states
    # degrade - and this one is repaired by re-running ingest, which needs a live server.
    #
    # Belt and braces on top of `index_health`'s own guarantee never to raise. A
    # *diagnostic* must not be able to take down the thing it diagnoses, and this one
    # already did once: before the guard, a fresh install (no collection until the first
    # ingest) raised here and uvicorn refused to start.
    try:
        corpus = index_health()
    except Exception:
        log.warning("corpus health check failed at boot - starting anyway", exc_info=True)
        corpus = {"ok": True, "checked": 0, "incomplete": []}
    if not corpus["ok"]:
        log.error("indexed documents are missing page images - re-run ingest to repair",
                  extra={"incomplete": corpus["incomplete"], "checked": corpus["checked"]})
    log.info("server warm", extra={"model_loaded": is_loaded(), "auth": auth_enabled(),
                                   "corpus_ok": corpus["ok"], "corpus_checked": corpus["checked"]})
    yield
    close_client()                               # the one place the server closes Qdrant


app = FastAPI(title="Vision RAG API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
PAGE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=PAGE_IMAGES_DIR), name="images")

# Every real endpoint hangs off this router, so auth + rate limiting are structural
# rather than per-route opt-ins. `/health` stays on `app` (orchestrators probe it
# without the secret) and so does the `/images` mount (`<img src>` can't send headers).
#
# rate_limit is listed FIRST deliberately. FastAPI solves dependencies in order, so
# auth-first would let a caller with no key burn unlimited 401s - the throttle would
# only ever apply to requests that had already passed the gate, leaving key guessing
# and unauthenticated floods unbounded. Limiting first throttles both.
api = APIRouter(dependencies=[Depends(rate_limit), Depends(require_api_key)])

# Serializes the single GPU-resident model across requests. Not a bare asyncio.Lock:
# it also bounds the wait (503 instead of a hung socket), refuses to release under a
# forward pass whose client has disconnected, and lets a long ingest hand the model
# back between page batches. See src/gpu_arbiter.py.
_gpu = GpuArbiter()


# --- Image helpers ---

def _to_url(request: Request, fs_path: str) -> str:
    """Map a filesystem path under page_images/ to its /images/... URL."""
    rel = Path(fs_path).resolve().relative_to(PAGE_IMAGES_DIR.resolve())
    return urljoin(str(request.base_url), f"images/{rel.as_posix()}")


async def _encode_data_uri(fs_path: str) -> str:
    """Read a PNG off disk and encode it as a base64 data-URI."""
    data = await asyncio.to_thread(Path(fs_path).read_bytes)
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


async def _image_ref(request: Request, fs_path: str | None, inline: bool) -> ImageRef | None:
    """An ImageRef for a stored image, or None when the path is missing/empty."""
    if not fs_path:
        return None
    return ImageRef(
        url=_to_url(request, fs_path),
        data_uri=await _encode_data_uri(fs_path) if inline else None,
    )


_CONFIDENCE_LEVELS: tuple[Confidence, ...] = ("high", "medium", "low")


def _coerce_confidence(value: object) -> Confidence:
    """Narrow an untyped citation confidence to the Confidence literal; unknown -> "medium".

    `citation` is a plain dict here, so its `confidence` is statically `str | Any`. Coercing
    it keeps the typed response model honest (an unexpected value degrades to "medium" rather
    than raising when CitationOut is built) and lets mypy verify the field type.
    """
    return next((level for level in _CONFIDENCE_LEVELS if level == value), "medium")


async def _build_query_response(request: Request, result: dict, inline: bool) -> QueryResponse:
    """Shape a raw `run_query` result dict into the HTTP contract.

    Translates filesystem image paths to URLs/data-URIs and enriches the citation with
    the cited page's pdf/page_number (resolving the 1-based source_page once, here, so
    the UI never re-implements the indexing that bit the CLI).
    """
    retrieved = result.get("retrieved", [])
    page_images = await asyncio.gather(*[
        _image_ref(request, hit.get("image_path"), inline)
        for hit in retrieved
    ])
    pages = [
        PageHit(
            index=i,
            pdf=hit["pdf"],
            page_number=hit["page_number"],
            score=hit["score"],
            # vector_store.search() drops hits whose image_path is missing/stale, so
            # every retrieved page resolves and _image_ref never returns None here
            # (unlike the optional crop/annotated paths).
            image=page_images[i - 1],  # type: ignore[arg-type]
        )
        for i, hit in enumerate(retrieved, start=1)
    ]

    citation = result.get("citation") or {}
    source_page = citation.get("source_page", 0)
    found = bool(citation.get("found"))
    cited = cited_hit(retrieved, source_page)
    # Enforce not-found invariant: "low" confidence when not found, "medium" fallback when found.
    confidence: Confidence = "low" if not found else _coerce_confidence(citation.get("confidence"))

    # Every validated region highlight produced, each with its own crop image. The list
    # is authoritative (already validated + cropped upstream), so no re-indexing here.
    cited_regions = result.get("cited_regions", [])
    region_crops = await asyncio.gather(*[
        _image_ref(request, r.get("crop_path"), inline) for r in cited_regions
    ])
    regions_out = []
    for r, crop_ref in zip(cited_regions, region_crops):
        hit = cited_hit(retrieved, r["source_page"])
        regions_out.append(
            RegionOut(
                source_page=r["source_page"],
                box=r["box"],
                pdf=hit["pdf"] if hit else None,
                page_number=hit["page_number"] if hit else None,
                crop=crop_ref,
            )
        )

    citation_out = CitationOut(
        found=found,
        source_page=source_page,
        box=citation.get("box") or [],
        pdf=cited["pdf"] if cited else None,
        page_number=cited["page_number"] if cited else None,
        confidence=confidence,
        regions=regions_out,
    )

    crop, annotated = await asyncio.gather(
        _image_ref(request, result.get("crop_path"), inline),
        _image_ref(request, result.get("annotated_path"), inline),
    )

    return QueryResponse(
        question=result.get("question", ""),
        answer=result.get("answer", ""),
        citation=citation_out,
        pages=pages,
        crop=crop,
        annotated=annotated,
        meta=QueryMeta(**{**result.get("meta", {}), "retrieve_k": RETRIEVE_K}),
    )


# --- Endpoints ---

def _corpus_status() -> str:
    """`index_health` as a one-line status string for /health (never raises).

    The failure branch logs the exception and returns a bare "unknown" rather than the
    exception text: /health is on `app`, not the gated `api` router, so its body is world
    -readable and a raw exception can carry internal hostnames or filesystem paths.
    (The `qdrant` field above still interpolates its exception - pre-existing behaviour a
    test asserts on, so changing it belongs in its own change, not this one.)
    """
    try:
        health = index_health()
    except Exception:
        log.warning("corpus health check failed", exc_info=True)
        return "unknown"
    if health["ok"]:
        return "ok"
    count = len(health["incomplete"])
    return (f"{count} document(s) missing page images "
            f"({', '.join(d['pdf'] for d in health['incomplete'][:5])}) - re-run ingest")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Model-loaded flag, Qdrant reachability, corpus integrity. 503 when Qdrant is down."""
    try:
        await asyncio.to_thread(ping)
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "model_loaded": is_loaded(),
                     "qdrant": f"unreachable: {exc}", "corpus": "unknown"},
        )
    return HealthResponse(
        status="ok",
        model_loaded=is_loaded(),
        qdrant="ok",
        corpus=await asyncio.to_thread(_corpus_status),
    )


@api.get("/corpus", response_model=CorpusResponse)
async def corpus():
    """Indexed documents + page counts, for the corpus rail. 503 if Qdrant is down."""
    try:
        docs = await asyncio.to_thread(list_documents)
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"documents": [], "total_pages": 0, "qdrant": f"unreachable: {exc}"},
        )
    documents = [DocumentInfo(**d) for d in docs]
    return CorpusResponse(
        documents=documents,
        total_pages=sum(d.page_count for d in documents),
        qdrant="ok",
    )


async def _indexed_document(pdf: str) -> tuple[str, dict]:
    """Normalize a `{pdf}` path parameter and resolve its index entry; 404 when unindexed.

    The one guard behind every /corpus/{pdf} route, so the three of them cannot drift
    apart - and drifting apart here means unlinking or serving the wrong file.

    Two layers. A `{pdf}` path parameter does not match `/`, so anything carrying a
    separator 404s in the router before this runs; `Path(pdf).name` normalizes what is
    left. The layer that actually constrains the filesystem is index membership: only a
    document the corpus already owns is addressable at all. That is what keeps DELETE from
    reaching a PDF nobody ingested, and what keeps `GET .../file` from being a read
    primitive over PDFS_DIR - which also holds every upload `_save_upload` wrote before its
    ingest ran, and the fetched eval corpus.
    """
    name = Path(pdf).name
    index = await asyncio.to_thread(document_index)
    entry = index.get(name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"{pdf} is not indexed.")
    return name, entry


def _page_files(name: str, page_count: int) -> tuple[list[Path | None], bool]:
    """One document's page PNGs (None where missing), plus whether its source PDF is there.

    One stat per *page*, deliberately not `pdf_render.page_image_numbers()`. That is the
    bulk form, for callers needing every document at once (`index_health`, `ingest._sync`);
    it costs a stat per page image in the whole corpus, because `Path.iterdir` discards the
    DirEntry and its `is_file()` then re-stats each one - 363 stats today to answer a
    15-page question, and the ratio only worsens as the corpus grows. Both routes resolve
    names through `page_image_path`, so they can never disagree about what "present" means.

    Blocking on purpose - PAGE_IMAGES_DIR may be a mounted disk (see `config._dir_from_env`)
    - so callers hand it to a thread.
    """
    paths = [page_image_path(name, n) for n in range(1, page_count + 1)]
    return [p if p.is_file() else None for p in paths], (PDFS_DIR / name).is_file()


@api.get("/corpus/{pdf}/pages", response_model=DocumentPagesResponse)
async def corpus_document_pages(pdf: str, request: Request):
    """Every page of one indexed document, for the full-screen viewer. 404 when it isn't
    indexed.

    Takes no GPU lock and touches no model - a Qdrant scroll plus one stat per page - so
    opening the viewer stays instant while a query or an ingest holds the GPU. Putting it
    behind the arbiter would let a page listing be shed with 503 for want of a GPU it never
    uses, and would queue it FIFO *ahead* of real model work.

    Citation-agnostic by design: the box the viewer overlays already came back on /query's
    response, so this stays a stateless per-document manifest rather than growing a
    `?highlight=` parameter.

    No `?inline=` data-URI mode, unlike /query. Inlining would read and base64-encode every
    page of the document (~1.33x the bytes) to render one, and would throw away the
    conditional responses the /images StaticFiles mount already gives page images for free.
    The ImageRef shape leaves room to add per-page inlining if a sandboxed UI ever needs it.
    """
    name, entry = await _indexed_document(pdf)
    page_count = entry["page_count"]
    paths, has_original = await asyncio.to_thread(_page_files, name, page_count)
    images = [
        await _image_ref(request, str(path) if path else None, False) for path in paths
    ]
    return DocumentPagesResponse(
        pdf=name,
        page_count=page_count,
        has_original=has_original,
        pages=[DocumentPage(page_number=n, image=image)
               for n, image in enumerate(images, start=1)],
    )


@api.get("/corpus/{pdf}/file", response_class=FileResponse)
async def corpus_document_file(pdf: str):
    """Download one indexed document's original PDF. 404 when it isn't indexed, and a
    distinct 404 when the index has it but the file is gone.

    **Gated like everything else on this router, and the UI pays for it.** An
    `<a href download>` cannot send X-API-Key, so the UI fetches this, wraps the body in an
    object URL and clicks a synthetic link. The /images exemption is deliberately not
    extended here: that hole is scoped to derived page rasters (src/auth.py says so), while
    this hands over whole source documents from a directory nothing has ever served over
    HTTP. If a header-free link is ever needed, it is a short-lived signed token on this
    route - not a third permanent exemption in a design whose gating is structural.

    The indexed-but-no-file case is named rather than served as an empty 200 or a 500. It is
    the same half-corpus class `index_health` reports for page images, and it is the state
    `has_original` on /pages exists to keep out of the UI in the first place.

    No GPU lock, and no to_thread around the response: FileResponse stats and streams the
    file on anyio's own thread pool after this handler returns. Its `stat_result` is
    deliberately left unset - re-stating costs one syscall and buys the case where the file
    vanishes in between, which then surfaces as a clean 503 through the RuntimeError handler
    instead of a truncated body after the headers have gone out.
    """
    name, _ = await _indexed_document(pdf)
    path = PDFS_DIR / name
    if not await asyncio.to_thread(path.is_file):
        raise HTTPException(
            status_code=404,
            detail=f"{name} is indexed but its source PDF is not on disk.",
        )
    return FileResponse(path, media_type="application/pdf", filename=name)


def _remove_document(name: str) -> int:
    """Drop one document's vectors, then its page/crop PNGs, then its source PDF.

    Points go first so the window in which a query can retrieve a page whose image is
    already unlinked stays as small as possible - and even inside it `search()` drops
    hits whose `image_path` is gone from disk, so a racing query degrades rather than
    breaks. File removal is best-effort: a leftover PNG is cosmetic, and failing the
    request after the vectors are gone would be a worse lie than a warning.
    """
    removed = delete_document(name)
    for path in [*page_images_for(name), *crop_images_for(name), PDFS_DIR / name]:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            log.warning("failed to remove file for deleted document",
                        extra={"pdf": name, "path": str(path)}, exc_info=True)
    return removed


@api.delete("/corpus/{pdf}", response_model=DeleteResponse)
async def delete_corpus_document(pdf: str):
    """Remove one document from the corpus entirely: vectors, page images, crops, and
    the stored PDF. 404 when it isn't indexed.

    Takes no GPU lock - deletion is pure Qdrant + filesystem work, so it stays responsive
    while a query or ingest is running.

    The two guards that make it safe to unlink files named by a URL path parameter live in
    `_indexed_document`, shared with the two read routes above.
    """
    name, _ = await _indexed_document(pdf)
    removed = await asyncio.to_thread(_remove_document, name)
    return DeleteResponse(pdf=name, removed_pages=removed)


@api.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request, inline: bool = Query(default=False)):
    """Answer one question. Serializes on the model; base64 reads happen outside the lock.

    503 (with `Retry-After`) rather than a hung request when the model stays busy past
    `GPU_WAIT_TIMEOUT_S` - though a concurrent ingest yields between page batches, so the
    normal wait behind one is a page, not a document.
    """
    result = await _gpu.run_exclusive(run_query, req.question)
    return await _build_query_response(request, result, inline)


@api.post("/heatmap", response_model=HeatmapResponse)
async def heatmap(req: HeatmapRequest):
    """Per-patch MaxSim heatmap for one page - which patches the query lit up ("why this
    page?"). On-demand (the UI's toggle), not folded into /query, because it costs two
    extra model forward passes; stateless - the page is named explicitly by (pdf, page).
    Serializes on the same arbiter as /query so the one GPU model runs one pass at a time."""
    image_path = page_image_path(req.pdf, req.page_number)
    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No indexed page image for {req.pdf} p.{req.page_number}.",
        )
    grid, n_x, n_y = await _gpu.run_exclusive(page_similarity, req.question, image_path)
    return HeatmapResponse(
        pdf=req.pdf, page_number=req.page_number, n_x=n_x, n_y=n_y, grid=grid
    )


async def _save_upload(file: UploadFile) -> tuple[str, list[Path]]:
    """Validate + persist an uploaded PDF under PDFS_DIR; return (name, [saved path]).

    Shared by the blocking and streaming ingest endpoints so the size/type checks and
    the save live in exactly one place. Only the uploaded file is handed to the ingest -
    this used to pass every PDF in PDFS_DIR, which made each upload re-embed the entire
    corpus through the ~2B model.
    """
    name = Path(file.filename or "").name          # strip any path components
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf uploads are accepted.")
    cap = MAX_UPLOAD_MB * 1024 * 1024
    if file.size is not None and file.size > cap:
        raise HTTPException(status_code=413, detail=f"PDF exceeds the {MAX_UPLOAD_MB} MB limit.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(data) > cap:
        raise HTTPException(status_code=413, detail=f"PDF exceeds the {MAX_UPLOAD_MB} MB limit.")

    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    dest = PDFS_DIR / name
    await asyncio.to_thread(dest.write_bytes, data)
    return name, [dest]


def _sse(event: dict) -> str:
    """Format one dict as a Server-Sent Events `data:` frame."""
    return f"data: {json.dumps(event)}\n\n"


@api.post("/ingest", response_model=IngestResponse, dependencies=[Depends(rate_limit_ingest)])
async def ingest(file: UploadFile = File(...)):
    """Upload and index a single PDF. Blocking and long-running - the response does not
    come back until the whole render/embed/upsert build has finished.

    It no longer *monopolizes* the model for that whole time, though: the gate hands the
    GPU to any queued query at each page boundary, so a concurrent /query waits about a
    page rather than the rest of the document.

    Carries the hourly ingest limit on top of the router's per-minute one, because a
    single upload costs minutes of GPU time rather than one forward pass."""
    name, all_pdfs = await _save_upload(file)
    gate = _gpu.thread_gate(asyncio.get_running_loop())
    indexed = await _gpu.run_exclusive(partial(run_ingest, all_pdfs, gate=gate))
    return IngestResponse(pdf=name, indexed_pages=indexed)


@api.post("/ingest/stream", dependencies=[Depends(rate_limit_ingest)])
async def ingest_stream(file: UploadFile = File(...)):
    """Same as /ingest, but streams per-page progress as Server-Sent Events.

    The render/embed loop runs in a worker thread (holding the model the whole time,
    exactly like /ingest, and yielding it at the same page boundaries); its progress
    callback hops each event back onto the loop via an asyncio.Queue, and an async
    generator drains the queue into SSE frames. A done-callback pushes a sentinel so the
    generator knows the build finished, then re-raises any build failure as a final
    `error` event.
    """
    name, all_pdfs = await _save_upload(file)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    done = object()
    gate = _gpu.thread_gate(loop)

    def progress(event: dict) -> None:
        """Hop one worker-thread progress event back onto the event loop's queue."""
        loop.call_soon_threadsafe(queue.put_nowait, event)   # called from the worker thread

    # Shedding has to be decided BEFORE the response starts and the lock must never be
    # held across the endpoint/generator boundary. Those pull in opposite directions, and
    # getting either wrong is worse than the problem:
    #
    #   * Acquiring inside event_stream() puts GpuBusy permanently out of reach of
    #     _gpu_busy_handler - FastAPI sends a StreamingResponse's 200 headers before it
    #     ever iterates the generator - so an overloaded server answers a truncated 200
    #     with no Retry-After while plain /ingest correctly sheds.
    #   * Acquiring here and closing the context *inside* the generator leaks the lock
    #     outright. If the connection breaks before the headers go out, Starlette raises
    #     from send() and never starts the body at all, so its `finally` never runs -
    #     measured, not assumed - and the model stays locked for the life of the process.
    #     Every later request then sheds forever. A wedged server is far worse than a
    #     truncated stream.
    #
    # So: probe at the door with the real timeout to make the shed decision, release it,
    # and let the generator take the lock for the actual build. Acquire and release then
    # sit in the same frame and cannot leak. The probe is advisory - another request can
    # slip in between - which is why the generator's own acquire still degrades to a
    # terminal `error` frame rather than dying silently. That race costs a queued client
    # a wait; the alternatives cost correctness.
    async with _gpu.acquire():                               # raises GpuBusy -> 503 here
        pass

    async def event_stream():
        """Drain the progress queue into SSE frames until the build finishes or fails."""
        try:
            async with _gpu.acquire():                       # holds the model for the build
                task = asyncio.create_task(
                    asyncio.to_thread(partial(run_ingest, all_pdfs, progress, gate=gate))
                )
                task.add_done_callback(
                    lambda _t: loop.call_soon_threadsafe(queue.put_nowait, done))
                try:
                    while True:
                        event = await queue.get()
                        if event is done:
                            break
                        yield _sse(event)
                    indexed = task.result()                  # task is finished here; re-raises on failure
                    yield _sse({"phase": "done", "pdf": name, "indexed_pages": indexed})
                except Exception as exc:
                    log.warning("ingest stream failed", exc_info=exc)
                    yield _sse({"phase": "error", "detail": str(exc)})
                finally:
                    # If the client disconnected mid-build, don't release the model lock
                    # until the in-flight embed loop actually finishes (a concurrent query
                    # forward pass on the one GPU model would otherwise race it).
                    if not task.done():
                        await asyncio.shield(task)
        except GpuBusy as exc:
            # Lost the race after the door probe. The headers are already out, so this
            # cannot be a 503 - say so in the stream instead of truncating it.
            log.warning("ingest stream shed after the response started")
            yield _sse({"phase": "error", "detail": str(exc),
                        "retry_after": exc.retry_after})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.exception_handler(RuntimeError)
async def _runtime_error_handler(request: Request, exc: RuntimeError):
    """Surface our actionable RuntimeErrors (e.g. Qdrant unreachable) as 503, not 500."""
    log.warning("request failed", exc_info=exc, extra={"path": request.url.path})
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(GpuBusy)
async def _gpu_busy_handler(request: Request, exc: GpuBusy):
    """Shed a request that waited too long for the model: 503 + Retry-After.

    Deliberately shaped like ratelimit's 429 - a client that is told to come back can
    back off, where one left hanging on the socket can only time out and guess."""
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc)},
        headers={"Retry-After": str(exc.retry_after)},
    )


# --- The UI, served from the same origin as the API ---

# Registered here, after every route function above has been attached to it.
app.include_router(api)

# The built Vite bundle (see the Dockerfile's node stage), served alongside the API so
# a deployment is one container on one origin and CORS never enters the picture.
#
# Deliberately NOT `mount("/", StaticFiles(html=True))`. A mount at "/" matches every
# path *before* the method is considered, so it turns a DELETE to an unrouted path into
# StaticFiles' 405 instead of the router's 404 - which is exactly what the path-traversal
# guard on DELETE /corpus/{pdf} relies on. Mounting only /assets and serving index.html
# from an explicit GET shadows nothing.
#
# This works because the UI has no client-side router: "/" is the only HTML entry point,
# so there are no deep links needing an index.html fallback. If react-router is ever
# added, that fallback becomes a catch-all `@app.get("/{path:path}")` - a GET-only route,
# so unrouted non-GET requests still 404 correctly.
#
# Guarded on existence: a dev checkout has no ui/dist until `npm run build` runs, and
# StaticFiles raises on a missing directory - the API must not refuse to boot just
# because the frontend was never built.
#
# A function rather than a bare `if` so both branches stay testable: ui/dist is
# gitignored, so CI only ever sees the not-built case, and a test can drive either by
# pointing UI_DIST_DIR at a tmp_path and mounting onto a throwaway app.
def _mount_ui(target: FastAPI) -> bool:
    """Attach the built UI to `target`; return whether ui/dist was there to attach."""
    index = UI_DIST_DIR / "index.html"
    assets = UI_DIST_DIR / "assets"
    # index.html, not the directory, is the signal that a build actually completed -
    # an empty ui/dist is left behind by an interrupted one.
    if not index.is_file() or not assets.is_dir():
        log.info("ui/dist not built - serving the API only", extra={"path": str(UI_DIST_DIR)})
        return False

    target.mount("/assets", StaticFiles(directory=assets), name="ui-assets")

    @target.get("/", include_in_schema=False)
    async def ui_index():
        """Serve the built UI's entry point."""
        return FileResponse(index)

    return True


_mount_ui(app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.server:app", host=SERVER_HOST, port=SERVER_PORT)
