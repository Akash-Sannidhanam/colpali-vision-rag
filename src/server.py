"""Warm FastAPI serving layer for the vision RAG pipeline.

A single-worker HTTP surface over the LangGraph pipeline. The ~2B ColQwen2 model is
loaded once at boot (`lifespan`), so the cold start is paid at startup, not per query.
All heavy work runs in a threadpool behind an `asyncio.Lock`, so the one GPU-resident
model is never asked to run two forward passes at once and `/health` stays responsive.

Endpoints:
  POST /query   {question}          -> answer + visual citation + used pages + meta
  POST /heatmap {question,pdf,page_number} -> per-patch MaxSim heatmap grid for one page (on-demand)
  GET  /health                      -> model-loaded flag + Qdrant reachability (503 if down)
  GET  /corpus                      -> indexed documents + page counts (for the UI rail)
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
from pathlib import Path
from urllib.parse import urljoin

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.answerer import Confidence
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
from src.graph import get_graph
from src.heatmap import page_similarity
from src.ingest import run_ingest
from src.logging_setup import get_logger
from src.main import run_query
from src.pdf_render import crop_images_for, page_image_path, page_images_for
from src.ratelimit import rate_limit, rate_limit_ingest
from src.vector_store import close_client, delete_document, document_index, list_documents, ping

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


class HealthResponse(BaseModel):
    """Liveness: whether the model is warm and whether Qdrant is reachable."""

    status: str
    model_loaded: bool
    qdrant: str


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
    log.info("server warm", extra={"model_loaded": is_loaded(), "auth": auth_enabled()})
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

_gpu_lock = asyncio.Lock()     # serialize the single GPU-resident model across requests


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
    cited = retrieved[source_page - 1] if 1 <= source_page <= len(retrieved) else None
    # Enforce not-found invariant: "low" confidence when not found, "medium" fallback when found.
    confidence: Confidence = "low" if not found else _coerce_confidence(citation.get("confidence"))

    # Every validated region highlight produced, each with its own crop image. The list
    # is authoritative (already validated + cropped upstream), so no re-indexing here.
    cited_regions = result.get("cited_regions", [])
    region_crops = await asyncio.gather(*[
        _image_ref(request, r.get("crop_path"), inline) for r in cited_regions
    ])
    regions_out = [
        RegionOut(
            source_page=r["source_page"],
            box=r["box"],
            pdf=retrieved[r["source_page"] - 1]["pdf"]
            if 1 <= r["source_page"] <= len(retrieved) else None,
            page_number=retrieved[r["source_page"] - 1]["page_number"]
            if 1 <= r["source_page"] <= len(retrieved) else None,
            crop=crop_ref,
        )
        for r, crop_ref in zip(cited_regions, region_crops)
    ]

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

@app.get("/health", response_model=HealthResponse)
async def health():
    """Model-loaded flag + Qdrant reachability. 503 (degraded) when Qdrant is down."""
    try:
        await asyncio.to_thread(ping)
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "model_loaded": is_loaded(),
                     "qdrant": f"unreachable: {exc}"},
        )
    return HealthResponse(status="ok", model_loaded=is_loaded(), qdrant="ok")


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

    Two guards, because this unlinks files named by a URL path parameter. A `{pdf}` path
    parameter does not match `/`, so anything carrying a separator 404s in the router
    before this runs; `Path(pdf).name` then normalizes what is left. The one that actually
    constrains the filesystem is the index-membership check: the normalized name must
    already be indexed, so only a document the corpus owns can ever be deleted.
    """
    name = Path(pdf).name
    index = await asyncio.to_thread(document_index)
    if name not in index:
        raise HTTPException(status_code=404, detail=f"{pdf} is not indexed.")

    removed = await asyncio.to_thread(_remove_document, name)
    return DeleteResponse(pdf=name, removed_pages=removed)


@api.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request, inline: bool = Query(default=False)):
    """Answer one question. Serializes on the model lock; base64 reads happen outside it."""
    async with _gpu_lock:
        result = await asyncio.to_thread(run_query, req.question)
    return await _build_query_response(request, result, inline)


@api.post("/heatmap", response_model=HeatmapResponse)
async def heatmap(req: HeatmapRequest):
    """Per-patch MaxSim heatmap for one page - which patches the query lit up ("why this
    page?"). On-demand (the UI's toggle), not folded into /query, because it costs two
    extra model forward passes; stateless - the page is named explicitly by (pdf, page).
    Serializes on the same model lock as /query so the one GPU model runs one pass at a time."""
    image_path = page_image_path(req.pdf, req.page_number)
    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No indexed page image for {req.pdf} p.{req.page_number}.",
        )
    async with _gpu_lock:
        grid, n_x, n_y = await asyncio.to_thread(page_similarity, req.question, image_path)
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
    """Upload and index a single PDF. Blocking and long-running - it holds the model
    lock for the whole render/embed/upsert build, so queries wait while it runs.

    Carries the hourly ingest limit on top of the router's per-minute one, because a
    single upload costs minutes of exclusive GPU time rather than one forward pass."""
    name, all_pdfs = await _save_upload(file)
    async with _gpu_lock:
        indexed = await asyncio.to_thread(run_ingest, all_pdfs)
    return IngestResponse(pdf=name, indexed_pages=indexed)


@api.post("/ingest/stream", dependencies=[Depends(rate_limit_ingest)])
async def ingest_stream(file: UploadFile = File(...)):
    """Same as /ingest, but streams per-page progress as Server-Sent Events.

    The render/embed loop runs in a worker thread (holding the model lock the whole
    time, exactly like /ingest); its progress callback hops each event back onto the
    loop via an asyncio.Queue, and an async generator drains the queue into SSE frames.
    A done-callback pushes a sentinel so the generator knows the build finished, then
    re-raises any build failure as a final `error` event.
    """
    name, all_pdfs = await _save_upload(file)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    done = object()

    def progress(event: dict) -> None:
        """Hop one worker-thread progress event back onto the event loop's queue."""
        loop.call_soon_threadsafe(queue.put_nowait, event)   # called from the worker thread

    async def event_stream():
        """Drain the progress queue into SSE frames until the build finishes or fails."""
        async with _gpu_lock:                                # hold the model lock for the build
            task = asyncio.create_task(asyncio.to_thread(run_ingest, all_pdfs, progress))
            task.add_done_callback(lambda _t: loop.call_soon_threadsafe(queue.put_nowait, done))
            try:
                while True:
                    event = await queue.get()
                    if event is done:
                        break
                    yield _sse(event)
                indexed = task.result()                      # task is finished here; re-raises on failure
                yield _sse({"phase": "done", "pdf": name, "indexed_pages": indexed})
            except Exception as exc:
                log.warning("ingest stream failed", exc_info=exc)
                yield _sse({"phase": "error", "detail": str(exc)})
            finally:
                # If the client disconnected mid-build, don't release the model lock until
                # the in-flight embed loop actually finishes (a concurrent query forward
                # pass on the one GPU model would otherwise race it).
                if not task.done():
                    await asyncio.shield(task)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.exception_handler(RuntimeError)
async def _runtime_error_handler(request: Request, exc: RuntimeError):
    """Surface our actionable RuntimeErrors (e.g. Qdrant unreachable) as 503, not 500."""
    log.warning("request failed", exc_info=exc, extra={"path": request.url.path})
    return JSONResponse(status_code=503, content={"detail": str(exc)})


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
