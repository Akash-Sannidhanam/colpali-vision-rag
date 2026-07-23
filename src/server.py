"""Warm FastAPI serving layer for the vision RAG pipeline.

A single-worker HTTP surface over the LangGraph pipeline. The ~2B ColQwen2 model is
loaded once at boot (`lifespan`), so the cold start is paid at startup, not per query.
All heavy work runs in a threadpool behind an `asyncio.Lock`, so the one GPU-resident
model is never asked to run two forward passes at once and `/health` stays responsive.

Endpoints:
  POST /query   {question}          -> answer + visual citation + used pages + meta
  POST /heatmap {question,pdf,page} -> per-patch MaxSim heatmap grid for one page (on-demand)
  GET  /health                      -> model-loaded flag + Qdrant reachability (503 if down)
  GET  /corpus                      -> indexed documents + page counts (for the UI rail)
  POST /ingest  (multipart PDF)     -> render/embed/index a PDF (blocking, holds the lock)
  /images/...                       -> static page/crop/annotated PNGs

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

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.answerer import Confidence
from src.config import (
    CORS_ALLOW_ORIGINS,
    MAX_UPLOAD_MB,
    PAGE_IMAGES_DIR,
    PDFS_DIR,
    RETRIEVE_K,
    SERVER_HOST,
    SERVER_PORT,
    validate,
)
from src.embedder import is_loaded, load_model
from src.graph import get_graph
from src.heatmap import page_similarity
from src.ingest import run_ingest
from src.logging_setup import get_logger
from src.main import run_query
from src.pdf_render import page_image_path
from src.vector_store import close_client, list_documents, ping

log = get_logger("server")


# --- Response / request models (the contract the UI is built against) ---

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class ImageRef(BaseModel):
    """One image, delivered as a static URL and/or an inline base64 data-URI.

    `url` is always set; `data_uri` is populated only when the request asks for it
    (`?inline=true`), so the same shape serves a normal web UI and a sandboxed one.
    """

    url: str | None = None
    data_uri: str | None = None


class PageHit(BaseModel):
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
    found: bool
    source_page: int    # 1-based index into pages[]; 0 when not found (primary region)
    box: list[int]      # [ymin, xmin, ymax, xmax] on a 0-1000 scale; [] when not found
    pdf: str | None = None          # enriched from pages[source_page-1]
    page_number: int | None = None
    confidence: Confidence = "low"  # the model's self-reported answer confidence
    regions: list[RegionOut] = []   # every cited region (primary first); [] when not found


class StageMeta(BaseModel):
    node: str
    latency_ms: float
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    est_cost_usd: float = 0.0
    gemini_calls: int = 0


class QueryMeta(BaseModel):
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
    question: str
    answer: str
    citation: CitationOut
    pages: list[PageHit]
    crop: ImageRef | None = None        # null when citation.found is false
    annotated: ImageRef | None = None
    meta: QueryMeta


class DocumentInfo(BaseModel):
    pdf: str
    page_count: int


class CorpusResponse(BaseModel):
    documents: list[DocumentInfo]
    total_pages: int
    qdrant: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    qdrant: str


class IngestResponse(BaseModel):
    pdf: str
    indexed_pages: int


class HeatmapRequest(BaseModel):
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
    log.info("server warm", extra={"model_loaded": is_loaded()})
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


@app.get("/corpus", response_model=CorpusResponse)
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


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request, inline: bool = Query(default=False)):
    """Answer one question. Serializes on the model lock; base64 reads happen outside it."""
    async with _gpu_lock:
        result = await asyncio.to_thread(run_query, req.question)
    return await _build_query_response(request, result, inline)


@app.post("/heatmap", response_model=HeatmapResponse)
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
    """Validate + persist an uploaded PDF under PDFS_DIR; return (name, all pdfs to index).

    Shared by the blocking and streaming ingest endpoints so the size/type checks and
    the save live in exactly one place.
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
    return name, sorted(PDFS_DIR.glob("*.pdf"))


def _sse(event: dict) -> str:
    """Format one dict as a Server-Sent Events `data:` frame."""
    return f"data: {json.dumps(event)}\n\n"


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """Upload and index a single PDF. Blocking and long-running - it holds the model
    lock for the whole render/embed/upsert build, so queries wait while it runs."""
    name, all_pdfs = await _save_upload(file)
    async with _gpu_lock:
        indexed = await asyncio.to_thread(run_ingest, all_pdfs)
    return IngestResponse(pdf=name, indexed_pages=indexed)


@app.post("/ingest/stream")
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
        loop.call_soon_threadsafe(queue.put_nowait, event)   # called from the worker thread

    async def event_stream():
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.server:app", host=SERVER_HOST, port=SERVER_PORT)
