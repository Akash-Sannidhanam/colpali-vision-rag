"""Warm FastAPI serving layer for the vision RAG pipeline.

A single-worker HTTP surface over the LangGraph pipeline. The ~2B ColQwen2 model is
loaded once at boot (`lifespan`), so the cold start is paid at startup, not per query.
All heavy work runs in a threadpool behind an `asyncio.Lock`, so the one GPU-resident
model is never asked to run two forward passes at once and `/health` stays responsive.

Endpoints:
  POST /query   {question}          -> answer + visual citation + used pages + meta
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
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urljoin

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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
from src.ingest import run_ingest
from src.logging_setup import get_logger
from src.main import run_query
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


class CitationOut(BaseModel):
    found: bool
    source_page: int    # 1-based index into pages[]; 0 when not found
    box: list[int]      # [ymin, xmin, ymax, xmax] on a 0-1000 scale; [] when not found
    pdf: str | None = None          # enriched from pages[source_page-1]
    page_number: int | None = None
    confidence: str = "low"         # the model's self-reported answer confidence


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
    cited = retrieved[source_page - 1] if 1 <= source_page <= len(retrieved) else None
    citation_out = CitationOut(
        found=bool(citation.get("found")),
        source_page=source_page,
        box=citation.get("box") or [],
        pdf=cited["pdf"] if cited else None,
        page_number=cited["page_number"] if cited else None,
        confidence=citation.get("confidence", "low"),
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


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """Upload and index a single PDF. Blocking and long-running - it holds the model
    lock for the whole render/embed/upsert build, so queries wait while it runs."""
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
    all_pdfs = sorted(PDFS_DIR.glob("*.pdf"))
    async with _gpu_lock:
        indexed = await asyncio.to_thread(run_ingest, all_pdfs)
    return IngestResponse(pdf=name, indexed_pages=indexed)


@app.exception_handler(RuntimeError)
async def _runtime_error_handler(request: Request, exc: RuntimeError):
    """Surface our actionable RuntimeErrors (e.g. Qdrant unreachable) as 503, not 500."""
    log.warning("request failed", exc_info=exc, extra={"path": request.url.path})
    return JSONResponse(status_code=503, content={"detail": str(exc)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.server:app", host=SERVER_HOST, port=SERVER_PORT)
