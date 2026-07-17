"""Qdrant multivector store - one multivector per page, ranked by MaxSim"""

from pathlib import Path
from qdrant_client import QdrantClient 
from qdrant_client import models as qm 

from src.config import (
    COLLECTION_NAME,
    QDRANT_API_KEY,
    QDRANT_PATH,
    QDRANT_URL,
    RETRIEVE_K,
    VECTOR_DIM,
)

_client: QdrantClient | None = None

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

def ensure_collection(reset: bool = False) -> None:
    """Create the multivector if it is missing."""
    client = get_client()
    if reset and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
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

def build_point(point_id, multivector, pdf_name, page_number, image_path) -> qm.PointStruct:
    """Build one page's point: its multivector plus source metadata."""
    return qm.PointStruct(
        id = point_id,
        vector = multivector,
        payload = {"pdf": pdf_name, "page_number": page_number, "image_path": str(image_path)},
    )

def upsert_pages(points: list[qm.PointStruct]) -> None:
    """Store a batch of page points in a single round-trip (skips empty batches)."""
    if not points:
        return
    client = get_client()
    client.upsert(collection_name=COLLECTION_NAME, points=points)

def search(query_multivector: list[list[float]], top_k: int = RETRIEVE_K) -> list[dict]:
    """Return the top_k pages for a query multivector, best score first."""
    client = get_client()
    response = client.query_points(
        collection_name = COLLECTION_NAME,
        query = query_multivector,
        limit = top_k,
        with_payload = True,
        # Binary quantization is lossy: pull extra candidates on the fast quantized
        # pass, then rescore them against the full-precision vectors to keep recall.
        search_params = qm.SearchParams(
            quantization = qm.QuantizationSearchParams(rescore=True, oversampling=2.0),
        ),
    )
    return [{**p.payload, "score": round(p.score, 4)} for p in response.points]


