"""ColPali (ColQwen2 / ColQwen2.5) visual embedder- reads pages by sight."""
from collections.abc import Iterator

import torch
from colpali_engine.models import (
    ColQwen2,
    ColQwen2_5,
    ColQwen2_5_Processor,
    ColQwen2Processor,
)
from PIL import Image

from src.config import COLPALI_MODEL, EMBED_BATCH_SIZE
from src.logging_setup import get_logger

logger = get_logger(__name__)

_model: ColQwen2 | ColQwen2_5 | None = None
_processor: ColQwen2Processor | ColQwen2_5_Processor | None = None

# Substrings that identify an allocation failure across backends: CUDA raises
# torch.OutOfMemoryError (a RuntimeError), MPS a plain RuntimeError, CPU a MemoryError.
# Matched on the message so a genuine non-OOM RuntimeError is re-raised instead of being
# silently retried at a smaller batch, which would hide a real bug as a slow ingest.
_OOM_MARKERS = ("out of memory", "can't allocate", "cannot allocate")

# How far two embeddings of the same page may drift before batching is called untrustworthy.
# See _batching_is_trustworthy: correct backends land at 0 or ~1e-6, the known-bad one at ~0.4.
BATCH_EQUIVALENCE_TOLERANCE = 1e-2

# Cached verdict from _batching_is_trustworthy: None until the first real multi-page batch
# has been checked. `False` permanently pins this process to single-page forward passes.
_batching_verified: bool | None = None


def _model_classes() -> tuple[
    type[ColQwen2 | ColQwen2_5], type[ColQwen2Processor | ColQwen2_5_Processor]
]:
    """Pick the (model, processor) classes for the configured checkpoint.

    colqwen2.5 checkpoints need the ColQwen2_5 loader; colqwen2 uses ColQwen2. Keyed
    off COLPALI_MODEL so a model swap is config-only (see config.COLPALI_MODEL)."""
    name = COLPALI_MODEL.lower()
    if "colqwen2.5" in name or "colqwen2_5" in name:
        return ColQwen2_5, ColQwen2_5_Processor
    return ColQwen2, ColQwen2Processor

def _device_and_dtype() -> tuple[str, torch.dtype]:
    """Pick the best available GPU: CUDA (NVIDIA) or MPS (Apple Silicon) with
    bfloat16, else CPU with float32."""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32

def load_model() -> tuple[ColQwen2 | ColQwen2_5, ColQwen2Processor | ColQwen2_5_Processor]:
    """Load the ColQwen model and processor once, then cache them."""
    global _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor
    
    device, dtype = _device_and_dtype()
    model_cls, processor_cls = _model_classes()
    _model = model_cls.from_pretrained(
        COLPALI_MODEL, torch_dtype=dtype, device_map=device).eval()
    _processor = processor_cls.from_pretrained(COLPALI_MODEL)
    print(f"Loaded {COLPALI_MODEL} on {device} ({dtype})")
    return _model, _processor

def _to_multivector(embedding: torch.Tensor) -> list[list[float]]:
    """Convert one (n_patches, 128) tensor into plain floats for Qdrant."""
    return embedding.to(torch.float32).cpu().numpy().tolist()

def _is_oom(exc: BaseException) -> bool:
    """True when `exc` is an allocation failure rather than a genuine error."""
    return isinstance(exc, MemoryError) or any(m in str(exc).lower() for m in _OOM_MARKERS)

def _release_memory() -> None:
    """Drop the allocator's cached blocks so a retry isn't defeated by fragmentation."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

def _embed_batch(images: list[Image.Image]) -> list[list[list[float]]]:
    """One forward pass over several pages, trimmed back to each page's own patches.

    **The trim is not optional.** `process_images` pads the batch to its longest member
    (`padding="longest"`, `padding_side="left"`) and ColQwen2's forward *zeroes* the pad
    positions but does not remove them - so `out[i]` still carries leading all-zero rows
    for every page shorter than the longest one. Storing those would change retrieval:
    Qdrant's MaxSim takes, per query token, the max dot product over a page's vectors, and
    a zero vector wins that max whenever every real dot is negative. Indexing the
    attention mask drops exactly the padding, leaving the same vectors a batch of one
    produces - which is what lets EMBED_BATCH_SIZE stay out of EMBED_VERSION.
    """
    model, processor = load_model()
    inputs = processor.process_images(images).to(model.device)
    with torch.no_grad():
        out = model(**inputs)  # (batch, padded_patches, 128)
    mask = inputs["attention_mask"].bool()  # (batch, padded_patches)
    return [_to_multivector(out[i][mask[i]]) for i in range(len(images))]

def _vectors_agree(a: list[list[float]], b: list[list[float]]) -> bool:
    """True when two embeddings of the same page match to within backend float noise.

    The gap this separates is not subtle: a correct backend reproduces a batched page
    bit-for-bit or within ~1e-6, while the corruption below rewrites components of a
    unit-norm vector by ~0.4. BATCH_EQUIVALENCE_TOLERANCE sits three orders of magnitude
    above the noise and two below the corruption, so it is not a threshold that needs tuning.
    """
    if len(a) != len(b):
        return False
    return all(abs(x - y) <= BATCH_EQUIVALENCE_TOLERANCE
               for row_a, row_b in zip(a, b) for x, y in zip(row_a, row_b))

def _batching_is_trustworthy(images: list[Image.Image], batched: list[list[list[float]]]) -> bool:
    """Re-embed this batch's first page alone and check the batch agreed with it.

    **Why this exists.** On MPS + bfloat16 (observed on torch 2.11.0 / transformers 5.3.0,
    and independent of `attn_implementation` - eager, sdpa and the default all fail) a
    batched forward pass silently corrupts the FIRST sequence in the batch. The same page
    placed at slot 0 of a batch of two comes back ~0.4 per component away from the same page
    embedded alone, while slots 1..n-1 are bit-identical to it. The same comparison on CPU
    float32, and on MPS float32, is exact - so the arithmetic is fine and the batching code
    is fine; the bf16 kernel is not. Compare pytorch/pytorch#162592 and #163597.

    Nothing about that failure is visible from the outside: no error, no NaN, correct patch
    counts, and an ingest that writes a `content_hash` marking the document current. It would
    poison one page in every EMBED_BATCH_SIZE of the index and no test that stubs the model
    could ever see it.

    **Why it probes with real pages.** A cheap synthetic image cannot stand in: at 28-112 px
    the corruption does not reproduce at all (sequence lengths 15-27 come back clean), while
    a rendered page is ~755 tokens and does. A guard built on a throwaway image would return
    a confident all-clear on exactly the configuration it exists to catch.

    Costs one extra single-page forward pass per process, only when batching is actually
    used, and the verdict is cached - so it is ~1 page per ingest run, not per batch or per
    document. It is a measurement rather than a device blocklist, so a fixed torch turns
    batching back on by itself.
    """
    return _vectors_agree(_embed_batch(images[:1])[0], batched[0])

def iter_embedded(
    images: list[Image.Image], batch_size: int | None = None,
) -> Iterator[tuple[int, list[list[list[float]]]]]:
    """Embed page images a batch at a time, yielding `(start_index, vectors)` per batch.

    The forward pass is 98% of a page's ingest cost (bench/reports/ingest_baseline.json), and
    it is the only stage batching helps, so this is the seam ingest goes through. It streams
    rather than returning a list
    so a caller can report per-page progress *and* span the whole document in one call -
    which is what makes the backoff below stick.

    On an out-of-memory error the batch halves and the *same* pages are retried. The size
    only ever shrinks, and it lives for the life of the generator, so a document that OOMs
    at the configured size pays one failed forward pass in total rather than one per batch.
    Pre-chunking pages and calling this once per chunk gives that guarantee up, because each
    call starts back at EMBED_BATCH_SIZE - iterate it over the whole page list instead.

    A single image that will not fit re-raises, because backing off further cannot help.
    Worth the handling: an ingest killed halfway leaves a truncated document that a later
    sync will consider current (see the fingerprint note in src/ingest.py).

    The first real multi-page batch is checked against a single-page embedding of its own
    first page before anything is yielded, because some backends corrupt a batched forward
    pass outright - see `_batching_is_trustworthy`. A backend that fails drops to single-page
    passes for the rest of the process and re-embeds the batch it was about to hand back, so
    a poisoned vector is never yielded even once.
    """
    global _batching_verified
    size = max(1, batch_size or EMBED_BATCH_SIZE)
    if _batching_verified is False:
        size = 1
    start = 0
    while start < len(images):
        chunk = images[start:start + size]
        try:
            vectors = _embed_batch(chunk)
            if len(chunk) > 1 and _batching_verified is None:
                _batching_verified = _batching_is_trustworthy(chunk, vectors)
                if not _batching_verified:
                    size = 1
                    logger.error(
                        "batched embedding does not match single-page embedding on this "
                        "backend; falling back to one page per forward pass for this process",
                        extra={"device": str(load_model()[0].device),
                               "dtype": str(load_model()[0].dtype),
                               "batch_size": len(chunk)},
                    )
                    continue  # same pages, one at a time - never yield the suspect vectors
        except (RuntimeError, MemoryError) as exc:
            if size == 1 or not _is_oom(exc):
                raise
            size = size // 2
            _release_memory()
            logger.warning(
                "embed batch ran out of memory; retrying smaller",
                extra={"batch_size": size, "pages_remaining": len(images) - start},
            )
            continue  # same pages, smaller batch
        yield start, vectors
        start += len(chunk)

def embed_images(
    images: list[Image.Image], batch_size: int | None = None,
) -> list[list[list[float]]]:
    """Embed page images into patch-level multivectors, in order. Flattens `iter_embedded`
    for callers that want the whole document at once rather than per-batch progress."""
    return [vector for _, batch in iter_embedded(images, batch_size) for vector in batch]

def embed_image(image: Image.Image) -> list[list[float]]:
    """Embed a single page image into its patch-level multivector."""
    return embed_images([image], batch_size=1)[0]

def embed_query(text: str) -> list[list[float]]:
    """Embed a text query into its token-level multivector."""
    model, processor = load_model()
    inputs = processor.process_queries([text]).to(model.device)
    with torch.no_grad():
        out = model(**inputs) # shape: (1, n_tokens, 128)
        return _to_multivector(out[0])

def is_loaded() -> bool:
    """True once the model + processor are cached. A public probe for the server's
    /health and lifespan, so callers never touch the private globals directly."""
    return _model is not None and _processor is not None