"""ColPali (ColQwen2 / ColQwen2.5) visual embedder- reads pages by sight."""
import time
from collections.abc import Iterator
from functools import cache

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

def _synchronize() -> None:
    """Block until queued GPU work has actually finished.

    Only ever called while collecting `stats`. CUDA and MPS both dispatch asynchronously, so
    `model(**inputs)` returns long before the compute does and the wait is instead paid by
    the first thing that touches the result - the `.cpu()` inside `_to_multivector`. Timing
    the two halves without this would charge nearly all of the GPU's time to the decode and
    report a forward pass that looks ~free, which is precisely backwards.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()

def preprocess(images: list[Image.Image]) -> object:
    """The CPU half of a forward pass: resize, patchify and pad a batch of pages.

    Split from `forward` because it is the only part that never touches the GPU, which is
    what lets ingest run it on a producer thread while the GPU is still busy with the
    previous batch (see `ingest.ingest_pdf`), and what lets the profiler charge the two
    halves separately. Returns the processor's batch, still on the CPU.
    """
    _, processor = load_model()
    return processor.process_images(images)

def forward(inputs, stats: dict | None = None) -> list[list[list[float]]]:
    """The GPU half: run the model over a preprocessed batch, trimmed to each page's patches.

    **The trim is not optional.** `process_images` pads the batch to its longest member
    (`padding="longest"`, `padding_side="left"`) and ColQwen2's forward *zeroes* the pad
    positions but does not remove them - so `out[i]` still carries leading all-zero rows
    for every page shorter than the longest one. Storing those would change retrieval:
    Qdrant's MaxSim takes, per query token, the max dot product over a page's vectors, and
    a zero vector wins that max whenever every real dot is negative. Indexing the
    attention mask drops exactly the padding, leaving the same vectors a batch of one
    produces - which is what lets EMBED_BATCH_SIZE stay out of EMBED_VERSION.

    The batch size is read off the attention mask rather than passed in, so a caller that
    hands work across a thread boundary need only carry the preprocessed batch.
    """
    model, _ = load_model()
    inputs = inputs.to(model.device)
    with torch.no_grad():
        out = model(**inputs)  # (batch, padded_patches, 128)
    mask = inputs["attention_mask"].bool()  # (batch, padded_patches)
    if stats is not None:
        _synchronize()  # see _synchronize: otherwise the decode is charged for the GPU
        stats["forward_s"] = stats.get("forward_s", 0.0) + (time.perf_counter() - stats["_t"])
        stats["_t"] = time.perf_counter()
    vectors = [_to_multivector(out[i][mask[i]]) for i in range(mask.shape[0])]
    if stats is not None:
        stats["decode_s"] = stats.get("decode_s", 0.0) + (time.perf_counter() - stats["_t"])
        stats["patches"] = stats.get("patches", 0) + sum(len(v) for v in vectors)
    return vectors

def _embed_batch(
    images: list[Image.Image], stats: dict | None = None,
) -> list[list[list[float]]]:
    """One forward pass over several pages: `preprocess` then `forward`.

    `stats`, when given, accumulates `preprocess_s` / `forward_s` / `decode_s` / `patches`
    across every batch it is passed to. Production callers pass nothing; it exists so
    `scripts/profile_ingest.py` can charge the three sub-stages separately without a second
    copy of this function drifting away from the one ingest actually runs.
    """
    if stats is None:
        return forward(preprocess(images))
    stats["_t"] = time.perf_counter()
    inputs = preprocess(images)
    stats["preprocess_s"] = stats.get("preprocess_s", 0.0) + (time.perf_counter() - stats["_t"])
    stats["_t"] = time.perf_counter()
    return forward(inputs, stats)

@cache
def _batching_is_supported() -> bool:
    """False on backends that miscompute a batched forward pass. Evaluated once per process.

    **MPS is excluded.** On MPS + bfloat16 (torch 2.11.0 / transformers 5.3.0, and
    independent of `attn_implementation` - eager, sdpa and the default all fail) a batched
    forward pass silently corrupts the FIRST sequence in the batch: the same page placed at
    slot 0 of a batch of two comes back ~0.4 per component from the same page embedded alone,
    while slots 1..n-1 are bit-identical to it. CPU float32 and MPS float32 are both exact, so
    the arithmetic and this module are fine; the bf16 kernel is not. Compare
    pytorch/pytorch#162592 and #163597.

    Nothing about that failure is visible from the outside: no error, no NaN, correct patch
    counts, and an ingest that writes a `content_hash` marking the document current. It would
    poison one page in every EMBED_BATCH_SIZE of the index, and no test that stubs the model
    could ever see it.

    This is a **device check, not a measurement** - deliberately the blunt version. Two things
    follow from that and are the price of it being simple: MPS float32 would in fact batch
    correctly and is refused anyway, and a torch release that fixes the kernel will not
    re-enable batching until someone deletes this. `profile_ingest.py --verify-equivalence`
    is how you would find out, and is what should be re-run on any dtype/device/torch change.
    """
    device, _ = _device_and_dtype()
    if device == "mps":
        logger.warning(
            "batching disabled: MPS miscomputes the first sequence of a batched forward "
            "pass, so pages are embedded one at a time (EMBED_BATCH_SIZE is ignored here)",
            extra={"device": device},
        )
        return False
    return True

def iter_embedded(
    images: list[Image.Image], batch_size: int | None = None, stats: dict | None = None,
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

    On a backend that miscomputes batched forward passes the size is forced to 1 up front,
    so a corrupted vector is never produced in the first place - see `_batching_is_supported`.

    `stats` is profiling-only and passed straight through to `_embed_batch`; see there.
    """
    size = max(1, batch_size or EMBED_BATCH_SIZE)
    if size > 1 and not _batching_is_supported():
        size = 1
    start = 0
    while start < len(images):
        chunk = images[start:start + size]
        try:
            vectors = _embed_batch(chunk, stats)
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