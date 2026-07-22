"""ColPali (ColQwen2 / ColQwen2.5) visual embedder- reads pages by sight."""
import torch
from colpali_engine.models import (
    ColQwen2,
    ColQwen2_5,
    ColQwen2_5_Processor,
    ColQwen2Processor,
)
from PIL import Image

from src.config import COLPALI_MODEL

_model: ColQwen2 | ColQwen2_5 | None = None
_processor: ColQwen2Processor | ColQwen2_5_Processor | None = None


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

def embed_image(image: Image.Image) -> list[list[float]]:
    """Embed a single page image into its patch-level multivector."""
    model, processor = load_model()
    inputs = processor.process_images([image]).to(model.device)
    with torch.no_grad():
        out = model(**inputs) # shape: (1, n_patches, 128)
        return _to_multivector(out[0])

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