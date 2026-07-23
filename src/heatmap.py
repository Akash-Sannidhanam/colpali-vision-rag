"""Patch-level MaxSim heatmap - which page patches a query lit up ("why this page?").

Complements the answer crop (which shows *where* Gemini read the answer) with the
retrieval side of the story: for a cited page, ColQwen2's per-patch token embeddings are
scored against the query tokens and reduced to a small [0,1] grid, so the UI can tint the
patches the query matched most strongly over the page image.

Kept separate from `embedder.py` (which owns the model) so this module stays a thin,
interpretability-only helper. The query x patch similarity is a short pure-torch inline of
colpali_engine's `get_similarity_maps_from_embeddings` recipe - reused directly it would
drag in `colpali_engine.interpretability`, whose package `__init__` imports matplotlib
(plotting we never use). Inlining keeps the dependency surface to torch/PIL and lets the
whole tail be unit-tested without a model or processor; `load_model` is imported lazily so
`import src.heatmap` stays cheap.
"""

from pathlib import Path

import torch
from PIL import Image


def _similarity_map(
    image_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    n_patches: tuple[int, int],
    image_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-token query x patch similarity for one image, shape (query_tokens, n_x, n_y).

    Pure-torch equivalent of colpali_engine's `get_similarity_maps_from_embeddings`: drop
    the non-image tokens via `image_mask`, fold the remaining patch tokens back into the
    (n_x, n_y) grid (they come out row-major - y outer, x inner), then dot each query token
    against every patch. Inputs are the batch-of-1 model outputs (index 0 used here).
    """
    n_x, n_y = int(n_patches[0]), int(n_patches[1])
    patches = image_embeddings[0][image_mask[0]]              # (n_x*n_y, dim)
    if patches.shape[0] != n_x * n_y:
        raise ValueError(
            f"image patch count {patches.shape[0]} != n_x*n_y {n_x * n_y} "
            "- get_n_patches / image_mask mismatch"
        )
    grid = patches.reshape(n_y, n_x, -1).permute(1, 0, 2)     # (n_x, n_y, dim)
    return torch.einsum("qd,xyd->qxy", query_embeddings[0], grid)  # (query_tokens, n_x, n_y)


def _grid_from_maps(sim: torch.Tensor, n_patches: tuple[int, int]) -> tuple[list[list[float]], int, int]:
    """Reduce a per-token similarity map to a normalized `grid[y][x]` in [0, 1].

    `sim` is (query_tokens, n_x, n_y). We take the max over query tokens (the patches that
    win any token are the MaxSim-relevant ones), min/max normalize to [0, 1] (mirrors
    colpali's `normalize_similarity_map`, inlined), then transpose x-major -> row-major so
    rows index y - the layout the UI paints onto the page.
    """
    n_x, n_y = int(n_patches[0]), int(n_patches[1])
    agg = sim.amax(dim=0)                                     # (n_x, n_y): strongest token per patch
    lo, hi = agg.min(), agg.max()
    rng = hi - lo
    # A flat map (rng == 0) carries no signal -> all zeros, never a divide-by-zero artifact.
    agg = (agg - lo) / rng if rng > 0 else torch.zeros_like(agg)
    grid = agg.transpose(0, 1).cpu().tolist()                # (n_x, n_y) -> (n_y, n_x) rows=y
    return grid, n_x, n_y


def page_similarity(question: str, image_path: Path) -> tuple[list[list[float]], int, int]:
    """Compute the query->page patch heatmap for one page image.

    Runs two forward passes (page + query) on the cached ColQwen2 model, so callers must
    serialize it on the same GPU lock as the rest of the pipeline. Returns
    `(grid, n_x, n_y)` where `grid[y][x]` in [0, 1] is the query's match strength at patch
    (x, y).
    """
    from src.embedder import load_model  # lazy: keep module import light for tests

    model, processor = load_model()
    with Image.open(image_path) as page_file:
        image = page_file.convert("RGB")
        batch_images = processor.process_images([image]).to(model.device)
        batch_queries = processor.process_queries([question]).to(model.device)
        with torch.no_grad():
            image_embeddings = model(**batch_images)         # (1, image_tokens, 128)
            query_embeddings = model(**batch_queries)        # (1, query_tokens, 128)
        # image.size is (width, height) - exactly what get_n_patches expects (image_size[0]
        # = width). spatial_merge_size is a ColQwen2/2.5 model property.
        n_patches = processor.get_n_patches(
            image_size=image.size, spatial_merge_size=model.spatial_merge_size
        )
        image_mask = processor.get_image_mask(batch_images)  # (1, image_tokens) bool

    # .float() before the einsum sidesteps bf16-on-MPS quirks (matches embed_image's cast).
    sim = _similarity_map(
        image_embeddings.float(), query_embeddings.float(), n_patches, image_mask
    )
    return _grid_from_maps(sim, n_patches)
