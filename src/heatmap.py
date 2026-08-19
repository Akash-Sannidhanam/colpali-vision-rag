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
import torch.nn.functional as F
from PIL import Image

from src.config import HEATMAP_SMOOTH_SIGMA


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


def _smooth(grid: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur over the (n_x, n_y) patch grid, replicating at the edges.

    The raw per-patch scores are noisy at this granularity - a page is only ~24x31
    patches - and single-patch spikes in blank margins are what made the overlay read as
    noise. Smoothing is *not* a display tweak dressed up: unlike a renormalization it is
    not monotone, so it moves the patch ranking and can be scored. Measured against 44
    answer regions located in the PDF text layer it lifts ROC AUC 0.662 -> 0.756
    (sign-test p = 1.3e-05), and a random map blurred the same way stays at chance, so
    the gain is signal rather than an artifact of favouring contiguous targets.
    """
    if sigma <= 0:
        return grid
    radius = max(1, round(3 * sigma))
    offsets = torch.arange(-radius, radius + 1, dtype=grid.dtype, device=grid.device)
    kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = kernel / kernel.sum()
    out = grid[None, None]                                   # (1, 1, n_x, n_y)
    # Two 1-D passes rather than one 2-D kernel: same result, and `replicate` padding
    # keeps a hot patch at the edge from being pulled toward zero by implicit zeros.
    out = F.conv2d(F.pad(out, (0, 0, radius, radius), mode="replicate"), kernel.view(1, 1, -1, 1))
    out = F.conv2d(F.pad(out, (radius, radius, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))
    return out[0, 0]


def _grid_from_maps(
    sim: torch.Tensor, n_patches: tuple[int, int], sigma: float | None = None
) -> tuple[list[list[float]], int, int]:
    """Reduce a per-token similarity map to a normalized `grid[y][x]` in [0, 1].

    `sim` is (query_tokens, n_x, n_y). We take the max over query tokens (the patches that
    win any token are the MaxSim-relevant ones), smooth the grid (see `_smooth`), min/max
    normalize to [0, 1] (mirrors colpali's `normalize_similarity_map`, inlined), then
    transpose x-major -> row-major so rows index y - the layout the UI paints onto the page.

    The max-over-tokens reduction is kept because it was *measured* to be the best of ten
    candidates, not because it was first: dropping the query's padding tokens, subtracting
    the per-patch baseline, per-token z-scoring and decomposing the MaxSim score by which
    patch wins each token all scored strictly worse. See docs/EXPERIMENTS.md.

    `sigma` overrides HEATMAP_SMOOTH_SIGMA (tests pass it explicitly; 0 disables).
    """
    n_x, n_y = int(n_patches[0]), int(n_patches[1])
    agg = sim.amax(dim=0)                                     # (n_x, n_y): strongest token per patch
    agg = _smooth(agg, HEATMAP_SMOOTH_SIGMA if sigma is None else sigma)
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
