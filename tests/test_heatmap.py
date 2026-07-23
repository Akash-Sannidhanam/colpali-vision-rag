"""Unit tests for the patch-heatmap post-processing (src.heatmap._grid_from_maps).

Exercises the pure tensor -> grid tail with a hand-built similarity map, so no model,
processor, colpali_engine, or PNG is touched (page_similarity's model calls are covered
by the live E2E, not here). Locks the two things easy to get wrong: the x-major -> row-
major(y) transpose, and the [0,1] normalization.
"""

import torch

from src.heatmap import _grid_from_maps, _similarity_map


def test_grid_shape_is_rows_y_cols_x():
    # sim: (q_tokens, n_x, n_y) = (2 tokens, 3 wide, 4 tall)
    grid, n_x, n_y = _grid_from_maps(torch.zeros(2, 3, 4), (3, 4))
    assert (n_x, n_y) == (3, 4)
    assert len(grid) == 4                          # n_y rows
    assert all(len(row) == 3 for row in grid)      # n_x cols each


def test_orientation_hot_patch_lands_at_grid_y_x():
    # one query token; hot at x=0, y=2 (i.e. sim[token][x=0][y=2])
    sim = torch.zeros(1, 3, 4)
    sim[0, 0, 2] = 5.0
    grid, _, _ = _grid_from_maps(sim, (3, 4))
    assert grid[2][0] == 1.0     # row-major [y][x]: the hot cell, normalized to the max
    assert grid[0][1] == 0.0     # a cold cell


def test_values_normalized_to_unit_range():
    sim = torch.tensor([[[1.0, 3.0], [2.0, 5.0]]])  # (1 token, n_x=2, n_y=2)
    grid, _, _ = _grid_from_maps(sim, (2, 2))
    flat = [v for row in grid for v in row]
    assert min(flat) == 0.0 and max(flat) == 1.0
    assert all(0.0 <= v <= 1.0 for v in flat)


def test_max_over_query_tokens_lights_both_patches():
    # two tokens, each hottest on a different patch -> both light up after amax(dim=0)
    sim = torch.zeros(2, 2, 2)
    sim[0, 0, 0] = 10.0          # token 0 hottest at (x0, y0)
    sim[1, 1, 1] = 10.0          # token 1 hottest at (x1, y1)
    grid, _, _ = _grid_from_maps(sim, (2, 2))
    assert grid[0][0] == 1.0     # (y0, x0)
    assert grid[1][1] == 1.0     # (y1, x1)
    assert grid[0][1] == 0.0 and grid[1][0] == 0.0


def test_flat_map_returns_all_zeros_no_spurious_hot():
    # a uniform map has no informative patch -> all zeros, never a divide-by-zero artifact
    grid, _, _ = _grid_from_maps(torch.full((3, 2, 2), 4.0), (2, 2))
    assert all(v == 0.0 for row in grid for v in row)


def test_similarity_map_masks_reshapes_and_scores():
    # 2x2 patch grid, dim=3. Image tokens are row-major (y outer, x inner): the flat token
    # order is (x0,y0),(x1,y0),(x0,y1),(x1,y1). Give each patch a distinct one-hot embedding.
    n_x, n_y, dim = 2, 2, 3
    patch_embs = torch.tensor(
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]  # 4 patches x dim
    )
    # Prepend one non-image token that the mask must drop.
    image_embeddings = torch.cat([torch.full((1, dim), 9.0), patch_embs]).unsqueeze(0)  # (1,5,dim)
    image_mask = torch.tensor([[False, True, True, True, True]])
    # One query token aligned with patch index 2 = (x0, y1).
    query_embeddings = torch.tensor([[0.0, 0.0, 1.0]]).unsqueeze(0)  # (1,1,dim)

    sim = _similarity_map(image_embeddings, query_embeddings, (n_x, n_y), image_mask)
    assert sim.shape == (1, n_x, n_y)                 # (q_tokens, n_x, n_y)
    # token dots highest with patch (x0, y1) -> sim[0, x=0, y=1] is the max
    assert sim[0, 0, 1] == sim.max()
    assert sim[0, 0, 1] == 1.0


def test_similarity_map_raises_on_patch_count_mismatch():
    image_embeddings = torch.zeros(1, 3, 4)           # 3 tokens, all "image"
    image_mask = torch.tensor([[True, True, True]])
    query_embeddings = torch.zeros(1, 1, 4)
    try:
        _similarity_map(image_embeddings, query_embeddings, (2, 2), image_mask)  # needs 4 patches
    except ValueError as exc:
        assert "mismatch" in str(exc)
    else:
        raise AssertionError("expected a ValueError on patch-count mismatch")
