"""Tests for `config.validate()`, the startup fail-fast check.

Patch the by-value module globals on `src.config` itself - they are read inside
`validate()` at call time, so monkeypatching them is what steers the branch.
"""

import math

import pytest

from src import config


def _ok(monkeypatch):
    """Put config in a state where validate() passes, so a test can break one thing."""
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(config, "CANDIDATE_FANOUT", 2.0)
    monkeypatch.setattr(config, "COLLECTION_NAME", "pdf_pages")
    monkeypatch.setattr(config, "EMBED_VISUAL_TOKENS", None)
    monkeypatch.setattr(config, "HEATMAP_SMOOTH_SIGMA", 1.5)


def test_validate_passes_on_a_sane_config(monkeypatch):
    """The happy path returns None rather than raising."""
    _ok(monkeypatch)
    assert config.validate() is None


def test_validate_rejects_a_missing_api_key(monkeypatch):
    """An empty key is caught at startup, not at the first Gemini call."""
    _ok(monkeypatch)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        config.validate()


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_validate_rejects_a_non_finite_fanout(monkeypatch, bad):
    """float() parses "inf"/"nan" happily, and round() then raises on *every* search.

    Catching it at startup turns a per-query OverflowError/ValueError deep inside
    `vector_store.search` into one clear message naming the knob.
    """
    _ok(monkeypatch)
    monkeypatch.setattr(config, "CANDIDATE_FANOUT", bad)

    with pytest.raises(RuntimeError, match="CANDIDATE_FANOUT"):
        config.validate()


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_validate_rejects_a_non_finite_heatmap_sigma(monkeypatch, bad):
    """float() parses "inf"/"nan" happily, and gaussian_filter() would fail on them.

    Catching it at startup turns a per-/heatmap-call error deep inside scipy into
    one clear message naming the knob.
    """
    _ok(monkeypatch)
    monkeypatch.setattr(config, "HEATMAP_SMOOTH_SIGMA", bad)

    with pytest.raises(RuntimeError, match="HEATMAP_SMOOTH_SIGMA"):
        config.validate()


def test_validate_allows_zero_heatmap_sigma(monkeypatch):
    """Zero is valid - it disables smoothing and restores the raw grid."""
    _ok(monkeypatch)
    monkeypatch.setattr(config, "HEATMAP_SMOOTH_SIGMA", 0.0)

    assert config.validate() is None


@pytest.mark.parametrize("bad", ["", "pdf_pages_7", "arm_512"])
def test_validate_rejects_a_collection_name_shaped_like_a_physical_one(monkeypatch, bad):
    """`pdf_pages_7` is how vector_store names a *version* of the alias `pdf_pages`.

    Allowing it as an alias would let one arm's orphan sweep delete another's index mid
    rebuild - silent, and only visible as a collection that vanished. Refusing the name is
    cheaper than teaching the sweep to tell the two apart.
    """
    _ok(monkeypatch)
    monkeypatch.setattr(config, "COLLECTION_NAME", bad)

    with pytest.raises(RuntimeError, match="COLLECTION_NAME"):
        config.validate()


def test_validate_allows_a_non_numeric_arm_suffix(monkeypatch):
    """The documented arm shape is fine: only an all-digit suffix collides."""
    _ok(monkeypatch)
    monkeypatch.setattr(config, "COLLECTION_NAME", "pdf_pages_vt512")

    assert config.validate() is None


@pytest.mark.parametrize("bad", [0, -1])
def test_validate_rejects_a_visual_token_budget_below_one(monkeypatch, bad):
    """A budget under one patch cannot describe a page, and the failure would be silent-ish.

    The processor would either raise deep inside a forward pass or hand back something empty
    that indexes perfectly cleanly - an index of nothing, recorded as current.
    """
    _ok(monkeypatch)
    monkeypatch.setattr(config, "EMBED_VISUAL_TOKENS", bad)

    with pytest.raises(RuntimeError, match="EMBED_VISUAL_TOKENS"):
        config.validate()


def test_validate_allows_an_unset_visual_token_budget(monkeypatch):
    """Unset means the checkpoint's own budget, which is what the stored vectors were built
    with - so it must stay a legal value rather than being normalised to a number."""
    _ok(monkeypatch)
    monkeypatch.setattr(config, "EMBED_VISUAL_TOKENS", None)

    assert config.validate() is None


# --- data directory overrides (see config._dir_from_env) ---

def test_data_dirs_default_under_the_repo_root(monkeypatch):
    """Unset env keeps every existing command and test pointing where it always did."""
    monkeypatch.delenv("PAGE_IMAGES_DIR", raising=False)

    assert config._dir_from_env("PAGE_IMAGES_DIR", config.ROOT_DIR / "page_images") == \
        config.ROOT_DIR / "page_images"


def test_data_dirs_honour_their_env_override(monkeypatch, tmp_path):
    """Overridable so a deployment can put its state on a mounted disk.

    Without this the only way to persist page_images/ is to bind-mount over the
    application directory, which is what left it ephemeral in the first place.
    """
    monkeypatch.setenv("PAGE_IMAGES_DIR", str(tmp_path / "elsewhere"))

    assert config._dir_from_env("PAGE_IMAGES_DIR", config.ROOT_DIR / "page_images") == \
        tmp_path / "elsewhere"


def test_a_relative_override_is_resolved_to_an_absolute_path(monkeypatch):
    """These paths are compared against each other, so a relative one would break that.

    `server._to_url` takes a page image's path relative to PAGE_IMAGES_DIR, and
    `vector_store._store_image_path` does the same on write - both need a stable
    absolute root regardless of the process's working directory.
    """
    monkeypatch.setenv("PAGE_IMAGES_DIR", "images")

    result = config._dir_from_env("PAGE_IMAGES_DIR", config.ROOT_DIR / "page_images")

    assert result.is_absolute()
