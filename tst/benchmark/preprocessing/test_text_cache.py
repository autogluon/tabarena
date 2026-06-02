"""Tests for the semantic-text embedding cache contract (location / key / versioning / reuse)."""

from __future__ import annotations

import numpy as np
import pytest

from tabarena.benchmark.preprocessing import text_cache as tc
from tabarena.loaders import set_tabarena_cache_root


@pytest.fixture
def cache_root(tmp_path):
    """Isolate the TabArena cache root so tests don't touch the real cache."""
    set_tabarena_cache_root(tmp_path)
    try:
        yield tmp_path
    finally:
        set_tabarena_cache_root(None)


def test_embedding_id_encodes_model_dim_and_revision():
    spec = tc.SEMANTIC_EMBEDDING
    assert tc.embedding_id() == f"{spec.model.rsplit('/', 1)[-1]}-d{spec.truncate_dim}-r{spec.revision}"
    assert tc.embedding_id() == spec.id


def test_embedding_id_changes_with_revision():
    from tabarena.benchmark.preprocessing.text_cache import SemanticEmbeddingSpec

    base = dict(model="Org/Model", truncate_dim=16, description="x")
    assert SemanticEmbeddingSpec(revision=1, **base).id == "Model-d16-r1"
    # A config-only change (same model/dim) still re-keys the cache via the revision bump.
    assert SemanticEmbeddingSpec(revision=2, **base).id == "Model-d16-r2"


def test_key_is_slug_or_int():
    assert tc.text_cache_key(363612) == "363612"

    class _UserTaskLike:
        slug = "blood_transfusion-c0b8fbb104f3"

    assert tc.text_cache_key(_UserTaskLike()) == "blood_transfusion-c0b8fbb104f3"


def test_path_is_versioned_under_tabarena_root(cache_root):
    path = tc.text_cache_path("ds-abc")
    assert path.parent == cache_root / "text_cache" / tc.embedding_id()
    assert path.name == "ds-abc_cache.parquet"


def test_resolve_and_has_cache(cache_root):
    assert tc.resolve_existing_cache_path("ds-abc") is None
    assert tc.has_text_cache("ds-abc") is False

    tc.save_text_cache({"hello": np.ones(4, dtype=np.float32)}, tc.text_cache_path("ds-abc"))
    assert tc.has_text_cache("ds-abc") is True
    assert tc.resolve_existing_cache_path("ds-abc") == tc.text_cache_path("ds-abc")


def test_save_load_roundtrip(cache_root):
    cache = {"a": np.array([1.0, 2.0], dtype=np.float32), "b": np.array([3.0, 4.0], dtype=np.float32)}
    path = tc.text_cache_path("ds-rt")
    tc.save_text_cache(cache, path)
    loaded = tc.load_text_cache(path)
    assert set(loaded) == {"a", "b"}
    assert np.allclose(loaded["a"], cache["a"])


def test_legacy_fallback_used_when_canonical_absent(cache_root, tmp_path, monkeypatch):
    # No canonical cache, but a legacy one exists -> resolve returns the legacy path.
    legacy = tmp_path / "legacy" / "ds-legacy_cache.parquet"
    tc.save_text_cache({"x": np.ones(2, dtype=np.float32)}, legacy)
    monkeypatch.setattr(tc, "_legacy_text_cache_path", lambda key: legacy if key == "ds-legacy" else None)

    assert tc.resolve_existing_cache_path("ds-legacy") == legacy
    assert tc.has_text_cache("ds-legacy") is True


class _UT:
    slug = "ds-ctx"


def _gen():
    from tabarena.benchmark.preprocessing.text_feature_generators import SemanticTextFeatureGenerator

    return SemanticTextFeatureGenerator


def test_use_cache_loads_and_restores(cache_root):
    gen = _gen()
    tc.save_text_cache({"hi": np.ones(3, dtype=np.float32)}, tc.text_cache_path("ds-ctx"))

    prev_lookup, prev_flag = gen._embedding_look_up, gen.only_load_from_cache
    with tc.use_text_cache_for_task(_UT(), has_text=True, mode="require"):
        assert gen.only_load_from_cache is True
        assert "hi" in gen._embedding_look_up
    # State restored on exit (no leakage across tasks).
    assert gen._embedding_look_up is prev_lookup
    assert gen.only_load_from_cache == prev_flag


def test_require_raises_when_missing_for_text_task(cache_root):
    with (
        pytest.raises(FileNotFoundError, match="Text cache not found"),
        tc.use_text_cache_for_task(
            _UT(),
            has_text=True,
            mode="require",
        ),
    ):
        pass


def test_auto_does_not_raise_when_missing(cache_root):
    gen = _gen()
    with tc.use_text_cache_for_task(_UT(), has_text=True, mode="auto"):
        assert gen.only_load_from_cache is False  # falls back to on-the-fly computation


def test_no_text_task_is_noop_even_in_require(cache_root):
    gen = _gen()
    with tc.use_text_cache_for_task(_UT(), has_text=False, mode="require"):
        assert gen.only_load_from_cache is False  # nothing loaded; no raise


def test_off_skips_cache_even_when_present(cache_root):
    gen = _gen()
    tc.save_text_cache({"hi": np.ones(3, dtype=np.float32)}, tc.text_cache_path("ds-ctx"))
    with tc.use_text_cache_for_task(_UT(), has_text=True, mode="off"):
        assert gen.only_load_from_cache is False


def test_import_text_cache_from_container(cache_root, tmp_path):
    """The container's bundled text-cache extra artifact is copied to the canonical location."""
    from tabarena.benchmark.task.data_foundry.text_cache import (
        container_text_cache_filename,
        import_text_cache_from_container,
    )

    # The bundled artifact carries the encoder id (e.g. tabarena_text_cache_Qwen3-Embedding-8B-d32.parquet).
    filename = container_text_cache_filename()
    assert filename == f"tabarena_text_cache_{tc.embedding_id()}.parquet"

    # Minimal stand-in for a downloaded CuratedContainer with the bundled extra artifact.
    container_dir = tmp_path / "california_house_prices_2020" / "uuid-123"
    container_dir.mkdir(parents=True)
    tc.save_text_cache({"hi": np.ones(3, dtype=np.float32)}, container_dir / filename)

    class _FakeContainer:
        def has_extra_file(self, name):
            return (container_dir / name).is_file()

        def extra_file_path(self, name):
            return container_dir / name

    dst = import_text_cache_from_container(_FakeContainer(), "california_house_prices_2020-abc123")
    assert dst == tc.text_cache_path("california_house_prices_2020-abc123")
    assert dst.is_file()
    assert "hi" in tc.load_text_cache(dst)


def test_import_text_cache_noop_when_container_has_none(cache_root, tmp_path):
    from tabarena.benchmark.task.data_foundry.text_cache import import_text_cache_from_container

    class _NoExtra:
        def has_extra_file(self, name):
            return False

    assert import_text_cache_from_container(_NoExtra(), "ds-x") is None
