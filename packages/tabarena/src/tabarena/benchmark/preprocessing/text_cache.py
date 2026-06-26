"""Canonical location, keying, and (re)use of the semantic-text embedding cache.

Computing semantic text embeddings (via :class:`SemanticTextFeatureGenerator`) is expensive, so
they are precomputed once per task and cached to a parquet file (text value -> embedding). This
module owns the *contract* for that cache so generation, loading at fit time, and the
download/prefetch path all agree:

* **Location** — under the TabArena cache root (``get_tabarena_cache_root()/text_cache/``),
  consistent with results/artifacts, with a read-time fallback to the legacy OpenML-root location.
* **Versioning** — embeddings are tied to a specific encoder (model + truncation dim), so caches
  live under an ``<embedding-id>`` subfolder. Changing the encoder
  changes the subfolder, so a stale cache can never be silently loaded.
* **Key** — the readable :attr:`UserTask.slug` for data-foundry / user tasks (matching the result
  caches and committed metadata), or the integer OpenML task id otherwise.

Embeddings are model-agnostic across tasks (a text value embeds the same regardless of task), but
they are cached per task so a benchmark only ships/loads the caches for the tasks it runs.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticEmbeddingSpec:
    """Identity + config of the semantic-text embedding, and the cache version tag it derives.

    The embeddings depend on more than the model + truncation dim (the ``query`` prompt, L2
    normalization, the fp16 dtype, the text preprocessing in :func:`sanitize_text`, and the pinned
    ``sentence-transformers`` behavior). To keep the cache id *forward-safe*, the id is

        ``<model-basename>-d<truncate_dim>-r<revision>``

    where ``model`` / ``truncate_dim`` flip the id automatically, and ``revision`` is the explicit
    knob to bump for *any other* change that alters the produced embeddings — so a cache made by an
    older encoder can never be silently loaded against a newer one. This spec is the single source
    of truth: the encoder is built from it (see ``TabArenaDefaultTextEncoder.get_default_encoder``)
    and the cache path / bundled artifact filename are keyed by :attr:`id`.
    """

    model: str
    """Hugging Face model id of the sentence-transformer encoder."""
    truncate_dim: int
    """Matryoshka truncation dimension applied to the embeddings."""
    revision: int
    """Bump for embedding-affecting changes NOT captured by ``model``/``truncate_dim`` (prompt,
    normalization, dtype, text preprocessing, an ST upgrade that changes outputs, ...)."""
    description: str
    """Human-readable note describing this embedding configuration (for maintainers)."""

    @property
    def id(self) -> str:
        """Forward-safe cache id, e.g. ``Qwen3-Embedding-8B-d32-r1``."""
        return f"{self.model.rsplit('/', 1)[-1]}-d{self.truncate_dim}-r{self.revision}"


#: The active semantic-text embedding. Change ``model``/``truncate_dim`` and/or bump ``revision``
#: (per its docstring) whenever the produced embeddings change; this re-keys the cache so stale
#: caches are ignored rather than mis-loaded.
SEMANTIC_EMBEDDING = SemanticEmbeddingSpec(
    model="Qwen/Qwen3-Embedding-8B",
    truncate_dim=32,
    revision=1,
    description=(
        "Qwen3-Embedding-8B with Matryoshka truncation to 32 dims; encoded with prompt_name='query', "
        "L2-normalized, float16. Produced/consumed by SemanticTextFeatureGenerator."
    ),
)

TextCacheMode = Literal["require", "auto", "off"]
"""How the loader treats a (text) task's cache: ``require`` = must be present (raise if missing),
``auto`` = use it if present else compute on the fly, ``off`` = ignore the cache (always compute)."""


def embedding_id() -> str:
    """Forward-safe id of the active embedding; the cache version tag (see :class:`SemanticEmbeddingSpec`)."""
    return SEMANTIC_EMBEDDING.id


def text_cache_root() -> Path:
    """Directory holding the embedding caches for the current encoder (created on demand)."""
    from tabarena.loaders import get_tabarena_cache_root

    root = get_tabarena_cache_root() / "text_cache" / embedding_id()
    root.mkdir(parents=True, exist_ok=True)
    return root


def text_cache_key(task_id_or_object) -> str:
    """Cache key for a task: the UserTask slug, or the integer OpenML task id as a string."""
    if isinstance(task_id_or_object, int):
        return str(task_id_or_object)
    slug = getattr(task_id_or_object, "slug", None)
    if slug is not None:
        return str(slug)
    return str(task_id_or_object)


def text_cache_path(task_key: str) -> Path:
    """Canonical (write) path for ``task_key`` under the current encoder's cache dir."""
    return text_cache_root() / f"{task_key}_cache.parquet"


def _legacy_text_cache_path(task_key: str) -> Path | None:
    """Pre-refactor location: ``<openml_root>/tabarena_text_cache/text_cache/<key>_cache.parquet``.

    Un-versioned and under the OpenML cache root. Returns ``None`` if OpenML is unavailable.
    """
    try:
        import openml
    except ImportError:
        return None
    base = (openml.config._root_cache_directory / "tabarena_text_cache").expanduser().resolve() / "text_cache"
    return base / f"{task_key}_cache.parquet"


def resolve_existing_cache_path(task_key: str) -> Path | None:
    """Return the path to an existing cache for ``task_key`` (canonical first, then legacy), or None."""
    canonical = text_cache_path(task_key)
    if canonical.exists():
        return canonical
    legacy = _legacy_text_cache_path(task_key)
    if legacy is not None and legacy.exists():
        return legacy
    return canonical if canonical.exists() else None


def has_text_cache(task_key: str) -> bool:
    """Whether an embedding cache (canonical or legacy) exists for ``task_key``."""
    return resolve_existing_cache_path(task_key) is not None


def load_text_cache(path: str | Path) -> dict[str, np.ndarray]:
    """Load an embedding cache parquet into a ``{text_value: embedding}`` dict."""
    from tabarena.benchmark.preprocessing.text_feature_generators import SemanticTextFeatureGenerator

    return SemanticTextFeatureGenerator.load_embedding_cache(path=path)


def save_text_cache(cache: dict[str, np.ndarray], path: str | Path) -> None:
    """Write a ``{text_value: embedding}`` dict to a parquet at ``path``."""
    from tabarena.benchmark.preprocessing.text_feature_generators import SemanticTextFeatureGenerator

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    SemanticTextFeatureGenerator.save_embedding_cache(cache=cache, path=path)


def generate_text_cache(task_id_or_object, *, ignore_cache: bool = False) -> Path:
    """Compute + cache the semantic embeddings for one task (maintainer / pre-generation path).

    Runs the model-agnostic preprocessing with only semantic-text features enabled over the task's
    feature matrix, then writes the embedding cache to :func:`text_cache_path`. Skips computation if
    a cache already exists (unless ``ignore_cache``).
    """
    from tabarena.benchmark.preprocessing.model_agnostic_default_preprocessing import (
        TabArenaModelAgnosticPreprocessing,
    )
    from tabarena.benchmark.preprocessing.text_feature_generators import SemanticTextFeatureGenerator
    from tabarena.benchmark.task.openml import OpenMLTaskWrapper
    from tabarena.benchmark.task.user_task import UserTask

    task_key = text_cache_key(task_id_or_object)
    out_path = text_cache_path(task_key)
    if (not ignore_cache) and out_path.exists():
        print(f"Text cache already exists for {task_key} at {out_path}; skipping generation.")
        return out_path

    if isinstance(task_id_or_object, UserTask):
        task = task_id_or_object.load()
    else:
        task = OpenMLTaskWrapper(task=task_id_or_object)  # already an OpenML task object
    X, _ = task.get_X_y()
    print(f"Loaded {task_key}: {len(X)} rows x {len(X.columns)} columns.")

    preprocessing = TabArenaModelAgnosticPreprocessing(
        enable_sematic_text_features=True,
        enable_raw_text_features=False,
        enable_text_special_features=False,
        enable_statistical_text_features=False,
        enable_text_ngram_features=False,
        enable_datetime_features=False,
        verbosity=4,
    )
    preprocessing.fit_transform(X=X)

    save_text_cache(cache=SemanticTextFeatureGenerator._embedding_look_up, path=out_path)
    SemanticTextFeatureGenerator._embedding_look_up.clear()
    print(f"Text cache generated and saved to: {out_path}")
    return out_path


@contextmanager
def use_text_cache_for_task(task_id_or_object, *, has_text: bool, mode: TextCacheMode = "require"):
    """Load a task's embedding cache into the generator for the duration of a fit, then restore.

    On enter (when ``mode != "off"`` and the task ``has_text``): load the cache into the shared
    ``SemanticTextFeatureGenerator._embedding_look_up`` and set ``only_load_from_cache=True`` so any
    unseen text raises rather than silently recomputing. If the cache is missing, ``require`` raises
    while ``auto`` falls back to on-the-fly computation. On exit, the generator's prior class state
    is always restored, so per-task caches never leak across tasks in a long-lived process.
    """
    from tabarena.benchmark.preprocessing.text_feature_generators import SemanticTextFeatureGenerator

    prev_lookup = SemanticTextFeatureGenerator._embedding_look_up
    prev_only_load = SemanticTextFeatureGenerator.only_load_from_cache
    try:
        if mode != "off" and has_text:
            task_key = text_cache_key(task_id_or_object)
            cache_path = resolve_existing_cache_path(task_key)
            if cache_path is not None:
                logger.debug("[TEXT CACHE] Loading embeddings for %s from %s", task_key, cache_path)
                SemanticTextFeatureGenerator._embedding_look_up = load_text_cache(cache_path)
                SemanticTextFeatureGenerator.only_load_from_cache = True
            elif mode == "require":
                raise FileNotFoundError(
                    f"Text cache not found for task {task_key!r} (encoder {embedding_id()!r}). "
                    f"Expected at {text_cache_path(task_key)}. Pre-generate or download it "
                    f"(text_cache mode='require'); use mode='auto' to compute on the fly instead.",
                )
        yield
    finally:
        SemanticTextFeatureGenerator._embedding_look_up = prev_lookup
        SemanticTextFeatureGenerator.only_load_from_cache = prev_only_load
