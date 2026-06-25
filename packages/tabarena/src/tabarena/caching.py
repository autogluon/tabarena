"""A single place to declare every cache directory TabArena uses.

TabArena reads and writes four independent caches. Historically each was process-global
state a user had to configure by hand on *every* process — the driver and every distributed
worker — because the runner API exposed no cache parameter. :class:`CacheConfig` collapses
that into one object you declare once and :meth:`CacheConfig.apply` wherever a process needs
it; :class:`~tabarena.contexts.TabArenaContext` does the applying for you (on construction and
again inside ``run_jobs``), and the SLURM worker setup reuses the same object.

See :class:`CacheConfig` for the authoritative description of each cache and its role.
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

__all__ = ["CacheConfig"]

# The path-valued fields (vs. the policy flags `apply_on_run` / `scope_openml`); used by
# `from_root` to lay each cache out under a single parent directory.
_LOCATION_FIELDS = ("openml", "huggingface", "tabarena", "results")
# All constructor fields; used by `from_dict` to ignore any unknown keys.
_CONFIG_FIELDS = (*_LOCATION_FIELDS, "apply_on_run", "scope_openml")


@dataclass(frozen=True)
class CacheConfig:
    """Declarative configuration for every cache directory TabArena uses.

    Construct one and hand it to :class:`~tabarena.contexts.TabArenaContext`
    (``TabArenaContext(cache_config=CacheConfig(...))``); the context calls
    :meth:`apply` on construction so the driver process is configured, and again inside
    ``run_jobs`` so any worker that runs the context inherits the same locations. You can
    also call :meth:`apply` directly in a process you control (e.g. a custom distributed
    worker's entrypoint).

    A field left ``None`` is *not touched* — the existing environment variable / library
    default stays in effect. :meth:`apply` is idempotent and creates no directories (the
    underlying libraries create them lazily on first write).

    The four caches and their roles:

    openml:
        **The most important cache.** OpenML's root cache directory, set via
        ``openml.config.set_root_cache_directory`` (no environment variable is honored by
        the code). It holds the materialized datasets and their cross-validation splits, and
        — because TabArena hangs its derived artifacts off this root at runtime — also
        ``tabarena_tasks/`` (materialized task specs), ``tabarena_text_cache/`` (text-feature
        embeddings), ``tabarena_metadata_cache/`` (data-foundry reference manifests) and
        ``local/datasets/`` (local dataset payloads). Point this at a large disk; the library
        default is ``~/.cache/openml``. :meth:`apply` sets this on the (global) ``openml.config``;
        set ``scope_openml=True`` (or use :meth:`scoped_openml`) to point TabArena at it only for
        the duration of a run and restore any pre-existing ``openml.config`` location after.
    huggingface:
        The HuggingFace Hub cache, set via the ``HF_HOME`` environment variable. It has two
        roles: (a) foundation-model **weights** — TabPFN / Mitra / LimiX / OrionMSP / SAP-RPT
        call ``hf_hub_download`` with no explicit ``cache_dir``, so the weights land under
        ``HF_HOME``; and (b) the one-time **raw dataset download** for data-foundry /
        BeyondArena collections, which is then materialized (converted) *into* the OpenML
        cache above. Setting this controls both — the data-foundry download honors ``HF_HOME``
        because :meth:`apply` runs before any materialization. (For per-source control of just
        the data-foundry download, pass ``cache_dir=`` to ``DataFoundrySource`` directly.) The
        library default is ``~/.cache/huggingface/hub``.
    tabarena:
        TabArena's own artifact cache — the benchmark results, baselines and leaderboard data
        downloaded by ``load_results`` and friends. Set via
        ``tabarena.loaders.set_tabarena_cache_root`` (equivalently the ``$TABARENA_CACHE``
        environment variable). Default ``~/.cache/tabarena``.
    results:
        The benchmark *run-output* cache, used as the default ``expname`` for ``run_jobs`` —
        under which the runner writes ``{expname}/data/{method}/{task}/{repeat}_{fold}/results.pkl``
        (``cache_mode`` controls resume/skip). Unlike the other three this is **not** global
        process state, so :meth:`apply` does not touch it. It is used *only* when ``run_jobs`` is
        called without an ``expname`` argument; it does not change the meaning of an explicit
        ``expname=None`` (still a throwaway temp dir). ``None`` here means "no default" (``run_jobs``
        then requires an explicit ``expname``, as before).

    Besides the locations, two flags describe *how* the config is applied by a context (the
    location fields are plain data; these are policy):

    apply_on_run:
        When ``True`` (default), the context re-applies this config at the start of each
        ``run_jobs`` / ``build_jobs(pre_materialize=True)`` — so a distributed worker that
        reconstructed the context inherits the same cache locations. Set ``False`` if the
        process is already configured and you don't want the runner touching it.
    scope_openml:
        When ``True``, the OpenML root is set only for the duration of each data operation and
        the pre-existing ``openml.config`` location is restored afterwards (via
        :meth:`scoped_openml`); the TabArena/HuggingFace caches stay applied. Use this to run
        against a dedicated OpenML
        cache without permanently changing an ambient ``openml.config`` the rest of your process
        relies on. When ``False`` (default), :meth:`apply` sets the OpenML root for the process.
    """

    openml: str | Path | None = None
    huggingface: str | Path | None = None
    tabarena: str | Path | None = None
    results: str | Path | None = None
    apply_on_run: bool = True
    scope_openml: bool = False

    @classmethod
    def from_root(cls, root: str | Path, **overrides) -> CacheConfig:
        """Put every cache under a single parent directory ``root``.

        ``CacheConfig.from_root("/scratch/me")`` resolves to ``/scratch/me/openml``,
        ``/scratch/me/huggingface``, ``/scratch/me/tabarena`` and ``/scratch/me/results``.
        Handy when one large/shared disk should hold everything. Any field can be pinned via
        a keyword override (e.g. ``from_root(root, results=None)`` to keep run outputs in a
        throwaway temp dir, or ``from_root(root, scope_openml=True)`` to set a policy flag).
        """
        root = Path(root)
        base: dict = {name: root / name for name in _LOCATION_FIELDS}
        base.update(overrides)
        return cls(**base)

    def to_dict(self) -> dict:
        """Serialize to a plain JSON-able dict (``Path`` location fields rendered as ``str``).

        Used to persist the config alongside a sweep (e.g. in a ``JobBatch``) so a compute node
        can reconstruct and :meth:`apply` it. Round-trips with :meth:`from_dict`.
        """
        return {
            "openml": None if self.openml is None else str(self.openml),
            "huggingface": None if self.huggingface is None else str(self.huggingface),
            "tabarena": None if self.tabarena is None else str(self.tabarena),
            "results": None if self.results is None else str(self.results),
            "apply_on_run": self.apply_on_run,
            "scope_openml": self.scope_openml,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CacheConfig:
        """Reconstruct a :class:`CacheConfig` from its :meth:`to_dict` form."""
        return cls(**{field: data[field] for field in data if field in _CONFIG_FIELDS})

    def apply(self, *, openml: bool = True) -> None:
        """Point the current process at these caches (idempotent; ``None`` fields are no-ops).

        Applies ``openml`` via ``openml.config.set_root_cache_directory``, ``tabarena`` via
        ``tabarena.loaders.set_tabarena_cache_root``, and ``huggingface`` via the ``HF_HOME``
        environment variable (see :meth:`_apply_huggingface` for the import-timing handling).
        ``results`` is intentionally not applied — it is a ``run_jobs`` default, not global
        state.

        Args:
            openml: When ``False``, skip the OpenML field. Used by callers that want to set the
                TabArena/HuggingFace caches now but defer the OpenML root to a temporary
                :meth:`scoped_openml` override (so the ambient ``openml.config`` is left untouched
                between operations).
        """
        if openml and self.openml is not None:
            import openml as openml_lib

            openml_lib.config.set_root_cache_directory(str(Path(self.openml).expanduser()))
        if self.tabarena is not None:
            from tabarena.loaders import set_tabarena_cache_root

            set_tabarena_cache_root(Path(self.tabarena).expanduser())
        if self.huggingface is not None:
            self._apply_huggingface(Path(self.huggingface).expanduser())

    @contextlib.contextmanager
    def scoped_openml(self) -> Iterator[None]:
        """Apply the caches for the duration of the block, then restore *only* the OpenML root.

        Unlike :meth:`apply` (which sets the OpenML root permanently), this saves the current
        ``openml.config`` root cache directory, applies this config, and restores that saved value
        on exit. Use it to run TabArena against a dedicated OpenML cache *X* without disturbing an
        ambient ``openml.config`` the rest of the process relies on — outside the block OpenML keeps
        reading its original location.

        Scoping is deliberately OpenML-only (hence the name):

        * **OpenML** re-reads ``openml.config._root_cache_directory`` on every call, so a
          save/restore reliably redirects it for the block and reverts afterwards.
        * **HuggingFace** *cannot* be scoped reliably: ``huggingface_hub`` freezes its cache paths
          into module constants at import time, and that import happens during the fit (inside the
          block). Restoring ``HF_HOME`` afterwards would not redirect later HF use in the same
          process, so ``huggingface`` is left applied rather than promising a restore it can't keep.
        * **TabArena** is private state (nothing outside TabArena reads ``set_tabarena_cache_root``)
          and ``compare`` needs it to persist, so it is left applied. ``results`` is never global.
        """
        import openml as openml_lib

        saved_openml_root = openml_lib.config._root_cache_directory
        self.apply()
        try:
            yield
        finally:
            openml_lib.config.set_root_cache_directory(str(saved_openml_root))

    @staticmethod
    def _apply_huggingface(path: Path) -> None:
        """Point the HuggingFace Hub cache at ``path`` via ``HF_HOME``.

        ``huggingface_hub`` is only ever imported lazily inside model fits, so in a fresh
        process setting ``HF_HOME`` before the first fit is sufficient. We also clear the
        more specific ``HF_HUB_CACHE`` / ``HUGGINGFACE_HUB_CACHE`` env vars so a stale value
        cannot shadow ``HF_HOME``. Finally, if ``huggingface_hub.constants`` is *already*
        imported (its cache paths are frozen at import time), we repoint those constants too —
        but we never force-import the module, keeping it lazy and dependency-light.
        """
        os.environ["HF_HOME"] = str(path)
        os.environ.pop("HF_HUB_CACHE", None)
        os.environ.pop("HUGGINGFACE_HUB_CACHE", None)

        constants = sys.modules.get("huggingface_hub.constants")
        if constants is not None:
            hub_cache = str(path / "hub")
            if hasattr(constants, "HF_HOME"):
                constants.HF_HOME = str(path)
            for attr in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
                if hasattr(constants, attr):
                    setattr(constants, attr, hub_cache)
