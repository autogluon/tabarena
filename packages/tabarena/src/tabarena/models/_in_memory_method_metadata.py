"""``InMemoryMethodMetadata`` — a :class:`MethodMetadata` backed by in-memory artifacts.

PROTOTYPE. Lets locally-produced results be registered with an arena context at init time
(via the existing ``methods=`` / ``extra_methods=`` arguments) so they flow through
``compare`` and the leaderboard/website machinery exactly like cached baseline methods —
without ever being written to disk or S3.

Only the *results-loading* slice of the :class:`MethodMetadata` contract is served from
memory (``load_results`` always; ``load_processed`` if a repo is supplied). The disk/S3-only
operations (``load_raw``, ``generate_repo``, ``method_downloader`` / ``method_uploader``,
``to_yaml``) are unsupported and raise, since there is no on-disk artifact behind them.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Self

import yaml

from tabarena.models._method_metadata import MethodMetadata

if TYPE_CHECKING:
    import pandas as pd

    from tabarena.end_to_end.method_results import MethodResults
    from tabarena.repository.evaluation_repository import EvaluationRepository

# Results filename by method_type, mirroring MethodMetadata.load_results' dispatch, so a
# directory written by to_dir is also loadable by the disk-backed MethodMetadata.from_yaml(path=).
_RESULTS_FILENAME_BY_TYPE = {
    "config": "hpo_results.parquet",
    "baseline": "model_results.parquet",
    "portfolio": "portfolio_results.parquet",
}


class InMemoryMethodMetadata(MethodMetadata):
    """A :class:`MethodMetadata` whose result artifacts live in memory rather than on disk.

    The in-memory results frame and (optional) repo are held in private slots that are
    excluded from the ``__dict__``-derived :meth:`to_info_dict`, so the metadata info table
    (and any YAML serialization) never tries to embed a DataFrame.
    """

    is_in_memory: bool = True

    #: Instance attributes holding in-memory artifacts — kept out of ``to_info_dict`` so a
    #: DataFrame never lands in ``MethodMetadataCollection.info()`` (and thus the website
    #: leaderboard merge).
    _IN_MEMORY_SLOTS = ("_results", "_repo", "_hpo_trajectories")

    def __init__(
        self,
        *,
        results: pd.DataFrame,
        repo: EvaluationRepository | None = None,
        hpo_trajectories: pd.DataFrame | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._results = results
        self._repo = repo
        self._hpo_trajectories = hpo_trajectories

    @classmethod
    def from_results_single(
        cls,
        results_single: MethodResults,
        *,
        new_result_prefix: str | None = None,
        use_suite_in_prefix: bool = False,
        use_model_results: bool = False,
    ) -> Self:
        """Build from a ``MethodResults`` (its ``method_metadata`` + in-memory results).

        ``new_result_prefix`` is treated as part of the method's *identity*: it is applied
        both to the results frame (via ``MethodResults.get_results``) and to the
        metadata's name fields, so the website leaderboard merge on ``(ta_name, ta_suite)``
        still matches.

        ``use_suite_in_prefix`` / ``use_model_results`` are forwarded to
        :meth:`MethodResults.get_results`. When the artifact-name prefix is applied, the
        same ``[suite] `` segment get_results prepends to the frame's identity columns is
        baked into the metadata identity here too, so the website merge still matches.
        """
        base = results_single.method_metadata
        results = results_single.get_results(
            new_result_prefix=new_result_prefix,
            use_suite_in_prefix=use_suite_in_prefix,
            use_model_results=use_model_results,
        )

        # The total prefix get_results prepended to the frame's method/config_type/ta_name/
        # ta_suite columns, mirrored so the metadata identity stays in lock-step.
        identity_prefix = new_result_prefix or ""
        if use_suite_in_prefix:
            identity_prefix = identity_prefix + f"[{base.suite}] "

        kwargs = base.to_info_dict()
        if identity_prefix:
            # Bake the prefix into identity so it matches the (already-prefixed) frame:
            #   * method/suite -> ta_name/ta_suite (the website merge key)
            #   * model_key            -> config_type (the rename-map / family key)
            #   * display_name         -> rendered method name
            for field in ("method", "suite", "model_key", "display_name"):
                value = kwargs.get(field)
                if isinstance(value, str):
                    kwargs[field] = identity_prefix + value

        return cls(results=results, repo=results_single_repo(results_single), **kwargs)

    @classmethod
    def from_results_df(
        cls,
        results: pd.DataFrame,
        *,
        method: str,
        suite: str,
        config_type: str | None = None,
        method_type: str = "config",
        repo: EvaluationRepository | None = None,
        hpo_trajectories: pd.DataFrame | None = None,
        display_name: str | None = None,
        can_hpo: bool = True,
    ) -> Self:
        """Wrap an already-computed results frame as a registerable in-memory method.

        The DataFrame-first complement to :meth:`from_results_single`, for results produced
        directly as a frame rather than via a ``MethodResults`` — e.g. the
        default / tuned / tuned+ensemble frame returned by
        :meth:`~tabarena.contexts.abstract_arena_context.AbstractArenaContext.combine_hpo`.
        The frame is returned verbatim by :meth:`load_results`, so it must already carry the
        leaderboard columns (``method`` / ``dataset`` / ``fold`` / ``metric_error`` / ...).

        For a ``"config"`` method, ``config_type`` is the family key: it becomes ``model_key``
        (hence :attr:`MethodMetadata.config_type`), matching the frame's ``config_type`` column
        and the ``"<config_type> (tuned)"`` style method names so the leaderboard renders them as
        one tunable family. Config-only fields are not accepted for other ``method_type`` values.

        ``hpo_trajectories`` is an optional tuning-trajectory frame (e.g. from
        :meth:`~tabarena.contexts.abstract_arena_context.AbstractArenaContext.generate_portfolio_trajectories`),
        served by :meth:`load_hpo_trajectories` and picked up by the tuning-trajectory plots.
        """
        # A frame-built method never has a raw artifact, and only has a "processed" repo when one
        # is supplied (and must genuinely belong to this method — i.e. hold its config_type's
        # configs). These flags are descriptive (not used to gate loading), but kept accurate so
        # ``load_processed`` raising for ``repo=None`` matches ``has_processed=False``.
        has_processed = repo is not None
        if method_type == "config":
            return cls(
                results=results,
                repo=repo,
                hpo_trajectories=hpo_trajectories,
                method=method,
                suite=suite,
                method_type="config",
                ag_key=config_type,
                model_key=config_type,
                can_hpo=can_hpo,
                display_name=display_name,
                has_raw=False,
                has_processed=has_processed,
            )
        return cls(
            results=results,
            repo=repo,
            hpo_trajectories=hpo_trajectories,
            method=method,
            suite=suite,
            method_type=method_type,
            display_name=display_name,
            has_raw=False,
            has_processed=has_processed,
        )

    # ------------------------------------------------------------------ serialization
    def to_info_dict(self) -> dict:
        # Build on the base exclusion (drops the runtime-only ``artifact_dir`` / ``cache_root``
        # overrides), then
        # also drop the in-memory artifact slots so no DataFrame/repo lands in the info table.
        return {k: v for k, v in super().to_info_dict().items() if k not in self._IN_MEMORY_SLOTS}

    # ------------------------------------------------------------------ persist / cache to disk
    def to_dir(self, path: str | Path) -> Path:
        """Persist this in-memory method to a self-contained artifact directory.

        Writes ``metadata.yaml`` + ``results/<...>.parquet`` (and ``processed/`` when a repo is
        held) in the standard :class:`MethodMetadata` on-disk layout, so the result is reloadable
        both by :meth:`from_dir` (back into an ``InMemoryMethodMetadata``) and by the disk-backed
        :meth:`MethodMetadata.from_yaml` (``path=``). Lets an expensive computation (e.g.
        :meth:`~tabarena.contexts.abstract_arena_context.AbstractArenaContext.combine_hpo`) be
        cached once and reloaded on later runs instead of recomputed. Returns the directory.
        """
        if self.method_type not in _RESULTS_FILENAME_BY_TYPE:
            raise ValueError(
                f"Cannot persist method_type={self.method_type!r}; "
                f"expected one of {sorted(_RESULTS_FILENAME_BY_TYPE)}.",
            )
        path = Path(path)
        results_dir = path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        # to_info_dict already omits the in-memory slots + runtime-only path overrides, leaving
        # exactly the dataclass fields from_dir feeds back into the constructor.
        with open(path / "metadata.yaml", "w") as f:
            yaml.dump(self.to_info_dict(), f, default_flow_style=False)
        self._results.to_parquet(results_dir / _RESULTS_FILENAME_BY_TYPE[self.method_type], index=False)
        if self._hpo_trajectories is not None:
            self._hpo_trajectories.to_parquet(results_dir / "hpo_trajectories.parquet", index=False)
        if self._repo is not None:
            self._repo.to_dir(path / "processed")
        return path

    @classmethod
    def from_dir(cls, path: str | Path, *, prediction_format: str = "memmap") -> Self:
        """Reconstruct an ``InMemoryMethodMetadata`` previously written by :meth:`to_dir`.

        Loads the directory as a disk-backed :class:`MethodMetadata` (via
        :meth:`MethodMetadata.from_yaml`) and lifts its artifacts into memory — reusing that
        method's own ``load_results`` (results parquet, dispatched by ``method_type``),
        ``load_hpo_trajectories``, and ``load_processed`` (the ``processed/`` repo, if present,
        with ``prediction_format``) rather than re-reading the yaml / parquet / repo by hand.
        ``to_info_dict`` drops the runtime-only ``artifact_dir``/``cache_root``, so the rebuilt
        method is fully in-memory.
        """
        base = MethodMetadata.from_yaml(path=path)
        repo = base.load_processed(prediction_format=prediction_format) if base.path_processed_exists else None
        hpo_trajectories = base.load_hpo_trajectories() if base.has_hpo_trajectories else None
        return cls(results=base.load_results(), repo=repo, hpo_trajectories=hpo_trajectories, **base.to_info_dict())

    # ------------------------------------------------------------------ in-memory artifacts
    def load_results(self) -> pd.DataFrame:
        return self._results.copy(deep=True)

    def load_model_results(self) -> pd.DataFrame:
        return self._results.copy(deep=True)

    def load_hpo_results(self) -> pd.DataFrame:
        return self._results.copy(deep=True)

    @property
    def has_hpo_trajectories(self) -> bool:
        return self._hpo_trajectories is not None

    def load_hpo_trajectories(self, download: bool | str = "auto") -> pd.DataFrame:
        if self._hpo_trajectories is None:
            raise self._unsupported("load_hpo_trajectories (no in-memory hpo_trajectories frame was supplied)")
        return self._hpo_trajectories.copy(deep=True)

    def load_processed(self, *args, **kwargs) -> EvaluationRepository:
        if self._repo is None:
            raise NotImplementedError(
                f"{type(self).__name__}(method={self.method!r}) has no in-memory repo; "
                f"repo-backed operations (load_repo / simulation / HPO) are unavailable. "
                f"Build it from a MethodResults that retains the repo (EndToEnd.from_raw) to enable these.",
            )
        return self._repo

    # ------------------------------------------------------------------ disabled disk/S3 ops
    def _unsupported(self, op: str):
        return NotImplementedError(
            f"{op} is unsupported for {type(self).__name__}(method={self.method!r}): "
            f"its artifacts live in memory, with no on-disk/S3 backing.",
        )

    def to_yaml(self, *args, **kwargs):
        raise self._unsupported("to_yaml")

    def to_yaml_fileobj(self, *args, **kwargs):
        raise self._unsupported("to_yaml_fileobj")

    def load_raw(self, *args, **kwargs):
        raise self._unsupported("load_raw")

    def generate_repo(self, *args, **kwargs):
        raise self._unsupported("generate_repo")

    def method_downloader(self, *args, **kwargs):
        raise self._unsupported("method_downloader")

    def method_uploader(self, *args, **kwargs):
        raise self._unsupported("method_uploader")


def results_single_repo(results_single: MethodResults) -> EvaluationRepository | None:
    """Return ``results_single.repo`` if it carries one (only in-memory pipeline runs set it),
    else ``None`` — keeps the in-memory repo when available for simulation paths.
    """
    return getattr(results_single, "repo", None)
