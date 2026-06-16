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

from typing import TYPE_CHECKING, Self

from tabarena.models._method_metadata import MethodMetadata

if TYPE_CHECKING:
    import pandas as pd

    from tabarena.nips2025_utils.end_to_end_single import EndToEndResultsSingle
    from tabarena.repository.evaluation_repository import EvaluationRepository


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
    _IN_MEMORY_SLOTS = ("_results", "_repo")

    def __init__(
        self,
        *,
        results: pd.DataFrame,
        repo: EvaluationRepository | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._results = results
        self._repo = repo

    @classmethod
    def from_results_single(
        cls,
        results_single: EndToEndResultsSingle,
        *,
        new_result_prefix: str | None = None,
    ) -> Self:
        """Build from an ``EndToEndResultsSingle`` (its ``method_metadata`` + in-memory results).

        ``new_result_prefix`` is treated as part of the method's *identity*: it is applied
        both to the results frame (via ``EndToEndResultsSingle.get_results``) and to the
        metadata's name fields, so the website leaderboard merge on ``(ta_name, ta_suite)``
        still matches.
        """
        base = results_single.method_metadata
        results = results_single.get_results(new_result_prefix=new_result_prefix)

        kwargs = base.to_info_dict()
        if new_result_prefix:
            # Bake the prefix into identity so it matches the (already-prefixed) frame:
            #   * method/artifact_name -> ta_name/ta_suite (the website merge key)
            #   * model_key            -> config_type (the rename-map / family key)
            #   * display_name         -> rendered method name
            for field in ("method", "artifact_name", "model_key", "display_name"):
                value = kwargs.get(field)
                if isinstance(value, str):
                    kwargs[field] = new_result_prefix + value

        return cls(results=results, repo=results_single_repo(results_single), **kwargs)

    # ------------------------------------------------------------------ serialization
    def to_info_dict(self) -> dict:
        # Build on the base exclusion (drops the runtime-only ``cache_root`` override), then
        # also drop the in-memory artifact slots so no DataFrame/repo lands in the info table.
        return {k: v for k, v in super().to_info_dict().items() if k not in self._IN_MEMORY_SLOTS}

    # ------------------------------------------------------------------ in-memory artifacts
    def load_results(self) -> pd.DataFrame:
        return self._results.copy(deep=True)

    def load_model_results(self) -> pd.DataFrame:
        return self._results.copy(deep=True)

    def load_hpo_results(self) -> pd.DataFrame:
        return self._results.copy(deep=True)

    def load_processed(self, *args, **kwargs) -> EvaluationRepository:
        if self._repo is None:
            raise NotImplementedError(
                f"{type(self).__name__}(method={self.method!r}) has no in-memory repo; "
                f"repo-backed operations (load_repo / simulation / HPO) are unavailable. "
                f"Build it from an EndToEndSingle (which retains the repo) to enable these.",
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


def results_single_repo(results_single: EndToEndResultsSingle) -> EvaluationRepository | None:
    """Return ``results_single.repo`` if it carries one (``EndToEndSingle.to_results`` does
    not), else ``None`` — keeps the in-memory repo when available for simulation paths.
    """
    return getattr(results_single, "repo", None)
