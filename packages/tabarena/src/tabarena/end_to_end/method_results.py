"""``MethodResults`` — one method's TabArena results (metadata + result frames)."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Self

import pandas as pd
from autogluon.common.savers import save_pd

if TYPE_CHECKING:
    from tabarena.models._method_metadata import MethodMetadata
    from tabarena.repository import EvaluationRepository


class MethodResults:
    """Results of a single method: its ``MethodMetadata`` plus the per-task result frames.

    Parameters
    ----------
    method_metadata : MethodMetadata
        Method identity and on-disk artifact layout.
    model_results : pd.DataFrame or None
        Raw per-task model results prior to HPO / model selection. Loaded from the method's
        cached artifacts when ``None``.
    hpo_results : pd.DataFrame or None
        TabArena HPO simulation results (one row per (task, config, seed)). Loaded from the
        method's cached artifacts when ``None`` (config methods only).
    repo : EvaluationRepository or None
        Optional processed repository backing these results. Populated by the in-memory
        pipeline (``EndToEnd.from_raw``) so repo-backed operations (e.g. simulation via
        ``InMemoryMethodMetadata``) stay available; ``None`` when results were produced
        per-task or loaded from cache.
    """

    def __init__(
        self,
        method_metadata: MethodMetadata,
        *,
        model_results: pd.DataFrame = None,
        hpo_results: pd.DataFrame = None,
        repo: EvaluationRepository | None = None,
    ):
        self.method_metadata = method_metadata
        if model_results is None:
            model_results = self.method_metadata.load_model_results()
        if hpo_results is None and self.method_metadata.method_type == "config":
            hpo_results = self.method_metadata.load_hpo_results()
        self.model_results = model_results
        self.hpo_results = hpo_results
        self.repo = repo

    def to_method_metadata(
        self,
        *,
        new_result_prefix: str | None = None,
        use_suite_in_prefix: bool = False,
        use_model_results: bool = False,
    ):
        """Vend this method as an :class:`InMemoryMethodMetadata` (metadata + in-memory results).

        The returned object can be passed to an arena context's ``methods=`` /
        ``extra_methods=`` so the method is registered at init and flows through
        ``compare`` and the leaderboard/website machinery like a cached baseline.

        ``use_suite_in_prefix`` / ``use_model_results`` mirror
        :meth:`get_results` (forwarded through :meth:`InMemoryMethodMetadata.from_results_single`).
        """
        from tabarena.models._in_memory_method_metadata import InMemoryMethodMetadata

        return InMemoryMethodMetadata.from_results_single(
            self,
            new_result_prefix=new_result_prefix,
            use_suite_in_prefix=use_suite_in_prefix,
            use_model_results=use_model_results,
        )

    def get_results(
        self,
        new_result_prefix: str | None = None,
        use_suite_in_prefix: bool = False,
        use_model_results: bool = False,
    ) -> pd.DataFrame:
        """Get data to compare results on TabArena leaderboard.

        Args:
                new_result_prefix (str | None): If not None, add a prefix to the new
                    results to distinguish new results from the original TabArena results.
                    Use this, for example, if you re-run a model from TabArena.
        """
        use_model_results = self.method_metadata.method_type != "config" or use_model_results

        df_results = copy.deepcopy(self.model_results) if use_model_results else copy.deepcopy(self.hpo_results)

        if use_suite_in_prefix:
            if new_result_prefix is None:
                new_result_prefix = ""
            new_result_prefix = new_result_prefix + f"[{self.method_metadata.suite}] "
        if new_result_prefix is not None:
            df_results = self.add_prefix_to_results(results=df_results, prefix=new_result_prefix, inplace=True)

        return df_results

    @classmethod
    def add_prefix_to_results(cls, results: pd.DataFrame, prefix: str, inplace: bool = False) -> pd.DataFrame:
        if not inplace:
            results = results.copy()
        for col in ["method", "config_type", "ta_name", "ta_suite"]:
            if col in results:
                results[col] = prefix + results[col]
        return results

    def cache(self):
        self.method_metadata.to_yaml()
        if self.hpo_results is not None:
            save_pd.save(path=str(self.method_metadata.path_results_hpo()), df=self.hpo_results)
        if self.model_results is not None:
            save_pd.save(path=str(self.method_metadata.path_results_model()), df=self.model_results)

    @classmethod
    def concat(cls, results_lst: list[MethodResults]) -> Self:
        """Merge per-task results of one method into a single ``MethodResults``.

        The metadata merge resolves the disagreements that per-task inference can produce
        (see the ``config_default`` note below); any other metadata mismatch raises.
        """
        method_metadata = copy.deepcopy(results_lst[0].method_metadata)
        hpo_results = copy.deepcopy(results_lst[0].hpo_results)
        model_results = copy.deepcopy(results_lst[0].model_results)
        for results_other in results_lst[1:]:
            method_metadata_other = results_other.method_metadata
            hpo_results_other = results_other.hpo_results
            model_results_other = results_other.model_results

            # Capture the any() in metadata creation.
            if method_metadata.is_bag or method_metadata_other.is_bag:
                method_metadata.is_bag = True
                method_metadata_other = copy.deepcopy(method_metadata_other)
                method_metadata_other.is_bag = True

            if method_metadata.config_default != method_metadata_other.config_default:
                # The two sides disagree on the default config. Either one already spans multiple
                # configs (config_default None), or — when merging per-task results from partially
                # completed runs — different tasks each finished a single, *different* config, which
                # MethodMetadata.from_raw infers as that task's config_default. Across the merged
                # set the method therefore has multiple configs and no single default: defer it to
                # None (resolved later via get_config_default(use_first_if_missing=True)) and mark
                # the method HPO-capable. This matches what a single pass over all of the method's
                # results produces, and keeps the merge order-independent (a later single-config
                # task can never overwrite the deferred None).
                method_metadata.config_default = None
                method_metadata_other.config_default = None
                method_metadata.can_hpo = True
                method_metadata_other.can_hpo = True
            if method_metadata.can_hpo != method_metadata_other.can_hpo:
                method_metadata.can_hpo = True
                method_metadata_other.can_hpo = True
            if method_metadata.__dict__ != method_metadata_other.__dict__:
                diffs = {
                    k: (v, method_metadata_other.__dict__.get(k))
                    for k, v in method_metadata.__dict__.items()
                    if v != method_metadata_other.__dict__.get(k)
                }
                diff_str = "\n".join(f"  {k}: {v1!r} != {v2!r}" for k, (v1, v2) in diffs.items())
                raise ValueError(
                    f"Method metadata mismatch! The following fields differ:\n{diff_str}",
                )

            # merge results
            hpo_results_to_concat = [r for r in [hpo_results, hpo_results_other] if r is not None]
            if hpo_results_to_concat:
                hpo_results = pd.concat(hpo_results_to_concat, ignore_index=True)
            model_results_to_concat = [r for r in [model_results, model_results_other] if r is not None]
            if model_results_to_concat:
                model_results = pd.concat(model_results_to_concat, ignore_index=True)
        return cls(
            method_metadata=method_metadata,
            hpo_results=hpo_results,
            model_results=model_results,
        )
