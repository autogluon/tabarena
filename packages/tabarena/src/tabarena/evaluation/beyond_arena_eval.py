"""BeyondArena evaluation runner — supports comparing multiple benchmark runs in one leaderboard.

Counterpart to :func:`tabarena.evaluation.benchmark_eval.run_eval` (TabArena v0.1). Where v0.1
compares against the published paper baselines, BeyondArena builds its leaderboard purely from the
runs you point it at, filtered by the data-foundry subset predicates
(:attr:`~tabarena.contexts.beyondarena.context.BeyondArenaContext.SUBSET_PREDICATES`).

Each :class:`BenchmarkRun` is a completed benchmark output dir + its cache ``benchmark_name`` + the
models to pull from it. Because each run caches under its own ``benchmark_name`` (artifact name),
the *same* model (e.g. ``TabPFN-3``) can appear in several runs without colliding — which is how a
new TabPFN-3-only run is compared head-to-head with an older full-suite run.

Task metadata is loaded once from the self-contained committed BeyondArena reference CSV (see
:func:`~tabarena.evaluation.beyond_metadata.load_beyond_task_metadata_collection`) as a native
``TaskMetadataCollection``; its ``per_dataset_frame()`` doubles as the subset-filtering frame — no
warehouse merge needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from tabarena.benchmark.task.metadata import TaskMetadataCollection

#: Internal config-type -> display-name fixups for BeyondArena methods, applied on top of the
#: TabArena rename map. Mirrors the hardcoded map in the legacy ``run_eval.py``.
DEFAULT_METHOD_RENAME_MAP: dict[str, str] = {
    "TA-REALMLP": "RealMLP",
    "TA-TABM": "TabM",
    "TA-TABDPT": "TabDPT",
    "TA-TABPFN-2.6": "TabPFN-2.6",
    "TA-TABICLv2": "TabICLv2",
}

#: Default auxiliary problem_type -> metric map (adds an ``aux_metric_error`` column).
DEFAULT_AUX_METRIC_MAP: dict[str, str] = {
    "binary": "balanced_accuracy",
    "multiclass": "balanced_accuracy",
    "regression": "r2",
}


@dataclass
class BenchmarkRun:
    """One completed benchmark run to fold into the leaderboard."""

    benchmark_name: str
    """Cache artifact name for this run's results (distinct per run disambiguates shared methods)."""
    output_dir: str | Path
    """The run's output directory; raw ``results.pkl`` live under ``<output_dir>/data``."""
    models: list[str]
    """Model-registry names to include from this run (e.g. ``["TabPFN-3"]``)."""
    result_suffix: str | None = None
    """Optional suffix appended to every method name from this run, to distinguish it in the
    leaderboard (e.g. ``" [new]"``). Baked into the cache at post-process time."""
    only_load_cache: bool | list[str] = False
    """Which of this run's models load from cache vs. re-generate from raw results:

    * ``False`` (default): re-generate every model from raw.
    * ``True``: load every model from cache (skip raw post-processing).
    * ``list[str]``: only the named models load from cache; the rest are re-generated. Lets you
      keep expensive already-cached models as-is while regenerating just the ones you changed.
    """
    ag_name_overrides: dict[str, str] | None = None
    """Optional per-model AG-name overrides for custom methods not in the model registry."""

    def loads_from_cache(self, model: str) -> bool:
        """Whether ``model`` should be loaded from cache (vs. re-generated) for this run."""
        if isinstance(self.only_load_cache, bool):
            return self.only_load_cache
        return model in self.only_load_cache


@dataclass
class BeyondArenaEvalConfig:
    """Inputs for :func:`run_beyond_arena_eval`."""

    runs: list[BenchmarkRun]
    """Runs whose results are combined into one leaderboard (single-run = a list of one)."""
    figure_output_dir: str | Path
    """Where leaderboards / figures / ``dataset_subsets.json`` are written."""
    metadata_source: str | Path = "BeyondArena"
    """BeyondArena source name (committed package CSV) or a path to a ``*_tasks_metadata.csv``."""
    subsets_to_evaluate: list[list[str]] | None = None
    """Each entry is a subset spec (e.g. ``["regression"]``); ``[]`` = full. ``None`` = full only."""
    tabarena_cache_path: str | None = None
    openml_cache_path: str | None = None
    compute_aux_metric: bool = False
    """If True, compute the auxiliary metric (per ``aux_metric_map``) during raw post-processing,
    adding an ``aux_metric_error`` column. Off by default — it slows down post-processing."""
    aux_metric_map: dict[str, str] | None = field(default_factory=lambda: dict(DEFAULT_AUX_METRIC_MAP))
    """Auxiliary problem_type -> metric map; only published when ``compute_aux_metric=True``."""
    reference_model_name: str | None = "XGB (default)"
    """Calibration framework passed to ``compare`` (``calibration_framework``)."""
    imputed_model_name: str | None = "RF (default)"
    """Fillna method passed to ``compare`` (``fillna``)."""
    require_task_metadata_to_match: bool = True
    """If True, raise when df_results contains datasets missing from task_metadata; else filter."""
    method_rename_map_extra: dict[str, str] | None = None
    """Extra method-name fixups merged on top of the default rename map (e.g. a new run's
    config_type -> display name)."""
    num_cpus: int | None = None
    save_leaderboards: bool = True
    save_result_plots: bool = True
    """If True (and at least two subsets were evaluated), save the per-family / per-model
    overview plots across subsets under ``figure_output_dir/result_plots`` (see
    :func:`tabarena.plot.subset_results.plot_subset_results`)."""
    result_plot_metrics: tuple[str, ...] = ("elo", "improvability")
    """Metrics to render in the result plots (keys of ``METRIC_SPECS`` or custom specs)."""
    contender_models: list[str] | None = None
    """Methods highlighted as contenders in the result plots: drawn as their own standalone
    line in the per-family plot and star-marked in the per-model plot (e.g. ``["TabPFN-3"]``)."""

    def subsets_to_run(self) -> list[list[str]]:
        """Subset specs to evaluate; defaults to the full benchmark only."""
        return self.subsets_to_evaluate if self.subsets_to_evaluate is not None else [[]]

    def effective_aux_metric_map(self) -> dict[str, str] | None:
        """The aux-metric map to publish; ``None`` (disabled) unless ``compute_aux_metric``."""
        return self.aux_metric_map if self.compute_aux_metric else None


def run_beyond_arena_eval(config: BeyondArenaEvalConfig) -> dict[str, pd.DataFrame]:
    """Build a BeyondArena leaderboard per subset from one or more benchmark runs.

    Post-processes each run's models into the cache (keyed by the run's ``benchmark_name``),
    combines them, then for each subset filters via the data-foundry predicates and compares.

    Returns ``{subset_label: leaderboard_df}``.
    """
    from tabarena.evaluation._eval_common import (
        MethodArtifact,
        init_aux_metric_env,
        init_caches,
        post_process_to_results,
        resolve_ag_name,
    )
    from tabarena.evaluation.beyond_metadata import load_beyond_task_metadata_collection

    init_caches(config.tabarena_cache_path, config.openml_cache_path)
    # Passing None also clears a stale env var, so a previous run can't silently re-enable it.
    init_aux_metric_env(config.effective_aux_metric_map())

    # One self-contained native collection, shared by post-processing, compare, and subset filtering.
    task_metadata = load_beyond_task_metadata_collection(config.metadata_source)

    # Flatten (run, model) into post-processing specs; distinct suite per run keeps a
    # method that appears in multiple runs from colliding in the cache.
    artifacts = [
        MethodArtifact(
            ag_name=resolve_ag_name(model, (run.ag_name_overrides or {}).get(model)),
            path_raw=Path(run.output_dir) / "data",
            suite=run.benchmark_name,
            result_suffix=run.result_suffix,
            only_load_cache=run.loads_from_cache(model),
        )
        for run in config.runs
        for model in run.models
    ]
    results = post_process_to_results(artifacts, task_metadata=task_metadata, num_cpus=config.num_cpus)

    leaderboards = evaluate_beyond_subsets(
        df_results=results.get_results(),
        task_metadata=task_metadata,
        figure_output_dir=config.figure_output_dir,
        subsets=config.subsets_to_run(),
        fillna=config.imputed_model_name,
        calibration_framework=config.reference_model_name,
        method_rename_map_extra=config.method_rename_map_extra,
        require_task_metadata_to_match=config.require_task_metadata_to_match,
        save_leaderboards=config.save_leaderboards,
    )

    if config.save_result_plots and len(leaderboards) >= 2:
        from tabarena.plot.subset_results import plot_subset_results

        plot_subset_results(
            leaderboards,
            Path(config.figure_output_dir) / "result_plots",
            metrics=config.result_plot_metrics,
            contenders=config.contender_models or (),
        )

    return leaderboards


def evaluate_beyond_subsets(
    *,
    df_results: pd.DataFrame,
    task_metadata: TaskMetadataCollection,
    figure_output_dir: str | Path,
    subsets: list[list[str]],
    data_foundry_metadata: pd.DataFrame | None = None,
    fillna: str | None = "RF (default)",
    calibration_framework: str | None = "XGB (default)",
    method_rename_map: dict[str, str] | None = None,
    method_rename_map_extra: dict[str, str] | None = None,
    predicates=None,
    require_task_metadata_to_match: bool = True,
    save_leaderboards: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the per-subset compare/leaderboard loop for the data-foundry (BeyondArena) flow.

    ``df_results`` is filtered per subset via ``subset_tasks_data_foundry`` using
    ``data_foundry_metadata`` (defaults to ``task_metadata.per_dataset_frame()`` — self-contained,
    carries every predicate column incl. ``max_train_rows``) and the BeyondArena subset
    predicates, then compared (no paper baselines).

    Args:
        df_results: Combined per-fold results across all runs/methods.
        task_metadata: Native per-task metadata collection (the eval ``task_metadata``).
        figure_output_dir: Where ``subsets/<label>`` figures, ``dataset_subsets.json`` and the
            ``all_leaderboards`` CSVs are written.
        subsets: Subset specs to evaluate (``[]`` = full benchmark).
        data_foundry_metadata: Frame the subset predicates filter on; defaults to
            ``task_metadata.per_dataset_frame()``.
        fillna: Imputed-method name passed to ``compare`` (``fillna``).
        calibration_framework: Calibration-method name passed to ``compare``.
        method_rename_map: Full rename map; if None, built from ``TabArenaContext`` + the
            BeyondArena defaults (plus ``method_rename_map_extra``).
        method_rename_map_extra: Extra fixups merged onto the default rename map (ignored when
            ``method_rename_map`` is given).
        predicates: Subset predicates; defaults to ``BeyondArenaContext.SUBSET_PREDICATES``.
        require_task_metadata_to_match: Raise (vs. filter) when results contain datasets absent
            from ``task_metadata``.
        save_leaderboards: If True, write each subset's leaderboard CSV.

    Returns:
        ``{subset_label: leaderboard_df}``.
    """
    import json

    from tabarena.contexts.beyondarena.context import BeyondArenaContext
    from tabarena.contexts.tabarena.context import TabArenaContext
    from tabarena.evaluation._eval_common import save_leaderboard, subset_label
    from tabarena.nips2025_utils.compare import compare, get_subsets_per_dataset, subset_tasks_data_foundry
    from tabarena.website.website_format import format_leaderboard

    if data_foundry_metadata is None:
        data_foundry_metadata = task_metadata.per_dataset_frame()
    if predicates is None:
        predicates = BeyondArenaContext.SUBSET_PREDICATES
    if method_rename_map is None:
        method_rename_map = TabArenaContext().get_method_rename_map()
        method_rename_map = {**method_rename_map, **DEFAULT_METHOD_RENAME_MAP, **(method_rename_map_extra or {})}

    df_results = _align_results_to_task_metadata(
        df_results,
        task_metadata,
        require=require_task_metadata_to_match,
    )

    figure_output_dir = Path(figure_output_dir)
    figure_output_dir.mkdir(parents=True, exist_ok=True)

    # Record, per dataset present in the results, which subsets it belongs to.
    subsets_per_dataset = get_subsets_per_dataset(data_foundry_metadata, predicates=predicates)
    subsets_per_dataset = {ds: subsets_per_dataset.get(ds, []) for ds in sorted(df_results["dataset"].unique())}
    subsets_path = figure_output_dir / "dataset_subsets.json"
    with subsets_path.open("w") as f:
        json.dump(subsets_per_dataset, f, indent=2)
    print(f"Saved dataset subset memberships to: {subsets_path}")

    leaderboards: dict[str, pd.DataFrame] = {}
    for subset in subsets:
        subset_sorted = sorted(subset)
        label = subset_label(subset_sorted)
        print(f"\n\n############### Evaluating subset: {label}")

        leaderboard = compare(
            df_results=subset_tasks_data_foundry(
                df_results=df_results,
                subset=subset_sorted,
                data_foundry_metadata=data_foundry_metadata,
                predicates=predicates,
            ),
            output_dir=figure_output_dir / "subsets" / label,
            task_metadata=task_metadata,
            fillna=fillna,
            calibration_framework=calibration_framework,
            remove_imputed=False,
            method_rename_map=method_rename_map,
            add_dataset_count=True,
        )
        print(format_leaderboard(df_leaderboard=leaderboard).to_markdown(index=False))

        if save_leaderboards:
            # Apply the rename (incl. tuned/ensemble variants) to the saved method column.
            full_rename = {
                **method_rename_map,
                **{
                    f"{k} {suffix}": f"{v} {suffix}"
                    for k, v in method_rename_map.items()
                    for suffix in ["(tuned + ensemble)", "(tuned)", "(default)"]
                },
            }
            leaderboard["method"] = leaderboard["method"].map(full_rename).fillna(leaderboard["method"])
            save_leaderboard(leaderboard, figure_output_dir, label, subdir="all_leaderboards")
        leaderboards[label] = leaderboard

    return leaderboards


def _align_results_to_task_metadata(
    df_results: pd.DataFrame,
    task_metadata: TaskMetadataCollection,
    *,
    require: bool,
) -> pd.DataFrame:
    """Ensure every dataset in ``df_results`` has task metadata; raise or filter on mismatch."""
    import warnings

    task_metadata_datasets = set(task_metadata.dataset_names())
    results_datasets = set(df_results["dataset"].unique())
    missing = results_datasets - task_metadata_datasets
    if not missing:
        return df_results

    detail = (
        f"Number of datasets in task_metadata: {len(task_metadata_datasets)}\n"
        f"Number of datasets in df_results: {len(results_datasets)}\n"
        f"Number of missing datasets: {len(missing)}\n"
        f"Missing datasets: {sorted(missing)}"
    )
    if require:
        raise AssertionError(
            "Found datasets in df_results that are missing from task_metadata. This likely means "
            "task_metadata is incomplete or df_results contains results for datasets not part of "
            f"the current evaluation context.\n{detail}",
        )
    warnings.warn(
        f"Found datasets in df_results missing from task_metadata; filtering them out.\n{detail}",
        UserWarning,
        stacklevel=2,
    )
    return df_results[df_results["dataset"].isin(task_metadata_datasets)].copy()
