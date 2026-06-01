"""Shared building blocks for the TabArena-v0.1 and BeyondArena evaluation runners.

Both runners follow the same skeleton — init caches/env, post-process each method's raw
artifacts into the cache (or load it), then run a per-subset compare/leaderboard loop. The
pieces that are identical across flavors live here so the two runners stay thin and consistent;
the *comparison* step differs (v0.1 compares against paper baselines via ``compare_on_tabarena``;
BeyondArena uses ``compare`` + data-foundry subset predicates) and is kept in each runner.

Heavy imports (``end_to_end*``, model registry) are deferred to call time so importing
``tabarena.evaluation`` stays cheap and free of the historical import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tabarena.loaders import set_tabarena_cache_root

if TYPE_CHECKING:
    import pandas as pd

    from tabarena.nips2025_utils.end_to_end import EndToEndResults


@dataclass
class MethodArtifact:
    """One method's raw-result location + cache identity, for post-processing.

    A flat, runner-agnostic spec: the v0.1 runner builds one per ``EvalMethod`` (all sharing the
    config's ``path_raw``/``benchmark_name``); the BeyondArena runner builds one per (run, model),
    so a single method name can appear under multiple ``artifact_name``s.
    """

    ag_name: str
    """The model's AutoGluon name — both the raw-folder prefix and the cache method name."""
    path_raw: Path
    """Directory holding the raw ``results.pkl`` artifacts (the run's ``<output_dir>/data``)."""
    artifact_name: str
    """Cache artifact name (upper cache dir); distinct values disambiguate the same method
    across different runs."""
    result_suffix: str | None = None
    """Optional suffix appended to the method name in the leaderboard (baked in at post-process)."""
    only_load_cache: bool = False
    """If True, skip raw->cache post-processing and load the existing cache instead."""


def init_caches(tabarena_cache_path: str | None = None, openml_cache_path: str | None = None) -> None:
    """Point TabArena/OpenML at the configured caches (resolved lazily, so order-independent)."""
    if tabarena_cache_path is not None:
        set_tabarena_cache_root(tabarena_cache_path)
        print("Set TabArena cache root to:", tabarena_cache_path)
    if openml_cache_path is not None:
        import openml

        openml.config.set_root_cache_directory(str(Path(openml_cache_path).expanduser()))
        print("Set OpenML cache root to:", openml_cache_path)


def init_aux_metric_env(aux_metric_map: dict[str, str] | None) -> None:
    """Publish/clear the auxiliary problem_type->metric map consumed during post-processing.

    When set, an ``aux_metric_error`` column is added to per-config and HPO/ensemble result rows.
    Passing ``None`` clears the env var (auxiliary metric disabled).
    """
    import json
    import os

    from tabarena.utils.aux_metric import AUX_METRIC_ENV_VAR

    if aux_metric_map is None:
        os.environ.pop(AUX_METRIC_ENV_VAR, None)
        return
    os.environ[AUX_METRIC_ENV_VAR] = json.dumps(aux_metric_map)
    print(f"Set {AUX_METRIC_ENV_VAR} to: {os.environ[AUX_METRIC_ENV_VAR]}")


def resolve_ag_name(name: str, ag_name_override: str | None = None) -> str:
    """Resolve a model-registry name (e.g. ``"TabPFN-3"``) to its AutoGluon name.

    ``ag_name_override`` short-circuits the registry lookup (for custom methods not registered in
    ``tabarena.models.utils.get_configs_generator_from_name``).
    """
    if ag_name_override is not None:
        return ag_name_override
    from tabarena.models.utils import get_configs_generator_from_name

    return get_configs_generator_from_name(name).model_cls.ag_name


def post_process_to_results(
    method_artifacts: list[MethodArtifact],
    *,
    task_metadata: pd.DataFrame | None = None,
    num_cpus: int | None = None,
) -> EndToEndResults:
    """Post-process each method's raw artifacts into the cache and collect them as results.

    For each artifact, either loads the existing cache (``only_load_cache``) or processes the raw
    ``results.pkl`` files into per-task results (caching them), then bundles all into a single
    :class:`EndToEndResults`. ``task_metadata`` is forwarded to the raw post-processing so custom
    (e.g. BeyondArena) task sets match correctly; pass ``None`` to infer it from the results.
    """
    from tabarena.nips2025_utils.end_to_end import EndToEndResults
    from tabarena.nips2025_utils.end_to_end_single import EndToEndSingle

    singles = []
    for ma in method_artifacts:
        if ma.only_load_cache:
            single = EndToEndSingle.from_cache(method=ma.ag_name, artifact_name=ma.artifact_name)
        else:
            print(f"Post-processing raw results for ag_name={ma.ag_name} (artifact={ma.artifact_name})...")
            single = EndToEndSingle.from_path_raw_to_results(
                path_raw=ma.path_raw,
                name_prefix_raw=ma.ag_name,
                name_suffix=ma.result_suffix,
                method=ma.ag_name,
                artifact_name=ma.artifact_name,
                task_metadata=task_metadata,
                num_cpus=num_cpus,
            )
        singles.append(single)
    return EndToEndResults(end_to_end_results_lst=singles)


def subset_label(subset: list[str]) -> str:
    """Filesystem-friendly label for a subset spec (``[]`` -> ``"full"``)."""
    return "_".join(sorted(subset)) if subset else "full"


def save_leaderboard(
    leaderboard: pd.DataFrame,
    figure_output_dir: str | Path,
    label: str,
    *,
    subdir: str = "leaderboards",
) -> Path:
    """Write a subset's leaderboard CSV under ``figure_output_dir/<subdir>/<label>.csv``."""
    lb_dir = Path(figure_output_dir) / subdir
    lb_dir.mkdir(parents=True, exist_ok=True)
    path = lb_dir / f"{label}.csv"
    leaderboard.to_csv(path, index=False)
    print("Saved leaderboard to:", path)
    return path
