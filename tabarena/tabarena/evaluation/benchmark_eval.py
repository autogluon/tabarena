"""Native evaluation of a ``TabArenaBenchmarkPlan``'s results.

A benchmark run writes raw prediction artifacts to
``<output_dir>/data/<method>/<task>/<r{r}f{f}>/results/results.pkl`` where
``output_dir == PathSetup.get_output_path(benchmark_name)``. This module turns
those raw artifacts into a TabArena leaderboard, reusing the same
``benchmark_name`` the plan was launched with to locate them.

Currently implements the TabArena-v0.1 flow: post-process the raw
results into the TabArena cache, then compare against the
TabArena-v0.1 paper baselines via ``EndToEndResults.compare_on_tabarena``.

Beyond-arena (future): ``_compare_subset`` is the extension seam. The general
path would accept a ``TabArenaMetadataBundle``, convert it to the eval
``task_metadata`` DataFrame via ``TabArenaTaskMetadata.to_dataframe(...)`` and
pass it to ``EndToEndSingle.from_path_raw_to_results``, then compare with a
data-foundry context (e.g. ``BeyondArenaContext`` + ``subset_tasks_data_foundry``)
instead of ``compare_on_tabarena``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class EvalMethod:
    """A method to include in the leaderboard, identified like a ``ModelJob``."""

    name: str
    """Model-registry name (the same name used in `ModelJob`), e.g. "RandomForest"."""
    only_load_cache: bool = False
    """If True, skip raw->cache post-processing and just load the existing cache."""
    ag_name_override: str | None = None
    """Use this AG name instead of deriving it from the registry (for custom methods
    not in `tabarena.models.utils.get_configs_generator_from_name`)."""

    @property
    def ag_name(self) -> str:
        """AG name used as the raw-folder prefix and the cache method name."""
        if self.ag_name_override is not None:
            return self.ag_name_override
        from tabarena.models.utils import get_configs_generator_from_name

        return get_configs_generator_from_name(self.name).model_cls.ag_name


@dataclass
class TabArenaEvalConfig:
    """Inputs for `run_eval`. Use the SAME `benchmark_name` as the plan."""

    benchmark_name: str
    """Matches the plan's benchmark name; also used as the cache ``artifact_name``."""
    output_dir: str | Path
    """The run's output dir, i.e. ``PathSetup.get_output_path(benchmark_name)``."""
    methods: list[EvalMethod]
    """Methods to include in the leaderboard (alongside the TabArena baselines)."""
    figure_output_dir: str | Path
    """Where figures/leaderboards are written."""
    subsets: list[list[str]] | None = None
    """Each entry is a subset spec (e.g. ``["regression"]``); ``[]`` means the full
    benchmark. ``None`` is treated as ``[[]]`` (full only)."""
    include_unverified: bool = True
    """Passed to ``TabArenaContext`` so unverified baselines are included."""
    num_cpus: int | None = None
    """CPUs for raw-result post-processing (None = all available)."""
    tabarena_cache_path: str | None = None
    """If set, exported as ``TABARENA_CACHE`` before the heavy imports."""
    openml_cache_path: str | None = None
    """If set, used as the OpenML root cache (for fetching task metadata)."""
    save_leaderboards: bool = True
    """If True, save each subset's leaderboard CSV under ``figure_output_dir``."""

    @property
    def path_raw(self) -> Path:
        """Directory holding the raw ``results.pkl`` artifacts."""
        return Path(self.output_dir) / "data"

    def subsets_to_run(self) -> list[list[str]]:
        """Subset specs to evaluate; defaults to the full benchmark only."""
        return self.subsets if self.subsets is not None else [[]]

    def init_caches(self) -> None:
        """Point TabArena/OpenML at the configured caches (call before heavy imports)."""
        if self.tabarena_cache_path is not None:
            os.environ["TABARENA_CACHE"] = self.tabarena_cache_path
            print("Set TABARENA_CACHE to:", self.tabarena_cache_path)
        if self.openml_cache_path is not None:
            import openml

            openml.config.set_root_cache_directory(str(Path(self.openml_cache_path).expanduser()))


def _subset_label(subset: list[str]) -> str:
    """Filesystem-friendly label for a subset spec (``[]`` -> ``"full"``)."""
    return "_".join(sorted(subset)) if subset else "full"


def _compare_subset(results, subset: list[str], *, config: TabArenaEvalConfig, figure_output_dir: Path):
    """Leaderboard for one subset (TabArena-v0.1 baselines via ``compare_on_tabarena``).

    Beyond-arena extension point: override/replace this to compare against a
    data-foundry context instead.
    """
    return results.compare_on_tabarena(
        output_dir=figure_output_dir / "subsets" / _subset_label(subset),
        subset=subset or None,
        tabarena_context_kwargs={"include_unverified": config.include_unverified},
    )


def run_eval(config: TabArenaEvalConfig) -> dict[str, pd.DataFrame]:
    """Build a TabArena leaderboard per subset from a plan's raw results.

    Post-processes each non-cache-only method's raw artifacts into the TabArena
    cache (keyed by ``benchmark_name``), compares against the TabArena-v0.1
    baselines, prints each leaderboard, and (by default) saves it as CSV.

    Returns ``{subset_label: leaderboard_df}``.
    """
    config.init_caches()

    # Imported here so TABARENA_CACHE (set above) is honored.
    from tabarena.nips2025_utils.end_to_end import EndToEndResults
    from tabarena.nips2025_utils.end_to_end_single import EndToEndSingle
    from tabarena.website.website_format import format_leaderboard

    singles = []
    for method in config.methods:
        if method.only_load_cache:
            single = EndToEndSingle.from_cache(method=method.ag_name, artifact_name=config.benchmark_name)
        else:
            print(f"Post-processing raw results for {method.name} (ag_name={method.ag_name})...")
            single = EndToEndSingle.from_path_raw_to_results(
                path_raw=config.path_raw,
                name_prefix_raw=method.ag_name,
                method=method.ag_name,
                artifact_name=config.benchmark_name,
                num_cpus=config.num_cpus,
            )
        singles.append(single)

    results = EndToEndResults(end_to_end_results_lst=singles)

    figure_output_dir = Path(config.figure_output_dir)
    leaderboards: dict[str, pd.DataFrame] = {}
    for subset in config.subsets_to_run():
        label = _subset_label(subset)
        leaderboard = _compare_subset(results, subset, config=config, figure_output_dir=figure_output_dir)

        print(f"\n##### Leaderboard [{label}]")
        print(format_leaderboard(leaderboard).to_markdown(index=False))

        if config.save_leaderboards:
            lb_dir = figure_output_dir / "leaderboards"
            lb_dir.mkdir(parents=True, exist_ok=True)
            leaderboard.to_csv(lb_dir / f"{label}.csv", index=False)
        leaderboards[label] = leaderboard

    return leaderboards
