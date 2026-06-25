"""Native evaluation of a ``TabArenaBenchmarkPlan``'s results (TabArena v0.1 flow).

A benchmark run writes raw prediction artifacts to
``<output_dir>/data/<method>/<task>/<r{r}f{f}>/results/results.pkl``.
This module turns those raw artifacts into a TabArena leaderboard, reusing the same
``benchmark_name`` the plan was launched with to locate them.

It implements the TabArena-v0.1 flow: post-process the raw results into the TabArena cache
(shared with ``beyond_arena_eval`` via :mod:`tabarena.evaluation._eval_common`), then compare
against the TabArena-v0.1 paper baselines by registering the run's methods on a
``TabArenaContext`` (``extra_methods=``) and calling ``TabArenaContext.compare``. Each method
can be renamed in the leaderboard via its ``result_suffix`` (e.g. to distinguish a re-run).

For the data-foundry / BeyondArena flow (multiple runs, data-foundry subset predicates, no paper
baselines), see :mod:`tabarena.evaluation.beyond_arena_eval`.
"""

from __future__ import annotations

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
    result_suffix: str | None = None
    """Optional suffix appended to this method's name in the leaderboard, e.g.
    `" [Rerun]"` renders ``TabPFN-3`` as ``TabPFN-3 [Rerun]``. Useful to distinguish
    a re-run from the original TabArena baseline. Baked into the cached results at
    post-processing time (via the engine's ``name_suffix``); for ``only_load_cache``
    the cache must already have been built with the same suffix."""

    @property
    def ag_name(self) -> str:
        """AG name used as the raw-folder prefix and the cache method name."""
        from tabarena.evaluation._eval_common import resolve_ag_name

        return resolve_ag_name(self.name, self.ag_name_override)


@dataclass
class TabArenaEvalConfig:
    """Inputs for `run_eval`. Use the SAME `benchmark_name` as the plan."""

    benchmark_name: str
    """Matches the plan's benchmark name; also used as the cache ``suite``."""
    output_dir: str | Path
    """The run's output dir of TabArena runs."""
    methods: list[EvalMethod]
    """Methods to include in the leaderboard (alongside the TabArena baselines)."""
    figure_output_dir: str | Path
    """Where figures/leaderboards are written."""
    subsets: list[list[str]] | None = None
    """Each entry is a subset spec (e.g. ``["regression"]``); ``[]`` means the full
    benchmark. ``None`` is treated as ``[[]]`` (full only)."""
    num_cpus: int | None = None
    """CPUs for raw-result post-processing (None = all available)."""
    tabarena_cache_path: str | None = None
    """If set, used as the TabArena cache root (via ``set_tabarena_cache_root``)."""
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
        """Point TabArena/OpenML at the configured caches (resolved lazily, so order-independent)."""
        from tabarena.evaluation._eval_common import init_caches

        init_caches(self.tabarena_cache_path, self.openml_cache_path)


def _compare_subset(context, subset: list[str], *, figure_output_dir: Path):
    """Leaderboard for one subset: the run's registered methods vs the TabArena-v0.1 baselines.

    The comparison seam that differs from the BeyondArena flow (which uses data-foundry subset
    predicates in :mod:`tabarena.evaluation.beyond_arena_eval`). ``context`` already has the run's
    methods registered via ``extra_methods=``, so they flow through ``compare`` like cached baselines.
    """
    from tabarena.evaluation._eval_common import subset_label

    return context.compare(
        output_dir=figure_output_dir / "subsets" / subset_label(subset),
        subset=subset or None,
    )


def run_eval(config: TabArenaEvalConfig) -> dict[str, pd.DataFrame]:
    """Build a TabArena leaderboard per subset from raw results.

    Post-processes each non-cache-only method's raw artifacts into the TabArena
    cache (keyed by ``benchmark_name``, with its ``result_suffix`` baked into the
    method name), compares against the TabArena-v0.1 baselines, prints each
    leaderboard, and (by default) saves it as CSV.

    Returns ``{subset_label: leaderboard_df}``.
    """
    from tabarena.evaluation._eval_common import (
        MethodArtifact,
        post_process_to_results,
        save_leaderboard,
        subset_label,
    )
    from tabarena.website.website_format import format_leaderboard

    config.init_caches()

    artifacts = [
        MethodArtifact(
            ag_name=method.ag_name,
            path_raw=config.path_raw,
            suite=config.benchmark_name,
            result_suffix=method.result_suffix,
            only_load_cache=method.only_load_cache,
        )
        for method in config.methods
    ]
    results = post_process_to_results(artifacts, task_metadata=None, num_cpus=config.num_cpus)

    # Register the run's methods on a TabArena-v0.1 context (extra_methods=), so they flow through
    # `compare` against the paper baselines exactly like cached methods (fill of missing tasks is
    # handled by the context's fillna_method, as the old compare_on_tabarena did via get_results).
    from tabarena.contexts.tabarena_context import TabArenaContext

    context = TabArenaContext(extra_methods=results.to_method_metadata_lst())

    figure_output_dir = Path(config.figure_output_dir)
    leaderboards: dict[str, pd.DataFrame] = {}
    for subset in config.subsets_to_run():
        label = subset_label(subset)
        leaderboard = _compare_subset(context, subset, figure_output_dir=figure_output_dir)

        print(f"\n##### Leaderboard [{label}]")
        print(format_leaderboard(leaderboard).to_markdown(index=False))

        if config.save_leaderboards:
            save_leaderboard(leaderboard, figure_output_dir, label)
        leaderboards[label] = leaderboard

    return leaderboards
