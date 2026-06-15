"""Benchmark a model on TabArena as an *outer* (no-bagging) experiment — shown with TabICLv2.

Same workflow as ``benchmarking/run_quickstart_tabarena.py`` (collection -> bundle ->
build_jobs -> run_jobs -> EndToEnd -> compare), but ``outer_experiments=True`` makes the
bundle fit each model directly on all the training data (an ``AGModelWrapper``: no train/val
split, bagging, or ensemble).
"""

from __future__ import annotations

from pathlib import Path

from tabarena.benchmark.experiment import (
    ExperimentBatchRunner,
    TabArenaV0pt1ExperimentBundle,
    build_jobs,
)
from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.models import TabICLv2Model
from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from tabarena.utils.config_utils import ConfigGenerator

if __name__ == "__main__":
    # Output dirs, resolved next to this script so they don't depend on the working directory.
    here = Path(__file__).parent
    run_name = "tabiclv2"
    results_dir = str(here / "experiments" / run_name)  # the runner's `expname` (results cache)
    eval_dir = here / "eval" / run_name  # leaderboard / figures `output_dir`

    # 1: TabArena tasks -> filter to the small datasets, first split (`lite` == r0f0)
    task_collection = TaskMetadataCollection.from_preset("TabArena-v0.1").subset_tasks(subset=["small", "lite"])

    # 2: build the experiments. `outer_experiments=True` emits no-validation `AGModelWrapper`
    #    fits (no bagging) for each model. Here we run two TabICLv2 configs: the default and an
    #    `n_estimators=1` variant. A `ConfigGenerator` with explicit `manual_configs` (and 0
    #    random configs) runs exactly those — yielding `TA-TabICLv2_c1` (default) and
    #    `TA-TabICLv2_c2` (n_estimators=1). For a registry model at its default, pass the name
    #    instead, e.g. `("TabICLv2", 0)`; see `run_quickstart_tabarena.py` for HPO / custom models.
    tabiclv2 = ConfigGenerator(
        search_space={},
        model_cls=TabICLv2Model,
        manual_configs=[{}, {"n_estimators": 1}],
    )
    bundle = TabArenaV0pt1ExperimentBundle(
        models=[(tabiclv2, 0)],
        outer_experiments=True,
    )
    experiments = bundle.build_experiments()

    # 3: experiments x the collection's splits -> a flat list of jobs.
    jobs = build_jobs(experiments, task_collection)

    # 4: run the jobs. `debug_mode=True` -> in-process native backend.
    runner = ExperimentBatchRunner(
        expname=results_dir,
        task_metadata=task_collection,
        debug_mode=True,
    )
    results_lst = runner.run_jobs(jobs)

    # 5: aggregate the raw results into a tidy per-(method, dataset, fold) frame.
    df_results = EndToEnd.from_raw_to_results_df(
        results_lst=results_lst,
        task_metadata=task_collection,
        new_result_prefix="[New] ",
    )
    print("\n=== raw per-fold results ===")
    print(df_results[["method", "dataset", "fold", "metric", "metric_error"]].to_string(index=False))

    # 6: compare against the cached TabArena leaderboard baselines (restricted to the tasks we ran).
    ta_context = TabArenaContext()
    leaderboard = ta_context.compare(
        output_dir=eval_dir,
        new_results=df_results,
        only_valid_tasks=df_results["method"].unique(),
    )
    leaderboard_website = ta_context.leaderboard_to_website_format(leaderboard=leaderboard)
    print("\n=== TabArena leaderboard ===")
    print(leaderboard_website.to_markdown(index=False))
