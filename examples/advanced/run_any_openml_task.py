"""Run TabArena models on ANY OpenML task — including tasks not in the TabArena suite.

These tasks have no TabArena baselines, so a generic ``AbstractArenaContext`` (``methods=[]``)
computes the leaderboard purely from your own results.
"""

from __future__ import annotations

from pathlib import Path

from tabarena.benchmark.experiment import (
    ExperimentBatchRunner,
    TabArenaV0pt1ExperimentBundle,
    build_jobs,
)
from tabarena.benchmark.task.metadata.fetch_metadata import task_metadata_collection_from_openml
from tabarena.contexts import AbstractArenaContext
from tabarena.end_to_end import EndToEnd

if __name__ == "__main__":
    # Output dirs, resolved next to this script so they don't depend on the working directory.
    here = Path(__file__).parent
    run_name = "any_openml_task"
    results_dir = str(here / "experiments" / run_name)  # the runner's `expname` (results cache)
    eval_dir = here / "eval" / run_name  # leaderboard / figures `output_dir`

    # OpenML task ids to benchmark on (any supervised classification/regression task).
    # List ids via: openml.tasks.list_tasks(output_format="dataframe").
    tids = [7592, 168908]

    # 1: build a TaskMetadataCollection for these OpenML tasks. `task_metadata_collection_from_openml`
    #    prefers TabArena's cached metadata and falls back to OpenML for any missing id
    #    (~1 min on first use). `subset_tasks` then restricts to exactly these tasks and to
    #    their first split (`lite` == r0f0), mirroring `run_quickstart_tabarena.py`.
    task_collection = task_metadata_collection_from_openml(tids=tids).subset_tasks(
        task_ids=tids,
        split_indices="lite",
    )

    # 2: models to run, each at its default config. See `run_quickstart_tabarena.py` for
    #    custom models + HPO. Registry names: `tabarena.models.utils.get_configs_generator_from_name`.
    bundle = TabArenaV0pt1ExperimentBundle(
        models=[
            ("LightGBM", 0),
            ("XGBoost", 0),
        ],
    )
    experiments = bundle.build_experiments()

    # 3: experiments x the collection's splits -> a flat list of jobs.
    jobs = build_jobs(experiments, task_collection)

    # 4: run the jobs. `debug_mode=True` -> in-process native backend. The runner downloads
    #    each OpenML task's data on first use.
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

    # 6: Custom leaderboard
    context = AbstractArenaContext(task_metadata=task_collection, methods=[])
    leaderboard = context.compare(output_dir=eval_dir, new_results=df_results)
    print("\n=== leaderboard ===")
    print(leaderboard.to_string())
