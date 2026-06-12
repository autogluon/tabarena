"""End-to-end example: BeyondArena tasks through ``ExperimentBatchRunner``.

Counterpart to ``run_custom_datasets_via_experiment_batch_runner.py``: the same local
benchmark loop (run jobs -> aggregate -> leaderboard), but instead of authoring custom
``UserTask``s the tasks come from the BeyondArena (Data Foundry) suite, acquired exactly
the way the tabflow_slurm benchmark setup does it (see
``tabflow_slurm.setup.benchmark.TabArenaBenchmarkSetup.get_jobs_to_run``):

1. Load the suite's metadata with ``TaskMetadataCollection.from_preset`` (committed
   reference CSV, no downloads) and filter it with ``subset_tasks`` — here one tiny
   dataset on its first split (the ``-lite`` preset keeps ``r0f0``; the full suite has
   142 datasets x 20 repeats x 3 folds).
2. ``materialize()`` the surviving tasks: download them from the Data Foundry warehouse
   and convert each into a local OpenML ``UserTask`` pickle (only the filtered subset
   is fetched; cached after the first run).
3. Build the experiments from ``BeyondArenaExperimentBundle`` — the suite's config
   generation, with the compute resources baked in at build time.
4. Enumerate the sweep with the core ``build_jobs`` (experiments x the collection's
   splits; model constraints are respected during enumeration).
5. Run via ``ExperimentBatchRunner.run_jobs``. Unlike the custom-datasets example, no
   ``user_tasks=`` registration is needed: a materialized task's ``task_id_str`` is a
   serialized ``UserTask`` id (``"UserTask|..."``), which the runner auto-registers
   from the collection — the same resolution the SLURM compute nodes use (see
   ``tabflow_slurm.run_tabarena_experiment``).
6. Aggregate the raw results with ``EndToEnd`` and compute a leaderboard via
   ``AbstractArenaContext.compare``, as in the custom-datasets example.

The first run downloads the selected dataset(s) from the Data Foundry warehouse (needs
the ``data_foundry`` dependency and network access).

Run with::

    python examples/benchmarking/run_beyondarena_via_experiment_batch_runner.py
"""

from __future__ import annotations

from pathlib import Path

from tabarena.benchmark.experiment import (
    BeyondArenaExperimentBundle,
    ExperimentBatchRunner,
    build_jobs,
)
from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.nips2025_utils.abstract_arena_context import AbstractArenaContext
from tabarena.nips2025_utils.end_to_end import EndToEnd

if __name__ == "__main__":
    here = Path(__file__).parent
    results_dir = str(here / "experiments" / "beyondarena_ebr")
    eval_dir = here / "eval" / "beyondarena_ebr"

    # 1 + 2: suite metadata -> filter -> materialize. Scope to the smallest BeyondArena
    # dataset (155 rows, binary) so the demo is fast; add more names to widen it — only
    # the selected datasets are downloaded.
    task_collection = (
        TaskMetadataCollection.from_preset("BeyondArena-lite")
        .subset_tasks(dataset_names=["hepatitis_survival_prediction"])
        .materialize()
    )

    # 3: the suite's experiment bundle builds the configs (default + random per model),
    # baking in modest laptop resources. Same `models` format as a tabflow_slurm
    # `ModelJob`: ("RandomForest", 0) = default config only, ("Linear", 1) = default + 1
    # random config.
    bundle = BeyondArenaExperimentBundle(models=[("RandomForest", 0), ("Linear", 1)])
    experiments = bundle.build_experiments(
        time_limit=600,
        num_cpus=4,
        num_gpus=0,
        memory_limit=8,
        time_limit_with_preprocessing=False,
    )

    # 4: experiments x the collection's splits — the same enumerator the SLURM setup
    # uses. With the lite preset this is one (dataset, fold=0, repeat=0) split.
    jobs = build_jobs(experiments, task_collection)

    # 5: materialized UserTasks auto-resolve from the collection; nothing to register.
    runner = ExperimentBatchRunner(
        expname=results_dir,
        task_metadata=task_collection,
        debug_mode=True,
    )
    results_lst = runner.run_jobs(jobs)

    # 6: aggregate the raw results into a tidy per-(method, dataset, fold) frame.
    end_to_end = EndToEnd.from_raw(
        results_lst=results_lst,
        task_metadata=task_collection,
        cache=False,
        cache_raw=False,
        backend="native",
    )
    df_results = end_to_end.to_results().get_results()
    print("\n=== raw per-fold results ===")
    print(df_results[["method", "dataset", "fold", "metric", "metric_error"]].to_string(index=False))

    # Leaderboard via the context's `compare`, on the BeyondArena task metadata. With
    # `methods=[]` it contributes no baseline results, so the leaderboard is computed
    # purely from the results passed as `new_results`.
    context = AbstractArenaContext(
        task_metadata=task_collection,
        methods=[],
    )
    leaderboard = context.compare(output_dir=eval_dir, new_results=df_results)
    print("\n=== leaderboard ===")
    print(leaderboard.to_string())
