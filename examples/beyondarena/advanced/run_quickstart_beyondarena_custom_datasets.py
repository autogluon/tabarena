"""Quickstart: benchmark TabArena models on a custom dataset that comes from a Data Foundry container.

Flow:

1. **load** a Data Foundry container (default: the toy container shipped inside ``data_foundry``, so
   this runs offline with no download).
2. **convert** it to a persisted TabArena ``UserTask`` via ``convert_curated_container_to_user_task``
   (the same converter BeyondArena's own loader uses).
3. wrap it in a ``TaskMetadataCollection`` via ``from_user_tasks`` (which loads the task's metadata
   for you). Your own data has no BeyondArena baselines, so the context uses ``methods=[]`` and the
   leaderboard is computed purely from your results.
4. ``context.build_and_run_jobs(experiments)`` pairs each experiment with every split of the task and
   runs the sweep. No ``user_tasks=`` override is needed: a converted task is "standardized" (saved to
   the default cache), so the runner resolves it from its portable ``task_id_str`` — exactly like
   BeyondArena's own datasets.
5. ``compare`` computes the leaderboard from the registered methods.

Requires the optional ``data-foundry`` dependency (``tabarena[data-foundry]``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from data_foundry.examples import load_toy_container

from tabarena.benchmark.experiment import BeyondArenaExperimentBundle
from tabarena.benchmark.task.data_foundry import (
    DEFAULT_EVAL_METRICS,
    convert_curated_container_to_user_task,
)
from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.contexts import BeyondArenaContext

if TYPE_CHECKING:
    from data_foundry.curation_container import CuratedContainer

    from tabarena.benchmark.task import UserTask


def load_container() -> CuratedContainer:
    """Step 1 — the Data Foundry "load": return a curated container to benchmark on.

    Defaults to the tiny synthetic container shipped inside ``data_foundry`` (a binary-classification
    IID task), so the example runs offline with no download. Swap in any other container, e.g.::

        # A real dataset from the official BeyondArena collection (downloads via its HuggingFaceSource):
        from data_foundry.collections import BEYOND_ARENA
        return BEYOND_ARENA.get_dataset("airfoil_self_noise")

        # A container you already have on disk:
        from data_foundry.curation_container import CuratedContainer
        return CuratedContainer.load("/path/to/warehouse/<unique_name>/<uuid>")
    """
    return load_toy_container()


def container_to_task(container: CuratedContainer) -> UserTask:
    """Step 2 — the TabArena "convert": turn the container into a persisted ``UserTask``.

    ``convert_curated_container_to_user_task`` is the same converter BeyondArena's loader uses, so the
    container's target/problem-type are validated, the eval metric is resolved against TabArena's
    supported set (``DEFAULT_EVAL_METRICS``), the splits and split-regime columns are carried over, any
    text-embedding cache is imported, and the ``UserTask`` is saved to the (default) task cache.
    """
    return convert_curated_container_to_user_task(
        container=container,
        evaluation_metrics=DEFAULT_EVAL_METRICS,
    )


if __name__ == "__main__":
    # Output dirs, resolved next to this script so they don't depend on the working directory.
    here = Path(__file__).parent
    run_name = "quickstart_beyondarena_custom_datasets"
    results_dir = str(here / "experiments" / run_name)  # the runner's `expname` (results cache)
    eval_dir = here / "eval" / run_name  # leaderboard / figures `output_dir`

    # 1 + 2: load a Data Foundry container, then convert it to a persisted TabArena UserTask.
    container = load_container()
    task = container_to_task(container)

    # 3: a metadata collection straight from the task — `from_user_tasks` loads the task's metadata
    #    itself, so we only ever deal with the `UserTask`.
    task_collection = TaskMetadataCollection.from_user_tasks(task)

    # 4: models to run, each at its default config, through the BeyondArena bundle (TabArena
    #    preprocessing + the dynamic, split-regime-aware validation protocol that reads the
    #    container's stratify/group/time metadata). Registry names:
    #    `tabarena.models.utils.get_configs_generator_from_name`.
    experiments = BeyondArenaExperimentBundle(
        models=[
            ("LightGBM", 0),
            ("RandomForest", 0),
        ],
    ).build_experiments()

    # 5: a BeyondArena context over the custom collection. `methods=[]` -> no BeyondArena baselines
    #    (a brand-new dataset has none), so the leaderboard is computed purely from our own results;
    #    `fillna_method`/`calibration_method=None` for the same reason (their BeyondArena defaults
    #    reference baseline methods that aren't present here).
    #    Don't scope with `subset="core"` (that predicate is specific to the official BeyondArena
    #    suite); the size/problem-type/split-regime predicates (e.g. `subset=["tiny"]`) do work, as
    #    they read the collection's own columns.
    context = BeyondArenaContext(
        task_metadata=task_collection,
        methods=[],
        fillna_method=None,
        calibration_method=None,
    )

    # 6: build the jobs (experiments x the task's splits) and run + register them. No `user_tasks=`
    #    override is needed — the converted task is standardized, so the runner resolves it from the
    #    collection's `task_id_str` (like BeyondArena's own datasets).
    context.build_and_run_jobs(
        experiments,
        expname=results_dir,
        debug_mode=True,  # <-- also lets you attach a local debugger
    )

    # 7: compute the leaderboard from the registered methods.
    leaderboard = context.compare(output_dir=eval_dir)
    print("\n=== leaderboard ===")
    print(leaderboard.to_markdown())
    print(f"\nView saved figures in {eval_dir}")
