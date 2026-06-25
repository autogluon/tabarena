"""Run TabPFN-3 on BeyondArena from non-default checkpoints, as an outer (no-bagging) experiment."""

from __future__ import annotations

from pathlib import Path

from tabarena.benchmark.experiment import BeyondArenaExperimentBundle
from tabarena.contexts import BeyondArenaContext
from tabarena.models import TabPFN3Model
from tabarena.utils.config_utils import ConfigGenerator

if __name__ == "__main__":
    here = Path(__file__).parent
    run_name = "beyondarena_tabpfn3_custom_checkpoint"
    results_dir = str(here / "experiments" / run_name)  # the runner's `expname`
    eval_dir = here / "eval" / run_name  # leaderboard `output_dir`

    # 1: suite metadata -> filter. `core` = the recommended protocol (each dataset's first
    #    `folds_to_use` splits); bounded to the tiny, non-high-dim datasets to keep it fast.
    subset = ["core", "tiny", "!high-dim"]

    # 2: a config generator for TabPFN-3 with non-default checkpoints. `checkpoint_per_problem_type`
    #    maps each problem type to a checkpoint (a bare filename resolved in the tabpfn cache dir, or
    #    an absolute path); TabPFN-3 picks the right one per task. Here we set the `classification`
    #    umbrella + `regression`, but you could split classification further into `binary` /
    #    `multiclass` (a more specific key wins). `search_space={}` + `n_configs=0` runs this one config.
    tabpfn3 = ConfigGenerator(
        search_space={},
        model_cls=TabPFN3Model,
        manual_configs=[
            {
                "checkpoint_per_problem_type": {
                    "classification": "tabpfn-v3-classifier-v3_20260417_multiclass.ckpt",
                    "regression": "tabpfn-v3-regressor-v3_20260417_mediumdata.ckpt",
                    # finer alternative: "binary": "...", "multiclass": "...",
                },
            },
        ],
    )

    # 3: build the experiments. `outer_experiments=True` emits no-validation `AGModelWrapper` fits
    #    (no bagging) for each model.
    experiments = BeyondArenaExperimentBundle(
        models=[(tabpfn3, 0)],
        outer_experiments=True,
    ).build_experiments()

    # 4: build_and_run_jobs scopes the context's BeyondArena task metadata to `subset`, pairs each
    #    config with each split, materializes the selected tasks, runs the model locally, and registers
    #    the results as in-memory methods (pre-filtering task_metadata to the tasks just run).
    context = BeyondArenaContext()
    context.build_and_run_jobs(
        experiments,
        expname=results_dir,
        subset=subset,
        new_result_prefix="[New] ",
    )

    # 5: compare against the cached BeyondArena baselines; the registered method is picked up
    #    automatically and the leaderboard is scoped to the tasks just run.
    leaderboard = context.compare(output_dir=eval_dir)
    print(leaderboard.to_markdown())
