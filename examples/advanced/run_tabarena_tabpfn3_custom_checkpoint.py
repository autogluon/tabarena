"""Run TabPFN-3 on TabArena from non-default checkpoints, as an outer (no-bagging) experiment."""

from __future__ import annotations

from pathlib import Path

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.models import TabPFN3Model
from tabarena.utils.config_utils import ConfigGenerator

if __name__ == "__main__":
    # Output dirs, resolved next to this script so they don't depend on the working directory.
    here = Path(__file__).parent
    run_name = "tabarena_tabpfn3_custom_checkpoint"
    results_dir = str(here / "experiments" / run_name)  # the runner's `expname` (results cache)
    eval_dir = here / "eval" / run_name  # leaderboard / figures `output_dir`

    # 1: a config generator for TabPFN-3 with non-default checkpoints. `checkpoint_per_problem_type`
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

    # 2: build the experiments. `outer_experiments=True` emits no-validation `AGModelWrapper` fits
    #    (no bagging) for each model.
    experiments = TabArenaV0pt1ExperimentBundle(
        models=[(tabpfn3, 0)],
        outer_experiments=True,
    ).build_experiments()

    # 3: the context is the hub. build_and_run_jobs scopes to the small datasets' first split
    #    (subset=["small", "lite"] == r0f0), pairs the config with each split, runs them locally,
    #    and registers the config as an in-memory method (pre-filtering task_metadata to the tasks
    #    just run, so `compare` scopes to them with nothing extra). debug_mode=True -> in-process
    #    native backend (also lets you attach a debugger).
    context = TabArenaContext()
    context.build_and_run_jobs(
        experiments,
        expname=results_dir,
        subset=["small", "lite"],
        new_result_prefix="[New] ",
        debug_mode=True,
    )

    # 4: compare against the cached TabArena baselines; the registered config is picked up
    #    automatically and carried into the website-format leaderboard with its metadata.
    leaderboard = context.compare(output_dir=eval_dir)
    leaderboard_website = context.leaderboard_to_website_format(leaderboard=leaderboard)
    print("\n=== TabArena leaderboard ===")
    print(leaderboard_website.to_markdown(index=False))
    print(f"\nView saved figures in {eval_dir}")
