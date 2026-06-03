from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tabarena.benchmark.experiment import AGModelBagExperiment
from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.nips2025_utils.tabarena_context import TabArenaContext

if __name__ == "__main__":
    expname = str(
        Path(__file__).parent / "experiments" / "quickstart"
    )  # folder location to save all experiment artifacts
    eval_dir = Path(__file__).parent / "eval" / "quickstart"
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    tabarena_context = TabArenaContext()
    task_metadata = tabarena_context.task_metadata

    # Sample for a quick demo
    datasets = ["anneal", "credit-g", "diabetes"]  # datasets = list(task_metadata["name"])

    # import your model classes
    from autogluon.tabular.models import LGBModel

    # This list of methods will be fit sequentially on each task (dataset x fold)
    methods = [
        # This will be a `config` in EvaluationRepository, because it computes out-of-fold predictions and thus can be used for post-hoc ensemble.
        AGModelBagExperiment(  # Wrapper for fitting a single bagged model via AutoGluon
            # The name you want the config to have
            name="LightGBM_c1_BAG_L1_Reproduced",
            # Supports any model that inherits from `autogluon.core.models.AbstractModel`
            model_cls=LGBModel,
            model_hyperparameters={
                # "ag_args_ensemble": {"fold_fitting_strategy": "sequential_local"},  # uncomment to fit folds sequentially, allowing for use of a debugger
            },  # The non-default model hyperparameters.
            num_bag_folds=8,  # num_bag_folds=8 was used in the TabArena 2025 paper
            time_limit=3600,  # time_limit=3600 was used in the TabArena 2025 paper
        ),
    ]

    # Build a runner scoped to the demo tasks via the TabArenaContext factory: the 3
    # datasets above at fold 0 / repeat 0. The "lite" subset keeps split 0
    # (== fold 0, repeat 0; split = n_folds * repeat + fold), and `datasets` restricts to
    # the demo datasets. `make_experiment_batch_runner` resolves these into the exact
    # (dataset, fold, repeat) triplets that `run_all` executes.
    exp_batch_runner = tabarena_context.make_experiment_batch_runner(
        expname=expname,
        datasets=datasets,
        subset="lite",
        cache_mode="ignore" if ignore_cache else "default",
    )

    # Get the run artifacts. Fits each method on each configured task.
    results_lst: list[dict[str, Any]] = exp_batch_runner.run_all(methods=methods)

    # compute results
    end_to_end = EndToEnd.from_raw(results_lst=results_lst, task_metadata=task_metadata, cache=False, cache_raw=False)
    end_to_end_results = end_to_end.to_results()

    print(f"New Configs Hyperparameters: {end_to_end.configs_hyperparameters()}")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Results:\n{end_to_end_results.model_results.head(100)}")

    new_results: pd.DataFrame = end_to_end_results.get_results(
        new_result_prefix="Demo_",
        use_model_results=True,  # If False: Will instead use the ensemble/HPO results
    )
    leaderboard: pd.DataFrame = tabarena_context.compare(
        output_dir=eval_dir,
        only_valid_tasks=new_results["method"].unique(),  # only compare on tasks ran in `results_lst`
        new_results=new_results,
    )
    leaderboard_website = tabarena_context.leaderboard_to_website_format(leaderboard)
    print(leaderboard_website.to_markdown(index=False))
