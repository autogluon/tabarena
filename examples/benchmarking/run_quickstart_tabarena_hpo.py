from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tabarena.benchmark.experiment import ExperimentBatchRunner
from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from tabarena.website.website_format import format_leaderboard


if __name__ == '__main__':
    expname = str(Path(__file__).parent / "experiments" / "quickstart_hpo")  # folder location to save all experiment artifacts
    eval_dir = Path(__file__).parent / "eval" / "quickstart_hpo"
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    tabarena_context = TabArenaContext()
    task_metadata = tabarena_context.task_metadata

    # Sample for a quick demo
    datasets = ["anneal", "credit-g", "diabetes"]  # datasets = list(task_metadata["name"])
    folds = [0]

    # import your model search spaces
    from tabarena.models.lightgbm.generate import gen_lightgbm
    from tabarena.models.random_forest.generate import gen_randomforest

    # run the default config + 5 random configurations
    experiments_lightgbm = gen_lightgbm.generate_all_bag_experiments(num_random_configs=5)
    experiments_rf = gen_randomforest.generate_all_bag_experiments(num_random_configs=5)

    # This list of methods will be fit sequentially on each task (dataset x fold)
    methods = [
        *experiments_lightgbm,
        *experiments_rf,
    ]

    exp_batch_runner = ExperimentBatchRunner(expname=expname, task_metadata=task_metadata)

    # Get the run artifacts.
    # Fits each method on each task (datasets * folds)
    results_lst: list[dict[str, Any]] = exp_batch_runner.run(
        datasets=datasets,
        folds=folds,
        methods=methods,
        ignore_cache=ignore_cache,
    )

    # compute results
    end_to_end = EndToEnd.from_raw(results_lst=results_lst, task_metadata=task_metadata, cache=False, cache_raw=False)
    end_to_end_results = end_to_end.to_results()

    print(f"New Configs Hyperparameters: {end_to_end.configs_hyperparameters()}")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Results:\n{end_to_end_results.model_results.head(100)}")

    leaderboard: pd.DataFrame = end_to_end_results.compare_on_tabarena(
        output_dir=eval_dir,
        only_valid_tasks=True,  # True: only compare on tasks ran in `results_lst`
        use_model_results=False,  # If False: Will instead use the ensemble/HPO results
        new_result_prefix="Demo_",
    )
    leaderboard_website = format_leaderboard(df_leaderboard=leaderboard)
    print(leaderboard_website.to_markdown(index=False))
