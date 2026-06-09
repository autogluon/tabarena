from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tabarena.benchmark.experiment import AGModelOuterExperiment
from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.nips2025_utils.tabarena_context import TabArenaContext

if __name__ == "__main__":
    expname = str(
        Path(__file__).parent / "experiments" / "quickstart"
    )  # folder location to save all experiment artifacts
    eval_dir = Path(__file__).parent / "eval" / "quickstart_tabiclv2"

    context = TabArenaContext()

    # import your model classes
    from tabarena.models import TabICLv2Model

    # This list of methods will be fit sequentially on each task
    methods = [
        AGModelOuterExperiment(
            name="TabICLv2",
            model_cls=TabICLv2Model,
            model_hyperparameters={},
        ),
        AGModelOuterExperiment(
            name="TabICLv2_n1",
            model_cls=TabICLv2Model,
            model_hyperparameters={"n_estimators": 1},
        ),
    ]

    experiment_batch_runner = context.make_experiment_batch_runner(
        expname=expname,
        subset=["small", "lite"],
    )

    # Get the run artifacts. Fits each method on each configured task.
    results_lst: list[dict[str, Any]] = experiment_batch_runner.run_all(methods=methods)

    # compute results
    end_to_end = EndToEnd.from_raw(
        results_lst=results_lst,
        task_metadata=context.task_metadata_collection,
        cache=False,
        cache_raw=False,
    )
    end_to_end_results = end_to_end.to_results()

    print(f"New Configs Hyperparameters: {end_to_end.configs_hyperparameters()}")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Results:\n{end_to_end_results.model_results.head(100)}")

    new_results: pd.DataFrame = end_to_end_results.get_results(
        new_result_prefix="Demo_",
        use_model_results=True,  # If False: Will instead use the ensemble/HPO results
    )
    leaderboard: pd.DataFrame = context.compare(
        output_dir=eval_dir,
        only_valid_tasks=new_results["method"].unique(),  # only compare on tasks ran in `results_lst`
        new_results=new_results,
        verbose=False,
    )
    leaderboard_website = context.leaderboard_to_website_format(leaderboard)
    print(leaderboard_website.to_markdown(index=False))
