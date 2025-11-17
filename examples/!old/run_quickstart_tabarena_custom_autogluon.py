from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tabarena import EvaluationRepository, Evaluator
from tabarena.benchmark.experiment import ExperimentBatchRunner, Experiment
from tabarena.benchmark.models.wrapper import AGWrapper
from tabarena.benchmark.result import ExperimentResults
from tabarena.nips2025_utils.fetch_metadata import load_task_metadata
from tabarena.nips2025_utils.tabarena_context import TabArenaContext


class MyCustomAGWrapper(AGWrapper):
    """
    Edit this however you like to alter the functionality of AutoGluon,
    or replace AutoGluon entirely by instead subclassing `AbstractExecModel` instead of `AGWrapper.
    """
    def _fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None, **kwargs):
        from autogluon.tabular import TabularPredictor

        train_data = X.copy()
        train_data[self.label] = y

        fit_kwargs = self.fit_kwargs.copy()

        if X_val is not None:
            tuning_data = X_val.copy()
            tuning_data[self.label] = y_val
            fit_kwargs["tuning_data"] = tuning_data

        self.predictor = TabularPredictor(label=self.label, problem_type=self.problem_type, eval_metric=self.eval_metric, **self.init_kwargs)
        self.predictor.fit(train_data=train_data, **fit_kwargs)
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        y_pred = self.predictor.predict(X)
        return y_pred

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        y_pred_proba = self.predictor.predict_proba(X)
        return y_pred_proba


if __name__ == '__main__':
    expname = str(Path(__file__).parent / "experiments" / "quickstart_custom_ag")  # folder location to save all experiment artifacts
    repo_dir = str(Path(__file__).parent / "repos" / "quickstart_custom_ag")  # Load the repo later via `EvaluationRepository.from_dir(repo_dir)`
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    task_metadata = load_task_metadata()

    # Sample for a quick demo
    datasets = ["anneal", "credit-g", "diabetes"]  # datasets = list(task_metadata["name"])
    folds = [0]

    methods = [
        # Should match `GBM (default)` on the leaderboard
        Experiment(
            name="MyCustomAutoGluon",
            method_cls=MyCustomAGWrapper,
            method_kwargs={
                "fit_kwargs": dict(
                    time_limit=3600,
                    hyperparameters={"GBM": [{}]},
                    num_bag_folds=8,
                )
            }
        ),
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

    experiment_results = ExperimentResults(task_metadata=task_metadata)

    # Convert the run artifacts into an EvaluationRepository
    repo: EvaluationRepository = experiment_results.repo_from_results(results_lst=results_lst)
    repo.print_info()

    repo.to_dir(path=repo_dir)  # Load the repo later via `EvaluationRepository.from_dir(repo_dir)`

    print(f"New Configs Hyperparameters: {repo.configs_hyperparameters()}")

    # get the local results
    evaluator = Evaluator(repo=repo)
    metrics: pd.DataFrame = evaluator.compare_metrics().reset_index().rename(columns={"framework": "method"})

    # load the TabArena paper results
    tabarena_context = TabArenaContext()
    tabarena_results = tabarena_context.load_results_paper(download_results="auto")
    tabarena_results = tabarena_results[[c for c in tabarena_results.columns if c in metrics.columns]]

    dataset_fold_map = metrics.groupby("dataset")["fold"].apply(set)

    def is_in(dataset: str, fold: int) -> bool:
        return (dataset in dataset_fold_map.index) and (fold in dataset_fold_map.loc[dataset])

    # filter tabarena_results to only the dataset, fold pairs that are present in `metrics`
    is_in_lst = [is_in(dataset, fold) for dataset, fold in zip(tabarena_results["dataset"], tabarena_results["fold"])]
    tabarena_results = tabarena_results[is_in_lst]

    metrics = pd.concat([
        metrics,
        tabarena_results,
    ], ignore_index=True)

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Results:\n{metrics.head(100)}")

    calibration_framework = "RF (default)"

    from bencheval.tabarena import TabArena
    tabarena = TabArena(
        method_col="method",
        task_col="dataset",
        seed_column="fold",
        error_col="metric_error",
        columns_to_agg_extra=[
            "time_train_s",
            "time_infer_s",
        ],
        groupby_columns=[
            "metric",
            "problem_type",
        ]
    )

    leaderboard = tabarena.leaderboard(
        data=metrics,
        include_elo=True,
        elo_kwargs={
            "calibration_framework": calibration_framework,
        },
    )

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(leaderboard)
