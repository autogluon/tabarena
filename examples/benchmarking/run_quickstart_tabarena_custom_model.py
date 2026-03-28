from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tabarena.benchmark.experiment import ExperimentBatchRunner, Experiment
from tabarena.benchmark.models.wrapper.abstract_class import AbstractExecModel
from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.nips2025_utils.tabarena_context import TabArenaContext


class SimpleLightGBM(AbstractExecModel):
    def __init__(self, hyperparameters: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        if hyperparameters is None:
            hyperparameters = {}
        self.hyperparameters = hyperparameters

    def get_model_cls(self):
        from lightgbm import LGBMClassifier, LGBMRegressor
        is_classification = self.problem_type in ['binary', 'multiclass']
        if is_classification:
            model_cls = LGBMClassifier
        elif self.problem_type == 'regression':
            model_cls = LGBMRegressor
        else:
            raise AssertionError(f"LightGBM does not recognize the problem_type='{self.problem_type}'")
        return model_cls

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        model_cls = self.get_model_cls()
        self.model = model_cls(**self.hyperparameters)
        self.model.fit(
            X=X,
            y=y
        )
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        y_pred = self.model.predict(X)
        return pd.Series(y_pred, index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        y_pred_proba = self.model.predict_proba(X)
        return pd.DataFrame(y_pred_proba, columns=self.model.classes_, index=X.index)


if __name__ == '__main__':
    expname = str(Path(__file__).parent / "experiments" / "quickstart_custom_model")  # folder location to save all experiment artifacts
    leaderboard_dir = str(Path(__file__).parent / "leaderboards" / "quickstart_custom_model")  # folder location to store tables, plots and figures
    ignore_cache = False  # set to True to overwrite existing caches and re-run experiments from scratch

    ta_context = TabArenaContext()
    task_metadata = ta_context.task_metadata.copy()

    # Sample for a quick demo
    datasets = ["anneal", "credit-g", "diabetes"]  # datasets = list(task_metadata["name"])
    folds = [0]

    methods = [
        Experiment(
            name="MyCustomModel",
            method_cls=SimpleLightGBM,
            method_kwargs={
                "hyperparameters": dict(
                    n_estimators=10,
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

    # compute results
    end_to_end = EndToEnd.from_raw(results_lst=results_lst, task_metadata=task_metadata, cache=False, cache_raw=False)
    end_to_end_results = end_to_end.to_results()

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Results:\n{end_to_end_results.model_results.head(100)}")

    leaderboard: pd.DataFrame = end_to_end_results.compare_on_tabarena(
        output_dir=leaderboard_dir,
        only_valid_tasks=True,  # True: only compare on tasks ran in `results_lst`
        plot_with_baselines=True,
    )
    leaderboard_website = ta_context.leaderboard_to_website_format(leaderboard=leaderboard)

    print(f"Leaderboard:")
    print(leaderboard_website.to_markdown(index=False))
    print(f"View saved figures in {leaderboard_dir}")
