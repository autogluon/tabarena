from __future__ import annotations

import numpy as np
import pandas as pd


class NormalizedScorer:
    def __init__(
        self,
        df_results: pd.DataFrame,
        tasks: list[str],
        baseline: str | None = None,
        metric_error_col: str = "metric_error",
        task_col: str = "task",
        framework_col: str = "framework",
    ):
        """:param df_results: Dataframe of method performance containing columns `metric_error_col`,
        `dataset_col` and `framework_col`.
        :param tasks: tasks to consider
        """
        if isinstance(tasks[0], tuple):
            task_col = ["dataset", "fold"]
            all_tasks = df_results[task_col].drop_duplicates().values.tolist()
            all_tasks = {tuple(task) for task in all_tasks}
        else:
            assert all(col in df_results for col in [metric_error_col, task_col, framework_col])
            all_tasks = set(df_results[task_col].unique())
        for task in tasks:
            assert task in all_tasks, f"{task_col} {task} not present in passed evaluations"
        self.topline_dict = df_results.groupby(task_col)[metric_error_col].min().to_dict()
        if baseline is not None:
            assert baseline in df_results[framework_col].unique()
            self.baseline_dict = (
                df_results[df_results[framework_col] == baseline].groupby(task_col)[metric_error_col].min().to_dict()
            )
        else:
            self.baseline_dict = df_results.groupby(task_col)[metric_error_col].median(numeric_only=True).to_dict()

        # The same values again, as arrays behind a lookup index. `rank_many` then resolves a whole
        # column with one hash of the task keys, rather than hashing a tuple per row through a dict.
        keys = list(self.topline_dict)
        self._task_keys = pd.MultiIndex.from_tuples(keys) if keys and isinstance(keys[0], tuple) else pd.Index(keys)
        self._topline_values = np.array([self.topline_dict[key] for key in keys], dtype=float)
        self._baseline_values = np.array([self.baseline_dict.get(key, np.nan) for key in keys], dtype=float)

    # TODO rename to score, create parent class
    def rank(self, task: str, error: float) -> float:
        baseline = self.baseline_dict[task]
        topline = self.topline_dict[task]
        res = (error - topline) / np.clip(baseline - topline, a_min=1e-5, a_max=None)
        return np.clip(res, 0, 1)

    def rank_many(self, tasks: pd.Index, errors: np.ndarray) -> np.ndarray:
        """:meth:`rank` over whole columns: one score per (task, error) pair.

        `tasks` indexes the per-task baseline and topline, so it must carry one entry per error --
        a `MultiIndex` when tasks are (dataset, fold) pairs. Scoring row by row instead spends most
        of its time in `np.clip`'s dispatch for two scalars, which dwarfs the arithmetic.
        """
        codes = self._task_keys.get_indexer(tasks)
        unknown = codes == -1
        baseline = self._baseline_values[np.where(unknown, 0, codes)]
        topline = self._topline_values[np.where(unknown, 0, codes)]

        res = (np.asarray(errors, dtype=float) - topline) / np.clip(baseline - topline, a_min=1e-5, a_max=None)
        scores = np.clip(res, 0, 1)
        # A task the scorer never saw has no baseline to score against.
        return np.where(unknown, np.nan, scores)
