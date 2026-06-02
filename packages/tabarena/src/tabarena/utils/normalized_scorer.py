from typing import List
import numpy as np
import pandas as pd


class NormalizedScorer:
    def __init__(self,
                 df_results: pd.DataFrame,
                 tasks: List[str],
                 baseline: str = None,
                 metric_error_col: str = 'metric_error',
                 task_col: str = 'task',
                 framework_col: str = 'framework',
                 ):
        """
        :param df_results: Dataframe of method performance containing columns `metric_error_col`,
        `dataset_col` and `framework_col`.
        :param tasks: tasks to consider
        """
        if isinstance(tasks[0], tuple):
            task_col = ["dataset", "fold"]
            all_tasks = df_results[task_col].drop_duplicates().values.tolist()
            all_tasks = set([tuple(task) for task in all_tasks])
        else:
            assert all(col in df_results for col in [metric_error_col, task_col, framework_col])
            all_tasks = set(df_results[task_col].unique())
        for task in tasks:
            assert task in all_tasks, f"{task_col} {task} not present in passed evaluations"
        self.topline_dict = df_results.groupby(task_col)[metric_error_col].min().to_dict()
        if baseline is not None:
            assert baseline in df_results[framework_col].unique()
            self.baseline_dict = df_results[
                df_results[framework_col] == baseline
                ].groupby(task_col)[metric_error_col].min().to_dict()
        else:
            self.baseline_dict = df_results.groupby(task_col)[
                metric_error_col].median(numeric_only=True).to_dict()

    # TODO rename to score, create parent class
    def rank(self, task: str, error: float) -> float:
        baseline = self.baseline_dict[task]
        topline = self.topline_dict[task]
        res = (error - topline) / np.clip(baseline - topline, a_min=1e-5, a_max=None)
        return np.clip(res, 0, 1)
