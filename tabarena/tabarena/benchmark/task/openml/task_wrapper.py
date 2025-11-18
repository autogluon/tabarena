from __future__ import annotations

import io
import logging
import math
import numpy as np
import pandas as pd

from openml.tasks.task import OpenMLSupervisedTask
from typing_extensions import Self

from autogluon.common.savers import save_pd, save_json
from autogluon.core.utils import generate_train_test_split

from .task_utils import get_task_data, get_ag_problem_type, get_task_with_retry
from ....utils.s3_utils import download_task_from_s3, upload_task_to_s3

logger = logging.getLogger(__name__)


class OpenMLTaskWrapper:
    def __init__(self, task: OpenMLSupervisedTask):
        assert isinstance(task, OpenMLSupervisedTask)
        self.task: OpenMLSupervisedTask = task
        self.X, self.y = get_task_data(task=self.task)
        self.problem_type = get_ag_problem_type(self.task)
        self.label = self.task.target_name

    @classmethod
    def from_task_id(cls, task_id: int) -> Self:
        task = get_task_with_retry(task_id=task_id)
        return cls(task)

    @property
    def task_id(self) -> int:
        return self.task.task_id

    @property
    def dataset_id(self) -> int:
        return self.task.dataset_id

    @property
    def eval_metric(self) -> str:
        metric_map = {
            "binary": "roc_auc",
            "multiclass": "log_loss",
            "regression": "root_mean_squared_error",
        }
        return metric_map[self.problem_type]

    def get_split_dimensions(self) -> tuple[int, int, int]:
        n_repeats, n_folds, n_samples = self.task.get_split_dimensions()
        return n_repeats, n_folds, n_samples

    def combine_X_y(self) -> pd.DataFrame:
        return pd.concat([self.X, self.y.to_frame(name=self.label)], axis=1)

    def save_data(self, path: str, file_type='.csv', train_indices=None, test_indices=None):
        data = self.combine_X_y()
        if train_indices is not None and test_indices is not None:
            train_data = data.loc[train_indices]
            test_data = data.loc[test_indices]
            save_pd.save(f"{path}train{file_type}", train_data)
            save_pd.save(f"{path}test{file_type}", test_data)
        else:
            save_pd.save(f"{path}data{file_type}", data)

    def save_metadata(self, path: str):
        metadata = dict(
            label=self.label,
            problem_type=self.problem_type,
            num_rows=len(self.X),
            num_cols=len(self.X.columns),
            task_id=self.task.task_id,
            dataset_id=self.task.dataset_id,
            openml_url=self.task.openml_url,
        )
        path_full = f"{path}metadata.json"
        save_json.save(path=path_full, obj=metadata)

    def get_split_indices(self, fold: int = 0, repeat: int = 0, sample: int = 0) -> tuple[np.ndarray, np.ndarray]:
        train_indices, test_indices = self.task.get_train_test_split_indices(fold=fold, repeat=repeat, sample=sample)
        return train_indices, test_indices

    def get_split_idx(self, fold: int = 0, repeat: int = 0, sample: int = 0) -> int:
        n_repeats, n_folds, n_samples = self.get_split_dimensions()
        assert fold < n_folds
        assert repeat < n_repeats
        assert sample < n_samples
        split_idx = n_folds * n_samples * repeat + n_samples * fold + sample
        return split_idx

    def split_vals_from_split_idx(self, split_idx: int) -> tuple[int, int, int]:
        n_repeats, n_folds, n_samples = self.get_split_dimensions()

        repeat = math.floor(split_idx / (n_folds * n_samples))
        remainder = split_idx % (n_folds * n_samples)
        fold = math.floor(remainder / n_samples)
        sample = remainder % n_samples

        assert fold < n_folds
        assert repeat < n_repeats
        assert sample < n_samples
        return repeat, fold, sample

    def get_train_test_split(
        self,
        fold: int = 0,
        repeat: int = 0,
        sample: int = 0,
        train_indices: np.ndarray = None,
        test_indices: np.ndarray = None,
        train_size: int | float = None,
        test_size: int | float = None,
        random_state: int = 0,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        if train_indices is None or test_indices is None:
            train_indices, test_indices = self.get_split_indices(fold=fold, repeat=repeat, sample=sample)
        X_train = self.X.loc[train_indices]
        y_train = self.y[train_indices]
        X_test = self.X.loc[test_indices]
        y_test = self.y[test_indices]

        if train_size is not None:
            X_train, y_train = self.subsample(X=X_train, y=y_train, size=train_size, random_state=random_state)
        if test_size is not None:
            X_test, y_test = self.subsample(X=X_test, y=y_test, size=test_size, random_state=random_state)

        return X_train, y_train, X_test, y_test

    @classmethod
    def to_csv_format(cls, X: pd.DataFrame) -> pd.DataFrame:
        """
        Converts X to the dtypes that it would have if it were saved to a CSV and then loaded.
        """
        s_buf = io.StringIO()
        X_index = X.index
        X.to_csv(s_buf, index=False)
        s_buf.seek(0)
        X = pd.read_csv(s_buf, low_memory=False)
        X.index = X_index
        return X

    def subsample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        size: int | float,
        random_state: int = 0,
    ) -> tuple[pd.DataFrame, pd.Series]:
        if isinstance(size, int) and size >= len(X):
            return X, y
        if isinstance(size, float) and size >= 1:
            return X, y
        X, _, y, _ = generate_train_test_split(
            X=X, y=y, problem_type=self.problem_type, train_size=size, random_state=random_state
        )
        return X, y

    def get_train_test_split_combined(
        self,
        fold: int = 0,
        repeat: int = 0,
        sample: int = 0,
        train_indices: np.ndarray = None,
        test_indices: np.ndarray = None,
        train_size: int | float = None,
        test_size: int | float = None,
        random_state: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        X_train, y_train, X_test, y_test = self.get_train_test_split(
            fold=fold,
            repeat=repeat,
            sample=sample,
            train_indices=train_indices,
            test_indices=test_indices,
            train_size=train_size,
            test_size=test_size,
            random_state=random_state,
        )
        train_data = pd.concat([X_train, y_train.to_frame(name=self.label)], axis=1)
        test_data = pd.concat([X_test, y_test.to_frame(name=self.label)], axis=1)
        return train_data, test_data

    def subsample_combined(
        self,
        data: pd.DataFrame,
        size: int | float,
        random_state: int = 0,
    ) -> pd.DataFrame:
        data, _ = self.subsample(X=data, y=data[self.label], size=size, random_state=random_state)
        return data


class OpenMLS3TaskWrapper(OpenMLTaskWrapper):
    """
    Class which uses S3 cache to download task splits.
    """
    @classmethod
    def from_task_id(cls, task_id: int, s3_dataset_cache: str) -> Self:
        assert s3_dataset_cache is not None
        download_task_from_s3(task_id, s3_dataset_cache=s3_dataset_cache)
        task = get_task_with_retry(task_id=task_id)
        return cls(task)

    @classmethod
    def update_s3_cache(cls, task_id: int, dataset_id: int, s3_dataset_cache: str):
        assert s3_dataset_cache is not None
        upload_task_to_s3(task_id=task_id, dataset_id=dataset_id, s3_dataset_cache=s3_dataset_cache)
