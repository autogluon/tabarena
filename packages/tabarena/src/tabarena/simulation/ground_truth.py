from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from tabarena.simulation.label_files import as_label_array, widen_labels, write_labels_dat


class GroundTruth:
    """Per-task validation and test labels, as 1-D arrays in row order.

    Integer labels are kept in the narrowest integer dtype that holds them (see
    :func:`~tabarena.simulation.label_files.as_label_array`) and widened to int64 by
    :meth:`labels_val` / :meth:`labels_test`; floats keep their dtype.
    """

    def __init__(self, label_val_dict: dict[str, dict[int, object]], label_test_dict: dict[str, dict[int, object]]):
        """:param label_val_dict: dataset -> fold -> labels (array, Series or single-column DataFrame;
        normalized to arrays, row ids of a Series/DataFrame index are dropped)
        :param label_test_dict: same as `label_val_dict`
        """
        assert set(label_val_dict.keys()) == set(label_test_dict.keys())
        self._label_val_dict = {
            d: {f: as_label_array(v) for f, v in folds.items()} for d, folds in label_val_dict.items()
        }
        self._label_test_dict = {
            d: {f: as_label_array(v) for f, v in folds.items()} for d, folds in label_test_dict.items()
        }

    @property
    def datasets(self) -> list[str]:
        return sorted(self._label_val_dict.keys())

    def dataset_folds(self, dataset: str) -> list[int]:
        return sorted(self._label_val_dict[dataset].keys())

    def dataset_fold_lst(self) -> list[tuple[str, int]]:
        datasets = self.datasets
        dataset_fold_lst = []
        for dataset in datasets:
            dataset_folds = self.dataset_folds(dataset=dataset)
            for fold in dataset_folds:
                dataset_fold_lst.append((dataset, fold))
        return dataset_fold_lst

    # FIXME: Add restrict instead, same as tabular_predictions
    def remove_dataset(self, dataset: str):
        self._label_val_dict.pop(dataset)
        self._label_test_dict.pop(dataset)

    def restrict_datasets(self, datasets: list[str]):
        datasets = set(datasets)
        datasets_cur = self.datasets
        for dataset in datasets_cur:
            if dataset not in datasets:
                self.remove_dataset(dataset=dataset)

    def restrict_folds(self, folds: list[int]):
        folds = set(folds)
        dataset_fold_lst = self.dataset_fold_lst()
        for dataset, fold in dataset_fold_lst:
            if fold not in folds:
                self._label_val_dict[dataset].pop(fold)
                self._label_test_dict[dataset].pop(fold)

    def labels_val(self, dataset: str, fold: int) -> np.ndarray:
        return self._labels(self._label_val_dict[dataset][fold])

    def labels_test(self, dataset: str, fold: int) -> np.ndarray:
        return self._labels(self._label_test_dict[dataset][fold])

    @staticmethod
    def _labels(labels) -> np.ndarray:
        if not isinstance(labels, np.ndarray):
            # GroundTruth objects pickled before labels were stored as arrays hold DataFrames.
            labels = as_label_array(labels)
        return widen_labels(labels)

    # TODO: Unit test
    @classmethod
    def from_dict(cls, label_dict):
        label_val_dict = dict()
        label_test_dict = dict()
        for dataset in label_dict:
            label_val_dict[dataset] = dict()
            label_test_dict[dataset] = dict()
            for fold in label_dict[dataset]:
                labels_val = label_dict[dataset][fold]["y_val"]
                labels_test = label_dict[dataset][fold]["y_test"]
                label_val_dict[dataset][fold] = labels_val
                label_test_dict[dataset][fold] = labels_test
        return cls(label_val_dict=label_val_dict, label_test_dict=label_test_dict)

    # TODO: Unit test
    def to_data_dir(self, data_dir: str):
        if data_dir[:2] == "s3":
            from autogluon.common.utils.s3_utils import is_s3_url, s3_path_to_bucket_prefix, upload_s3_folder

            if is_s3_url(str(data_dir)):
                s3_bucket, s3_prefix = s3_path_to_bucket_prefix(data_dir)
                with tempfile.TemporaryDirectory() as temp_dir:
                    self.to_data_dir(data_dir=temp_dir)
                    upload_s3_folder(bucket=s3_bucket, prefix=s3_prefix, folder_to_upload=temp_dir)
                return

        for dataset in self.datasets:
            for fold in self._label_val_dict[dataset]:
                write_labels_dat(
                    Path(data_dir) / dataset / str(fold),
                    labels_val=self._label_val_dict[dataset][fold],
                    labels_test=self._label_test_dict[dataset][fold],
                )
