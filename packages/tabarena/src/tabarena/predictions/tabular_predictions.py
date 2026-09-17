from __future__ import annotations

import copy
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

# dictionary mapping the config name to predictions for a given dataset fold split
ConfigPredictionsDict = dict[str, np.array]

# dictionary mapping a particular fold of a dataset (a task) to split to config name to predictions
TaskPredictionsDict = dict[str, ConfigPredictionsDict]

# dictionary mapping the folds of a dataset to split to config name to predictions
DatasetPredictionsDict = dict[int, TaskPredictionsDict]

# dictionary mapping dataset to fold to split to config name to predictions
TabularPredictionsDict = dict[str, DatasetPredictionsDict]


class TabularModelPredictions:
    def __init__(self, datasets: list[str] | None = None):
        """Contains a collection of model evaluations, can be instantiated by either `TabularPredictionsInMemory`
        which contains all evaluations in memory with a dictionary or `TabularPredictionsMemmap` which stores
        all model evaluations on disk and retrieve them on the fly with memmap.
        :param datasets: if specified, only consider the given list of dataset.
        """
        if datasets:
            self.restrict_datasets(datasets)

    @classmethod
    def from_dict(
        cls, pred_dict: TabularPredictionsDict, datasets: list[str] | None = None, output_dir: str | None = None
    ):
        """Instantiates from a dictionary of predictions
        :param pred_dict:
        :param datasets: if specified, only consider the given list of dataset
        :param output_dir: directory where files are written (used for `TabularPredictionsMemmap`)
        :return:
        """
        raise NotImplementedError()

    def to_dict(self) -> TabularPredictionsDict:
        """:return: the whole dictionary of predictions"""
        raise NotImplementedError()

    @classmethod
    def from_data_dir(
        cls,
        data_dir: str | Path,
        datasets: list[str] | None = None,
        metadata_by_dir: dict[str, dict] | None = None,
        metadata_files: list[str | Path] | None = None,
        metadata_dict: dict[str, dict[int, dict]] | None = None,
    ):
        raise NotImplementedError()

    def to_data_dir(self, data_dir: str | Path, dtype: str = "float32"):
        # TODO: No need to convert to dict for TabularPredictionsMemmap, can just clone the files to the new directory instead to be more efficient.
        self._cache_data(pred_dict=self.to_dict(), data_dir=data_dir, dtype=dtype)

    @staticmethod
    def _cache_data(pred_dict: TabularPredictionsDict, data_dir: str, dtype: str = "float32"):
        assert dtype in ["float16", "float32"]

        if data_dir[:2] == "s3":
            from autogluon.common.utils.s3_utils import is_s3_url, s3_path_to_bucket_prefix, upload_s3_folder

            if is_s3_url(str(data_dir)):
                s3_bucket, s3_prefix = s3_path_to_bucket_prefix(data_dir)
                with tempfile.TemporaryDirectory() as temp_dir:
                    TabularModelPredictions._cache_data(pred_dict=pred_dict, data_dir=temp_dir, dtype=dtype)
                    upload_s3_folder(bucket=s3_bucket, prefix=s3_prefix, folder_to_upload=temp_dir)
                return

        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        for dataset, folds_dict in pred_dict.items():
            for fold, folds in folds_dict.items():
                target_folder = path_memmap(data_dir, dataset, fold)
                target_folder.mkdir(exist_ok=True, parents=True)

                # print(f"Converting {dataset} fold {fold} to {target_folder}")

                models = list(folds["pred_proba_dict_val"].keys())
                assert set(folds["pred_proba_dict_test"].keys()) == set(models), (
                    "different models available on validation and testing"
                )

                def get_split(split_key, models):
                    model_results = folds[split_key]  # noqa: B023
                    return np.array([model_results[model] for model in models])

                pred_val = get_split("pred_proba_dict_val", models)
                pred_test = get_split("pred_proba_dict_test", models)

                # Save metadata that are required to retrieve the predictions, in particular the shape of model
                # predictions and the model list
                with open(target_folder / "metadata.json", "w") as f:
                    metadata_dict = {
                        "models": models,
                        "dataset": dataset,
                        "fold": fold,
                        "pred_val_shape": pred_val.shape,
                        "pred_test_shape": pred_test.shape,
                        "dtype": dtype,
                    }

                    f.write(json.dumps(metadata_dict))

                # Dumps data to memmap tensors, alternatively could use .npy but it would make model loading much
                # slower in cases when only some models are loaded
                fp = np.memmap(str(target_folder / "pred-val.dat"), dtype=dtype, mode="w+", shape=pred_val.shape)
                fp[:] = pred_val[:]
                fp.flush()
                fp = np.memmap(str(target_folder / "pred-test.dat"), dtype=dtype, mode="w+", shape=pred_test.shape)
                fp[:] = pred_test[:]
                fp.flush()

    def predict_val(
        self, dataset: str, fold: int, models: list[str] | None = None, model_fallback: str | None = None
    ) -> np.array:
        """Obtains validation predictions on a given dataset and fold for a list of models
        :param: model_fallback to use for fallback if one model of `models` is not found
        :return: predictions with shape (num_models, num_rows, num_classes) for classification and
        (num_models, num_rows, ) for regression.
        """
        raise NotImplementedError()

    def predict_test(
        self, dataset: str, fold: int, models: list[str] | None = None, model_fallback: str | None = None
    ) -> np.array:
        """Obtains test predictions on a given dataset and fold for a list of models
        :param: model_fallback to use for fallback if one model of `models` is not found
        :return: predictions with shape (num_models, num_rows, num_classes) for classification and
        (num_models, num_rows, ) for regression.
        """
        raise NotImplementedError()

    @property
    def datasets(self) -> list[str]:
        """:return: list of datasets that are present in the collection"""
        return sorted(self.model_available_dict().keys())

    def restrict_datasets(self, datasets: list[str]):
        raise NotImplementedError()

    @property
    def folds(self) -> list[int]:
        """:return: list of folds that are present in all datasets"""
        all_folds = []
        for _dataset, fold_dict in self.model_available_dict().items():
            all_folds.append(fold_dict.keys())
        return list(set.intersection(*map(set, all_folds))) if all_folds else []

    def dataset_fold_lst(self) -> list[tuple[str, int]]:
        datasets = self.datasets
        models_available_dict = self.model_available_dict()
        dataset_fold_lst = []
        for dataset in datasets:
            dataset_folds = list(models_available_dict[dataset].keys())
            for fold in dataset_folds:
                dataset_fold_lst.append((dataset, fold))
        return dataset_fold_lst

    def restrict_folds(self, folds: list[int]):
        raise NotImplementedError()

    @property
    def models(self) -> list[str]:
        """:return: list of models that are present in all datasets and folds"""
        all_models = []
        for _dataset, fold_dict in self.model_available_dict().items():
            for _fold, models in fold_dict.items():
                all_models.append(models)
        return list(set.intersection(*map(set, all_models))) if all_models else []

    def models_exist(self) -> list[str]:
        """:return: list of models that are present in any task"""
        all_models = set()
        for _dataset, fold_dict in self.model_available_dict().items():
            for _fold, models in fold_dict.items():
                all_models = all_models.union(models)
        return sorted(all_models)

    def restrict_models(self, models: list[str]):
        raise NotImplementedError()

    def model_available_dict(self) -> dict[str, dict[int, list[str]]]:
        """:return: a dictionary listing all evaluations available mapping dataset to fold to list of models available."""
        model_available_dict = self._model_available_dict()
        return self._filter_empty(model_available_dict)

    def _filter_empty(self, model_available_dict):
        # remove all possibly empty collections from the given nested dictionaries
        res = model_available_dict
        for dataset, folds in model_available_dict.items():
            for fold, models in folds.items():
                if models:
                    res[dataset][fold] = models
            res[dataset] = {fold: models for fold, models in folds.items() if models}
        return {dataset: folds for dataset, folds in res.items() if folds}

    def _model_available_dict(self) -> dict[str, dict[int, list[str]]]:
        raise NotImplementedError()

    @staticmethod
    def _keep_only_models_in_both_validation_and_test(pred_dict: TabularPredictionsDict):
        """:return: a dictionary where all splits contains the same set of models"""
        for dataset, folds in pred_dict.items():
            for fold, splits in folds.items():
                models = set.intersection(*[set(model_dict.keys()) for model_dict in splits.values()])
                for split, split_dict in splits.items():
                    restrict = lambda model_dict, models: {k: v for k, v in model_dict.items() if k in models}
                    pred_dict[dataset][fold][split] = restrict(split_dict, models)


class TabularPredictionsInMemory(TabularModelPredictions):
    def __init__(self, pred_dict: TabularPredictionsDict, datasets: list[str] | None = None):
        # TODO assert that models are all both in validation and test set
        self.pred_dict = pred_dict
        super().__init__(datasets=datasets)

    @classmethod
    def from_dict(
        cls, pred_dict: TabularPredictionsDict, datasets: list[str] | None = None, output_dir: str | None = None
    ):
        # Optional, avoids changing passed object
        pred_dict = copy.deepcopy(pred_dict)
        cls._keep_only_models_in_both_validation_and_test(pred_dict)
        return cls(pred_dict=pred_dict, datasets=datasets)

    def to_dict(self) -> TabularPredictionsDict:
        return self.pred_dict

    @classmethod
    def from_data_dir(
        cls,
        data_dir: str | Path,
        datasets: list[str] | None = None,
        metadata_by_dir: dict[str, dict] | None = None,
        metadata_files: list[str | Path] | None = None,
        metadata_dict: dict[str, dict[int, dict]] | None = None,
    ):
        memmap = TabularPredictionsMemmap.from_data_dir(
            data_dir=data_dir,
            datasets=datasets,
            metadata_by_dir=metadata_by_dir,
            metadata_files=metadata_files,
            metadata_dict=metadata_dict,
        )
        return cls.from_dict(pred_dict=memmap.to_dict(), datasets=datasets)

    def predict_val(
        self, dataset: str, fold: int, models: list[str] | None = None, model_fallback: str | None = None
    ) -> np.array:
        assert model_fallback is None, (
            "config_fallback not supported for in memory data-structure, "
            "try saving repo via `repo.to_dir(path)`, "
            "then calling `repo = EvaluationRepository.from_dir(path, prediction_format='memmap')` to enable config_fallback"
        )
        return self._load_pred(dataset=dataset, fold=fold, models=models, split="val")

    def predict_test(
        self, dataset: str, fold: int, models: list[str] | None = None, model_fallback: str | None = None
    ) -> np.array:
        assert model_fallback is None, (
            "config_fallback not supported for in memory data-structure, "
            "try saving repo via `repo.to_dir(path)`, "
            "then calling `repo = EvaluationRepository.from_dir(path, prediction_format='memmap')` to enable config_fallback"
        )
        return self._load_pred(dataset=dataset, fold=fold, models=models, split="test")

    def _load_pred(self, dataset: str, split: str, fold: int, models: list[str] | None = None):
        if models is None:
            models = self.models

        def get_split(split, models):
            split_key = "pred_proba_dict_test" if split == "test" else "pred_proba_dict_val"
            model_results = self.pred_dict[dataset][fold][split_key]
            return np.array([self._get_model_results(model=model, model_pred_probas=model_results) for model in models])

        return get_split(split, models)

    # TODO: Improve exception logging if model is missing, refer to exception logging in MemMap version
    def _get_model_results(self, model: str, model_pred_probas: dict) -> np.array:
        return model_pred_probas[model]

    def restrict_datasets(self, datasets: list[str]):
        self.pred_dict = {dataset: fold_dict for dataset, fold_dict in self.pred_dict.items() if dataset in datasets}

    def restrict_folds(self, folds: list[int]):
        for dataset, fold_dict in self.pred_dict.items():
            self.pred_dict[dataset] = {fold: fold_info for fold, fold_info in fold_dict.items() if fold in folds}

    def restrict_models(self, models: list[str]):
        selected_models = set(models)
        for dataset, fold_dict in self.pred_dict.items():
            for fold, fold_info in fold_dict.items():
                for split, model_dict in fold_info.items():
                    self.pred_dict[dataset][fold][split] = {
                        model: v for model, v in model_dict.items() if model in selected_models
                    }

    def _model_available_dict(self) -> dict[str, dict[int, list[str]]]:
        return {
            dataset: {fold: list(fold_info["pred_proba_dict_val"].keys()) for fold, fold_info in fold_dict.items()}
            for dataset, fold_dict in self.pred_dict.items()
        }


def path_memmap(folder_memmap: Path, dataset: str, fold: int):
    return folder_memmap / dataset / str(fold)


class _TaskTable:
    """The per-task prediction metadata of a memmap store, one array per field.

    A store holds one ``metadata.json`` worth of information per task: the model list, the two
    prediction shapes and the dtype. As one dict per task that was 44k small dicts for a
    12-method collection, which took 0.26 s and 46 MB of private memory to unpickle in every
    ray worker. Here each field is an array over tasks (row order = load order), the distinct
    model lists and dtypes are stored once and referenced by code, and per-task model
    availability (what :meth:`TabularPredictionsMemmap.restrict_models` narrows) is a boolean
    mask over the task's model list. The ``(dataset, fold) -> row`` index is built on first use
    and never pickled.
    """

    _SHAPE_PAD = -1

    def __init__(
        self,
        datasets: list[str],
        dataset_code: np.ndarray,
        fold: np.ndarray,
        model_lists: list[list[str]],
        model_list_code: np.ndarray,
        available: np.ndarray,
        val_shape: np.ndarray,
        test_shape: np.ndarray,
        dtypes: list[str],
        dtype_code: np.ndarray,
    ):
        self.datasets = datasets
        self.dataset_code = dataset_code
        self.fold = fold
        self.model_lists = model_lists
        self.model_list_code = model_list_code
        self.available = available
        self.val_shape = val_shape
        self.test_shape = test_shape
        self.dtypes = dtypes
        self.dtype_code = dtype_code

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "_index"}

    def __setstate__(self, state):
        self.__dict__.update(state)

    @classmethod
    def from_metadata_dict(cls, metadata_dict: dict[str, dict[int, dict]]) -> _TaskTable:
        """Build from ``dataset -> fold -> {"models", "pred_val_shape", "pred_test_shape", "dtype"}``.

        ``models_all`` (the file's model order) is used when present, with ``models`` as the
        currently available subset; otherwise ``models`` is both.
        """
        datasets: list[str] = []
        dataset_index: dict[str, int] = {}
        model_lists: list[list[str]] = []
        model_list_index: dict[tuple[str, ...], int] = {}
        dtypes: list[str] = []
        dtype_index: dict[str, int] = {}
        dataset_code, fold, model_list_code, dtype_code = [], [], [], []
        val_shapes, test_shapes, availability = [], [], []
        for dataset, fold_dict in metadata_dict.items():
            for fold_id, task in fold_dict.items():
                models_all = tuple(task.get("models_all", task["models"]))
                if models_all not in model_list_index:
                    model_list_index[models_all] = len(model_lists)
                    model_lists.append(list(models_all))
                dtype = str(task["dtype"])
                if dtype not in dtype_index:
                    dtype_index[dtype] = len(dtypes)
                    dtypes.append(dtype)
                if dataset not in dataset_index:
                    dataset_index[dataset] = len(datasets)
                    datasets.append(dataset)
                models = task["models"]
                if "models_all" not in task or len(models) == len(models_all):
                    availability.append(None)  # every model available (the state a store loads in)
                else:
                    selected = set(models)
                    availability.append([m in selected for m in models_all])
                dataset_code.append(dataset_index[dataset])
                fold.append(int(fold_id))
                model_list_code.append(model_list_index[models_all])
                dtype_code.append(dtype_index[dtype])
                val_shapes.append(tuple(int(x) for x in task["pred_val_shape"]))
                test_shapes.append(tuple(int(x) for x in task["pred_test_shape"]))
        n = len(fold)
        width = max((len(m) for m in model_lists), default=0)
        available = np.zeros((n, width), dtype=bool)
        for i, row in enumerate(availability):
            if row is None:
                available[i, : len(model_lists[model_list_code[i]])] = True
            else:
                available[i, : len(row)] = row
        return cls(
            datasets=datasets,
            dataset_code=np.asarray(dataset_code, dtype=np.int32),
            fold=np.asarray(fold, dtype=np.int64),
            model_lists=model_lists,
            model_list_code=np.asarray(model_list_code, dtype=np.int32),
            available=available,
            val_shape=cls._pad_shapes(val_shapes),
            test_shape=cls._pad_shapes(test_shapes),
            dtypes=dtypes,
            dtype_code=np.asarray(dtype_code, dtype=np.int16),
        )

    @classmethod
    def _pad_shapes(cls, shapes: list[tuple[int, ...]]) -> np.ndarray:
        width = max((len(sh) for sh in shapes), default=0)
        out = np.full((len(shapes), width), cls._SHAPE_PAD, dtype=np.int64)
        for i, sh in enumerate(shapes):
            out[i, : len(sh)] = sh
        return out

    def __len__(self) -> int:
        return len(self.fold)

    @property
    def index(self) -> dict[str, dict[int, int]]:
        """``dataset -> fold -> row``."""
        index = self.__dict__.get("_index")
        if index is None:
            index = {}
            for row, (code, fold) in enumerate(zip(self.dataset_code.tolist(), self.fold.tolist(), strict=True)):
                index.setdefault(self.datasets[code], {})[fold] = row
            self._index = index
        return index

    def row(self, dataset: str, fold: int) -> int | None:
        folds = self.index.get(dataset)
        return None if folds is None else folds.get(fold)

    def models_all(self, row: int) -> list[str]:
        return self.model_lists[self.model_list_code[row]]

    def models(self, row: int) -> list[str]:
        models_all = self.models_all(row)
        mask = self.available[row]
        return [m for i, m in enumerate(models_all) if mask[i]]

    def shape(self, row: int, split: str) -> tuple[int, ...]:
        padded = self.val_shape[row] if split == "val" else self.test_shape[row]
        return tuple(int(x) for x in padded if x != self._SHAPE_PAD)

    def dtype(self, row: int) -> str:
        return self.dtypes[self.dtype_code[row]]

    def take(self, keep: np.ndarray) -> _TaskTable:
        """The table restricted to the rows where ``keep`` is True (order preserved)."""
        return _TaskTable(
            datasets=self.datasets,
            dataset_code=self.dataset_code[keep],
            fold=self.fold[keep],
            model_lists=self.model_lists,
            model_list_code=self.model_list_code[keep],
            available=self.available[keep],
            val_shape=self.val_shape[keep],
            test_shape=self.test_shape[keep],
            dtypes=self.dtypes,
            dtype_code=self.dtype_code[keep],
        )

    def restrict_models(self, models: list[str]) -> None:
        selected = set(models)
        for code, models_all in enumerate(self.model_lists):
            membership = np.zeros(self.available.shape[1], dtype=bool)
            membership[: len(models_all)] = [m in selected for m in models_all]
            rows = self.model_list_code == code
            self.available[rows] &= membership

    def dataset_names(self) -> list[str]:
        """Datasets with at least one row, in row order."""
        seen = dict.fromkeys(self.dataset_code.tolist())
        return [self.datasets[code] for code in seen]

    def to_metadata_dict(self) -> dict[str, dict[int, dict]]:
        """The legacy ``dataset -> fold -> metadata`` view (``models``, ``models_all``,
        ``model_indices``, ``pred_val_shape``, ``pred_test_shape``, ``dtype``), built fresh.
        """
        out: dict[str, dict[int, dict]] = {}
        for row in range(len(self)):
            models_all = self.models_all(row)
            out.setdefault(self.datasets[self.dataset_code[row]], {})[int(self.fold[row])] = {
                "models": self.models(row),
                "pred_val_shape": list(self.shape(row, "val")),
                "pred_test_shape": list(self.shape(row, "test")),
                "dtype": self.dtype(row),
                "models_all": models_all,
                "model_indices": {m: i for i, m in enumerate(models_all)},
            }
        return out


class TabularPredictionsMemmap(TabularModelPredictions):
    def __init__(
        self,
        data_dir: str | Path,
        datasets: list[str] | None = None,
        metadata_by_dir: dict[str, dict] | None = None,
        metadata_files: list[str | Path] | None = None,
        metadata_dict: dict[str, dict[int, dict]] | None = None,
    ):
        """:param data_dir: data where the predictions has been saved
        :param datasets: if specified, the predictions only contains those datasets
        :param metadata_by_dir: optional cache of parsed per-task ``metadata.json`` keyed by
            task directory; cached directories skip the file read
        :param metadata_files: the per-task ``metadata.json`` paths to load. When given, only
            these tasks are loaded and ``data_dir`` is not walked. ``None`` walks ``data_dir`` for
            ``*metadata.json``.
        :param metadata_dict: the per-task metadata already loaded, ``dataset -> fold -> {"models",
            "pred_val_shape", "pred_test_shape", "dtype"}`` (e.g. from a dataset's ``tasks.dat``, see
            :mod:`tabarena.simulation.task_data`); no metadata file is read when given.
        """
        self.data_dir = Path(data_dir)
        if metadata_dict is None:
            metadata_dict = self._load_metadatas(
                data_dir, metadata_by_dir=metadata_by_dir, metadata_files=metadata_files
            )
        self._table = _TaskTable.from_metadata_dict(metadata_dict)
        super().__init__(datasets=datasets)

    @property
    def metadata_dict(self) -> dict[str, dict[int, dict]]:
        """Per-task metadata as ``dataset -> fold -> {"models", "models_all", "model_indices",
        "pred_val_shape", "pred_test_shape", "dtype"}``. A view built from the task table on each
        access; mutate the store through ``restrict_*``, not through this dict.
        """
        return self._table.to_metadata_dict()

    @classmethod
    def from_data_dir(
        cls,
        data_dir: str | Path,
        datasets: list[str] | None = None,
        metadata_by_dir: dict[str, dict] | None = None,
        metadata_files: list[str | Path] | None = None,
        metadata_dict: dict[str, dict[int, dict]] | None = None,
    ):
        return cls(
            data_dir=data_dir,
            datasets=datasets,
            metadata_by_dir=metadata_by_dir,
            metadata_files=metadata_files,
            metadata_dict=metadata_dict,
        )

    @classmethod
    def from_dict(
        cls,
        pred_dict: TabularPredictionsDict,
        output_dir: str | None = None,
        datasets: list[str] | None = None,
        dtype: str = "float32",
    ):
        cls._keep_only_models_in_both_validation_and_test(pred_dict)
        cls._cache_data(pred_dict=pred_dict, data_dir=output_dir, dtype=dtype)
        return cls(data_dir=output_dir, datasets=datasets)

    def to_dict(self) -> TabularPredictionsDict:
        model_available_dict = self.model_available_dict()
        return {
            dataset: {
                fold: {
                    "pred_proba_dict_val": {
                        model: self.predict_val(dataset, fold, [model]).squeeze() for model in models
                    },
                    "pred_proba_dict_test": {
                        model: self.predict_test(dataset, fold, [model]).squeeze() for model in models
                    },
                }
                for fold, models in fold_dict.items()
            }
            for dataset, fold_dict in model_available_dict.items()
        }

    @staticmethod
    def _load_metadatas(
        data_dir,
        metadata_by_dir: dict[str, dict] | None = None,
        metadata_files: list[str | Path] | None = None,
    ) -> dict[str, dict[int, dict]]:
        if metadata_by_dir is None:
            metadata_by_dir = {}
        res = defaultdict(dict)
        if metadata_files is None:
            metadata_files = list(Path(data_dir).rglob("*metadata.json"))
        for metadata_file in map(Path, metadata_files):
            cached = metadata_by_dir.get(str(metadata_file.parent))
            if cached is not None:
                metadata = dict(cached)  # copy: `pop` must not mutate the cache
            else:
                with open(metadata_file) as f:
                    metadata = json.load(f)
            dataset = metadata.pop("dataset")
            fold = metadata.pop("fold")
            res[dataset][fold] = metadata
        return res

    def __setstate__(self, state):
        legacy = state.pop("metadata_dict", None)
        self.__dict__.update(state)
        if "_table" not in self.__dict__:
            # pickled before the task table: one metadata dict per task
            self._table = _TaskTable.from_metadata_dict(legacy or {})

    def predict_val(
        self, dataset: str, fold: int, models: list[str] | None = None, model_fallback: str | None = None
    ) -> np.array:
        return self._load_pred(dataset=dataset, fold=fold, models=models, split="val", model_fallback=model_fallback)

    def predict_test(
        self, dataset: str, fold: int, models: list[str] | None = None, model_fallback: str | None = None
    ) -> np.array:
        return self._load_pred(dataset=dataset, fold=fold, models=models, split="test", model_fallback=model_fallback)

    def _load_pred(
        self, dataset: str, split: str, fold: int, models: list[str] | None = None, model_fallback: str | None = None
    ):
        assert dataset in self._table.index, f"{dataset} not available."
        row = self._table.row(dataset, fold)
        assert row is not None, f"Fold {fold} of {dataset} not available."

        assert split in ["val", "test"]
        task_folder = path_memmap(folder_memmap=self.data_dir, dataset=dataset, fold=fold)
        models_all = self._table.models_all(row)
        mask = self._table.available[row]
        model_indices_available = {m: i for i, m in enumerate(models_all) if mask[i]}
        if model_fallback:
            # we use the model fallback if a model is not present
            models = [m if m in model_indices_available else model_fallback for m in models]
        try:
            model_indices = [model_indices_available[m] for m in models]
        except KeyError as e:
            missing_models = [m for m in models if m not in model_indices_available]
            missing_model_fallback = model_fallback is not None and model_fallback in missing_models
            if missing_model_fallback:
                raise Exception(
                    f"Tried to get predictions on dataset={dataset}, fold={fold}, split={split}, model_fallback={model_fallback} | "
                    f"The specified `model_fallback` does not exist for this task ..."
                    f"\n\tPlease specify a valid `model_fallback`.",
                ) from e
            raise Exception(
                f"Tried to get predictions on dataset={dataset}, fold={fold}, split={split}, model_fallback={model_fallback} | "
                f"Missing {len(missing_models)} out of {len(models)} requested model results for this task: {missing_models}"
                f"\n\tEither remove these models from the request or specify `model_fallback` to fill missing values.",
            ) from e
        pred = np.memmap(
            str(task_folder / f"pred-{split}.dat"),
            dtype=self._table.dtype(row),
            mode="r",
            shape=self._table.shape(row, split),
        )
        return pred[model_indices]

    def restrict_datasets(self, datasets: list[str]):
        keep_codes = {self._table.datasets.index(d) for d in set(datasets) if d in self._table.datasets}
        keep = (
            np.isin(self._table.dataset_code, list(keep_codes))
            if keep_codes
            else np.zeros(len(self._table), dtype=bool)
        )
        self._table = self._table.take(keep)

    def restrict_folds(self, folds: list[int]):
        self._table = self._table.take(np.isin(self._table.fold, list(set(folds))))

    def restrict_models(self, models: list[str]):
        self._table.restrict_models(models)

    def _model_available_dict(self) -> dict[str, dict[int, list[str]]]:
        table = self._table
        out: dict[str, dict[int, list[str]]] = {}
        datasets = table.datasets
        for row, (code, fold) in enumerate(zip(table.dataset_code.tolist(), table.fold.tolist(), strict=True)):
            out.setdefault(datasets[code], {})[fold] = table.models(row)
        return out
