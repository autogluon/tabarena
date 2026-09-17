from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

import boto3
from autogluon.common.loaders import load_json, load_pd
from autogluon.common.savers import save_json
from autogluon.common.utils.s3_utils import is_s3_url, s3_path_to_bucket_prefix
from botocore.errorfactory import ClientError

from tabarena.loaders import Paths, load_configs, load_results
from tabarena.repository.evaluation_repository import EvaluationRepository
from tabarena.simulation.dense_utils import intersect_folds_and_datasets, prune_zeroshot_gt
from tabarena.simulation.simulation_context import ZeroshotSimulatorContext
from tabarena.simulation.task_data import TASKS_FILENAME

if TYPE_CHECKING:
    import pandas as pd

    from tabarena.predictions.tabular_predictions import TabularModelPredictions
    from tabarena.simulation.ground_truth import GroundTruth


@dataclass
class BenchmarkPaths:
    configs: str
    baselines: str = None
    task_metadata: str = None
    metadata_join_column: str = "dataset"
    path_pred_proba: str = None
    datasets: list[str] = None
    zs_pp: list[str] = None
    zs_gt: list[str] = None
    configs_hyperparameters: list[str] = None
    relative_path: str = None

    def __post_init__(self):
        if self.zs_pp is not None and isinstance(self.zs_pp, str):
            self.zs_pp = [self.zs_pp]
        if self.zs_gt is not None and isinstance(self.zs_gt, str):
            self.zs_gt = [self.zs_gt]
        if self.configs_hyperparameters is not None and isinstance(self.configs_hyperparameters, str):
            self.configs_hyperparameters = [self.configs_hyperparameters]

    @property
    def configs_full(self):
        return self._to_full(self.configs)

    @property
    def baselines_full(self):
        return self._to_full(self.baselines)

    @property
    def zs_pp_full(self):
        return self._to_full_lst(self.zs_pp)

    @property
    def zs_gt_full(self):
        return self._to_full_lst(self.zs_gt)

    @property
    def configs_hyperparameters_full(self):
        return self._to_full_lst(self.configs_hyperparameters)

    @property
    def task_metadata_full(self):
        return self._to_full(self.task_metadata)

    @property
    def path_pred_proba_full(self):
        return self._to_full(self.path_pred_proba)

    def _to_full(self, path: str) -> str | None:
        if self.relative_path is None:
            return path
        if path is None:
            return None
        return self._join(str(Path(self.relative_path)), path)

    def _to_full_lst(self, paths: list[str] | None) -> list[str] | None:
        if self.relative_path is None:
            return paths
        if paths is None:
            return None
        root = str(Path(self.relative_path))
        return [self._join(root, path) for path in paths]

    @staticmethod
    def _join(root: str, path: str) -> str:
        """``str(Path(root) / path)`` without constructing a ``Path`` per element.

        A context lists one entry per task file (tens of thousands), and ``pathlib`` costs about
        13 us per join against under 1 us for ``os.path.join``, which made these properties a
        quarter of a method's load. The plain join is used for the clean relative paths the
        contexts contain; anything ``pathlib`` would normalize differently (absolute paths,
        ``.`` components, repeated or trailing separators) takes the ``Path`` route, so the
        result is identical in every case.
        """
        if (
            not path
            or path.startswith((os.sep, "." + os.sep))
            or path.endswith(os.sep)
            or (os.sep + os.sep) in path
            or (os.sep + "." + os.sep) in path
            or os.path.isabs(path)
        ):
            return str(Path(root) / path)
        if root in ("", ".") or root.endswith(os.sep):
            return str(Path(root) / path)
        return root + os.sep + path

    def print_summary(self):
        max_str_len = max(len(key) for key in self.__dict__)
        print("BenchmarkPaths Summary:")
        print("\n".join(f"\t{key + ' ' * (max_str_len - len(key))} = {value}" for key, value in self.__dict__.items()))

    def get_file_paths(self, include_zs: bool = True) -> list[str]:
        file_paths = [
            self.configs_full,
            self.baselines_full,
            self.task_metadata_full,
        ]
        if include_zs:
            file_paths += self.zs_pp_full
            file_paths += self.zs_gt_full
        return [f for f in file_paths if f is not None]

    def assert_exists_all(self, check_zs=True):
        self._raise_if_missing(self.missing_files(check_zs=check_zs))

    @staticmethod
    def _raise_if_missing(missing_files: list[str]):
        if missing_files:
            missing_files_str = "".join(f'\n\t"{m}"' for m in missing_files)
            raise ValueError(f"Missing {len(missing_files)} required files: [{missing_files_str}\n]")

    def exists_all(self, check_zs: bool = True) -> bool:
        return not self.missing_files(check_zs=check_zs)

    def missing_files(self, check_zs: bool = True) -> list:
        required_files = self.get_file_paths(include_zs=check_zs)
        return self._missing_files(required_files)

    @classmethod
    def _missing_files(cls, filepaths: list[str]) -> list[str]:
        """Return the subset of ``filepaths`` that does not exist, minimizing filesystem
        round trips: local paths are grouped by parent directory and checked with one
        ``os.scandir`` per directory instead of one ``stat`` per file. Artifacts hold
        thousands of per-task files, so on network filesystems (e.g. NFS) per-file stats
        dominate loading time. S3 paths keep the per-file check.
        """
        missing = []
        local_by_dir: dict[str, list[tuple[str, str]]] = {}
        for f in filepaths:
            if is_s3_url(path=f):
                if not cls.exists(f):
                    missing.append(f)
            else:
                p = Path(f)
                local_by_dir.setdefault(str(p.parent), []).append((p.name, f))
        for parent, entries in local_by_dir.items():
            try:
                with os.scandir(parent) as it:
                    names = {e.name for e in it}
            except (FileNotFoundError, NotADirectoryError):
                missing.extend(f for _, f in entries)
                continue
            missing.extend(f for name, f in entries if name not in names)
        return missing

    @staticmethod
    def exists(filepath: str) -> bool:
        if filepath is None:
            raise AssertionError("Filepath cannot be None!")
        filepath = str(filepath)

        if is_s3_url(path=filepath):
            s3_bucket, s3_prefix = s3_path_to_bucket_prefix(s3_path=filepath)
            s3 = boto3.client("s3")
            try:
                s3.head_object(Bucket=s3_bucket, Key=s3_prefix)
            except ClientError:
                return False
        elif not Path(filepath).exists():
            return False
        return True

    def load_results(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_configs, df_metadata = load_results(
            path_configs=self.configs_full,
            path_metadata=self.task_metadata_full,
            metadata_join_column=self.metadata_join_column,
            require_tid_in_metadata=self.task_metadata is not None,
        )
        return df_configs, df_metadata

    def load_baselines(self) -> pd.DataFrame | None:
        if self.baselines is None:
            return None
        return load_pd.load(self.baselines_full)

    def load_predictions(
        self,
        zsc: ZeroshotSimulatorContext,
        prediction_format: str = "memmap",
        verbose: bool = True,
        validate: bool = True,
    ) -> tuple[TabularModelPredictions, GroundTruth, ZeroshotSimulatorContext]:
        """:param prediction_format: Determines the format of the loaded tabular_predictions. Default = "memmap".
        "memmap": Fast and low memory usage.
        "memopt": Very fast and high memory usage.
        "mem": Slow and high memory usage, simplest format to debug.
        :param validate: If True, checks that all zs files exist before loading. Callers that
        have already validated (e.g. :meth:`BenchmarkContext.load`) pass False to skip the
        filesystem sweep.
        """
        if validate:
            self._raise_if_missing(self._missing_files(list(self.zs_pp_full) + list(self.zs_gt_full)))
        zeroshot_pred_proba, zeroshot_gt, zsc = load_zeroshot_input(
            path_pred_proba=self.path_pred_proba_full,
            paths_gt=self.zs_gt_full,
            paths_pp=self.zs_pp_full,
            zsc=zsc,
            datasets=self.datasets,
            prediction_format=prediction_format,
            verbose=verbose,
        )
        return zeroshot_pred_proba, zeroshot_gt, zsc

    def load_configs_hyperparameters(self) -> dict:
        return load_configs(self.configs_hyperparameters_full)

    def to_dict(self) -> dict:
        return asdict(self)


class BenchmarkContext:
    def __init__(
        self,
        *,
        folds: list[int],
        benchmark_paths: BenchmarkPaths,
        name: str | None = None,
        description: str | None = None,
        date: str | None = None,
        config_fallback: str | None = None,
    ):
        self.folds = folds
        self.benchmark_paths = benchmark_paths
        self.name = name
        self.description = description
        self.date = date
        self.config_fallback = config_fallback

    @classmethod
    def from_paths(
        cls,
        *,
        folds: list[int],
        name: str | None = None,
        description: str | None = None,
        date: str | None = None,
        config_fallback: str | None = None,
        **paths,
    ):
        return cls(
            folds=folds,
            name=name,
            description=description,
            date=date,
            config_fallback=config_fallback,
            benchmark_paths=BenchmarkPaths(**paths),
        )

    def load(
        self,
        folds: list[int] | None = None,
        load_predictions: bool = True,
        prediction_format: str = "memmap",
        verbose: bool = True,
        validate: bool = False,
    ) -> tuple[ZeroshotSimulatorContext, TabularModelPredictions, GroundTruth]:
        """:param folds: If None, uses self.folds as default.
            If specified, must be a subset of `self.folds`. This will filter the results to only the specified folds.
            Restricting folds can be useful to speed up experiments.
        :param load_predictions: If True, loads zpp and gt files.
        :param prediction_format: Determines the format of the loaded tabular_predictions. Default = "memmap".
            "memmap": Fast and low memory usage.
            "memopt": Very fast and high memory usage.
            "mem": Slow and high memory usage, simplest format to debug.
        :param validate: If True, list every task directory first and raise ``FileNotFoundError``
            naming all required files that are missing, before anything is loaded. Off by default:
            processed artifacts arrive as one unpacked archive, the listings cost seconds per
            artifact on network filesystems, and a missing file still fails, just later and one at a
            time (results and label files when they are read here, prediction ``.dat`` files when
            they are first memmapped at predict time).
        :return: Returns three objects in the following order:
            zsc: ZeroshotSimulatorContext
                The zeroshot simulator context object.
            zeroshot_pred_proba: TabularModelPredictions
                # TODO: Consider making a part of zsc.
                The prediction probabilities of all configs for all tasks.
                Will be None if `load_predictions=False`.
            zeroshot_gt : dict
                # TODO: Make its own object instead of a raw dict.
                # TODO: Consider making a part of zsc or zeroshot_pred_proba
                The target ground truth for both validation and test samples for all tasks.
                Will be None if `load_predictions=False`.

        Artifacts are fetched as a whole beforehand (``MethodMetadata.method_downloader``), so
        loading itself never downloads.
        """
        assert prediction_format in ["memmap", "memopt", "mem"]
        if folds is None:
            folds = self.folds
        for f in folds:
            assert f in self.folds, f"Fold {f} does not exist in available folds! self.folds={self.folds}"

        if verbose:
            print(
                f"Loading BenchmarkContext:\n"
                f"\tname: {self.name}\n"
                f"\tdescription: {self.description}\n"
                f"\tdate: {self.date}\n"
                f"\tfolds: {folds}"
            )
        if validate:
            missing_files = self.benchmark_paths.missing_files(check_zs=load_predictions)
            if missing_files:
                missing_files_str = [f'\n\t"{m}"' for m in missing_files]
                raise FileNotFoundError(
                    f"Missing {len(missing_files)} required files: \n[{','.join(missing_files_str)}\n]"
                )

        configs_hyperparameters = self.load_configs_hyperparameters()
        zsc = self._load_zsc(folds=folds, configs_hyperparameters=configs_hyperparameters, verbose=verbose)

        if load_predictions:
            zeroshot_pred_proba, zeroshot_gt, zsc = self._load_predictions(
                zsc=zsc, prediction_format=prediction_format, verbose=verbose
            )
        else:
            zeroshot_pred_proba = None
            zeroshot_gt = None

        return zsc, zeroshot_pred_proba, zeroshot_gt

    def load_repo(
        self,
        folds: list[int] | None = None,
        load_predictions: bool = True,
        prediction_format: str = "memmap",
        verbose: bool = True,
        validate: bool = False,
    ) -> EvaluationRepository:
        zsc, zeroshot_pred_proba, zeroshot_gt = self.load(
            folds=folds,
            load_predictions=load_predictions,
            prediction_format=prediction_format,
            verbose=verbose,
            validate=validate,
        )
        return EvaluationRepository(
            zeroshot_context=zsc,
            tabular_predictions=zeroshot_pred_proba,
            ground_truth=zeroshot_gt,
            config_fallback=self.config_fallback,
        )

    def _load_results(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_configs, df_metadata = self.benchmark_paths.load_results()
        return df_configs, df_metadata

    def load_configs_hyperparameters(self) -> dict:
        return self.benchmark_paths.load_configs_hyperparameters()

    def _load_predictions(
        self,
        zsc: ZeroshotSimulatorContext,
        prediction_format: str,
        verbose: bool = True,
    ) -> tuple[TabularModelPredictions, GroundTruth, ZeroshotSimulatorContext]:
        # validate=False: `load` runs the existence check itself when asked to (its `validate`).
        return self.benchmark_paths.load_predictions(
            zsc=zsc, prediction_format=prediction_format, verbose=verbose, validate=False
        )

    def _load_zsc(
        self, folds: list[int], configs_hyperparameters: dict, verbose: bool = True
    ) -> ZeroshotSimulatorContext:
        df_configs, df_metadata = self._load_results()

        # Load in real framework results to score against
        if verbose:
            print(f"Loading baselines: {self.benchmark_paths.baselines}")
        df_baselines = self.benchmark_paths.load_baselines()

        score_against_only_baselines = df_baselines is not None

        return ZeroshotSimulatorContext(
            df_configs=df_configs,
            folds=folds,
            df_baselines=df_baselines,
            df_metadata=df_metadata,
            score_against_only_baselines=score_against_only_baselines,
            configs_hyperparameters=configs_hyperparameters,
        )

    def to_json(self, path: str):
        output = {
            "name": self.name,
            "date": self.date,
            "description": self.description,
            "folds": self.folds,
            "config_fallback": self.config_fallback,
            "benchmark_paths": self.benchmark_paths.to_dict(),
        }
        save_json.save(path=path, obj=output)

    @classmethod
    def from_json(cls, path: str) -> Self:
        kwargs = load_json.load(path)
        # Contexts written before the per-file download logic was removed carry this key (always
        # null in processed artifacts).
        kwargs.pop("s3_download_map", None)
        kwargs["benchmark_paths"] = BenchmarkPaths(**kwargs["benchmark_paths"])
        return cls(**kwargs)


def construct_context(
    name: str | None,
    datasets: list[str],
    folds: list[int],
    local_prefix: str,
    description: str | None = None,
    date: str | None = None,
    task_metadata: str | None = None,
    local_prefix_is_relative: bool = True,
    has_baselines: bool = True,
    metadata_join_column: str = "dataset",
    configs_hyperparameters: list[str] | None = None,
    is_relative: bool = False,
    config_fallback: str | None = None,
    dataset_fold_lst_pp: list[tuple[str, int]] | None = None,
    dataset_fold_lst_gt: list[tuple[str, int]] | None = None,
) -> BenchmarkContext:
    """Parameters
    ----------
    name
    description
    date
    datasets
    folds
    local_prefix: str, default = None
        The directory holding the input files.
    task_metadata

    Returns:
    -------
    BenchmarkContext object that is able to load the data.
    """
    if local_prefix_is_relative:
        path_context = str(Paths.results_root_cache / local_prefix) + os.sep
    else:
        path_context = str(Path(local_prefix)) + os.sep

    data_root = Paths.data_root_cache if local_prefix_is_relative else Path(path_context).parent

    split_key = str(Path(path_context) / "model_predictions") + os.sep

    if dataset_fold_lst_pp is None:
        dataset_fold_lst_pp = [(dataset, fold) for dataset in datasets for fold in folds]
    if dataset_fold_lst_gt is None:
        dataset_fold_lst_gt = [(dataset, fold) for dataset in datasets for fold in folds]

    files_pred = ["pred-test.dat", "pred-val.dat"]
    _files_pp = [f"{dataset}/{fold}/{f}" for dataset, fold in dataset_fold_lst_pp for f in files_pred]

    # prediction metadata and labels of all folds live in one file per dataset
    _files_gt = [f"{dataset}/{TASKS_FILENAME}" for dataset in sorted({dataset for dataset, _ in dataset_fold_lst_gt})]

    if is_relative:
        zs_pp = [str(Path("model_predictions") / f) for f in _files_pp]
        zs_gt = [str(Path("model_predictions") / f) for f in _files_gt]
    else:
        zs_pp = [f"{split_key}{f}" for f in _files_pp]
        zs_pp = [Paths.rel_to_abs(k, relative_to=data_root) for k in zs_pp]
        zs_gt = [f"{split_key}{f}" for f in _files_gt]
        zs_gt = [Paths.rel_to_abs(k, relative_to=data_root) for k in zs_gt]

    if is_relative:
        _result_paths = dict(configs="configs.parquet")
    else:
        _result_paths = dict(
            configs=str(Path(path_context) / "configs.parquet"),
        )

    if has_baselines:
        if is_relative:
            _result_paths["baselines"] = "baselines.parquet"
        else:
            _result_paths["baselines"] = str(Path(path_context) / "baselines.parquet")

    if task_metadata is not None:
        if is_relative:
            _task_metadata_path = dict(task_metadata=task_metadata)
        else:
            _task_metadata_path = dict(task_metadata=str(Path(path_context) / task_metadata))
    else:
        _task_metadata_path = dict()

    if is_relative:
        split_key = str(Path("model_predictions")) + os.path.sep

    relative_path = str(Path(path_context)) if is_relative else None

    _bag_zs_path = dict(
        zs_gt=zs_gt,
        zs_pp=zs_pp,
        path_pred_proba=split_key,
    )

    _configs_hyperparameters_path = dict()
    if configs_hyperparameters is not None:
        _configs_hyperparameters_path["configs_hyperparameters"] = configs_hyperparameters

    context: BenchmarkContext = BenchmarkContext.from_paths(
        name=name,
        description=description,
        date=date,
        folds=folds,
        datasets=datasets,
        metadata_join_column=metadata_join_column,
        relative_path=relative_path,
        config_fallback=config_fallback,
        **_result_paths,
        **_bag_zs_path,
        **_task_metadata_path,
        **_configs_hyperparameters_path,
    )
    return context


def load_zeroshot_input(
    path_pred_proba: str,
    paths_gt: list[str],
    datasets: list[str],
    zsc: ZeroshotSimulatorContext,
    prediction_format: str = "memmap",
    verbose: bool = True,
    paths_pp: list[str] | None = None,
) -> tuple[TabularModelPredictions, GroundTruth, ZeroshotSimulatorContext]:
    """Load labels and prediction metadata for the context's tasks in one pass
    (:meth:`ZeroshotSimulatorContext.load_task_data`: one ``tasks.dat`` per dataset, or the
    per-task files), then the predictions. The tasks come from ``paths_pp`` (the context's
    per-task prediction files) when given, else from ``paths_gt``.
    """
    if verbose:
        print(
            f"Loading ZS inputs:\n\tpred_proba:  {path_pred_proba}\n",
        )
    task_data = zsc.load_task_data(paths_pp if paths_pp else paths_gt)
    zeroshot_gt = zsc.load_groundtruth(paths_gt=paths_gt, task_data=task_data)
    metadata_dict: dict[str, dict[int, dict]] = {}
    for (dataset, fold), task in task_data.items():
        metadata_dict.setdefault(dataset, {})[fold] = task.metadata
    zeroshot_pred_proba = zsc.load_pred(
        path_pred_proba=path_pred_proba,
        datasets=datasets,
        prediction_format=prediction_format,
        metadata_dict=metadata_dict,
    )

    # keep only dataset whose folds are all present
    intersect_folds_and_datasets(zsc, zeroshot_pred_proba, zeroshot_gt=zeroshot_gt)
    zeroshot_pred_proba.restrict_models(zsc.get_configs())
    zeroshot_gt = prune_zeroshot_gt(
        dataset_to_tid_dict=zsc.dataset_to_tid_dict, zeroshot_pred_proba=zeroshot_pred_proba, zeroshot_gt=zeroshot_gt
    )

    return zeroshot_pred_proba, zeroshot_gt, zsc
