
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple
import os
from pathlib import Path

import boto3
from botocore.errorfactory import ClientError
import pandas as pd
from typing_extensions import Self

from autogluon.common.loaders import load_pd, load_json
from autogluon.common.savers import save_json
from autogluon.common.utils.s3_utils import download_s3_files
from autogluon.common.utils.s3_utils import is_s3_url, s3_path_to_bucket_prefix

from .utils import load_zeroshot_input
from ..loaders import load_configs, load_results, Paths
from ..simulation.ground_truth import GroundTruth
from ..simulation.simulation_context import ZeroshotSimulatorContext
from ..predictions.tabular_predictions import TabularModelPredictions
from ..repository.evaluation_repository import EvaluationRepository
from ..utils import catchtime
from ..utils.huggingfacehub_utils import download_from_huggingface
from ..utils.download import download_files


def download_from_s3(name: str, include_zs: bool, exists: str, dry_run: bool, s3_download_map, benchmark_paths, verbose: bool):
    print(f'Downloading files for {name} context... '
          f'(include_zs={include_zs}, exists="{exists}", dry_run={dry_run})')
    if dry_run:
        print(f'\tNOTE: `dry_run=True`! Files will not be downloaded.')
    assert exists in ["raise", "ignore", "overwrite"]
    assert s3_download_map is not None, \
        f'self.s3_download_map is None: download functionality is disabled'
    file_paths_expected = benchmark_paths.get_file_paths(include_zs=include_zs)

    file_paths_to_download = [f for f in file_paths_expected if f in s3_download_map]
    if len(file_paths_to_download) == 0:
        print(f'WARNING: Matching file paths to download is 0! '
              f'`self.s3_download_map` probably has incorrect keys.')
    file_paths_already_exist = [f for f in file_paths_to_download if benchmark_paths.exists(f)]
    file_paths_missing = [f for f in file_paths_to_download if not benchmark_paths.exists(f)]

    if exists == 'raise':
        if file_paths_already_exist:
            raise AssertionError(f'`exists="{exists}"`, '
                                 f'and found {len(file_paths_already_exist)} files that already exist locally!\n'
                                 f'\tExisting Files: {file_paths_already_exist}\n'
                                 f'\tMissing  Files: {file_paths_missing}\n'
                                 f'Either manually inspect and delete existing files, '
                                 f'set `exists="ignore"` to keep your local files and only download missing files, '
                                 f'or set `exists="overwrite"` to overwrite your existing local files.')
    elif exists == 'ignore':
        file_paths_to_download = file_paths_missing
    elif exists == 'overwrite':
        file_paths_to_download = file_paths_to_download
    else:
        raise ValueError(f'Invalid value for exists (`exists="{exists}"`). '
                         f'Valid values: {["raise", "ignore", "overwrite"]}')

    s3_to_local_tuple_list = [(val, key) for key, val in s3_download_map.items()
                              if key in file_paths_to_download]

    log_extra = ''

    num_exist = len(file_paths_already_exist)
    if exists == 'overwrite':
        if num_exist > 0:
            log_extra += f'\tWill overwrite {num_exist} files that exist locally:\n' \
                         f'\t\t{file_paths_already_exist}'
        else:
            log_extra = f''
    if exists == 'ignore':
        log_extra += f'\tWill skip {num_exist} files that exist locally:\n' \
                     f'\t\t{file_paths_already_exist}'
    if file_paths_missing:
        if log_extra:
            log_extra += '\n'
        log_extra += f'Will download {len(file_paths_missing)} files that are missing locally:\n' \
                     f'\t\t{file_paths_missing}'

    if log_extra:
        print(log_extra)
    print(f'\tDownloading {len(s3_to_local_tuple_list)} files from s3 to local...')
    for s3_path, local_path in s3_to_local_tuple_list:
        print(f'\t\t"{s3_path}" -> "{local_path}"')
    s3_required_list = [(s3_path, local_path) for s3_path, local_path in s3_to_local_tuple_list if
                        s3_path[:2] == "s3"]
    urllib_required_list = [(s3_path, local_path) for s3_path, local_path in s3_to_local_tuple_list if
                            s3_path[:2] != "s3"]
    if urllib_required_list:
        download_files(remote_to_local_tuple_list=urllib_required_list, dry_run=dry_run, verbose=verbose)
    if s3_required_list:
        download_s3_files(s3_to_local_tuple_list=s3_required_list, dry_run=dry_run, verbose=verbose)


@dataclass
class BenchmarkPaths:
    configs: str
    baselines: str = None
    task_metadata: str = None
    metadata_join_column: str = "dataset"
    path_pred_proba: str = None
    datasets: List[str] = None
    zs_pp: List[str] = None
    zs_gt: List[str] = None
    configs_hyperparameters: List[str] = None
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
        return str(Path(self.relative_path) / path)

    def _to_full_lst(self, paths: list[str] | None) -> list[str] | None:
        if self.relative_path is None:
            return paths
        if paths is None:
            return None
        return [self._to_full(path) for path in paths]

    def print_summary(self):
        max_str_len = max(len(key) for key in self.__dict__.keys())
        print(f'BenchmarkPaths Summary:')
        print("\n".join(f'\t{key + " "*(max_str_len - len(key))} = {value}' for key, value in self.__dict__.items()))

    def get_file_paths(self, include_zs: bool = True) -> List[str]:
        file_paths = [
            self.configs_full,
            self.baselines_full,
            self.task_metadata_full,
        ]
        if include_zs:
            file_paths += self.zs_pp_full
            file_paths += self.zs_gt_full
        file_paths = [f for f in file_paths if f is not None]
        return file_paths

    def assert_exists_all(self, check_zs=True):
        self._assert_exists(self.configs_full, 'configs')
        if self.baselines is not None:
            self._assert_exists(self.baselines_full, 'baselines')
        if self.task_metadata is not None:
            self._assert_exists(self.task_metadata_full, 'task_metadata')
        if check_zs:
            if self.zs_pp is not None:
                for f in self.zs_pp_full:
                    self._assert_exists(f, f'zs_pp | {f}')
            if self.zs_gt is not None:
                for f in self.zs_gt_full:
                    self._assert_exists(f, f'zs_gt | {f}')

    @staticmethod
    def _assert_exists(filepath: str, name: str = None):
        if filepath is None:
            raise AssertionError(f'Filepath for {name} cannot be None!')

        if is_s3_url(path=filepath):
            s3_bucket, s3_prefix = s3_path_to_bucket_prefix(s3_path=filepath)
            s3 = boto3.client('s3')
            try:
                s3.head_object(Bucket=s3_bucket, Key=s3_prefix)
            except ClientError as e:
                if e.response['Error']['Code'] == "404":
                    # The key does not exist.
                    raise ValueError(f'Filepath for {name} does not exist in S3! '
                                     f'(filepath="{filepath}")')
                elif e.response['Error']['Code'] == "403":
                    raise ValueError(f'Filepath for {name} does not exist in S3 or you lack permissions to read! '
                                     f'(filepath="{filepath}")')
                else:
                    raise e
        else:
            if not Path(filepath).exists():
                raise ValueError(f'Filepath for {name} does not exist on local filesystem! '
                                 f'(filepath="{filepath}")')

    def exists_all(self, check_zs: bool = True) -> bool:
        required_files = self.get_file_paths(include_zs=check_zs)
        return all(self.exists(f) for f in required_files)

    def missing_files(self, check_zs: bool = True) -> list:
        required_files = self.get_file_paths(include_zs=check_zs)
        missing_files = []
        for f in required_files:
            if not self.exists(f):
                missing_files.append(f)
        return missing_files

    @staticmethod
    def exists(filepath: str) -> bool:
        if filepath is None:
            raise AssertionError(f'Filepath cannot be None!')
        filepath = str(filepath)

        if is_s3_url(path=filepath):
            s3_bucket, s3_prefix = s3_path_to_bucket_prefix(s3_path=filepath)
            s3 = boto3.client('s3')
            try:
                s3.head_object(Bucket=s3_bucket, Key=s3_prefix)
            except ClientError as e:
                return False
        else:
            if not Path(filepath).exists():
                return False
        return True

    def load_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        df_baselines = load_pd.load(self.baselines_full)
        return df_baselines

    def load_predictions(
        self,
        zsc: ZeroshotSimulatorContext,
        prediction_format: str = "memmap",
        verbose: bool = True,
    ) -> Tuple[TabularModelPredictions, GroundTruth, ZeroshotSimulatorContext]:
        """
        :param prediction_format: Determines the format of the loaded tabular_predictions. Default = "memmap".
            "memmap": Fast and low memory usage.
            "memopt": Very fast and high memory usage.
            "mem": Slow and high memory usage, simplest format to debug.
        """
        for f in self.zs_pp_full:
            self._assert_exists(f, f'zs_pp | {f}')
        for f in self.zs_gt_full:
            self._assert_exists(f, name=f'zs_gt | {f}')
        zeroshot_pred_proba, zeroshot_gt, zsc = load_zeroshot_input(
            path_pred_proba=self.path_pred_proba_full,
            paths_gt=self.zs_gt_full,
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
    def __init__(self,
                 *,
                 folds: List[int],
                 benchmark_paths: BenchmarkPaths,
                 name: str = None,
                 description: str = None,
                 date: str = None,
                 s3_download_map: Dict[str, str] = None,
                 config_fallback: str = None,
                 ):
        self.folds = folds
        self.benchmark_paths = benchmark_paths
        self.name = name
        self.description = description
        self.date = date
        self.s3_download_map = s3_download_map
        self.config_fallback = config_fallback

    @classmethod
    def from_paths(cls,
                   *,
                   folds: List[int],
                   name: str = None,
                   description: str = None,
                   date: str = None,
                   s3_download_map: Dict[str, str] = None,
                   config_fallback: str = None,
                   **paths):
        return cls(folds=folds,
                   name=name,
                   description=description,
                   date=date,
                   s3_download_map=s3_download_map,
                   config_fallback=config_fallback,
                   benchmark_paths=BenchmarkPaths(**paths))

    def download(self,
                 include_zs: bool = True,
                 exists: str = 'raise',
                 verbose: bool = True,
                 dry_run: bool = False,
                 use_s3: bool = True,
    ):
        """
        Downloads all BenchmarkContext required files from s3 to local disk.

        :param include_zs: If True, downloads zpp and gt files if they exist.
        :param exists: This determines the behavior of the file download.
            Options: ['ignore', 'raise', 'overwrite']
            If 'ignore': Will only download missing files and ignore files that already exist locally.
                Note: Does not guarantee local and remote files are identical.
            If 'raise': Will raise an exception if any local files exist. Otherwise, it will download all remote files.
                Guarantees alignment between local and remote files (at the time of download)
            If 'overwrite': Will download all remote files, overwriting any pre-existing local files.
                Guarantees alignment between local and remote files (at the time of download)
        :param dry_run: If True, will not download files, but instead log what would have been downloaded.
        """
        if use_s3:
            download_from_s3(
                name=self.name, include_zs=include_zs, exists=exists, dry_run=dry_run,
                s3_download_map=self.s3_download_map, benchmark_paths=self.benchmark_paths, verbose=verbose
            )
        else:
            if verbose:
                print(f'Downloading files for {self.name} context... '
                      f'(include_zs={include_zs}, exists="{exists}")')
            download_from_huggingface(
                datasets=self.benchmark_paths.datasets,
            )

    def load(self,
             folds: List[int] = None,
             load_predictions: bool = True,
             download_files: bool = True,
             prediction_format: str = "memmap",
             exists: str = 'ignore',
             use_s3: bool = True,
             verbose: bool = True,
             ) -> Tuple[ZeroshotSimulatorContext, TabularModelPredictions, GroundTruth]:
        """
        :param folds: If None, uses self.folds as default.
            If specified, must be a subset of `self.folds`. This will filter the results to only the specified folds.
            Restricting folds can be useful to speed up experiments.
        :param load_predictions: If True, loads zpp and gt files.
        :param download_files: If True, will download required files from s3 if they don't already exist locally.
        :param prediction_format: Determines the format of the loaded tabular_predictions. Default = "memmap".
            "memmap": Fast and low memory usage.
            "memopt": Very fast and high memory usage.
            "mem": Slow and high memory usage, simplest format to debug.
        :param exists: If download_files=True, this determines the behavior of the file download.
            Options: ['ignore', 'raise', 'overwrite']
            Refer to `self.download` for details.
        :return: Returns four objects in the following order:
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
        """
        assert prediction_format in ["memmap", "memopt", "mem"]
        if folds is None:
            folds = self.folds
        for f in folds:
            assert f in self.folds, f'Fold {f} does not exist in available folds! self.folds={self.folds}'

        if verbose:
            print(f'Loading BenchmarkContext:\n'
                  f'\tname: {self.name}\n'
                  f'\tdescription: {self.description}\n'
                  f'\tdate: {self.date}\n'
                  f'\tfolds: {folds}')
        if download_files and exists == 'ignore':
            if self.benchmark_paths.exists_all(check_zs=load_predictions):
                download_files = False
        if download_files:
            if verbose:
                self.benchmark_paths.print_summary()
            if self.s3_download_map is None:
                missing_files = self.benchmark_paths.missing_files()
                if missing_files:
                    missing_files_str = [f'\n\t"{m}"' for m in missing_files]
                    raise FileNotFoundError(f'Missing {len(missing_files)} required files: \n[{",".join(missing_files_str)}\n]')
            if verbose:
                print(f'Downloading input files from s3...')
            self.download(include_zs=load_predictions, exists=exists, use_s3=use_s3, verbose=verbose)
        self.benchmark_paths.assert_exists_all(check_zs=load_predictions)

        configs_hyperparameters = self.load_configs_hyperparameters()
        zsc = self._load_zsc(folds=folds, configs_hyperparameters=configs_hyperparameters, verbose=verbose)

        if load_predictions:
            zeroshot_pred_proba, zeroshot_gt, zsc = self._load_predictions(zsc=zsc, prediction_format=prediction_format, verbose=verbose)
        else:
            zeroshot_pred_proba = None
            zeroshot_gt = None

        return zsc, zeroshot_pred_proba, zeroshot_gt

    def load_repo(
        self,
        folds: List[int] = None,
        load_predictions: bool = True,
        download_files: bool = True,
        prediction_format: str = "memmap",
        exists: str = 'ignore',
        use_s3: bool = True,
        verbose: bool = True,
    ) -> EvaluationRepository:
        zsc, zeroshot_pred_proba, zeroshot_gt = self.load(
            folds=folds,
            load_predictions=load_predictions,
            download_files=download_files,
            prediction_format=prediction_format,
            exists=exists,
            use_s3=use_s3,
            verbose=verbose,
        )
        repo = EvaluationRepository(
            zeroshot_context=zsc,
            tabular_predictions=zeroshot_pred_proba,
            ground_truth=zeroshot_gt,
            config_fallback=self.config_fallback,
        )
        return repo

    def _load_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_configs, df_metadata = self.benchmark_paths.load_results()
        return df_configs, df_metadata

    def load_configs_hyperparameters(self) -> dict:
        return self.benchmark_paths.load_configs_hyperparameters()

    def _load_predictions(
        self,
        zsc: ZeroshotSimulatorContext,
        prediction_format: str,
        verbose: bool = True,
    ) -> Tuple[TabularModelPredictions, GroundTruth, ZeroshotSimulatorContext]:
        return self.benchmark_paths.load_predictions(zsc=zsc, prediction_format=prediction_format, verbose=verbose)

    def _load_zsc(self, folds: List[int], configs_hyperparameters: dict, verbose: bool = True) -> ZeroshotSimulatorContext:
        df_configs, df_metadata = self._load_results()

        # Load in real framework results to score against
        if verbose:
            print(f'Loading baselines: {self.benchmark_paths.baselines}')
        df_baselines = self.benchmark_paths.load_baselines()

        score_against_only_baselines = df_baselines is not None

        zsc = ZeroshotSimulatorContext(
            df_configs=df_configs,
            folds=folds,
            df_baselines=df_baselines,
            df_metadata=df_metadata,
            score_against_only_baselines=score_against_only_baselines,
            configs_hyperparameters=configs_hyperparameters,
        )
        return zsc

    def to_json(self, path: str):
        output = {
            "name": self.name,
            "date": self.date,
            "description": self.description,
            "folds": self.folds,
            "s3_download_map": self.s3_download_map,
            "config_fallback": self.config_fallback,
            "benchmark_paths": self.benchmark_paths.to_dict()
        }
        save_json.save(path=path, obj=output)

    @classmethod
    def from_json(cls, path: str) -> Self:
        kwargs = load_json.load(path)
        kwargs["benchmark_paths"] = BenchmarkPaths(**kwargs["benchmark_paths"])
        return cls(**kwargs)


def construct_s3_download_map(
    s3_prefix: str,
    path_context: str,
    split_key: str,
    files_pp: List[str],
    files_gt: List[str],
    task_metadata: str | None = None,
) -> Dict[str, str]:
    split_value = f"{s3_prefix}model_predictions/"
    s3_download_map = {
        # FIXME: COMPARISON ROUNDING ERROR
        "configs.parquet": "configs.parquet",
        "baselines.parquet": "baselines.parquet",
    }
    if task_metadata is not None:
        s3_download_map[task_metadata] = task_metadata
    s3_download_map = {f'{path_context}{k}': f'{s3_prefix}{v}' for k, v in s3_download_map.items()}
    _s3_download_map_metadata_pp = {f"{split_key}{f}": f"{split_value}{f}" for f in files_pp}
    _s3_download_map_metadata_gt = {f"{split_key}{f}": f"{split_value}{f}" for f in files_gt}
    s3_download_map.update(_s3_download_map_metadata_pp)
    s3_download_map.update(_s3_download_map_metadata_gt)
    s3_download_map = {Paths.rel_to_abs(k, relative_to=Paths.data_root): v for k, v in s3_download_map.items()}
    return s3_download_map


def construct_context(
    name: str | None,
    datasets: list[str],
    folds: list[int],
    local_prefix: str,
    s3_prefix: str = None,
    description: str = None,
    date: str = None,
    task_metadata: str = None,
    local_prefix_is_relative: bool = True,
    has_baselines: bool = True,
    metadata_join_column: str = "dataset",
    configs_hyperparameters: list[str] = None,
    is_relative: bool = False,
    config_fallback: str = None,
    dataset_fold_lst_pp: list[tuple[str, int]] = None,
    dataset_fold_lst_gt: list[tuple[str, int]] = None,
) -> BenchmarkContext:
    """

    Parameters
    ----------
    name
    description
    date
    datasets
    folds
    local_prefix: str, default = None
        The location for input files to be downloaded to / located.
    s3_prefix: str, default = None
        The s3 location for input files to download from.
        If None, then all files must already exist locally in the `local_prefix` directory.
        Example: "s3://s3_bucket/foo/bar/2023_08_21/"
    task_metadata

    Returns
    -------
    BenchmarkContext object that is able to load the data.
    """
    if local_prefix_is_relative:
        path_context = str(Paths.results_root_cache / local_prefix) + os.sep
    else:
        path_context = str(Path(local_prefix)) + os.sep

    if local_prefix_is_relative:
        data_root = Paths.data_root_cache
    else:
        data_root = Path(path_context).parent

    split_key = str(Path(path_context) / "model_predictions") + os.sep

    if dataset_fold_lst_pp is None:
        dataset_fold_lst_pp = [(dataset, fold) for dataset in datasets for fold in folds]
    if dataset_fold_lst_gt is None:
        dataset_fold_lst_gt = [(dataset, fold) for dataset in datasets for fold in folds]

    files_pred = ["metadata.json", "pred-test.dat", "pred-val.dat"]
    _files_pp = [f"{dataset}/{fold}/{f}" for dataset, fold in dataset_fold_lst_pp for f in files_pred]

    files_label = ["label-test.csv.zip", "label-val.csv.zip"]
    _files_gt = [f"{dataset}/{fold}/{f}" for dataset, fold in dataset_fold_lst_gt for f in files_label]

    if s3_prefix is not None:
        _s3_download_map = construct_s3_download_map(
            s3_prefix=s3_prefix,
            path_context=path_context,
            split_key=split_key,
            files_pp=_files_pp,
            files_gt=_files_gt,
            task_metadata=task_metadata,
        )
    else:
        _s3_download_map = None

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

    if is_relative:
        relative_path = str(Path(path_context))
    else:
        relative_path = None

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
        s3_download_map=_s3_download_map,
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

