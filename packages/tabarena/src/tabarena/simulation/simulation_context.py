from __future__ import annotations

import copy
import os
import threading
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import pandas as pd
from autogluon.common.loaders import load_json, load_pd
from autogluon.common.savers import save_json, save_pd

from tabarena.predictions import (
    TabularModelPredictions,
    TabularPredictionsInMemory,
    TabularPredictionsInMemoryOpt,
    TabularPredictionsMemmap,
)
from tabarena.utils import task_to_tid_fold
from tabarena.utils.rank_utils import RankScorer

from .ground_truth import GroundTruth
from .sim_utils import (
    get_dataset_to_metric_problem_type,
    has_duplicate_keys,
)

if TYPE_CHECKING:
    from .task_data import TaskData


def _default_dict():
    # this free function is added as `defaultdict(lambda: defaultdict(dict))` cant be pickled
    return defaultdict(dict)


#: Threads used by :meth:`ZeroshotSimulatorContext.load_groundtruth` for the per-task file reads
#: when it runs on the main thread. Calls from a worker thread (a method-loading pool) use one
#: thread: eleven methods loading side by side already saturate the filesystem and the GIL, and
#: nesting more threads there measured slower.
LABEL_LOAD_THREADS = 8


def _default_label_load_threads() -> int:
    return LABEL_LOAD_THREADS if threading.current_thread() is threading.main_thread() else 1


class ZeroshotSimulatorContext:
    def __init__(
        self,
        df_configs: pd.DataFrame = None,
        df_baselines: pd.DataFrame = None,
        df_metadata: pd.DataFrame = None,
        configs_hyperparameters: dict[str, dict[str, Any]] | None = None,
        folds: list[int] | None = None,
        pct: bool = False,
        score_against_only_baselines: bool = True,
    ):
        """Encapsulates results evaluated on multiple base models/datasets/folds.
        :param df_configs: results of configs by multiple datasets/folds
        :param df_baselines: results of baseline systems by multiple datasets/folds
        :param folds: List of folds to be considered in a list of integers. If None, will not filter by fold.
        :param pct: whether to use percentage rather than rank numbers
        :param score_against_only_baselines: if `True`, the scores are ranks (or percentage if `pct` is True) over the baselines only
        baselines. If False, the scores are computed against both baselines and the configs.
        """
        if df_configs is None:
            df_configs = self._create_empty_df_configs()
        if df_baselines is None:
            df_baselines = self._create_empty_df_baselines()
        if configs_hyperparameters is None:
            configs_hyperparameters = {}

        self.folds = folds
        self.score_against_only_baselines = score_against_only_baselines
        self.pct = pct

        # TODO align_valid_folds returns 8 values and does many different things, it would help to break down to more
        #  modular functions
        (
            self.df_configs,
            self.df_baselines,
            self.df_metrics,
            self.df_metadata,
            self.task_to_dataset_dict,
            self.dataset_to_folds_dict,
            self.dataset_to_tid_dict,
            self.dataset_to_problem_type_dict,
            self.task_to_fold_dict,
            self.folds,
            self.unique_tasks,
            self.unique_datasets,
            self.configs_hyperparameters,
            self.rank_scorer,
        ) = self._align_valid_folds(
            df_configs=df_configs,
            df_baselines=df_baselines,
            df_metadata=df_metadata,
            configs_hyperparameters=configs_hyperparameters,
            folds=folds,
            score_against_only_automl=self.score_against_only_baselines,
            pct=self.pct,
        )
        self._df_configs_ranked: pd.DataFrame | None = None
        self.dataset_to_tasks_dict = self._compute_dataset_to_tasks()

    @property
    def df_configs_ranked(self) -> pd.DataFrame:
        """``df_configs`` plus a ``rank`` column: each result's rank among the comparison set
        (see :attr:`rank_scorer`), computed on first access.

        Loading and ensemble simulation never read the ranks, only portfolio and analysis code
        does, so the column is not built eagerly (0.35 s for a 12-method collection and a few
        hundredths per method inside the loading pool). The value is what eager construction gave:
        the ranks depend only on the scorer, which is fixed at construction and rebuilt by the
        subset methods exactly when the eager column was rebuilt.
        """
        cached = self.__dict__.get("_df_configs_ranked")
        if cached is None:
            # objects pickled before the column became lazy carry it as a plain attribute
            cached = self.__dict__.get("df_configs_ranked")
        if cached is None:
            cached = self.df_configs.copy()
            if len(cached) > 0:
                cached["rank"] = self.rank_scorer.rank_many(
                    tasks=cached["task"].to_numpy(),
                    errors=cached["metric_error"].to_numpy(),
                )
            else:
                cached["rank"] = None
            self._df_configs_ranked = cached
        return cached

    # Frames whose string columns are pickled as categoricals (see __getstate__).
    _PICKLE_CATEGORICAL_FRAMES = ("df_configs", "df_baselines", "_df_configs_ranked")

    def __getstate__(self):
        """Pickle the result frames' string columns as categoricals.

        The frames are hundreds of thousands of rows whose ``dataset`` / ``framework`` / ``task``
        / ``metric`` / ``problem_type`` columns hold a few thousand distinct strings. Pickled as
        object columns they cost one memo lookup per cell on both ends; as categoricals they are
        an integer code array (sent out of band) plus the distinct values. ``__setstate__``
        restores the object dtype, so nothing outside pickling ever sees a categorical and the
        pandas ``groupby`` / ``value_counts`` semantics of the frames are unchanged. Cuts the ray
        transfer of a 12-method collection's contexts by about half.
        """
        state = dict(self.__dict__)
        converted: dict[str, list[str]] = {}
        for attr in self._PICKLE_CATEGORICAL_FRAMES:
            df = state.get(attr)
            if df is None or len(df) == 0:
                continue
            columns = [
                c
                for c in df.columns
                if df[c].dtype == object and pd.api.types.infer_dtype(df[c], skipna=True) == "string"
            ]
            if not columns:
                continue
            state[attr] = df.astype(dict.fromkeys(columns, "category"))
            converted[attr] = columns
        state["_pickled_categorical_columns"] = converted
        return state

    def __setstate__(self, state):
        converted = state.pop("_pickled_categorical_columns", {})
        self.__dict__.update(state)
        for attr, columns in converted.items():
            df = self.__dict__[attr]
            # column by column: a frame-level astype would copy the numeric columns too, which
            # arrive as zero-copy views into ray's object store and should stay that way
            for c in columns:
                df[c] = df[c].astype(object)

    def _compute_dataset_to_tasks(self) -> dict:
        """Returns the mapping of dataset parent to dataset fold names.
        For example:
        {
            'DATASET_NAME': ['DATASET_NAME_1', 'DATASET_NAME_2', ..., 'DATASET_NAME_10'],
            ...,
        }.

        """
        dataset_to_tasks_dict = dict()
        for task in self.unique_tasks:
            dataset = self.task_to_dataset_dict[task]
            if dataset in dataset_to_tasks_dict:
                dataset_to_tasks_dict[dataset].append(task)
            else:
                dataset_to_tasks_dict[dataset] = [task]
        for task in dataset_to_tasks_dict:
            dataset_to_tasks_dict[task] = sorted(dataset_to_tasks_dict[task])
        return dataset_to_tasks_dict

    def _update_all(self, folds=None):
        if folds is None:
            folds = self.folds
        (
            self.df_configs,
            self.df_baselines,
            self.df_metrics,
            self.df_metadata,
            self.task_to_dataset_dict,
            self.dataset_to_folds_dict,
            self.dataset_to_tid_dict,
            self.dataset_to_problem_type_dict,
            self.task_to_fold_dict,
            self.folds,
            self.unique_tasks,
            self.unique_datasets,
            self.configs_hyperparameters,
            self.rank_scorer,
        ) = self._align_valid_folds(
            df_configs=self.df_configs,
            df_baselines=self.df_baselines,
            df_metadata=self.df_metadata,
            configs_hyperparameters=self.configs_hyperparameters,
            folds=folds,
            score_against_only_automl=self.score_against_only_baselines,
            pct=self.pct,
        )
        self._df_configs_ranked = None
        self.dataset_to_tasks_dict = self._compute_dataset_to_tasks()

    @classmethod
    def _align_valid_folds(
        cls,
        *,
        df_configs: pd.DataFrame,
        df_baselines: pd.DataFrame,
        df_metadata: pd.DataFrame,
        configs_hyperparameters: dict,
        folds: list[int] | None,
        score_against_only_automl: bool,
        pct: bool,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        dict[str, str],
        dict[str, list[int]],
        dict[str, int],
        dict[str, str],
        dict[str, int],
        list[int],
        list[str],
        list[str],
        dict[str, Any],
        RankScorer,
    ]:
        cls._validate_df_configs(df_configs=df_configs)
        cls._validate_df_baselines(df_baselines=df_baselines)
        # assert that each dataset contains only one problem type
        dataset_problem_types = cls._compute_dataset_problem_types(df_configs=df_configs, df_baselines=df_baselines)

        if df_metadata is not None and "dataset" not in df_metadata and "name" in df_metadata:
            df_metadata = df_metadata.copy(deep=True)
            df_metadata["dataset"] = df_metadata["name"]
        cls._validate_df_metadata(df_metadata=df_metadata)

        df_configs, df_baselines, dataset_tid = cls._align_tid(
            df_configs=df_configs,
            df_baselines=df_baselines,
            df_metadata=df_metadata,
        )

        df_baselines = df_baselines.drop(columns=["tid"], errors="ignore").merge(dataset_tid, on=["dataset"])
        df_baselines = df_baselines.drop(columns=["problem_type"], errors="ignore").merge(
            dataset_problem_types, on=["dataset"]
        )

        # One hashing pass per frame: integer codes for the (dataset, fold) pairs of configs and
        # baselines together, in first-occurrence order (what drop_duplicates would keep). Every
        # per-task structure below derives from these codes instead of re-hashing the string
        # columns of the full frame, which the pandas equivalents did about fifteen times.
        n_configs = len(df_configs)
        dataset_all = np.concatenate([df_configs["dataset"].to_numpy(), df_baselines["dataset"].to_numpy()])
        fold_all = np.concatenate([df_configs["fold"].to_numpy(), df_baselines["fold"].to_numpy()]).astype(np.int64)
        tid_all = np.concatenate([df_configs["tid"].to_numpy(), df_baselines["tid"].to_numpy()])
        assert fold_all.min(initial=0) >= 0, "folds must be non-negative"
        dataset_codes, dataset_uniques = pd.factorize(dataset_all)
        fold_span = int(fold_all.max(initial=0)) + 1
        pair_codes, pair_keys = pd.factorize(dataset_codes.astype(np.int64) * fold_span + fold_all)

        # every row's (dataset, fold) is in the union of both frames by construction, so the
        # former merge against that union only reset the index
        df_configs = df_configs.reset_index(drop=True)
        df_baselines = df_baselines.reset_index(drop=True)

        framework_codes, framework_uniques = pd.factorize(
            np.concatenate([df_configs["framework"].to_numpy(), df_baselines["framework"].to_numpy()])
        )
        result_keys = framework_codes.astype(np.int64) * len(pair_keys) + pair_codes
        for source, df_source, keys in (
            ("df_baselines", df_baselines, result_keys[n_configs:]),
            ("df_configs", df_configs, result_keys[:n_configs]),
        ):
            if has_duplicate_keys(keys, n_range=len(framework_uniques) * len(pair_keys)):
                config_task_counts = df_source[["framework", "dataset", "fold"]].value_counts()
                raise AssertionError(
                    f"Multiple rows in `{source}` exist for a config task pair! "
                    f"There should only ever be one row per config task pair. "
                    f"You might have multiple results from re-runs or other bugs that have not been de-duplicated.\n"
                    f"Config Task Counts:\n"
                    f"{config_task_counts}"
                )

        keep = np.ones(len(fold_all), dtype=bool)
        if folds is not None:
            keep = np.isin(fold_all, np.asarray(folds))
            df_configs = df_configs[keep[:n_configs]].reset_index(drop=True)
            df_baselines = df_baselines[keep[n_configs:]].reset_index(drop=True)
        pair_codes_c = pair_codes[:n_configs][keep[:n_configs]]
        pair_codes_b = pair_codes[n_configs:][keep[n_configs:]]

        # per unique pair: dataset, fold, tid (constant per dataset, see _compute_dataset_tid) and task name
        pair_first_row = np.empty(len(pair_keys), dtype=np.int64)
        pair_first_row[pair_codes[::-1]] = np.arange(len(pair_codes))[::-1]  # reversed, so the first row wins
        pair_dataset = dataset_uniques[pair_keys // fold_span]
        pair_fold = pair_keys % fold_span
        pair_tid = tid_all[pair_first_row]
        pair_task = (pd.Series(pair_tid).astype(str) + "_" + pd.Series(pair_fold).astype(str)).to_numpy()
        df_configs["task"] = pair_task[pair_codes_c]
        df_baselines["task"] = pair_task[pair_codes_b]

        present = np.zeros(len(pair_keys), dtype=bool)
        present[pair_codes_c] = True
        present[pair_codes_b] = True
        pairs_present = np.flatnonzero(present)
        unique_tasks = sorted(set(pair_task[pairs_present].tolist()))
        unique_datasets = sorted(set(pair_dataset[pairs_present].tolist()))

        unique_folds = cls._compute_folds_from_data(df_configs=df_configs, df_baselines=df_baselines)

        # FIXME: Remove scoring via baselines by default all-together, or maybe remove scoring period.
        if score_against_only_automl and len(df_baselines) > 0:
            assert len(df_baselines) != 0, (
                "`score_against_only_automl=True`, but `df_baselines` is empty. "
                "Either specify `df_baselines` or set `score_against_only_automl=False`."
            )
            df_comparison = df_baselines.copy(deep=True)
        else:
            df_comparison = pd.concat([d for d in [df_configs, df_baselines] if len(d) != 0], ignore_index=True)
        rank_scorer = RankScorer(
            df_results=df_comparison,
            tasks=unique_tasks,
            pct=pct,
        )
        task_to_dataset_dict = {pair_task[i]: pair_dataset[i] for i in pairs_present}
        dataset_to_tid_dict = {}
        for i in pairs_present:
            dataset_to_tid_dict.setdefault(
                pair_dataset[i], pair_tid[i].item() if hasattr(pair_tid[i], "item") else pair_tid[i]
            )
        assert len(unique_datasets) == len(dataset_to_tid_dict.keys())

        df_metrics = cls._dataset_metric_problem_type(
            df_configs=df_configs,
            df_baselines=df_baselines,
            dataset_codes=dataset_codes[keep],
            dataset_uniques=dataset_uniques,
        )

        cls._minimize_df_metadata(df_metadata=df_metadata, unique_datasets=unique_datasets)

        cls._verify_configs_hyperparameters(configs_hyperparameters=configs_hyperparameters)
        configs_hyperparameters = copy.deepcopy(configs_hyperparameters)
        unique_configs = sorted(df_configs["framework"].unique())
        # TODO: Avoid needing to do configs_base, make the names match to begin with
        configs_base = set([cls._config_name_to_config_base(config) for config in unique_configs] + unique_configs)
        configs_hyperparameters_keys = list(configs_hyperparameters.keys())
        for c in configs_hyperparameters_keys:
            if c not in configs_base:
                configs_hyperparameters.pop(c)

        dataset_to_problem_type_dict = df_metrics["problem_type"].to_dict()

        task_to_fold_dict = {pair_task[i]: int(pair_fold[i]) for i in pairs_present}
        dataset_to_folds_dict: dict[str, list[int]] = {}
        for i in pairs_present:
            dataset_to_folds_dict.setdefault(pair_dataset[i], []).append(int(pair_fold[i]))
        dataset_to_folds_dict = {d: sorted(f) for d, f in dataset_to_folds_dict.items()}

        return (
            df_configs,
            df_baselines,
            df_metrics,
            df_metadata,
            task_to_dataset_dict,
            dataset_to_folds_dict,
            dataset_to_tid_dict,
            dataset_to_problem_type_dict,
            task_to_fold_dict,
            unique_folds,
            unique_tasks,
            unique_datasets,
            configs_hyperparameters,
            rank_scorer,
        )

    @staticmethod
    def _dataset_metric_problem_type(
        *,
        df_configs: pd.DataFrame,
        df_baselines: pd.DataFrame,
        dataset_codes: np.ndarray,
        dataset_uniques: np.ndarray,
    ) -> pd.DataFrame:
        """``dataset -> (metric, problem_type)`` frame, one row per dataset in first-occurrence
        order across configs then baselines; raises through
        :func:`~tabarena.simulation.sim_utils.get_dataset_to_metric_problem_type` when a dataset
        carries more than one combination.
        """
        metric_all = np.concatenate([df_configs["metric"].to_numpy(), df_baselines["metric"].to_numpy()])
        problem_type_all = np.concatenate(
            [df_configs["problem_type"].to_numpy(), df_baselines["problem_type"].to_numpy()]
        )
        metric_codes, metric_uniques = pd.factorize(metric_all)
        pt_codes, pt_uniques = pd.factorize(problem_type_all)
        combo = (dataset_codes.astype(np.int64) * len(metric_uniques) + metric_codes) * len(pt_uniques) + pt_codes
        combo_codes, combo_uniques = pd.factorize(combo)
        combo_dataset = combo_uniques // (len(metric_uniques) * len(pt_uniques))
        if has_duplicate_keys(combo_dataset, n_range=len(dataset_uniques)):
            return get_dataset_to_metric_problem_type(df_configs=df_configs, df_baselines=df_baselines)
        return pd.DataFrame(
            {
                "dataset": dataset_uniques[combo_dataset],
                "metric": metric_uniques[(combo_uniques // len(pt_uniques)) % len(metric_uniques)],
                "problem_type": pt_uniques[combo_uniques % len(pt_uniques)],
            }
        ).set_index("dataset")

    @staticmethod
    def _validate_df_metadata(df_metadata: pd.DataFrame):
        if df_metadata is not None:
            assert "dataset" in df_metadata, (
                f"Missing required `dataset` column in metadata.\nColumns: {list(df_metadata.columns)}"
            )
            assert len(df_metadata) == len(df_metadata["dataset"].unique())

    @staticmethod
    def _minimize_df_metadata(df_metadata: pd.DataFrame, unique_datasets: list[str]):
        if df_metadata is not None:
            df_metadata = copy.deepcopy(df_metadata)
            df_metadata = df_metadata[df_metadata["dataset"].isin(unique_datasets)]
            metadata_datasets = sorted(df_metadata["dataset"].unique())
            present_datasets = sorted(unique_datasets)
            if metadata_datasets != present_datasets:
                missing_metadata_datasets = [d for d in present_datasets if d not in metadata_datasets]
                raise AssertionError(
                    f"Datasets are present in df_configs / df_baselines, but are missing in df_metadata!\n"
                    f"\t{len(missing_metadata_datasets)} missing datasets: {missing_metadata_datasets}",
                )

            assert len(df_metadata) == len(unique_datasets)
        return df_metadata

    @classmethod
    def _align_tid(
        cls, df_configs: pd.DataFrame, df_baselines: pd.DataFrame, df_metadata: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        datasets = sorted(set(list(df_configs["dataset"].unique()) + list(df_baselines["dataset"].unique())))
        if df_metadata is None or "tid" not in df_metadata.columns:
            if len(df_configs) > 0 and len(df_baselines) > 0:
                if "tid" in df_configs.columns:
                    assert "tid" in df_baselines.columns, (
                        "If `tid` not in df_metadata, then df_configs and df_baselines "
                        "must both contain `tid` or both not contain `tid` columns."
                    )
                else:
                    assert "tid" not in df_baselines.columns, (
                        "If `tid` not in df_metadata, then df_configs and df_baselines "
                        "must both contain `tid` or both not contain `tid` columns."
                    )

        df_configs = cls._fillna_tid(df=df_configs, df_metadata=df_metadata, datasets=datasets, name="df_configs")
        df_baselines = cls._fillna_tid(df=df_baselines, df_metadata=df_metadata, datasets=datasets, name="df_baselines")

        dataset_tid = cls._compute_dataset_tid(df_configs=df_configs, df_baselines=df_baselines)
        if df_metadata is not None and "tid" in df_metadata.columns:
            map_metadata_dataset_tid = df_metadata[["dataset", "tid"]].set_index("dataset")["tid"]
            map_dataset_tid = dataset_tid.set_index("dataset")["tid"]
            if not map_metadata_dataset_tid.loc[map_dataset_tid.index].equals(map_dataset_tid):
                raise AssertionError("Mismatched dataset -> tid map between metadata and configs/baselines!")

        return df_configs, df_baselines, dataset_tid

    @staticmethod
    def _fillna_tid(df: pd.DataFrame, df_metadata: pd.DataFrame, datasets: list[str], name: str) -> pd.DataFrame:
        if "tid" not in df.columns:
            df = df.copy(deep=True)
            if df_metadata is not None and "tid" in df_metadata.columns:
                dataset_tid_map = df_metadata.set_index("dataset")["tid"]
                df["tid"] = df["dataset"].map(dataset_tid_map)
            else:
                if len(df) > 0:
                    print(
                        f"Note: `tid` is missing from `{name}` columns. `tid` will be created by mapping the sorted unique `dataset` values to [0, n-1]. "
                        f"These values will be unrelated to OpenML task IDs."
                    )
                dataset_tid_map = {d: i for i, d in enumerate(datasets)}
                df["tid"] = df["dataset"].map(dataset_tid_map).astype(int)
        return df

    @staticmethod
    def _compute_dataset_tid(df_configs: pd.DataFrame, df_baselines: pd.DataFrame) -> pd.DataFrame:
        # assert that each dataset-tid combination is exclusive
        dataset_tid = df_configs[["dataset", "tid"]].drop_duplicates()
        if df_baselines is not None:
            dataset_tid_baselines = df_baselines[["dataset", "tid"]].drop_duplicates()
            dataset_tid = pd.concat([dataset_tid, dataset_tid_baselines], ignore_index=True).drop_duplicates()
        if len(dataset_tid) != len(dataset_tid["dataset"].unique()):
            dataset_counts = dataset_tid["dataset"].value_counts()
            non_unique_datasets = dataset_counts[dataset_counts > 1]
            dataset_tid_invalid = (
                dataset_tid[dataset_tid["dataset"].isin(non_unique_datasets.index)]
                .sort_values(by=["dataset", "tid"])
                .reset_index(drop=True)
            )
            raise ValueError(
                f"{len(non_unique_datasets)} invalid datasets encountered! Datasets contain different task IDs (tid) within `df_configs`. "
                f"Ensure the tid is unique.\nInvalid Datasets:\n{dataset_tid_invalid}",
            )

        assert len(dataset_tid) == len(dataset_tid["tid"].unique())
        return dataset_tid

    @staticmethod
    def _compute_dataset_problem_types(df_configs: pd.DataFrame, df_baselines: pd.DataFrame) -> pd.DataFrame:
        # assert that each dataset contains only one problem type
        dataset_problem_types = df_configs[["dataset", "problem_type"]].drop_duplicates()
        dataset_problem_types_baselines = df_baselines[["dataset", "problem_type"]].drop_duplicates()
        dataset_problem_types = pd.concat(
            [dataset_problem_types, dataset_problem_types_baselines], ignore_index=True
        ).drop_duplicates()
        assert len(dataset_problem_types) == len(dataset_problem_types["dataset"].unique()), (
            "Error: datasets exist in `df_configs` that contain multiple problem_types!"
        )

        dataset_problem_types_comparison = df_baselines[["dataset", "problem_type"]].drop_duplicates()
        assert len(dataset_problem_types_comparison) == len(dataset_problem_types_comparison["dataset"].unique()), (
            "Error: datasets exist in `df_baselines` that contain multiple problem_types!"
        )
        dataset_problem_types_map_configs = dataset_problem_types.set_index("dataset").squeeze(axis=1).to_dict()
        dataset_problem_types_map_baselines = (
            dataset_problem_types_comparison.set_index("dataset").squeeze(axis=1).to_dict()
        )
        for d in dataset_problem_types_map_configs:
            problem_type_configs = dataset_problem_types_map_configs[d]
            if d in dataset_problem_types_map_baselines:
                problem_type_baselines = dataset_problem_types_map_baselines[d]
                assert problem_type_configs == problem_type_baselines, (
                    f"Error: Dataset `{d}` has a different `problem_type` between `df_configs` and `df_baselines`. They must match:\n"
                    f"\tdf_configs  : {problem_type_configs}\n"
                    f"\tdf_baselines: {problem_type_baselines}"
                )
        return dataset_problem_types

    def df_dataset_folds(self) -> pd.DataFrame:
        return self.df_configs[["dataset", "fold"]].drop_duplicates().reset_index(drop=True)

    def dataset_folds(self) -> list[tuple]:
        dataset_folds = self.df_dataset_folds().values.tolist()
        return [tuple(dataset_fold) for dataset_fold in dataset_folds]

    def dataset_to_folds(self, dataset: str) -> list[int]:
        return self.dataset_to_folds_dict[dataset]

    @staticmethod
    def _validate_df(df: pd.DataFrame, name: str, required_columns: list[str]):
        df_columns = list(df.columns)
        missing_required_columns = [c for c in required_columns if c not in df_columns]
        if missing_required_columns:
            present_required_columns = [c for c in required_columns if c in df_columns]
            present_extra_columns = [c for c in df_columns if c not in required_columns]
            raise AssertionError(
                f"Missing required columns in `{name}`:\n"
                f"\tMissing Required: {missing_required_columns}\n"
                f"\tPresent Required: {present_required_columns}\n"
                f"\tPresent    Extra: {present_extra_columns}\n"
                f"{df}"
            )

    @classmethod
    def _validate_df_configs(cls, df_configs: pd.DataFrame):
        assert df_configs is not None
        assert isinstance(df_configs, pd.DataFrame)
        required_columns = [
            "dataset",
            "fold",
            "framework",
            "metric_error",
            "metric_error_val",
            "metric",
            "problem_type",
            "time_train_s",
            "time_infer_s",
        ]
        cls._validate_df(df=df_configs, name="df_configs", required_columns=required_columns)

    @classmethod
    def _validate_df_baselines(cls, df_baselines: pd.DataFrame):
        assert df_baselines is not None
        assert isinstance(df_baselines, pd.DataFrame)
        required_columns = [
            "dataset",
            "fold",
            "framework",
            "metric_error",
            "metric",
            "problem_type",
            "time_train_s",
            "time_infer_s",
        ]
        cls._validate_df(df=df_baselines, name="df_baselines", required_columns=required_columns)

    def print_info(self):
        out = "====== Zeroshot Simulator Context Info ======\n"
        out += f"# Configs: {len(self.get_configs())}\n"
        out += f"# Baselines: {len(self.get_baselines())}\n"
        out += f"# Datasets: {len(self.unique_datasets)}\n"
        if self.folds is not None:
            out += f"# Folds: {len(self.folds)}\n"
            out += f"Folds: {self.folds}\n"
        out += f"# Folds*Datasets: {len(self.unique_tasks)}\n"
        out += "=============================================\n"
        print(out)

    def get_datasets(
        self,
        *,
        configs: list[str] | None = None,
        problem_type: str | list[str] | None = None,
        union: bool = True,
    ) -> list[str]:
        datasets = self.unique_datasets
        if problem_type is not None:
            if isinstance(problem_type, list):
                datasets = [
                    dataset for dataset in datasets if self.dataset_to_problem_type_dict[dataset] in problem_type
                ]
            else:
                datasets = [
                    dataset for dataset in datasets if self.dataset_to_problem_type_dict[dataset] == problem_type
                ]
        df_configs = self.df_configs
        if configs is not None:
            configs_set = set(configs)
            df_configs = df_configs[df_configs["dataset"].isin(datasets) & df_configs["framework"].isin(configs_set)]
            datasets_remain = df_configs["dataset"].unique()
            datasets = [d for d in datasets if d in datasets_remain]
        if not union:
            if configs is None:
                configs = self.get_configs(union=True)
                configs_set = set(configs)
            n_configs = len(configs)
            df_configs_filtered = df_configs[df_configs["framework"].isin(configs_set)]
            value_counts = df_configs_filtered.value_counts(["dataset", "fold"])
            value_counts_valid = value_counts[value_counts == n_configs]
            dataset_value_counts = (
                value_counts_valid.reset_index(drop=False)[["dataset", "count"]]
                .groupby("dataset")["count"]
                .sum()
                .to_dict()
            )
            dataset_fold_counts = {d: len(folds) for d, folds in self.dataset_to_folds_dict.items()}

            # filter to only datasets that contain all configs
            datasets = [d for d in datasets if dataset_value_counts.get(d, 0) == (dataset_fold_counts[d] * n_configs)]
        return datasets

    def task_to_fold(self, task) -> int:
        return self.task_to_fold_dict[task]

    def get_tids(self, problem_type=None) -> list[int]:
        datasets = self.get_datasets(problem_type=problem_type)
        return [self.dataset_to_tid_dict[dataset] for dataset in datasets]

    def get_tasks(
        self,
        datasets: list[str] | None = None,
        problem_type: str | list[str] | None = None,
        as_dataset_fold: bool = False,
    ) -> list[str] | list[tuple[str, int]]:
        """:param datasets: a list of dataset parent names, only return folds that have a parent in this list
        :param problem_type: a problem type from AutoGluon in "multiclass", "binary", ... or list of problem types
        :return: List of datasets-folds formatted as `['359987_8', '359933_3', ...]` where the dataset is encoded before
        the "_" and the fold after.
        """
        tasks = self._get_tasks_from_datasets(datasets=datasets) if datasets is not None else self.unique_tasks
        if problem_type is not None:
            if not isinstance(problem_type, list):
                problem_type = [problem_type]
            tasks = [
                task
                for task in tasks
                if self.dataset_to_problem_type_dict[self.task_to_dataset_dict[task]] in problem_type
            ]
        if as_dataset_fold:
            # Resolve the tid -> dataset map once; `tid_to_dataset_dict` is a property that
            # rebuilds the dict on every access, which made this loop quadratic in practice.
            tid_to_dataset = self.tid_to_dataset_dict
            tasks = [self._task_to_dataset_fold(task, tid_to_dataset=tid_to_dataset) for task in tasks]
        return tasks

    def _task_to_dataset_fold(self, task: str, tid_to_dataset: dict[int, str] | None = None) -> tuple[str, int]:
        if tid_to_dataset is None:
            tid_to_dataset = self.tid_to_dataset_dict
        tid, fold = task_to_tid_fold(task=task)
        dataset = tid_to_dataset[tid]
        return dataset, fold

    def _get_tasks_from_datasets(self, datasets: list[str]):
        dataset_folds = []
        for d in datasets:
            dataset_folds += self.dataset_to_tasks_dict[d]
        return dataset_folds

    @property
    def tid_to_dataset_dict(self):
        return {v: k for k, v in self.dataset_to_tid_dict.items()}

    @staticmethod
    def task_name_from_tid(tid: int, fold: int) -> str:
        return f"{tid}_{fold}"

    def get_configs(
        self,
        *,
        datasets: list[str] | None = None,
        tasks: list[tuple[str, int]] | None = None,
        config_types: list[str] | None = None,
        union: bool = True,
    ) -> list[str]:
        """Return all valid configs.
        By default, will return all configs that appear in any task at least once.

        Parameters
        ----------
        datasets : list[str], default = None
            If specified, will only consider the configs present in the given datasets
        tasks: list[tuple[str, int]], default = None
            If specified, will only consider the configs present in the given tasks
        config_types: list[str], default = None
            If specified, will only consider the configs with a config type in `config_types`.
        union: bool, default = True
            If True, will return the union of configs present in each task.
            If False, will return the intersection of configs present in each task.

        Returns:
        -------
        A list of config names satisfying the above conditions.
        """
        df = self.df_configs
        if df is None:
            return []
        df = self._filter_df_by_datasets(df=df, datasets=datasets, tasks=tasks)
        configs = self._get_configs_from_df(df=df, union=union)
        if config_types is not None:
            configs_type = self.configs_type
            configs = [c for c in configs if configs_type[c] in config_types]
        return configs

    @staticmethod
    def task_entries_from_paths(paths: list[str]) -> dict[Path, list[int] | None]:
        """Group a context's per-task file paths (``<dataset>/<fold>/<file>``) or per-dataset
        ``tasks.dat`` paths into ``dataset directory -> folds`` (``None`` = every fold of the file).
        """
        from tabarena.simulation.task_data import TASKS_FILENAME

        # String splits rather than one Path per entry: a context lists one entry per task file
        # (tens of thousands) and pathlib costs about 13 us each.
        entries: dict[Path, list[int] | None] = {}
        seen: dict[str, set[int]] = {}
        for p in paths:
            p = str(p).rstrip(os.sep)
            head, _, name = p.rpartition(os.sep)
            if name == TASKS_FILENAME:
                entries.setdefault(Path(head), None)
                continue
            dataset_dir, _, fold = head.rpartition(os.sep)
            seen.setdefault(dataset_dir, set()).add(int(fold))
        for dataset_dir, folds in seen.items():
            entries[Path(dataset_dir)] = sorted(folds)
        return entries

    def load_task_data(self, paths: list[str], n_threads: int | None = None) -> dict[tuple[str, int], TaskData]:
        """Prediction metadata and labels of the tasks behind ``paths``, from each dataset's
        ``tasks.dat`` or its per-task files (see :mod:`tabarena.simulation.task_data`), through the
        active :class:`~tabarena.simulation.label_cache.LabelFileCache` when one is set.

        The reads run on ``n_threads`` threads over datasets: :data:`LABEL_LOAD_THREADS` on the
        main thread and 1 inside a worker thread (see the constant).
        """
        from tabarena.simulation.label_cache import get_active_label_cache
        from tabarena.simulation.task_data import load_task_data

        if n_threads is None:
            n_threads = _default_label_load_threads()
        unique_datasets = set(self.unique_datasets)
        entries = {d: f for d, f in self.task_entries_from_paths(paths).items() if d.name in unique_datasets}
        return load_task_data(entries, n_threads=n_threads, label_cache=get_active_label_cache())

    def load_groundtruth(
        self,
        paths_gt: list[str],
        metadata_by_dir: dict[str, dict] | None = None,
        n_threads: int | None = None,
        task_data: dict[tuple[str, int], TaskData] | None = None,
    ) -> GroundTruth:
        """Load the per-task labels behind ``paths_gt`` (see :meth:`load_task_data`); pass
        ``task_data`` to reuse an earlier load. ``metadata_by_dir`` is filled with each task
        directory's prediction metadata when given.
        """
        if task_data is None:
            task_data = self.load_task_data(paths_gt, n_threads=n_threads)
        gt_val = defaultdict(_default_dict)
        gt_test = defaultdict(_default_dict)
        for (dataset, fold), task in task_data.items():
            gt_val[dataset][fold] = task.labels_val
            gt_test[dataset][fold] = task.labels_test
            if metadata_by_dir is not None:
                metadata_by_dir[str(Path(paths_gt[0]).parent.parent.parent / dataset / str(fold))] = {
                    **task.metadata,
                    "dataset": dataset,
                    "fold": fold,
                }
        # task-data arrays are already in the stored form (see GroundTruth.__init__)
        return GroundTruth(label_val_dict=gt_val, label_test_dict=gt_test, normalize=False)

    def load_pred(
        self,
        path_pred_proba: Path | str,
        datasets: list[str],
        prediction_format: str = "memmap",
        metadata_by_dir: dict[str, dict] | None = None,
        metadata_files: list[str | Path] | None = None,
        metadata_dict: dict[str, dict[int, dict]] | None = None,
    ) -> TabularModelPredictions:
        """:param prediction_format: Determines the format of the loaded tabular_predictions. Default = "memmap".
        "memmap": Fast and low memory usage.
        "memopt": Very fast and high memory usage.
        "mem": Slow and high memory usage, simplest format to debug.
        :param metadata_by_dir: Optional cache of parsed per-task ``metadata.json`` keyed by
        task directory.
        :param metadata_files: The per-task ``metadata.json`` paths to load; ``None`` walks
        ``path_pred_proba`` for them (see ``TabularPredictionsMemmap``).
        :param metadata_dict: The per-task prediction metadata already loaded (``dataset -> fold ->
        metadata``), e.g. from :meth:`load_task_data`; no metadata file is read when given.
        """
        assert prediction_format in ["memmap", "memopt", "mem"]

        class_map = {
            "memmap": TabularPredictionsMemmap,
            "memopt": TabularPredictionsInMemoryOpt,
            "mem": TabularPredictionsInMemory,
        }

        path_pred_proba = Path(path_pred_proba)
        zeroshot_pred_proba: TabularModelPredictions = class_map[prediction_format].from_data_dir(
            data_dir=path_pred_proba,
            datasets=datasets,
            metadata_by_dir=metadata_by_dir,
            metadata_files=metadata_files,
            metadata_dict=metadata_dict,
        )
        all_datasets = self.get_datasets()
        valid_datasets = [d for d in zeroshot_pred_proba.datasets if d in all_datasets]
        zeroshot_pred_proba.restrict_datasets(datasets=valid_datasets)
        return zeroshot_pred_proba

    def subset_datasets(self, datasets: list[str], only_configs: bool = False):
        """Only keep the provided datasets, drop all others."""
        for d in datasets:
            if d not in self.unique_datasets:
                raise ValueError(f"Missing expected dataset {d} in ZeroshotSimulatorContext!")

        # Remove datasets from internal dataframes
        self.df_configs = self.df_configs[self.df_configs["dataset"].isin(datasets)]
        self._df_configs_ranked = None
        if only_configs:
            datasets_baselines = list(set(self.df_baselines["dataset"]))
            datasets = [d for d in self.unique_datasets if d in datasets or d in datasets_baselines]
        unique_datasets_subset = []
        for d in self.unique_datasets:
            if d in datasets:
                unique_datasets_subset.append(d)
        self._update_unique_datasets(unique_datasets_subset)

        if self.df_baselines is not None:
            self.df_baselines = self.df_baselines[self.df_baselines["dataset"].isin(self.unique_datasets)]
        if self.df_metadata is not None:
            self.df_metadata = self.df_metadata[self.df_metadata["dataset"].isin(self.unique_datasets)]
        self.folds = self._compute_folds_from_data(df_configs=self.df_configs, df_baselines=self.df_baselines)
        self.dataset_to_tid_dict = {d: t for d, t in self.dataset_to_tid_dict.items() if d in self.unique_datasets}
        self.dataset_to_folds_dict = {d: t for d, t in self.dataset_to_folds_dict.items() if d in self.unique_datasets}

    @classmethod
    def _compute_folds_from_data(cls, df_configs: pd.DataFrame, df_baselines: pd.DataFrame) -> list[int]:
        return sorted(set(list(df_configs["fold"].unique()) + list(df_baselines["fold"].unique())))

    def subset_configs(self, configs: list[str]):
        """Only keep the provided configs, drop all others."""
        self.df_configs = self.df_configs[self.df_configs["framework"].isin(configs)]
        self._update_all()

    def subset_baselines(self, baselines: list[str]):
        """Only keep the provided configs, drop all others."""
        self.df_baselines = self.df_baselines[self.df_baselines["framework"].isin(baselines)]
        self._update_all()

    def subset_folds(self, folds: list[int]):
        """Only keep the provided folds, drop all others."""
        self._update_all(folds=folds)

    def _update_unique_datasets(self, unique_datasets):
        for d in unique_datasets:
            assert d in self.unique_datasets
        unique_tasks = []
        dataset_to_tasks_dict = dict()
        for d in unique_datasets:
            dataset_to_tasks_dict[d] = self.dataset_to_tasks_dict[d]
            unique_tasks += dataset_to_tasks_dict[d]
        self.unique_datasets = unique_datasets
        self.unique_tasks = unique_tasks
        self.dataset_to_tasks_dict = dataset_to_tasks_dict

    def _get_configs_from_df(self, df: pd.DataFrame, union: bool = True) -> list[str]:
        """Parameters
        ----------
        df: pd.DataFrame
            A DataFrame containing the columns "task" and "framework".
        union: bool, default = True
            If True, will return the union of configs present in each task.
            If False, will return the intersection of configs present in each task.

        Returns:
        -------
        The list of "framework" values present in `df` that satisfy the `union` value logic.
        """
        if union:
            res = df["framework"].unique()
        else:
            datasets = list(df["dataset"].unique())
            tasks = [t for d in datasets for t in self.dataset_to_tasks_dict[d]]
            res = None
            for task in tasks:
                methods = set(df.loc[df["task"] == task, "framework"].unique())
                res = methods if res is None else res.intersection(methods)
        if res is None:
            res = []
        return sorted(res)

    def get_baselines(
        self, *, datasets: list[str] | None = None, tasks: list[tuple[str, int]] | None = None, union: bool = True
    ) -> list[str]:
        """Return all valid baselines.
        By default, will return all baselines that appear in any task at least once.

        Parameters
        ----------
        datasets : list[str], default = None
            If specified, will only consider the baselines present in the given datasets
        tasks: list[tuple[str, int]], default = None
            If specified, will only consider the baselines present in the given tasks
        union: bool, default = True
            If True, will return the union of baselines in each task.
            If False, will return the intersection of baselines present in each task.

        Returns:
        -------
        A list of baseline names satisfying the above conditions.
        """
        df = self.df_baselines
        if df is None:
            return []
        df = self._filter_df_by_datasets(df=df, datasets=datasets, tasks=tasks)
        return self._get_configs_from_df(df=df, union=union)

    def _filter_df_by_datasets(
        self,
        df: pd.DataFrame,
        configs: list[str] | None = None,
        datasets: list[str] | None = None,
        tasks: list[tuple[str, int]] | None = None,
    ) -> pd.DataFrame:
        if configs is not None:
            df = df[df["framework"].isin(configs)]
        if datasets is not None:
            datasets_all = set(self.get_datasets())
            datasets_invalid = set(datasets).difference(datasets_all)
            if len(datasets_invalid) != 0:
                raise ValueError(f"Invalid datasets specified: {sorted(datasets_invalid)}")
            df = df[df["dataset"].isin(datasets)]
        if tasks is not None:
            tasks_all = set(self.get_tasks(as_dataset_fold=True))
            tasks_invalid = set(tasks).difference(tasks_all)
            if len(tasks_invalid) != 0:
                raise ValueError(f"Invalid tasks specified: {sorted(tasks_invalid)}")

            tasks_df = pd.DataFrame(tasks, columns=["dataset", "fold"])
            df = df.merge(tasks_df, on=["dataset", "fold"])

        return df

    def get_configs_hyperparameters(
        self, configs: list[str] | None = None, include_ag_args: bool = True
    ) -> dict[str, dict | None]:
        if configs is None:
            configs = self.get_configs()
        else:
            valid_configs = set(self.get_configs())
            for config in configs:
                if config not in valid_configs:
                    raise ValueError(f"User-specified config does not exist: '{config}'")
        return {
            c: self.get_config_hyperparameters(config=c, include_ag_args=include_ag_args, check_valid=False)
            for c in configs
        }

    def get_config_hyperparameters(
        self, config: str, include_ag_args: bool = True, check_valid: bool = True
    ) -> dict | None:
        if check_valid and config not in self.get_configs():
            raise ValueError(f"User-specified config does not exist: '{config}'")
        if self.configs_hyperparameters is None:
            return None
        if config in self.configs_hyperparameters:
            config_base = config
        else:
            # FIXME: (TabRepo v2) Hardcoding _BAG name, avoid this
            config_base = self._config_name_to_config_base(config=config)
            if config_base not in self.configs_hyperparameters:
                return None
        config_hyperparameters = self.configs_hyperparameters[config_base]
        if not include_ag_args:
            config_hyperparameters = copy.deepcopy(config_hyperparameters)
            config_hyperparameters["hyperparameters"].pop("ag_args", None)
        return config_hyperparameters

    # FIXME: This should be removed eventually
    @classmethod
    def _config_name_to_config_base(cls, config: str) -> str:
        return config.rsplit("_BAG_", 1)[0]

    @property
    def configs_type(self) -> dict[str, str | None]:
        """Returns a dict of config name -> config type.
        If model type is unknown, the value will be `None`.

        For example:
        "CatBoost_c1_BAG_L1": "CAT"
        "ExtraTrees_r13_BAG_L1": "XT"
        """
        configs = self.get_configs()
        configs_type = {}
        for config in configs:
            config_hyperparameters = self.get_config_hyperparameters(config=config, check_valid=False)
            if config_hyperparameters is None:
                configs_type[config] = None
            else:
                configs_type[config] = config_hyperparameters.get("model_type", None)
        return configs_type

    def df_configs_task(self, dataset: str, fold: int, configs: list[str] | None = None) -> pd.DataFrame:
        df_configs_task = self.df_configs[(self.df_configs["dataset"] == dataset) & (self.df_configs["fold"] == fold)]
        if configs is not None:
            configs = set(configs)
            df_configs_task = df_configs_task[df_configs_task["framework"].isin(configs)]
        return df_configs_task

    # TODO: Support max_models and max_models_per_type for simulation
    def get_top_configs(
        self,
        dataset: str,
        fold: int,
        configs: list[str] | None = None,
        max_models: int | None = None,
        max_models_per_type: int | None = None,
    ) -> list[str]:
        df_configs_task = self.df_configs_task(dataset=dataset, fold=fold, configs=configs)
        df_configs_task_sorted = df_configs_task.sort_values(by=["metric_error_val", "framework"])
        df_configs_task_sorted["type"] = df_configs_task_sorted["framework"].map(self.configs_type).fillna("nan")

        if max_models_per_type is not None:
            df_configs_task_sorted["rank_by_type"] = (
                df_configs_task_sorted.groupby(["type", "task"])["metric_error_val"].rank("first").astype(int)
            )
            df_configs_task_sorted = df_configs_task_sorted[
                df_configs_task_sorted["rank_by_type"] <= max_models_per_type
            ]
        if max_models is not None:
            df_configs_task_sorted["rank"] = (
                df_configs_task_sorted.groupby("task")["metric_error_val"].rank("first").astype(int)
            )
            df_configs_task_sorted = df_configs_task_sorted[df_configs_task_sorted["rank"] <= max_models]
        return list(df_configs_task_sorted["framework"])

    # TODO: Potentially change the format in future
    # TODO: Check for `model_type` key?
    # TODO: What about `name_prefix` and `name_suffix`?
    @classmethod
    def _verify_configs_hyperparameters(cls, configs_hyperparameters: dict[str, dict[str, str | dict]]):
        for config, v in configs_hyperparameters.items():
            assert isinstance(v, dict), (
                f"configs_hyperparameters value for key {config} must be of type dict, found: {type(v)}"
            )
            assert "hyperparameters" in v, (
                f"configs_hyperparameters value for key {config} must include a `hyperparameters` key"
            )
            assert isinstance(v["hyperparameters"], dict), (
                f"configs_hyperparameters['{config}']['hyperparameters'] "
                f"must be of type dict, found: {type(v['hyperparameters'])}"
            )

    @classmethod
    def _create_empty_df_configs(cls) -> pd.DataFrame:
        required_columns = [
            "dataset",
            "fold",
            "framework",
            "metric_error",
            "metric_error_val",
            "metric",
            "problem_type",
            "time_train_s",
            "time_infer_s",
        ]
        return pd.DataFrame(columns=required_columns)

    @classmethod
    def _create_empty_df_baselines(cls) -> pd.DataFrame:
        required_columns = [
            "dataset",
            "fold",
            "framework",
            "metric_error",
            "metric",
            "problem_type",
            "time_train_s",
            "time_infer_s",
        ]
        return pd.DataFrame(columns=required_columns)

    def to_dir(self, path: str) -> dict:
        path_configs = "configs.parquet"
        save_pd.save(path=str(Path(path) / path_configs), df=self.df_configs)

        path_baselines = None
        if self.df_baselines is not None:
            path_baselines = "baselines.parquet"
            save_pd.save(path=str(Path(path) / path_baselines), df=self.df_baselines)

        path_metadata = None
        if self.df_metadata is not None:
            path_metadata = "task_metadata.parquet"
            save_pd.save(path=str(Path(path) / path_metadata), df=self.df_metadata)

        path_configs_hyperparameters = None
        if self.configs_hyperparameters is not None:
            path_configs_hyperparameters = "configs_hyperparameters.json"
            save_json.save(path=str(Path(path) / path_configs_hyperparameters), obj=self.configs_hyperparameters)

        metadata = {
            "df_configs": path_configs,
            "df_baselines": path_baselines,
            "df_metadata": path_metadata,
            "configs_hyperparameters": path_configs_hyperparameters,
            "pct": self.pct,
            "score_against_only_baselines": self.score_against_only_baselines,
        }
        path_metadata_json = "metadata.json"
        save_json.save(path=str(Path(path) / path_metadata_json), obj=metadata)
        return metadata

    @classmethod
    def from_dir(cls, path: str) -> Self:
        path_metadata_json = "metadata.json"
        metadata = load_json.load(path=path_metadata_json)

        path_configs = metadata["df_configs"]
        df_configs = load_pd.load(str(Path(path) / path_configs))

        df_baselines = None
        path_baselines = metadata["df_baselines"]
        if path_baselines is not None:
            df_baselines = load_pd.load(str(Path(path) / path_baselines))

        df_metadata = None
        path_metadata = metadata["df_metadata"]
        if path_metadata is not None:
            df_metadata = load_pd.load(str(Path(path) / path_metadata))

        configs_hyperparameters = None
        path_configs_hyperparameters = metadata["configs_hyperparameters"]
        if path_configs_hyperparameters is not None:
            configs_hyperparameters = load_json.load(str(Path(path) / path_configs_hyperparameters))

        pct = metadata["pct"]
        score_against_only_baselines = metadata["score_against_only_baselines"]

        return cls(
            df_configs=df_configs,
            df_baselines=df_baselines,
            df_metadata=df_metadata,
            configs_hyperparameters=configs_hyperparameters,
            pct=pct,
            score_against_only_baselines=score_against_only_baselines,
        )
