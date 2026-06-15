from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
from autogluon.common.loaders import load_pkl
from autogluon.common.savers import save_pkl

from tabarena.benchmark.result.abstract_result import AbstractResult


class BaselineResult(AbstractResult):
    def __init__(self, result: dict, convert_format: bool = True, inplace: bool = False):
        super().__init__(result=result, inplace=inplace)
        if convert_format:
            if inplace:
                self.result = copy.deepcopy(self.result)
            self.result = self._align_result_input_format()

        required_keys = [
            "framework",
            "task_metadata",
            "metric_error",
            "metric",
            "problem_type",
            "time_train_s",
            "time_infer_s",
        ]
        for key in required_keys:
            assert key in self.result, f"Missing {key} in result dict!"

    @classmethod
    def from_dict(cls, result: dict | BaselineResult) -> BaselineResult:
        """Converts results in old format to new format
        Keeps results in new format as-is.

        This enables the use of results in the old format alongside results in the new format.

        When `simulation_artifacts` is present but `method_metadata` and/or
        `metric_error_val` are missing, sensible defaults are populated so the
        result is treated as a `ConfigResult` rather than a `BaselineResult`:
        `method_metadata` is filled with the framework name as `model_type`,
        and `metric_error_val` is recomputed from the validation predictions
        in `simulation_artifacts`.
        """
        from tabarena.benchmark.result.ag_bag_result import AGBagResult
        from tabarena.benchmark.result.config_result import ConfigResult

        if isinstance(result, BaselineResult):
            return result
        assert isinstance(result, dict)
        result_cls = BaselineResult
        sim_artifacts = result.get("simulation_artifacts", None)
        method_metadata = result.get("method_metadata", None)
        if sim_artifacts is not None and method_metadata is None:
            framework = result["framework"]
            method_metadata = {
                "model_type": framework,
                "model_hyperparameters": {},
                "name_prefix": framework,
            }
            result["method_metadata"] = method_metadata
        if sim_artifacts is not None and method_metadata is not None:
            assert isinstance(sim_artifacts, dict)
            if "task_metadata" in result:
                dataset = result["task_metadata"]["name"]
                split_idx = result["task_metadata"]["split_idx"]
            else:
                dataset = result["dataset"]
                split_idx = result["fold"]
            result_cls = ConfigResult
            if list(sim_artifacts.keys()) == [dataset]:
                sim_artifacts = sim_artifacts[dataset][split_idx]
            if "metric_error_val" not in result:
                metric_error_val = cls._compute_metric_error_val(
                    result=result,
                    sim_artifacts=sim_artifacts,
                )
                if metric_error_val is not None:
                    result["metric_error_val"] = metric_error_val
            bag_info = sim_artifacts.get("bag_info", None)
            if bag_info is not None:
                assert isinstance(bag_info, dict)
                result_cls = AGBagResult
        return result_cls(result=result, convert_format=True, inplace=False)

    @staticmethod
    def _compute_metric_error_val(result: dict, sim_artifacts: dict) -> float | None:
        framework = result["framework"]
        y_val = sim_artifacts.get("y_val")
        pred_proba_val = sim_artifacts.get("pred_val")
        if pred_proba_val is None:
            pred_proba_dict_val = sim_artifacts.get("pred_proba_dict_val")
            if isinstance(pred_proba_dict_val, dict):
                pred_proba_val = pred_proba_dict_val.get(framework)
        if y_val is None or pred_proba_val is None:
            return None
        from autogluon.core.metrics import get_metric
        from autogluon.core.utils.utils import get_pred_from_proba

        ag_metric = get_metric(metric=result["metric"], problem_type=result["problem_type"])
        if ag_metric.needs_class:
            y_pred_val = get_pred_from_proba(
                y_pred_proba=pred_proba_val,
                problem_type=result["problem_type"],
            )
            return ag_metric.error(y_val, y_pred_val)
        if result["problem_type"] == "binary" and len(pred_proba_val.shape) != 1:
            pred_proba_val = pred_proba_val[:, 1]
        return ag_metric.error(y_val, pred_proba_val)

    @classmethod
    def from_pickle(cls, path: str | Path) -> BaselineResult:
        result: dict | BaselineResult = load_pkl.load(path)
        return cls.from_dict(result=result)

    def update_name(self, name: str | None = None, name_prefix: str | None = None, name_suffix: str | None = None):
        assert name is not None or name_prefix is not None or name_suffix is not None, (
            "Must specify one of `name`, `name_prefix`, `name_suffix`."
        )
        assert name is None or name_prefix is None, "Must only specify one of `name`, `name_prefix`."
        assert name is None or name_suffix is None, "Must only specify one of `name`, `name_suffix`."
        if name is not None:
            self.result["framework"] = name
            return
        if name_prefix is not None:
            self.result["framework"] = f"{name_prefix}{self.framework}"
        if name_suffix is not None:
            self.result["framework"] = f"{self.framework}{name_suffix}"

    @property
    def framework(self) -> str:
        return self.result["framework"]

    @property
    def dataset(self) -> str:
        return self.task_metadata["name"]

    @property
    def problem_type(self) -> str:
        return self.result["problem_type"]

    @property
    def split_idx(self) -> int:
        return self.task_metadata["split_idx"]

    @property
    def repeat(self) -> int:
        return self.task_metadata["repeat"]

    @property
    def fold(self) -> int:
        return self.task_metadata["fold"]

    @property
    def sample(self) -> int:
        return self.task_metadata["sample"]

    @property
    def task_metadata(self) -> dict:
        return self.result["task_metadata"]

    def _align_result_input_format(self) -> dict:
        """Converts results in old format to new format
        Keeps results in new format as-is.

        This enables the use of results in the old format alongside results in the new format.

        Returns:
        -------

        """
        if "metric_error_val" in self.result:
            self.result["metric_error_val"] = float(self.result["metric_error_val"])
        # Canonicalize metric-name aliases for every result (e.g. AutoGluon's
        # ``root_mean_squared_error`` == TabArena's ``rmse``), so downstream comparisons join
        # on one name regardless of whether the result carries simulation artifacts.
        if "metric" in self.result:
            from tabarena.benchmark.task.metrics import normalize_eval_metric

            self.result["metric"] = normalize_eval_metric(self.result["metric"])
        if "df_results" in self.result:
            self.result.pop("df_results")
        if "task_metadata" not in self.result:
            self.result["task_metadata"] = dict(
                fold=self.result["fold"],
                repeat=0,
                sample=0,
                split_idx=self.result["fold"],
                tid=self.result["tid"],
                name=self.result["dataset"],
            )
            self.result.pop("fold")
            self.result.pop("tid")
            self.result.pop("dataset")
        return self.result

    def compute_df_result(self) -> pd.DataFrame:
        required_columns = [
            "framework",
            "metric_error",
            "metric",
            "time_train_s",
            "time_infer_s",
            "problem_type",
        ]

        data = {
            "dataset": self.dataset,
            "fold": self.split_idx,
        }

        optional_columns = [
            "metric_error_val",
        ]

        columns_to_use = copy.deepcopy(required_columns)

        for c in required_columns:
            assert c in self.result
        for c in optional_columns:
            if c in self.result:
                columns_to_use.append(c)

        data.update({c: self.result[c] for c in columns_to_use})

        if "tid" in self.result["task_metadata"]:
            data.update({"tid": self.result["task_metadata"]["tid"]})

        if "method_metadata" in self.result:
            method_metadata = self.result["method_metadata"]

            optional_metadata_columns = [
                "num_cpus",
                "num_gpus",
                "disk_usage",
            ]

            for col in optional_metadata_columns:
                if col in method_metadata:
                    assert col not in data
                    data.update({col: method_metadata[col]})

        return pd.DataFrame([data])

    def to_dir(self, path: str | Path):
        suffix = Path(f"{self.framework}")
        if "tid" in self.result["task_metadata"]:
            suffix = suffix / str(self.result["task_metadata"]["tid"])
        else:
            suffix = suffix / str(self.dataset)
        suffix = suffix / f"{self.repeat}_{self.fold}"
        path_full = Path(path) / suffix
        path_file = path_full / "results.pkl"
        save_pkl.save(path=str(path_file), object=self.result)
