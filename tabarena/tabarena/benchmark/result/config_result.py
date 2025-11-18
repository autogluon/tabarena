from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from typing import Any
from typing_extensions import Self

from tabarena.benchmark.result.baseline_result import BaselineResult


class ConfigResult(BaselineResult):
    def __init__(self, result: dict, convert_format: bool = True, inplace: bool = False):
        super().__init__(result=result, convert_format=convert_format, inplace=inplace)

        required_keys = [
            "simulation_artifacts",
            "method_metadata",
            "metric_error_val",
        ]
        for key in required_keys:
            assert key in self.result, f"Missing {key} in result dict!"

    @property
    def name_prefix(self) -> str:
        return self.result["method_metadata"].get("name_prefix", "")

    @property
    def name_suffix(self) -> str:
        return self.framework.removeprefix(self.name_prefix)

    @property
    def model_type(self) -> str:
        return self.result["method_metadata"]["model_type"]

    @property
    def ag_key(self) -> str:
        return self.result["method_metadata"].get("ag_key", self.model_type)

    def update_name(self, name: str = None, name_prefix: str = None, name_suffix: str = None, keep_suffix: bool = True):
        assert name is not None or name_prefix is not None or name_suffix is not None, \
            f"Must specify one of `name`, `name_prefix`, `name_suffix`."
        assert name is None or name_prefix is None, f"Must only specify one of `name`, `name_prefix`."
        assert name is None or name_suffix is None, f"Must only specify one of `name`, `name_suffix`."
        if name is not None:
            if keep_suffix:
                self.result["framework"] = f"{name}{self.name_suffix}"
            else:
                self.result["framework"] = name
            self.result["method_metadata"]["name_prefix"] = name
            return
        if name_prefix is not None:
            og_name = self.framework
            assert og_name.startswith(self.name_prefix), (
                f"Tried updating name with `name_prefix='{name_prefix}'`, "
                f"but name did not contain expected prefix '{self.name_prefix}'!"
                f"\n\tname: '{og_name}'"
            )
            new_name_prefix = f"{name_prefix}{self.name_prefix}"
            new_name = f"{new_name_prefix}{og_name.removeprefix(self.name_prefix)}"
            self.result["method_metadata"]["name_prefix"] = new_name_prefix
            self.result["framework"] = new_name
        if name_suffix is not None:
            og_name = self.framework
            assert og_name.startswith(self.name_prefix), (
                f"Tried updating name with `name_suffix='{name_suffix}'`, "
                f"but name did not contain expected prefix '{self.name_prefix}'!"
                f"\n\tname: '{og_name}'"
            )
            new_name_prefix = f"{self.name_prefix}{name_suffix}"
            new_name = f"{new_name_prefix}{og_name.removeprefix(self.name_prefix)}"
            self.result["method_metadata"]["name_prefix"] = new_name_prefix
            self.result["framework"] = new_name

    def update_model_type(self, name: str = None, name_prefix: str = None, name_suffix: str = None):
        assert name is not None or name_prefix is not None or name_suffix is not None, \
            f"Must specify one of `name`, `name_prefix`, `name_suffix`."
        assert name is None or name_prefix is None, f"Must only specify one of `name`, `name_prefix`."
        assert name is None or name_suffix is None, f"Must only specify one of `name`, `name_suffix`."
        if "ag_key" not in self.result["method_metadata"]:
            self.result["method_metadata"]["ag_key"] = self.model_type
        if name is not None:
            self.result["method_metadata"]["model_type"] = name
            return
        if name_prefix is not None:
            self.result["method_metadata"]["model_type"] = f"{name_prefix}{self.model_type}"
        if name_suffix is not None:
            self.result["method_metadata"]["model_type"] = f"{self.model_type}{name_suffix}"

    @property
    def simulation_artifacts(self) -> dict:
        return self.result["simulation_artifacts"]

    @property
    def y_test(self) -> np.ndarray:
        return self.simulation_artifacts["y_test"]

    @property
    def y_val(self) -> np.ndarray:
        return self.simulation_artifacts["y_val"]

    @property
    def y_test_idx(self) -> np.ndarray:
        return self.simulation_artifacts["y_test_idx"]

    @property
    def y_val_idx(self) -> np.ndarray:
        return self.simulation_artifacts["y_val_idx"]

    @property
    def y_pred_proba_test(self) -> np.ndarray:
        return self.simulation_artifacts["pred_test"]

    @property
    def y_pred_proba_val(self) -> np.ndarray:
        return self.simulation_artifacts["pred_val"]

    @property
    def y_pred_proba_test_as_pd(self) -> pd.DataFrame | pd.Series:
        if self.problem_type == "multiclass":
            ordered_class_labels = self.simulation_artifacts["ordered_class_labels"]
            out = pd.DataFrame(data=self.y_pred_proba_test, index=self.y_test_idx, columns=ordered_class_labels)
        elif self.problem_type in ["binary", "regression"]:
            out = pd.Series(data=self.y_pred_proba_test, index=self.y_test_idx, name=self.simulation_artifacts["label"])
        else:
            raise ValueError(f"Unsupported problem_type={self.problem_type}")
        return out

    @property
    def y_pred_proba_val_as_pd(self) -> pd.DataFrame | pd.Series:
        if self.problem_type == "multiclass":
            ordered_class_labels = self.simulation_artifacts["ordered_class_labels"]
            out = pd.DataFrame(data=self.y_pred_proba_val, index=self.y_val_idx, columns=ordered_class_labels)
        elif self.problem_type in ["binary", "regression"]:
            out = pd.Series(data=self.y_pred_proba_val, index=self.y_val_idx, name=self.simulation_artifacts["label"])
        else:
            raise ValueError(f"Unsupported problem_type={self.problem_type}")
        return out

    @property
    def label(self) -> str | int:
        return self.simulation_artifacts["label"]

    def _align_result_input_format(self) -> dict:
        self.result = super()._align_result_input_format()
        dataset = self.result["task_metadata"]["name"]
        split_idx = self.result["task_metadata"]["split_idx"]
        framework = self.result["framework"]

        if list(self.result["simulation_artifacts"].keys()) == [dataset]:
            # if old format
            new_sim_artifacts = self.result["simulation_artifacts"][dataset][split_idx]
            self.result["simulation_artifacts"] = new_sim_artifacts
        if "pred_proba_dict_test" in self.result["simulation_artifacts"]:
            pred_proba_dict_test = self.result["simulation_artifacts"].pop("pred_proba_dict_test")
            pred_proba_test = pred_proba_dict_test[framework]
            self.result["simulation_artifacts"]["pred_test"] = pred_proba_test
        if "pred_proba_dict_val" in self.result["simulation_artifacts"]:
            pred_proba_dict_val = self.result["simulation_artifacts"].pop("pred_proba_dict_val")
            pred_proba_val = pred_proba_dict_val[framework]
            self.result["simulation_artifacts"]["pred_val"] = pred_proba_val
        if "eval_metric" in self.result["simulation_artifacts"]:
            if self.result["simulation_artifacts"]["eval_metric"] == "root_mean_squared_error":
                self.result["simulation_artifacts"]["eval_metric"] = "rmse"
            assert self.result["simulation_artifacts"]["eval_metric"] == self.result["metric"]
            self.result["simulation_artifacts"].pop("eval_metric")
        if "metric" in self.result["simulation_artifacts"]:
            assert self.result["simulation_artifacts"]["metric"] == self.result["metric"]
            self.result["simulation_artifacts"].pop("metric")
        if "problem_type_transform" in self.result["simulation_artifacts"]:
            assert self.result["simulation_artifacts"]["problem_type_transform"] == self.result["problem_type"]
            self.result["simulation_artifacts"].pop("problem_type_transform")
        if "problem_type" in self.result["simulation_artifacts"]:
            assert self.result["simulation_artifacts"]["problem_type"] == self.result["problem_type"]
            self.result["simulation_artifacts"].pop("problem_type")

        return self.result

    def _pred_val_from_children(self) -> np.ndarray:
        num_samples_val = len(self.simulation_artifacts["y_val_idx"])
        if len(self.bag_info["pred_val_per_child"][0].shape) == 1:
            pred_val = np.zeros(dtype=np.float64, shape=num_samples_val)
        else:
            pred_val = np.zeros(dtype=np.float64, shape=(num_samples_val, self.bag_info["pred_val_per_child"][0].shape[1]))
        val_child_count = np.zeros(dtype=int, shape=num_samples_val)
        for val_idx_child, pred_val_child in zip(self.bag_info["val_idx_per_child"], self.bag_info["pred_val_per_child"]):
            val_child_count[val_idx_child] += 1
            pred_val[val_idx_child] += pred_val_child
            pass
        pred_val = pred_val / val_child_count[:, None]
        pred_val = pred_val.astype(np.float32)
        return pred_val

    def _pred_test_from_children(self) -> np.ndarray:
        num_samples_test = len(self.simulation_artifacts["y_test_idx"])
        if len(self.bag_info["pred_val_per_child"][0].shape) == 1:
            pred_test = np.zeros(dtype=np.float64, shape=num_samples_test)
        else:
            pred_test = np.zeros(dtype=np.float64, shape=(num_samples_test, self.bag_info["pred_test_per_child"][0].shape[1]))
        num_children = len(self.bag_info["pred_test_per_child"])
        for pred_test_child in self.bag_info["pred_test_per_child"]:
            pred_test += pred_test_child
        pred_test = pred_test / num_children
        pred_test = pred_test.astype(np.float32)
        return pred_test

    # TODO: Maybe calibrating model binary pred proba will improve ensemble roc_auc?
    def temp_scale(self, y_val, y_pred_proba_val, method: str = "v2"):
        init_val = 1.0
        max_iter = 200
        lr = 0.1
        from tabarena.utils.temp_scaling.calibrators import (
            AutoGluonTemperatureScalingCalibrator,
            TemperatureScalingCalibrator,
            AutoGluonTemperatureScalingCalibratorFixed,
            TemperatureScalingCalibratorFixed,
        )
        if method == "v1":
            calibrator = AutoGluonTemperatureScalingCalibrator(init_val=init_val, max_iter=max_iter, lr=lr)
        elif method == "v2":
            calibrator = TemperatureScalingCalibrator(max_iter=max_iter, lr=lr)
        elif method == "v1_fix":
            calibrator = AutoGluonTemperatureScalingCalibratorFixed(init_val=init_val, max_iter=max_iter, lr=lr)
        elif method == "v2_fix":
            calibrator = TemperatureScalingCalibratorFixed(max_iter=max_iter, lr=lr)
        else:
            raise ValueError(f"Unknown temp_scale method: {method}")
        calibrator.fit(X=y_pred_proba_val, y=y_val)
        return calibrator

    def generate_calibrated(self, method: str = "v2", name_suffix: str = "_CAL") -> Self:
        result = self.result
        sim_artifact = result["simulation_artifacts"]
        metric = result["metric"]
        from autogluon.core.metrics import get_metric
        problem_type = result["problem_type"]
        ag_metric = get_metric(metric=metric, problem_type=problem_type)
        y_test = sim_artifact["y_test"]

        y_val = sim_artifact["y_val"]
        y_pred_proba_val = sim_artifact["pred_val"]
        calibrator = self.temp_scale(y_val=y_val, y_pred_proba_val=y_pred_proba_val, method=method)
        y_pred_proba_test = sim_artifact["pred_test"]
        y_pred_proba_test_scaled = calibrator.predict_proba(y_pred_proba_test)
        y_pred_proba_val_scaled = calibrator.predict_proba(y_pred_proba_val)

        # metric_error_og = ag_metric.error(y_test, y_pred_proba_test)
        metric_error_cal = ag_metric.error(y_test, y_pred_proba_test_scaled)
        metric_error_val_og = ag_metric.error(y_val, y_pred_proba_val)
        metric_error_val_cal = ag_metric.error(y_val, y_pred_proba_val_scaled)

        if metric_error_val_cal > metric_error_val_og:
            print(f"WARNING:")
            print(metric_error_val_cal, metric_error_val_og)
            print(result["framework"], result["dataset"], result["fold"])

        result_calibrated = copy.deepcopy(self)
        result_calibrated.result["metric_error"] = metric_error_cal
        result_calibrated.result["metric_error_val"] = metric_error_val_cal
        result_calibrated.result["simulation_artifacts"]["pred_test"] = y_pred_proba_test_scaled
        result_calibrated.result["simulation_artifacts"]["pred_val"] = y_pred_proba_val_scaled
        result_calibrated.result["framework"] = result_calibrated.result["framework"] + name_suffix
        # FIXME: Fix bag children? Should they be calibrated?

        return result_calibrated

    def compute_metric_test(self, metric, decision_threshold: float | str = 0.5, as_sklearn: bool = False, calibrate: bool = False):
        from autogluon.core.metrics import get_metric
        from autogluon.core.utils.utils import get_pred_from_proba
        ag_metric = get_metric(metric=metric, problem_type=self.problem_type)
        y_test = self.simulation_artifacts["y_test"]
        y_val = self.simulation_artifacts["y_val"]
        y_pred_proba_val = self.simulation_artifacts["pred_val"]
        y_pred_proba_test = self.simulation_artifacts["pred_test"]

        if ag_metric.needs_class:
            # y_pred_val = get_pred_from_proba(y_pred_proba=y_pred_proba_val, problem_type=self.problem_type, decision_threshold=decision_threshold)
            # y_pred_test = get_pred_from_proba(y_pred_proba=y_pred_proba_test, problem_type=self.problem_type, decision_threshold=decision_threshold)
            # metric_error_val = ag_metric.error(y_val, y_pred_val)
            # metric_error_test = ag_metric.error(y_test, y_pred_test)

            if decision_threshold == "auto":
                min_val_rows_for_calibration = 10000
                if self.problem_type == "binary":
                    from autogluon.core.calibrate import calibrate_decision_threshold
                    decision_threshold = calibrate_decision_threshold(
                        y=y_val,
                        y_pred_proba=y_pred_proba_val,
                        metric=ag_metric,
                    )
                else:
                    decision_threshold = 0.5

            y_pred_val = get_pred_from_proba(y_pred_proba=y_pred_proba_val, problem_type=self.problem_type, decision_threshold=decision_threshold)
            y_pred_test = get_pred_from_proba(y_pred_proba=y_pred_proba_test, problem_type=self.problem_type, decision_threshold=decision_threshold)
            metric_error_val = ag_metric.error(y_val, y_pred_val)
            metric_error_test = ag_metric.error(y_test, y_pred_test)
        else:
            if calibrate:
                from autogluon.core.data.label_cleaner import LabelCleanerMulticlassToBinary
                if self.problem_type == "binary":
                    y_pred_proba_val = LabelCleanerMulticlassToBinary.convert_binary_proba_to_multiclass_proba(y_pred_proba_val)
                    y_pred_proba_test = LabelCleanerMulticlassToBinary.convert_binary_proba_to_multiclass_proba(y_pred_proba_test)
                calibrator = self.temp_scale(y_val=y_val, y_pred_proba_val=y_pred_proba_val)
                y_pred_proba_val = calibrator.predict_proba(y_pred_proba_val)
                y_pred_proba_test = calibrator.predict_proba(y_pred_proba_test)
                if self.problem_type == "binary":
                    y_pred_proba_val = y_pred_proba_val[:, 1]
                    y_pred_proba_test = y_pred_proba_test[:, 1]
            metric_error_val = ag_metric.error(y_val, y_pred_proba_val)
            metric_error_test = ag_metric.error(y_test, y_pred_proba_test)

        # print(f"{ag_metric.name}\ttest: {metric_error_test:.4f}\tval: {metric_error_val:.4f}\t{self.problem_type}")
        if as_sklearn:
            metric_score_test = ag_metric.convert_score_to_original(score=ag_metric.convert_error_to_score(error=metric_error_test))
            metric_score_val = ag_metric.convert_score_to_original(score=ag_metric.convert_error_to_score(error=metric_error_val))
            return metric_score_test, metric_score_val
        else:
            return metric_error_test, metric_error_val

    def generate_old_sim_artifact(self) -> dict[str, dict[int, dict[str, Any]]]:
        sim_artifacts = copy.deepcopy(self.simulation_artifacts)
        framework = self.framework
        dataset = self.dataset
        split_idx = self.split_idx

        sim_artifacts["pred_proba_dict_test"] = {framework: sim_artifacts.pop("pred_test")}
        sim_artifacts["pred_proba_dict_val"] = {framework: sim_artifacts.pop("pred_val")}
        sim_artifacts["eval_metric"] = self.result["metric"]
        sim_artifacts["problem_type"] = self.result["problem_type"]
        sim_artifacts["problem_type_transform"] = self.result["problem_type"]
        sim_artifacts = {dataset: {split_idx: sim_artifacts}}
        return sim_artifacts

    @property
    def hyperparameters(self) -> dict:
        if "method_metadata" in self.result and "model_hyperparameters" in self.result["method_metadata"]:
            method_metadata = self.result["method_metadata"]
            model_hyperparameters = method_metadata["model_hyperparameters"]
            model_cls = method_metadata.get("model_cls", None)
            model_type = method_metadata.get("model_type", None)
            ag_key = method_metadata.get("ag_key", model_type)
            name_prefix = method_metadata.get("name_prefix", None)

            config_hyperparameters = dict(
                model_cls=model_cls,
                model_type=model_type,
                ag_key=ag_key,
                name_prefix=name_prefix,
                hyperparameters=model_hyperparameters,
            )
        else:
            config_hyperparameters = dict(
                model_cls=None,
                model_type=None,
                ag_key=None,
                name_prefix=None,
                hyperparameters={},
            )
        return config_hyperparameters
