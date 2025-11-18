
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from tabarena.benchmark.result.baseline_result import BaselineResult
from tabarena.benchmark.result.config_result import ConfigResult


class AGBagResult(ConfigResult):
    @property
    def bag_info(self) -> dict:
        return self.simulation_artifacts["bag_info"]

    @property
    def y_pred_proba_test_per_child(self) -> list[np.ndarray]:
        return self.bag_info["pred_test_per_child"]

    @property
    def y_pred_proba_val_per_child(self) -> list[np.ndarray]:
        return self.bag_info["pred_val_per_child"]

    def y_pred_proba_test_child(self, idx: int) -> np.ndarray:
        return self.y_pred_proba_test_per_child[idx]

    def y_pred_proba_val_child(self, idx: int) -> np.ndarray:
        return self.y_pred_proba_val_per_child[idx]

    @property
    def val_idx_per_child(self) -> list[np.ndarray]:
        return self.bag_info["val_idx_per_child"]

    def val_idx_child(self, idx: int) -> np.ndarray:
        return self.val_idx_per_child[idx]

    def y_pred_proba_val_child_as_pd(self, idx: int) -> pd.DataFrame | pd.Series:
        y_pred_proba_val_child = self.y_pred_proba_val_child(idx=idx)
        val_idx_child = self.val_idx_child(idx=idx)

        if self.problem_type == "multiclass":
            ordered_class_labels = self.simulation_artifacts["ordered_class_labels"]
            out = pd.DataFrame(data=y_pred_proba_val_child, index=val_idx_child, columns=ordered_class_labels)
        elif self.problem_type in ["binary", "regression"]:
            out = pd.Series(data=y_pred_proba_val_child, index=val_idx_child, name=self.simulation_artifacts["label"])
        else:
            raise ValueError(f"Unsupported problem_type={self.problem_type}")
        return out

    def y_pred_proba_test_child_as_pd(self, idx: int) -> pd.DataFrame | pd.Series:
        y_pred_proba_test_child = self.y_pred_proba_test_child(idx=idx)

        if self.problem_type == "multiclass":
            ordered_class_labels = self.simulation_artifacts["ordered_class_labels"]
            out = pd.DataFrame(data=y_pred_proba_test_child, index=self.y_test_idx, columns=ordered_class_labels)
        elif self.problem_type in ["binary", "regression"]:
            out = pd.Series(data=y_pred_proba_test_child, index=self.y_test_idx, name=self.simulation_artifacts["label"])
        else:
            raise ValueError(f"Unsupported problem_type={self.problem_type}")
        return out

    def _align_result_input_format(self) -> dict:
        self.result = super()._align_result_input_format()

        bag_info = self.result["simulation_artifacts"]["bag_info"]
        if "pred_proba_test_per_child" in bag_info:
            bag_info["pred_test_per_child"] = bag_info.pop("pred_proba_test_per_child")
        num_samples_val = len(self.result["simulation_artifacts"]["y_val_idx"])
        if "val_idx_per_child" in bag_info and "pred_val_per_child" not in bag_info:
            # Ensure no repeated bagging
            assert num_samples_val == sum([len(val_idx_child) for val_idx_child in bag_info["val_idx_per_child"]])
            # convert to pred_val_per_child
            pred_val_per_child = []
            if len(bag_info["val_idx_per_child"]) == 1:
                # FIXME: Bug in AutoGluon's output! Wrong indices. This logic fixes that.
                y_val_idx = self.result["simulation_artifacts"]["y_val_idx"]
                value_to_index = {value: idx for idx, value in enumerate(y_val_idx)}
                val_idx_child = bag_info["val_idx_per_child"][0]
                val_idx_child_iloc = np.array([value_to_index[v] for v in val_idx_child])
                bag_info["val_idx_per_child"][0] = val_idx_child_iloc
            for val_idx_child in bag_info["val_idx_per_child"]:
                pred_val_child_cur = self.result["simulation_artifacts"]["pred_val"][val_idx_child]
                pred_val_per_child.append(pred_val_child_cur)
            bag_info["pred_val_per_child"] = pred_val_per_child

        pred_val = self._pred_val_from_children()
        if "pred_val" in self.result["simulation_artifacts"]:
            assert pred_val.shape == self.simulation_artifacts["pred_val"].shape
            if not np.isclose(pred_val, self.simulation_artifacts["pred_val"]).all():
                print(f"WARNING: Not close  VAL: {self.result['task_metadata']['name']}, {self.result['task_metadata']['split_idx']}, {self.result['framework']}")
        self.simulation_artifacts["pred_val"] = pred_val
        pred_test = self._pred_test_from_children()
        if "pred_test" in self.result["simulation_artifacts"]:
            assert pred_test.shape == self.simulation_artifacts["pred_test"].shape
            is_close_lst = np.isclose(pred_test, self.simulation_artifacts["pred_test"], rtol=5e-4)
            if not is_close_lst.all():
                print(
                    f"WARNING: Not close TEST: {self.result['task_metadata']['name']}, {self.result['task_metadata']['split_idx']}, {self.result['framework']}"
                    f" |\t{((1 - is_close_lst.mean()) * 100):.3f}% of samples were not close!"
                )
        self.simulation_artifacts["pred_test"] = pred_test
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
        if len(pred_val.shape) == 1:
            pred_val = pred_val / val_child_count
        else:
            pred_val = pred_val / val_child_count[:, None]
        pred_val = pred_val.astype(np.float32)
        return pred_val

    def _pred_test_from_children(self) -> np.ndarray:
        num_samples_test = len(self.simulation_artifacts["y_test_idx"])
        if len(self.bag_info["pred_val_per_child"][0].shape) == 1:
            pred_test = np.zeros(dtype=np.float64, shape=num_samples_test)
        else:
            pred_test = np.zeros(dtype=np.float64, shape=(num_samples_test, self.bag_info["pred_test_per_child"][0].shape[1]))
        for pred_test_child in self.bag_info["pred_test_per_child"]:
            pred_test += pred_test_child
        pred_test = pred_test / self.num_children
        pred_test = pred_test.astype(np.float32)
        return pred_test

    @property
    def num_children(self) -> int:
        return len(self.bag_info["pred_test_per_child"])

    def bag_artifacts(self, as_baseline: bool = True) -> list[BaselineResult]:
        """
        Logic that gets the holdout artifact from the bag by only using the first child model.

        Parameters
        ----------
        as_baseline

        Returns
        -------

        """
        results_new = []
        sim_artifact = self.simulation_artifacts
        pred_proba_test_per_child = sim_artifact["bag_info"]["pred_test_per_child"]
        num_children = self.num_children
        # metric = self.result["metric"]
        framework = self.result["framework"]

        # problem_type = self.result["problem_type"]
        # ag_metric = get_metric(metric=metric, problem_type=problem_type)
        # y_test = sim_artifact["y_test"]

        # y_test_idx = sim_artifact["y_test_idx"]
        # y_test = pd.Series(data=y_test, index=y_test_idx)

        if num_children > 1:
            for i, c in enumerate(pred_proba_test_per_child):
                if i != 0:
                    break
                # if problem_type == "multiclass":
                #     y_pred_proba_test_child = pd.DataFrame(data=c, index=y_test_idx, columns=sim_artifact["ordered_class_labels_transformed"])
                # else:
                #     y_pred_proba_test_child = pd.Series(data=c, index=y_test_idx)
                #
                # # FIXME: needs to work with predictions too, not just pred proba
                # metric_error = ag_metric.error(y_test, y_pred_proba_test_child)

                result_baseline_new = copy.deepcopy(self.result)

                holdout_name = framework + "_HOLDOUT"
                result_baseline_new["framework"] = holdout_name
                # result_baseline_new["metric_error"] = metric_error
                result_baseline_new["time_train_s"] /= num_children
                result_baseline_new["time_infer_s"] /= num_children

                if not as_baseline:
                    sim_artifacts_holdout = result_baseline_new["simulation_artifacts"]

                    y_val_idx_to_keep = sim_artifacts_holdout["bag_info"]["val_idx_per_child"][i]

                    sim_artifacts_holdout["y_val_idx"] = sim_artifacts_holdout["y_val_idx"][y_val_idx_to_keep]
                    sim_artifacts_holdout["y_val"] = sim_artifacts_holdout["y_val"][y_val_idx_to_keep]
                    sim_artifacts_holdout["pred_val"] = sim_artifacts_holdout["pred_val"][y_val_idx_to_keep]

                    # FIXME: Can be more elegant
                    sim_artifacts_holdout["pred_val"] = sim_artifacts_holdout["bag_info"]["pred_val_per_child"][i]
                    sim_artifacts_holdout["pred_test"] = sim_artifacts_holdout["bag_info"]["pred_test_per_child"][i]

                    sim_artifacts_holdout["bag_info"]["val_idx_per_child"] = [val_idx_child for i_cur, val_idx_child in enumerate(sim_artifacts_holdout["bag_info"]["val_idx_per_child"]) if i_cur == i]
                    sim_artifacts_holdout["bag_info"]["pred_val_per_child"] = [pred_val_child for i_cur, pred_val_child in
                                                                              enumerate(sim_artifacts_holdout["bag_info"]["pred_val_per_child"]) if i_cur == i]
                    sim_artifacts_holdout["bag_info"]["pred_test_per_child"] = [pred_test_child for i_cur, pred_test_child in
                                                                              enumerate(sim_artifacts_holdout["bag_info"]["pred_test_per_child"]) if i_cur == i]

                    # result_baseline_new["metric_error_val"] = ag_metric.error(y_test, y_pred_proba_test_child)

                    ag_bag_result_holdout = AGBagResult(result_baseline_new, convert_format=False, inplace=True)

                    metric_error_test_holdout, metric_error_val_holdout = ag_bag_result_holdout.compute_metric_test(
                        metric=ag_bag_result_holdout.result["metric"],
                    )

                    result_baseline_new["metric_error_val"] = metric_error_val_holdout
                    result_baseline_new["metric_error"] = metric_error_test_holdout

                    result_baseline_new["method_metadata"]["disk_usage"] = result_baseline_new["method_metadata"]["disk_usage"] // num_children


                    # FIXME: support repeat bagging

                if as_baseline:
                    result_baseline_new = BaselineResult(result=result_baseline_new, convert_format=False, inplace=True)
                else:
                    result_baseline_new = AGBagResult(result=result_baseline_new, convert_format=False, inplace=True)
                results_new.append(result_baseline_new)

            # print("ens", self.result["metric_error"])
        return results_new
