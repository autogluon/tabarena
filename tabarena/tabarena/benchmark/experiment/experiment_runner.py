from __future__ import annotations

import datetime
from typing import Literal, Type

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype

from autogluon.core.data.label_cleaner import LabelCleaner, LabelCleanerDummy
from autogluon.core.metrics import get_metric, Scorer
from tabarena.benchmark.task.openml import OpenMLTaskWrapper
from tabarena.utils.cache import AbstractCacheFunction, CacheFunctionDummy, CacheFunctionDF
from tabarena.benchmark.models.wrapper.abstract_class import AbstractExecModel


# TODO: make a dataclass so type hinter is happy with subclasses?
class ExperimentRunner:
    def __init__(
        self,
        *,
        method_cls: Type[AbstractExecModel],
        task: OpenMLTaskWrapper,
        fold: int,
        task_name: str,
        method: str,
        repeat: int = 0,
        sample: int = 0,
        fit_args: dict | None = None,
        cleanup: bool = True,
        input_format: Literal["openml", "csv"] = "openml",
        cacher: AbstractCacheFunction | None = None,
        debug_mode: bool = True,
    ):
        """

        Parameters
        ----------
        method_cls
        task
        fold
        task_name
        method
        fit_args
        cleanup
        input_format
        cacher
        debug_mode: bool, default True
            If True, will operate in a manner best suited for local model development.
            This mode will be friendly to local debuggers and will avoid subprocesses/threads
            and complex try/except logic.

            IF False, will operate in a manner best suited for large-scale benchmarking.
            This mode will try to record information when method's fail
            and might not work well with local debuggers.
        """
        assert input_format in ["openml", "csv"]
        self.method_cls = method_cls
        self.task = task
        self.fold = fold
        self.repeat = repeat
        self.sample = sample
        self.task_name = task_name
        self.method = method
        self.fit_args = fit_args
        self.cleanup = cleanup
        self.input_format = input_format
        ag_eval_metric_map = {
            'binary': 'roc_auc',
            'multiclass': 'log_loss',
            'regression': 'rmse',
        }
        self.eval_metric_name = ag_eval_metric_map[self.task.problem_type]  # FIXME: Don't hardcode eval metric
        self.eval_metric: Scorer = get_metric(metric=self.eval_metric_name, problem_type=self.task.problem_type)
        self.model: AbstractExecModel | None = None
        self.task_split_idx = self.task.get_split_idx(fold=self.fold, repeat=self.repeat, sample=self.sample)
        self.X, self.y, self.X_test, self.y_test = self.task.get_train_test_split(fold=self.fold, repeat=self.repeat, sample=self.sample)
        if input_format == "csv":
            self.X = self.task.to_csv_format(X=self.X)
            self.X_test = self.task.to_csv_format(X=self.X_test)
        self.label_cleaner = LabelCleaner.construct(problem_type=self.task.problem_type, y=self.y)
        if cacher is None:
            cacher = CacheFunctionDummy()
        self.cacher = cacher
        self.debug_mode = debug_mode

    def init_method(self) -> AbstractExecModel:
        model = self.method_cls(
            problem_type=self.task.problem_type,
            eval_metric=self.eval_metric,
            **self.fit_args,
        )
        return model

    def run_model_fit(self) -> dict:
        return self.model.fit_custom(X=self.X, y=self.y, X_test=self.X_test)

    def run(self) -> dict:
        out = self._run()
        if self.cleanup:
            self._cleanup()
        return out

    @classmethod
    def init_and_run(
        cls,
        method_cls: Type[AbstractExecModel],
        task: OpenMLTaskWrapper,
        fold: int,
        task_name: str,
        method: str,
        fit_args: dict = None,
        cleanup: bool = True,
        input_format: Literal["openml", "csv"] = "openml",
        cacher: AbstractCacheFunction | None = None,
        debug_mode: bool = True,
        **kwargs,
    ) -> dict:
        obj = cls(
            method_cls=method_cls,
            task=task,
            fold=fold,
            task_name=task_name,
            method=method,
            fit_args=fit_args,
            cleanup=cleanup,
            input_format=input_format,
            cacher=cacher,
            debug_mode=debug_mode,
            **kwargs,
        )
        return obj.run()

    def _run(self) -> dict:
        utc_time = datetime.datetime.now(datetime.timezone.utc)
        time_start_str = utc_time.strftime('%Y-%m-%d %H:%M:%S')
        time_start = utc_time.timestamp()
        self.model = self.init_method()
        try:
            out = self.run_model_fit()
        except Exception as exc:
            if not self.debug_mode:
                # Only do this in benchmark mode, since it could mess with a local debugger.
                self.handle_failure(exc=exc)
            raise
        out = self.post_fit(out=out)
        out["metric_error"] = self.evaluate(
            y_true=self.y_test,
            y_pred=out["predictions"],
            y_pred_proba=out["probabilities"],
        )
        out = self.post_evaluate(out=out)
        out["experiment_metadata"] = self._experiment_metadata(time_start=time_start, time_start_str=time_start_str)
        out = self.convert_to_output(out=out)
        return out

    def handle_failure(self, exc: Exception):
        # TODO: This is autogluon specific, make a subclass AGExperimentRunner?
        failures = self.model.failure_artifact
        if not hasattr(self.cacher, "cache_path") or self.cacher.cache_path is None:
            return
        if failures is None:
            try:
                failures = self.model.get_metadata_failure()
            except:
                return
        if failures is None:
            return
        if "model_failures" in failures:
            model_failures = failures["model_failures"]
            if len(model_failures) > 0:
                cacher_model_failures = CacheFunctionDF(cache_path=self.cacher.cache_path, cache_name="model_failures")
                cacher_model_failures.save_cache(data=model_failures)

    def post_fit(self, out: dict) -> dict:
        return out

    def post_evaluate(self, out: dict) -> dict:
        out["task_metadata"] = {
            "tid": self.task.task_id,
            "name": self.task_name,
            "fold": self.fold,
            "repeat": self.repeat,
            "sample": self.sample,
            "split_idx": self.task_split_idx,
        }
        out["framework"] = self.method
        out["problem_type"] = self.task.problem_type
        out["metric"] = self.eval_metric_name

        out["simulation_artifacts"] = None
        if hasattr(self.model, "get_metadata"):
            out["method_metadata"] = self.model.get_metadata()
        if self.model.can_get_error_val:
            out["metric_error_val"] = self.model.get_metric_error_val()
        return out

    def _experiment_metadata(self, time_start: float, time_start_str: str) -> dict:
        metadata = {}
        metadata["experiment_cls"] = self.__class__.__name__
        metadata["method_cls"] = self.method_cls.__name__
        time_end = datetime.datetime.now(datetime.timezone.utc).timestamp()
        metadata["time_start"] = time_start
        metadata["time_end"] = time_end
        metadata["total_duration"] = time_end - time_start
        metadata["time_start_str"] = time_start_str
        return metadata

    def convert_to_output(self, out: dict) -> dict:
        out.pop("predictions")
        out.pop("probabilities")
        return out

    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_pred_proba: pd.Series | pd.DataFrame | None,
    ) -> float:
        return evaluate(
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            scorer=self.eval_metric,
            label_cleaner=self.label_cleaner,
            problem_type=self.task.problem_type,
        )

    def _cleanup(self):
        self.model.cleanup()


class OOFExperimentRunner(ExperimentRunner):
    def __init__(
        self,
        *,
        compute_simulation_artifacts: bool = True,
        compute_bag_info: bool = True,
        optimize_simulation_artifacts_memory: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.compute_simulation_artifacts = compute_simulation_artifacts
        self.compute_bag_info = compute_bag_info
        self.optimize_simulation_artifacts_memory = optimize_simulation_artifacts_memory

    def post_evaluate(self, out: dict) -> dict:
        out = super().post_evaluate(out=out)
        if self.compute_simulation_artifacts and self.model.can_get_oof:
            simulation_artifact = self.model.get_oof()
            if self.task.problem_type == "regression":
                simulation_artifact["pred_proba_dict_test"] = self.label_cleaner.transform(out["predictions"])
            else:
                simulation_artifact["pred_proba_dict_test"] = self.label_cleaner.transform_proba(out["probabilities"], as_pandas=True)
                if self.task.problem_type == "binary":
                    simulation_artifact["pred_proba_dict_test"] = simulation_artifact["pred_proba_dict_test"].iloc[:, 1]
            simulation_artifact["y_test"] = self.label_cleaner.transform(self.y_test)

            if self.optimize_simulation_artifacts_memory:
                # optimize memory
                simulation_artifact["y_test"].index = pd.to_numeric(simulation_artifact["y_test"].index, downcast="integer")
                simulation_artifact["y_val"].index = pd.to_numeric(simulation_artifact["y_val"].index, downcast="integer")

                simulation_artifact["y_test_idx"] = simulation_artifact["y_test"].index.values
                simulation_artifact["y_val_idx"] = simulation_artifact["y_val"].index.values

                simulation_artifact["y_test"] = simulation_artifact["y_test"].values
                simulation_artifact["y_val"] = simulation_artifact["y_val"].values
                if is_integer_dtype(simulation_artifact["y_test"]):
                    simulation_artifact["y_test"] = pd.to_numeric(simulation_artifact["y_test"], downcast="integer")
                if is_integer_dtype(simulation_artifact["y_val"]):
                    simulation_artifact["y_val"] = pd.to_numeric(simulation_artifact["y_val"], downcast="integer")

                simulation_artifact["pred_proba_dict_test"] = simulation_artifact["pred_proba_dict_test"].astype(np.float32)
                simulation_artifact["pred_proba_dict_val"] = simulation_artifact["pred_proba_dict_val"].astype(np.float32)

                simulation_artifact["pred_proba_dict_test"] = simulation_artifact["pred_proba_dict_test"].values
                simulation_artifact["pred_proba_dict_val"] = simulation_artifact["pred_proba_dict_val"].values

            simulation_artifact["label"] = self.task.label
            simulation_artifact["metric"] = self.eval_metric_name

            if self.compute_bag_info and (self.model.can_get_per_child_oof and self.model.can_get_per_child_val_idx):
                simulation_artifact["bag_info"] = self.model.bag_artifact(X_test=self.X_test)


            simulation_artifact["pred_proba_dict_val"] = {self.method: simulation_artifact["pred_proba_dict_val"]}
            simulation_artifact["pred_proba_dict_test"] = {self.method: simulation_artifact["pred_proba_dict_test"]}
        else:
            simulation_artifact = None
        out["simulation_artifacts"] = simulation_artifact
        return out


def evaluate(y_true: pd.Series, y_pred: pd.Series, y_pred_proba: pd.DataFrame, scorer: Scorer, problem_type: str, label_cleaner: LabelCleaner = None) -> float:
    if label_cleaner is None:
        label_cleaner = LabelCleanerDummy(problem_type=problem_type)
    y_true = label_cleaner.transform(y_true)
    if scorer.needs_pred:
        y_pred = label_cleaner.transform(y_pred)
        error = scorer.error(y_true=y_true, y_pred=y_pred)
    elif problem_type == "binary":
        y_pred_proba = label_cleaner.transform_proba(y_pred_proba, as_pandas=True)
        error = scorer.error(y_true=y_true, y_pred=pd.DataFrame(y_pred_proba).iloc[:, 1])
    else:
        y_pred_proba = label_cleaner.transform_proba(y_pred_proba, as_pandas=True)
        error = scorer.error(y_true=y_true, y_pred=y_pred_proba)
    return error
