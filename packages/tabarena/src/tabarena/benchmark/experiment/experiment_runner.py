"""Runners that fit a single method on one (task, fold, repeat, sample) and score it.

``ExperimentRunner`` is the base: it loads the split, instantiates the method, fits it
(tracking time/memory), evaluates the test metric, and assembles the result dict.
``OOFExperimentRunner`` additionally collects the out-of-fold ensemble-simulation artifact.
``Experiment.run`` (in ``experiment_constructor``) drives these via ``init_and_run``.
"""

from __future__ import annotations

import datetime
import traceback
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from autogluon.core.data.label_cleaner import LabelCleaner, LabelCleanerDummy
from autogluon.core.metrics import Scorer, get_metric
from pandas.api.types import is_integer_dtype

from tabarena.benchmark.task.metrics import default_eval_metric
from tabarena.utils.cache import AbstractCacheFunction, CacheFunctionDF, CacheFunctionDummy

if TYPE_CHECKING:
    from tabarena.benchmark.exec_models.base import AbstractExecModel
    from tabarena.benchmark.task import TaskWrapper
    from tabarena.models.warmup import WarmupReport


# TODO: make a dataclass so type hinter is happy with subclasses?
class ExperimentRunner:
    """Fit and evaluate a single method on one (task, fold, repeat, sample).

    Loads the train/test split, instantiates ``method_cls``, fits it (recording time + memory
    via the method's ``fit_custom``), evaluates the test ``metric_error``, and returns a result
    dict with predictions, metadata, and timing. Subclasses can hook into ``post_fit`` /
    ``post_evaluate`` to enrich the result. The usual entry point is the ``init_and_run``
    classmethod, which constructs a runner and calls ``run``.
    """

    def __init__(
        self,
        *,
        method_cls: type[AbstractExecModel],
        task: TaskWrapper,
        fold: int,
        task_name: str,
        method: str,
        repeat: int = 0,
        sample: int = 0,
        fit_args: dict | None = None,
        cleanup: bool = True,
        cacher: AbstractCacheFunction | None = None,
        debug_mode: bool = True,
        eval_metric_name: str | None = None,
        require_warmup: bool = True,
        warmup: bool = True,
        cleanup_on_failure: bool = False,
        cpu_budget_check: Literal["error", "warn", "off"] = "warn",
    ):
        """Configure the runner and load the split for ``(fold, repeat, sample)``.

        Parameters
        ----------
        method_cls:
            The method (``AbstractExecModel`` subclass) to instantiate and fit.
        task:
            The loaded task providing the data splits and problem metadata.
        fold, repeat, sample:
            The split coordinates to fit/evaluate on.
        task_name:
            Display name recorded on the result (used downstream as the ``dataset`` key).
        method:
            Name of the method/framework recorded on the result.
        fit_args:
            Kwargs forwarded to ``method_cls(...)`` (alongside ``problem_type`` / ``eval_metric``).
        cleanup:
            If True, call the model's ``cleanup`` after running (frees files/GPU memory).
        cacher:
            Cacher used to persist side artifacts (e.g. model failures); defaults to a no-op.
        debug_mode: bool, default True
            If True, will operate in a manner best suited for local model development.
            This mode will be friendly to local debuggers and will avoid subprocesses/threads
            and complex try/except logic.

            IF False, will operate in a manner best suited for large-scale benchmarking.
            This mode will try to record information when method's fail
            and might not work well with local debuggers.
        eval_metric_name: str, default None
            If provided, will override the default evaluation metric for the task.
            If None, will use the default metric based on the task's problem type.
        require_warmup: bool, default True
            If True, a warm-up that raised or left steps failed (``WarmupReport.status`` ``"failed"``
            or ``"partial"``) aborts the run before the timed fit with a ``RuntimeError`` naming the
            failed steps, so a cold timed section is never recorded silently (see
            ``check_warmup_report``); ``"none"`` (nothing to warm) and ``"disabled"`` pass. Set
            False to record the report and fit anyway.
        warmup: bool, default True
            If True, run the method's untimed environment warm-up before fitting
            (see ``run_warmup``).
        cleanup_on_failure: bool, default False
            If True, call the model's ``cleanup`` also when the run fails (after ``handle_failure``
            wrote the failure artifact), so a multi-item worker does not leak predictor
            directories or served GPU models. Off by default so local debugging keeps the
            artifacts of a failed fit; the SLURM runner turns it on.
        cpu_budget_check: {"error", "warn", "off"}, default "warn"
            What to do when the method's CPU budget (``num_cpus``) differs from the CPUs this
            process may run on (see ``run_cpu_budget_check``): raise, warn, or only record it.
            Local and debug runs warn; the SLURM runner sets ``"error"``.
        """
        if eval_metric_name is None:
            eval_metric_name = default_eval_metric(task.problem_type)
        if cacher is None:
            cacher = CacheFunctionDummy()

        self.method_cls = method_cls
        self.task = task
        self.fold = fold
        self.repeat = repeat
        self.sample = sample
        self.task_name = task_name
        self.method = method
        self.fit_args = fit_args or {}
        self.cleanup = cleanup
        self.eval_metric_name = eval_metric_name
        self.eval_metric: Scorer = get_metric(metric=self.eval_metric_name, problem_type=self.task.problem_type)
        self.model: AbstractExecModel | None = None
        self.task_split_idx = self.task.get_split_idx(fold=self.fold, repeat=self.repeat, sample=self.sample)
        self.cacher = cacher
        self.debug_mode = debug_mode
        self.require_warmup = require_warmup
        self.warmup = warmup
        self.cleanup_on_failure = cleanup_on_failure
        self.cpu_budget_check = cpu_budget_check
        self.cpu_thread_info: dict | None = None
        self.time_warmup_s: float | None = None
        self.warmup_report: WarmupReport | None = None
        self.timing_audit: dict | None = None
        # Post-fit memo of the lazily loaded ``(X_test, y_test)``; see ``_reload_post_fit_split``.
        self._post_fit_split: tuple[pd.DataFrame, pd.Series] | None = None

        # When lazy-loading, keep the split frames as None and (re)load them on demand; we only
        # materialize ``y`` here to fit the label cleaner.
        if self.task.lazy_load_data:
            self.X, self.y, self.X_test, self.y_test = None, None, None, None
            _, y, _, _ = self._train_test_split()
        else:
            self.X, self.y, self.X_test, self.y_test = self._train_test_split()
            y = self.y

        self.label_cleaner = LabelCleaner.construct(problem_type=self.task.problem_type, y=y)

    @classmethod
    def init_and_run(
        cls,
        method_cls: type[AbstractExecModel],
        task: TaskWrapper,
        fold: int,
        task_name: str,
        method: str,
        fit_args: dict | None = None,
        cleanup: bool = True,
        cacher: AbstractCacheFunction | None = None,
        debug_mode: bool = True,
        **kwargs,
    ) -> dict:
        """Construct a runner with the given config and immediately ``run`` it."""
        obj = cls(
            method_cls=method_cls,
            task=task,
            fold=fold,
            task_name=task_name,
            method=method,
            fit_args=fit_args,
            cleanup=cleanup,
            cacher=cacher,
            debug_mode=debug_mode,
            **kwargs,
        )
        return obj.run()

    def init_method(self) -> AbstractExecModel:
        """Instantiate ``method_cls`` for this task from ``fit_args``."""
        return self.method_cls(
            problem_type=self.task.problem_type,
            eval_metric=self.eval_metric,
            **self.fit_args,
        )

    # --- Data loading helpers ---------------------------------------------------------
    def _train_test_split(self) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Load ``(X, y, X_test, y_test)`` for this runner's fixed fold/repeat/sample."""
        return self.task.get_train_test_split(fold=self.fold, repeat=self.repeat, sample=self.sample)

    def _load_y_test(self) -> pd.Series:
        """Return the test labels, from the post-fit memo when data is lazy-loaded."""
        if self.task.lazy_load_data:
            return self._reload_post_fit_split()[1]
        return self.y_test

    def _load_x_test(self) -> pd.DataFrame:
        """Return the test features, from the post-fit memo when data is lazy-loaded."""
        if self.task.lazy_load_data:
            return self._reload_post_fit_split()[0]
        return self.X_test

    def _reload_post_fit_split(self) -> tuple[pd.DataFrame, pd.Series]:
        """The lazily loaded ``(X_test, y_test)`` after the fit, loaded once and kept until ``_run`` ends.

        ``evaluate``, ``post_evaluate`` and ``bag_artifact`` all read the test split after the fit;
        the memo loads it once instead of once per reader. It is filled on first use after the fit
        (never during the timed fit, whose own reload lives in ``fit_custom``; ``run_model_fit`` clears
        it first) and dropped in ``_run`` after ``post_evaluate``, so the peak memory of the timed fit
        is unchanged. Only the test half is kept.
        """
        if self._post_fit_split is None:
            _, _, X_test, y_test = self._train_test_split()
            self._post_fit_split = (X_test, y_test)
        return self._post_fit_split

    @property
    def split_seed(self):
        """We use the split index as a source for a seed that creates different randomness per split."""
        return self.task_split_idx

    # --- Fit / run lifecycle ----------------------------------------------------------
    def run_cpu_budget_check(self) -> dict:
        """Snapshot the thread configuration and check the method's CPU budget against the usable CPUs.

        Runs after the method is constructed and before the warm-up. The snapshot
        (``tabarena.utils.thread_utils.cpu_thread_info``) records the affinity count, the OMP-style
        thread variables and the pools already created; ``budget_check`` holds the outcome of
        ``check_cpu_budget`` for ``self.model.num_cpus_budget`` under ``cpu_budget_check``. A
        mismatch means the thread pools are sized from the machine rather than from the
        ``num_cpus`` the results report.
        """
        from tabarena.utils.thread_utils import check_cpu_budget, cpu_thread_info

        info = cpu_thread_info()
        # A runner built without ``__init__`` (test doubles) has no mode; a duck-typed model has no budget.
        mode = getattr(self, "cpu_budget_check", "warn")
        info["budget_check"] = check_cpu_budget(getattr(self.model, "num_cpus_budget", None), on_mismatch=mode)
        return info

    def run_warmup(self) -> WarmupReport:
        """Run the method's untimed environment warm-up (see ``AbstractExecModel.warmup_fn``).

        Runs after the method is constructed and before any fit timing starts, so warm-up
        cost (imports, JIT/kernel compilation, CUDA context init, Ray startup, shared weights)
        never counts toward ``time_train_s`` / ``time_infer_s`` or a fit time limit, matching
        the one-time per-environment costs a real deployment would not pay per fit. A failure is
        logged and recorded in the report; ``check_warmup_report`` decides whether the fit may
        proceed cold. It warms this process and
        disk-backed caches; parallel fold workers start cold unless the opt-in worker pool
        covered them (see ``tabarena.models.warmup`` for the full scope).

        Returns:
            The ``WarmupReport``. Its ``duration_s`` is ``None`` exactly where the warm-up did not
            run (``status`` ``"disabled"``, ``"none"`` or ``"failed"``), so ``time_warmup_s``
            keeps its meaning.
        """
        from tabarena.models.warmup import WarmupReport, run_warmup_fn

        if not self.warmup:
            return WarmupReport(status="disabled", label=self.method)
        warmup_fn = self.model.warmup_fn
        if warmup_fn is None:
            return WarmupReport(status="none", label=self.method)
        return run_warmup_fn(warmup_fn, label=self.method)

    def check_warmup_report(self, report: WarmupReport) -> None:
        """Raise when ``require_warmup`` is set and the warm-up did not fully succeed.

        ``"failed"`` (the warm-up raised) and ``"partial"`` (some steps failed) are the two states
        in which the timed fit would run cold; ``"ok"``, ``"none"`` and ``"disabled"`` pass.
        """
        if not getattr(self, "require_warmup", True) or report.status not in ("failed", "partial"):
            return
        detail = f"failed steps: {report.failed_steps}" if report.failed_steps else f"error: {report.error}"
        raise RuntimeError(
            f"Warm-up of method {self.method!r} did not succeed (status {report.status!r}; {detail}); the timed "
            "fit would run cold. Fix the warm-up (python -P -m tabarena.tools.audit_warmup --model ...) or pass "
            "require_warmup=False to fit anyway."
        )

    def run_model_fit(self) -> dict:
        """Fit the model and predict on the test split (via the method's ``fit_custom``).

        Passes the loaded frames directly, or a lazy-load callback when the task lazy-loads
        its data (so the large arrays are only materialized inside ``fit_custom``).
        """
        self._post_fit_split = None
        if self.task.lazy_load_data:
            lazy_load_function = self._lazy_load_for_run_model_fit
            X, y, X_test = None, None, None
        else:
            lazy_load_function = None
            X, y, X_test = self.X, self.y, self.X_test
        return self.model.fit_custom(
            X=X, y=y, X_test=X_test, split_seed=self.split_seed, lazy_load_function=lazy_load_function
        )

    def _lazy_load_for_run_model_fit(self):
        """Lazy-load callback for ``fit_custom``: return ``(X, y, X_test)`` for this split."""
        X, y, X_test, _ = self._train_test_split()
        return X, y, X_test

    def run(self) -> dict:
        """Run the full fit and evaluate flow and return the result dict (cleaning up if enabled).

        The model's ``cleanup`` runs after a successful ``_run`` when ``cleanup`` is set. After a
        failure it runs only when ``cleanup_on_failure`` is set, after ``handle_failure`` inside
        ``_run`` wrote the failure artifact; an error from that cleanup is printed and the original
        exception propagates.
        """
        try:
            out = self._run()
        except Exception:
            if self.cleanup_on_failure:
                self._cleanup_after_failure()
            raise
        if self.cleanup:
            self._cleanup()
        return out

    def _run(self) -> dict:
        """Instantiate, fit, evaluate, and assemble the result dict for this split.

        The served model stays resident through ``evaluate`` and ``post_evaluate`` (metadata, OOF and
        bag artifacts reuse it); ``run`` releases it afterwards through ``_cleanup``.
        """
        utc_time = datetime.datetime.now(datetime.UTC)
        time_start_str = utc_time.strftime("%Y-%m-%d %H:%M:%S")
        time_start = utc_time.timestamp()
        self.model = self.init_method()
        self.cpu_thread_info = self.run_cpu_budget_check()
        self.warmup_report = self.run_warmup()
        self.time_warmup_s = self.warmup_report.duration_s
        self.check_warmup_report(self.warmup_report)
        try:
            out = self.run_model_fit()
        except Exception as exc:
            if not self.debug_mode:
                # Only do this in benchmark mode, since it could mess with a local debugger.
                self.handle_failure(exc=exc)
            raise
        # The environment audit of the timed sections belongs to the experiment metadata, not to the
        # method's result keys.
        self.timing_audit = out.pop("timing_audit", None)
        out = self.post_fit(out=out)

        out["metric_error"] = self.evaluate(
            y_true=self._load_y_test(),
            y_pred=out["predictions"],
            y_pred_proba=out["probabilities"],
        )

        out = self.post_evaluate(out=out)
        self._post_fit_split = None  # the memoized test split is not needed past post_evaluate
        out["experiment_metadata"] = self._experiment_metadata(time_start=time_start, time_start_str=time_start_str)
        return self.convert_to_output(out=out)

    def _cleanup_after_failure(self) -> None:
        """Best-effort ``cleanup`` of the model after a failed run; never raises."""
        model = self.model
        if model is None:
            return
        try:
            model.cleanup()
        except Exception:
            print("Cleanup after the failed run raised; the original exception is re-raised:")
            traceback.print_exc()

    def handle_failure(self, exc: Exception):
        """Persist any per-model failure artifacts to the cache (benchmark mode only).

        No-op unless the cacher has a ``cache_path`` and the model exposes ``model_failures``.
        """
        # TODO: This is autogluon specific, make a subclass AGExperimentRunner?
        failures = self.model.failure_artifact
        if not hasattr(self.cacher, "cache_path") or self.cacher.cache_path is None:
            return
        if failures is None:
            try:
                failures = self.model.get_metadata_failure()
            except Exception:
                return
        if failures is None:
            return
        if "model_failures" in failures:
            model_failures = failures["model_failures"]
            if len(model_failures) > 0:
                cacher_model_failures = CacheFunctionDF(cache_path=self.cacher.cache_path, cache_name="model_failures")
                cacher_model_failures.save_cache(data=model_failures)

    # --- Result hooks -----------------------------------------------------------------
    def post_fit(self, out: dict) -> dict:
        """Hook run on the fit output before evaluation. Default: no-op."""
        return out

    def post_evaluate(self, out: dict) -> dict:
        """Attach task/method metadata (and optional method metadata + val error) to ``out``."""
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
        """Build the timing/identity metadata block (start/end timestamps + duration).

        ``total_duration`` includes the warm-up; ``time_warmup_s`` records it separately
        (None = nothing was warmed) so the untimed share stays auditable, and ``warmup_report``
        lists what the warm-up imported, primed and dummy-fitted (``None`` when the runner never
        got to the warm-up). ``timing_audit`` is the environment audit ``fit_custom`` took around the
        fit and predict timers (``None`` when the method's ``fit_custom`` returned none).
        ``cpu_thread_info`` is the thread-configuration snapshot and CPU budget check taken before
        the warm-up (see ``run_cpu_budget_check``).
        """
        time_end = datetime.datetime.now(datetime.UTC).timestamp()
        return {
            "experiment_cls": self.__class__.__name__,
            "method_cls": self.method_cls.__name__,
            "time_start": time_start,
            "time_end": time_end,
            "total_duration": time_end - time_start,
            "time_start_str": time_start_str,
            "time_warmup_s": self.time_warmup_s,
            "warmup_report": self.warmup_report.to_dict() if self.warmup_report is not None else None,
            "timing_audit": getattr(self, "timing_audit", None),
            "cpu_thread_info": getattr(self, "cpu_thread_info", None),
        }

    def convert_to_output(self, out: dict) -> dict:
        """Drop the raw predictions/probabilities from the result (kept only for evaluation)."""
        out.pop("predictions")
        out.pop("probabilities")
        return out

    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_pred_proba: pd.Series | pd.DataFrame | None,
    ) -> float:
        """Compute this task's metric error for the predictions (see the module ``evaluate``)."""
        return evaluate(
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            scorer=self.eval_metric,
            label_cleaner=self.label_cleaner,
            problem_type=self.task.problem_type,
        )

    def _cleanup(self):
        """Release any resources held by the fitted model."""
        self.model.cleanup()


class OOFExperimentRunner(ExperimentRunner):
    """``ExperimentRunner`` that also collects the out-of-fold ensemble-simulation artifact.

    On top of the base result, ``post_evaluate`` gathers the model's OOF validation predictions
    and test predictions (in label-cleaned space) needed for TabArena ensemble simulation,
    optionally including per-bagged-child info, and optionally shrinking it for storage.
    """

    def __init__(
        self,
        *,
        compute_simulation_artifacts: bool = True,
        compute_bag_info: bool = True,
        optimize_simulation_artifacts_memory: bool = True,
        **kwargs,
    ):
        """Configure simulation-artifact collection; ``**kwargs`` go to ``ExperimentRunner``.

        Parameters
        ----------
        compute_simulation_artifacts:
            If True (and the model supports OOF), build the ensemble-simulation artifact.
        compute_bag_info:
            If True, include per-bagged-child OOF/test info when the model exposes it.
        optimize_simulation_artifacts_memory:
            If True, shrink the artifact in place (downcast indices, drop pandas wrappers,
            float32 probabilities) before returning it.
        """
        super().__init__(**kwargs)
        self.compute_simulation_artifacts = compute_simulation_artifacts
        self.compute_bag_info = compute_bag_info
        self.optimize_simulation_artifacts_memory = optimize_simulation_artifacts_memory

    def post_evaluate(self, out: dict) -> dict:
        """Attach the ensemble-simulation artifact (OOF + test predictions) to ``out``.

        Built only when simulation artifacts are requested and the model can produce OOF
        predictions; otherwise ``simulation_artifacts`` stays ``None`` (set by the base). The timed
        test predictions are handed to ``bag_artifact`` so a bag whose output equals its single
        child's can reuse them instead of predicting again; the path the model took is then written
        to ``method_metadata["per_child_test_source"]`` (the metadata was collected before the
        artifact).
        """
        out = super().post_evaluate(out=out)
        if not (self.compute_simulation_artifacts and self.model.can_get_oof):
            return out

        simulation_artifact = self.model.get_oof()

        # Test predictions in the model's internal (label-cleaned) space.
        if self.task.problem_type == "regression":
            simulation_artifact["pred_proba_dict_test"] = self.label_cleaner.transform(out["predictions"])
        else:
            simulation_artifact["pred_proba_dict_test"] = self.label_cleaner.transform_proba(
                out["probabilities"], as_pandas=True
            )
            if self.task.problem_type == "binary":
                simulation_artifact["pred_proba_dict_test"] = simulation_artifact["pred_proba_dict_test"].iloc[:, 1]

        simulation_artifact["y_test"] = self.label_cleaner.transform(self._load_y_test())

        if self.optimize_simulation_artifacts_memory:
            self._optimize_simulation_artifact_memory(simulation_artifact)

        simulation_artifact["label"] = self.task.label
        simulation_artifact["metric"] = self.eval_metric_name

        if self.compute_bag_info and self.model.can_get_per_child_oof and self.model.can_get_per_child_val_idx:
            simulation_artifact["bag_info"] = self.model.bag_artifact(
                X_test=self._load_x_test(),
                y_pred=out["predictions"],
                y_pred_proba=out["probabilities"],
            )
            source = getattr(self.model, "per_child_test_source", None)
            if source is not None and isinstance(out.get("method_metadata"), dict):
                out["method_metadata"]["per_child_test_source"] = source

        simulation_artifact["pred_proba_dict_val"] = {self.method: simulation_artifact["pred_proba_dict_val"]}
        simulation_artifact["pred_proba_dict_test"] = {self.method: simulation_artifact["pred_proba_dict_test"]}
        out["simulation_artifacts"] = simulation_artifact
        return out

    @staticmethod
    def _optimize_simulation_artifact_memory(artifact: dict) -> None:
        """Shrink a simulation artifact in place for cheaper storage/transfer.

        Downcasts the ``y`` indices to integers (stored separately as ``*_idx``), replaces the
        pandas ``y`` / predicted-probability objects with raw numpy arrays, and casts predicted
        probabilities to ``float32``.
        """
        artifact["y_test"].index = pd.to_numeric(artifact["y_test"].index, downcast="integer")
        artifact["y_val"].index = pd.to_numeric(artifact["y_val"].index, downcast="integer")

        artifact["y_test_idx"] = artifact["y_test"].index.values
        artifact["y_val_idx"] = artifact["y_val"].index.values

        artifact["y_test"] = artifact["y_test"].values
        artifact["y_val"] = artifact["y_val"].values
        if is_integer_dtype(artifact["y_test"]):
            artifact["y_test"] = pd.to_numeric(artifact["y_test"], downcast="integer")
        if is_integer_dtype(artifact["y_val"]):
            artifact["y_val"] = pd.to_numeric(artifact["y_val"], downcast="integer")

        artifact["pred_proba_dict_test"] = artifact["pred_proba_dict_test"].astype(np.float32)
        artifact["pred_proba_dict_val"] = artifact["pred_proba_dict_val"].astype(np.float32)

        artifact["pred_proba_dict_test"] = artifact["pred_proba_dict_test"].values
        artifact["pred_proba_dict_val"] = artifact["pred_proba_dict_val"].values


def evaluate(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_pred_proba: pd.Series | pd.DataFrame | None,
    scorer: Scorer,
    problem_type: str,
    label_cleaner: LabelCleaner | None = None,
) -> float:
    """Return the metric *error* for predictions, in the scorer's expected space.

    Cleans ``y_true`` (and, as needed, the predictions) via ``label_cleaner`` so the inputs
    match the model's internal label space, then scores: point predictions for ``needs_pred``
    metrics, the positive-class probability for binary, else the full probability matrix.
    """
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
