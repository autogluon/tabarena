"""CTBoost model wrapper for TabArena.

CTBoost is a conditional-inference-tree gradient booster by Markus Maier and
the CTBoost contributors. The Apache-2.0 implementation is available at
https://github.com/captnmarkus/ctboost.
"""

from __future__ import annotations

import math
import os
import threading
import time
from contextlib import contextmanager
from itertools import combinations
from typing import Any

import numpy as np
from autogluon.core.models import AbstractModel

_HISTOGRAM_THREAD_ENV = "CTBOOST_HIST_THREADS"
_HISTOGRAM_THREAD_LOCK = threading.Lock()
_MIN_TRAINING_BUDGET_FRACTION = 0.4
_TRAINING_TIME_LIMIT_FRACTION = 0.95
_PAIR_BUDGET_PARAM = "tabarena_categorical_pair_budget"
_PAIR_BUDGET_LIMIT = 4
_PAIR_CANDIDATE_COLUMN_LIMIT = 16
_PAIR_JOINT_CARDINALITY_LIMIT = 4096


def _stopping_metric_name(metric: Any) -> str | None:
    """Return AutoGluon's scorer name in a comparison-friendly form."""
    if metric is None:
        return None
    name = getattr(metric, "name", None)
    if callable(name):
        try:
            name = name()
        except TypeError:
            name = None
    if not isinstance(name, str):
        if isinstance(metric, str):
            name = metric
        else:
            name = getattr(metric, "__name__", None)
    if not isinstance(name, str) or not name.strip():
        return None
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _ctboost_eval_metric(problem_type: Any, stopping_metric: Any) -> str | None:
    """Translate supported AutoGluon stopping scorers to CTBoost metrics."""
    resolved_problem_type = str(problem_type).strip().lower()
    resolved_metric = _stopping_metric_name(stopping_metric)
    mappings = {
        "binary": {"roc_auc": "AUC", "log_loss": "Logloss"},
        "multiclass": {"log_loss": "MultiClass"},
        "regression": {
            "rmse": "RMSE",
            "root_mean_squared_error": "RMSE",
        },
    }
    return mappings.get(resolved_problem_type, {}).get(resolved_metric)


def _resolve_time_limit(time_limit: float | None) -> float | None:
    if time_limit is None:
        return None
    resolved = float(time_limit)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("time_limit must be finite and positive when provided")
    return resolved


def _callback_list(callbacks: Any) -> list[Any]:
    if callbacks is None:
        return []
    if callable(callbacks):
        return [callbacks]
    try:
        return list(callbacks)
    except TypeError as exc:
        raise TypeError("callbacks must be callable or an iterable of callables") from exc


def _raise_time_limit_exceeded() -> None:
    from autogluon.core.utils.exceptions import TimeLimitExceeded

    raise TimeLimitExceeded("Insufficient AutoGluon fit budget remains after CTBoost adapter setup")


def _deadline_callback(deadline: float, training_started_at: float) -> Any:
    """Stop when the budget cannot safely fit two average tree iterations."""

    def _stop_at_deadline(env: Any) -> bool:
        now = time.monotonic()
        if now >= deadline:
            return True
        completed_iterations = max(1, int(env.iteration) - int(env.begin_iteration) + 1)
        average_iteration_time = max(0.0, (now - training_started_at) / completed_iterations)
        return now + 2.0 * average_iteration_time >= deadline

    return _stop_at_deadline


@contextmanager
def _ctboost_histogram_threads(num_cpus: Any):
    """Keep native histogram workers inside TabArena's per-fit CPU budget."""
    resolved = 1 if num_cpus is None else max(1, int(num_cpus))
    with _HISTOGRAM_THREAD_LOCK:
        previous = os.environ.get(_HISTOGRAM_THREAD_ENV)
        os.environ[_HISTOGRAM_THREAD_ENV] = str(resolved)
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(_HISTOGRAM_THREAD_ENV, None)
            else:
                os.environ[_HISTOGRAM_THREAD_ENV] = previous


def _categorical_columns(frame: Any) -> list[str]:
    """Return columns that should use CTBoost's native categorical pipeline."""
    columns: list[str] = []
    for name, dtype in frame.dtypes.items():
        dtype_name = str(dtype).lower()
        if dtype_name in {"object", "category", "string"} or dtype_name.startswith("string["):
            columns.append(str(name))
    return columns


def _resolve_categorical_pair_budget(value: Any) -> int:
    """Validate the small adapter-only categorical-pair budget."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{_PAIR_BUDGET_PARAM} must be an integer")
    resolved = int(value)
    if not 0 <= resolved <= _PAIR_BUDGET_LIMIT:
        raise ValueError(f"{_PAIR_BUDGET_PARAM} must be between 0 and {_PAIR_BUDGET_LIMIT}")
    return resolved


def normalize_tabarena_frame(
    frame: Any,
    *,
    categorical_columns: list[str] | tuple[str, ...] | None = None,
) -> tuple[Any, list[str]]:
    """Preserve native categorical, missing, boolean, and unseen-value semantics."""
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    normalized_columns = [str(name) for name in frame.columns]
    if list(frame.columns) == normalized_columns:
        normalized = frame
    else:
        normalized = frame.copy(deep=False)
        normalized.columns = normalized_columns

    resolved_categoricals = (
        _categorical_columns(normalized) if categorical_columns is None else [str(name) for name in categorical_columns]
    )
    missing_columns = [name for name in resolved_categoricals if name not in normalized.columns]
    if missing_columns:
        raise ValueError(
            "TabArena prediction data is missing categorical columns seen during fit: " + ", ".join(missing_columns)
        )
    return normalized, resolved_categoricals


def _bounded_categorical_pairs(
    frame: Any,
    categorical_columns: list[str],
    *,
    max_pairs: int,
    max_joint_cardinality: int = _PAIR_JOINT_CARDINALITY_LIMIT,
) -> list[list[str]]:
    """Select a deterministic set of inexpensive categorical pairs from training data."""
    budget = max(0, int(max_pairs))
    if budget == 0 or len(categorical_columns) < 2:
        return []

    column_order = {name: index for index, name in enumerate(categorical_columns)}
    cardinalities: list[tuple[int, int, str]] = []
    for name in categorical_columns:
        cardinality = int(frame[name].nunique(dropna=False))
        if 1 < cardinality <= max_joint_cardinality:
            cardinalities.append((cardinality, column_order[name], name))

    cardinalities.sort()
    candidates = cardinalities[:_PAIR_CANDIDATE_COLUMN_LIMIT]
    ranked_pairs: list[tuple[int, int, int, str, str]] = []
    for left, right in combinations(candidates, 2):
        joint_upper_bound = left[0] * right[0]
        if joint_upper_bound <= max_joint_cardinality:
            left_order, right_order = sorted((left[1], right[1]))
            left_name = categorical_columns[left_order]
            right_name = categorical_columns[right_order]
            ranked_pairs.append((joint_upper_bound, left_order, right_order, left_name, right_name))

    ranked_pairs.sort()
    return [[left, right] for _, _, _, left, right in ranked_pairs[:budget]]


class CTBoostModel(AbstractModel):
    """AutoGluon wrapper for CTBoost's classifier and regressor estimators."""

    ag_key = "CTB"
    ag_name = "CTBoost"
    ag_priority = 65
    seed_name = "random_seed"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    _default_auxiliary_params_extra = {
        "valid_raw_types": ["bool", "int", "float", "category", "object"],
        "ignored_type_group_special": ["datetime_as_object"],
    }
    default_resources_physical_cores_only = True
    default_num_gpus = 0

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._ctboost_categorical_columns: list[str] = []

    def _preprocess(self, X: Any, is_train: bool = False, **kwargs: Any) -> Any:
        X = super()._preprocess(X, **kwargs)
        categorical_columns = None if is_train else self._ctboost_categorical_columns
        X, resolved = normalize_tabarena_frame(X, categorical_columns=categorical_columns)
        if is_train:
            self._ctboost_categorical_columns = resolved
        return X

    def _fit(
        self,
        X: Any,
        y: Any,
        X_val: Any = None,
        y_val: Any = None,
        sample_weight: Any = None,
        num_cpus: int = 1,
        num_gpus: float = 0,
        time_limit: float | None = None,
        callbacks: Any = None,
        **kwargs: Any,
    ) -> None:
        del num_gpus, kwargs
        resolved_time_limit = _resolve_time_limit(time_limit)
        training_deadline = None
        if resolved_time_limit is not None:
            fit_started_at = time.monotonic()
            training_deadline = fit_started_at + resolved_time_limit * _TRAINING_TIME_LIMIT_FRACTION

        from ctboost import CTBoostClassifier, CTBoostRegressor

        X_train = self.preprocess(X, y=y, is_train=True)
        X_validation = None
        if X_val is not None:
            X_validation = self.preprocess(X_val, is_train=False)

        params = dict(self._get_model_params())
        early_stopping_rounds = int(params.pop("early_stopping_rounds", 50))
        categorical_pair_budget = _resolve_categorical_pair_budget(params.pop(_PAIR_BUDGET_PARAM, 0))
        configured_callbacks = _callback_list(params.pop("callbacks", None))
        configured_callbacks.extend(_callback_list(callbacks))
        if "eval_metric" not in params:
            eval_metric = _ctboost_eval_metric(self.problem_type, getattr(self, "stopping_metric", None))
            if eval_metric is not None:
                params["eval_metric"] = eval_metric
        params["cat_features"] = self._ctboost_categorical_columns or None
        if (
            categorical_pair_budget
            and "categorical_combinations" not in params
            and not params.get("pairwise_categorical_combinations", False)
        ):
            categorical_pairs = _bounded_categorical_pairs(
                X_train,
                self._ctboost_categorical_columns,
                max_pairs=categorical_pair_budget,
            )
            if categorical_pairs:
                params["categorical_combinations"] = categorical_pairs
        params["task_type"] = "CPU"

        if self.problem_type == "regression":
            self.model = CTBoostRegressor(**params)
        else:
            self.model = CTBoostClassifier(**params)

        fit_kwargs: dict[str, Any] = {"sample_weight": sample_weight}
        if X_validation is not None and y_val is not None:
            fit_kwargs.update(
                {
                    "eval_set": (X_validation, np.asarray(y_val)),
                    "early_stopping_rounds": early_stopping_rounds,
                }
            )
        with _ctboost_histogram_threads(num_cpus):
            if training_deadline is not None:
                training_started_at = time.monotonic()
                remaining_time = training_deadline - training_started_at
                if remaining_time <= resolved_time_limit * _MIN_TRAINING_BUDGET_FRACTION:
                    _raise_time_limit_exceeded()
                configured_callbacks.append(_deadline_callback(training_deadline, training_started_at))
            if configured_callbacks:
                fit_kwargs["callbacks"] = configured_callbacks
            self.model.fit(X_train, np.asarray(y), **fit_kwargs)
        best_iteration = self.model.get_best_iteration()
        if best_iteration is not None and int(best_iteration) >= 0:
            self.params_trained["iterations"] = int(best_iteration) + 1

    def _set_default_params(self) -> None:
        defaults = {
            "iterations": 1000,
            "learning_rate": 0.05,
            "max_depth": 6,
            "alpha": 0.05,
            "lambda_l2": 1.0,
            "subsample": 0.8,
            "bootstrap_type": "Bernoulli",
            "ordered_ctr": True,
            "max_cat_threshold": 64,
            "early_stopping_rounds": 50,
            "verbose": False,
        }
        for name, value in defaults.items():
            self._set_default_param_value(name, value)

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: Any,
        hyperparameters: dict[str, Any] | None = None,
        num_classes: int | None = 1,
        **kwargs: Any,
    ) -> int:
        """Conservatively estimate peak fit memory for fold scheduling."""
        del kwargs
        params = dict(hyperparameters or {})
        rows, raw_columns = (int(value) for value in X.shape)
        classes = max(1, int(num_classes or 1))
        try:
            from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage

            input_bytes = int(get_approximate_df_mem_usage(X).sum())
        except (ImportError, AttributeError, TypeError):
            memory_usage = getattr(X, "memory_usage", None)
            input_bytes = (
                int(memory_usage(index=True, deep=True).sum()) if callable(memory_usage) else rows * raw_columns * 8
            )

        categorical_columns = _categorical_columns(X)
        categorical_count = len(categorical_columns)
        one_hot_max_size = max(0, int(params.get("one_hot_max_size", 0)))
        max_cat_threshold = max(0, int(params.get("max_cat_threshold", 64)))
        ordered_ctr = bool(params.get("ordered_ctr", True))
        transformed_columns = raw_columns - categorical_count
        for name in categorical_columns:
            # CTBoost either one-hot expands a low-cardinality source or keeps
            # one encoded source plus one ordered-CTR column per class.
            cardinality = max(1, int(X[name].nunique(dropna=False)))
            if max_cat_threshold > 1:
                cardinality = min(cardinality, max_cat_threshold)
            if 0 < cardinality <= one_hot_max_size:
                transformed_columns += cardinality
            else:
                transformed_columns += 1 + (classes if ordered_ctr else 0)

        pair_budget = _resolve_categorical_pair_budget(params.get(_PAIR_BUDGET_PARAM, 0))
        transformed_columns += min(pair_budget, _PAIR_BUDGET_LIMIT) * (1 + (classes if ordered_ctr else 0))
        columns = max(raw_columns, transformed_columns)
        max_bins = max(1, int(params.get("max_bins", 256)))
        bin_width = 1 if max_bins <= 256 else 2 if max_bins <= 65_535 else 4
        quantized_bytes = rows * columns * bin_width
        statistic_bytes = rows * classes * 8 * 6
        histogram_bytes = columns * max_bins * classes * 8 * 3

        depth = max(0, int(params.get("max_depth", params.get("depth", 6))))
        depth_leaves = 1 << min(depth, 20)
        configured_leaves = int(params.get("max_leaves", 0) or 0)
        leaves = min(depth_leaves, configured_leaves) if configured_leaves > 0 else depth_leaves
        iterations = max(1, int(params.get("iterations", params.get("n_estimators", 1000))))
        tree_bytes = iterations * classes * max(1, 2 * leaves - 1) * 64

        baseline_bytes = 512 * 1024 * 1024
        return int(
            baseline_bytes + 4 * input_bytes + 2 * quantized_bytes + statistic_bytes + histogram_bytes + tree_bytes
        )
