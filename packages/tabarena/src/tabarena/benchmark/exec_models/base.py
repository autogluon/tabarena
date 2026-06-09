from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from autogluon.core.data.label_cleaner import LabelCleaner, LabelCleanerDummy
from autogluon.features import AutoMLPipelineFeatureGenerator

from tabarena.benchmark.exec_models.utils import _apply_inv_perm, _make_perm
from tabarena.utils.time_utils import Timer

if TYPE_CHECKING:
    from collections.abc import Callable

    from autogluon.core.metrics import Scorer


class AbstractExecModel:
    """Base class for a benchmarked *method* (an "execution model").

    A subclass wraps some underlying model behind a small, uniform interface so the
    experiment runner can fit it, collect predictions, and record metadata the same way
    for every method. Concrete subclasses only have to implement the protected hooks
    (``_fit`` and ``_predict`` / ``_predict_proba``); everything else here is shared
    orchestration.

    Lifecycle (driven by the experiment runner):

    1. ``__init__`` — configure preprocessing and inference-shuffle behavior.
    2. ``fit_custom`` — the end-to-end harness: optionally shuffle features, fit (while
       tracking time + memory), then predict on the test data and undo any shuffling.
    3. ``cleanup`` — release any resources (files, GPU memory, ...).

    The public ``fit`` / ``predict`` / ``predict_proba`` methods handle label and feature
    (pre)processing, then delegate to the ``_fit`` / ``_predict`` / ``_predict_proba``
    hooks. Preprocessing can be turned off via the ``preprocess_*`` flags when the
    underlying model does its own.

    Optional capabilities are advertised by the ``can_get_*`` class flags below. When a
    flag is ``True`` the runner calls the matching method (e.g. ``can_get_oof`` ->
    ``get_oof``); subclasses that flip a flag must implement the corresponding method.
    ``get_metadata`` is detected separately via ``hasattr`` by the runner.
    """

    # --- Optional-capability flags (queried by the experiment runner) -----------------
    can_get_error_val = False
    """Whether the method can report a validation metric error (see ``get_metric_error_val``)."""
    can_get_oof = False
    """Whether the method can produce out-of-fold predictions (see ``get_oof``)."""
    can_get_per_child_oof = False
    """Whether per-bagged-child out-of-fold predictions are available (see ``bag_artifact``)."""
    can_get_per_child_test = False
    """Whether per-bagged-child test predictions are available (see ``bag_artifact``)."""
    can_get_per_child_val_idx = False
    """Whether per-bagged-child validation indices are available (see ``bag_artifact``)."""

    _can_use_data_in_place: bool
    """Whether the training data may be used in place rather than defensively copied.
    Set to True by ``fit_custom`` when data is lazy-loaded (and thus owned by this
    object), letting wrappers skip a copy of the training frame.
    """
    _split_seed: Literal["NOTSET"] | None | int
    """The per-split seed passed to ``fit_custom`` (``"NOTSET"`` until a fit runs).
    Source of per-split randomness, e.g. for ``_shuffle_features``.
    """
    label_cleaner: LabelCleaner | None
    """The fitted label cleaner (a no-op ``LabelCleanerDummy`` when ``preprocess_label`` is off).
    Set during ``fit``; used to encode/decode labels and probabilities.
    """
    _feature_generator: AutoMLPipelineFeatureGenerator | None
    """The fitted feature generator when ``preprocess_data`` is on, else ``None``.
    Set during ``fit`` and applied to feature inputs via ``transform_X``.
    """
    failure_artifact: dict | None
    """Optional record of a fit failure, populated by some wrappers (e.g. ``AGSingleWrapper.post_fit``).
    Read by the experiment runner when handling failures.
    """

    # --- Preprocessing / inference-shuffle config (class-level defaults) ---------------
    # These are plain class attributes so a subclass can change a default by simply
    # re-declaring it (e.g. ``preprocess_data = False``) instead of threading it through
    # every ``__init__``. Each one can still be overridden per instance by passing the same
    # name as a keyword argument to ``__init__`` (see ``_CONFIG_ATTRS``).
    preprocess_data: bool = True
    """If True, fit an ``AutoMLPipelineFeatureGenerator`` on ``X`` and transform all feature
    inputs through it (see ``transform_X``). Subclasses whose underlying model preprocesses
    features themselves set this to False."""
    preprocess_label: bool = True
    """If True, clean/encode the label via an AutoGluon ``LabelCleaner``; set False to pass
    labels through unchanged."""
    shuffle_test: bool = True
    """If True, deterministically permute the test rows before inference and invert the
    permutation on the outputs (guards against models that depend on row order). See
    ``_shuffle_test_rows``."""
    shuffle_seed: int = 0
    """Seed for the test-row permutation (see ``shuffle_test``)."""
    reset_index_test: bool = True
    """If True, reset the test frame's index before inference (the original index is
    restored on the outputs)."""
    shuffle_features: bool = False
    """If True, deterministically permute the feature columns (per split) before fitting.
    Requires a ``split_seed`` in ``fit_custom``. See ``_shuffle_features``."""

    #: Config attributes (above) that ``__init__`` accepts as per-instance overrides.
    _CONFIG_ATTRS = (
        "preprocess_data",
        "preprocess_label",
        "shuffle_test",
        "shuffle_seed",
        "reset_index_test",
        "shuffle_features",
    )

    def __init__(
        self,
        problem_type: str,
        eval_metric: Scorer,
        **kwargs,
    ):
        """Configure the method.

        Parameters
        ----------
        problem_type:
            One of ``"binary"``, ``"multiclass"``, ``"regression"``.
        eval_metric:
            AutoGluon scorer used for evaluation (and, for some wrappers, model fitting).
        **kwargs:
            Per-instance overrides for any of the class-level config attributes
            (``preprocess_data``, ``preprocess_label``, ``shuffle_test``, ``shuffle_seed``,
            ``reset_index_test``, ``shuffle_features``); the remaining keys are forwarded up
            the MRO (used by cooperative mixins such as the validation protocol on the
            AutoGluon wrappers).
        """
        # Apply any per-instance overrides of the class-level config defaults, then forward
        # the remaining kwargs up the MRO (e.g. to the validation-protocol mixin).
        for name in self._CONFIG_ATTRS:
            if name in kwargs:
                setattr(self, name, kwargs.pop(name))
        super().__init__(**kwargs)
        self.problem_type = problem_type
        self.eval_metric = eval_metric

        # Defaults for internal state
        self._can_use_data_in_place = False
        self._split_seed = "NOTSET"
        self.label_cleaner = None
        self._feature_generator = None
        self.failure_artifact = None

    # --- Label / feature (pre)processing ----------------------------------------------
    def transform_y(self, y: pd.Series) -> pd.Series:
        """Encode labels into the model's internal label space via the label cleaner."""
        return self.label_cleaner.transform(y)

    def inverse_transform_y(self, y: pd.Series) -> pd.Series:
        """Decode internal label predictions back to the original label space."""
        return self.label_cleaner.inverse_transform(y)

    def transform_y_pred_proba(self, y_pred_proba: pd.DataFrame) -> pd.DataFrame:
        """Map probabilities from the original class space into the internal one."""
        return self.label_cleaner.transform_proba(y_pred_proba, as_pandas=True)

    def inverse_transform_y_pred_proba(self, y_pred_proba: pd.DataFrame) -> pd.DataFrame:
        """Map probabilities from the internal class space back to the original one."""
        return self.label_cleaner.inverse_transform_proba(y_pred_proba, as_pandas=True)

    def transform_X(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted feature generator to ``X`` (a no-op if ``preprocess_data`` is off)."""
        if self.preprocess_data:
            return self._feature_generator.transform(X)
        return X

    def _preprocess_fit_transform(self, X: pd.DataFrame, y: pd.Series):
        """Fit the label cleaner and (optionally) feature generator, then transform ``X``/``y``.

        Called once at the start of ``fit``. Sets ``self.label_cleaner`` and, when
        ``preprocess_data`` is enabled, ``self._feature_generator``.
        """
        if self.preprocess_label:
            self.label_cleaner = LabelCleaner.construct(problem_type=self.problem_type, y=y)
        else:
            self.label_cleaner = LabelCleanerDummy(problem_type=self.problem_type)
        if self.preprocess_data:
            self._feature_generator = AutoMLPipelineFeatureGenerator()
            X = self._feature_generator.fit_transform(X=X, y=y)
        y = self.transform_y(y)
        return X, y

    # --- Fit / predict lifecycle hooks (overridable) ----------------------------------
    def post_fit(self, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame):
        """Hook run after fitting, before inference. Default: no-op.

        ``X``/``y`` are the (reloaded) training data and ``X_test`` the test features, both
        in the order/layout used for inference.
        """

    def pre_predict(self):
        """Hook run immediately before inference (e.g. to persist a model). Default: no-op."""

    def post_predict(self):
        """Hook run immediately after inference (e.g. to unpersist a model). Default: no-op."""

    # --- End-to-end execution harness -------------------------------------------------
    def fit_custom(
        self,
        X: pd.DataFrame | None,
        y: pd.Series | None,
        X_test: pd.DataFrame | None,
        *,
        split_seed: int | None = None,
        lazy_load_function: Callable | None = None,
    ) -> dict:
        """Fit the method and predict on ``X_test``, recording timing and memory usage.

        This is the single entry point used by the experiment runner. It fits the model
        (via ``fit`` -> ``_fit``) while tracking wall-clock time and CPU/GPU memory, then
        produces predictions (probabilities for classification, point predictions for
        regression) on ``X_test``, undoing any test-row shuffle so outputs align with the
        caller's original ``X_test``.

        Parameters
        ----------
        X, y, X_test:
            Training features/labels and test features. Must all be ``None`` iff
            ``lazy_load_function`` is provided.
        split_seed:
            If not None, the per-split seed used to shuffle features (required when
            ``shuffle_features`` is True).
        lazy_load_function:
            If provided, a callable returning ``(X, y, X_test)`` used to load the data only
            when needed (to save memory). The data is loaded once for fitting and reloaded
            afterwards so the training arrays can be used in place.

        Returns:
        -------
        dict
            Keys: ``predictions``, ``probabilities`` (None for regression), ``time_train_s``,
            ``time_infer_s``, ``memory_usage``.
        """
        from tabarena.utils.memory_utils import CpuMemoryTracker, GpuMemoryTracker

        self._split_seed = split_seed

        if lazy_load_function is not None:
            assert X is None and y is None and X_test is None, "If lazy_load_function is provided, X and y must be None"  # noqa: PT018
            X, y, _ = lazy_load_function()
            self._can_use_data_in_place = True

        X, shuffled_features = self._shuffle_features(X, split_seed=split_seed)

        with CpuMemoryTracker() as cpu_tracker, GpuMemoryTracker(device=0) as gpu_tracker, Timer() as timer_fit:
            self.fit(X, y)

        # Reload all, allows X,y to be used in-place
        if lazy_load_function is not None:
            del X, y, X_test  # Free memory from previous load
            X, y, X_test = lazy_load_function()

        X_test, inv_perm, og_index = self._shuffle_test_rows(X_test)
        if shuffled_features is not None:
            X_test = X_test[shuffled_features]
            X = X[shuffled_features]

        self.post_fit(X=X, y=y, X_test=X_test)

        self.pre_predict()
        if self.problem_type in ["binary", "multiclass"]:
            with Timer() as timer_predict:
                y_pred_proba = self.predict_proba(X_test)
            y_pred = self.predict_from_proba(y_pred_proba)
        else:
            with Timer() as timer_predict:
                y_pred = self.predict(X_test)
            y_pred_proba = None
        self.post_predict()

        return {
            "predictions": self._restore_prediction_order(y_pred, inv_perm, og_index),
            "probabilities": self._restore_prediction_order(y_pred_proba, inv_perm, og_index),
            "time_train_s": timer_fit.duration,
            "time_infer_s": timer_predict.duration,
            "memory_usage": self._collect_memory_usage(cpu_tracker, gpu_tracker),
        }

    def _shuffle_features(self, X: pd.DataFrame, *, split_seed: int | None) -> tuple[pd.DataFrame, list | None]:
        """Deterministically permute the feature columns of ``X`` when ``shuffle_features``.

        Returns ``(X, shuffled_features)`` where ``shuffled_features`` is the permuted
        column order (or None when shuffling is disabled), so the same order can later be
        applied to ``X_test``.
        """
        if not self.shuffle_features:
            return X, None
        assert split_seed is not None, "If shuffle_features is True, split_seed must not be None!"
        shuffled_features = list(X.columns)
        rng = np.random.default_rng(seed=split_seed)
        rng.shuffle(shuffled_features)
        return X[shuffled_features], shuffled_features

    def _shuffle_test_rows(self, X_test: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray | None, pd.Index]:
        """Apply the deterministic test-row shuffle and index reset used for inference.

        Shuffling guards against models that (incorrectly) depend on row order; the
        permutation is inverted afterwards (see ``_restore_prediction_order``) so outputs
        line up with the caller's original ``X_test``.

        Returns:
        -------
        (X_test, inv_perm, original_index)
            ``inv_perm`` is None when ``shuffle_test`` is disabled.
        """
        original_index = X_test.index
        inv_perm = None
        if self.shuffle_test:
            perm, inv_perm = _make_perm(len(X_test), seed=self.shuffle_seed)
            X_test = X_test.iloc[perm]
        if self.reset_index_test:
            X_test = X_test.reset_index(drop=True)
        return X_test, inv_perm, original_index

    def _restore_prediction_order(self, predictions, inv_perm: np.ndarray | None, original_index: pd.Index):
        """Map model outputs back onto the caller's original ``X_test`` order/index.

        Inverts the test-row shuffle when ``shuffle_test`` is enabled; otherwise restores
        the original index (for pandas outputs) when ``reset_index_test`` reset it.
        ``None`` is passed through unchanged (e.g. absent probabilities for regression).
        """
        if predictions is None:
            return None
        if self.shuffle_test:
            return _apply_inv_perm(predictions, inv_perm, index=original_index)
        if self.reset_index_test and hasattr(predictions, "index"):
            predictions.index = original_index
        return predictions

    @staticmethod
    def _collect_memory_usage(cpu_tracker, gpu_tracker) -> dict:
        """Snapshot the CPU/GPU memory trackers into the result dict's ``memory_usage`` block."""
        return dict(
            peak_mem_cpu=cpu_tracker.peak_rss,
            min_mem_cpu=cpu_tracker.min_rss,
            peak_mem_gpu=gpu_tracker.peak_allocated,
            peak_mem_gpu_reserved=gpu_tracker.peak_reserved,
            min_mem_gpu=gpu_tracker.min_allocated,
            min_mem_gpu_reserved=gpu_tracker.min_reserved,
            gpu_tracking_enabled=gpu_tracker.enabled,
        )

    # --- Fit / predict (public + protected hooks) -------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None):
        """Preprocess the data and delegate to ``_fit``.

        Fits the label cleaner + feature generator on ``X``/``y`` and transforms any
        provided validation data the same way before calling the subclass ``_fit``.
        """
        X, y = self._preprocess_fit_transform(X=X, y=y)
        if X_val is not None:
            X_val = self.transform_X(X_val)
            y_val = self.transform_y(y_val)

        return self._fit(X=X, y=y, X_val=X_val, y_val=y_val)

    def _fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None):
        """Fit the underlying model on already-preprocessed data. Must be implemented."""
        raise NotImplementedError

    def predict_from_proba(self, y_pred_proba: pd.DataFrame) -> pd.Series:
        """Derive class predictions (argmax) from predicted probabilities."""
        if isinstance(y_pred_proba, pd.DataFrame):
            return y_pred_proba.idxmax(axis=1)
        return np.argmax(y_pred_proba, axis=1)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict labels for ``X`` (preprocess features, call ``_predict``, decode labels)."""
        X = self.transform_X(X=X)
        y_pred = self._predict(X)
        return self.inverse_transform_y(y=y_pred)

    def _predict(self, X: pd.DataFrame):
        """Predict labels on already-preprocessed features. Must be implemented."""
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict class probabilities for ``X`` (preprocess, call ``_predict_proba``, decode)."""
        X = self.transform_X(X=X)
        y_pred_proba = self._predict_proba(X=X)
        return self.inverse_transform_y_pred_proba(y_pred_proba=y_pred_proba)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict class probabilities on already-preprocessed features. Must be implemented."""
        raise NotImplementedError

    # --- Resource management ----------------------------------------------------------
    def cleanup(self):
        """Release any resources held by the method (files, GPU memory, ...). Default: no-op."""

    # --- Optional capabilities (gated by the ``can_get_*`` flags) ---------------------
    def get_metric_error_val(self) -> float:
        """Return the validation metric error. Implement when ``can_get_error_val`` is True."""
        raise NotImplementedError

    def get_oof(self) -> dict:
        """Return out-of-fold simulation artifacts. Implement when ``can_get_oof`` is True."""
        raise NotImplementedError

    def bag_artifact(self, X_test: pd.DataFrame) -> dict:
        """Return per-bagged-child OOF/test artifacts.

        Implement when ``can_get_per_child_oof`` / ``can_get_per_child_val_idx`` are True.
        """
        raise NotImplementedError
