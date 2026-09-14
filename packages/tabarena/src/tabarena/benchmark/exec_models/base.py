from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
import pandas as pd
from autogluon.core.data.label_cleaner import LabelCleaner, LabelCleanerDummy
from autogluon.features import AutoMLPipelineFeatureGenerator

from tabarena.benchmark.exec_models.utils import _apply_inv_perm, _make_perm
from tabarena.benchmark.task.metadata import ValidationMetadata
from tabarena.utils.time_utils import Timer

if TYPE_CHECKING:
    from collections.abc import Callable

    from autogluon.core.metrics import Scorer

    from tabarena.models.warmup import WarmupReport


#: Stored under ``timing_audit["scope"]`` so a reader of one result knows what the audit can see.
TIMING_AUDIT_SCOPE = (
    "Main-process view only: parallel fold workers import in their own processes, and libraries vendored "
    "under tabarena.models appear as tabarena submodules rather than as new packages."
)


class AbstractExecModel:
    """Base class for a benchmarked *method* (an "execution model").

    A subclass wraps some underlying model behind a small, uniform interface so the
    experiment runner can fit it, collect predictions, and record metadata the same way
    for every method. Concrete subclasses only have to implement the protected hooks
    (``_fit`` and ``_predict`` / ``_predict_proba``); everything else here is shared
    orchestration.

    Lifecycle (driven by the experiment runner):

    1. ``__init__`` — configure preprocessing and inference-shuffle behavior.
    2. ``warmup_fn`` — optional untimed environment warm-up, run before any timing starts
       (see the property).
    3. ``fit_custom`` — the end-to-end harness: optionally shuffle features, fit (while
       tracking time + memory), then predict on the test data and undo any shuffling.
    4. post-evaluate consumers (metadata, OOF and bag artifacts) run after ``fit_custom`` and
       before ``cleanup``, so a model brought into serving state in ``pre_predict`` is still
       resident for them.
    5. ``cleanup`` — release any resources (files, served models, GPU memory, ...).

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

    @classmethod
    def uses_ray(cls, method_kwargs: dict, *, problem_type: str | None = None) -> bool:
        """Whether a fit configured with these constructor kwargs may start or use a Ray runtime.

        Consulted by ``Experiment.uses_ray`` so a SLURM worker can skip starting a Ray runtime (GCS,
        raylet, agents, prestarted workers: several seconds of wall-clock per item and 0.5 to 1.5 GB
        of RSS counted into the fit's memory numbers) for jobs that never touch Ray. The default is
        True (start Ray). A subclass returns False only when it can show that no code path calls
        ``ray.init`` or ``ray.remote`` during fit, predict or artifact collection: a wrong False makes
        AutoGluon start Ray inside the timed fit with its own defaults (no SLURM-safe temp dir).

        Args:
            method_kwargs: The constructor kwargs of this exec model, as serialized on the experiment
                (before the run-time resource detection).
            problem_type: The task's problem type when known; a model's default ensemble arguments
                may depend on it.
        """
        return True

    # --- Declarative warm-up (untimed; see the ``warmup_fn`` property) ------------------
    warmup_modules: ClassVar[tuple[str, ...]] = ()
    """Modules this exec model wants imported untimed before the fit (merged over the MRO).

    Declare the libraries the fit and predict import lazily; a ``"torch"`` entry also creates the
    CUDA context when the compute budget allows it. A subclass that overrides ``warmup_fn`` must
    start from ``self._declared_warmup()`` for these declarations to apply."""
    warmup_torch_device: ClassVar[bool] = False
    """If True, the default warm-up imports torch and creates the CUDA context (see ``_warmup_cuda``)."""
    warmup_dummy_fit: ClassVar[bool] = True
    """Whether the warm-up may fit and predict the wrapped model classes on a small synthetic dataset.

    Read by the AutoGluon wrappers and passed to ``tabarena.models.warmup.warmup_model_cls``; systems
    that fit a whole pipeline turn it off (see ``ExternalSystemModel``)."""

    _can_use_data_in_place: bool
    """Whether ``fit`` may mutate the training frames it receives (for example append the label
    column) instead of copying them first.

    ``fit_custom`` sets it to True for the duration of every fit it drives and restores the previous
    value afterwards: the frames reaching ``fit`` are either lazy-loaded (owned by this object),
    the new frame a column shuffle produced, the feature generator's output when
    ``preprocess_data`` is on, or a copy taken once before the memory trackers and the fit timer
    start, so no wrapper pays a defensive copy inside the timed fit. It stays False for direct
    ``fit`` callers (tests, notebooks), where ``AGWrapper._attach_label`` and
    ``ExternalSystemModel._fit`` still copy to protect the caller's frame.
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
    validation_metadata: ValidationMetadata
    """Task-derived metadata for validation information. Available to *every* method uniformly:
    the experiment runner populates it from the task at fit time.

    This is read-only *data*: holding it does not change behavior. Whether a method
    *acts* on it is the method's own decision (e.g. the AutoGluon wrappers gate task-specific
    validation splits on ``use_task_specific_validation``; an external system reads whatever fields
    it needs in ``_fit``). The shared evaluation protocol (splits, feature names, test-row shuffle,
    scoring) is fixed upstream by the task and runner regardless of this object.
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
    verbosity: int | None = None
    """Verbosity for the ``preprocess_data`` feature generator. ``None`` (default) leaves
    AutoGluon's logger untouched — so a standalone fit (no ``TabularPredictor`` to configure
    logging) stays silent, as before. An int turns logging on for the fit: it raises the
    AutoGluon logger to the matching level (via ``set_logger_verbosity``) and sets the feature
    generator's verbosity, so its preprocessing output is shown (2 = the usual key prints,
    3+ = more detail). Wrappers that delegate preprocessing to ``TabularPredictor`` (which
    handles its own verbosity) leave ``preprocess_data=False`` and are unaffected."""

    #: Config attributes (above) that ``__init__`` accepts as per-instance overrides.
    _CONFIG_ATTRS = (
        "preprocess_data",
        "preprocess_label",
        "shuffle_test",
        "shuffle_seed",
        "reset_index_test",
        "shuffle_features",
        "verbosity",
    )

    def __init__(
        self,
        problem_type: str,
        eval_metric: Scorer,
        *,
        validation_metadata: ValidationMetadata | dict | None = None,
        **kwargs,
    ):
        """Configure the method.

        Parameters
        ----------
        problem_type:
            One of ``"binary"``, ``"multiclass"``, ``"regression"``.
        eval_metric:
            AutoGluon scorer used for evaluation (and, for some wrappers, model fitting).
        validation_metadata:
            Task-derived split metadata (or a kwargs dict / ``None`` for one), normalized via
            ``ValidationMetadata.from_config`` and exposed on ``self.validation_metadata`` for
            every method (see that attribute). Normally injected by the experiment runner from
            the task; a bare ``None`` yields an empty ``ValidationMetadata()``.
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
        # Uniform, read-only task metadata available to all methods (empty when unset).
        self.validation_metadata = ValidationMetadata.from_config(validation_metadata)

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

    def _make_feature_generator(self):
        """Build the (unfitted) model-agnostic feature generator for ``preprocess_data``.

        Default is AutoGluon's standard ``AutoMLPipelineFeatureGenerator``. Subclasses override
        this to use a different model-agnostic pipeline (e.g. ``AGModelWrapper`` resolves the
        TabArena ``tabarena_default`` generator), keeping the rest of the preprocessing flow shared.
        """
        return AutoMLPipelineFeatureGenerator()

    def _preprocess_fit_transform(self, X: pd.DataFrame, y: pd.Series):
        """Fit the label cleaner and (optionally) feature generator, then transform ``X``/``y``.

        Called once at the start of ``fit``. Sets ``self.label_cleaner`` and, when
        ``preprocess_data`` is enabled, ``self._feature_generator`` (built by
        ``_make_feature_generator``).
        """
        if self.preprocess_label:
            self.label_cleaner = LabelCleaner.construct(problem_type=self.problem_type, y=y)
        else:
            self.label_cleaner = LabelCleanerDummy(problem_type=self.problem_type)
        if self.preprocess_data:
            self._feature_generator = self._make_feature_generator()
            if self.verbosity is not None:
                # Surface the feature generator's logs: raise AutoGluon's logger to the matching
                # level (TabularPredictor does this in the validation path; a standalone fit does
                # not) and set the generator's own verbosity.
                from autogluon.common.utils.log_utils import set_logger_verbosity

                set_logger_verbosity(self.verbosity)
                self._feature_generator.set_verbosity(self.verbosity)
            X = self._feature_generator.fit_transform(X=X, y=y)
        y = self.transform_y(y)
        return X, y

    # --- Warm-up (untimed) -------------------------------------------------------------
    @property
    def warmup_fn(self) -> Callable[[], WarmupReport | None] | None:
        """Optional zero-arg callable warming the execution environment before the timed fit.

        The experiment runner calls it (when not ``None``) after constructing the method and
        *before* ``fit_custom``, so nothing it does counts toward the measured ``time_train_s``
        / ``time_infer_s`` or any fit time limit. This mirrors reality: one-time per-environment
        costs (library imports, JIT/kernel compilation, CUDA context initialization, Ray startup,
        shared checkpoint weights) are not paid per fit by a long-lived deployment, so they should
        not inflate a method's measured speed. The runner records a failed or partial warm-up in
        ``warmup_report`` and, with ``require_warmup`` (the default), aborts the run before the timed
        fit instead of measuring a cold process.

        Returning a ``WarmupReport`` lets the runner record what was warmed; returning ``None`` is
        allowed (the runner still records the imported modules and the CUDA state around the call
        through ``tabarena.models.warmup.run_warmup_fn``).

        This is an *environment* warm-up, not a fit-only one: it runs once per job, and since
        the same process serves the timed fit and the timed inference, everything it warms
        (imports, CUDA context, compiled kernels) also benefits ``time_infer_s``. Two things it
        deliberately does **not** cover: (1) data-dependent first-call inference work (e.g. lazy
        compilation triggered by the first ``predict`` on real data) stays in the measured
        inference time, with ``pre_predict`` / ``post_predict`` as the untimed hooks around
        inference (used e.g. for model persistence); (2) parallel (Ray) fold workers spawned inside
        the fit start cold unless the opt-in worker pool covered them; disk-backed caches carry
        over (see ``tabarena.models.warmup``).

        Implementations may use everything known at construction time (``problem_type``,
        ``eval_metric``, hyperparameters, compute budget) but never the task's data, and must
        not carry task- or data-specific state into the fit.

        Default: ``self._declared_warmup`` when the class declares ``warmup_modules`` or
        ``warmup_torch_device``, else ``None`` (nothing to warm). A subclass that overrides this
        property must call ``self._declared_warmup()`` itself so the declarations still apply.
        """
        from tabarena.models.warmup import collect_warmup_modules

        if not (collect_warmup_modules(type(self)) or self.warmup_torch_device):
            return None
        return self._declared_warmup

    def _warmup_cuda(self) -> bool | None:
        """Whether the default torch warm-up should create the CUDA context; ``None`` auto-detects.

        Reads ``self.num_gpus`` when the exec model stores its compute budget there (systems do);
        wrappers that keep it elsewhere override this.
        """
        num_gpus = getattr(self, "num_gpus", None)
        return None if num_gpus is None else num_gpus > 0

    @property
    def num_cpus_budget(self) -> int | None:
        """The CPU budget (``num_cpus``) this method fits under, or ``None`` when it declares none.

        The experiment runner compares it with the CPUs the process may actually use before the
        fit (see :func:`tabarena.utils.thread_utils.check_cpu_budget`); ``None`` is resolved to the
        auto-detected count there. Wrappers that carry a ``num_cpus`` return it.
        """
        return None

    def _declared_warmup(self, report: WarmupReport | None = None) -> WarmupReport:
        """Apply the declarative warm-up (``warmup_torch_device`` then ``warmup_modules``) into a report.

        The building block every ``warmup_fn`` starts from: the default property returns it directly
        and the AutoGluon wrappers call it before their own layers.
        """
        from tabarena.models.warmup import WarmupReport, apply_warmup_entries, collect_warmup_modules, warmup_torch_step

        if report is None:
            report = WarmupReport()
        cuda = self._warmup_cuda()
        torch_done = False
        if self.warmup_torch_device:
            torch_done = warmup_torch_step(cuda=cuda, report=report)
        apply_warmup_entries(collect_warmup_modules(type(self)), cuda=cuda, report=report, torch_done=torch_done)
        return report

    # --- Fit / predict lifecycle hooks (overridable) ----------------------------------
    def post_fit(self, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame):
        """Hook run after fitting, before inference. Default: no-op.

        ``X``/``y`` are the (reloaded) training data and ``X_test`` the test features, both
        in the order/layout used for inference.
        """

    def pre_predict(self):
        """Hook run once immediately before inference, outside the inference timer. Default: no-op.

        The untimed inference-side counterpart of ``warmup_fn``: use it to bring the fitted model
        into serving state, the way a deployment would before serving (persist it in memory, see
        ``AGWrapper.persist``; place its weights on the inference device). It may touch the fitted
        model but never the test data, and must not precompute anything prediction-specific.

        A wrapped model object may declare ``prepare_for_inference(self) -> None``. The exec models
        call it untimed on every persisted object (a bag and each of its loaded children) and record
        which calls succeeded in the method metadata (``prepared_for_inference``). This is the one
        place the hook's contract is written; other docstrings refer here.

        The hook must be idempotent and model-only. It may:

        * reload or reattach its own pretrained weights, from the process-wide weights registry
          (``tabarena.models._weights``) or the local checkpoint cache;
        * move tensors it already owns to the inference device;
        * build configuration-driven pipelines that read no data (for example a library predictor
          object constructed from the estimator's constructor arguments and its checkpoint);
        * synchronize the device and switch modules to eval mode.

        It must not:

        * read training, validation or test data;
        * compute or cache anything derived from the stored training context;
        * run a forward pass, also not on dummy inputs;
        * change parameter dtypes.

        Data-dependent first-call work (kernel dispatch on the real test batch, in-context staging)
        stays inside the predict timer. A failing hook is isolated per object and logged; the timed
        predict then runs on that object unprepared.
        """

    def post_predict(self):
        """Hook run immediately after inference, outside the timer. Default: no-op.

        Releasing served models does not belong here when post-evaluate consumers (method metadata,
        OOF and bag artifacts) reuse them; that release happens in ``cleanup``.
        """

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
        """Fit the method and predict on ``X_test``, recording timing, memory usage and an environment audit.

        The single entry point used by the experiment runner. It fits the model (via ``fit``, then
        ``_fit``) while tracking wall-clock time and CPU/GPU memory, then produces predictions
        (probabilities for classification, point predictions for regression) on ``X_test``, undoing
        any test-row shuffle so outputs align with the caller's original ``X_test``.

        Frame ownership: ``fit`` receives frames this object owns, so no wrapper pays a defensive copy
        inside the timed fit (``_can_use_data_in_place`` is True for the duration of the fit and the
        previous value is restored afterwards). A lazy-loaded frame is owned already, a column shuffle
        produces a new frame, and the feature generator's output is new when ``preprocess_data`` is
        on; in every other case ``X`` and ``y`` are copied once here, before the memory trackers and
        the fit timer start. ``post_fit`` still receives the caller's (unmodified) frames.

        Timer boundaries: the memory trackers and the fit timer bracket ``fit`` only; ``pre_predict``
        is untimed; the predict timer brackets ``predict_proba`` (classification) or ``predict``
        (regression) only; ``predict_from_proba`` and ``post_predict`` are untimed. An
        ``EnvironmentSnapshot`` (``tabarena.utils.timing_audit``) is taken immediately before and after
        each timer, so the audit shows exactly what the timed section imported or initialized.

        Args:
            X: Training features. Must be ``None`` iff ``lazy_load_function`` is provided.
            y: Training labels, aligned with ``X``; same ``None`` rule.
            X_test: Test features; same ``None`` rule.
            split_seed: If not None, the per-split seed used to shuffle features (required when
                ``shuffle_features`` is True).
            lazy_load_function: If provided, a callable returning ``(X, y, X_test)`` used to load the
                data only when needed (to save memory). The data is loaded once for fitting and
                reloaded afterwards so the training frames can be used in place.

        Returns:
            A dict with ``predictions``, ``probabilities`` (None for regression), ``time_train_s``,
            ``time_infer_s``, ``memory_usage`` (see ``_collect_memory_usage`` for its keys) and
            ``timing_audit``: ``{"fit": {...}, "predict": {...}, "scope": str}``, where each timed
            section's dict is ``EnvironmentSnapshot.diff`` output (``new_modules``, ``new_packages``,
            ``new_submodule_packages``, ``cuda_initialized_before`` / ``_after``,
            ``ray_initialized_before`` / ``_after``) or ``None`` when the audit failed, and ``scope`` is
            ``TIMING_AUDIT_SCOPE``.
        """
        from tabarena.utils.memory_utils import CpuMemoryTracker, GpuMemoryTracker
        from tabarena.utils.timing_audit import audit_since, take_snapshot

        self._split_seed = split_seed

        owned = lazy_load_function is not None
        if owned:
            assert X is None and y is None and X_test is None, "If lazy_load_function is provided, X and y must be None"  # noqa: PT018
            X, y, _ = lazy_load_function()

        X_fit, shuffled_features = self._shuffle_features(X, split_seed=split_seed)
        y_fit = y
        if not owned and shuffled_features is None and not self.preprocess_data:
            # The caller keeps its frames untouched: copy once here, outside the trackers and the fit
            # timer. A lazy-loaded frame is already ours, a column shuffle produced a new frame, and with
            # ``preprocess_data`` the frame reaching ``_fit`` is the feature generator's own output.
            X_fit = X.copy()
            y_fit = y.copy()

        can_use_data_in_place_before = self._can_use_data_in_place
        self._can_use_data_in_place = True
        try:
            # Both trackers are constructed before either is entered, so the GPU tracker's torch import
            # is part of the CPU baseline rather than of the fit's memory curve.
            cpu_tracker = CpuMemoryTracker()
            gpu_tracker = GpuMemoryTracker(device=0)
            with cpu_tracker, gpu_tracker:
                # Snapshot after the trackers are entered (the GPU tracker creates the CUDA context) and
                # right before the timer starts, so neither is attributed to the fit.
                before_fit = take_snapshot()
                with Timer() as timer_fit:
                    self.fit(X_fit, y_fit)
                audit_fit = audit_since(before_fit)
        finally:
            self._can_use_data_in_place = can_use_data_in_place_before
        del X_fit, y_fit

        # Reload all, allows X,y to be used in-place
        if owned:
            del X, y, X_test  # Free memory from previous load
            X, y, X_test = lazy_load_function()

        X_test, inv_perm, og_index = self._shuffle_test_rows(X_test)
        if shuffled_features is not None:
            X_test = X_test[shuffled_features]
            X = X[shuffled_features]

        self.post_fit(X=X, y=y, X_test=X_test)

        self.pre_predict()
        before_predict = take_snapshot()
        if self.problem_type in ["binary", "multiclass"]:
            with Timer() as timer_predict:
                y_pred_proba = self.predict_proba(X_test)
            audit_predict = audit_since(before_predict)
            y_pred = self.predict_from_proba(y_pred_proba)
        else:
            with Timer() as timer_predict:
                y_pred = self.predict(X_test)
            audit_predict = audit_since(before_predict)
            y_pred_proba = None
        self.post_predict()

        return {
            "predictions": self._restore_prediction_order(y_pred, inv_perm, og_index),
            "probabilities": self._restore_prediction_order(y_pred_proba, inv_perm, og_index),
            "time_train_s": timer_fit.duration,
            "time_infer_s": timer_predict.duration,
            "memory_usage": self._collect_memory_usage(cpu_tracker, gpu_tracker),
            "timing_audit": {"fit": audit_fit, "predict": audit_predict, "scope": TIMING_AUDIT_SCOPE},
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
        """Snapshot the CPU/GPU memory trackers into the result dict's ``memory_usage`` block.

        Keys, all sampled over the timed fit only (bytes):

        ``peak_mem_cpu`` / ``min_mem_cpu``
            Highest and lowest RSS of this process plus its descendants (Ray daemons and fold
            workers included).
        ``peak_mem_gpu`` / ``min_mem_gpu`` and the ``_reserved`` pair
            torch's allocated and reserved CUDA memory, absolute values; ``gpu_tracking_enabled`` says
            whether a CUDA device was tracked at all.
        ``baseline_mem_cpu`` / ``baseline_mem_cpu_self``
            RSS of the subtree and of the main process alone right when the tracker started, so a
            daemon baseline (for example Ray) can be subtracted in analysis: the difference of the two
            is the descendants' share. ``cpu_tracking_backend`` names the sampler (``"procfs"`` or
            ``"psutil"``); both report identical values per sample.

        Baseline shifts to keep in mind when comparing numbers across TabArena versions (the existing
        keys keep their meaning, their values move):

        * the GPU tracker reports absolute allocation, so weights pre-loaded by the warm-up into the
          shared registry (``tabarena.models._weights``) are part of ``min_mem_gpu`` and set the floor
          of ``peak_mem_gpu``; this is intended (a served model keeps its weights resident) and recorded
          in ``experiment_metadata["warmup_report"]`` and ``method_metadata["shared_weights"]``;
        * ``min_mem_cpu`` and the baselines include the warm-up's imports, the torch import the GPU
          tracker performs on nodes where torch is installed (0.3 to 0.5 GB, also for CPU models) and,
          on a local run of a bagging preset, Ray's daemons; sequential_local SLURM jobs no longer start
          Ray, so they lose the 0.5 to 1.5 GB Ray daemon baseline;
        * ``peak_mem_cpu`` may rise by the size of ``X`` for fits shorter than one sampling interval,
          because ``fit_custom`` copies the caller's frame before the tracker starts;
        * ``info["memory_size"]`` and ``disk_usage`` in the method metadata move with the weightless
          pickles of the foundation-model wrappers (pickle plus tensor bytes for ``memory_size``);
        * the shared registry is released between in-process sweep items only when the model class
          changes (see ``experiment_runner_api``), so consecutive items of one model start with the
          weights resident.
        """
        return dict(
            peak_mem_cpu=cpu_tracker.peak_rss,
            min_mem_cpu=cpu_tracker.min_rss,
            peak_mem_gpu=gpu_tracker.peak_allocated,
            peak_mem_gpu_reserved=gpu_tracker.peak_reserved,
            min_mem_gpu=gpu_tracker.min_allocated,
            min_mem_gpu_reserved=gpu_tracker.min_reserved,
            gpu_tracking_enabled=gpu_tracker.enabled,
            baseline_mem_cpu=getattr(cpu_tracker, "start_rss", None),
            baseline_mem_cpu_self=getattr(cpu_tracker, "start_rss_self", None),
            cpu_tracking_backend=getattr(cpu_tracker, "backend", None),
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

    def bag_artifact(self, X_test: pd.DataFrame, *, y_pred=None, y_pred_proba=None) -> dict:
        """Return per-bagged-child OOF/test artifacts.

        Implement when ``can_get_per_child_oof`` / ``can_get_per_child_val_idx`` are True. Runs after
        ``fit_custom`` and before ``cleanup``, so a model brought into serving state in ``pre_predict``
        is still resident.

        Args:
            X_test: The test features in the caller's original row order.
            y_pred: The timed test predictions (the runner's ``predictions``) in the caller's original
                row order, when the runner has them.
            y_pred_proba: The timed test probabilities (the runner's ``probabilities``), same order;
                ``None`` for regression.

        Implementations may derive the per-child test artifact from the timed outputs when the bag's
        output equals its child's (a single-child bag with no post-hoc transform) and must compute it
        otherwise.
        """
        raise NotImplementedError
