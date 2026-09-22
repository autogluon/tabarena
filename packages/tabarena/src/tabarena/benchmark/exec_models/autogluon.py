from __future__ import annotations

import copy
import os
import shutil
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
import pandas as pd
from autogluon.core.data.label_cleaner import LabelCleanerMulticlassToBinary
from autogluon.core.models import AbstractModel
from autogluon.core.utils import get_pred_from_proba
from loguru import logger

from tabarena.benchmark.exec_models.autogluon_utils import (
    SPLIT_RANDOM_STATE,
    get_num_group_instances,
    resolve_holdout_split,
    resolve_model_cls,
    resolve_validation_splits,
    validation_structure_from_metadata,
)
from tabarena.benchmark.exec_models.base import AbstractExecModel
from tabarena.benchmark.exec_models.persist_inference import (
    InferencePersistence,
    dispatch_prepare_for_inference,
    free_inference_memory,
    persist_for_inference,
    release_after_inference,
)
from tabarena.benchmark.exec_models.utils import _apply_inv_perm, _make_perm
from tabarena.benchmark.preprocessing.pipeline import build_feature_generator, resolve_preprocessing_pipeline
from tabarena.benchmark.validation_protocol import (
    ValidationProtocol,
    ValidationProtocolError,
    ValidationResolution,
    structure_of,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from autogluon.tabular import TabularPredictor

    from tabarena.models.warmup import WarmupReport

#: Environment variable that, when set together with ``init_kwargs["default_base_path"]``, replaces the
#: configured AutoGluon artifact root at fit time (a SLURM job points it at node-local scratch).
MODEL_ARTIFACTS_BASE_PATH_ENV = "TABARENA_MODEL_ARTIFACTS_BASE_PATH"


class AGWrapper(AbstractExecModel):
    """An AutoGluon ``TabularPredictor`` wrapped as an exec model.

    Fits a full ``TabularPredictor`` (whatever ``init_kwargs`` / ``fit_kwargs`` describe)
    and exposes it through the common exec-model interface. Feature/label preprocessing
    is disabled by default here, since AutoGluon does its own; the label column is appended
    to the frame internally under the name resolved in ``_build_predictor_args``
    (the validation metadata's ``target_name``, else ``"__label__"``).

    The inner validation follows ``self.validation_protocol`` (a
    :class:`~tabarena.benchmark.validation_protocol.ValidationProtocol`, injected by the experiment
    runner from the experiment) and is resolved during ``_fit`` against ``self.validation_metadata``.
    A bagged fit (``AGSingleBagWrapper``) takes its fold and repeat counts from the protocol; with
    ``task_specific_validation`` it uses ``resolve_validation_splits`` to produce ``k`` group/time-aware
    folds (re-injected as ``ag_args_ensemble['custom_splits']``). A holdout fit (``AGSingleWrapper``)
    uses ``resolve_holdout_split`` to produce a single group/time-aware train/validation split, fed to
    ``TabularPredictor`` as ``tuning_data`` (a single model does not consume ``custom_splits``). A full
    predictor (``AGWrapper`` itself) fits the counts ``fit_kwargs`` name, or lets AutoGluon decide
    without them; with ``task_specific_validation`` it declares the task's grouped / temporal
    structure to ``TabularPredictor.fit(validation_structure=...)`` and AutoGluon builds the folds,
    or the holdout split, itself (one fit trains many models, so TabArena cannot pre-resolve splits
    per model). The learner is then seeded with ``split_random_state`` so those splits follow the
    same seed as TabArena's resolved ones; a task without such structure keeps AutoGluon's default
    splitter and seed.

    Parameters
    ----------
    init_kwargs:
        Extra keyword arguments for the ``TabularPredictor(...)`` constructor.
    fit_kwargs:
        Extra keyword arguments for ``TabularPredictor.fit(...)``. The bagging counts are not
        passed here: a single-model wrapper takes them from ``validation_protocol``; a full
        predictor may name ``num_bag_folds`` / ``num_bag_sets`` explicitly (fit as given).
    temporal_forward_only:
        Full predictor only. Ask AutoGluon for forward-chaining temporal validation on a ``time_on``
        task instead of leave-one-block-out: fold *i* validates time block *i+1* and trains only on
        earlier blocks. Costs the earliest block (never validated) and trains each fold on less data;
        no effect on a task without ``time_on``; cannot be combined with stacking (AutoGluon raises).
    split_random_state:
        Full predictor only. Seed for AutoGluon's structure-aware splitting, injected as the
        learner's ``random_state`` when a structure is declared. Defaults to ``data_foundry``'s
        seed (:data:`~tabarena.benchmark.exec_models.autogluon_utils.SPLIT_RANDOM_STATE`) so the
        folds match TabArena's resolved ones; an explicit ``init_kwargs["learner_kwargs"]["random_state"]``
        wins.
    persist:
        If True (default), persist the fitted model in memory before inference (untimed, in
        ``pre_predict`` through ``persist_for_inference``), so the measured inference time is that of
        a served, in-memory model rather than including its load from disk. Memory-guarded by
        AutoGluon's ``max_memory`` (``persist_max_memory``): when the model (incl. a bag's children)
        doesn't fit, nothing is persisted and inference falls back to on-demand disk loads. The
        served models stay resident through metadata and bag-artifact collection and are released
        in ``cleanup`` (so ``ExperimentRunner(cleanup=False)`` keeps them resident until the exec model
        is garbage collected). The fit metadata records which models were in memory
        (``persisted_models``) and which persisted objects ran their ``prepare_for_inference`` hook
        (``prepared_for_inference``).
    validation_metadata:
        Task-derived ``ValidationMetadata`` (or a kwargs dict for one) describing the
        validation-split structure. Inherited from ``AbstractExecModel`` (the runner injects it
        from the task); the label column appended to the training frame is named via
        ``validation_metadata.get_target_name()`` (its ``target_name``, else ``"__label__"``).
    validation_protocol:
        The :class:`~tabarena.benchmark.validation_protocol.ValidationProtocol` (or its dict form)
        this fit follows: fold and repeat counts, the tiny-data regime, whether the inner splits are
        task-specific and whether AutoGluon may adapt the fold count to the class count. ``None``
        for a full predictor that leaves the validation to AutoGluon.
    """

    persist_max_memory: float | None = 0.4
    """``max_memory`` for the ``pre_predict`` persist, or ``None`` to skip the check.

    AutoGluon decides whether the models fit by pickling each one to measure its size, which for
    a large model can cost more than the loading the check guards against. Subclasses whose
    memory use is bounded by construction can set this to ``None``.
    """

    release_shared_weights_on_cleanup: bool = False
    """Whether ``cleanup`` also drops the process-wide shared-weights registry (``tabarena.models._weights``).

    The registry is process-scoped. A SLURM job runs one item per process, so nothing needs releasing
    there, and an in-process sweep releases it between items only when the model class changes (see
    ``experiment_runner_api``), so consecutive items of one model keep hitting the primed entries. Set
    True to free the weights (GPU memory included) after every item.
    """

    # Default AutoGluon can return a validation score
    can_get_error_val = True
    # Default AutoGluon can return OOF predictions for the best model.
    can_get_oof = True

    # AutoGluon does its own feature/label preprocessing, so disable ours by default.
    preprocess_data = False
    preprocess_label = False

    predictor: TabularPredictor
    """The fitted AutoGluon ``TabularPredictor`` (set by ``_fit``)."""

    bagged_fit: ClassVar[bool | None] = None
    """Whether this wrapper fits a bag (``True``), a single holdout model (``False``) or a full
    predictor whose ``fit_kwargs`` and presets decide (``None``)."""

    def __init__(
        self,
        init_kwargs: dict | None = None,
        fit_kwargs: dict | None = None,
        persist: bool = True,
        validation_protocol: ValidationProtocol | dict | None = None,
        temporal_forward_only: bool = False,
        split_random_state: int = SPLIT_RANDOM_STATE,
        **kwargs,
    ):
        if self.bagged_fit is not None and (temporal_forward_only or split_random_state != SPLIT_RANDOM_STATE):
            raise ValueError(
                "`temporal_forward_only` and `split_random_state` act on the full predictor, which declares the "
                "task's structure to AutoGluon; a single-model wrapper resolves its splits in TabArena."
            )
        if "use_task_specific_validation" in kwargs:
            raise TypeError(
                "`use_task_specific_validation` is not a wrapper argument; task-specific validation is the "
                "`task_specific_validation` field of the experiment's ValidationProtocol.",
            )
        super().__init__(**kwargs)
        if init_kwargs is None:
            init_kwargs = {}
        if fit_kwargs is None:
            fit_kwargs = {}
        self.init_kwargs = init_kwargs
        self.fit_kwargs = fit_kwargs
        self.validation_protocol = ValidationProtocol.from_config(validation_protocol)
        self.temporal_forward_only = temporal_forward_only
        self.split_random_state = split_random_state
        self.persist = persist
        #: A single fit on all rows with no validation split; only ``AGSingleWrapper`` turns it on.
        self.no_validation: bool = False
        self._persisted_models: list[str] | None = None
        self._validation_resolution: ValidationResolution | None = None
        self._prepared_models: list[str] | None = None

    # --- Warm-up (untimed) --------------------------------------------------------------
    @property
    def warmup_fn(self) -> Callable[[], WarmupReport] | None:
        """Warm the AutoGluon stack, every configured model class and the feature generator (untimed).

        Configured classes come from ``fit_kwargs`` (a ``hyperparameters`` dict, a config name or
        presets); the resolution is for warm-up only and nothing is passed into the fit. Returns a
        ``WarmupReport``.
        """
        return self._warmup

    def _warmup_cuda(self) -> bool | None:
        num_gpus = self.fit_kwargs.get("num_gpus")
        return None if num_gpus is None else num_gpus > 0

    @property
    def num_cpus_budget(self) -> int | None:
        """The ``num_cpus`` in ``fit_kwargs`` (``None`` when the fit is left to auto-detect it)."""
        return self.fit_kwargs.get("num_cpus")

    def _warmup(self) -> WarmupReport:
        from tabarena.benchmark.exec_models.autogluon_utils import configured_model_classes
        from tabarena.models.warmup import warmup_ag_stack, warmup_feature_generator_cls, warmup_model_classes

        report = self._declared_warmup()
        warmup_ag_stack(report=report)
        configured = configured_model_classes(self.fit_kwargs)
        warmup_model_classes(
            configured,
            problem_type=self.problem_type,
            num_cpus=self.fit_kwargs.get("num_cpus"),
            num_gpus=self.fit_kwargs.get("num_gpus"),
            report=report,
            dummy_fit=self.warmup_dummy_fit,
        )
        # ``feature_generator_cls`` is still on ``self.fit_kwargs`` here: ``_build_predictor_args``
        # pops it from a deep copy at fit time.
        warmup_feature_generator_cls(
            self.fit_kwargs.get("feature_generator_cls"),
            self.fit_kwargs.get("feature_generator_kwargs"),
            report=report,
        )
        self._warmup_ray(configured, report)
        return report

    def _warmup_ray(self, configured: list[tuple[type[AbstractModel], dict | None]], report: WarmupReport) -> None:
        """Start Ray (and optionally an import-only worker pool) for a CPU bag with parallel fold fitting.

        AutoGluon's ``ParallelLocalFoldFittingStrategy`` starts Ray inside the timed fit when it is
        not running; starting it here with the same ``num_cpus`` / ``num_gpus`` moves that runtime
        startup out of the timer without changing fold scheduling. Only CPU bags qualify
        (``num_gpus == 0`` and ``num_bag_folds > 1`` after presets, and at least one configured class
        resolving to ``parallel_local``); GPU bags and fits with an unknown GPU count are left to
        AutoGluon. The worker pool is opt-in via ``TABARENA_RAY_WORKER_WARMUP`` and everything obeys
        the ``TABARENA_DISABLE_RAY_WARMUP`` kill switch. Outcomes land in ``report.ray``.
        """
        from tabarena.benchmark.exec_models.autogluon_utils import resolve_effective_fit_kwargs
        from tabarena.models.warmup import ray_worker_warmup_modules
        from tabarena.utils.ray_utils import (
            ensure_ray_initialized,
            plan_ray_worker_pool,
            ray_worker_warmup_enabled,
            warmup_ray_workers,
        )

        effective = resolve_effective_fit_kwargs(self.fit_kwargs)
        num_bag_folds = effective.get("num_bag_folds")
        num_cpus = effective.get("num_cpus")
        num_gpus = effective.get("num_gpus")
        skipped = self._ray_warmup_skip_reason(configured, report, num_bag_folds=num_bag_folds, num_gpus=num_gpus)
        if skipped is not None:
            report.ray["skipped"] = skipped
            return
        try:
            report.ray.update(ensure_ray_initialized(num_cpus=num_cpus, num_gpus=num_gpus))
            report.step("ray:init")
        except Exception as exc:
            logger.warning(f"Warm-up could not start Ray ({exc!r}); AutoGluon starts it inside the fit.")
            report.step("ray:init", failed=True, error=exc)
            return
        if not ray_worker_warmup_enabled():
            return
        try:
            num_workers, cpus_per_worker = plan_ray_worker_pool(
                num_cpus=int(num_cpus or 0), num_jobs=int(num_bag_folds)
            )
            modules: list[str] = []
            for cls, _hps in configured:
                modules.extend(ray_worker_warmup_modules(cls))
            report.ray["pool"] = warmup_ray_workers(modules, num_workers, cpus_per_worker=cpus_per_worker)
            report.step("ray:pool")
        except Exception as exc:
            logger.warning(f"Warm-up of the Ray worker pool failed ({exc!r}); folds start cold.")
            report.step("ray:pool", failed=True, error=exc)

    def _ray_warmup_skip_reason(
        self,
        configured: list[tuple[type[AbstractModel], dict | None]],
        report: WarmupReport,
        *,
        num_bag_folds: int | float | None,
        num_gpus: int | float | None,
    ) -> str | None:
        """Why ``_warmup_ray`` must not start Ray for this fit, or ``None`` when it may.

        Records the resolved fold fitting strategies on ``report.ray`` when it gets that far.
        """
        from tabarena.models.warmup import resolve_fold_fitting_strategy
        from tabarena.utils.ray_utils import DISABLE_RAY_WARMUP_ENV, ray_warmup_disabled

        if ray_warmup_disabled():
            return DISABLE_RAY_WARMUP_ENV
        if not (isinstance(num_bag_folds, int | float) and num_bag_folds > 1):
            return "not a bagged fit"
        if num_gpus is None:
            return "num_gpus unknown"
        if num_gpus > 0:
            return "GPU bag; AutoGluon starts Ray itself"
        strategies = {
            resolve_fold_fitting_strategy(cls, hps, problem_type=self.problem_type, num_gpus=num_gpus)
            for cls, hps in configured
        }
        report.ray["fold_fitting_strategies"] = sorted(strategies)
        if "parallel_local" not in strategies:
            return "no configured class uses parallel fold fitting"
        return None

    def _configured_model_classes(self) -> Iterator[tuple[type[AbstractModel], dict | None]]:
        """Yield ``(model_cls, hyperparameters)`` for each model configured in ``fit_kwargs``.

        Delegates to ``autogluon_utils.configured_model_classes``: a ``hyperparameters`` dict, a
        config name and presets all resolve (best effort; unresolvable keys are skipped).
        """
        from tabarena.benchmark.exec_models.autogluon_utils import configured_model_classes

        yield from configured_model_classes(self.fit_kwargs)

    def _build_predictor_args(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
        data_owned: bool = False,
    ) -> tuple[pd.DataFrame, dict, dict]:
        """Build the ``(train_data, init_kwargs, fit_kwargs)`` for ``TabularPredictor.fit``.

        Works on deep copies of the configured ``init_kwargs`` / ``fit_kwargs`` so the
        wrapper can be re-fit. The steps:

        1. Resolve the bagging counts from the validation protocol (``_apply_validation_splits``).
           For a bagged single-model wrapper a task-specific protocol may clamp the fold / repeat
           counts and produce explicit ``custom_splits`` (re-injected into ``ag_args_ensemble``);
           for a full predictor it declares the task's structure as ``validation_structure``
           instead, and the learner is seeded with ``split_random_state`` so AutoGluon's splits
           follow TabArena's seed (an explicit ``learner_kwargs["random_state"]`` wins).
        2. On the single-model holdout path, carve a single task-aware validation split off the
           training data (``_apply_task_specific_holdout``) and hand it to ``TabularPredictor`` as
           explicit ``tuning_data`` — a single model does not consume the bagged ``custom_splits``.
        3. If ``feature_generator_cls`` is given, instantiate it (forwarding any group/time
           split columns it accepts) into ``fit_kwargs["feature_generator"]``.
        4. Assemble ``train_data`` by appending the label column; attach tuning/validation
           data when provided.

        When both ``init_kwargs["default_base_path"]`` and the environment variable
        ``TABARENA_MODEL_ARTIFACTS_BASE_PATH`` are set, the artifact root of this fit is the
        environment value (a SLURM job points it at node-local scratch). The override happens on the
        deep copy, so the configured ``init_kwargs`` (and the ``init_kwargs_extra`` metadata) are
        untouched.

        Returns:
        -------
        (train_data, init_kwargs, fit_kwargs)
            Ready to pass to ``TabularPredictor(**init_kwargs).fit(train_data, **fit_kwargs)``.
        """
        init_kwargs = copy.deepcopy(self.init_kwargs)
        fit_kwargs = copy.deepcopy(self.fit_kwargs)

        base_path_override = os.environ.get(MODEL_ARTIFACTS_BASE_PATH_ENV)
        if base_path_override and "default_base_path" in init_kwargs:
            init_kwargs["default_base_path"] = base_path_override

        # Name the internal label column from the validation metadata (a sentinel when unset). The
        # name is purely internal — predictions / artifacts use the task's own label + cleaner.
        label = self.validation_metadata.get_target_name()
        init_kwargs["label"] = label

        num_folds = self._apply_validation_splits(fit_kwargs, X=X, y=y)
        if X_val is None and not self.no_validation:
            X, y, X_val, y_val = self._apply_task_specific_holdout(X=X, y=y, num_folds=num_folds)
        if fit_kwargs.get("validation_structure") is not None:
            # The learner's random_state also seeds AutoGluon's default splitter, so it is set only
            # alongside a declared structure: an unstructured task keeps the default splitter's seed.
            init_kwargs.setdefault("learner_kwargs", {}).setdefault("random_state", self.split_random_state)
        self._apply_feature_generator(fit_kwargs)

        # TODO: think about if we can reset the index here without breaking simulation artifacts
        train_data = self._attach_label(X, y, label=label, data_owned=data_owned)
        if X_val is not None:
            fit_kwargs["tuning_data"] = self._attach_label(X_val, y_val, label=label, data_owned=data_owned)

        return train_data, init_kwargs, fit_kwargs

    def _apply_validation_splits(self, fit_kwargs: dict, *, X: pd.DataFrame, y: pd.Series) -> int | None:
        """Resolve the bagging counts (+ any custom splits) from the protocol into ``fit_kwargs`` in place.

        A bagged wrapper (``bagged_fit=True``) takes ``num_bag_folds`` / ``num_bag_sets`` from
        ``self.validation_protocol``: resolved for the training split's (group) instance count when
        the protocol has a tiny-data regime, and with ``task_specific_validation`` run through
        ``resolve_validation_splits`` (data-dependent clamps plus explicit ``custom_splits``). A
        holdout wrapper (``bagged_fit=False``) resolves nothing here. A full predictor
        (``bagged_fit=None``) passes its own ``fit_kwargs`` counts through and, when the protocol
        asks for task-specific validation, declares the task's grouped / temporal structure as
        ``fit_kwargs["validation_structure"]`` (``_declare_validation_structure``): AutoGluon then
        builds the folds, or the holdout split, itself. What was resolved is kept on
        ``self._validation_resolution`` for the result record.

        Returns the effective ``num_folds``: ``None`` (or ``<= 1``) signals the non-bagged holdout
        path, which ``_build_predictor_args`` then handles via a single task-aware split for a
        single-model wrapper.
        """
        num_folds = fit_kwargs.pop("num_bag_folds", None)
        num_repeats = fit_kwargs.pop("num_bag_sets", None)

        if self.bagged_fit is True:
            if num_folds is not None or num_repeats is not None:
                given = (("num_bag_folds", num_folds), ("num_bag_sets", num_repeats))
                replacement = ", ".join(f"{key}={value!r}" for key, value in given if value is not None)
                raise ValueError(
                    "A bagged wrapper takes its counts from `validation_protocol`: pass "
                    f"validation_protocol=ValidationProtocol({replacement}) instead of `num_bag_folds` / "
                    "`num_bag_sets` in `fit_kwargs`."
                )
            if self.validation_protocol is None:
                raise ValidationProtocolError(
                    "A bagged fit needs a validation protocol: run the experiment through an arena context "
                    "(which supplies its official protocol) or pass validation_protocol=ValidationProtocol(...).",
                )
            return self._resolve_bagged_fit(fit_kwargs, protocol=self.validation_protocol, X=X, y=y)

        if self.bagged_fit is False:
            if num_folds is not None or num_repeats is not None:
                raise ValueError(
                    "A holdout wrapper fits no bag; drop `num_bag_folds` / `num_bag_sets` from `fit_kwargs`."
                )
            return None

        # Full predictor: the counts are fit as given (without them AutoGluon decides); the task's
        # structure, when the protocol asks for it, is declared rather than resolved here.
        if num_folds is not None:
            fit_kwargs["num_bag_folds"] = num_folds
        if num_repeats is not None:
            fit_kwargs["num_bag_sets"] = num_repeats
        self._declare_validation_structure(fit_kwargs, num_folds=num_folds, num_repeats=num_repeats)
        return num_folds

    def _declare_validation_structure(
        self, fit_kwargs: dict, *, num_folds: int | None, num_repeats: int | None
    ) -> None:
        """Declare the task's structure to AutoGluon for a full-predictor fit (in place).

        With a task-specific protocol and a task that has grouped or temporal structure,
        ``fit_kwargs["validation_structure"]`` is set from the validation metadata
        (:func:`~tabarena.benchmark.exec_models.autogluon_utils.validation_structure_from_metadata`)
        and AutoGluon resolves the splits: group-disjoint or time-blocked folds for a bagged fit,
        the matching holdout for a non-bagged one, and any clamping (fewer groups than folds,
        repeats collapsed) inside its ``ValidationStructure``. A task without such structure keeps
        AutoGluon's default splitter, as the single-model wrappers do (``resolve_validation_splits``
        returns no ``custom_splits`` for it). The class-count adaptation is switched on when the
        protocol asks for it. The resolution record notes what was declared.
        """
        protocol = self.validation_protocol
        structure = None
        if protocol is not None and protocol.task_specific_validation:
            structure = validation_structure_from_metadata(
                self.validation_metadata, temporal_forward_only=self.temporal_forward_only
            )
            if structure is None:
                logger.info(
                    "Task declares no grouped or temporal structure; AutoGluon's default splitter is used "
                    f"(num_bag_folds={num_folds}, num_bag_sets={num_repeats}).",
                )
            else:
                fit_kwargs["validation_structure"] = structure
                logger.info(
                    f"Delegating validation splitting to AutoGluon: {structure} (num_bag_folds={num_folds}, "
                    f"num_bag_sets={num_repeats}, split_random_state={self.split_random_state}).",
                )
        bagged = num_folds is not None and num_folds > 1
        if protocol is not None and protocol.adapt_num_folds_to_n_classes and bagged:
            fit_kwargs["adapt_num_bag_folds_to_n_classes"] = True
        self._validation_resolution = ValidationResolution(
            regime="explicit" if bagged else None,
            num_bag_folds_nominal=num_folds,
            num_bag_sets_nominal=num_repeats,
            num_bag_folds_resolved=num_folds if bagged else None,
            num_bag_sets_resolved=(num_repeats if num_repeats is not None else 1) if bagged else None,
            validation_structure=structure is not None,
            structure=structure_of(self.validation_metadata),
        )

    def _resolve_bagged_fit(
        self,
        fit_kwargs: dict,
        *,
        protocol: ValidationProtocol,
        X: pd.DataFrame,
        y: pd.Series,
        regime: Literal["explicit"] | None = None,
    ) -> int:
        """Write ``protocol``'s resolved bagging into ``fit_kwargs`` and record the resolution.

        The tiny-data regime is decided on the training split's (group) instance count. With
        ``task_specific_validation`` the counts and custom splits come from ``resolve_validation_splits``
        (whose data-dependent clamps are recorded), otherwise straight from the protocol. AutoGluon's
        class-count adaptation is switched on only when the protocol asks for it (AutoGluon's default
        is off). Returns the resolved ``num_folds``.
        """
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        clamps: list[str] = []
        custom_splits = None
        if protocol.task_specific_validation:
            num_group_instances = get_num_group_instances(self.validation_metadata, X=X)
            custom_splits, num_folds, num_repeats = resolve_validation_splits(
                self.validation_metadata, protocol, X=X, y=y, clamps=clamps
            )
        else:
            num_group_instances = (
                get_num_group_instances(self.validation_metadata, X=X) if protocol.has_tiny_regime else None
            )
            num_folds, num_repeats = protocol.resolve_num_splits(num_group_instances)
        nominal_folds, nominal_repeats = protocol.resolve_num_splits(num_group_instances)

        logger.info(f"Using num_folds: {num_folds}")
        fit_kwargs["num_bag_folds"] = num_folds
        logger.info(f"Using num_repeats: {num_repeats}")
        fit_kwargs["num_bag_sets"] = num_repeats
        if custom_splits is not None:
            logger.info("Using custom_splits for validation protocol.")
            fit_kwargs.setdefault("ag_args_ensemble", {})["custom_splits"] = custom_splits
        if protocol.adapt_num_folds_to_n_classes:
            fit_kwargs["adapt_num_bag_folds_to_n_classes"] = True

        self._validation_resolution = ValidationResolution(
            num_group_instances=num_group_instances,
            regime=regime if regime is not None else protocol.regime(num_group_instances),
            num_bag_folds_nominal=nominal_folds,
            num_bag_sets_nominal=nominal_repeats,
            num_bag_folds_resolved=num_folds,
            num_bag_sets_resolved=num_repeats,
            clamps=tuple(clamps),
            custom_splits=custom_splits is not None,
            num_custom_splits=len(custom_splits) if custom_splits is not None else None,
            structure=structure_of(self.validation_metadata),
        )
        return num_folds

    def _apply_task_specific_holdout(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        num_folds: int | None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame | None, pd.Series | None]:
        """Carve a single task-aware (group/temporal) validation split off the training data.

        Non-bagged counterpart of the bagged ``custom_splits`` path: a single ``TabularPredictor``
        fit does not consume ``ag_args_ensemble['custom_splits']`` (that is read only by the bagged
        ensemble), so the resolved holdout rows are returned as explicit ``X_val`` / ``y_val`` and
        fed to ``TabularPredictor`` as ``tuning_data`` instead.

        Only acts on the single-model holdout path: a task-specific validation protocol and no
        bagging (``num_folds`` is ``None`` / ``<= 1``). A full predictor (``bagged_fit=None``)
        never carves here: its holdout follows the ``validation_structure`` it declared, which
        AutoGluon resolves itself. Otherwise, or when the task carries no grouped/temporal
        structure (``resolve_holdout_split`` returns ``None``), returns the data unchanged with
        ``X_val=None`` so AutoGluon's built-in holdout is used.

        Returns ``(X_train, y_train, X_val, y_val)``; rows keep their original index.
        """
        protocol = self.validation_protocol
        if self.bagged_fit is None or protocol is None or not protocol.task_specific_validation:
            return X, y, None, None
        if num_folds is not None and num_folds > 1:
            return X, y, None, None

        X_reset = X.reset_index(drop=True)
        num_group_instances = get_num_group_instances(self.validation_metadata, X=X_reset)
        nominal_folds, nominal_repeats = protocol.resolve_num_splits(num_group_instances)
        split = resolve_holdout_split(self.validation_metadata, protocol, X=X_reset, y=y.reset_index(drop=True))
        self._validation_resolution = ValidationResolution(
            num_group_instances=num_group_instances,
            regime=protocol.regime(num_group_instances),
            num_bag_folds_nominal=nominal_folds,
            num_bag_sets_nominal=nominal_repeats,
            task_specific_holdout=split is not None,
            holdout_rows=None if split is None else len(split[1]),
            structure=structure_of(self.validation_metadata),
        )
        if split is None:
            return X, y, None, None

        train_idx, val_idx = split
        logger.info(
            f"Using task-specific holdout split as tuning_data: {len(train_idx)} train / "
            f"{len(val_idx)} validation rows.",
        )
        # Return standalone copies (not ``.iloc`` views): ``_attach_label`` sets the label column
        # on an owned frame in place, which would raise a pandas ``SettingWithCopyWarning`` on a
        # slice. The original index is preserved.
        return (
            X.iloc[train_idx].copy(),
            y.iloc[train_idx].copy(),
            X.iloc[val_idx].copy(),
            y.iloc[val_idx].copy(),
        )

    def _apply_feature_generator(self, fit_kwargs: dict) -> None:
        """Instantiate ``feature_generator_cls`` into ``fit_kwargs["feature_generator"]`` (in place).

        No-op when ``feature_generator_cls`` is absent. The task's group/time split columns (from
        ``self.validation_metadata``) are always forwarded via the shared
        :func:`~tabarena.benchmark.preprocessing.build_feature_generator`, which passes them only to a
        generator that accepts them — so they take effect for the group-aware TabArena generator and
        are ignored by AutoGluon's default one. No gate is needed (the consuming generator is only
        present for the TabArena pipeline, which is the right setting).
        """
        feature_generator_cls = fit_kwargs.pop("feature_generator_cls", None)
        feature_generator_kwargs = fit_kwargs.pop("feature_generator_kwargs", {})
        if feature_generator_cls is None:
            return

        fit_kwargs["feature_generator"] = build_feature_generator(
            feature_generator_cls,
            feature_generator_kwargs,
            group_cols=self.validation_metadata.group_on,
            group_labels=self.validation_metadata.group_labels,
            group_time_on=self.validation_metadata.group_time_on,
        )

    def _attach_label(self, X: pd.DataFrame, y: pd.Series, *, label: str, data_owned: bool) -> pd.DataFrame:
        """Return ``X`` with ``y`` appended as the ``label`` column.

        An owned frame is edited in place (``fit_custom`` hands over frames it copied before the fit
        timer, or that it loaded itself); a caller's frame is copied first.
        """
        data = X if data_owned else X.copy()
        data[label] = y
        return data

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs,
    ):
        """Resolve the validation protocol, then construct and fit the ``TabularPredictor``."""
        from autogluon.tabular import TabularPredictor

        train_data, init_kwargs, fit_kwargs = self._build_predictor_args(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
            data_owned=kwargs.get("data_owned", False),
        )

        self.predictor = TabularPredictor(
            problem_type=self.problem_type,
            eval_metric=self.eval_metric,
            **init_kwargs,
        )
        self.predictor.fit(
            train_data=train_data,
            **fit_kwargs,
        )

        return self

    def get_validation_record(self) -> dict:
        """What this fit resolved from the protocol, plus the counts AutoGluon's trainer fitted.

        The resolved part comes from ``_apply_validation_splits`` / ``_apply_task_specific_holdout``
        (empty for a full predictor that left the validation to AutoGluon); the fitted part reads the
        trainer's ``k_fold`` / ``n_repeats``.
        """
        record = self._validation_resolution.to_record() if self._validation_resolution is not None else {}
        trainer = getattr(getattr(self, "predictor", None), "_trainer", None)
        if trainer is not None:
            record["num_bag_folds_fitted"] = getattr(trainer, "k_fold", None)
            record["num_bag_sets_fitted"] = getattr(trainer, "n_repeats", None)
        return record

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict labels with the fitted predictor (already-preprocessed ``X``)."""
        return self.predictor.predict(X)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict class probabilities with the fitted predictor (already-preprocessed ``X``)."""
        return self.predictor.predict_proba(X)

    def pre_predict(self):
        """Persist the fitted model(s) in memory and run their untimed inference preparation.

        Runs once, outside the inference timer, through
        :func:`~tabarena.benchmark.exec_models.persist_inference.persist_for_inference`: the best
        model and its ancestors are persisted (memory-guarded by ``persist_max_memory``; when the
        models don't fit nothing is persisted and inference loads from disk on demand), a bag's
        children that are still path strings are loaded, and every persisted object (the bag and its
        loaded children) that declares ``prepare_for_inference`` runs it under the contract written on
        ``AbstractExecModel.pre_predict``. A failing hook is isolated per object and leaves the persist
        outcome recorded. The outcome lands on ``self._persisted_models`` and ``self._prepared_models``
        for the fit metadata. The served models stay resident until ``cleanup``.
        """
        if not self.persist:
            return
        outcome = persist_for_inference(self.predictor, max_memory=self.persist_max_memory)
        self._persisted_models = outcome.persisted_models
        self._prepared_models = outcome.prepared_models

    def get_oof(self) -> dict:
        """Return the predictor's simulation artifact, narrowed to the best model's val proba."""
        # TODO: Rename method
        if self.no_validation:
            return self._simulation_artifact_without_validation()
        simulation_artifact = self.predictor.simulation_artifact()
        simulation_artifact["pred_proba_dict_val"] = simulation_artifact["pred_proba_dict_val"][
            self.predictor.model_best
        ]
        return simulation_artifact

    def _simulation_artifact_without_validation(self) -> dict:
        """The simulation artifact of a fit without validation rows: metadata plus an empty validation slot.

        ``pred_proba_dict_val`` holds a zero-row frame (multiclass) or series (binary, regression) in the
        predictor's internal label space, already narrowed to the one model as :meth:`get_oof` returns it,
        and ``y_val`` an empty series, so every consumer that indexes the validation slot sees the usual
        types with no rows; the runner fills the test predictions.
        """
        label_cleaner = self.predictor._learner.label_cleaner
        num_classes = label_cleaner.num_classes
        if self.problem_type == "multiclass" and num_classes is not None and num_classes > 2:
            pred_val = pd.DataFrame(np.empty((0, num_classes), dtype=np.float32), columns=list(range(num_classes)))
        else:
            pred_val = pd.Series(np.empty(0, dtype=np.float32), dtype=np.float32)
        y_val = pd.Series(np.empty(0, dtype=np.float64 if self.problem_type == "regression" else np.int64))
        return {
            "pred_proba_dict_val": pred_val,  # narrowed to the one model, as ``get_oof`` returns it
            "y_val": y_val,
            "eval_metric": self.predictor.eval_metric.name,
            "problem_type": self.predictor.problem_type,
            "problem_type_transform": label_cleaner.problem_type_transform,
            "ordered_class_labels": label_cleaner.ordered_class_labels,
            "ordered_class_labels_transformed": label_cleaner.ordered_class_labels_transformed,
            "num_classes": num_classes,
            "label": self.predictor.label,
        }

    def get_metric_error_val(self) -> float:
        """Return the best model's validation metric error from the predictor leaderboard.

        A no-validation fit has no validation rows, so it has no error (None, as the leaderboard reports it) and the
        leaderboard is not built.
        """
        if self.no_validation:
            return None
        # FIXME: this shouldn't be calculating its own val score, that should be external. This should simply give val pred and val pred proba
        leaderboard = self.predictor.leaderboard(score_format="error", set_refit_score_to_parent=True)
        metric_error_val = leaderboard.set_index("model").loc[self.predictor.model_best]["metric_error_val"]
        if metric_error_val is not None and not np.isnan(metric_error_val):
            metric_error_val = float(metric_error_val)
        return metric_error_val

    def cleanup(self):
        """Release the served models, delete the predictor's on-disk artifacts and free CPU/GPU memory.

        The served models are released here rather than in ``post_predict`` because metadata and
        bag-artifact collection reuse them. Guarded on the predictor existing, so it is safe to call
        after a failed fit (``ExperimentRunner(cleanup_on_failure=True)``); ``empty_cache`` failures are
        logged and never mask an exception being propagated. The shared-weights registry is dropped
        only when ``release_shared_weights_on_cleanup`` is set.
        """
        predictor = getattr(self, "predictor", None)
        if predictor is not None and self.persist:
            release_after_inference(predictor)
        else:
            free_inference_memory()
        if predictor is not None:
            shutil.rmtree(predictor.path, ignore_errors=True)
        if self.release_shared_weights_on_cleanup:
            from tabarena.models import _weights

            _weights.release()


def _hyperparameters_user_from_info(info: dict) -> dict:
    """The user-specified hyperparameters of a model, read from its ``get_info()`` output.

    Same result as ``TabularPredictor.model_hyperparameters(model, output_format="user")``,
    which computes a fresh ``get_info()`` internally; this reads an already collected one. For a
    bagged model the child's hyperparameters are returned, with the bag's own under
    ``ag_args_ensemble`` when any were given.
    """
    if "bagged_info" in info:
        hyperparameters = info["bagged_info"]["child_hyperparameters_user"].copy()
        if info["hyperparameters_user"]:
            hyperparameters["ag_args_ensemble"] = info["hyperparameters_user"]
        return hyperparameters
    return info["hyperparameters_user"]


_CUSTOM_SPLITS_KEY = "custom_splits"

#: The bag-level AutoGluon params worth recording next to the protocol: they shape how many
#: children a bag really has and how they are fit (model-class defaults included).
_BAG_PARAM_KEYS = ("use_child_oof", "refit_folds", "num_folds", "max_sets", "fold_fitting_strategy", "stratify", "bin")

#: ``TabularPredictor.fit`` arguments that change the validation behind the wrapper's protocol.
_VALIDATION_BYPASS_FIT_KEYS = (
    "validation_structure",
    "validation_size_curves",
    "validation_mode",
    "use_bag_holdout",
    "holdout_frac",
    "tuning_data",
    "dynamic_stacking",
    "auto_stack",
    "refit_full",
    "set_best_to_refit_full",
    "ds_args",
)

#: ``TabularPredictor(...)`` arguments that change the validation behind the wrapper's protocol
#: (``groups`` pins the folds to the group count and one repeat inside the learner).
_VALIDATION_BYPASS_INIT_KEYS = ("groups", "learner_kwargs")


def _without_custom_splits(params: Any) -> tuple[Any, int]:
    """Copy of a hyperparameter dict without ``custom_splits`` (top level or under ``ag_args_ensemble``).

    Returns the object itself with a count of 0 when there is nothing to strip.
    """
    if not isinstance(params, dict):
        return params, 0
    stripped = 0
    if _CUSTOM_SPLITS_KEY in params:
        params = {k: v for k, v in params.items() if k != _CUSTOM_SPLITS_KEY}
        stripped += 1
    nested = params.get("ag_args_ensemble")
    if isinstance(nested, dict) and _CUSTOM_SPLITS_KEY in nested:
        params = {**params, "ag_args_ensemble": {k: v for k, v in nested.items() if k != _CUSTOM_SPLITS_KEY}}
        stripped += 1
    return params, stripped


def _strip_custom_splits_from_info(info: dict) -> tuple[dict, int]:
    """Copy of a model's ``get_info()`` dict without the custom-split index arrays.

    Task-specific folds are passed as predictor-level ``ag_args_ensemble["custom_splits"]``, which
    AutoGluon merges into every model's user params, so the index arrays (one per fold, the size of
    the training split) would otherwise be pickled into every result and into the processed
    hyperparameter tables. Returns ``(info, 0)`` with the very same object when nothing had to be
    stripped, so callers can rely on identity in that case.
    """
    total = 0
    new_info = dict(info)
    for key in ("hyperparameters_user", "hyperparameters"):
        if key in new_info:
            new_info[key], stripped = _without_custom_splits(new_info[key])
            total += stripped
    bagged_info = new_info.get("bagged_info")
    if isinstance(bagged_info, dict):
        new_bagged_info = dict(bagged_info)
        for key in ("child_hyperparameters_user", "child_hyperparameters"):
            if key in new_bagged_info:
                new_bagged_info[key], stripped = _without_custom_splits(new_bagged_info[key])
                total += stripped
        new_info["bagged_info"] = new_bagged_info
    return (new_info, total) if total else (info, 0)


class AGSingleWrapper(AGWrapper):
    """Fit a single AutoGluon model (no weighted ensemble) inside a ``TabularPredictor``.

    This is the common path for benchmarking one model family: it forces
    ``fit_weighted_ensemble=False`` and passes ``{model_cls: model_hyperparameters}`` as the
    predictor's ``hyperparameters``. Predictor/ensemble-level options that would conflict
    with fitting a single model are rejected up front (see ``_validate_fit_kwargs``);
    model-level options belong in ``model_hyperparameters``.

    Parameters
    ----------
    model_cls: str | type[AbstractModel]
        The model class (or its AutoGluon registry key) to fit, as used in
        ``predictor.fit(..., hyperparameters={model_cls: model_hyperparameters})``.
    model_hyperparameters: dict
        Hyperparameters for ``model_cls`` (including any ``ag_args_fit`` / ``ag_args_ensemble``).

        Persisting skips AutoGluon's memory check here (``persist_max_memory = None``): one model,
        or one bag of it, is bounded by construction, so the check's cost -- pickling every model
        to size it -- buys nothing.
    calibrate: bool | str, default False
        Forwarded to ``TabularPredictor.fit(calibrate=...)``.
    no_validation: bool, default False
        Fit the one model on every training row with no validation split at all (AutoGluon's
        ``validation_mode="none"``): no bag, no holdout, one fit, the same predictions as an outer fit
        of the model. The simulation artifact still carries the test predictions (an empty validation
        slot), so the run's predictions can be cached and combined afterwards by methods that need no
        validation estimate. Incompatible with a bagged wrapper.
    init_kwargs, fit_kwargs:
        Extra predictor constructor / fit kwargs (the "extra" kwargs recorded in metadata).
    """

    persist_max_memory: float | None = None

    bagged_fit: ClassVar[bool | None] = False
    per_child_test_source: str | None = None
    """How the per-child test artifact was produced, when a bag artifact was collected.

    ``"timed_children_pass"`` when the timed predict ran the children pass and kept each child's
    output (see ``AGSingleBagWrapper.pre_predict``), ``"timed_prediction"`` when a single-child bag
    reused the timed prediction, ``"child_forward_pass"`` when the children predicted again, ``None``
    before ``bag_artifact`` ran (and always for a non-bagged wrapper). Recorded in the fit metadata so
    reruns of nondeterministic GPU models can be interpreted without logs.
    """

    def __init__(
        self,
        model_cls: str | type[AbstractModel],
        model_hyperparameters: dict,
        calibrate: bool | str = False,
        init_kwargs: dict | None = None,
        fit_kwargs: dict | None = None,
        no_validation: bool = False,
        **kwargs,
    ):
        assert isinstance(model_cls, str) or issubclass(model_cls, AbstractModel)
        assert isinstance(model_hyperparameters, dict)

        if fit_kwargs is None:
            fit_kwargs = {}
        if init_kwargs is None:
            init_kwargs = {}
        self._validate_fit_kwargs(fit_kwargs)
        self._validate_init_kwargs(init_kwargs)
        if no_validation and self.bagged_fit:
            raise ValueError("`no_validation` fits one model on all rows; it cannot be combined with a bagged wrapper.")

        # Record the user-provided "extra" kwargs (used for metadata), then derive the
        # effective fit kwargs by forcing the single-model contract on top of them.
        self.init_kwargs_extra = init_kwargs

        fit_kwargs = copy.deepcopy(fit_kwargs)
        fit_kwargs["calibrate"] = calibrate
        self.fit_kwargs_extra = fit_kwargs

        fit_kwargs = copy.deepcopy(fit_kwargs)
        fit_kwargs["fit_weighted_ensemble"] = False
        if no_validation:
            fit_kwargs["validation_mode"] = "none"
        fit_kwargs["hyperparameters"] = {model_cls: model_hyperparameters}

        self._model_cls = model_cls
        self.model_hyperparameters = model_hyperparameters
        # The best model's `get_info()` as collected by `get_metadata` (custom splits stripped), reused
        # by `get_validation_record` so the bag is inspected once.
        self._collected_info: dict | None = None
        self._collected_child_oof: bool | None = None

        super().__init__(
            init_kwargs=init_kwargs,
            fit_kwargs=fit_kwargs,
            **kwargs,
        )
        self.no_validation = no_validation

    @staticmethod
    def _validate_fit_kwargs(fit_kwargs: dict) -> None:
        """Reject ``fit_kwargs`` incompatible with fitting a single model under the validation protocol.

        Options interpreted at the predictor/ensemble level (``presets``,
        ``num_stack_levels``, ``fit_weighted_ensemble``, ...) or with a dedicated wrapper
        argument (``calibrate``) must not be passed here; model-level options such as
        ``ag_args_fit`` / ``ag_args_ensemble`` belong in ``model_hyperparameters``. Predictor-level
        validation knobs (``validation_structure``, ``use_bag_holdout``, ``holdout_frac``,
        ``tuning_data``, ...) would change the validation behind the protocol's back; an
        AutoGluon-driven validation belongs to a full predictor (``AGExperiment``) or a system.
        """
        disallowed = {
            "hyperparameters": "Must not specify `hyperparameters` in AGSingleWrapper.",
            "num_stack_levels": "num_stack_levels is not allowed for AGSingleWrapper.",
            "presets": "AGSingleWrapper does not support `presets`.",
            "fit_weighted_ensemble": (
                "Must not specify `fit_weighted_ensemble` in AGSingleWrapper... It is always set to False."
            ),
            "calibrate": "Specify calibrate directly rather than in `fit_kwargs`.",
            "ag_args_fit": "ag_args_fit must be specified in `model_hyperparameters`, not in `fit_kwargs`.",
            "ag_args_ensemble": "ag_args_ensemble must be specified in `model_hyperparameters`, not in `fit_kwargs`.",
        }
        for key in _VALIDATION_BYPASS_FIT_KEYS:
            disallowed[key] = (
                f"`{key}` changes the validation behind the validation protocol; an AutoGluon-driven "
                "validation belongs to a full predictor (AGExperiment) or a system."
            )
        for key, message in disallowed.items():
            assert key not in fit_kwargs, message

    @staticmethod
    def _validate_init_kwargs(init_kwargs: dict) -> None:
        """Reject ``TabularPredictor(...)`` arguments that would re-derive the validation splits."""
        for key in _VALIDATION_BYPASS_INIT_KEYS:
            assert key not in init_kwargs, (
                f"`{key}` changes the validation behind the validation protocol; an AutoGluon-driven "
                "validation belongs to a full predictor (AGExperiment) or a system."
            )

    @classmethod
    def uses_ray(cls, method_kwargs: dict, *, problem_type: str | None = None) -> bool:
        """Whether this single-model fit can reach Ray (see ``AbstractExecModel.uses_ray``).

        Three predictor-level ``fit_kwargs`` start Ray on their own and answer True right away:
        ``fit_strategy`` other than ``"sequential"``, ``dynamic_stacking`` and ``auto_stack``
        (``presets`` and ``num_stack_levels`` are rejected by ``_validate_fit_kwargs`` and need no
        check). Without a bag (``num_bag_folds`` absent or at most 1) no fold-fitting strategy exists
        and the answer is False; the task-specific validation protocol can only lower the fold count,
        never turn a non-bagged fit into a bag. For a bag the answer is the fold fitting strategy
        AutoGluon would pick (``tabarena.models.warmup.resolve_fold_fitting_strategy`` over the model's
        default ``ag_args_ensemble`` merged under the user's), and when ``num_gpus`` is not known yet
        both the CPU and the GPU variant are considered, so an unknown GPU count errs towards True.
        ``AGSingleBagWrapper`` inherits this.
        """
        from tabarena.models.warmup import PARALLEL_FOLD_FITTING_STRATEGIES, resolve_fold_fitting_strategy

        fit_kwargs = method_kwargs.get("fit_kwargs") or {}
        if fit_kwargs.get("fit_strategy", "sequential") != "sequential":
            return True
        if fit_kwargs.get("dynamic_stacking") or fit_kwargs.get("auto_stack"):
            return True
        num_bag_folds = fit_kwargs.get("num_bag_folds")
        if isinstance(num_bag_folds, bool) or not isinstance(num_bag_folds, int | float) or num_bag_folds <= 1:
            return False
        model_cls = resolve_model_cls(method_kwargs["model_cls"])
        hyperparameters = method_kwargs.get("model_hyperparameters") or {}
        num_gpus = fit_kwargs.get("num_gpus")
        gpu_cases = [num_gpus] if num_gpus is not None else [0, 1]
        return any(
            resolve_fold_fitting_strategy(model_cls, hyperparameters, problem_type=problem_type, num_gpus=gpus)
            in PARALLEL_FOLD_FITTING_STRATEGIES
            for gpus in gpu_cases
        )

    def post_fit(self, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame):
        """Capture any model fit failures so the runner can record them on a crash."""
        self.failure_artifact = self.get_metadata_failure()

    def get_hyperparameters(self, info: dict | None = None) -> dict:
        """Return the best model's hyperparameters in user-facing form.

        ``info`` is the best model's ``get_info()`` output. ``get_metadata`` passes the one it
        already collected, because ``get_info`` on a bagged model reloads its children from disk
        and pickles every model to measure its size, which for a foundation model means
        serialising the weights.
        """
        if info is None:
            info = self._load_model(assert_single_model=False).get_info(include_feature_metadata=False)
        return _hyperparameters_user_from_info(info)

    @property
    def model_cls(self) -> type[AbstractModel]:
        """The model class, resolving an AutoGluon registry key string when needed."""
        return resolve_model_cls(self._model_cls)

    def _load_model(self, assert_single_model: bool = True):
        """Load the fitted model object from the predictor's trainer.

        When ``assert_single_model`` is True, assert exactly one inferable model exists and
        load it; otherwise load the predictor's ``model_best``.
        """
        model_names = self.predictor.model_names(can_infer=True)
        if assert_single_model:
            assert len(model_names) == 1
            model_name = self.predictor.model_names()[0]
        else:
            model_name = self.predictor.model_best
        return self.predictor._trainer.load_model(model_name)

    def get_metadata_init(self, info: dict | None = None) -> dict:
        """Metadata known at construction time (model class, hyperparameters, extra kwargs).

        ``info`` is the best model's ``get_info()`` output, see ``get_hyperparameters``.
        """
        metadata = {}
        metadata["hyperparameters"] = self.get_hyperparameters(info=info)
        metadata["model_cls"] = self.model_cls.__name__
        metadata["model_type"] = self.model_cls.ag_key
        metadata["name_prefix"] = self.model_cls.ag_name
        metadata["model_hyperparameters"] = self.model_hyperparameters
        metadata["init_kwargs_extra"] = self.init_kwargs_extra
        metadata["fit_kwargs_extra"] = self.fit_kwargs_extra
        return metadata

    def get_metadata_fit(self, model: AbstractModel | None = None, info: dict | None = None) -> dict:
        """Metadata available only after fitting (info, disk/compute usage, fit metadata).

        ``model`` is the loaded best model and ``info`` its ``get_info()`` output; either is
        collected here when not given. Since the served models stay resident until ``cleanup``,
        ``_load_model`` returns the served object and ``get_info`` sizes the served children.

        Besides the existing keys the block carries the untimed inference-side bookkeeping:
        ``persist`` / ``persisted_models`` (which models were in memory during the timed predict;
        ``None`` = disabled, not run or failed, ``[]`` = skipped by the memory guard),
        ``prepared_for_inference`` (persisted objects whose ``prepare_for_inference`` hook ran),
        ``shared_weights`` (the process-wide registry ``report()`` and each bagged child's
        ``info["shared_weights"]`` when present; see ``tabarena.models._weights``) and
        ``per_child_test_source`` (how the bag artifact's per-child test predictions were produced:
        ``"timed_children_pass"``, ``"timed_prediction"`` or ``"child_forward_pass"``; ``None``
        until ``bag_artifact`` ran; the OOF runner refreshes it after the artifact).
        """
        metadata = {}
        metadata["persist"] = self.persist
        metadata["persisted_models"] = self._persisted_models
        # Names of the persisted objects whose untimed inference prep ran without error.
        metadata["prepared_for_inference"] = self._prepared_models
        if model is None:
            model = self._load_model(assert_single_model=False)
        if info is None:
            info = model.get_info(include_feature_metadata=False)
        metadata["info"] = info
        metadata["shared_weights"] = self._shared_weights_metadata(info)
        metadata["per_child_test_source"] = self.per_child_test_source
        metadata["disk_usage"] = model.disk_usage()
        metadata["num_cpus"] = model.fit_num_cpus
        metadata["num_gpus"] = model.fit_num_gpus
        metadata["num_cpus_child"] = model.fit_num_cpus_child
        metadata["num_gpus_child"] = model.fit_num_gpus_child
        metadata["fit_metadata"] = model.get_fit_metadata()
        if hasattr(model, "_memory_usage_estimate"):
            metadata["memory_usage_estimate"] = model._memory_usage_estimate
        return metadata

    @staticmethod
    def _shared_weights_metadata(info: dict | None) -> dict:
        """The ``shared_weights`` metadata block: the registry report plus each bagged child's own entry.

        ``children`` maps a bagged child's name to its ``info["shared_weights"]`` (``None`` for a
        child that shares nothing); for a non-bagged model the block is in ``info`` itself and
        ``children`` is empty. The registry module is torch-free and imported by name at call time.
        """
        from tabarena.models import _weights

        children_info = info.get("children_info") if isinstance(info, dict) else None
        children = {}
        if isinstance(children_info, dict):
            children = {
                name: child.get("shared_weights") for name, child in children_info.items() if isinstance(child, dict)
            }
        return {"registry": _weights.report(), "children": children}

    def get_metadata_failure(self) -> dict:
        """Record any per-model fit failures reported by the predictor."""
        return {
            "model_failures": self.predictor.model_failures(),
        }

    def get_metadata(self) -> dict:
        """Combined construction-time and post-fit metadata for this model.

        The best model is loaded and its ``get_info()`` collected once, then shared by both parts
        and kept for ``get_validation_record``. Custom-split index arrays are stripped from the
        collected info first (see ``_strip_custom_splits_from_info``).
        """
        model = self._load_model(assert_single_model=False)
        info, _ = _strip_custom_splits_from_info(model.get_info(include_feature_metadata=False))
        self._collected_info = info
        self._collected_child_oof = getattr(model, "_child_oof", None)
        metadata = self.get_metadata_init(info=info)
        metadata.update(self.get_metadata_fit(model=model, info=info))
        return metadata

    def get_validation_record(self) -> dict:
        """The resolved / fitted counts plus what the bag itself reports.

        Extends ``AGWrapper.get_validation_record`` with the bag's own bookkeeping from the info
        collected by ``get_metadata``: the child count, the folds per repeat, whether the children
        were replaced by a single child with its own out-of-fold estimate (``child_oof``, the
        ``use_child_oof`` case) and the bag-level params that shape the children
        (``use_child_oof``, ``refit_folds``, ``fold_fitting_strategy``, ...).
        """
        record = super().get_validation_record()
        info = self._collected_info
        if info is None:
            return record
        bagged_info = info.get("bagged_info")
        if isinstance(bagged_info, dict):
            record["num_child_models"] = bagged_info.get("num_child_models")
            record["n_repeats_bag"] = bagged_info.get("_n_repeats")
            record["k_per_n_repeat"] = bagged_info.get("_k_per_n_repeat")
            record["bagged_mode"] = bagged_info.get("bagged_mode")
        record["child_oof"] = self._collected_child_oof
        params = info.get("hyperparameters")
        if isinstance(params, dict):
            record["bag_params"] = {key: params[key] for key in _BAG_PARAM_KEYS if key in params}
        return record


class AGSingleBagWrapper(AGSingleWrapper):
    """A bagged ``AGSingleWrapper`` that also exposes its per-child (per-fold) artifacts.

    Identical fitting to ``AGSingleWrapper`` except that the fold and repeat counts come from the
    validation protocol, and it advertises and provides the per-bagged-child out-of-fold validation
    indices and test predictions needed for ensemble simulation.

    The per-child test predictions come out of the timed predict. When the served bag supports it
    (AutoGluon's ``predict_proba_children`` and ``predict_proba_from_children``), the timed predict
    runs the children pass: every child predicts once, its array is kept, and the bag's own output is
    derived from the arrays with the arithmetic ``predict_proba`` uses, so ``bag_artifact`` hands them
    over instead of predicting again. Otherwise the timed predict takes the predictor's path; a
    single-child bag's artifact is then its timed output, and every other bag predicts a second time
    per child.
    """

    bagged_fit: ClassVar[bool | None] = True

    # Bagging exposes per-child OOF predictions and their validation indices.
    can_get_per_child_oof = True
    can_get_per_child_val_idx = True

    _timed_children_bag = None
    """The served bag ``pre_predict`` cleared for the children pass; ``None`` when the timed predict takes the predictor's path."""
    _timed_children: list[np.ndarray] | None = None
    """The per-child arrays the timed predict's children pass kept (child output space, timed row order), until
    ``bag_artifact`` takes them or ``cleanup`` drops them."""

    def pre_predict(self):
        """Persist the served bag (``AGWrapper.pre_predict``), then decide the timed predict's path.

        When the served bag supports it, the timed predict runs the children pass
        (``_predict_children_pass``): ``predict_proba_children`` once, ``predict_proba_from_children``
        for the bag's own output, the learner's post-processing for the user-facing frame; the per-child
        arrays it keeps become the bag artifact, so no child predicts a second time. The checks run
        here, outside the timer. A bag that fails them (an AutoGluon without
        ``predict_proba_from_children``, a learner without the post-processing hooks, a label cleaner
        that changes the problem type) keeps the predictor's own ``predict_proba`` / ``predict``, and
        the artifact predicts per child afterwards.
        """
        super().pre_predict()
        self._timed_children = None
        self._timed_children_bag = None
        try:
            model = self._load_model()
            reason = self._children_pass_unavailable_reason(model)
        except Exception as exc:
            reason = f"the served bag could not be inspected: {exc!r}"
        if reason is None:
            self._timed_children_bag = model
        else:
            logger.info(
                f"The timed predict takes the predictor's path; the bag artifact predicts per child ({reason})."
            )

    def cleanup(self):
        """Drop the timed predict's per-child arrays, then release the served models (``AGWrapper.cleanup``)."""
        self._timed_children = None
        self._timed_children_bag = None
        super().cleanup()

    def _children_pass_unavailable_reason(self, model) -> str | None:
        """Why the timed predict cannot run the children pass on ``model``, or ``None`` when it can."""
        for method in ("predict_proba_children", "predict_proba_from_children"):
            if not callable(getattr(model, method, None)):
                return f"the served bag has no {method}, an AutoGluon older than the per-child pass"
        if self.problem_type not in ("regression", "quantile") and not model.can_predict_proba():
            return "the served bag cannot predict probabilities"
        learner = self.predictor._learner
        for method in ("transform_features", "_post_process_predict_proba", "_post_process_predict"):
            if not callable(getattr(learner, method, None)):
                return f"the learner has no {method}"
        if learner.label_cleaner.problem_type_transform != self.problem_type:
            return "the learner's label cleaner changes the problem type"
        return None

    @staticmethod
    def _children_pass_covers(model, *, n_rows: int) -> bool:
        """Whether the children pass reproduces ``predict_proba`` on ``n_rows`` rows: no bag-level chunking applies.

        ``predict_proba_children`` does not chunk by the bag's own ``ag.max_batch_size`` (each child
        chunks by its own), so a bag whose limit is below the row count keeps the predictor's path.
        """
        max_batch_size = model._get_max_batch_size()
        return max_batch_size is None or max_batch_size >= n_rows

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """The timed ``predict_proba``: the children pass when ``pre_predict`` cleared it, else the predictor's own."""
        bag = self._timed_children_bag
        if bag is None or not self._children_pass_covers(bag, n_rows=len(X)):
            return super()._predict_proba(X)
        y_pred_proba, children = self._predict_children_pass(bag, X)
        y_pred_proba = self.predictor._learner._post_process_predict_proba(
            y_pred_proba=y_pred_proba, as_pandas=True, index=X.index, as_multiclass=True, inverse_transform=True
        )
        self._timed_children = children
        return y_pred_proba

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        """The timed ``predict`` (regression): the children pass when ``pre_predict`` cleared it, else the predictor's own."""
        bag = self._timed_children_bag
        if bag is None or not self._children_pass_covers(bag, n_rows=len(X)):
            return super()._predict(X)
        learner = self.predictor._learner
        y_pred_proba, children = self._predict_children_pass(bag, X)
        problem_type = learner.label_cleaner.problem_type_transform or learner.problem_type
        y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=problem_type)
        y_pred = learner._post_process_predict(y_pred=y_pred, as_pandas=True, index=X.index, inverse_transform=True)
        self._timed_children = children
        return y_pred

    def _predict_children_pass(self, bag, X: pd.DataFrame) -> tuple[np.ndarray, list[np.ndarray]]:
        """``TabularPredictor.predict_proba``'s work for the served bag, keeping every child's array.

        The learner's feature transform, then ``BaggedEnsembleModel.predict_proba_children`` (one pass
        over the children, each array kept) and ``predict_proba_from_children`` (their mean and the
        bag's calibration, the arithmetic ``predict_proba`` uses), so the bag's output is what the
        predictor's path computes and the per-child arrays cost no second pass. Returns the bag's
        output in the internal label space and the per-child arrays: each child's ``predict_proba`` on
        the preprocessed rows, before the bag's calibration, in ``X``'s row order.
        """
        X_inner = self.predictor._learner.transform_features(X)
        children = bag.predict_proba_children(X=X_inner)
        return bag.predict_proba_from_children(children), children

    def bag_artifact(self, X_test: pd.DataFrame, *, y_pred=None, y_pred_proba=None) -> dict:
        """Collect per-child test predictions and validation indices for the bagged model.

        The timed outputs (``y_pred`` / ``y_pred_proba`` in the caller's original row order) are
        forwarded to ``get_per_child_test``, which takes the arrays the timed predict's children pass
        kept, or reuses the outputs for a single-child bag, instead of predicting again;
        ``per_child_test_source`` records which path produced the artifact.
        """
        model = self._load_model()
        bag_info = {}
        bag_info["pred_proba_test_per_child"] = self.get_per_child_test(
            X_test=X_test, model=model, y_pred=y_pred, y_pred_proba=y_pred_proba
        )
        bag_info["val_idx_per_child"] = self.get_per_child_val_idx(model=model)
        return bag_info

    def get_per_child_val_idx(self, model=None) -> list[np.ndarray]:
        """Return each child's out-of-fold validation indices (into the internal train data).

        A refit bag (``_refit_oof``) carries the indices of the folds it discarded and
        ``get_oof_fold_val_idx`` returns that stored list without reading ``X`` or ``y``, so the
        internal training data is loaded only for the other branches (the splitters need it).
        """
        if model is None:
            model = self._load_model()

        get_oof_fold_val_idx = getattr(model, "get_oof_fold_val_idx", None)
        if (
            get_oof_fold_val_idx is not None
            and getattr(model, "_refit_oof", False)
            and getattr(model, "_oof_fold_val_idx", None) is not None
        ):
            # The refit branch of AutoGluon's get_oof_fold_val_idx returns the stored indices and never
            # touches X or y, so the internal data stays on disk.
            val_idx_per_child = get_oof_fold_val_idx(X=None, y=None)
        else:
            X, y = self.predictor.load_data_internal()
            if get_oof_fold_val_idx is not None:
                val_idx_per_child = get_oof_fold_val_idx(X=X, y=y)
            else:
                # LEGACY: drop this branch once AutoGluon >= 1.6.2 is the floor, and call
                # `model.get_oof_fold_val_idx` unconditionally.
                #
                # It reproduces that method for older AutoGluon, where a bagged model exposes only
                # its splitters. Note what it *cannot* reproduce: a `refit_folds` model there reports
                # a single child covering every row, because the folds that made its OOF were
                # discarded along with their splitters, so the fold structure is simply unavailable.
                all_kfolds = []
                if model._child_oof:
                    all_kfolds = [(None, X.index.values)]
                else:
                    for n_repeat, k in enumerate(model._k_per_n_repeat):
                        kfolds = model._cv_splitters[n_repeat].split(X=X, y=y)
                        all_kfolds += kfolds[n_repeat * k : (n_repeat + 1) * k]
                val_idx_per_child = [val_idx for _train_idx, val_idx in all_kfolds]

        return [pd.to_numeric(val_idx, downcast="integer") for val_idx in val_idx_per_child]  # memory opt

    def get_per_child_test(
        self,
        X_test: pd.DataFrame,
        model=None,
        *,
        y_pred=None,
        y_pred_proba=None,
    ) -> list[np.ndarray]:
        """Return each child's predictions on ``X_test`` (float32), in the original row order.

        Three paths, tried in this order; ``per_child_test_source`` records which one produced the
        artifact. The arrays the timed predict's children pass kept are the artifact
        (``_take_timed_children``). A single-child bag whose output equals its child's reuses the timed
        prediction (``_per_child_test_from_timed_output``). Otherwise every child predicts on the served
        bag: the same deterministic test-row shuffle as inference is applied (see ``_shuffle_test_rows``)
        and inverted on the per-child outputs. The pass's arrays are taken, and dropped, before the
        paths are tried, so they never outlive the artifact.
        """
        if model is None:
            model = self._load_model()

        timed_children = self._take_timed_children(model=model, n_rows=len(X_test))
        if timed_children is not None:
            self.per_child_test_source = "timed_children_pass"
            logger.info("Per-child test artifact taken from the timed predict's children pass.")
            return timed_children
        reused = self._per_child_test_from_timed_output(
            model=model, n_rows=len(X_test), y_pred=y_pred, y_pred_proba=y_pred_proba
        )
        if reused is not None:
            self.per_child_test_source = "timed_prediction"
            logger.info("Per-child test artifact reused the timed prediction (single-child bag).")
            return reused
        self.per_child_test_source = "child_forward_pass"
        logger.info("Per-child test artifact computed by a per-child forward pass.")

        X_test, inv_perm, original_index = self._shuffle_test_rows(X_test)

        X_test = self.transform_X(X=X_test)

        X_test_inner = self.predictor.transform_features(data=X_test, model=model.name)

        if model.can_predict_proba():
            per_child_test_preds = model.predict_proba_children(X=X_test_inner)
        else:
            per_child_test_preds = model.predict_children(X=X_test_inner)

        if self.shuffle_test:
            # Inverse-permute outputs back to original X_test order
            per_child_test_preds = [
                _apply_inv_perm(y_pred, inv_perm, index=original_index) for y_pred in per_child_test_preds
            ]

        return [preds_child.astype(np.float32) for preds_child in per_child_test_preds]  # memory opt

    def _per_child_test_from_timed_output(
        self,
        *,
        model,
        n_rows: int,
        y_pred,
        y_pred_proba,
    ) -> list[np.ndarray] | None:
        """The single child's test artifact derived from the timed prediction, or ``None`` to fall back.

        Why the result equals ``predict_proba_children`` / ``predict_children`` on the child: a
        one-child bag returns ``child_output / 1`` (AutoGluon's
        ``BaggedEnsembleModel._predict_proba_internal``, float32 stays float32), the trainer applies
        no further transform to an L1 model, and the learner's ``_post_process_predict_proba`` is
        inverted exactly by its own ``_pre_process_predict_proba``: multiclass stays float32 end to
        end (zero columns for unseen classes added, then dropped), binary passes through a float64
        two-column buffer whose values are the float32 ones (the cast back is lossless), and regression
        is the child's float32 values (``LabelCleanerDummy`` identities). The timed output is already in
        the caller's original row order, so no shuffle inversion is needed.

        Reuse happens only when every guard holds, otherwise the forward-pass path runs:
        ``model.n_children == 1``; no post-hoc ``temperature_scalar`` or ``conformalize``; no
        bag-level chunking (``_get_max_batch_size()`` is ``None`` or covers all rows); the learner has
        ``_pre_process_predict_proba`` and its label cleaner's ``problem_type_transform`` equals this
        wrapper's ``problem_type`` (rules out the multiclass-to-binary cleaner); the timed object is
        present with ``n_rows`` rows (``y_pred_proba`` for classification, ``y_pred`` for regression).
        Any exception during the conversion also falls back.

        Returns:
            ``[array]`` with one float32 array in the child's internal output space, or ``None``.
        """
        timed = y_pred if self.problem_type == "regression" else y_pred_proba
        try:
            if not self._bag_output_equals_child_output(model, n_rows=n_rows) or timed is None or len(timed) != n_rows:
                return None
            if self.problem_type == "regression":
                arr = timed.to_numpy() if hasattr(timed, "to_numpy") else timed
            else:
                arr = self.predictor._learner._pre_process_predict_proba(
                    timed, as_multiclass=True, inverse_transform=True
                )
            arr = np.asarray(arr)
        except Exception as exc:
            logger.warning(
                "Reusing the timed prediction as the per-child test artifact failed; predicting with the child "
                f"instead. ({exc!r})"
            )
            return None
        return [arr.astype(np.float32)]

    def _take_timed_children(self, *, model, n_rows: int) -> list[np.ndarray] | None:
        """The per-child test artifact from the timed predict's children pass, or ``None`` to fall back.

        Takes, and drops, the arrays ``_predict_children_pass`` kept: one per child in ``model.models``
        order, each child's ``predict_proba`` on the preprocessed rows before the bag's calibration, the
        very arrays ``predict_proba_children`` returns. The timed predict ran on the shuffled test rows,
        so the inverse of the same deterministic permutation (``shuffle_seed``) is applied, as the
        forward-pass path does to its own outputs; the arrays are then cast to float32.

        Falls back when the pass did not run, when ``model`` is not the bag it ran on, or when the
        number of arrays or their row count does not match the bag's children and ``n_rows`` (a predict
        that ran on other rows).
        """
        children, self._timed_children = self._timed_children, None
        if children is None:
            return None
        bag = self._timed_children_bag
        if bag is None or getattr(model, "name", None) != getattr(bag, "name", None):
            logger.warning("The timed predict's children pass ran on another bag; predicting per child instead.")
            return None
        n_children = getattr(model, "n_children", None)
        if n_children is None:
            n_children = len(getattr(model, "models", None) or [])
        if len(children) != n_children or any(len(arr) != n_rows for arr in children):
            logger.warning(
                f"The timed predict's children pass kept {len(children)} arrays with {[len(arr) for arr in children]} "
                f"rows for a bag of {n_children} children on {n_rows} rows; predicting per child instead."
            )
            return None
        if self.shuffle_test:
            _perm, inv_perm = _make_perm(n_rows, seed=self.shuffle_seed)
            children = [_apply_inv_perm(arr, inv_perm) for arr in children]
        return [np.asarray(arr).astype(np.float32, copy=False) for arr in children]  # memory opt

    def _bag_output_equals_child_output(self, model, *, n_rows: int) -> bool:
        """Whether the served bag's prediction on ``n_rows`` rows is exactly its single child's output.

        The guards of ``_per_child_test_from_timed_output``: one child, no post-hoc transform, no
        bag-level chunking, and a learner whose post-processing is invertible for this problem type.
        """
        if getattr(model, "n_children", None) != 1:
            return False
        if getattr(model, "temperature_scalar", None) is not None or getattr(model, "conformalize", None) is not None:
            return False
        max_batch_size = model._get_max_batch_size()
        if max_batch_size is not None and max_batch_size < n_rows:
            return False
        learner = self.predictor._learner
        if getattr(learner, "_pre_process_predict_proba", None) is None:
            return False
        return learner.label_cleaner.problem_type_transform == self.problem_type


class AGModelWrapper(AbstractExecModel):
    """Fit a single AutoGluon model directly, bypassing ``TabularPredictor``.

    Instantiates ``model_cls`` and calls its ``fit`` on all of ``X``/``y`` (no train/val
    split, no bagging, no ensemble). Used to benchmark methods that want to train on the
    full data (e.g. via ``AGModelOuterExperiment``). Unlike ``AGWrapper`` this does not
    carry the validation protocol and provides no OOF / metadata capabilities.

    Preprocessing is shared with the validation path via a named ``preprocessing_pipeline``
    (see :func:`~tabarena.benchmark.preprocessing.resolve_preprocessing_pipeline`): its
    model-agnostic feature generator is applied through ``AbstractExecModel``'s
    ``preprocess_data`` path (``_make_feature_generator``), and its model-specific step is
    injected into ``hyperparameters`` (which the AutoGluon model applies in its own ``fit``).
    ``None`` / ``"default"`` keeps AutoGluon's standard ``AutoMLPipelineFeatureGenerator``;
    ``"tabarena_default"`` uses the TabArena pipeline — functionally the same preprocessing as
    an ``AGWrapper`` configured with the same pipeline.

    Parameters
    ----------
    model_cls: type[AbstractModel]
        AutoGluon model class to fit.
    hyperparameters: dict, optional
        Model hyperparameters; the resolved pipeline's model-specific step is merged in.
    preprocessing_pipeline: str | None, default None
        Pipeline name passed to ``resolve_preprocessing_pipeline``.

    Grouped-task columns are sourced from the task's ``validation_metadata`` (injected by the
    runner) — the same source ``AGWrapper`` uses — so there is nothing group-related to pass here.
    """

    model: AbstractModel
    """The fitted single AutoGluon model (set by ``_fit``)."""

    def __init__(
        self,
        model_cls: type[AbstractModel],
        hyperparameters: dict | None = None,
        *,
        fit_kwargs: dict | None = None,
        preprocessing_pipeline: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert issubclass(model_cls, AbstractModel)
        self.model_cls = model_cls
        self._prepared_models: list[str] | None = None
        if hyperparameters is None:
            hyperparameters = {}
        # Passed straight to the model's `fit` (e.g. num_cpus / num_gpus / time_limit). The
        # AutoGluon model applies/ignores each as appropriate.
        self.fit_kwargs = fit_kwargs or {}

        pipeline = resolve_preprocessing_pipeline(preprocessing_pipeline)
        self.preprocessing_pipeline = preprocessing_pipeline
        self._feature_generator_cls = pipeline.feature_generator_cls
        self._feature_generator_kwargs = pipeline.feature_generator_kwargs
        # Model-specific preprocessing rides on the model's hyperparameters (the AutoGluon model
        # applies it in its own fit), so this works the same here as in the AGWrapper path.
        self.hyperparameters = pipeline.apply_model_specific(hyperparameters)

    @property
    def warmup_fn(self) -> Callable[[], WarmupReport] | None:
        """Warm this model class's environment (imports, kernels, CUDA context, shared weights, dummy fit).

        No AutoGluon-stack or Ray warm-up on this path (no trainer, no parallel folds). Returns a
        ``WarmupReport``.
        """
        return self._warmup

    def _warmup_cuda(self) -> bool | None:
        num_gpus = self.fit_kwargs.get("num_gpus")
        return None if num_gpus is None else num_gpus > 0

    @property
    def num_cpus_budget(self) -> int | None:
        """The ``num_cpus`` in ``fit_kwargs`` (``None`` when the fit is left to auto-detect it)."""
        return self.fit_kwargs.get("num_cpus")

    def _warmup(self) -> WarmupReport:
        from tabarena.models.warmup import warmup_feature_generator_cls, warmup_model_cls

        report = self._declared_warmup()
        warmup_model_cls(
            self.model_cls,
            problem_type=self.problem_type,
            num_cpus=self.fit_kwargs.get("num_cpus"),
            num_gpus=self.fit_kwargs.get("num_gpus"),
            hyperparameters=self.hyperparameters,
            report=report,
            dummy_fit=self.warmup_dummy_fit,
        )
        if self.preprocess_data:
            warmup_feature_generator_cls(self._feature_generator_cls, self._feature_generator_kwargs, report=report)
        return report

    def _make_feature_generator(self):
        """Build the pipeline's model-agnostic feature generator (shared with ``AGWrapper``).

        Group/time split columns come from the task's ``validation_metadata`` (injected uniformly by
        the runner) and are forwarded only to a generator that accepts them.
        """
        metadata = self.validation_metadata
        return build_feature_generator(
            self._feature_generator_cls,
            self._feature_generator_kwargs,
            group_cols=metadata.group_on,
            group_labels=metadata.group_labels,
            group_time_on=metadata.group_time_on,
        )

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Instantiate ``model_cls`` and fit it directly on the (preprocessed) data.

        The fitted model-agnostic generator's feature metadata goes along, so the special types it
        assigned (text embeddings, an exposed group key, ...) reach the model instead of being
        re-inferred from the frame's dtypes; the ``TabularPredictor`` path passes them the same way.
        """
        self.model = self.model_cls(
            path="",
            name=self.model_cls.__name__,
            problem_type=self.problem_type,
            eval_metric=self.eval_metric,
            hyperparameters=self.hyperparameters,
        )
        fit_kwargs = dict(self.fit_kwargs)
        if self.preprocess_data and self._feature_generator is not None and "feature_metadata" not in fit_kwargs:
            fit_kwargs["feature_metadata"] = self._feature_generator.feature_metadata
        self.model.fit(X=X, y=y, **fit_kwargs)
        return self

    @classmethod
    def uses_ray(cls, method_kwargs: dict, *, problem_type: str | None = None) -> bool:
        """False: a direct ``model_cls.fit`` on the full data has no bag, no trainer and no fold-fitting strategy."""
        return False

    def pre_predict(self):
        """Run the fitted model's untimed inference preparation (``prepare_for_inference``).

        The directly fitted model already lives in memory, so unlike ``AGWrapper`` there is no persist
        step; this only dispatches the optional model-level hook under the contract written on
        ``AbstractExecModel.pre_predict`` (a failure is logged and inference proceeds unprepared).
        ``get_metadata`` exposes the outcome.
        """
        self._prepared_models = dispatch_prepare_for_inference([self.model])

    def get_metadata(self) -> dict:
        """The inference-prep outcome (``prepared_for_inference``); there is no persist step on this path."""
        return InferencePersistence(prepared_models=self._prepared_models).as_metadata(persist=False)

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict labels with the fitted model, preserving ``X``'s index."""
        y_pred = self.model.predict(X)
        return pd.Series(y_pred, index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict class probabilities, widening binary output to two columns."""
        y_pred_proba = self.model.predict_proba(X)
        if self.problem_type == "binary":
            y_pred_proba = LabelCleanerMulticlassToBinary.convert_binary_proba_to_multiclass_proba(y_pred_proba)
        return pd.DataFrame(y_pred_proba, index=X.index)
