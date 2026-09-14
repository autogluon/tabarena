from __future__ import annotations

import copy
import dataclasses
import importlib
import inspect
import traceback
import warnings
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np
import yaml

from tabarena.benchmark.exec_models.autogluon import (
    AGModelWrapper,
    AGSingleBagWrapper,
    AGSingleWrapper,
    AGWrapper,
)
from tabarena.benchmark.exec_models.registry import infer_model_cls
from tabarena.benchmark.experiment.experiment_runner import ExperimentRunner, OOFExperimentRunner
from tabarena.benchmark.experiment.model_constraints import ModelConstraints
from tabarena.benchmark.validation_protocol import (
    CACHE_AUX_NAME,
    ValidationProtocol,
    ValidationProtocolError,
    validation_protocol_key,
)
from tabarena.utils.cache import AbstractCacheFunction, CacheFunctionDummy

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from autogluon.core.models import AbstractModel

    from tabarena.benchmark.exec_models.base import AbstractExecModel
    from tabarena.benchmark.preprocessing.text_cache import TextCacheMode
    from tabarena.benchmark.task import TaskWrapper
    from tabarena.benchmark.task.metadata import ValidationMetadata
    from tabarena.benchmark.validation_protocol import ValidationFlavour


class Experiment:
    """Experiment contains a method and the logic to run it on any task.
    Experiment is fully generic, and can accept any method_cls so long as it inherits from `AbstractExecModel`.

    Parameters
    ----------
    name: str
        The name of the experiment / method.
        Should be descriptive and unique compared to other methods.
        For example, `"LightGBM_c1_BAG_L1"`
    method_cls: Type[AbstractExecModel]
        The method class to be fit and evaluated.
    method_kwargs: dict
        The kwargs passed to the init of `method_cls`.
    experiment_cls: Type[ExperimentRunner], default OOFExperimentRunner
        The experiment class that wraps the method_cls.
        This class will track metadata information such as fit time, inference time, system resources, etc.
        It will also calculate the test `metric_error`, to ensure that the method_cls is evaluated correctly.
    experiment_kwargs: dict, optional
        The kwargs passed to the init of `experiment_cls`.
    preprocessing_pipeline: str | None, default None
        Name of an optional preprocessing pipeline to apply (e.g. ``"tabarena_default"``).
        Applied lazily at run time via ``_apply_preprocessing``; ``None`` applies none.
    validation_protocol: ValidationProtocol | dict | None, default None
        The :class:`~tabarena.benchmark.validation_protocol.ValidationProtocol` this experiment's
        inner validation follows (fold and repeat counts, tiny-data regime, task-specific splits,
        class-adaptive folds). A dict is accepted for the YAML round-trip and normalized. ``None``
        until an arena context stamps its protocol at ``build_jobs`` / ``run_jobs`` (see
        ``set_validation_protocol``); a bagged or holdout experiment cannot fit without one. Only
        ``AGWrapper``-based experiments act on it; an outer fit or a system ignores it.
    text_cache_mode: {"require", "auto", "off"}, default "off"
        How a text task's semantic-embedding cache is treated at fit time (see
        ``task_cache_scope``): ``require`` = the cache must be present (raise if missing),
        ``auto`` = use it if present else compute on the fly, ``off`` = ignore the cache.
        No-op for non-text tasks. Defaults to ``off`` for a standalone experiment; the curated
        ``TabArenaExperimentBundle`` suites set ``require`` on every experiment they build (see
        ``set_text_cache_mode``), so a prepared cache is mandatory there.
    model_constraints: ModelConstraints | dict | None, default None
        Optional dataset-compatibility constraints of this experiment's model (e.g. a max
        train-set size). A dict is accepted for the YAML round-trip and normalized to a
        :class:`~tabarena.benchmark.experiment.model_constraints.ModelConstraints`.
        Respected wherever jobs are enumerated or run (``build_jobs``, ``run_jobs``, the
        SLURM dispatch): incompatible (experiment, task-split) pairs are skipped. ``None``
        means unconstrained. ``TabArenaExperimentBundle`` attaches these automatically at
        build time (see ``set_model_constraints``).

    """

    # YAML: when True, ``from_yaml`` reads the wrapped class from ``model_cls`` (resolved via the
    # model registry, with ``ag_args_fit`` string-eval); when False, from ``method_cls``.
    _yaml_resolves_model_cls: bool = False

    #: How this experiment validates (see ``ValidationFlavour``). ``None`` for a bare experiment
    #: over a custom exec model; subclasses declare ``bagged`` / ``holdout`` / ``outer`` /
    #: ``system`` / ``predictor``. Arena contexts treat ``bagged`` and ``system`` as official.
    VALIDATION_FLAVOUR: ClassVar[ValidationFlavour | None] = None

    # --- Construction ----------------------------------------------------------------
    def __init__(
        self,
        name: str,
        method_cls: type[AbstractExecModel],
        method_kwargs: dict,
        *,
        experiment_cls: type[ExperimentRunner] = OOFExperimentRunner,
        experiment_kwargs: dict | None = None,
        preprocessing_pipeline: str | None = None,
        text_cache_mode: TextCacheMode = "off",
        model_constraints: ModelConstraints | dict | None = None,
        validation_protocol: ValidationProtocol | dict | None = None,
    ):
        if experiment_kwargs is None:
            experiment_kwargs = {}
        method_kwargs = copy.deepcopy(method_kwargs)
        experiment_kwargs = copy.deepcopy(experiment_kwargs)
        assert isinstance(name, str)
        assert len(name) > 0, "Name cannot be empty!"
        assert isinstance(method_kwargs, dict)
        assert isinstance(experiment_kwargs, dict)
        self.name = name
        self.method_cls = method_cls
        self.method_kwargs = method_kwargs
        self.experiment_cls = experiment_cls
        self.experiment_kwargs = experiment_kwargs
        # Name of an optional preprocessing pipeline to apply, applied lazily at
        # run time via `_apply_preprocessing` (see that method).
        self.preprocessing_pipeline = preprocessing_pipeline
        # How a text task's semantic-embedding cache is treated at fit time (require/auto/off),
        # independent of the validation protocol (see `task_cache_scope`). No-op for non-text tasks.
        self.text_cache_mode = text_cache_mode
        # Dataset-compatibility constraints of this experiment's model (None = unconstrained).
        self.model_constraints = self._normalize_model_constraints(model_constraints)
        # The validation protocol this experiment runs under (None until an arena context stamps
        # its own, or for flavours without inner validation). `_locals` keeps the normalized object
        # so the YAML round trip reproduces the experiment exactly.
        self.validation_protocol = ValidationProtocol.from_config(validation_protocol)
        if "validation_protocol" in self._locals:
            self._locals["validation_protocol"] = self.validation_protocol

    def __new__(cls, *args, **kwargs):
        """Capture the constructor arguments on ``self._locals`` for YAML round-tripping.

        Runs before ``__init__`` and records the (deep-copied) positional + keyword args
        keyed by parameter name, so ``to_yaml_dict`` can later re-emit exactly what this
        experiment was built from. Positional args are matched to names via the subclass
        ``__init__`` signature.
        """
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        _args = copy.deepcopy(args)
        _kwargs = copy.deepcopy(kwargs)
        arg_names = [param.name for param in params.values() if param.name != "self"]
        for i, arg in enumerate(_args):
            arg_name = arg_names[i]
            assert arg_name not in kwargs
            _kwargs[arg_name] = arg

        instance = super().__new__(cls)
        instance._locals = {**_kwargs}
        return instance

    @staticmethod
    def _normalize_model_constraints(
        model_constraints: ModelConstraints | dict | None,
    ) -> ModelConstraints | None:
        """Normalize the ``model_constraints`` input (a dict comes from the YAML form)."""
        if isinstance(model_constraints, dict):
            return ModelConstraints(**model_constraints)
        return model_constraints

    def set_model_constraints(self, model_constraints: ModelConstraints | dict | None) -> None:
        """Attach dataset-compatibility constraints after construction.

        Equivalent to passing ``model_constraints`` to the constructor, including the
        YAML round-trip (``_locals`` is kept in sync). Used by builders that obtain
        experiments from config generators rather than calling the constructor directly
        (e.g. ``TabArenaExperimentBundle``).
        """
        self.model_constraints = self._normalize_model_constraints(model_constraints)
        if self.model_constraints is None:
            self._locals.pop("model_constraints", None)
        else:
            self._locals["model_constraints"] = self.model_constraints

    def set_text_cache_mode(self, text_cache_mode: TextCacheMode) -> None:
        """Set the text-embedding cache mode after construction (kept in sync for YAML).

        Equivalent to passing ``text_cache_mode`` to the constructor. Used by builders that
        obtain experiments from config generators rather than the constructor directly — e.g.
        ``TabArenaExperimentBundle``, which enforces ``require`` on every experiment it builds.
        """
        self.text_cache_mode = text_cache_mode
        self._locals["text_cache_mode"] = text_cache_mode

    def set_validation_protocol(self, validation_protocol: ValidationProtocol | dict | None) -> None:
        """Set the validation protocol after construction (kept in sync for YAML).

        Equivalent to passing ``validation_protocol`` to the constructor. Used by the arena context,
        which stamps its protocol onto the experiments it runs (``AbstractArenaContext.build_jobs`` /
        ``run_jobs``), so the protocol travels with the experiment into the ``JobBatch`` and to the
        compute node.
        """
        self.validation_protocol = ValidationProtocol.from_config(validation_protocol)
        if self.validation_protocol is None:
            self._locals.pop("validation_protocol", None)
        else:
            self._locals["validation_protocol"] = self.validation_protocol

    def validation_record(self) -> dict:
        """The static part of a result's ``validation_protocol`` record: flavour, protocol and key.

        The runner merges the fitted part (what the wrapper resolved and fitted) on top; the
        ``key`` is the composite identity :func:`~tabarena.benchmark.validation_protocol.validation_protocol_key`
        computes for the record, so a holdout or outer fit never reads as the bagged protocol.
        """
        protocol = None if self.validation_protocol is None else self.validation_protocol.to_dict()
        record: dict[str, Any] = {"flavour": self.VALIDATION_FLAVOUR, "protocol": protocol}
        record["key"] = validation_protocol_key(record)
        return record

    def uses_ray(self, *, problem_type: str | None = None) -> bool:
        """Whether a fit of this experiment may start or use a Ray runtime.

        Consulted by the SLURM worker to decide whether to start Ray before the fit (see
        ``AbstractExecModel.uses_ray``). Delegates to ``method_cls.uses_ray`` with a deep copy of
        ``method_kwargs`` (the experiment is never mutated) and answers True on any error, the safe
        default: Ray started up front rather than cold inside the timed fit.

        Args:
            problem_type: The task's problem type when known; a model's default ensemble arguments
                may depend on it.
        """
        try:
            return bool(self.method_cls.uses_ray(copy.deepcopy(self.method_kwargs), problem_type=problem_type))
        except Exception:
            return True

    # --- Execution (fit / run) -------------------------------------------------------
    def run(
        self,
        task: TaskWrapper | None,
        fold: int,
        task_name: str,
        *,
        cache_task_key: int | str,
        repeat: int = 0,
        sample: int = 0,
        cacher: AbstractCacheFunction | None = None,
        ignore_cache: bool = False,
        raise_on_failure: bool = True,
        **experiment_kwargs,
    ) -> dict | None:
        """Fit this experiment on a single (task, fold, repeat) and return its results.

        Single entry point for executing one experiment job end to end, so callers (e.g.
        ``_run_job_specs``) only hand over the task and cache and read back a result.
        It splits into two flows:

        - **Load flow** (``task is None``): load and return the cached ``results`` directly.
          No task configuration, preprocessing resolution, or fit-failure guard is needed —
          reached e.g. on a default-mode cache hit, where the caller passes no task.
        - **Fit flow** (``task`` provided):
            1. Open the task's text-embedding cache scope for the fit (``task_cache_scope``).
            2. Build the finalized ``method_kwargs`` (``fit_args``) — validation metadata,
               preprocessing, resources — via ``init_method_kwargs``.
            3. Fit via ``experiment_cls.init_and_run`` through ``cacher`` — which still
               short-circuits to the cached ``results`` on a hit instead of refitting.
            4. Guard against a non-finite final metric error (``_enforce_finite_metric``).
            5. Write the validation-protocol side record next to a freshly fitted result.

        In both flows an existing cache is first checked against this experiment's validation
        protocol (``_check_cached_validation_protocol``): a result fit under another protocol is
        refused rather than reused, unless ``ignore_cache`` refits it.

        Fit-flow failures are handled here: when ``raise_on_failure`` is False, a fit
        exception (or a non-finite-metric failure) is swallowed and ``None`` is returned;
        otherwise it propagates.

        Parameters
        ----------
        task: TaskWrapper | None
            The loaded task to fit on. ``None`` is allowed only on a cache hit (the cached
            ``results`` is loaded without a task); fitting with ``task=None`` is a no-op load.
        fold: int
            The fold index to fit.
        repeat: int, default 0
            The repeat index to fit.
        sample: int, default 0
            The sample index to fit.
        task_name: str
            Display name recorded on the results (used downstream as the ``dataset`` key).
        cache_task_key: int | str
            Canonical task identifier (OpenML task id or ``UserTask.slug``) used to key the
            task's text-embedding cache in ``task_cache_scope``.
        cacher: AbstractCacheFunction | None, default None
            Cacher for this job's ``results`` artifact. Defaults to a no-op in-memory cacher.
        ignore_cache: bool, default False
            If True, refit and overwrite the cache even when a cached result exists.
        raise_on_failure: bool, default True
            If True, fit exceptions and non-finite-metric failures propagate; if False, they
            are swallowed and ``None`` is returned.
        **experiment_kwargs
            Extra kwargs forwarded to ``experiment_cls.init_and_run`` (e.g. ``debug_mode``,
            ``eval_metric_name``).

        Returns:
        -------
        dict | None
            The experiment results, or ``None`` when the job failed (and was not raised).
        """
        if cacher is None:
            cacher = CacheFunctionDummy()

        cache_existed = cacher.exists
        if cache_existed and not ignore_cache:
            self._check_cached_validation_protocol(cacher)

        # Load flow: with no task to fit, load the cached result directly
        if task is None:
            return cacher.cache(fun=None, fun_kwargs=None, ignore_cache=ignore_cache)

        # Fit flow: create the task's cache scope eagerly (outside the try, so a misconfigured
        # task/experiment surfaces immediately), then prepare the task-specific fit inputs and
        # fit through the cacher.
        task_cache_cm = self.task_cache_scope(task=task, cache_task_key=cache_task_key)
        try:
            with task_cache_cm:
                fit_args = self.init_method_kwargs(task=task, debug_mode=bool(experiment_kwargs.get("debug_mode")))
                out = cacher.cache(
                    fun=self._init_and_run_recorded,
                    fun_kwargs=dict(
                        method_cls=self.method_cls,
                        task=task,
                        fold=fold,
                        repeat=repeat,
                        sample=sample,
                        task_name=task_name,
                        method=self.name,
                        fit_args=fit_args,
                        **self.experiment_kwargs,
                        **experiment_kwargs,
                    ),
                    ignore_cache=ignore_cache,
                )
                self._enforce_finite_metric(out=out, cacher=cacher)
                if ignore_cache or not cache_existed:
                    cacher.save_aux(CACHE_AUX_NAME, self.validation_aux_record())
        except Exception:
            if raise_on_failure:
                raise
            print(f"Experiment {self.name!r} failed on {task_name} (fold={fold}, repeat={repeat}):")
            traceback.print_exc()
            return None

        return out

    def validation_aux_record(self) -> dict:
        """The side record stored next to a cached result: the protocol ``key`` and ``flavour`` it was fit under."""
        record = self.validation_record()
        return {"key": record["key"], "flavour": record["flavour"]}

    def _check_cached_validation_protocol(self, cacher: AbstractCacheFunction) -> None:
        """Refuse an existing cached result that was fit under another validation protocol.

        Reads the side record written by a previous run (see ``validation_aux_record``); a cache without
        one (written before side records existed) is trusted. The results cache is keyed by experiment
        name only, so without this check a rerun under another protocol would silently reuse the old
        result as its own.
        """
        cached = cacher.load_aux(CACHE_AUX_NAME)
        if cached is None:
            return
        expected = self.validation_aux_record()["key"]
        if cached.get("key") != expected:
            raise ValidationProtocolError(
                f"The cached result of experiment {self.name!r} ({cacher.cache_file}) was fit under validation "
                f"protocol {cached.get('key')!r} (flavour {cached.get('flavour')!r}); this run asks for {expected!r}. "
                "Use a new expname / benchmark_name to keep both, or ignore_cache=True to refit and overwrite it.",
            )

    def _init_and_run_recorded(self, **kwargs) -> dict:
        """Run ``experiment_cls.init_and_run`` and complete the result's ``validation_protocol`` record.

        The runner writes what the fitted method reports (``get_validation_record``); this adds the
        experiment's static part (flavour, protocol, key; see ``validation_record``) underneath it, so
        the pickled result carries the whole record.
        """
        out = self.experiment_cls.init_and_run(**kwargs)
        out["validation_protocol"] = {**self.validation_record(), **(out.get("validation_protocol") or {})}
        return out

    def task_cache_scope(
        self,
        *,
        task: TaskWrapper,
        cache_task_key: int | str,
    ) -> AbstractContextManager:
        """Return the task's text-embedding cache scope for the fit (a null scope for non-text tasks).

        The text-embedding cache is loaded for *any* text task per :attr:`text_cache_mode`
        (``off`` for a standalone experiment, ``require`` when built by a TabArena bundle),
        independent of the validation protocol.

        This *also* validates *eagerly* (when called, not on ``__enter__``) that the experiment's
        validation protocol can run on this task: a bagged or holdout experiment must carry one, and a
        task-specific protocol needs a task wrapper that exposes the split structure. So a
        misconfiguration surfaces immediately at the call site, before any fit. The validation
        metadata itself is applied later, in ``init_method_kwargs``. This is part of the fit flow, so a
        task object is required.

        Parameters
        ----------
        task: TaskWrapper
            The loaded task to adapt to.
        cache_task_key: int | str
            The canonical task identifier used to key the task-specific text-embedding
            cache (OpenML task id, or ``UserTask.slug``). This is the same key used for
            the results cache path. Only the loader (not generation) is needed here, so
            the key alone suffices — the original task handle is not required.

        Returns:
        -------
        A context manager to wrap this experiment's ``run(...)`` call.
        """
        from contextlib import nullcontext

        # The validation protocol is checked eagerly, before any fit.
        self._validate_validation_protocol(task)

        # Non-text tasks have no embedding cache to load.
        if not task.has_text:
            return nullcontext()

        # Load this task's semantic-text embedding cache for the duration of the fit
        # (slug-keyed + encoder-versioned; restored afterwards). Under ``require`` (the bundle
        # default) a text task with no cache fails fast — warm it first via the prefetch/download
        # path or pre-generation; ``auto`` computes on the fly and ``off`` skips the cache.
        from tabarena.benchmark.preprocessing.text_cache import use_text_cache_for_task

        return use_text_cache_for_task(
            cache_task_key,
            has_text=True,
            mode=self.text_cache_mode,
        )

    def init_method_kwargs(self, *, task: TaskWrapper, debug_mode: bool = False) -> dict:
        """Build the finalized ``method_kwargs`` (passed as ``fit_args``) for a single fit.

        Works on a fresh deep copy of ``self.method_kwargs`` (the experiment is never mutated),
        chaining the ``_apply_*`` steps — each takes ``method_kwargs`` and returns the
        (possibly new) ``method_kwargs``:

        1. the task's validation metadata, injected uniformly for *every* method as read-only data
           (``_apply_validation_metadata``);
        2. the experiment's validation protocol, injected for the ``AGWrapper`` family, which acts on
           the metadata according to it (fold counts, task-specific splits);
        3. any preprocessing pipeline (``_apply_preprocessing``);
        4. any auto-detected compute resources (``_apply_resources``);
        5. in-process fold fitting when ``debug_mode`` (``_apply_debug_fold_fitting``).
        """
        method_kwargs = copy.deepcopy(self.method_kwargs)
        # Uniform: hand every method the task's validation metadata (read-only data), layering any
        # user-provided value over the task-derived base. Every ``AbstractExecModel`` accepts it.
        method_kwargs = self._apply_validation_metadata(method_kwargs, task_metadata=task.get_validation_metadata())
        # Policy: the AutoGluon wrappers act on the metadata according to the experiment's protocol.
        if self.validation_protocol is not None and issubclass(self.method_cls, AGWrapper):
            method_kwargs["validation_protocol"] = self.validation_protocol
        method_kwargs = self._apply_preprocessing(method_kwargs)
        method_kwargs = self._apply_resources(method_kwargs)
        if debug_mode:
            method_kwargs = self._apply_debug_fold_fitting(method_kwargs)
        return method_kwargs

    def _apply_debug_fold_fitting(self, method_kwargs: dict) -> dict:
        """In debug mode, default a bagged AG model to in-process fold fitting and return it.

        Bagged AutoGluon models otherwise fit their folds in parallel Ray workers.

        Only applies to :class:`AGSingleBagWrapper`-based experiments, and only when no
        ``fold_fitting_strategy`` is already set (an explicit choice — e.g. from a model's
        ``ag_args_ensemble`` or ``sequential_local_fold_fitting`` — always wins). No-op otherwise.
        """
        if not (isinstance(self.method_cls, type) and issubclass(self.method_cls, AGSingleBagWrapper)):
            return method_kwargs
        hyperparameters = method_kwargs.get("model_hyperparameters")
        if not isinstance(hyperparameters, dict):
            return method_kwargs
        hyperparameters.setdefault("ag_args_ensemble", {}).setdefault("fold_fitting_strategy", "sequential_local")
        return method_kwargs

    def _apply_preprocessing(self, method_kwargs: dict) -> dict:
        """Apply ``self.preprocessing_pipeline`` to ``method_kwargs`` and return it.

        Edits the given ``method_kwargs`` in place for the built-in pipelines (and returns
        it); a no-op when no pipeline is configured.
        """
        pipeline = self.preprocessing_pipeline
        if pipeline is None or pipeline == "default":
            return method_kwargs

        if pipeline == "tabarena_default":
            from tabarena.benchmark.preprocessing import (
                TabArenaModelAgnosticPreprocessing,
                TabArenaModelSpecificPreprocessing,
            )

            # Model-agnostic step: a feature generator AutoGluon applies to all data. Set the same
            # way for every flavour; ``AGWrapper`` builds it from these keys (forwarding the task's
            # group/time split columns) for both the single-model wrappers and the full predictor.
            method_kwargs["fit_kwargs"]["feature_generator_cls"] = TabArenaModelAgnosticPreprocessing
            method_kwargs["fit_kwargs"]["feature_generator_kwargs"] = {}
            # Model-specific step: augments each model's hyperparameters. The shape differs by
            # experiment type (single ``model_hyperparameters`` vs a full predictor's per-model
            # ``hyperparameters`` dict), so it is delegated to an overridable hook.
            self._apply_model_specific_preprocessing(method_kwargs, TabArenaModelSpecificPreprocessing)
            return method_kwargs

        if pipeline.startswith("FSBench__"):
            # Logic for the (experimental) feature selection benchmark. The helper works on a
            # whole experiment; we only take its resolved method_kwargs.
            from tabarena.benchmark.feature_selection_methods.feature_selection_benchmark_utils import (
                apply_fs_bench_preprocessing,
            )

            return apply_fs_bench_preprocessing(preprocessing_name=pipeline, experiment=self).method_kwargs

        raise ValueError(f"Preprocessing pipeline name '{pipeline}' not recognized.")

    def _apply_model_specific_preprocessing(self, method_kwargs: dict, model_specific_cls) -> None:
        """Augment the model hyperparameters in place with the model-specific preprocessing.

        Default (single-model config experiments — ``AGModelExperiment`` / ``AGModelBagExperiment``):
        wrap the single ``method_kwargs["model_hyperparameters"]``. ``AGExperiment`` (a full
        ``TabularPredictor`` run) overrides this to walk its per-model ``hyperparameters`` dict.
        """
        method_kwargs["model_hyperparameters"] = model_specific_cls.add_to_hyperparameters(
            method_kwargs["model_hyperparameters"],
        )

    @staticmethod
    def _apply_resources(method_kwargs: dict) -> dict:
        """Return ``method_kwargs`` with any ``None`` compute resources auto-detected.

        A serialized experiment can carry ``num_cpus=None`` / ``num_gpus=None`` /
        ``memory_limit=None`` to mean "detect on whatever node I run on", preserving
        per-node auto-detection while keeping the experiment self-contained. Returns the
        input unchanged when there is nothing to detect (no copy); otherwise a
        deep-copied ``method_kwargs`` with the detected values filled in.
        """
        fit_kwargs = method_kwargs.get("fit_kwargs") or {}
        detect_cpus = fit_kwargs.get("num_cpus", 0) is None
        detect_gpus = fit_kwargs.get("num_gpus", 0) is None
        detect_memory = fit_kwargs.get("memory_limit", 0) is None
        if not (detect_cpus or detect_gpus or detect_memory):
            return method_kwargs

        from tabarena.utils.resources import detect_memory_limit_gb, detect_num_cpus, detect_num_gpus

        method_kwargs = copy.deepcopy(method_kwargs)
        fit_kwargs = method_kwargs["fit_kwargs"]
        if detect_cpus:
            fit_kwargs["num_cpus"] = detect_num_cpus()
            print(f"num_cpus not provided, using detected number of CPUs: {fit_kwargs['num_cpus']}")
        if detect_gpus:
            fit_kwargs["num_gpus"] = detect_num_gpus()
            print(f"num_gpus not provided, using detected number of GPUs: {fit_kwargs['num_gpus']}")
        if detect_memory:
            fit_kwargs["memory_limit"] = detect_memory_limit_gb()
            print(f"memory_limit not provided, using detected memory size: {fit_kwargs['memory_limit']} GB")
        return method_kwargs

    @staticmethod
    def _apply_validation_metadata(method_kwargs: dict, *, task_metadata: ValidationMetadata) -> dict:
        """Set ``method_kwargs["validation_metadata"]`` from the task (in place) and return it.

        The task-derived ``task_metadata`` is the base, with any value already present in
        ``method_kwargs`` layered over it (per-key for a dict; see ``ValidationMetadata.from_config``).
        This runs uniformly for *every* experiment and only provides the metadata as read-only data;
        whether and how a method acts on it is decided by its validation protocol. Every
        ``AbstractExecModel`` accepts ``validation_metadata`` (handled by the base).
        """
        from tabarena.benchmark.task.metadata import ValidationMetadata

        method_kwargs["validation_metadata"] = ValidationMetadata.from_config(
            method_kwargs.get("validation_metadata"),
            base=task_metadata,
        )
        return method_kwargs

    # --- Utils -----------------------------------------------------------
    @staticmethod
    def _enforce_finite_metric(*, out: dict | None, cacher: AbstractCacheFunction) -> None:
        """Raise on a result with a non-finite final metric error, deleting its cache.

        Always enforced: this guards against silently accepting results from models that
        overflow during fitting. Inspects ``metric_error`` / ``metric_error_val``; if one
        is non-finite, the result's cache file is deleted and a ``RuntimeError`` is raised.
        Whether that propagates or is swallowed (the run returning ``None``) is decided by
        the ``raise_on_failure`` handling in ``run``, which wraps this call. A finite (or
        absent) metric is a no-op.
        """
        if out is None:
            return
        for metric_error_key in ("metric_error", "metric_error_val"):
            if metric_error_key not in out:
                continue
            if not np.isfinite(out[metric_error_key]):
                print(f"Non-finite final metric error detected: \t{metric_error_key}={out[metric_error_key]}. ")
                print("\tDeleting cache file and counting as failure.")
                cacher.delete_cache()
                raise RuntimeError(f"Non-finite metric error detected for key {metric_error_key!r}.")

    def _validate_validation_protocol(self, task: TaskWrapper) -> None:
        """Assert this experiment's validation protocol can run on ``task``.

        A bagged or holdout experiment needs a protocol (an arena context stamps its official one; a
        standalone run passes ``validation_protocol=`` explicitly). A task-specific protocol
        additionally needs a task wrapper whose ``get_validation_metadata`` carries the real split
        structure (see ``_validate_task_specific_validation_supported``). Flavours without inner
        validation (outer fits, systems) have nothing to check.
        """
        protocol = self.validation_protocol
        if protocol is None:
            if self.VALIDATION_FLAVOUR in ("bagged", "holdout"):
                raise ValidationProtocolError(
                    f"Experiment {self.name!r} has no validation protocol. Run it through an arena context "
                    "(TabArenaContext / BeyondArenaContext supply their official protocol) or pass "
                    "validation_protocol=ValidationProtocol(...) to the experiment or its bundle.",
                )
            return
        if protocol.task_specific_validation:
            self._validate_task_specific_validation_supported(task)

    def _validate_task_specific_validation_supported(self, task: TaskWrapper) -> None:
        """Assert task-specific validation is supported for ``task`` and this experiment.

        Requires a task wrapper whose ``get_validation_metadata`` carries the real split
        structure, an ``InMemoryTaskWrapper`` (projected from its task metadata) or an
        ``OpenMLTaskWrapper`` around a ``TabArenaOpenMLSupervisedTask``, and an
        ``AGWrapper``-based ``method_cls`` (the family that builds task-specific splits).
        """
        from tabarena.benchmark.task.in_memory import InMemoryTaskWrapper
        from tabarena.benchmark.task.openml import OpenMLTaskWrapper, TabArenaOpenMLSupervisedTask

        supported = isinstance(task, InMemoryTaskWrapper) or (
            isinstance(task, OpenMLTaskWrapper) and isinstance(task.task, TabArenaOpenMLSupervisedTask)
        )
        if not supported:
            raise ValueError(
                "A validation protocol with `task_specific_validation=True` requires a task wrapper with "
                "split-aware validation metadata: an `InMemoryTaskWrapper` or an `OpenMLTaskWrapper` around a "
                f"`TabArenaOpenMLSupervisedTask`; got {type(task).__name__}.",
            )

        if not issubclass(self.method_cls, AGWrapper):
            raise NotImplementedError(
                "A validation protocol with `task_specific_validation=True` requires an `AGWrapper`-based "
                f"method_cls, got {self.method_cls.__name__}.",
            )

    # --- (De)serialization -----------------------------------------------------------
    def to_yaml_dict(self) -> dict:
        """Return the YAML-serializable dict for this experiment: ``type`` + constructor args."""
        locals = self._locals
        locals_new = self._to_yaml_dict(locals=locals)
        assert "type" not in locals_new, "The `type` key is reserved for the class name."
        return dict(
            type=class_to_path(self.__class__),
            **locals_new,
        )

    def to_yaml(self, path: str):
        """Write this experiment to a YAML file at ``path``."""
        assert path.endswith(".yaml")

        yaml_out = self.to_yaml_dict()
        with open(path, "w") as outfile:
            yaml.dump(yaml_out, outfile, default_flow_style=False)

    def to_yaml_str(self) -> str:
        """Return this experiment serialized to a YAML string."""
        yaml_out = self.to_yaml_dict()
        return yaml.safe_dump(yaml_out, sort_keys=False, allow_unicode=True)

    def _to_yaml_dict(self, locals: dict) -> dict:
        """Convert captured constructor args to YAML-friendly values.

        Classes become import paths; a ``ModelConstraints`` or ``ValidationProtocol`` value becomes
        its plain field dict (``__init__`` normalizes it back on load).
        """
        locals_new = {}
        for k, v in locals.items():
            if inspect.isclass(v):
                v = class_to_path(v)
            elif isinstance(v, ModelConstraints):
                v = dataclasses.asdict(v)
            elif isinstance(v, ValidationProtocol):
                v = v.to_dict()
            locals_new[k] = v
        return locals_new

    @classmethod
    def from_yaml(cls, _context=None, **kwargs) -> Self:
        """Reconstruct an experiment from its YAML constructor args, resolving class-valued fields.

        Two shapes, selected by ``_yaml_resolves_model_cls``:

        - default — the wrapped class is stored under ``method_cls`` (resolved without the model
          registry); ``experiment_cls`` is resolved too when present.
        - model-experiment — the wrapped class is stored under ``model_cls`` (resolved via the
          model registry), and ``model_hyperparameters["ag_args_fit"]`` expression strings are
          evaluated.

        ``from_yaml`` is always invoked with keyword arguments (see
        ``YamlSingleExperimentSerializer.parse_method``), so the class field arrives in ``kwargs``.
        Experiments serialized before the validation protocol existed are migrated first (see
        :func:`_migrate_legacy_validation_kwargs`).
        """
        if _context is None:
            _context = globals()
        kwargs = _migrate_legacy_validation_kwargs(cls, kwargs)

        if cls._yaml_resolves_model_cls:
            kwargs["model_cls"] = resolve_class(
                kwargs["model_cls"], context=_context, registry_resolver=infer_model_cls
            )
            _eval_ag_args_fit_strings(kwargs.get("model_hyperparameters"), _context)
        else:
            kwargs["method_cls"] = resolve_class(kwargs["method_cls"], context=_context)
            if "experiment_cls" in kwargs:
                kwargs["experiment_cls"] = resolve_class(kwargs["experiment_cls"], context=_context)

        return cls(**kwargs)


class AGExperiment(Experiment):
    """Experiment wrapping a full AutoGluon ``TabularPredictor`` (fixes ``method_cls=AGWrapper``).

    Accepts the predictor ``init_kwargs`` / ``fit_kwargs`` directly (rather than nested in
    ``method_kwargs``) and defaults ``experiment_kwargs`` to disable simulation artifacts.

    Parameters
    ----------
    name: str
        The name of the experiment / method (descriptive + unique, e.g. ``"LightGBM_c1_BAG_L1"``).
    init_kwargs: dict, optional
        Extra ``TabularPredictor(...)`` constructor kwargs (stored as ``method_kwargs["init_kwargs"]``).
    fit_kwargs: dict, optional
        Extra ``TabularPredictor.fit(...)`` kwargs (stored as ``method_kwargs["fit_kwargs"]``);
        e.g. ``hyperparameters`` / ``presets`` / ``num_bag_folds`` / ``num_bag_sets``. A full
        predictor is the one experiment whose bagging AutoGluon (presets or these counts) decides;
        it is therefore not an official flavour of the arena contexts (see ``VALIDATION_FLAVOUR``).
    method_kwargs: dict, optional
        Extra kwargs for ``AGWrapper(...)`` — see that class for accepted keys (e.g.
        ``validation_metadata`` / ``persist``). Must not contain ``init_kwargs`` / ``fit_kwargs``
        (pass those directly).
    experiment_kwargs: dict, optional
        Runner kwargs, merged over the default ``{"compute_simulation_artifacts": False}``.
    **kwargs:
        Forwarded to ``Experiment.__init__`` (e.g. ``preprocessing_pipeline``,
        ``validation_protocol``, which a task-specific protocol uses for the task-aware holdout or
        custom splits of explicit counts).
    """

    VALIDATION_FLAVOUR: ClassVar[ValidationFlavour | None] = "predictor"
    _method_cls = AGWrapper
    _experiment_cls = OOFExperimentRunner

    def __init__(
        self,
        name: str,
        init_kwargs: dict | None = None,
        fit_kwargs: dict | None = None,
        method_kwargs: dict | None = None,
        experiment_kwargs: dict | None = None,
        **kwargs,
    ):
        # `init_kwargs` / `fit_kwargs` are first-class arguments here; reject them nested in
        # `method_kwargs` and copy before mutating so the caller's dict is left untouched.
        method_kwargs = copy.deepcopy(method_kwargs) if method_kwargs else {}
        assert "init_kwargs" not in method_kwargs, "Pass `init_kwargs` directly, not inside `method_kwargs`."
        assert "fit_kwargs" not in method_kwargs, "Pass `fit_kwargs` directly, not inside `method_kwargs`."
        if init_kwargs is not None:
            method_kwargs["init_kwargs"] = init_kwargs
        if fit_kwargs is not None:
            method_kwargs["fit_kwargs"] = fit_kwargs

        super().__init__(
            name=name,
            method_cls=self._method_cls,
            method_kwargs=method_kwargs,
            experiment_cls=self._experiment_cls,
            # Simulation artifacts off by default; the caller may override via experiment_kwargs.
            experiment_kwargs={"compute_simulation_artifacts": False, **(experiment_kwargs or {})},
            **kwargs,
        )

    def to_yaml_dict(self) -> dict:
        locals = super().to_yaml_dict()

        locals = copy.deepcopy(locals)
        items = list(locals.items())
        for k, v in items:
            if k == "fit_kwargs" and "hyperparameters" in v and isinstance(v["hyperparameters"], dict):
                hyperparameters = v["hyperparameters"]
                keys = list(hyperparameters.keys())
                for model in keys:
                    if inspect.isclass(model):
                        val = locals["fit_kwargs"]["hyperparameters"].pop(model)
                        locals["fit_kwargs"]["hyperparameters"][model.ag_key] = val
        return locals

    @classmethod
    def from_yaml(cls, _context=None, **kwargs) -> Self:
        if _context is None:
            _context = globals()
        kwargs = _migrate_legacy_validation_kwargs(cls, kwargs)
        from tabarena.benchmark.exec_models.registry import tabarena_model_registry

        tabarena_model_keys = tabarena_model_registry.keys

        if "experiment_cls" in kwargs:
            kwargs["experiment_cls"] = eval(kwargs["experiment_cls"], _context)  # noqa: S307
        if "fit_kwargs" in kwargs and "hyperparameters" in kwargs["fit_kwargs"]:
            if isinstance(kwargs["fit_kwargs"]["hyperparameters"], dict):
                hyperparameters = kwargs["fit_kwargs"]["hyperparameters"]
                keys = list(hyperparameters.keys())
                for model in keys:
                    if model in tabarena_model_keys:
                        val = kwargs["fit_kwargs"]["hyperparameters"].pop(model)
                        kwargs["fit_kwargs"]["hyperparameters"][tabarena_model_registry.key_to_cls(model)] = val
        return cls(**kwargs)

    def _apply_model_specific_preprocessing(self, method_kwargs: dict, model_specific_cls) -> None:
        """Augment a full predictor's per-model hyperparameters with model-specific preprocessing.

        A full ``TabularPredictor`` run has no single ``model_hyperparameters``; its
        ``fit_kwargs["hyperparameters"]`` maps each model key to a config dict *or a list of config
        dicts*. Every config is wrapped, so single- and multi-model AutoGluon experiments both get
        the same model-specific preprocessing as the single-model config experiments. A no-op when
        there is no ``hyperparameters`` dict (e.g. a preset-driven run) — the model-agnostic feature
        generator still applies.
        """
        hyperparameters = method_kwargs.get("fit_kwargs", {}).get("hyperparameters")
        if not isinstance(hyperparameters, dict):
            return
        for model_key, configs in hyperparameters.items():
            if isinstance(configs, list):
                hyperparameters[model_key] = [
                    model_specific_cls.add_to_hyperparameters(config) if isinstance(config, dict) else config
                    for config in configs
                ]
            elif isinstance(configs, dict):
                hyperparameters[model_key] = model_specific_cls.add_to_hyperparameters(configs)


class AGModelExperiment(Experiment):
    """Fit a single AutoGluon model (fixes ``method_cls=AGSingleWrapper``).

    Parameters
    ----------
    name: str
        The name of the experiment / method (descriptive + unique, e.g. ``"LightGBM_c1_BAG_L1"``).
    model_cls: type[AbstractModel]
        AutoGluon model class to fit.
    model_hyperparameters: dict
        AutoGluon model hyperparameters, identical to what you would pass to
        ``TabularPredictor.fit(..., hyperparameters={model_cls: model_hyperparameters})``.
    time_limit: float, optional
        Fit time limit in seconds. No limit if unspecified. Injected into the model's
        ``ag.max_time_limit`` (or into ``fit_kwargs["time_limit"]`` when
        ``time_limit_with_preprocessing``); must not be set inside ``fit_kwargs``.
    time_limit_with_preprocessing: bool, default False
        If True, ``time_limit`` also covers preprocessing time (set as ``fit_kwargs["time_limit"]``
        rather than the model's ``ag.max_time_limit``).
    raise_on_model_failure: bool, default True
        Raise any AutoGluon model failure in a debugger-friendly manner. Must not be set
        inside ``fit_kwargs``.
    method_kwargs: dict, optional
        Extra kwargs for ``AGSingleWrapper(...)`` — see that class for accepted keys (e.g.
        ``init_kwargs`` / ``fit_kwargs`` / ``validation_metadata``). ``fit_kwargs`` must not carry
        ``num_bag_folds`` / ``num_bag_sets`` / ``adapt_num_bag_folds_to_n_classes`` (the validation
        protocol owns them), and ``model_hyperparameters["ag_args_ensemble"]`` must not rewrite the
        bagging structure (``num_folds`` / ``max_sets`` / ``custom_splits``).
    **kwargs:
        Forwarded to ``Experiment.__init__`` (e.g. ``experiment_kwargs``,
        ``preprocessing_pipeline``, ``validation_protocol``).
    """

    VALIDATION_FLAVOUR: ClassVar[ValidationFlavour | None] = "holdout"
    _method_cls = AGSingleWrapper
    _experiment_cls = OOFExperimentRunner
    _yaml_resolves_model_cls = True

    #: Whether the fit ``time_limit`` bounds the whole bag (``ag_args_ensemble``) or the single model.
    _time_limit_on_bag: ClassVar[bool] = False
    #: ``ag_args_ensemble`` keys that rewrite the bagging structure the validation protocol owns.
    _BAGGING_STRUCTURE_KEYS: ClassVar[tuple[str, ...]] = ("num_folds", "max_sets", "custom_splits")

    def __init__(
        self,
        name: str,
        model_cls: type[AbstractModel],
        model_hyperparameters: dict,
        *,
        time_limit: float | None = None,
        time_limit_with_preprocessing: bool = False,
        raise_on_model_failure: bool = True,
        method_kwargs: dict | None = None,
        **kwargs,
    ):
        self._reject_legacy_bagging_kwargs(kwargs)
        method_kwargs = copy.deepcopy(method_kwargs) if method_kwargs else {}
        assert isinstance(model_hyperparameters, dict)
        self._validate_time_limit(time_limit)
        # These have dedicated constructor arguments; they must not be nested in fit_kwargs.
        self._reject_in_fit_kwargs(method_kwargs, "time_limit", "raise_on_model_failure")
        # The validation protocol owns the bagging: counts and class adaptation are resolved from it at
        # fit time, and no ag_args_ensemble key may rewrite the bag's structure.
        self._reject_in_fit_kwargs(method_kwargs, "num_bag_folds", "num_bag_sets", "adapt_num_bag_folds_to_n_classes")
        self._reject_bagging_structure_overrides(model_hyperparameters)

        fit_kwargs = method_kwargs.setdefault("fit_kwargs", {})
        if time_limit is not None:
            if time_limit_with_preprocessing:
                fit_kwargs["time_limit"] = time_limit
            else:
                model_hyperparameters = self._insert_time_limit(
                    model_hyperparameters=model_hyperparameters, time_limit=time_limit
                )
        fit_kwargs["raise_on_model_failure"] = raise_on_model_failure
        super().__init__(
            name=name,
            method_cls=self._method_cls,
            method_kwargs={
                "model_cls": model_cls,
                "model_hyperparameters": model_hyperparameters,
                **method_kwargs,
            },
            experiment_cls=self._experiment_cls,
            **kwargs,
        )

    def _to_yaml_dict(self, locals: dict) -> dict:
        """Serialize model_cls as an import path so custom/unregistered classes can
        be loaded without requiring TabArena registry registration.
        """
        locals = copy.deepcopy(locals)
        locals["model_cls"] = class_to_path(locals["model_cls"])
        return super()._to_yaml_dict(locals=locals)

    @staticmethod
    def _validate_time_limit(time_limit: float | None) -> None:
        """Assert ``time_limit`` is a positive number (when given)."""
        if time_limit is not None:
            assert isinstance(time_limit, (float, int)), "time_limit must be a number"
            assert time_limit > 0, "time_limit must be positive"

    def _reject_legacy_bagging_kwargs(self, kwargs: dict) -> None:
        """Raise a ``TypeError`` naming the replacement when the bagging counts arrive as constructor kwargs."""
        legacy = {key: kwargs[key] for key in ("num_bag_folds", "num_bag_sets") if key in kwargs}
        if legacy:
            given = ", ".join(f"{key}={value!r}" for key, value in legacy.items())
            raise TypeError(
                f"{self.__class__.__name__} takes the bagging counts from its validation protocol: pass "
                f"validation_protocol=ValidationProtocol({given}) instead of {given}"
            )

    def _reject_in_fit_kwargs(self, method_kwargs: dict, *keys: str) -> None:
        """Assert none of ``keys`` are set in ``method_kwargs['fit_kwargs']``.

        Each of these has a dedicated constructor argument and must be passed there rather
        than nested in ``fit_kwargs``.
        """
        fit_kwargs = method_kwargs.get("fit_kwargs") or {}
        protocol_fields = {"adapt_num_bag_folds_to_n_classes": "adapt_num_folds_to_n_classes"}
        for key in keys:
            assert key not in fit_kwargs, (
                f"Set `{key}` directly in {self.__class__.__name__} rather than in `fit_kwargs`"
                if key in ("time_limit", "raise_on_model_failure")
                else f"`{key}` is owned by the validation protocol; pass "
                f"`validation_protocol=ValidationProtocol({protocol_fields.get(key, key)}=...)` to "
                f"{self.__class__.__name__} rather than `{key}` in `fit_kwargs`"
            )

    def _reject_bagging_structure_overrides(self, model_hyperparameters: dict) -> None:
        """Assert no ``ag_args_ensemble`` key rewrites the bagging structure the protocol owns."""
        ensemble_args = model_hyperparameters.get("ag_args_ensemble") or {}
        offending = [key for key in self._BAGGING_STRUCTURE_KEYS if key in ensemble_args]
        assert not offending, (
            f"ag_args_ensemble {offending} rewrite the bagging structure; the validation protocol owns it "
            f"(pass `validation_protocol=ValidationProtocol(...)` to {self.__class__.__name__} instead)"
        )

    def _insert_time_limit(self, model_hyperparameters: dict, time_limit: float | None) -> dict:
        """Return ``model_hyperparameters`` with the fit ``time_limit`` injected.

        For a bagged experiment (``_time_limit_on_bag``) the limit goes under
        ``ag_args_ensemble["ag.max_time_limit"]``, so the whole bag shares it; otherwise under
        top-level ``ag.max_time_limit``. Asserts the key isn't already set (it must be passed via the
        experiment's ``time_limit`` argument).
        """
        is_bag = self._time_limit_on_bag
        model_hyperparameters = copy.deepcopy(model_hyperparameters)
        if is_bag:
            if "ag_args_ensemble" in model_hyperparameters:
                assert "ag.max_time_limit" not in model_hyperparameters["ag_args_ensemble"], (
                    f"Set `time_limit` directly in {self.__class__.__name__} rather than in `ag_args_ensemble`"
                )
            else:
                model_hyperparameters["ag_args_ensemble"] = {}
            model_hyperparameters["ag_args_ensemble"]["ag.max_time_limit"] = time_limit
        else:
            assert "ag.max_time_limit" not in model_hyperparameters, (
                f"Set `time_limit` directly in {self.__class__.__name__} rather than in `model_hyperparameters`"
            )
            model_hyperparameters["ag.max_time_limit"] = time_limit
        return model_hyperparameters


class AGModelBagExperiment(AGModelExperiment):
    """Fit a single *bagged* AutoGluon model (fixes ``method_cls=AGSingleBagWrapper``).

    All models fit this way generate out-of-fold predictions on the entire training set and
    are compatible with ensemble simulations in TabArena. The bag fits ``num_bag_folds`` x
    ``num_bag_sets`` children, both taken from the experiment's validation protocol (see
    ``validation_protocol`` on ``Experiment``): an arena context stamps its official protocol at
    ``build_jobs`` / ``run_jobs``, or a caller passes one explicitly. This is the official flavour
    of the TabArena and BeyondArena leaderboards for models.

    Parameters
    ----------
    name: str
        The name of the experiment / method (descriptive + unique, e.g. ``"LightGBM_c1_BAG_L1"``).
    model_cls: type[AbstractModel]
        AutoGluon model class to fit.
    model_hyperparameters: dict
        AutoGluon model hyperparameters (see ``AGModelExperiment``).
    extra_model_hyperparameters: dict, optional
        Hyperparameters merged into ``model_hyperparameters`` (must not share keys with it).
    method_kwargs: dict, optional
        Extra kwargs for ``AGSingleBagWrapper(...)`` — see that class / ``AGModelExperiment``.
    **kwargs:
        Forwarded to ``AGModelExperiment.__init__`` (e.g. ``time_limit``,
        ``time_limit_with_preprocessing``, ``raise_on_model_failure``, ``experiment_kwargs``,
        ``preprocessing_pipeline``, ``validation_protocol``).
    """

    VALIDATION_FLAVOUR: ClassVar[ValidationFlavour | None] = "bagged"
    _method_cls = AGSingleBagWrapper
    _time_limit_on_bag: ClassVar[bool] = True

    def __init__(
        self,
        name: str,
        model_cls: type[AbstractModel],
        model_hyperparameters: dict,
        *,
        extra_model_hyperparameters: dict | None = None,
        method_kwargs: dict | None = None,
        **kwargs,
    ):
        method_kwargs = copy.deepcopy(method_kwargs) if method_kwargs else {}
        extra_model_hyperparameters = self._resolve_extra_model_hyperparameters(
            extra_model_hyperparameters, method_kwargs
        )
        self._warn_if_nested_model_hyperparameters(method_kwargs)
        model_hyperparameters = self._merge_model_hyperparameters(model_hyperparameters, extra_model_hyperparameters)

        super().__init__(
            name=name,
            model_cls=model_cls,
            model_hyperparameters=model_hyperparameters,
            method_kwargs=method_kwargs,
            **kwargs,
        )

    @staticmethod
    def _resolve_extra_model_hyperparameters(extra_model_hyperparameters: dict | None, method_kwargs: dict) -> dict:
        """Return the effective ``extra_model_hyperparameters``, popping any in ``method_kwargs``.

        The argument and the (legacy) ``method_kwargs["extra_model_hyperparameters"]`` location
        are mutually exclusive; defaults to an empty dict when neither is given.
        """
        if extra_model_hyperparameters is None:
            return method_kwargs.pop("extra_model_hyperparameters", {})
        assert "extra_model_hyperparameters" not in method_kwargs, (
            "Set only one of `extra_model_hyperparameters` and `method_kwargs['extra_model_hyperparameters']`"
        )
        return extra_model_hyperparameters

    def _warn_if_nested_model_hyperparameters(self, method_kwargs: dict) -> None:
        """Backward compat: ``model_hyperparameters`` used to be nestable in ``method_kwargs``.

        It is now a direct constructor argument. If a stale nested copy is present, warn that
        the direct ``model_hyperparameters`` argument overrides it, and drop the nested one.
        """
        if "model_hyperparameters" in method_kwargs:
            warnings.warn(
                f"`model_hyperparameters` was passed inside `method_kwargs` to {self.__class__.__name__}; "
                "this is deprecated and is overridden by the direct `model_hyperparameters` argument.",
                DeprecationWarning,
                stacklevel=2,
            )
            del method_kwargs["model_hyperparameters"]

    @staticmethod
    def _merge_model_hyperparameters(model_hyperparameters: dict, extra_model_hyperparameters: dict) -> dict:
        """Return a deep copy of ``model_hyperparameters`` with ``extra_*`` merged in.

        Asserts the two share no keys so an ``extra`` entry can never silently overwrite a
        primary hyperparameter.
        """
        overlapping_keys = set(extra_model_hyperparameters).intersection(model_hyperparameters)
        assert not overlapping_keys, (
            "extra_model_hyperparameters cannot have overlapping keys with model_hyperparameters. "
            f"Overlapping keys: {overlapping_keys}"
        )
        merged = copy.deepcopy(model_hyperparameters)
        merged.update(extra_model_hyperparameters)
        return merged


class AGModelOuterExperiment(Experiment):
    """Fit a single AutoGluon model on all data, with no train/val split.

    Passes all data as ``X, y`` into ``model_cls.fit`` (fixes ``method_cls=AGModelWrapper``).
    Useful to benchmark methods that don't fine-tune, such as TabPFNv2 / TabICL, which want
    to use all the data for training.

    Parameters
    ----------
    name: str
        The name of the experiment / method (descriptive + unique, e.g. ``"LightGBM_c1_BAG_L1"``).
    model_cls: type[AbstractModel]
        AutoGluon model class to fit.
    model_hyperparameters: dict
        AutoGluon model hyperparameters (stored as ``method_kwargs["hyperparameters"]``).
    preprocessing_pipeline: str | None, default None
        Named preprocessing pipeline forwarded to ``AGModelWrapper`` (``None`` / ``"default"``
        or ``"tabarena_default"``). Unlike the ``AGWrapper`` family, this is resolved *inside*
        the wrapper (``AbstractExecModel`` preprocessing), not via ``Experiment._apply_preprocessing``
        — so the experiment-level preprocessing hook stays a no-op for outer experiments.
    method_kwargs: dict, optional
        Extra kwargs for ``AGModelWrapper(...)`` — see that class for accepted keys (e.g.
        ``shuffle_features``, ``group_cols``).
    experiment_kwargs: dict, optional
        The kwargs passed to the runner (``experiment_cls``).
    **kwargs:
        Forwarded to ``Experiment.__init__`` (e.g. ``model_constraints``). An outer fit has no inner
        validation, so it runs outside the arena's validation protocol and its results are recorded
        as such (flavour ``outer``).
    """

    VALIDATION_FLAVOUR: ClassVar[ValidationFlavour | None] = "outer"
    _method_cls = AGModelWrapper
    _experiment_cls = OOFExperimentRunner
    _yaml_resolves_model_cls = True

    def __init__(
        self,
        name: str,
        model_cls: type[AbstractModel],
        model_hyperparameters: dict,
        *,
        preprocessing_pipeline: str | None = None,
        method_kwargs: dict | None = None,
        experiment_kwargs: dict | None = None,
        **kwargs,
    ):
        method_kwargs = copy.deepcopy(method_kwargs) if method_kwargs else {}
        super().__init__(
            name=name,
            method_cls=self._method_cls,
            method_kwargs={
                "model_cls": model_cls,
                "hyperparameters": model_hyperparameters,
                "preprocessing_pipeline": preprocessing_pipeline,
                **method_kwargs,
            },
            experiment_cls=self._experiment_cls,
            experiment_kwargs=experiment_kwargs,
            **kwargs,
        )


class ExternalSystemExperiment(Experiment):
    """Run a self-contained ML system (one that isn't a single AutoGluon model) on all the training
    data (no validation split).

    Parameters
    ----------
    name: str
        The name of the experiment / method (descriptive + unique).
    system_cls: type[AbstractExecModel]
        The system's exec-model class to fit (e.g. an ``ExternalSystemModel`` subclass).
    system_hyperparameters: dict, optional
        Keyword arguments for ``system_cls(...)`` (the per-config init kwargs). Merged under
        ``method_kwargs`` (which wins on key collisions).
    method_kwargs: dict, optional
        Extra ``system_cls`` kwargs shared across configs (e.g. ``shuffle_features``).
    experiment_kwargs: dict, optional
        The kwargs passed to the runner (``experiment_cls``).
    **kwargs:
        Forwarded to ``Experiment.__init__`` (e.g. ``model_constraints``, ``text_cache_mode``). A
        system owns its validation, so the arena's validation protocol is not applied to it; systems
        are an official flavour of the arena contexts and their results are recorded as ``system``.
    """

    VALIDATION_FLAVOUR: ClassVar[ValidationFlavour | None] = "system"
    _experiment_cls = OOFExperimentRunner

    def __init__(
        self,
        name: str,
        system_cls: type[AbstractExecModel],
        system_hyperparameters: dict | None = None,
        *,
        method_kwargs: dict | None = None,
        experiment_kwargs: dict | None = None,
        **kwargs,
    ):
        method_kwargs = copy.deepcopy(method_kwargs) if method_kwargs else {}
        super().__init__(
            name=name,
            method_cls=system_cls,
            method_kwargs={**(system_hyperparameters or {}), **method_kwargs},
            experiment_cls=self._experiment_cls,
            experiment_kwargs=experiment_kwargs,
            **kwargs,
        )

    @classmethod
    def from_yaml(cls, _context=None, **kwargs) -> Self:
        """Reconstruct from YAML, resolving ``system_cls`` from its import path.

        ``system_cls`` is serialized as an import path by the base ``_to_yaml_dict`` (it is a plain
        class, not an AutoGluon registry model), so it is resolved here rather than via ``method_cls``.
        """
        if _context is None:
            _context = globals()
        kwargs = _migrate_legacy_validation_kwargs(cls, kwargs)
        kwargs["system_cls"] = resolve_class(kwargs["system_cls"], context=_context)
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Class (de)serialization helpers (shared by the experiments + YAML serializers)
# ---------------------------------------------------------------------------
def class_to_path(cls: type) -> str:
    """Serialize a class to a stable, fully qualified import path.

    Example:
    -------
    autogluon.tabular.models.TabPFNv3preModel
    """
    return f"{cls.__module__}.{cls.__qualname__}"


def import_class(path: str) -> type:
    """Import a class from a fully qualified import path.

    Example:
    -------
    autogluon.tabular.models.TabPFNv3preModel
    """
    module_path, _, class_name = path.rpartition(".")
    if not module_path:
        raise ValueError(f"Expected fully qualified class path, got: {path!r}")

    module = importlib.import_module(module_path)

    obj: Any = module
    for part in class_name.split("."):
        obj = getattr(obj, part)

    if not inspect.isclass(obj):
        raise TypeError(f"Imported object is not a class: {path!r}")

    return obj


def resolve_class(
    value: str | type,
    *,
    context: dict | None = None,
    registry_resolver=None,
) -> type:
    """Resolve a class from, in priority order:

    1. an already-materialized class,
    2. ``context`` / globals (by name),
    3. a registry resolver, such as ``infer_model_cls``,
    4. a fully qualified import path.
    """
    if inspect.isclass(value):
        return value

    if not isinstance(value, str):
        raise TypeError(f"Expected class or string, got: {type(value)}")

    if context is not None and value in context:
        resolved = context[value]
        if inspect.isclass(resolved):
            return resolved

    if registry_resolver is not None:
        try:
            return registry_resolver(value)
        except Exception:  # noqa: S110
            pass

    return import_class(value)


def _eval_ag_args_fit_strings(model_hyperparameters: dict | None, context: dict) -> None:
    """Evaluate string expressions in ``model_hyperparameters["ag_args_fit"]`` in place.

    YAML stores some ``ag_args_fit`` values as expression strings (e.g. a class reference);
    each is ``eval``'d against ``context``, leaving non-evaluable strings (``NameError``)
    untouched. No-op when there is no ``ag_args_fit`` dict. Shared by the model-experiment
    ``from_yaml`` paths.
    """
    if not (isinstance(model_hyperparameters, dict) and isinstance(model_hyperparameters.get("ag_args_fit"), dict)):
        return
    for key, value in model_hyperparameters["ag_args_fit"].items():
        if isinstance(value, str):
            try:
                model_hyperparameters["ag_args_fit"][key] = eval(value, context)  # noqa: S307
            except NameError:
                pass  # If eval fails (e.g. unknown name), keep the original string value


def _migrate_legacy_validation_kwargs(cls: type[Experiment], kwargs: dict) -> dict:
    """Translate the pre-protocol YAML knobs of an experiment into ``validation_protocol`` (in place).

    Experiments serialized before the protocol object existed carried
    ``dynamic_tabarena_validation_protocol`` (task-specific inner splits), ``num_bag_folds`` /
    ``num_bag_sets`` (ints or ``"auto"``) and ``method_kwargs["fit_kwargs"]["adapt_num_bag_folds_to_n_classes"]``.
    They map onto one :class:`ValidationProtocol` named ``"legacy"``: explicit counts become the
    protocol's counts (no tiny-data regime); ``"auto"`` or absent counts under task-specific validation
    become the pre-protocol policy (8x1 with 5x5 at or below 500 training group instances), and 8x1
    otherwise. Flavours without inner validation only drop the knobs, as does a YAML that already names
    ``validation_protocol``. A YAML without any of the knobs is returned untouched.
    """
    dynamic = kwargs.pop("dynamic_tabarena_validation_protocol", None)
    num_bag_folds = kwargs.pop("num_bag_folds", None)
    num_bag_sets = kwargs.pop("num_bag_sets", None)
    fit_kwargs = (kwargs.get("method_kwargs") or {}).get("fit_kwargs")
    adapt = fit_kwargs.pop("adapt_num_bag_folds_to_n_classes", None) if isinstance(fit_kwargs, dict) else None
    if dynamic is None and num_bag_folds is None and num_bag_sets is None and adapt is None:
        return kwargs
    if kwargs.get("validation_protocol") is not None or cls.VALIDATION_FLAVOUR not in (
        "bagged",
        "holdout",
        "predictor",
    ):
        return kwargs

    task_specific = bool(dynamic)
    adapt = bool(adapt)
    if isinstance(num_bag_folds, int) and not isinstance(num_bag_folds, bool):
        protocol = ValidationProtocol(
            num_bag_folds=num_bag_folds,
            num_bag_sets=num_bag_sets if isinstance(num_bag_sets, int) and not isinstance(num_bag_sets, bool) else 1,
            task_specific_validation=task_specific,
            adapt_num_folds_to_n_classes=adapt,
            name="legacy",
        )
    elif task_specific:
        protocol = ValidationProtocol(
            tiny_num_bag_folds=5,
            tiny_num_bag_sets=5,
            tiny_max_group_instances=500,
            task_specific_validation=True,
            adapt_num_folds_to_n_classes=adapt,
            name="legacy",
        )
    else:
        protocol = ValidationProtocol(adapt_num_folds_to_n_classes=adapt, name="legacy")
    kwargs["validation_protocol"] = protocol
    return kwargs


class YamlSingleExperimentSerializer:
    """(De)serialize a single ``Experiment`` to/from YAML (a ``type`` + constructor-args dict)."""

    @classmethod
    def parse_method(cls, method_config: dict, context=None) -> Experiment:
        """Parse a method configuration dictionary and return an instance of the method class.
        This function evaluates the 'type' field in the method_config to determine the class to instantiate.
        It also evaluates any string values in the configuration that are meant to be Python expressions.
        """
        # Creating copy as we perform pop() which can lead to errors in subsequent calls
        method_config = method_config.copy()

        if context is None:
            context = globals()

        method_type_raw = method_config.pop("type")
        method_type = resolve_class(method_type_raw, context=context)

        return method_type.from_yaml(**method_config, _context=context)

    @classmethod
    def from_yaml(cls, path: str, context=None) -> Experiment:
        yaml_out = cls.load_yaml(path=path)
        return cls.parse_method(yaml_out, context=context)

    @classmethod
    def load_yaml(cls, path: str) -> dict:
        assert path.endswith(".yaml")
        with open(path) as file:
            return yaml.safe_load(file)

    @classmethod
    def to_yaml(cls, experiment: Experiment, path: str):
        assert path.endswith(".yaml")
        yaml_out = cls._to_yaml_format(experiment=experiment)
        with open(path, "w") as outfile:
            yaml.dump(yaml_out, outfile, default_flow_style=False)

    @classmethod
    def to_yaml_str(cls, experiment: Experiment) -> str:
        yaml_out = cls._to_yaml_format(experiment=experiment)
        return yaml.dump(yaml_out)

    @classmethod
    def _to_yaml_format(cls, experiment: Experiment) -> dict:
        return experiment.to_yaml_dict()


class YamlExperimentSerializer:
    """(De)serialize a list of ``Experiment`` objects to/from a YAML file's ``methods`` list."""

    @classmethod
    def from_yaml(cls, path: str, context=None, config_index: list[int] | None = None) -> list[Experiment]:
        """Load experiments from a YAML file.

        If `config_index` is given, only the experiments at those indices
        (into the file's `methods` list) are parsed and returned; otherwise all
        experiments are returned.
        """
        yaml_out = cls.load_yaml(path=path)

        experiments = []
        for m_i, experiment in enumerate(yaml_out):
            if (config_index is not None) and (m_i not in config_index):
                continue
            experiments.append(
                YamlSingleExperimentSerializer.parse_method(
                    experiment,
                    context=context,
                ),
            )

        return experiments

    @classmethod
    def from_yaml_str(cls, yaml_str: str, context=None) -> list[Experiment]:
        """Parse a YAML string containing multiple experiment definitions
        and return a list of Experiment instances.
        """
        yaml_out = yaml.safe_load(yaml_str)
        methods = yaml_out["methods"]

        experiments = []
        for experiment in methods:
            experiments.append(
                YamlSingleExperimentSerializer.parse_method(
                    experiment,
                    context=context,
                ),
            )

        return experiments

    @classmethod
    def load_yaml(cls, path: str) -> list[dict]:
        assert path.endswith(".yaml")

        with open(path) as file:
            yaml_out = yaml.safe_load(file)
        return yaml_out["methods"]

    @classmethod
    def to_yaml(cls, experiments: list[Experiment], path: str):
        assert path.endswith(".yaml")
        yaml_out = cls._to_yaml_format(experiments=experiments)
        with open(path, "w") as outfile:
            yaml.dump(yaml_out, outfile, default_flow_style=False)

    @classmethod
    def to_yaml_str(cls, experiments: list[Experiment]) -> str:
        yaml_out = cls._to_yaml_format(experiments=experiments)
        return yaml.dump(yaml_out)

    @classmethod
    def _to_yaml_format(cls, experiments: list[Experiment]) -> dict[str, list[dict]]:
        yaml_lst = []
        for experiment in experiments:
            yaml_lst.append(experiment.to_yaml_dict())
        return {"methods": yaml_lst}
