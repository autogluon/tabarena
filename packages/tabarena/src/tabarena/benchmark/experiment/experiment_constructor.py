from __future__ import annotations

import copy
import importlib
import inspect
import traceback
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import yaml

from tabarena.benchmark.exec_models.ag_single_model import AGModelWrapper
from tabarena.benchmark.exec_models.autogluon import AGSingleBagWrapper, AGSingleWrapper, AGWrapper
from tabarena.benchmark.exec_models.registry import infer_model_cls
from tabarena.benchmark.experiment.experiment_runner import ExperimentRunner, OOFExperimentRunner
from tabarena.utils.cache import AbstractCacheFunction, CacheFunctionDummy

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from autogluon.core.models import AbstractModel

    from tabarena.benchmark.exec_models.base import AbstractExecModel
    from tabarena.benchmark.task.metadata import GroupLabelTypes, SplitTimeHorizonTypes, SplitTimeHorizonUnitTypes
    from tabarena.benchmark.task.openml import OpenMLTaskWrapper


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
        Resolved lazily at run time via ``_resolve_preprocessing``; ``None`` applies none.
    dynamic_tabarena_validation_protocol: bool, default False
        If True, this experiment's validation split is configured dynamically from the
        task (type and dataset metadata) at run time, via ``prepare_for_task``. Only
        supported for bagged AutoGluon experiments.

    """

    def __new__(cls, *args, **kwargs):
        # Logic executed before __init__
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

    def to_yaml_dict(self) -> dict:
        locals = self._locals
        locals_new = self._to_yaml_dict(locals=locals)
        assert "type" not in locals_new, "The `type` key is reserved for the class name."
        return dict(
            type=class_to_path(self.__class__),
            **locals_new,
        )

    def to_yaml(self, path: str):
        assert path.endswith(".yaml")

        yaml_out = self.to_yaml_dict()
        with open(path, "w") as outfile:
            yaml.dump(yaml_out, outfile, default_flow_style=False)

    def to_yaml_str(self) -> str:
        yaml_out = self.to_yaml_dict()
        return yaml.safe_dump(yaml_out, sort_keys=False, allow_unicode=True)

    def _to_yaml_dict(self, locals: dict) -> dict:
        locals_new = {}
        for k, v in locals.items():
            if inspect.isclass(v):
                v = class_to_path(v)
            locals_new[k] = v
        return locals_new

    @classmethod
    def from_yaml(cls, method_cls, _context=None, **kwargs) -> Self:
        if _context is None:
            _context = globals()

        method_cls = resolve_class(method_cls, context=_context)

        if "experiment_cls" in kwargs:
            kwargs["experiment_cls"] = resolve_class(kwargs["experiment_cls"], context=_context)

        return cls(method_cls=method_cls, **kwargs)

    def __init__(
        self,
        name: str,
        method_cls: type[AbstractExecModel],
        method_kwargs: dict,
        *,
        experiment_cls: type[ExperimentRunner] = OOFExperimentRunner,
        experiment_kwargs: dict | None = None,
        preprocessing_pipeline: str | None = None,
        dynamic_tabarena_validation_protocol: bool = False,
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
        # Name of an optional preprocessing pipeline to apply, resolved lazily at
        # run time via `_resolve_preprocessing` (see that method).
        self.preprocessing_pipeline = preprocessing_pipeline
        # Whether `run` should adapt this experiment's validation data dynamically
        # based on the task it runs on (task-dependent, so applied at run time rather
        # than baked into method_kwargs).
        self.dynamic_tabarena_validation_protocol = dynamic_tabarena_validation_protocol

    def construct_method(self, problem_type: str, eval_metric) -> AbstractExecModel:
        return self.method_cls(
            problem_type=problem_type,
            eval_metric=eval_metric,
            **self.method_kwargs,
        )

    def run(
        self,
        task: OpenMLTaskWrapper | None,
        fold: int,
        task_name: str,
        *,
        cache_task_key: int | str,
        repeat: int = 0,
        sample: int = 0,
        cacher: AbstractCacheFunction | None = None,
        ignore_cache: bool = False,
        raise_on_failure: bool = True,
        verbose: bool = True,
        **experiment_kwargs,
    ) -> dict | None:
        """Fit this experiment on a single (task, fold, repeat) and return its results.

        Single entry point for executing one experiment job end to end, so callers (e.g.
        ``run_experiments_new``) only hand over the task and cache and read back a result.
        It splits into two flows:

        - **Load flow** (``task is None``): load and return the cached ``results`` directly.
          No task configuration, preprocessing resolution, or fit-failure guard is needed —
          reached e.g. on a default-mode cache hit, where the caller passes no task.
        - **Fit flow** (``task`` provided):
            1. Configure the task-dependent validation protocol and open the task's
               text-embedding cache scope for the fit (``prepare_for_task``).
            2. Resolve any preprocessing pipeline (``_resolve_preprocessing``).
            3. Fit via ``experiment_cls.init_and_run`` through ``cacher`` — which still
               short-circuits to the cached ``results`` on a hit instead of refitting.
            4. Guard against a non-finite final metric error (``_enforce_finite_metric``).

        Fit-flow failures are handled here: when ``raise_on_failure`` is False, a fit
        exception (or a non-finite-metric failure) is swallowed and ``None`` is returned;
        otherwise it propagates.

        Parameters
        ----------
        task: OpenMLTaskWrapper | None
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
            task's text-embedding cache in ``prepare_for_task``.
        cacher: AbstractCacheFunction | None, default None
            Cacher for this job's ``results`` artifact. Defaults to a no-op in-memory cacher.
        ignore_cache: bool, default False
            If True, refit and overwrite the cache even when a cached result exists.
        raise_on_failure: bool, default True
            If True, fit exceptions and non-finite-metric failures propagate; if False, they
            are swallowed and ``None`` is returned.
        verbose: bool, default True
            Verbosity for the default cacher used when ``cacher`` is None.
        **experiment_kwargs
            Extra kwargs forwarded to ``experiment_cls.init_and_run`` (e.g. ``debug_mode``,
            ``eval_metric_name``).

        Returns:
        -------
        dict | None
            The experiment results, or ``None`` when the job failed (and was not raised).
        """
        if cacher is None:
            cacher = CacheFunctionDummy(verbose=verbose)

        # Load flow: with no task to fit, load the cached result directly
        if task is None:
            return cacher.cache(fun=None, fun_kwargs=None, ignore_cache=ignore_cache)

        # Fit flow: configure for the task (dynamic validation protocol) and obtain the
        # cache scope wrapping the fit. Eager config happens here; a null scope is returned
        # when not applicable.
        task_cache_cm = self.prepare_for_task(task=task, cache_task_key=cache_task_key)
        try:
            with task_cache_cm:
                # Resolve the preprocessing pipeline (if any) into a ready-to-run experiment.
                resolved = self._resolve_preprocessing()
                out = cacher.cache(
                    fun=resolved.experiment_cls.init_and_run,
                    fun_kwargs=dict(
                        method_cls=resolved.method_cls,
                        task=task,
                        fold=fold,
                        repeat=repeat,
                        sample=sample,
                        task_name=task_name,
                        method=resolved.name,
                        fit_args=self._autodetect_resources(resolved.method_kwargs),
                        **resolved.experiment_kwargs,
                        **experiment_kwargs,
                    ),
                    ignore_cache=ignore_cache,
                )
                self._enforce_finite_metric(out=out, cacher=cacher)
        except Exception:
            if raise_on_failure:
                raise
            print(f"Experiment {self.name!r} failed on {task_name} (fold={fold}, repeat={repeat}):")
            traceback.print_exc()
            return None

        return out

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

    def _resolve_preprocessing(self) -> Experiment:
        """Resolve `self.preprocessing_pipeline` into a ready-to-run Experiment.

        Returns `self` when there is nothing to resolve; otherwise returns a
        deep-copied Experiment with the pipeline applied and
        `preprocessing_pipeline` cleared.
        """
        pipeline = self.preprocessing_pipeline
        if pipeline is None or pipeline == "default":
            return self

        if pipeline == "tabarena_default":
            from tabarena.benchmark.preprocessing import (
                TabArenaModelAgnosticPreprocessing,
                TabArenaModelSpecificPreprocessing,
            )

            resolved = copy.deepcopy(self)
            resolved.preprocessing_pipeline = None
            resolved.method_kwargs["fit_kwargs"]["feature_generator_cls"] = TabArenaModelAgnosticPreprocessing
            resolved.method_kwargs["fit_kwargs"]["feature_generator_kwargs"] = {}
            resolved.method_kwargs["model_hyperparameters"] = TabArenaModelSpecificPreprocessing.add_to_hyperparameters(
                resolved.method_kwargs["model_hyperparameters"],
            )
            return resolved

        if pipeline.startswith("FSBench__"):
            # Logic for the (experimental) feature selection benchmark.
            from tabarena.benchmark.feature_selection_methods.feature_selection_benchmark_utils import (
                apply_fs_bench_preprocessing,
            )

            resolved = apply_fs_bench_preprocessing(preprocessing_name=pipeline, experiment=self)
            resolved.preprocessing_pipeline = None
            return resolved

        raise ValueError(f"Preprocessing pipeline name '{pipeline}' not recognized.")

    def set_resources(
        self,
        *,
        num_cpus: int | None = None,
        num_gpus: int = 0,
        memory_limit: int | None = None,
    ) -> None:
        """Bake compute resources into this experiment's ``fit_kwargs``.

        Useful for attaching resources to a manually-built Experiment (the
        benchmark builder does the equivalent at construction time). ``num_cpus``
        and ``memory_limit`` may be left ``None`` to request run-time
        auto-detection of the node's resources (see ``_autodetect_resources``).
        """
        fit_kwargs = self.method_kwargs.setdefault("fit_kwargs", {})
        fit_kwargs["num_cpus"] = num_cpus
        fit_kwargs["num_gpus"] = num_gpus
        fit_kwargs["memory_limit"] = memory_limit

    @staticmethod
    def _autodetect_resources(method_kwargs: dict) -> dict:
        """Return ``method_kwargs`` with any ``None`` compute resources auto-detected.

        A serialized experiment can carry ``num_cpus=None`` / ``memory_limit=None``
        to mean "detect on whatever node I run on", preserving per-node
        auto-detection while keeping the experiment self-contained. Returns the
        input unchanged when there is nothing to detect (no copy); otherwise a
        deep-copied ``method_kwargs`` with the detected values filled in.
        """
        fit_kwargs = method_kwargs.get("fit_kwargs") or {}
        detect_cpus = fit_kwargs.get("num_cpus", 0) is None
        detect_memory = fit_kwargs.get("memory_limit", 0) is None
        if not (detect_cpus or detect_memory):
            return method_kwargs

        from tabarena.utils.resources import detect_memory_limit_gb, detect_num_cpus

        method_kwargs = copy.deepcopy(method_kwargs)
        fit_kwargs = method_kwargs["fit_kwargs"]
        if detect_cpus:
            fit_kwargs["num_cpus"] = detect_num_cpus()
            print(f"num_cpus not provided, using detected number of CPUs: {fit_kwargs['num_cpus']}")
        if detect_memory:
            fit_kwargs["memory_limit"] = detect_memory_limit_gb()
            print(f"memory_limit not provided, using detected memory size: {fit_kwargs['memory_limit']} GB")
        return method_kwargs

    def load_validation_split_metadata(
        self,
        *,
        use_task_specific_validation: bool,
        target_name: str | None = None,
        stratify_on: str | None = None,
        group_on: str | list[str] | None = None,
        time_on: str | None = None,
        group_time_on: str | None = None,
        group_labels: GroupLabelTypes | None = None,
        split_time_horizon: SplitTimeHorizonTypes | None = None,
        split_time_horizon_unit: SplitTimeHorizonUnitTypes | None = None,
        overwrite_existing: bool = False,
    ) -> None:
        """Load validation split metadata into the experiment's method_kwargs.

        Parameter
        ---------
        use_task_specific_validation: bool
            If True, we will adapt the validation protocol of the experiment
            based on the metadat from the task.
        target_name: str, optional
            The name of the target column in the dataset.
        stratify_on: str, optional
            The name of the column to stratify on when creating validation splits.
        group_on: str or list of str, optional
            The name(s) of the column(s) to group on when creating validation splits.
        time_on: str, optional
            The name of the column to use for time-based validation splits.
        group_time_on: str, optional
            The name of the column to use for group time-based validation splits.
        group_labels:
            Whether the group_on column(s) contain labels for each sample, or for each group.
        split_time_horizon:
            The time horizon of the test data.
        split_time_horizon_unit:
            The unit for the time horizon.
        overwrite_existing: bool, default False
            If True, will overwrite existing validation split metadata in method_kwargs.
        """
        print(
            "Loading validation split metadata into experiment:"
            f"\n\tUse task specific validation: {use_task_specific_validation}"
            f"\n\ttarget_name: {target_name}"
            f"\n\tstratify_on: {stratify_on}"
            f"\n\ttime_on: {time_on}"
            f"\n\tsplit_time_horizon: {split_time_horizon}"
            f"\n\tsplit_time_horizon_unit: {split_time_horizon_unit}"
            f"\n\tgroup_on: {group_on}"
            f"\n\tgroup_time_on: {group_time_on}"
            f"\n\tgroup_labels: {group_labels}",
        )
        params = {
            "use_task_specific_validation": use_task_specific_validation,
            "target_name": target_name,
            "stratify_on": stratify_on,
            "group_on": group_on,
            "time_on": time_on,
            "group_time_on": group_time_on,
            "group_labels": group_labels,
            "split_time_horizon": split_time_horizon,
            "split_time_horizon_unit": split_time_horizon_unit,
        }

        for key, value in params.items():
            if (not overwrite_existing) and (key in self.method_kwargs):
                print(
                    f"{key} already exists, using existing value: \n\t{self.method_kwargs[key]}",
                )
            else:
                self.method_kwargs[key] = value

    def prepare_for_task(
        self,
        *,
        task: OpenMLTaskWrapper,
        cache_task_key: int | str,
    ) -> AbstractContextManager:
        """Configure this experiment for ``task`` and return its task-specific cache scope.

        Bundles the (experimental) ``dynamic_tabarena_validation_protocol`` setup into a
        single call: it adapts this experiment's validation-split metadata to the task
        (eagerly, when called) and returns a context manager that loads the task's
        semantic-text embedding cache for the duration of the fit, restoring the prior
        state on exit. When the protocol is disabled, configuration is skipped and a null
        context is returned. This is part of the fit flow, so a task object is required.

        Configuration runs eagerly here (rather than on ``__enter__``) so a misconfigured
        task/experiment surfaces immediately at the call site; only the returned scope is
        meant to wrap the ``run(...)`` call.

        Parameters
        ----------
        task: OpenMLTaskWrapper
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

        if not self.dynamic_tabarena_validation_protocol:
            return nullcontext()

        from tabarena.benchmark.task.openml import TabArenaOpenMLSupervisedTask

        if not isinstance(task.task, TabArenaOpenMLSupervisedTask):
            raise ValueError(
                "`dynamic_tabarena_validation_protocol` is only implemented for `TabArenaOpenMLSupervisedTask`!",
            )

        if not isinstance(self, AGModelBagExperiment):
            # TODO: add support
            raise NotImplementedError(
                f"Validation split kwargs only implemented for AGModelBagExperiment for now, got {type(self)}",
            )

        # Add info about group and time for the pipeline to handle.
        self.load_validation_split_metadata(
            use_task_specific_validation=True,
            **task.get_validation_split_kwargs(),
        )

        # Load this task's semantic-text embedding cache for the duration of the fit
        # (slug-keyed + encoder-versioned; restored afterwards). Default ``require``: a
        # text task with no cache fails fast — warm it first via the prefetch/download
        # path or pre-generation.
        from tabarena.benchmark.preprocessing.text_cache import use_text_cache_for_task

        return use_text_cache_for_task(
            cache_task_key,
            has_text=task._has_text,
            mode="require",
        )


class AGModelOuterExperiment(Experiment):
    """Simplified Experiment class
    for fitting a single model using AutoGluon without doing a train/val split,
    simply passing all data as X, y into `model_cls.fit`.

    This can be useful to benchmark methods that don't perform fine-tuning,
    such as TabPFNv2 and TabICL, where they instead want to use all the data for training.
    """

    def __init__(
        self,
        name: str,
        model_cls: type[AbstractModel],
        model_hyperparameters: dict,
        *,
        method_kwargs: dict | None = None,
        experiment_kwargs: dict | None = None,
    ):
        if method_kwargs is None:
            method_kwargs = {}
        super().__init__(
            name=name,
            method_cls=AGModelWrapper,
            method_kwargs={
                "model_cls": model_cls,
                "hyperparameters": model_hyperparameters,
                **method_kwargs,
            },
            experiment_cls=OOFExperimentRunner,
            experiment_kwargs=experiment_kwargs,
        )

    @classmethod
    def from_yaml(cls, model_cls, _context=None, **kwargs) -> Self:
        if _context is None:
            _context = globals()
        model_cls = resolve_class(
            model_cls,
            context=_context,
            registry_resolver=infer_model_cls,
        )

        # Evaluate all values in ag_args_fit
        if "model_hyperparameters" in kwargs and "ag_args_fit" in kwargs["model_hyperparameters"]:
            for key, value in kwargs["model_hyperparameters"]["ag_args_fit"].items():
                if isinstance(value, str):
                    try:
                        kwargs["model_hyperparameters"]["ag_args_fit"][key] = eval(value, _context)  # noqa: S307
                    except NameError:
                        pass  # If eval fails, keep the original string value
        return cls(model_cls=model_cls, **kwargs)


class AGExperiment(Experiment):
    _method_cls = AGWrapper

    def __init__(
        self,
        name: str,
        init_kwargs: dict | None = None,
        fit_kwargs: dict | None = None,
        method_kwargs: dict | None = None,
        experiment_kwargs: dict | None = None,
    ):
        _experiment_kwargs = {"compute_simulation_artifacts": False}
        if experiment_kwargs is None:
            experiment_kwargs = {}
        if method_kwargs is None:
            method_kwargs = {}
        experiment_kwargs = copy.deepcopy(experiment_kwargs)
        method_kwargs = copy.deepcopy(method_kwargs)
        _experiment_kwargs.update(experiment_kwargs)
        assert "fit_kwargs" not in method_kwargs
        assert "init_kwargs" not in method_kwargs
        if init_kwargs is not None:
            method_kwargs["init_kwargs"] = init_kwargs
        if fit_kwargs is not None:
            method_kwargs["fit_kwargs"] = fit_kwargs
        super().__init__(
            name=name,
            method_cls=self._method_cls,
            method_kwargs=method_kwargs,
            experiment_cls=OOFExperimentRunner,
            experiment_kwargs=_experiment_kwargs,
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


# convenience wrapper
class AGModelExperiment(Experiment):
    """Simplified Experiment class specifically for fitting a single model using AutoGluon.
    The following arguments are fixed:
        method_cls = AGSingleWrapper
        experiment_cls = OOFExperimentRunner.

    Parameters
    ----------
    name: str
        The name of the experiment / method.
        Should be descriptive and unique compared to other methods.
        For example, `"LightGBM_c1_BAG_L1"`
    model_cls: Type[AbstractModel]
        AutoGluon model class to fit
    model_hyperparameters: dict
        AutoGluon model hyperparameters
        Identical to what you would pass to `TabularPredictor.fit(..., hyperparameters={model_cls: [model_hyperparameters]})
    time_limit: float, optional
        The time limit in seconds the model is allowed to fit for.
        If unspecified, no time limit is enforced.
    raise_on_model_failure: bool, default True
        By default sets raise_on_model_failure to True
        so that any AutoGluon model failure will be raised in a debugger friendly manner.
    method_kwargs: dict, optional
        The kwargs passed to the init of `method_cls`.
    experiment_kwargs: dict, optional
        The kwargs passed to the init of `experiment_cls`.
    """

    _method_cls = AGSingleWrapper
    _experiment_cls = OOFExperimentRunner

    def __init__(
        self,
        name: str,
        model_cls: type[AbstractModel],
        model_hyperparameters: dict,
        *,
        time_limit: float | None = None,
        raise_on_model_failure: bool = True,
        method_kwargs: dict | None = None,
        experiment_kwargs: dict | None = None,
        time_limit_with_preprocessing: bool = False,
        preprocessing_pipeline: str | None = None,
        dynamic_tabarena_validation_protocol: bool = False,
    ):
        if method_kwargs is None:
            method_kwargs = {}
        method_kwargs = copy.deepcopy(method_kwargs)
        if time_limit is not None:
            assert isinstance(time_limit, (float, int))
            assert time_limit > 0
        if "fit_kwargs" in method_kwargs:
            assert "time_limit" not in method_kwargs["fit_kwargs"], (
                f"Set `time_limit` directly in {self.__class__.__name__} rather than in `fit_kwargs`"
            )
        assert isinstance(model_hyperparameters, dict)
        if "fit_kwargs" not in method_kwargs:
            method_kwargs["fit_kwargs"] = {}
        if time_limit is not None:
            if time_limit_with_preprocessing:
                method_kwargs["fit_kwargs"]["time_limit"] = time_limit
            else:
                model_hyperparameters = self._insert_time_limit(
                    model_hyperparameters=model_hyperparameters, time_limit=time_limit, method_kwargs=method_kwargs
                )
        assert "raise_on_model_failure" not in method_kwargs["fit_kwargs"], (
            f"Set `raise_on_model_failure` directly in {self.__class__.__name__} rather than in `fit_kwargs`"
        )
        method_kwargs["fit_kwargs"]["raise_on_model_failure"] = raise_on_model_failure
        super().__init__(
            name=name,
            method_cls=self._method_cls,
            method_kwargs={
                "model_cls": model_cls,
                "model_hyperparameters": model_hyperparameters,
                **method_kwargs,
            },
            experiment_cls=self._experiment_cls,
            experiment_kwargs=experiment_kwargs,
            preprocessing_pipeline=preprocessing_pipeline,
            dynamic_tabarena_validation_protocol=dynamic_tabarena_validation_protocol,
        )

    def _to_yaml_dict(self, locals: dict) -> dict:
        """Serialize model_cls as an import path so custom/unregistered classes can
        be loaded without requiring TabArena registry registration.
        """
        locals = copy.deepcopy(locals)
        locals["model_cls"] = class_to_path(locals["model_cls"])
        return super()._to_yaml_dict(locals=locals)

    @classmethod
    def from_yaml(cls, model_cls, _context=None, **kwargs) -> Self:
        if _context is None:
            _context = globals()
        model_cls = resolve_class(
            model_cls,
            context=_context,
            registry_resolver=infer_model_cls,
        )

        # Evaluate all values in ag_args_fit
        if "model_hyperparameters" in kwargs and "ag_args_fit" in kwargs["model_hyperparameters"]:
            for key, value in kwargs["model_hyperparameters"]["ag_args_fit"].items():
                if isinstance(value, str):
                    try:
                        kwargs["model_hyperparameters"]["ag_args_fit"][key] = eval(value, _context)  # noqa: S307
                    except NameError:
                        pass  # If eval fails, keep the original string value
        return cls(model_cls=model_cls, **kwargs)

    def _insert_time_limit(self, model_hyperparameters: dict, time_limit: float | None, method_kwargs: dict) -> dict:
        is_bag = False
        if "fit_kwargs" in method_kwargs and "num_bag_folds" in method_kwargs["fit_kwargs"]:
            if method_kwargs["fit_kwargs"]["num_bag_folds"] > 1:
                is_bag = True
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


# convenience wrapper
class AGModelBagExperiment(AGModelExperiment):
    """Simplified Experiment class specifically for fitting a single bagged model using AutoGluon.
    The following arguments are fixed:
        method_cls = AGSingleWrapper
        experiment_cls = OOFExperimentRunner.

    All models fit this way will generate out-of-fold predictions on the entire training set,
    and will be compatible with ensemble simulations in TabArena.

    Will fit the model with `num_bag_folds` folds and `num_bag_sets` sets (aka repeats).
    In total will fit `num_bag_folds * num_bag_sets` models in the bag.

    Parameters
    ----------
    name: str
        The name of the experiment / method.
        Should be descriptive and unique compared to other methods.
        For example, `"LightGBM_c1_BAG_L1"`
    model_cls: Type[AbstractModel]
    model_hyperparameters: dict
        Identical to what you would pass to `TabularPredictor.fit(..., hyperparameters={model_cls: [model_hyperparameters]})
    time_limit: float, optional
    num_bag_folds: int, default 8
    num_bag_sets: int, default 1
    method_kwargs: dict, optional
    experiment_kwargs: dict, optional
    time_limit_with_preprocessing: bool, default False
            If True, time limit also captures the time it takes for preprocessing.
    """

    _method_cls = AGSingleBagWrapper

    def __init__(
        self,
        name: str,
        model_cls: type[AbstractModel],
        model_hyperparameters: dict,
        *,
        time_limit: float | None = None,
        num_bag_folds: int = 8,
        num_bag_sets: int = 1,
        raise_on_model_failure: bool = True,
        method_kwargs: dict | None = None,
        experiment_kwargs: dict | None = None,
        time_limit_with_preprocessing: bool = False,
        extra_model_hyperparameters: dict | None = None,
        preprocessing_pipeline: str | None = None,
        dynamic_tabarena_validation_protocol: bool = False,
    ):
        if method_kwargs is None:
            method_kwargs = {}
        method_kwargs = copy.deepcopy(method_kwargs)
        if extra_model_hyperparameters is None:
            if "extra_model_hyperparameters" in method_kwargs:
                extra_model_hyperparameters = method_kwargs["extra_model_hyperparameters"]
                del method_kwargs["extra_model_hyperparameters"]
            else:
                extra_model_hyperparameters = {}
        else:
            assert "extra_model_hyperparameters" not in method_kwargs, (
                "Set only one of `extra_model_hyperparameters` and `method_kwargs['extra_model_hyperparameters']`"
            )
        assert isinstance(num_bag_folds, int)
        assert isinstance(num_bag_sets, int)
        assert isinstance(method_kwargs, dict)
        assert num_bag_folds >= 2
        assert num_bag_sets >= 1
        if "fit_kwargs" in method_kwargs:
            assert "num_bag_folds" not in method_kwargs["fit_kwargs"], (
                f"Set `num_bag_folds` directly in {self.__class__.__name__} rather than in `fit_kwargs`"
            )
            assert "num_bag_sets" not in method_kwargs["fit_kwargs"], (
                f"Set `num_bag_sets` directly in {self.__class__.__name__} rather than in `fit_kwargs`"
            )
            method_kwargs["fit_kwargs"] = copy.deepcopy(method_kwargs["fit_kwargs"])
        else:
            method_kwargs["fit_kwargs"] = {}
        method_kwargs["fit_kwargs"]["num_bag_folds"] = num_bag_folds
        method_kwargs["fit_kwargs"]["num_bag_sets"] = num_bag_sets
        if "model_hyperparameters" in method_kwargs:
            assert "model_hyperparameters" in method_kwargs["model_hyperparameters"], (
                "model_hyperparameters should be passed directly to AGModelBagExperiment rather than in method_kwargs"
            )
            model_hyperparameters = copy.deepcopy(model_hyperparameters)
            del method_kwargs["model_hyperparameters"]
        if extra_model_hyperparameters is not None:
            assert isinstance(extra_model_hyperparameters, dict)
            # check no key overlap!
            overlapping_keys = set(extra_model_hyperparameters.keys()).intersection(set(model_hyperparameters.keys()))
            assert not overlapping_keys, (
                f"extra_model_hyperparameters cannot have overlapping keys with model_hyperparameters. "
                f"Overlapping keys: {overlapping_keys}"
            )
            model_hyperparameters = copy.deepcopy(model_hyperparameters)
            model_hyperparameters.update(extra_model_hyperparameters)

        super().__init__(
            name=name,
            model_cls=model_cls,
            model_hyperparameters=model_hyperparameters,
            time_limit=time_limit,
            raise_on_model_failure=raise_on_model_failure,
            method_kwargs=method_kwargs,
            experiment_kwargs=experiment_kwargs,
            time_limit_with_preprocessing=time_limit_with_preprocessing,
            preprocessing_pipeline=preprocessing_pipeline,
            dynamic_tabarena_validation_protocol=dynamic_tabarena_validation_protocol,
        )


class YamlSingleExperimentSerializer:
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


def class_to_path(cls: type) -> str:
    """Serialize a class to a stable import path.

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
    """Resolve a class from:
    1. an already-materialized class
    2. context/globals
    3. registry resolver, such as infer_model_cls
    4. fully qualified import path.
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
