from __future__ import annotations

import copy
import inspect
from typing import Type

import yaml
from typing_extensions import Self

from autogluon.core.models import AbstractModel

from tabarena.benchmark.models.wrapper.abstract_class import AbstractExecModel
from tabarena.benchmark.models.wrapper.AutoGluon_class import AGWrapper, AGSingleWrapper, AGSingleBagWrapper
from tabarena.benchmark.models.wrapper.ag_model import AGModelWrapper
from tabarena.benchmark.experiment.experiment_runner import ExperimentRunner, OOFExperimentRunner
from tabarena.benchmark.models.model_registry import infer_model_cls
from tabarena.utils.cache import AbstractCacheFunction, CacheFunctionDummy
from tabarena.benchmark.task.openml import OpenMLTaskWrapper


class Experiment:
    """
    Experiment contains a method and the logic to run it on any task.
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

    """

    def __new__(cls, *args, **kwargs):
        # Logic executed before __init__
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        _args = copy.deepcopy(args)
        _kwargs = copy.deepcopy(kwargs)
        arg_names = [param.name for param in params.values() if param.name != 'self']
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
        assert "type" not in locals_new, f"The `type` key is reserved for the class name."
        locals_new = dict(
            type=self.__class__.__name__,
            **locals_new
        )
        return locals_new

    def to_yaml(self, path: str):
        assert path.endswith(".yaml")

        yaml_out = self.to_yaml_dict()
        with open(path, 'w') as outfile:
            yaml.dump(yaml_out, outfile, default_flow_style=False)

    def to_yaml_str(self) -> str:
        yaml_out = self.to_yaml_dict()
        return yaml.safe_dump(yaml_out, sort_keys=False, allow_unicode=True)

    def _to_yaml_dict(self, locals: dict) -> dict:
        locals_new = {}
        for k, v in locals.items():
            if inspect.isclass(v):
                v = v.__name__
            locals_new[k] = v
        return locals_new

    @classmethod
    def from_yaml(cls, method_cls, _context=None, **kwargs) -> Self:
        if _context is None:
            _context = globals()
        # Convert string class names to actual class references
        method_cls = eval(method_cls, _context)

        if "experiment_cls" in kwargs:
            kwargs["experiment_cls"] = eval(kwargs["experiment_cls"], _context)

        obj = cls(method_cls=method_cls, **kwargs)
        return obj

    def __init__(
        self,
        name: str,
        method_cls: Type[AbstractExecModel],
        method_kwargs: dict,
        *,
        experiment_cls: Type[ExperimentRunner] = OOFExperimentRunner,
        experiment_kwargs: dict = None,
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
        repeat: int = 0,
        sample: int = 0,
        cacher: AbstractCacheFunction | None = None,
        ignore_cache: bool = False,
        **experiment_kwargs,
    ) -> dict:
        if cacher is None:
            cacher = CacheFunctionDummy()
        if task is not None:
            out = cacher.cache(
                fun=self.experiment_cls.init_and_run,
                fun_kwargs=dict(
                    method_cls=self.method_cls,
                    task=task,
                    fold=fold,
                    repeat=repeat,
                    sample=sample,
                    task_name=task_name,
                    method=self.name,
                    fit_args=self.method_kwargs,
                    **self.experiment_kwargs,
                    **experiment_kwargs,
                ),
                ignore_cache=ignore_cache,
            )
        else:
            # load cache, no need to load task
            out = cacher.cache(fun=None, fun_kwargs=None, ignore_cache=ignore_cache)
        return out


class AGModelOuterExperiment(Experiment):
    """
    Simplified Experiment class
    for fitting a single model using AutoGluon without doing a train/val split,
    simply passing all data as X, y into `model_cls.fit`.

    This can be useful to benchmark methods that don't perform fine-tuning,
    such as TabPFNv2 and TabICL, where they instead want to use all the data for training.
    """
    def __init__(
        self,
        name: str,
        model_cls: Type[AbstractModel],
        model_hyperparameters: dict,
        *,
        method_kwargs: dict = None,
        experiment_kwargs: dict = None,
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
        model_cls = _context.get(model_cls, infer_model_cls(model_cls))

        # Evaluate all values in ag_args_fit
        if "model_hyperparameters" in kwargs:
            if "ag_args_fit" in kwargs["model_hyperparameters"]:
                for key, value in kwargs["model_hyperparameters"]["ag_args_fit"].items():
                    if isinstance(value, str):
                        try:
                            kwargs["model_hyperparameters"]["ag_args_fit"][key] = eval(value, _context)
                        except NameError:
                            pass  # If eval fails, keep the original string value
        obj = cls(model_cls=model_cls, **kwargs)
        return obj


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
            if k == "fit_kwargs":
                if "hyperparameters" in v and isinstance(v["hyperparameters"], dict):
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
        from tabarena.benchmark.models.model_registry import tabarena_model_registry
        tabarena_model_keys = tabarena_model_registry.keys

        if "experiment_cls" in kwargs:
            kwargs["experiment_cls"] = eval(kwargs["experiment_cls"], _context)
        if "fit_kwargs" in kwargs:
            if "hyperparameters" in kwargs["fit_kwargs"]:
                if isinstance("hyperparameters", dict):
                    hyperparameters = kwargs["fit_kwargs"]["hyperparameters"]
                    keys = list(hyperparameters.keys())
                    for model in keys:
                        if model in tabarena_model_keys:
                            val = kwargs["fit_kwargs"]["hyperparameters"].pop(model)
                            kwargs["fit_kwargs"]["hyperparameters"][tabarena_model_registry.key_to_cls(model)] = val
        obj = cls(**kwargs)
        return obj


# convenience wrapper
class AGModelExperiment(Experiment):
    """
    Simplified Experiment class specifically for fitting a single model using AutoGluon.
    The following arguments are fixed:
        method_cls = AGSingleWrapper
        experiment_cls = OOFExperimentRunner

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
        model_cls: Type[AbstractModel],
        model_hyperparameters: dict,
        *,
        time_limit: float | None = None,
        raise_on_model_failure: bool = True,
        method_kwargs: dict = None,
        experiment_kwargs: dict = None,
    ):
        if method_kwargs is None:
            method_kwargs = {}
        method_kwargs = copy.deepcopy(method_kwargs)
        if time_limit is not None:
            assert isinstance(time_limit, (float, int))
            assert time_limit > 0
        if "fit_kwargs" in method_kwargs:
            assert "time_limit" not in method_kwargs["fit_kwargs"], \
                f"Set `time_limit` directly in {self.__class__.__name__} rather than in `fit_kwargs`"
        assert isinstance(model_hyperparameters, dict)
        if time_limit is not None:
            model_hyperparameters = self._insert_time_limit(model_hyperparameters=model_hyperparameters, time_limit=time_limit, method_kwargs=method_kwargs)
        if "fit_kwargs" not in method_kwargs:
            method_kwargs["fit_kwargs"] = {}
        assert "raise_on_model_failure" not in method_kwargs["fit_kwargs"], \
            f"Set `raise_on_model_failure` directly in {self.__class__.__name__} rather than in `fit_kwargs`"
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
        )

    def _to_yaml_dict(self, locals: dict) -> dict:
        """
        Convert model_cls to ag_key, since ag_key is a unique identifier, whereas model_cls is not necessarily unique.
        """
        ag_key = locals["model_cls"].ag_key
        locals = copy.deepcopy(locals)
        locals["model_cls"] = ag_key
        return super()._to_yaml_dict(locals=locals)

    @classmethod
    def from_yaml(cls, model_cls, _context=None, **kwargs) -> Self:
        if _context is None:
            _context = globals()
        model_cls = _context.get(model_cls, infer_model_cls(model_cls))

        # Evaluate all values in ag_args_fit
        if "model_hyperparameters" in kwargs:
            if "ag_args_fit" in kwargs["model_hyperparameters"]:
                for key, value in kwargs["model_hyperparameters"]["ag_args_fit"].items():
                    if isinstance(value, str):
                        try:
                            kwargs["model_hyperparameters"]["ag_args_fit"][key] = eval(value, _context)
                        except NameError:
                            pass  # If eval fails, keep the original string value
        obj = cls(model_cls=model_cls, **kwargs)
        return obj

    def _insert_time_limit(self, model_hyperparameters: dict, time_limit: float | None, method_kwargs: dict) -> dict:
        is_bag = False
        if "fit_kwargs" in method_kwargs and "num_bag_folds" in method_kwargs["fit_kwargs"]:
            if method_kwargs["fit_kwargs"]["num_bag_folds"] > 1:
                is_bag = True
        model_hyperparameters = copy.deepcopy(model_hyperparameters)
        if is_bag:
            if "ag_args_ensemble" in model_hyperparameters:
                assert "ag.max_time_limit" not in model_hyperparameters["ag_args_ensemble"], \
                    f"Set `time_limit` directly in {self.__class__.__name__} rather than in `ag_args_ensemble`"
            else:
                model_hyperparameters["ag_args_ensemble"] = {}
            model_hyperparameters["ag_args_ensemble"]["ag.max_time_limit"] = time_limit
        else:
            assert "ag.max_time_limit" not in model_hyperparameters, \
                f"Set `time_limit` directly in {self.__class__.__name__} rather than in `model_hyperparameters`"
            model_hyperparameters["ag.max_time_limit"] = time_limit
        return model_hyperparameters


# convenience wrapper
class AGModelBagExperiment(AGModelExperiment):
    """
    Simplified Experiment class specifically for fitting a single bagged model using AutoGluon.
    The following arguments are fixed:
        method_cls = AGSingleWrapper
        experiment_cls = OOFExperimentRunner

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
    """
    _method_cls = AGSingleBagWrapper

    def __init__(
        self,
        name: str,
        model_cls: Type[AbstractModel],
        model_hyperparameters: dict,
        *,
        time_limit: float | None = None,
        num_bag_folds: int = 8,
        num_bag_sets: int = 1,
        raise_on_model_failure: bool = True,
        method_kwargs: dict = None,
        experiment_kwargs: dict = None,
    ):
        if method_kwargs is None:
            method_kwargs = {}
        method_kwargs = copy.deepcopy(method_kwargs)
        assert isinstance(num_bag_folds, int)
        assert isinstance(num_bag_sets, int)
        assert isinstance(method_kwargs, dict)
        assert num_bag_folds >= 2
        assert num_bag_sets >= 1
        if "fit_kwargs" in method_kwargs:
            assert "num_bag_folds" not in method_kwargs["fit_kwargs"], f"Set `num_bag_folds` directly in {self.__class__.__name__} rather than in `fit_kwargs`"
            assert "num_bag_sets" not in method_kwargs["fit_kwargs"], f"Set `num_bag_sets` directly in {self.__class__.__name__} rather than in `fit_kwargs`"
            method_kwargs["fit_kwargs"] = copy.deepcopy(method_kwargs["fit_kwargs"])
        else:
            method_kwargs["fit_kwargs"] = {}
        method_kwargs["fit_kwargs"]["num_bag_folds"] = num_bag_folds
        method_kwargs["fit_kwargs"]["num_bag_sets"] = num_bag_sets
        super().__init__(
            name=name,
            model_cls=model_cls,
            model_hyperparameters=model_hyperparameters,
            time_limit=time_limit,
            raise_on_model_failure=raise_on_model_failure,
            method_kwargs=method_kwargs,
            experiment_kwargs=experiment_kwargs,
        )


class YamlSingleExperimentSerializer:
    @classmethod
    def parse_method(cls, method_config: dict, context=None) -> Experiment:
        """
        Parse a method configuration dictionary and return an instance of the method class.
        This function evaluates the 'type' field in the method_config to determine the class to instantiate.
        It also evaluates any string values in the configuration that are meant to be Python expressions.
        """
        # Creating copy as we perform pop() which can lead to errors in subsequent calls
        method_config = method_config.copy()

        if context is None:
            context = globals()

        method_type = eval(method_config.pop('type'), context)
        method_obj = method_type.from_yaml(**method_config, _context=context)
        return method_obj

    @classmethod
    def from_yaml(cls, path: str, context=None) -> Experiment:
        yaml_out = cls.load_yaml(path=path)
        experiment = cls.parse_method(yaml_out, context=context)
        return experiment

    @classmethod
    def load_yaml(cls, path: str) -> dict:
        assert path.endswith(".yaml")
        with open(path, 'r') as file:
            yaml_out = yaml.safe_load(file)
        return yaml_out

    @classmethod
    def to_yaml(cls, experiment: Experiment, path: str):
        assert path.endswith(".yaml")
        yaml_out = cls._to_yaml_format(experiment=experiment)
        with open(path, 'w') as outfile:
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
    def from_yaml(cls, path: str, context=None) -> list[Experiment]:
        yaml_out = cls.load_yaml(path=path)

        experiments = []
        for experiment in yaml_out:
            experiments.append(YamlSingleExperimentSerializer.parse_method(experiment, context=context))

        return experiments

    @classmethod
    def load_yaml(cls, path: str) -> list[dict]:
        assert path.endswith(".yaml")

        with open(path, 'r') as file:
            yaml_out = yaml.safe_load(file)
        return yaml_out["methods"]

    @classmethod
    def to_yaml(cls, experiments: list[Experiment], path: str):
        assert path.endswith(".yaml")
        yaml_out = cls._to_yaml_format(experiments=experiments)
        with open(path, 'w') as outfile:
            yaml.dump(yaml_out, outfile, default_flow_style=False)

    @classmethod
    def to_yaml_str(cls, experiments: list[Experiment]) -> str:
        yaml_out = cls._to_yaml_format(experiments=experiments)
        return yaml.dump(yaml_out)

    @classmethod
    def _to_yaml_format(cls, experiments: list[Experiment]) -> dict[str, list[dict]]:
        yaml_lst = []
        for experiment in experiments:
            yaml_dict = experiment.to_yaml_dict()
            yaml_lst.append(yaml_dict)
        yaml_out = {"methods": yaml_lst}
        return yaml_out
