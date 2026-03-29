from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from itertools import product
from typing import ClassVar


@dataclass
class FSBenchConfig:
    """Parsed representation of an FSBench preprocessing pipeline config string.

    Config string format::

        FSBench__<fs_method>__<total_budget>__<budget_index>__<proxy_model>__<fs_time>

    ``fs_time`` is seconds when >= 1, a fraction of the total time limit when < 1.
    """

    fs_method: str
    total_budget: int
    budget_index: int
    proxy_model: str
    fs_time: int | float

    _PREFIX: ClassVar[str] = "FSBench"
    _SEP: ClassVar[str] = "__"

    def to_string(self) -> str:
        """Serialise to the config string used in benchmark pipelines."""
        parts = [self._PREFIX, self.fs_method, self.total_budget, self.budget_index, self.proxy_model, self.fs_time]
        return self._SEP.join(str(p) for p in parts)

    @classmethod
    def from_string(cls, s: str) -> FSBenchConfig:
        """Parse a config string into an :class:`FSBenchConfig`."""
        _, fs_method, total_budget, budget_index, proxy_model, raw_time = s.split(cls._SEP)
        fs_time: int | float = float(raw_time) if "." in raw_time else int(raw_time)
        return cls(
            fs_method=fs_method,
            total_budget=int(total_budget),
            budget_index=int(budget_index),
            proxy_model=proxy_model,
            fs_time=fs_time,
        )

    @classmethod
    def is_fs_bench_string(cls, s: str) -> bool:
        """Return True if ``s`` looks like an FSBench config string."""
        return s.startswith(cls._PREFIX + cls._SEP)


def get_num_selected_features(d: int, b: int, thr: int = 100) -> list[int]:
    """Return the list of feature counts to evaluate for a given dataset dimensionality and budget.

    Parameters
    ----------
    d : int
        Dataset feature dimensionality.
    b : int
        Budget — number of different feature counts to evaluate.
    thr : int
        Threshold separating the uniform (d <= thr) from the exponential (d > thr) regime.

    Returns
    -------
    list[int]
        List of b feature counts in increasing order.
    """
    if d <= thr:
        return [math.ceil(d * (i / (b + 1))) for i in range(1, b + 1)]
    return [math.ceil(math.pow(d, i / (b + 1))) for i in range(1, b + 1)]


def _get_budget_feature_count(d: int, *, b: int, idx: int) -> int:
    """Return the ``idx``-th feature count from the budget schedule for a dataset with ``d`` features."""
    return get_num_selected_features(d=d, b=b)[idx]


def get_fs_benchmark_preprocessing_pipelines(
    fs_methods: list[str],
    proxy_model_config: list[str],
    time_limit: list[int],
    total_budget: int,
    include_default: bool = True,
) -> list[str]:
    """Build the list of preprocessing pipeline config strings for the feature selection benchmark.

    Parameters
    ----------
    fs_methods : list[str]
        Feature selector method names.
    proxy_model_config : list[str]
        Proxy model identifiers (e.g. ``["lgbm"]``).
    time_limit : list[int]
        Time limits in seconds to include.
    total_budget : int
        Total number of budget indices (b in the feature-count formula).
    include_default : bool
        If True, append ``"default"`` as the last pipeline entry.

    Returns
    -------
    list[str]
        Ordered list of pipeline config strings.
    """
    pipelines = [
        FSBenchConfig(
            fs_method=fs_method,
            total_budget=total_budget,
            budget_index=budget_index,
            proxy_model=proxy_model,
            fs_time=fs_time,
        ).to_string()
        for fs_method, budget_index, proxy_model, fs_time in product(
            fs_methods, range(total_budget), proxy_model_config, time_limit
        )
    ]
    if include_default:
        pipelines.append("default")
    return pipelines


def apply_fs_bench_preprocessing(*, preprocessing_name: str, experiment):
    """Apply a FSBench preprocessing pipeline to a deep-copied experiment.

    Parses ``preprocessing_name`` into an :class:`FSBenchConfig`, attaches the
    corresponding feature selector (with a deferred ``max_features`` callable),
    and sets the preprocessing time limit on the experiment.

    Parameters
    ----------
    preprocessing_name : str
        Config string of the form
        ``FSBench__<fs_method>__<total_budget>__<budget_index>__<proxy_model>__<fs_time>``.
    experiment :
        A ``YamlSingleExperimentSerializer``-parsed experiment object (will be deep-copied).

    Returns
    -------
    experiment
        A new experiment object with the feature selection pipeline applied.
    """
    from copy import deepcopy

    from autogluon.features.generators.drop_duplicates import DropDuplicatesFeatureGenerator
    from autogluon.features.generators.drop_unique import DropUniqueFeatureGenerator
    from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import ProxyModelConfig
    from tabarena.benchmark.feature_selection_methods.feature_selection_methods_register import (
        FEATURE_SELECTION_METHODS_WITH_PROXY_MODEL,
        get_feature_selector_from_name,
    )

    config = FSBenchConfig.from_string(preprocessing_name)
    print(
        f"Using preprocessing pipeline: {config.to_string()}\n"
        f"  fs_method:   {config.fs_method}\n"
        f"  budget:      index {config.budget_index} of {config.total_budget} (max_features resolved at fit time)\n"
        f"  proxy_model: {config.proxy_model}\n"
        f"  fs_time:     {config.fs_time}"
    )

    proxy_mode_config = None
    if config.fs_method in FEATURE_SELECTION_METHODS_WITH_PROXY_MODEL:
        if config.proxy_model == "lgbm":
            proxy_mode_config = ProxyModelConfig(model_hyperparameters={})
        else:
            raise ValueError(
                f"Proxy model '{config.proxy_model}' not recognised for pipeline '{preprocessing_name}'."
            )

    max_features_fn = partial(_get_budget_feature_count, b=config.total_budget, idx=config.budget_index)
    selector_cls = get_feature_selector_from_name(name=config.fs_method)
    selector = selector_cls(max_features=max_features_fn, proxy_mode_config=proxy_mode_config)


    # TODO: refactor: make this its own model agnostic preprocessing class instead.
    # Inject feature selection config into experiment
    new_experiment = deepcopy(experiment)
    fit_kwargs = new_experiment.method_kwargs["fit_kwargs"]
    prep_pipeline = fit_kwargs.get("_feature_generator_kwargs", {})
    prep_pipeline["post_generators"] = prep_pipeline.get("post_generators", []) + [
        # Default post generators
        DropUniqueFeatureGenerator(),
        DropDuplicatesFeatureGenerator(post_drop_duplicates=False),
        # Selector Generator
        selector,
    ]
    prep_pipeline["post_drop_duplicates"] = False  # Not needed anymore.
    fit_kwargs["_feature_generator_kwargs"] = prep_pipeline
    fit_kwargs["time_limit_preprocessing"] = config.fs_time

    return new_experiment
