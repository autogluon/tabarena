"""Generate benchmark experiments from a model's hyperparameter search space.

A *config generator* pairs a model class with a way to enumerate hyperparameter
configurations — a fixed list of ``manual_configs`` plus ``N`` randomly sampled ones:

* :class:`ConfigGenerator` — samples a ConfigSpace ``search_space``.
* :class:`CustomAGConfigGenerator` — delegates sampling to a ``search_space_func``.

Its ``generate_all_*_experiments`` methods turn those configs into ready-to-run
:class:`~tabarena.benchmark.experiment.experiment_constructor.Experiment` objects of a given
flavour, which differ only in how the model is validated:

* **bag** — :class:`~...AGModelBagExperiment`: cross-validated bagging (the TabArena default).
* **outer** — :class:`~...AGModelOuterExperiment`: no validation, train on all the data.

Both flavours share the per-config naming (``{ag_name}{name_suffix}{bag_suffix}``) and iteration via
:func:`_build_experiments`; each ``config`` carries its display suffix under ``ag_args`` and any
validation/bagging settings under ``ag_args_ensemble`` (see the config-dict helpers below).
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Literal

from autogluon.core.searcher.local_random_searcher import LocalRandomSearcher

from tabarena.benchmark.experiment import AGModelBagExperiment, AGModelOuterExperiment

if TYPE_CHECKING:
    from collections.abc import Callable

    from autogluon.core.models import AbstractModel

AddSeed = Literal["static", "fold-wise", "fold-config-wise"]


# ---------------------------------------------------------------------------
# Config-dict helpers
#
# A "config" is a model-hyperparameters dict that additionally carries AutoGluon's two reserved
# sub-dicts: ``ag_args`` (here only ``name_suffix``, used to name the experiment) and
# ``ag_args_ensemble`` (bagging/validation settings: random seed, fold-fitting strategy).
# ---------------------------------------------------------------------------
def add_suffix_to_config(config: dict, suffix: str) -> dict:
    """Return a copy of ``config`` tagged with ``ag_args.name_suffix = suffix``.

    The suffix is what names the resulting experiment (``{ag_name}{suffix}``). Raises if the
    config already carries ``ag_args`` (each config is tagged exactly once).
    """
    if "ag_args" in config:
        raise AssertionError("ag_args already exists in config!")
    config = copy.deepcopy(config)
    config["ag_args"] = {"name_suffix": suffix}
    return config


def _with_ag_args_ensemble(config: dict, **values) -> dict:
    """Return a copy of ``config`` with ``values`` merged into its ``ag_args_ensemble`` dict."""
    config = copy.deepcopy(config)
    config.setdefault("ag_args_ensemble", {}).update(values)
    return config


def add_seed_logic(config: dict, random_seed: int, vary_seed_across_folds: bool) -> dict:
    """Return a copy of ``config`` with the bagged model's random seed set in ``ag_args_ensemble``.

    ``vary_seed_across_folds`` makes each bagged child use a different seed (offset from
    ``random_seed``) rather than all children sharing it.
    """
    return _with_ag_args_ensemble(config, model_random_seed=random_seed, vary_seed_across_folds=vary_seed_across_folds)


def add_fold_fitting_strategy(config: dict, fold_fitting_strategy: str) -> dict:
    """Return a copy of ``config`` with the bagged ``fold_fitting_strategy`` set in ``ag_args_ensemble``."""
    return _with_ag_args_ensemble(config, fold_fitting_strategy=fold_fitting_strategy)


def combine_manual_and_random_configs(
    manual_configs: list[dict],
    random_configs: list[dict],
    name_id_suffix: str = "",
) -> list[dict]:
    """Tag and concatenate the manual + random configs, naming them ``_c{i}`` / ``_r{i}``.

    Manual configs get a ``_c{i}`` suffix (curated configs), random ones ``_r{i}``; ``name_id_suffix``
    is appended to every suffix to disambiguate parallel runs of the same model.
    """
    return [
        *(add_suffix_to_config(config, suffix=f"_c{i + 1}{name_id_suffix}") for i, config in enumerate(manual_configs)),
        *(add_suffix_to_config(config, suffix=f"_r{i + 1}{name_id_suffix}") for i, config in enumerate(random_configs)),
    ]


def configs_to_name_dict(configs: list[dict], name_prefix: str, model_type: str) -> dict[str, dict]:
    """Index already-tagged ``configs`` by their full name (``{name_prefix}{name_suffix}``).

    Returns ``{config_name: {hyperparameters, name_prefix, name_suffix, model_type}}`` — the
    dict form consumed by callers that want named configs rather than ready ``Experiment`` objects.
    """
    return {
        f"{name_prefix}{config['ag_args']['name_suffix']}": {
            "hyperparameters": config,
            "name_prefix": name_prefix,
            "name_suffix": config["ag_args"]["name_suffix"],
            "model_type": model_type,
        }
        for config in configs
    }


def get_random_searcher(search_space: dict, num_configs: int | None = None) -> LocalRandomSearcher:
    """Build a ``LocalRandomSearcher`` over ``search_space``.

    By default the searcher's first (default) configuration is discarded so sampling returns purely
    random configs. The exception: when the space has exactly ``num_configs`` total configurations
    and we want them all, the default is kept so none is dropped.
    """
    searcher = LocalRandomSearcher(search_space=search_space)

    keep_default_config = False
    if num_configs is not None:
        total_configs = searcher._get_num_configs()
        keep_default_config = total_configs is not None and num_configs <= total_configs

    if not keep_default_config:
        searcher.get_config()  # discard the default config
    return searcher


# ---------------------------------------------------------------------------
# Config generators
# ---------------------------------------------------------------------------
class AGConfigGenerator:
    """Base config generator: a model class + a list of manual configs.

    Subclasses implement :meth:`get_searcher_configs` to additionally sample random configs.
    ``name`` / ``model_type`` default to the model class's ``ag_name`` / ``ag_key``.
    """

    def __init__(
        self,
        model_cls: type[AbstractModel],
        name: str | None = None,
        model_type: str | None = None,
        manual_configs: list[dict] | None = None,
    ):
        self.model_cls = model_cls
        self.name = name if name is not None else model_cls.ag_name
        self.model_type = model_type if model_type is not None else model_cls.ag_key
        assert self.name is not None, "set `ag_name` and `ag_key` in the model class!"
        self.manual_configs = manual_configs if manual_configs is not None else []

    def get_searcher_configs(self, num_configs: int) -> list[dict]:
        """Sample ``num_configs`` random configurations from the search space. Implemented by subclasses."""
        raise NotImplementedError

    def generate_all_configs_lst(self, num_random_configs: int, name_id_suffix: str = "") -> list[dict]:
        """The manual + ``num_random_configs`` random configs as a flat, name-tagged list."""
        random_configs = self.get_searcher_configs(num_random_configs) if num_random_configs > 0 else []
        return combine_manual_and_random_configs(
            manual_configs=self.manual_configs, random_configs=random_configs, name_id_suffix=name_id_suffix
        )

    def generate_all_configs(self, num_random_configs: int) -> dict[str, dict]:
        """Like :meth:`generate_all_configs_lst`, but indexed by config name (see :func:`configs_to_name_dict`)."""
        configs = self.generate_all_configs_lst(num_random_configs=num_random_configs)
        return configs_to_name_dict(configs=configs, name_prefix=self.name, model_type=self.model_type)

    def generate_all_bag_experiments(
        self,
        num_random_configs: int,
        name_id_suffix: str = "",
        add_seed: AddSeed = "static",
        method_kwargs: dict | None = None,
        fold_fitting_strategy: Literal["sequential_local"] | None = None,
        **kwargs,
    ) -> list[AGModelBagExperiment]:
        """Build a bagged (:class:`AGModelBagExperiment`) experiment per config.

        Parameters
        ----------
        num_random_configs:
            Number of random configurations to sample (on top of the manual configs).
        name_id_suffix:
            Suffix appended to every config name, to distinguish parallel runs of the same model.
        add_seed:
            How the bagged random seed is assigned (see :func:`generate_bag_experiments`).
        method_kwargs:
            Extra kwargs forwarded to each experiment (e.g. ``init_kwargs`` / ``fit_kwargs``).
        fold_fitting_strategy:
            If set (``"sequential_local"``), fit folds in-process rather than in parallel Ray
            workers — recommended for local / debugger runs.
        **kwargs:
            Forwarded to :func:`generate_bag_experiments` (e.g. ``time_limit``,
            ``preprocessing_pipeline``, ``dynamic_tabarena_validation_protocol``).
        """
        configs = self.generate_all_configs_lst(num_random_configs=num_random_configs, name_id_suffix=name_id_suffix)
        return generate_bag_experiments(
            model_cls=self.model_cls,
            configs=configs,
            name_suffix_from_ag_args=True,
            add_seed=add_seed,
            method_kwargs=method_kwargs,
            fold_fitting_strategy=fold_fitting_strategy,
            **kwargs,
        )

    def generate_all_outer_experiments(
        self,
        num_random_configs: int,
        name_id_suffix: str = "",
        method_kwargs: dict | None = None,
        preprocessing_pipeline: str | None = None,
        extra_model_hyperparameters: dict | None = None,
        **kwargs,
    ) -> list[AGModelOuterExperiment]:
        """Build a no-validation 'outer' (:class:`AGModelOuterExperiment`) experiment per config.

        Each config becomes an experiment that trains on all the data with no train/val split,
        bagging, or ensemble. ``preprocessing_pipeline`` is resolved inside the wrapper, and
        ``extra_model_hyperparameters`` (e.g. ``ag.verbosity``) are merged into each model's
        hyperparameters — so the pipeline matches the bagged path.

        Parameters
        ----------
        num_random_configs / name_id_suffix:
            As in :meth:`generate_all_bag_experiments`.
        method_kwargs:
            Extra ``AGModelWrapper`` kwargs (e.g. ``{"shuffle_features": True}``).
        preprocessing_pipeline:
            Pipeline name forwarded to ``AGModelOuterExperiment`` / ``AGModelWrapper``.
        extra_model_hyperparameters:
            Hyperparameters merged into every model's hyperparameters.
        """
        configs = self.generate_all_configs_lst(num_random_configs=num_random_configs, name_id_suffix=name_id_suffix)
        return generate_outer_experiments(
            model_cls=self.model_cls,
            configs=configs,
            name_suffix_from_ag_args=True,
            method_kwargs=method_kwargs,
            preprocessing_pipeline=preprocessing_pipeline,
            extra_model_hyperparameters=extra_model_hyperparameters,
            **kwargs,
        )


class ConfigGenerator(AGConfigGenerator):
    """A config generator that samples random configs from a ConfigSpace ``search_space``."""

    def __init__(
        self,
        search_space: dict,
        model_cls: type[AbstractModel],
        name: str | None = None,
        manual_configs: list[dict] | None = None,
    ):
        super().__init__(model_cls=model_cls, name=name, manual_configs=manual_configs)
        self.search_space = search_space

    def get_searcher_configs(self, num_configs: int) -> list[dict]:
        searcher = get_random_searcher(self.search_space, num_configs=num_configs)
        return [searcher.get_config() for _ in range(num_configs)]


class CustomAGConfigGenerator(AGConfigGenerator):
    """A config generator that delegates random sampling to a ``search_space_func(num_configs)``."""

    def __init__(
        self,
        model_cls: type[AbstractModel],
        search_space_func: Callable[[int], list[dict]],
        name: str | None = None,
        manual_configs: list[dict] | None = None,
    ):
        super().__init__(model_cls=model_cls, name=name, manual_configs=manual_configs)
        self.search_space_func = search_space_func

    def get_searcher_configs(self, num_configs: int) -> list[dict]:
        return self.search_space_func(num_configs)


# ---------------------------------------------------------------------------
# Experiment builders (shared core + per-flavour wrappers)
# ---------------------------------------------------------------------------
def _resolve_config_name_suffix(
    config: dict,
    index: int,
    *,
    name_suffix_from_ag_args: bool,
    name_id_prefix: str,
    name_id_suffix: str,
    add_name_suffix_to_params: bool,
) -> tuple[str, dict]:
    """Resolve the display ``name_suffix`` for ``config``, returning ``(name_suffix, config)``.

    When ``name_suffix_from_ag_args`` the suffix is read from the config's ``ag_args.name_suffix``
    (set upstream by :func:`combine_manual_and_random_configs`); otherwise a positional ``_r{index+1}``
    suffix is generated and (when ``add_name_suffix_to_params``) tagged onto a copy of the config.
    """
    if name_suffix_from_ag_args:
        return config.get("ag_args", {}).get("name_suffix", ""), config
    name_suffix = f"_{name_id_prefix}{index + 1}{name_id_suffix}"
    if add_name_suffix_to_params:
        config = add_suffix_to_config(config=config, suffix=name_suffix)
    return name_suffix, config


def _build_experiments(
    model_cls: type[AbstractModel],
    configs: list[dict],
    *,
    build_experiment: Callable[[str, dict], object],
    name_bag_suffix: str = "",
    name_suffix_from_ag_args: bool,
    name_id_prefix: str,
    name_id_suffix: str,
    add_name_suffix_to_params: bool,
) -> list:
    """Name each config and build its experiment via ``build_experiment(name, config)``.

    The single iteration + naming routine shared by every ``generate_*_experiments`` flavour: the
    experiment name is ``{model_cls.ag_name}{name_suffix}{name_bag_suffix}``. Only ``build_experiment``
    (the per-flavour construction) differs between flavours.
    """
    experiments = []
    for index, config in enumerate(configs):
        name_suffix, config = _resolve_config_name_suffix(
            config,
            index,
            name_suffix_from_ag_args=name_suffix_from_ag_args,
            name_id_prefix=name_id_prefix,
            name_id_suffix=name_id_suffix,
            add_name_suffix_to_params=add_name_suffix_to_params,
        )
        name = f"{model_cls.ag_name}{name_suffix}{name_bag_suffix}"
        experiments.append(build_experiment(name, config))
    return experiments


def _apply_seed_to_bag_configs(
    configs: list[dict],
    add_seed: AddSeed,
    *,
    num_bag_folds: int,
    num_bag_sets: int,
) -> list[dict]:
    """Tag each bagged config with a ``model_random_seed`` according to ``add_seed``.

    * ``"static"`` — every config (and fold) uses seed 0.
    * ``"fold-wise"`` — seed 0, varied across the folds of each bag.
    * ``"fold-config-wise"`` — additionally offset each config's seed by ``num_bag_sets * num_bag_folds``
      so different configs explore disjoint seed ranges.
    """
    if add_seed == "static":
        return [add_seed_logic(config, random_seed=0, vary_seed_across_folds=False) for config in configs]
    if add_seed == "fold-wise":
        return [add_seed_logic(config, random_seed=0, vary_seed_across_folds=True) for config in configs]
    if add_seed == "fold-config-wise":
        offset_between_configs = num_bag_sets * num_bag_folds
        return [
            add_seed_logic(config, random_seed=i * offset_between_configs, vary_seed_across_folds=True)
            for i, config in enumerate(configs)
        ]
    raise ValueError(
        f"Invalid add_seed value: {add_seed!r}. Choose from 'static', 'fold-wise', or 'fold-config-wise'.",
    )


def generate_bag_experiments(
    model_cls: type[AbstractModel],
    configs: list[dict],
    time_limit: float | None = 3600,
    num_bag_folds: int = 8,
    num_bag_sets: int = 1,
    name_suffix_from_ag_args: bool = False,
    name_id_prefix: str = "r",
    name_id_suffix: str = "",
    name_bag_suffix: str = "_BAG_L1",
    add_name_suffix_to_params: bool = True,
    add_seed: AddSeed = "static",
    fold_fitting_strategy: Literal["sequential_local"] | None = None,
    **kwargs,
) -> list[AGModelBagExperiment]:
    """Build a bagged :class:`AGModelBagExperiment` per config (``num_bag_folds`` x ``num_bag_sets`` children).

    Each config is first tagged with its random seed (``add_seed``, see
    :func:`_apply_seed_to_bag_configs`) and any ``fold_fitting_strategy``; experiments are then named
    ``{ag_name}{name_suffix}{name_bag_suffix}`` and built. ``**kwargs`` are forwarded to
    :class:`AGModelBagExperiment` (e.g. ``preprocessing_pipeline``,
    ``dynamic_tabarena_validation_protocol``).
    """
    configs = _apply_seed_to_bag_configs(configs, add_seed, num_bag_folds=num_bag_folds, num_bag_sets=num_bag_sets)
    if fold_fitting_strategy is not None:
        configs = [add_fold_fitting_strategy(config, fold_fitting_strategy=fold_fitting_strategy) for config in configs]

    def build_experiment(name: str, config: dict) -> AGModelBagExperiment:
        return AGModelBagExperiment(
            name=name,
            model_cls=model_cls,
            model_hyperparameters=config,
            num_bag_folds=num_bag_folds,
            num_bag_sets=num_bag_sets,
            time_limit=time_limit,
            **kwargs,
        )

    return _build_experiments(
        model_cls,
        configs,
        build_experiment=build_experiment,
        name_bag_suffix=name_bag_suffix,
        name_suffix_from_ag_args=name_suffix_from_ag_args,
        name_id_prefix=name_id_prefix,
        name_id_suffix=name_id_suffix,
        add_name_suffix_to_params=add_name_suffix_to_params,
    )


def generate_outer_experiments(
    model_cls: type[AbstractModel],
    configs: list[dict],
    name_suffix_from_ag_args: bool = False,
    name_id_prefix: str = "r",
    name_id_suffix: str = "",
    add_name_suffix_to_params: bool = True,
    preprocessing_pipeline: str | None = None,
    method_kwargs: dict | None = None,
    extra_model_hyperparameters: dict | None = None,
    **kwargs,
) -> list[AGModelOuterExperiment]:
    """Build a no-validation :class:`AGModelOuterExperiment` per config (train on all data).

    ``ag_args`` / ``ag_args_ensemble`` are predictor/bagging-level keys and are dropped from the model
    hyperparameters (there is no ``TabularPredictor`` here to consume them); the per-config
    ``ag_args.name_suffix`` is still used to name the experiment. ``extra_model_hyperparameters`` are
    merged into each model's hyperparameters (they must not collide).
    """
    extra_model_hyperparameters = extra_model_hyperparameters or {}

    def build_experiment(name: str, config: dict) -> AGModelOuterExperiment:
        model_hyperparameters = {k: v for k, v in config.items() if k not in ("ag_args", "ag_args_ensemble")}
        overlapping = set(extra_model_hyperparameters).intersection(model_hyperparameters)
        assert not overlapping, f"extra_model_hyperparameters overlap with model hyperparameters: {overlapping}"
        return AGModelOuterExperiment(
            name=name,
            model_cls=model_cls,
            model_hyperparameters={**model_hyperparameters, **extra_model_hyperparameters},
            preprocessing_pipeline=preprocessing_pipeline,
            method_kwargs=method_kwargs,
            **kwargs,
        )

    return _build_experiments(
        model_cls,
        configs,
        build_experiment=build_experiment,
        name_suffix_from_ag_args=name_suffix_from_ag_args,
        name_id_prefix=name_id_prefix,
        name_id_suffix=name_id_suffix,
        add_name_suffix_to_params=add_name_suffix_to_params,
    )
