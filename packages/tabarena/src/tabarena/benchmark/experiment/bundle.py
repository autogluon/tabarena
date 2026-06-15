"""Experiment selection and config generation: the TabArenaExperimentBundle."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from tabarena.benchmark.experiment.experiment_constructor import Experiment
from tabarena.benchmark.experiment.model_constraints import (
    TABICL_CONSTRAINTS,
    TABPFNV2_CONSTRAINTS,
    ModelConstraints,
)

if TYPE_CHECKING:
    from tabarena.utils.config_utils import AGConfigGenerator


def _experiment_ag_key(experiment: Experiment) -> str:
    """Resolve the AutoGluon model key an experiment's constraints are looked up under.

    The wrapped class is read from the experiment's captured constructor args
    (``model_cls`` for model experiments, ``method_cls`` for plain ones — the same keys
    the serialized YAML form carries) and its ``ag_key`` is used. Full-pipeline
    AutoGluon experiments (neither key) map to ``"AutoGluon"``; classes without an
    ``ag_key`` fall back to their class name, which simply won't match any constraint
    key (i.e. unconstrained).
    """
    ctor_args = getattr(experiment, "_locals", None) or {}
    cls_obj = ctor_args.get("model_cls") or ctor_args.get("method_cls")
    if cls_obj is not None:
        return getattr(cls_obj, "ag_key", None) or cls_obj.__name__
    if experiment.name.startswith("AutoGluon"):
        return "AutoGluon"
    return experiment.name


@dataclass(kw_only=True)
class TabArenaExperimentBundle:
    """Defines which models/experiments to run in a benchmark and builds them.

    Encapsulates the list of models with per-model config counts, the
    preprocessing pipelines applied to each model, and miscellaneous model-level
    settings (predict batching, verbosity, artifact paths, fold fitting).

    `build_experiments` / `generate_configs_yaml` turn this into ready-to-run
    `Experiment` objects, baking the compute resources (passed at build time)
    into each experiment so the resulting YAML is self-contained.
    """

    models: list[tuple[str | AGConfigGenerator, int | str | dict] | Experiment] = field(default_factory=list)
    """List of models to run in the benchmark with metadata.
    Metadata keys from left to right:
        - model name: str
        - number of random hyperparameter configurations to run: int or str
            Some special cases are:
                - If 0, only the default configuration is run.
                - If "all", `n_random_configs`-many configurations are run.
                - If dict, kwargs for AGExperiment

    Custom (non-registry) models — two ways to include one:
        - RECOMMENDED: pass a `(config_generator, n_configs)` tuple, where
          `config_generator` is an `AGConfigGenerator` (e.g. `ConfigGenerator`)
          wrapping your `model_cls`. This goes through the same path as a registry
          name, so the bundle bakes in ALL its settings — compute resources,
          `preprocessing_pipeline`, `dynamic_tabarena_validation_protocol`,
          `time_limit`, fold-fitting strategy — exactly as for registry models. Use
          `manual_configs=[{}]` (+ `n_configs=0`) for a default-only run, or a real
          search space with `n_configs > 0` for HPO.
        - Full manual control: pass a fully-built `Experiment` (e.g. an
          `AGModelBagExperiment`) directly. It is used *verbatim* — the bundle does
          NOT bake its resources / preprocessing / validation protocol into it, so
          set those on the experiment yourself.
      Either way, dataset-compatibility constraints are still resolved/attached like
      any other (see `_attach_model_constraints`).

    Example usage:
        # Run all random configs for LightGBM, 10 random configs for Random Forest,
        and only the default for TabDPT.
        default_factory=lambda: [
                ("LightGBM", "all"),
                ("RandomForest", 10),
                ("TabDPT", 0),
            ]
        )

    Models one can use: "CatBoost", "EBM", "ExtraTrees", "FastaiMLP", "KNN", "LightGBM",
    "Linear", "ModernNCA", "TorchMLP", "RandomForest", "RealMLP", "TabDPT", "TabICL",
    "TabM", "TabPFNv2", "XGBoost", "Mitra", "xRFM", "RealTabPFN-v2.5", "SAP-RPT-OSS",
    "TabICLv2", "PerpetualBooster", "TabSTAR"

    For the newest set of available models, see:
    `tabarena.models.utils.get_configs_generator_from_name`
    """
    n_random_configs: int
    """Number of random hyperparameter configurations to run for each model"""
    preprocessing_pipelines: list[str]
    """EXPERIMENTAL!
    Preprocessing pipelines to add to the configurations we want to run.

    Each options multiplies the number of configurations to run by the number of
    pipelines. For example, if we have 10 configurations and 2 pipelines, we will
    run 20 configurations.

    Options:
        - "default": Use the default preprocessing pipeline.
        - "tabarena_default": new model agnostic and model specific preprocessing
            updates for TabArena (experimental, can be buggy!).
        - Any other string points to custom experimental code for now.
    """
    model_agnostic_preprocessing: bool = True
    """Whether to use model-agnostic preprocessing or not.
    By default, we use AutoGluon's automatic preprocessing for all models.
    This can be disabled by setting this to False. Warning: the model then needs
    to be able to handle this!
    """

    max_predict_batch_size: int | None = None
    """Maximal batch size for the predict function of the models.
    This is used at validation and test predict time. Thus, it trades off speed for memory usage.
    If None, no limit is applied.
    """
    sequential_local_fold_fitting: bool = False
    """Use Ray for local fold fitting. This is used to speed up the local fold fitting
    and force this behavior if True. If False the default strategy of running the
    local fold fitting is used, as determined by AutoGluon and the model's
    default_ag_args_ensemble parameters."""
    outer_experiments: bool = False
    """If True, build no-validation 'outer' experiments instead of bagged ones: each model is
    fit directly on all the data (``AGModelWrapper``) with no train/val split, bagging, or
    ensemble. The bundle's other settings still apply. Useful for full-data
    methods and quick no-validation runs. 
    A pre-built ``Experiment`` passed in ``models`` is still used verbatim."""
    model_artifacts_base_path: str | Path | None = "/tmp"  # noqa: S108
    """Adapt the default temporary directory used for model artifacts in TabArena.
        - If None, the default temporary directory is used: "./AutoGluonModels".
        - If a string or Path, the directory is used as the base path for the temporary
        and any model artifacts will be stored in time-stamped subdirectories.
    """
    verbosity: int = 2
    """AutoGluon verbosity level for the fit, passed via `init_kwargs['verbosity']`."""
    model_verbosity: int | None = 4
    """Verbosity level passed to the model via model_hyperparameters['verbose'].
    Controls model-level logging (e.g. CatBoost iteration logs, LightGBM verbosity)
    independently of AutoGluon's overall verbosity. If None, no model-level verbosity is set."""
    adapt_num_folds_to_n_classes: bool = True
    """Whether to adapt the number of folds to the number of classes for classification tasks.
    Ensures that each fold has at least one sample of each class.
    """
    shuffle_features: bool = True
    """Whether to shuffle the features of the datasets. Only here for backward compatibility
    with the original TabArena setup, but not recommended to change."""
    dynamic_tabarena_validation_protocol: bool = True
    """If True, experiments built by this bundle adapt their validation data
    dynamically based on the task at run time (handled by the run engine).
    WARNING: this can overwrite the configured validation of a configuration!"""

    custom_model_constraints: dict[str, ModelConstraints] = field(default_factory=dict)
    """Per-model overrides of dataset compatibility, keyed by AG model key.

    Entries here are merged on top of `DEFAULT_MODEL_CONSTRAINTS` (custom wins
    on key collisions). Models not listed in either map are considered
    compatible with every dataset. Consumed by the benchmark orchestration to
    skip incompatible `(model, dataset)` jobs.
    """

    DEFAULT_TIME_LIMIT: ClassVar[int] = 3600
    """Default per-model fit ``time_limit`` (seconds) when ``build_experiments`` is called
    without one. 1 hour for the generic TabArena setup; subclasses override (BeyondArena
    uses 4 hours)."""

    DEFAULT_MODEL_CONSTRAINTS: ClassVar[dict[str, ModelConstraints]] = {
        "TABICL": TABICL_CONSTRAINTS,
        "TA-TABICL": TABICL_CONSTRAINTS,
        "TABPFNV2": TABPFNV2_CONSTRAINTS,
        "TA-TABPFNV2": TABPFNV2_CONSTRAINTS,
        "MITRA": TABPFNV2_CONSTRAINTS,
    }

    @property
    def model_constraints(self) -> dict[str, ModelConstraints]:
        """Effective model constraints (defaults overridden by `custom_model_constraints`)."""
        return {**self.DEFAULT_MODEL_CONSTRAINTS, **self.custom_model_constraints}

    def generate_configs_yaml(
        self,
        *,
        configs_path: str,
        time_limit: int | None = None,
        num_cpus: int | None = None,
        num_gpus: int | None = None,
        memory_limit: int | None = None,
        time_limit_with_preprocessing: bool = False,
    ) -> list[dict]:
        """Build experiments, write them to `configs_path`, and return the parsed methods list.

        Compute resources (`num_cpus`/`num_gpus`/`memory_limit`/`time_limit`) are
        baked into each experiment alongside the model-specific overrides this
        bundle owns (including `verbosity`). All have defaults; see
        :meth:`build_experiments` for how each is resolved (``None`` -> auto-detect
        for resources, the bundle's :attr:`DEFAULT_TIME_LIMIT` for ``time_limit``).
        """
        import yaml

        from tabarena.benchmark.experiment import YamlExperimentSerializer

        experiments_all = self.build_experiments(
            time_limit=time_limit,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory_limit=memory_limit,
            time_limit_with_preprocessing=time_limit_with_preprocessing,
        )

        # Verify no duplicate names
        experiment_names = set()
        for experiment in experiments_all:
            if experiment.name in experiment_names:
                raise AssertionError(
                    f"Found multiple instances of experiment named {experiment.name}. "
                    f"All experiment names must be unique!",
                )
            experiment_names.add(experiment.name)

        YamlExperimentSerializer.to_yaml(
            experiments=experiments_all,
            path=configs_path,
        )

        # Read YAML file and return the methods list.
        with Path(configs_path).open() as file:
            return yaml.safe_load(file)["methods"]

    def build_experiments(
        self,
        *,
        time_limit: int | None = None,
        num_cpus: int | None = None,
        num_gpus: int | None = None,
        memory_limit: int | None = None,
        time_limit_with_preprocessing: bool = False,
    ) -> list[Experiment]:
        """Build the full list of experiments (one per model x pipeline x random config).

        Each experiment leaves here self-contained: compute resources are baked into its
        kwargs, and its dataset-compatibility :class:`ModelConstraints` (resolved from
        :attr:`model_constraints` by AG model key) are attached to the experiment itself —
        so downstream consumers (``build_jobs``, ``run_jobs``, the SLURM dispatch) respect
        them without being handed a separate constraints mapping.

        All arguments have defaults, so ``build_experiments()`` works with no arguments:

        - ``time_limit``: ``None`` uses this bundle's :attr:`DEFAULT_TIME_LIMIT`
          (1 hour for TabArena, 4 hours for BeyondArena).
        - ``num_cpus`` / ``num_gpus`` / ``memory_limit``: ``None`` is baked into the
          experiment as "auto-detect on whatever node runs it" — resolved to the running
          machine's resources at fit time (see ``Experiment._apply_resources``).
        - ``time_limit_with_preprocessing``: defaults to ``False`` (the ``time_limit``
          bounds model fit only, excluding preprocessing).
        """
        if time_limit is None:
            time_limit = self.DEFAULT_TIME_LIMIT
        method_kwargs = self._init_base_method_kwargs(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory_limit=memory_limit,
        )
        self._print_experiment_summary(method_kwargs)
        experiments = [
            exp
            for pipeline_name in self.preprocessing_pipelines
            for exp in self._build_experiments_for_pipeline(
                pipeline_name,
                method_kwargs,
                time_limit=time_limit,
                time_limit_with_preprocessing=time_limit_with_preprocessing,
            )
        ]
        self._attach_model_constraints(experiments)
        return experiments

    def _attach_model_constraints(self, experiments: list[Experiment]) -> None:
        """Attach each experiment's :class:`ModelConstraints` (keyed by its AG model key).

        Experiments that already carry explicit constraints keep them; models without an
        entry in :attr:`model_constraints` stay unconstrained (``None``).
        """
        constraints_by_key = self.model_constraints
        for experiment in experiments:
            if experiment.model_constraints is None:
                constraints = constraints_by_key.get(_experiment_ag_key(experiment))
                if constraints is not None:
                    experiment.set_model_constraints(constraints)

    def _init_base_method_kwargs(
        self,
        *,
        num_cpus: int | None,
        num_gpus: int | None,
        memory_limit: int | None,
    ) -> dict:
        """Build the per-method kwargs: the bench-level base, plus this bundle's
        model-level overrides and the (baked) compute resources.
        """
        mk = {
            "init_kwargs": {"verbosity": self.verbosity},
            "shuffle_features": self.shuffle_features,
            "fit_kwargs": {},
            "extra_model_hyperparameters": {},
        }
        if self.model_artifacts_base_path is not None:
            mk["init_kwargs"]["default_base_path"] = self.model_artifacts_base_path
        if not self.model_agnostic_preprocessing:
            mk["fit_kwargs"]["feature_generator"] = None
        if self.adapt_num_folds_to_n_classes:
            mk["fit_kwargs"]["adapt_num_bag_folds_to_n_classes"] = True
        if self.max_predict_batch_size is not None:
            mk["extra_model_hyperparameters"]["ag.max_batch_size"] = self.max_predict_batch_size
        if self.model_verbosity is not None:
            mk["extra_model_hyperparameters"]["ag.verbosity"] = self.model_verbosity
        # Bake compute resources into the experiment so the serialized config is
        # self-contained. `None` num_cpus/num_gpus/memory_limit are auto-detected at
        # run time by the Experiment (see `Experiment._apply_resources`).
        mk["fit_kwargs"]["num_cpus"] = num_cpus
        mk["fit_kwargs"]["num_gpus"] = num_gpus
        mk["fit_kwargs"]["memory_limit"] = memory_limit
        return mk

    def _print_experiment_summary(self, method_kwargs: dict) -> None:
        print(
            "Generating experiments for models...",
            f"\n\t`all` := number of configs: {self.n_random_configs}",
            f"\n\t{len(self.models)} models: {self.models}",
            f"\n\t{len(self.preprocessing_pipelines)} preprocessing pipelines: {self.preprocessing_pipelines}",
            f"\n\tMethod kwargs: {method_kwargs}",
        )

    def _build_experiments_for_pipeline(
        self,
        pipeline_name: str,
        method_kwargs: dict,
        *,
        time_limit: int,
        time_limit_with_preprocessing: bool,
    ) -> list:
        """Per-pipeline overrides + per-model dispatch."""
        pipeline_kwargs = deepcopy(method_kwargs)
        name_id_suffix = ""
        preprocessing_pipeline = None
        if self.model_agnostic_preprocessing:
            if pipeline_name != "default":
                preprocessing_pipeline = pipeline_name
            if pipeline_name != "tabarena_default":
                name_id_suffix = f"_{pipeline_name}"

        experiments: list = []
        for model in self.models:
            experiments.extend(
                self._build_experiments_for_model(
                    model,
                    pipeline_kwargs,
                    name_id_suffix,
                    preprocessing_pipeline=preprocessing_pipeline,
                    time_limit=time_limit,
                    time_limit_with_preprocessing=time_limit_with_preprocessing,
                ),
            )
        return experiments

    def _build_experiments_for_model(
        self,
        model: Experiment | tuple,
        pipeline_method_kwargs: dict,
        name_id_suffix: str,
        *,
        preprocessing_pipeline: str | None,
        time_limit: int,
        time_limit_with_preprocessing: bool,
    ) -> list:
        if isinstance(model, Experiment):
            return [model]
        model_name, n_configs_or_kwargs = model[0], model[1]
        if isinstance(model_name, str) and model_name.startswith("AutoGluon"):
            return self._generate_autogluon_config(
                model_name=model_name,
                agexp_kwargs=n_configs_or_kwargs,
                pipeline_method_kwargs=pipeline_method_kwargs,
                time_limit=time_limit,
            )
        return self._generate_model_configs(
            model_name=model_name,
            n_configs=n_configs_or_kwargs,
            pipeline_method_kwargs=pipeline_method_kwargs,
            name_id_suffix=name_id_suffix,
            preprocessing_pipeline=preprocessing_pipeline,
            time_limit=time_limit,
            time_limit_with_preprocessing=time_limit_with_preprocessing,
        )

    @staticmethod
    def _generate_autogluon_config(
        *,
        model_name: str,
        agexp_kwargs: dict,
        pipeline_method_kwargs: dict,
        time_limit: int,
    ) -> list:
        """Parse the AutoGluon config from the models."""
        from tabarena.benchmark.experiment.experiment_constructor import (
            AGExperiment,
        )

        # deepcopy: shallow .copy() leaves nested `fit_kwargs` / `init_kwargs` shared with
        # the caller's dict, and the subsequent .update() / item assignment would mutate
        # the user's `self.models` entry across calls.
        agexp_kwargs = deepcopy(agexp_kwargs)
        for key in ["fit_kwargs", "init_kwargs"]:
            agexp_kwargs.setdefault(key, {})
            if key in pipeline_method_kwargs:
                agexp_kwargs[key].update(pipeline_method_kwargs[key])
        agexp_kwargs["fit_kwargs"]["time_limit"] = time_limit

        return [AGExperiment(name=model_name, **agexp_kwargs)]

    def _generate_model_configs(
        self,
        *,
        model_name: str,
        n_configs: int | str,
        pipeline_method_kwargs: dict,
        name_id_suffix: str,
        preprocessing_pipeline: str | None,
        time_limit: int,
        time_limit_with_preprocessing: bool,
        default_seed_config: str = "fold-config-wise",
    ) -> list:
        from tabarena.models.utils import get_configs_generator_from_name

        if isinstance(n_configs, str) and n_configs == "all":
            n_configs = self.n_random_configs
        elif not isinstance(n_configs, int):
            raise ValueError(
                f"Invalid number of configurations for model {model_name}: {n_configs}. Must be an integer or 'all'.",
            )
        if isinstance(model_name, str):
            config_generator = get_configs_generator_from_name(model_name)
        else:
            config_generator = deepcopy(model_name)

        if self.outer_experiments:
            # No-validation path: one direct ``AGModelWrapper`` fit on all data, no bagging /
            # ensemble. The compute resources + time_limit are forwarded to the model's `fit`,
            # and the preprocessing pipeline + shuffle_features still apply (resolved inside the
            # wrapper) — so the only thing dropped vs the bagged path is the validation/bagging.
            base_fit_kwargs = pipeline_method_kwargs.get("fit_kwargs") or {}
            outer_fit_kwargs = {
                "num_cpus": base_fit_kwargs.get("num_cpus"),
                "num_gpus": base_fit_kwargs.get("num_gpus"),
                "time_limit": time_limit,
            }
            return config_generator.generate_all_outer_experiments(
                num_random_configs=n_configs,
                name_id_suffix=name_id_suffix,
                method_kwargs={
                    "shuffle_features": pipeline_method_kwargs["shuffle_features"],
                    "fit_kwargs": outer_fit_kwargs,
                },
                preprocessing_pipeline=preprocessing_pipeline,
                extra_model_hyperparameters=pipeline_method_kwargs.get("extra_model_hyperparameters") or None,
            )

        return config_generator.generate_all_bag_experiments(
            num_random_configs=n_configs,
            add_seed=default_seed_config,
            name_id_suffix=name_id_suffix,
            method_kwargs=pipeline_method_kwargs,
            time_limit=time_limit,
            time_limit_with_preprocessing=time_limit_with_preprocessing,
            preprocessing_pipeline=preprocessing_pipeline,
            fold_fitting_strategy="sequential_local" if self.sequential_local_fold_fitting else None,
            dynamic_tabarena_validation_protocol=self.dynamic_tabarena_validation_protocol,
        )


@dataclass(kw_only=True)
class TabArenaV0pt1ExperimentBundle(TabArenaExperimentBundle):
    """The original TabArena-v0.1 experiment bundle, for backward compatibility."""

    n_random_configs: int = 200
    """TabArena-v0.1 used 200 configs per model."""
    shuffle_features: bool = False
    """TabArena-v0.1 default"""
    dynamic_tabarena_validation_protocol: bool = False
    """Only used in v0.2 or larger with new data foundry task metadata integration."""
    preprocessing_pipelines: list[str] = field(default_factory=lambda: ["default"])
    """Use AutoGluon default preprocessing only."""
    adapt_num_folds_to_n_classes: bool = False
    """TabArena-v0.1 did not adapt the number of folds to the number of classes."""


@dataclass(kw_only=True)
class BeyondArenaExperimentBundle(TabArenaExperimentBundle):
    """Experiment bundle for the BeyondArena paper.
    It used the current defaults, 25 configs, and the new TabArena preprocessing.
    """

    n_random_configs: int = 25
    preprocessing_pipelines: list[str] = field(default_factory=lambda: ["tabarena_default"])

    DEFAULT_TIME_LIMIT: ClassVar[int] = 4 * 3600
    """BeyondArena used a 4-hour per-model fit time limit."""
