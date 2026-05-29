"""Model/pipeline selection and experiment-config generation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import yaml
from tabarena.benchmark.experiment.experiment_constructor import Experiment

from tabflow_slurm.setup.constraints import TABICL_CONSTRAINTS, TABPFNV2_CONSTRAINTS, ModelConstraints

if TYPE_CHECKING:
    from tabflow_slurm.setup.resources import ResourcesSetup


@dataclass
class ModelPipelinesToRunSetup:
    """Defines which models to run in the benchmark, plus model-related behavior.

    Encapsulates the list of models with per-model config counts, preprocessing
    pipelines applied to each model, and miscellaneous model-level settings
    (predict batching, verbosity, artifact paths, fake memory, fold fitting).
    """

    n_random_configs: int = 50
    """Number of random hyperparameter configurations to run for each model"""
    models: list[tuple[str, int | str | dict]] = field(default_factory=list)
    """List of models to run in the benchmark with metadata.
    Metadata keys from left to right:
        - model name: str
        - number of random hyperparameter configurations to run: int or str
            Some special cases are:
                - If 0, only the default configuration is run.
                - If "all", `n_random_configs`-many configurations are run.
                - If dict, kwargs for AGExperiment
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
    model_agnostic_preprocessing: bool = True
    """Whether to use model-agnostic preprocessing or not.
    By default, we use AutoGluon's automatic preprocessing for all models.
    This can be disabled by setting this to False. Warning: the model then needs
    to be able to handle this!
    """
    preprocessing_pipelines: list[str] = field(default_factory=lambda: ["tabarena_default"])
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
    model_artifacts_base_path: str | Path | None = "/tmp"  # noqa: S108
    """Adapt the default temporary directory used for model artifacts in TabArena.
        - If None, the default temporary directory is used: "./AutoGluonModels".
        - If a string or Path, the directory is used as the base path for the temporary
        and any model artifacts will be stored in time-stamped subdirectories.
    """
    model_verbosity: int | None = None
    """Verbosity level passed to the model via model_hyperparameters['verbose'].
    Controls model-level logging (e.g. CatBoost iteration logs, LightGBM verbosity)
    independently of AutoGluon's overall verbosity. If None, no model-level verbosity is set."""
    adapt_num_folds_to_n_classes: bool = True
    """Whether to adapt the number of folds to the number of classes for classification tasks.
    Ensures that each fold has at least one sample of each class.
    """
    dynamic_tabarena_validation_protocol: bool = True
    """If True, the validation data will be adapted dynamically based on the task.
    WARNING: this can overwrite the configured validation of a configuration!"""

    custom_model_constraints: dict[str, ModelConstraints] = field(default_factory=dict)
    """Per-model overrides of dataset compatibility, keyed by AG model key.

    Entries here are merged on top of `DEFAULT_MODEL_CONSTRAINTS` (custom wins
    on key collisions). Models not listed in either map are considered
    compatible with every dataset.
    """

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
        resources_setup: ResourcesSetup,
        verbosity: int,
        shuffle_features: bool,
    ) -> list[dict]:
        """Build experiments, write them to `configs_path`, and return the parsed methods list.

        `verbosity` and `shuffle_features` are bench-level toggles threaded
        through into the per-method kwargs alongside the model-specific
        overrides this dataclass owns.
        """
        from tabarena.benchmark.experiment import (
            YamlExperimentSerializer,
        )

        base_method_kwargs = {
            "init_kwargs": {"verbosity": verbosity},
            "shuffle_features": shuffle_features,
            "fit_kwargs": {},
            "extra_model_hyperparameters": {},
        }

        experiments_all = self.generate_experiments(
            base_method_kwargs=base_method_kwargs,
            resources_setup=resources_setup,
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

    def generate_experiments(
        self,
        *,
        base_method_kwargs: dict,
        resources_setup: ResourcesSetup,
    ) -> list:
        """Build the full list of experiment configs (one per model x pipeline x random config)."""
        method_kwargs = self._enrich_method_kwargs(base_method_kwargs)
        self._print_experiment_summary(method_kwargs)
        return [
            exp
            for pipeline_name in self.preprocessing_pipelines
            for exp in self._build_experiments_for_pipeline(pipeline_name, method_kwargs, resources_setup)
        ]

    def _enrich_method_kwargs(self, base_method_kwargs: dict) -> dict:
        """Layer this dataclass's model-level overrides on top of the bench-level base."""
        mk = deepcopy(base_method_kwargs)
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
        resources_setup: ResourcesSetup,
    ) -> list:
        """Per-pipeline overrides + per-model dispatch."""
        pipeline_kwargs = deepcopy(method_kwargs)
        name_id_suffix = ""
        if self.model_agnostic_preprocessing:
            if pipeline_name != "default":
                pipeline_kwargs["preprocessing_pipeline"] = pipeline_name
            if pipeline_name != "tabarena_default":
                name_id_suffix = f"_{pipeline_name}"

        experiments: list = []
        for model in self.models:
            experiments.extend(
                self._build_experiments_for_model(model, pipeline_kwargs, name_id_suffix, resources_setup)
            )
        return experiments

    def _build_experiments_for_model(
        self,
        model: Experiment | tuple,
        pipeline_method_kwargs: dict,
        name_id_suffix: str,
        resources_setup: ResourcesSetup,
    ) -> list:
        if isinstance(model, Experiment):
            return [model]
        model_name, n_configs_or_kwargs = model[0], model[1]
        if isinstance(model_name, str) and model_name.startswith("AutoGluon"):
            return self._generate_autogluon_config(
                model_name=model_name,
                agexp_kwargs=n_configs_or_kwargs,
                pipeline_method_kwargs=pipeline_method_kwargs,
                resources_setup=resources_setup,
            )
        return self._generate_model_configs(
            model_name=model_name,
            n_configs=n_configs_or_kwargs,
            pipeline_method_kwargs=pipeline_method_kwargs,
            name_id_suffix=name_id_suffix,
            resources_setup=resources_setup,
        )

    @staticmethod
    def _generate_autogluon_config(
        *,
        model_name: str,
        agexp_kwargs: dict,
        pipeline_method_kwargs: dict,
        resources_setup: ResourcesSetup,
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
        agexp_kwargs["fit_kwargs"]["time_limit"] = resources_setup.time_limit

        return [AGExperiment(name=model_name, **agexp_kwargs)]

    def _generate_model_configs(
        self,
        *,
        model_name: str,
        n_configs: int | str,
        pipeline_method_kwargs: dict,
        name_id_suffix: str,
        resources_setup: ResourcesSetup,
        default_seed_config: str = "fold-config-wise",
    ) -> list:
        from tabarena.models.utils import get_configs_generator_from_name

        if isinstance(n_configs, str) and n_configs == "all":
            n_configs = self.n_random_configs
        elif not isinstance(n_configs, int):
            raise ValueError(
                f"Invalid number of configurations for model {model_name}: {n_configs}. Must be an integer or 'all'."
            )
        if isinstance(model_name, str):
            config_generator = get_configs_generator_from_name(model_name)
        else:
            config_generator = deepcopy(model_name)
        return config_generator.generate_all_bag_experiments(
            num_random_configs=n_configs,
            add_seed=default_seed_config,
            name_id_suffix=name_id_suffix,
            method_kwargs=pipeline_method_kwargs,
            time_limit=resources_setup.time_limit,
            time_limit_with_preprocessing=resources_setup.time_limit_with_model_agnostic_preprocessing,
        )
