from __future__ import annotations

import copy
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from tabarena.benchmark.exec_models import ExternalSystemModel
from tabarena.benchmark.validation_protocol import TABARENA_V0PT1_VALIDATION_PROTOCOL, ValidationProtocol

if TYPE_CHECKING:
    import pandas as pd
    from autogluon.core.metrics import Scorer

    from tabarena.benchmark.task.metadata import ValidationMetadata

CONFIGS_PATH = Path(__file__).with_name("configs.json")
NAME = "LGBM-Prep-FTE25"
ENSEMBLE_SIZE = 40
"""Greedy ensemble selection iterations, as in TabArena's tuned + ensembled simulation."""


@dataclass(frozen=True)
class _ArenaSetup:
    """How the bags validate and preprocess on one benchmark: the way that benchmark fits its models."""

    protocol: ValidationProtocol
    preprocessing_pipeline: str | None
    shuffle_features: bool
    """Whether the benchmark permutes the feature columns per split (``AbstractExecModel.shuffle_features``).

    The benchmark applies it to a system's input as well, so the system's bags see the same column order as
    the per-config model runs; only :func:`make_bag_experiment`, which builds those model runs, sets it."""


ARENA_SETUPS: dict[str, _ArenaSetup] = {
    # TabArena-v0.1: 8 folds x 1 set, plain stratified inner splits, no model-agnostic preprocessing.
    "tabarena": _ArenaSetup(
        protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL, preprocessing_pipeline=None, shuffle_features=False
    ),
    # BeyondArena: 8 folds x 1 set with group / time-aware inner splits and class-adaptive folds, and TabArena's
    # default preprocessing. This is BeyondArena's official protocol without its tiny-data 5 x 5 regime, which
    # the reference runs predate; the system keeps 8 x 1 on every task.
    "beyondarena": _ArenaSetup(
        protocol=ValidationProtocol(
            name="LGBM-Prep-FTE25 (BeyondArena)", task_specific_validation=True, adapt_num_folds_to_n_classes=True
        ),
        preprocessing_pipeline="tabarena_default",
        shuffle_features=True,
    ),
}


def load_configs() -> list[dict[str, Any]]:
    """The 26 FTE25 configs in portfolio order: ``c1`` is the LightGBM default."""
    return json.loads(CONFIGS_PATH.read_text())["configs"]


def make_bag_experiment(
    config: dict[str, Any],
    *,
    arena: str,
    bag_time_limit: float | None,
    num_cpus: int | None = None,
    memory_limit: float | None = None,
    artifact_path: str | None = None,
):
    """The ``AGModelBagExperiment`` of one config: what TabArena would run for it as a bagged model.

    The system fits exactly these bags. Run through an arena context, the same experiment produces the
    config's own per-split result (with its out-of-fold artifact), which is how a reference run of the system
    can be assembled from per-config fits.
    """
    from tabarena.benchmark.experiment import AGModelBagExperiment
    from tabarena.systems.lgbm_prep_fte25._model import LGBMDefaultModel, LGBMPrepModel

    setup = ARENA_SETUPS[arena]
    if config["model"] == "lightgbm_default":
        model_cls, hyperparameters = LGBMDefaultModel, {}
    else:
        model_cls, hyperparameters = LGBMPrepModel, copy.deepcopy(config["hyperparameters"])
    hyperparameters["ag_args_ensemble"] = {
        "model_random_seed": config["bag_seed"],
        "vary_seed_across_folds": True,
        "num_folds_parallel": 1,
        "fold_fitting_strategy": "sequential_local",
    }
    fit_kwargs: dict[str, Any] = {"num_gpus": 0}
    if num_cpus is not None:
        fit_kwargs["num_cpus"] = num_cpus
    if memory_limit is not None:
        fit_kwargs["memory_limit"] = memory_limit
    init_kwargs: dict[str, Any] = {"verbosity": 0}
    if artifact_path is not None:
        init_kwargs["default_base_path"] = artifact_path
    return AGModelBagExperiment(
        name=f"{NAME}_{config['name']}",
        model_cls=model_cls,
        model_hyperparameters=hyperparameters,
        time_limit=bag_time_limit,
        preprocessing_pipeline=setup.preprocessing_pipeline,
        validation_protocol=setup.protocol,
        method_kwargs={
            "init_kwargs": init_kwargs,
            "fit_kwargs": fit_kwargs,
            "shuffle_features": setup.shuffle_features,
        },
    )


def ensemble_weights(
    oof_predictions: list[np.ndarray], y_oof: np.ndarray, *, problem_type: str, metric: Scorer
) -> np.ndarray:
    """Greedy (Caruana) ensemble weights over the bags, fit on their out-of-fold predictions.

    ``oof_predictions`` are in AutoGluon's internal label space (positive-class probability for binary), in
    portfolio order; ties go to the earlier bag. Shared with the result rebuild so both select identically.
    """
    from tabarena.simulation.ensemble.greedy_ensembler import GreedyEnsembler

    ensembler = GreedyEnsembler(problem_type=problem_type, metric=metric, ensemble_size=ENSEMBLE_SIZE)
    ensembler.fit(predictions=np.stack(oof_predictions), labels=np.asarray(y_oof))
    return np.asarray(ensembler.weights_, dtype=float)


class LGBMPrepFTE25SystemModel(ExternalSystemModel):
    """LightGBM + TabPrep with a fixed filtered portfolio, bagged and ensembled.

    The pipeline, per training split:

    1. Fit 26 bagged LightGBM models: AutoGluon's LightGBM default and 25 configs that each pair LightGBM
       parameters with TabPrep feature engineering (8 of them) or none (17). The 25 were selected offline, on
       synthetic data only, for the tuned + ensembled objective out of a bank of 4,000 configs drawn from
       TabArena's LightGBM space (TabPrep on every fourth). Each bag uses 8 inner folds; TabPrep is fitted inside
       every fold on its training rows only, and at most 2,000 features (by absolute Spearman correlation on
       those rows) go into LightGBM.
    2. Greedy ensemble selection (40 iterations) on the bags' out-of-fold predictions.
    3. Predict with the weighted average of the selected bags' predictions (each bag averages its 8 fold models).

    Every bag is built with TabArena's own single-model bagged wrapper (``AGSingleBagWrapper``) through
    ``AGModelBagExperiment``, so its inner folds, bagging and preprocessing are those TabArena applies to a model
    on the same benchmark; ``arena`` selects that benchmark's setup (see ``ARENA_SETUPS``).

    Init hyperparameters:

    * ``arena`` -- ``"tabarena"`` (default) or ``"beyondarena"``.
    * ``bag_time_limit`` -- time limit in seconds of each bag (all its folds). ``None`` (default) splits the
      split's ``time_limit`` evenly over the 26 bags, or no limit when there is none.
    * ``path`` -- where the bags write their artifacts; ``None`` uses a temp dir removed in :meth:`cleanup`.

    Codebase of TabPrep: https://github.com/atschalz/tabprep
    """

    uses_ray = False  # folds are fitted sequentially in-process

    def __init__(
        self,
        *,
        arena: str = "tabarena",
        bag_time_limit: float | None = None,
        path: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if arena not in ARENA_SETUPS:
            raise ValueError(f"arena must be one of {sorted(ARENA_SETUPS)}, got {arena!r}")
        self.arena = arena
        self.bag_time_limit = bag_time_limit
        self.path = path
        self._bags: list = []
        self._weights: np.ndarray | None = None
        self._root: str | None = None
        self._fit_report: dict[str, Any] = {}

    def _make_bag(
        self,
        config: dict[str, Any],
        *,
        problem_type: str,
        eval_metric: Scorer,
        validation_metadata: ValidationMetadata,
        num_cpus: int | None,
        memory_limit: float | None,
        bag_time_limit: float | None,
    ):
        """One config's ``AGSingleBagWrapper``, configured the way TabArena configures a bagged model."""
        experiment = make_bag_experiment(
            config,
            arena=self.arena,
            bag_time_limit=bag_time_limit,
            num_cpus=num_cpus,
            memory_limit=memory_limit,
            artifact_path=str(Path(self._root) / config["name"]),
        )
        # The steps of ``Experiment.init_method_kwargs`` that need no task object: the task's validation
        # metadata, the protocol for the AutoGluon wrapper, the preprocessing pipeline, detected resources.
        setup = ARENA_SETUPS[self.arena]
        method_kwargs = copy.deepcopy(experiment.method_kwargs)
        # The benchmark already applied its per-split column permutation to the system's input.
        method_kwargs.pop("shuffle_features", None)
        method_kwargs = experiment._apply_validation_metadata(method_kwargs, task_metadata=validation_metadata)
        method_kwargs["validation_protocol"] = setup.protocol
        method_kwargs = experiment._apply_preprocessing(method_kwargs)
        method_kwargs = experiment._apply_resources(method_kwargs)
        return experiment.method_cls(problem_type=problem_type, eval_metric=eval_metric, **method_kwargs)

    def _fit_system(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        target_name: str,
        problem_type: str,
        eval_metric: Scorer,
        validation_metadata: ValidationMetadata,
        num_cpus: int | None,
        num_gpus: int | None,
        memory_limit: float | None,
        time_limit: float | None,
        random_state: int | None,
    ):
        """Fit the 26 bags on all training rows, then the ensemble weights on their out-of-fold predictions.

        The configs carry their own seeds, so ``random_state`` is not used. See ``ExternalSystemModel._fit_system``
        for the argument contract.
        """
        del target_name, num_gpus, random_state
        configs = load_configs()
        bag_time_limit = self.bag_time_limit
        if bag_time_limit is None and time_limit is not None:
            bag_time_limit = time_limit / len(configs)
        self._root = self.path or tempfile.mkdtemp(prefix="tabarena_lgbm_prep_fte25_")
        oof, y_oof, internal_problem_type = [], None, None
        for config in configs:
            bag = self._make_bag(
                config,
                problem_type=problem_type,
                eval_metric=eval_metric,
                validation_metadata=validation_metadata,
                num_cpus=num_cpus,
                memory_limit=memory_limit,
                bag_time_limit=bag_time_limit,
            )
            bag.fit(X, y)
            artifact = bag.get_oof()
            oof.append(np.asarray(artifact["pred_proba_dict_val"], dtype=np.float64))
            labels = np.asarray(artifact["y_val"])
            if y_oof is None:
                y_oof, internal_problem_type = labels, artifact["problem_type_transform"]
            elif not np.array_equal(labels, y_oof):
                raise RuntimeError(f"{config['name']}: out-of-fold labels differ between bags")
            self._bags.append(bag)
        self._weights = ensemble_weights(oof, y_oof, problem_type=internal_problem_type, metric=eval_metric)
        self._fit_report = {
            "arena": self.arena,
            "bag_time_limit": bag_time_limit,
            "weights": {config["name"]: float(w) for config, w in zip(configs, self._weights, strict=True)},
        }
        return self

    def _selected(self):
        return [(bag, w) for bag, w in zip(self._bags, self._weights, strict=True) if w > 0]

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        total = None
        for bag, w in self._selected():
            proba = bag.predict_proba(X) * w
            total = proba if total is None else total + proba
        return total

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        if self.problem_type != "regression":
            proba = self._predict_proba(X)
            return proba.idxmax(axis=1)
        total = None
        for bag, w in self._selected():
            pred = bag.predict(X) * w
            total = pred if total is None else total + pred
        return total

    def get_metadata(self) -> dict:
        """The ensemble weights per config, the arena setup and the bag time limit."""
        return dict(self._fit_report)

    def cleanup(self):
        # Only remove a directory we created: an explicit `path` belongs to the caller.
        if self._root and self.path is None:
            shutil.rmtree(self._root, ignore_errors=True)
