"""Per-model dataset-compatibility constraints.

A :class:`ModelConstraints` describes which dataset shapes a model can run on. It is a
property of the *experiment* (attach it via ``Experiment(model_constraints=...)`` or
``Experiment.set_model_constraints``; ``TabArenaExperimentBundle`` bakes them in at
build time) and is respected wherever jobs are enumerated or run (``build_jobs``,
``ExperimentBatchRunner.run_jobs``, the SLURM dispatch).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConstraints:
    """Per-model dataset-compatibility constraints.

    A constraint is "active" only when its corresponding field is set
    (non-`None`); unset fields impose no restriction. `regression_support`
    defaults to True — set False for classification-only models.
    """

    max_n_features: int | None = None
    max_n_samples_train_per_fold: int | None = None
    min_n_samples_train_per_fold: int | None = None
    max_n_classes: int | None = None
    regression_support: bool = True

    def applies(
        self,
        *,
        n_features: int,
        n_classes: int,
        n_samples_train_per_fold: int,
        problem_type: str | None = None,
    ) -> bool:
        """True if a dataset with these properties is compatible with the model.

        For regression datasets, `problem_type == "regression"` is the
        authoritative signal — `n_classes` from metadata can be 0/-1/None.
        """
        if problem_type == "regression" and not self.regression_support:
            return False
        if self.max_n_features is not None and n_features > self.max_n_features:
            return False
        if (
            self.max_n_samples_train_per_fold is not None
            and n_samples_train_per_fold > self.max_n_samples_train_per_fold
        ):
            return False
        if (
            self.min_n_samples_train_per_fold is not None
            and n_samples_train_per_fold < self.min_n_samples_train_per_fold
        ):
            return False
        return not (self.max_n_classes is not None and n_classes > self.max_n_classes)


# Shared constraints for model families (used by TabArenaExperimentBundle.DEFAULT_MODEL_CONSTRAINTS).
TABICL_CONSTRAINTS = ModelConstraints(
    max_n_samples_train_per_fold=100_000,
    max_n_features=500,
    regression_support=False,
)
TABPFNV2_CONSTRAINTS = ModelConstraints(
    max_n_samples_train_per_fold=10_000,
    max_n_features=500,
    max_n_classes=10,
)
