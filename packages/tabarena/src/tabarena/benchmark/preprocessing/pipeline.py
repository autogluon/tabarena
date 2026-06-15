"""Shared resolution of a named preprocessing pipeline.

A *pipeline* (``"default"`` or ``"tabarena_default"``) is two independent pieces:

* a **model-agnostic feature generator** (an AutoGluon feature generator class), and
* a **model-specific** step that injects ``ag.model_specific_feature_generator_kwargs`` into a
  single model's hyperparameters (consumed by the AutoGluon model's own ``fit``).

Both the validation path (``AGWrapper`` -> ``TabularPredictor.fit(feature_generator=...)``) and
the no-validation path (``AGModelWrapper`` -> ``AbstractExecModel`` preprocessing) resolve the
pipeline through :func:`resolve_preprocessing_pipeline` so the two stay functionally aligned, and
both instantiate the feature generator through :func:`build_feature_generator`.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from autogluon.features import AbstractFeatureGenerator

    from tabarena.benchmark.task.metadata.schema import GroupLabelTypes


def _identity_model_specific(hyperparameters: dict) -> dict:
    return hyperparameters


@dataclass(frozen=True)
class PreprocessingPipeline:
    """The resolved pieces of a named preprocessing pipeline.

    ``feature_generator_cls`` / ``feature_generator_kwargs`` describe the model-agnostic
    feature generator; ``apply_model_specific`` injects the model-specific preprocessing into a
    single model's hyperparameters (identity for pipelines that have none).
    """

    feature_generator_cls: type[AbstractFeatureGenerator]
    feature_generator_kwargs: dict = field(default_factory=dict)
    apply_model_specific: Callable[[dict], dict] = _identity_model_specific


def resolve_preprocessing_pipeline(name: str | None) -> PreprocessingPipeline:
    """Resolve a pipeline name to its model-agnostic generator + model-specific transform.

    Supported names:

    * ``None`` / ``"default"`` — AutoGluon's standard ``AutoMLPipelineFeatureGenerator`` and no
      model-specific step (the historical default for both wrapper families).
    * ``"tabarena_default"`` — ``TabArenaModelAgnosticPreprocessing`` plus
      ``TabArenaModelSpecificPreprocessing.add_to_hyperparameters``.
    """
    from autogluon.features import AutoMLPipelineFeatureGenerator

    if name is None or name == "default":
        return PreprocessingPipeline(feature_generator_cls=AutoMLPipelineFeatureGenerator)
    if name == "tabarena_default":
        from tabarena.benchmark.preprocessing import (
            TabArenaModelAgnosticPreprocessing,
            TabArenaModelSpecificPreprocessing,
        )

        return PreprocessingPipeline(
            feature_generator_cls=TabArenaModelAgnosticPreprocessing,
            apply_model_specific=TabArenaModelSpecificPreprocessing.add_to_hyperparameters,
        )
    raise ValueError(
        f"Preprocessing pipeline name {name!r} not recognized; expected 'default' or 'tabarena_default'.",
    )


def build_feature_generator(
    feature_generator_cls: type[AbstractFeatureGenerator],
    feature_generator_kwargs: dict | None = None,
    *,
    group_cols: str | list[str] | None = None,
    group_labels: GroupLabelTypes | None = None,
    group_time_on: str | None = None,
) -> AbstractFeatureGenerator:
    """Instantiate ``feature_generator_cls``, forwarding only the group/time params it accepts.

    The group/time columns are a task property; a generator that doesn't take them (e.g. the
    plain ``AutoMLPipelineFeatureGenerator``) simply doesn't receive them. Shared by the
    validation path (``AGWrapper``, which sources the group params from its validation metadata)
    and the no-validation path (which passes them explicitly, or not at all).
    """
    kwargs = dict(feature_generator_kwargs or {})
    accepted = inspect.signature(feature_generator_cls.__init__).parameters
    group_params = {
        "group_cols": group_cols,
        "group_labels": group_labels,
        "group_time_on": group_time_on,
    }
    kwargs.update({k: v for k, v in group_params.items() if k in accepted})
    return feature_generator_cls(**kwargs)
