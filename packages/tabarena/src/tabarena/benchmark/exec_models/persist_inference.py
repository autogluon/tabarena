"""The untimed inference bracket shared by the AutoGluon-backed exec models.

A served deployment predicts from a model that is resident in memory with its weights on the
inference device. The exec models reproduce that around the timed predict: ``pre_predict`` calls
:func:`persist_for_inference` (persist the fitted models, load a bag's children, run every
persisted object's optional ``prepare_for_inference`` hook), the predict timer then measures the
served state, and ``cleanup`` calls :func:`release_after_inference` (unpersist, collect garbage,
return cached CUDA blocks). The served models stay resident through the post-evaluate consumers
(method metadata, OOF and bag artifacts) so those reuse the served bag instead of reloading it and
every child from disk.

Memory guard
    ``predictor.persist(models="best", max_memory=...)`` is AutoGluon's own guard: when the models
    exceed that fraction of the available RAM nothing is persisted and inference falls back to
    on-demand disk loads. ``max_memory=None`` skips the check (the single-model wrappers do this,
    their memory is bounded by construction). The check sizes every model by pickling it, which
    for a foundation model serialises the weights; see ``AGWrapper.persist_max_memory``.

Recorded outcome
    :class:`InferencePersistence` records what happened so the untimed share of the measurement
    stays auditable in the method metadata: ``persisted_models`` is ``None`` when persist was
    disabled, did not run or failed, ``[]`` when the memory guard skipped it, and the persisted
    names otherwise; ``prepared_for_inference`` lists the persisted objects whose
    ``prepare_for_inference`` ran without error. The hook's contract is written once, in
    ``AbstractExecModel.pre_predict``.
"""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


@dataclass
class InferencePersistence:
    """Outcome of one :func:`persist_for_inference` call.

    Attributes:
        persisted_models: Names of the models resident during the timed predict. ``None`` when
            persist was disabled, never ran or failed (inference then loads from disk); ``[]`` when
            persist ran but AutoGluon's memory guard skipped it.
        prepared_models: Names of the persisted objects (bags and their loaded children) whose
            ``prepare_for_inference`` hook ran without error. ``None`` when persist did not run.
    """

    persisted_models: list[str] | None = None
    prepared_models: list[str] | None = None

    def as_metadata(self, *, persist: bool) -> dict:
        """The method-metadata block: ``persist``, ``persisted_models`` and ``prepared_for_inference``."""
        return {
            "persist": persist,
            "persisted_models": self.persisted_models,
            "prepared_for_inference": self.prepared_models,
        }


def persist_for_inference(predictor, *, max_memory: float | None) -> InferencePersistence:
    """Persist the predictor's best model (with its ancestors) and run the untimed inference prep.

    Calls ``predictor.persist(models="best", max_memory=max_memory)`` (AutoGluon's default
    ``with_ancestors=True`` applies, so a stacked predictor persists the base models its ensemble
    reads), loads any child of a persisted bag that is still a path string, and dispatches
    ``prepare_for_inference`` to every persisted object (see :func:`dispatch_prepare_for_inference`
    for the error isolation).

    When ``persist`` returns an empty list because every model was already resident (a trainer
    running with ``low_memory=False``), the resident models are recorded instead so the outcome
    stays accurate; that fallback lists every resident model, not only the best one and its
    ancestors. When the memory guard skipped persisting, the trainer holds no models and the
    outcome stays ``[]``.

    Any exception is logged at warning level and answered with an empty
    :class:`InferencePersistence` after a best-effort ``unpersist``, so ``persisted_models=None``
    never coexists with partially resident models; inference then predicts from disk as it would
    without the bracket.

    Args:
        predictor: A fitted ``TabularPredictor``.
        max_memory: AutoGluon's ``max_memory`` fraction, or ``None`` to skip the memory check.

    Returns:
        The recorded outcome.
    """
    try:
        persisted = list(predictor.persist(models="best", max_memory=max_memory))
        trainer = predictor._trainer
        if not persisted:
            persisted = list(trainer.models)
        objects = list(iter_persisted_model_objects(trainer, persisted))
        prepared = dispatch_prepare_for_inference(objects)
    except Exception as exc:
        logger.warning(f"Persisting the model for untimed inference prep failed; predicting from disk. ({exc!r})")
        unpersist_after_inference(predictor)
        return InferencePersistence()
    return InferencePersistence(persisted_models=persisted, prepared_models=prepared)


def iter_persisted_model_objects(trainer, model_names: Iterable[str]) -> Iterator[object]:
    """Yield the resident model object for each name, followed by a bag's loaded children.

    A bagged ensemble whose children are still path strings (persist short-circuited because the
    bag itself was already resident) gets ``persist_child_models()`` called first, inside its own
    try/except, so the children are objects before the hooks run. Names that are not resident in
    ``trainer.models`` are skipped.
    """
    for name in model_names:
        model = trainer.models.get(name)
        if model is None:
            continue
        children = getattr(model, "models", None)
        if children and hasattr(model, "persist_child_models") and any(isinstance(c, str) for c in children):
            try:
                model.persist_child_models()
            except Exception as exc:
                logger.warning(f"Loading the children of {name!r} for inference prep failed. ({exc!r})")
        yield model
        for child in getattr(model, "models", None) or []:
            if not isinstance(child, str):
                yield child


def dispatch_prepare_for_inference(models: Iterable[object]) -> list[str]:
    """Call ``prepare_for_inference()`` on every object that declares it; return the names that succeeded.

    Each hook runs inside its own try/except: a failure is logged at warning level with the
    exception text (so a later failure inside the timed predict is attributable) and the remaining
    objects are still prepared. A model without the hook is skipped. The contract the hook has to
    honour is documented on ``AbstractExecModel.pre_predict``.
    """
    prepared: list[str] = []
    for model in models:
        prepare = getattr(model, "prepare_for_inference", None)
        if not callable(prepare):
            continue
        name = getattr(model, "name", type(model).__name__)
        try:
            prepare()
        except Exception as exc:
            logger.warning(f"prepare_for_inference of {name!r} failed; predicting without inference prep. ({exc!r})")
            continue
        prepared.append(name)
    return prepared


def unpersist_after_inference(predictor) -> None:
    """Best-effort ``predictor.unpersist()``; a ``None`` predictor and any exception are tolerated."""
    if predictor is None:
        return
    try:
        predictor.unpersist()
    except Exception as exc:
        logger.warning(f"Unpersisting the served models failed. ({exc!r})")


def free_inference_memory() -> None:
    """Collect garbage and, when torch is imported and CUDA is available, return cached CUDA blocks.

    ``empty_cache`` is wrapped in its own try/except: a sticky CUDA fault from the fit must not mask
    the exception that is being propagated when this runs from a failure-path cleanup.
    """
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        logger.warning(f"torch.cuda.empty_cache failed during cleanup. ({exc!r})")


def release_after_inference(predictor) -> None:
    """Unpersist the served models, then free host and device memory (:func:`free_inference_memory`).

    Called from an exec model's ``cleanup`` after the post-evaluate consumers ran, not from
    ``post_predict``, so the served bag is still resident for metadata and bag-artifact collection.
    """
    unpersist_after_inference(predictor)
    free_inference_memory()
