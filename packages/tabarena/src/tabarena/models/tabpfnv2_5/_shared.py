"""The TabPFN family's preset of :class:`tabarena.models._shared_weights_model.SharedWeightsModelMixin`.

tabpfn's estimators accept a ``ModelSpecs`` container (the network, its architecture config, the
inference config and a regressor's criterion) as ``model_path``; ``tabpfn.base.initialize_tabpfn_model``
then returns the container's contents instead of reading a checkpoint. :class:`TabPFNSharedWeightsMixin`
drives that seam for TabPFN-3, TabPFN-2.5 / 2.6 and TabPFN-Wide:

* :data:`TABPFN_SPEC` is the spec the three wrappers derive theirs from (the ``models_`` network
  attribute, the configurations tabpfn mutates the module under).
* :meth:`TabPFNSharedWeightsMixin._resolve_shared_checkpoint` resolves the checkpoint the fit
  passes to tabpfn from the wrapper's ``checkpoint_param`` (a per-problem-type mapping or a
  ``[classifier, regressor]`` list), the class defaults and tabpfn's cache directory, downloading
  through tabpfn's own downloader when the fetch policy allows.
* :meth:`TabPFNSharedWeightsMixin._swap_in_shared_specs` and
  :meth:`TabPFNSharedWeightsMixin._finish_shared_fit` are the two lines a wrapper's ``_fit`` adds
  around the library call.
* The attach, detach and module hooks know the ``ManyClassClassifier`` wrapper that
  tabpfn_extensions nests around the base estimator for more than ten classes: the specs stay on
  the base estimator, whose predict-time clones share the network because the payload's
  ``__deepcopy__`` returns itself (see ``_estimators.py``).

Only :meth:`TabPFNSharedWeightsMixin._resolve_shared_checkpoint`, :meth:`TabPFNSharedWeightsMixin.prefetch_weights`
and the download helpers import tabpfn, inside the function; every other hook is library-free.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from tabarena.models._shared_estimators import check_payload_device, payload_modules
from tabarena.models._shared_weights_model import ResolvedCheckpoint, SharedWeightsModelMixin, SharedWeightsSpec
from tabarena.models._weights import normalize_device, shallow_copy
from tabarena.models.prefetch import WeightsUnavailableError

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

#: ``inference_precision`` values under which the network is never cast in place; a ``torch.dtype``
#: makes tabpfn's per-device cache call ``model.type(dtype)`` on the module.
SHAREABLE_INFERENCE_PRECISIONS: tuple[str, ...] = ("auto", "autocast")
#: The only ``fit_mode`` whose fit leaves the network untouched (tabpfn's default);
#: ``fit_with_cache`` writes the train-set representation into the module.
SHAREABLE_FIT_MODE = "fit_preprocessors"


def mutates_network(hyperparameters: Mapping[str, Any]) -> bool:
    """Whether tabpfn writes into or casts the module under these hyperparameters (then the fit loads its own)."""
    return (
        hyperparameters.get("fit_mode", SHAREABLE_FIT_MODE) != SHAREABLE_FIT_MODE
        or hyperparameters.get("inference_precision", "auto") not in SHAREABLE_INFERENCE_PRECISIONS
    )


#: The spec the TabPFN wrappers derive theirs from with ``dataclasses.replace``.
TABPFN_SPEC = SharedWeightsSpec(
    library="tabpfn",
    disable_when=("differentiable_input", "use_finetuning", mutates_network),
    # tabpfn takes a ``torch.dtype`` for a forced precision; any value outside the two strings casts.
    unshareable_examples=({"fit_mode": "fit_with_cache"}, {"inference_precision": "float16"}),
    network_attr="models_",
)


def tabpfn_checkpoint_source(checkpoint: str | Path, estimator_type: str) -> dict[str, Any]:
    """Provenance of a tabpfn checkpoint for the metadata: repository id and file name, never a host path.

    tabpfn downloads its checkpoints from per-version Hugging Face repositories
    (``tabpfn.model_loading.ModelSource``) into its own cache directory rather than the Hugging
    Face cache, so the repository is looked up from the version the file name encodes.
    """
    filename = Path(checkpoint).name
    version = repo_id = None
    try:
        from tabpfn.model_loading import ModelSource, resolve_model_version

        version = resolve_model_version(str(checkpoint)).value
        repo_id = getattr(ModelSource, f"get_{estimator_type}_{version.replace('.', '_')}")().repo_id
    except Exception:
        logger.debug("Could not resolve the tabpfn source of %s", filename, exc_info=True)
    return {"repo_id": repo_id, "filename": filename, "revision": None, "version": version}


def ensure_tabpfn_checkpoint(checkpoint: str | Path, estimator_type: str, *, allow_download: bool) -> Path:
    """Make sure a tabpfn checkpoint is on disk, downloading it exactly as tabpfn's own loader would.

    ``tabpfn.model_loading.load_model_criterion_config`` downloads a missing checkpoint through
    ``download_model`` before building the network. The shared-weights loader builds from the file
    only, so the download decision moves here, where the fetch policy gates it.

    Raises:
        WeightsUnavailableError: The file is missing and ``allow_download`` is False.
        RuntimeError: Every download source failed.
    """
    path = Path(checkpoint)
    if path.is_file():
        return path
    if not allow_download:
        raise WeightsUnavailableError(f"TabPFN checkpoint {path} is not cached locally")
    from tabpfn.model_loading import download_model, resolve_model_version

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading TabPFN checkpoint to %s", path)
    result = download_model(path, version=resolve_model_version(str(path)), which=estimator_type, model_name=path.name)
    if result != "ok":
        raise RuntimeError(f"Failed to download the TabPFN checkpoint {path.name} to {path}") from result[0]
    return path


def detach_fitted_estimator(estimator: Any, *, model_path: Any) -> Any:
    """A shallow copy of a fitted tabpfn estimator without its network; the live estimator keeps everything.

    ``models_`` becomes ``None`` and the inference engine copy loses its per-device model caches
    (the shape tabpfn's own ``save_fitted_tabpfn_model`` writes). ``model_path`` is set to
    ``model_path`` so a specs object never rides along, and a regressor's bar-distribution
    criteria are deep-copied to the CPU so the pickle holds no device tensors.
    """
    copy_ = shallow_copy(estimator)
    copy_.models_ = None
    executor = getattr(estimator, "executor_", None)
    if executor is not None:
        executor_copy = shallow_copy(executor)
        executor_copy.model_caches = None
        copy_.executor_ = executor_copy
    copy_.model_path = model_path
    for name in ("znorm_space_bardist_", "raw_space_bardist_"):
        criterion = getattr(estimator, name, None)
        if criterion is not None:
            setattr(copy_, name, copy.deepcopy(criterion).to("cpu"))
    return copy_


def attach_shared_specs(estimator: Any, specs: Any, device: str) -> None:
    """Point a fitted tabpfn estimator at the shared network of ``specs`` on device type ``device``.

    The sequence tabpfn's ``load_fitted_tabpfn_model`` uses: ``models_`` and the engine's model
    caches are rebuilt from the module, then ``estimator.to(device)`` refreshes the device
    bookkeeping. The module already lives on ``device`` (the registry key names it), so the
    engine's per-device cache finds it under its exact device key and moves nothing; a regressor's
    ``to`` also moves its own criterion copies.
    """
    estimator.models_ = [specs.model]
    executor = getattr(estimator, "executor_", None)
    if executor is not None:
        executor._set_models(estimator.models_)
    estimator.to(device)


class TabPFNSharedWeightsMixin(SharedWeightsModelMixin):
    """Shared-weights preset for wrappers around tabpfn's estimators; see the module docstring.

    A concrete wrapper declares ``shared_weights_spec`` (derived from :data:`TABPFN_SPEC`),
    ``checkpoint_param``, the two default checkpoint names and, when its checkpoints live outside
    tabpfn's cache, ``custom_model_dir``; its ``_fit`` wraps the library call in
    :meth:`_swap_in_shared_specs` and :meth:`_finish_shared_fit`.
    """

    #: Hyperparameter that overrides the checkpoint: a mapping from problem type (``"binary"``,
    #: ``"multiclass"``, ``"regression"``, or the ``"classification"`` umbrella) to a file, or a
    #: ``[classifier, regressor]`` list. A file is a bare name in tabpfn's cache directory
    #: (``custom_model_dir`` when set) or an absolute path.
    checkpoint_param: ClassVar[str | None] = None
    #: Directory holding the checkpoints instead of tabpfn's cache directory.
    custom_model_dir: ClassVar[str | None] = None
    default_classification_model: ClassVar[str | None] = None
    default_regression_model: ClassVar[str | None] = None
    #: Cheapness knob for the warm-up dummy fit; the ensemble size never touches the checkpoint or the key.
    warmup_dummy_fit_hyperparameters: ClassVar[dict] = {"n_estimators": 1}
    # Refitting is faster at inference and as good as the bag for an in-context model.
    _default_ag_args_ensemble_extra: ClassVar[dict] = {"refit_folds": True}

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # The ``model_path`` a fit passes to tabpfn without sharing; restored on the estimator after a shared fit.
        self._model_path_arg: Any = None

    def __setstate__(self, state: dict) -> None:
        state.setdefault("_model_path_arg", None)
        super().__setstate__(state)

    # --- checkpoint resolution -------------------------------------------------------------------

    @classmethod
    def _checkpoint_name(cls, *, problem_type: str, variant: str, hyperparameters: Mapping[str, Any]) -> str | None:
        """The checkpoint file name a fit of ``problem_type`` loads, or ``None`` for tabpfn's own default.

        A mapping override is read at the exact problem type, then the ``"classification"``
        umbrella for binary and multiclass, then the class default; a list override is read at the
        variant's position and taken as is.
        """
        override = hyperparameters.get(cls.checkpoint_param) if cls.checkpoint_param else None
        if isinstance(override, (list, tuple)):
            return override[0] if variant == "classifier" else override[1]
        name = None
        if override:
            name = override.get(problem_type)
            if name is None and variant == "classifier":
                name = override.get("classification")
        if name is None:
            name = cls.default_classification_model if variant == "classifier" else cls.default_regression_model
        return name

    @classmethod
    def _checkpoint_path(cls, *, problem_type: str, hyperparameters: Mapping[str, Any]) -> str | None:
        """The ``model_path`` string a fit passes to tabpfn, or ``None`` for tabpfn's own default.

        The name from :meth:`_checkpoint_name` under ``custom_model_dir`` or tabpfn's cache
        directory (``prepend_cache_path`` is a no-op for an absolute path).
        """
        variant = "classifier" if problem_type in ("binary", "multiclass") else "regressor"
        name = cls._checkpoint_name(problem_type=problem_type, variant=variant, hyperparameters=hyperparameters)
        if name is None:
            return None
        if cls.custom_model_dir is not None:
            return str(Path(cls.custom_model_dir) / name)
        from tabpfn.model_loading import prepend_cache_path

        return prepend_cache_path(name)

    @classmethod
    def _resolve_shared_checkpoint(
        cls,
        *,
        problem_type: str,
        variant: str,
        hyperparameters: Mapping[str, Any],
        allow_download: bool,
        stage: str = "fit",
    ) -> ResolvedCheckpoint | None:
        """The checkpoint of :meth:`_checkpoint_path`, on disk (downloaded when allowed), with tabpfn's Hub provenance."""
        del stage
        checkpoint = cls._checkpoint_path(problem_type=problem_type, hyperparameters=hyperparameters)
        if checkpoint is None:
            return None
        path = ensure_tabpfn_checkpoint(checkpoint, variant, allow_download=allow_download)
        return ResolvedCheckpoint(path=str(path.resolve()), source=tabpfn_checkpoint_source(path, variant))

    @classmethod
    def _build_shared_weights(cls, key):
        from tabarena.models.tabpfnv2_5._estimators import build_shared_model_specs

        return build_shared_model_specs(key.checkpoint, key.variant, key.device)

    @classmethod
    def prefetch_weights(cls) -> list[str]:
        """Download every tabpfn checkpoint into tabpfn's cache; returns the wrapper's two default checkpoint paths."""
        from tabpfn.model_loading import download_all_models, prepend_cache_path

        download_all_models(to=Path(prepend_cache_path("")))
        paths = [cls._checkpoint_path(problem_type=pt, hyperparameters={}) for pt in ("binary", "regression")]
        return [path for path in paths if path is not None]

    # --- the two lines a wrapper's _fit adds -----------------------------------------------------

    def _swap_in_shared_specs(self, hyperparameters: dict, payload: Any) -> dict:
        """``hyperparameters`` with the registry specs as ``model_path`` when ``payload`` is set; records the path it replaces."""
        if payload is None:
            return hyperparameters
        self._model_path_arg = hyperparameters.get("model_path")
        return {**hyperparameters, "model_path": payload}

    def _finish_shared_fit(self, payload: Any) -> None:
        """After a shared fit the estimator records the checkpoint path, not the specs, and owns its criterion.

        The path keeps the specs (and with them the network) out of ``get_params()`` and the
        pickle. tabpfn's fit assigns the shared regressor criterion to ``znorm_space_bardist_``;
        the deep copy decouples this estimator's pickle and later ``.to()`` calls from the registry
        object. A many-class wrapper keeps the specs on its base estimator: its predict-time clones
        of that estimator are what share the network.
        """
        if payload is None or self._many_class_wrapper() is not None:
            return
        self.model.model_path = self._model_path_arg
        criterion = getattr(self.model, "znorm_space_bardist_", None)
        if criterion is not None:
            self.model.znorm_space_bardist_ = copy.deepcopy(criterion)

    # --- many-class aware hooks -------------------------------------------------------------------

    def _many_class_wrapper(self) -> Any | None:
        """The ``ManyClassClassifier`` around the base estimator, or ``None`` when ``self.model`` is the estimator."""
        model = self.model
        return model if model is not None and hasattr(model, "estimator") and hasattr(model, "alphabet_size") else None

    def _base_estimator(self) -> Any:
        """The tabpfn estimator: ``self.model`` or the base estimator of a many-class wrapper."""
        wrapper = self._many_class_wrapper()
        return self.model if wrapper is None else wrapper.estimator

    def _network_attached(self) -> bool:
        wrapper = self._many_class_wrapper()
        if wrapper is None:
            return super()._network_attached()
        return getattr(wrapper.estimator.model_path, "model", None) is not None

    def _attach_shared_weights(self, payload: Any, device: str) -> None:
        """Point the estimator, or a many-class wrapper's base estimator and its fitted rows, at ``payload``."""
        device_type = normalize_device(device)
        check_payload_device(payload, device_type)
        wrapper = self._many_class_wrapper()
        if wrapper is None:
            attach_shared_specs(self.model, payload, device_type)
        else:
            wrapper.estimator.model_path = payload
            wrapper.estimator.device = device_type
            for row in getattr(wrapper, "estimators_", None) or []:
                attach_shared_specs(row, payload, device_type)
        self._apply_device_bookkeeping(device_type)

    def _detach_for_pickle(self, estimator: Any) -> Any:
        """The weightless copy: ``models_`` and the engine caches dropped, ``model_path`` the checkpoint path again."""
        wrapper = self._many_class_wrapper()
        if wrapper is None:
            return detach_fitted_estimator(estimator, model_path=self._model_path_arg)
        wrapper_copy = shallow_copy(wrapper)
        base = shallow_copy(wrapper.estimator)
        base.model_path = self._model_path_arg
        wrapper_copy.estimator = base
        rows = getattr(wrapper, "estimators_", None)
        if rows:
            wrapper_copy.estimators_ = [detach_fitted_estimator(row, model_path=self._model_path_arg) for row in rows]
        return wrapper_copy

    def _shared_modules(self):
        if not self._network_attached():
            return []
        wrapper = self._many_class_wrapper()
        if wrapper is None:
            return payload_modules(self.model.models_)
        return payload_modules(wrapper.estimator.model_path)

    def _move_owned_network(self, device: str) -> None:
        """Tabpfn's own ``to`` on the estimator that owns its network (the base estimator under a many-class wrapper)."""
        self._base_estimator().to(device)


__all__ = [
    "SHAREABLE_FIT_MODE",
    "SHAREABLE_INFERENCE_PRECISIONS",
    "TABPFN_SPEC",
    "TabPFNSharedWeightsMixin",
    "attach_shared_specs",
    "detach_fitted_estimator",
    "ensure_tabpfn_checkpoint",
    "mutates_network",
    "tabpfn_checkpoint_source",
]
