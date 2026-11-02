from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from autogluon.core.constants import BINARY, MULTICLASS
from autogluon.core.models.abstract import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd

#: Hugging Face commits of the released checkpoints, pinned so a push to a repository never changes the
#: benchmarked weights. Keys are neuralk-oss's model names.
HF_REVISIONS: dict[str, str] = {
    "seldon": "b113e1cebe6748538ddcd992e060aec5de772b27",
    "seldon-mini": "28507308fd67e3bb1796682294cb8ea3aeab4eec",
}


def to_signed_integers(X: pd.DataFrame) -> pd.DataFrame:
    """``X`` with its ``uint16`` / ``uint32`` / ``uint64`` columns cast to ``int64`` (``float64`` above its range).

    As in the Kumo Tabular wrapper: sdm keeps the unsigned dtype for the category values of such columns
    when it infers them as categorical, and torch cannot unpickle those tensors. Values are unchanged.
    """
    casts = {}
    for column, dtype in X.dtypes.items():
        if dtype.kind == "u" and dtype.itemsize > 1:
            fits = dtype.itemsize < 8 or X[column].max() <= np.iinfo(np.int64).max
            casts[column] = "int64" if fits else "float64"
    return X.astype(casts) if casts else X


@dataclass
class FittedSeldon:
    """The shared network next to the fitted classifier, whose estimator holds it only during a call.

    AutoGluon's shared-weights pickle finds the network by identity in the fitted state but treats a
    ``torch.nn.Module`` as one unit, and the library's estimator (``SeldonICL``) is a module holding the
    network as a submodule. Kept here, the network is left out of the pickle and put back from the
    registry on load; :meth:`classifier` attaches it before the library runs.
    """

    network: Any
    classifier: Any


def _attached(fitted: FittedSeldon):
    """The fitted classifier with the shared network attached to its estimator."""
    fitted.classifier.estimator.model = fitted.network
    return fitted.classifier


def _detach(fitted: FittedSeldon) -> None:
    fitted.classifier.estimator.model = None


class SeldonModel(AbstractTorchModel):
    """Seldon: Neuralk-AI's pretrained in-context-learning tabular foundation model.

    A thin pass-through to ``neuralk_oss.SeldonClassifier``: ``from_pretrained(<name>)`` builds the
    classifier with the released model's packaged config (ensemble size, per-member preprocessing
    recipe, context size, float16 autocast on CUDA), ``fit`` keeps the training table as context and
    ``predict_proba`` classifies the query rows against it. Nothing is added on top of the library:
    the wrapper only hands it AutoGluon's preprocessed frame, the fit's seed and device.

    Codebase: https://github.com/Neuralk-AI/neuralk_oss (weights: https://huggingface.co/Neuralk-AI/seldon)
    Authors: Neuralk-AI
    License: code under Apache-2.0, weights under the Seldon Research and Evaluation License
    (non-commercial research and internal evaluation; gated on Hugging Face).

    An in-context-learning model without a training loop, so ``X_val`` / ``y_val`` and ``time_limit`` are
    unused, like in the other foundation-model wrappers.
    """

    ag_key = "TA-SELDON"
    ag_name = "TA-Seldon"
    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types: ClassVar[list[str]] = [BINARY, MULTICLASS]
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 1
    _default_ag_args_ensemble_extra: ClassVar[dict] = {
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": True,
    }
    warmup_modules: ClassVar[tuple[str, ...]] = ("sdm", "neuralk_oss")
    #: One network per checkpoint, revision and device per process, loaded by the library's own loader.
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader="neuralk_oss.model:Seldon.from_pretrained", key=("name_or_path", "revision")
    )
    #: The two models are registered separately; each owns its ``share_weights`` class setting.
    class_settings_per_subclass = True

    #: The released model this class runs (neuralk-oss's model name).
    model_name: ClassVar[str] = "seldon"

    def _fit(self, X: pd.DataFrame, y: pd.Series, num_cpus: int = 1, num_gpus: int = 0, **kwargs):
        import torch
        from neuralk_oss import Seldon, SeldonClassifier

        X = self.preprocess(X, y=y)
        device = self._resolve_fit_device(num_gpus)
        # What ``SeldonClassifier.from_pretrained(<name>)`` does, with the network loaded through the shared
        # loader: the packaged config's settings, then the constructor on the loaded network.
        network = Seldon.from_pretrained(
            f"{self.model_name}-classifier", revision=HF_REVISIONS[self.model_name], device=device
        )
        settings = {
            **_packaged_settings(self.model_name),
            "device": device,
            "random_state": self.random_seed if isinstance(self.random_seed, int) else 0,
        }
        classifier = SeldonClassifier(network, **settings)
        self.model = FittedSeldon(network=network, classifier=classifier)
        self._num_cpus = num_cpus
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(num_cpus)
            classifier.fit(X, y)
        finally:
            torch.set_num_threads(previous_threads)
            _detach(self.model)

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        import torch

        X = self.preprocess(X, **kwargs)
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(self._num_cpus)
            classifier = _attached(self.model)
            proba = classifier.predict_proba(X)
        finally:
            torch.set_num_threads(previous_threads)
            _detach(self.model)
        # A bagged fold can miss a class; its column stays 0.
        predictions = np.zeros((len(X), self.num_classes), dtype=np.float32)
        predictions[:, np.asarray(self.model.classifier.classes_, dtype=int)] = proba
        return self._convert_proba_to_unified_form(predictions)

    def _preprocess(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return to_signed_integers(super()._preprocess(X, **kwargs))

    def get_device(self) -> str:
        param = next(self.model.network.parameters(), None)
        return str(param.device) if param is not None else "cpu"

    def _set_device(self, device: str):
        """Move the network, the estimator and its fitted sdm state (context cache, fitted processors)."""
        import torch

        classifier = self.model.classifier
        classifier.device = torch.device(device)
        self.model.network.to(device)
        estimator = classifier.estimator
        estimator.to(device)
        cache = getattr(estimator, "_cache", None)
        if cache is not None:
            estimator._cache = cache.to(device)
            recipe = estimator._cache["recipe_execution"].recipe
            for processor in (recipe.features, recipe.target, recipe.output):
                processor.to(device)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    @classmethod
    def prefetch_weights(cls) -> list:
        """Download the pinned checkpoint into the Hugging Face cache; return its local directory."""
        from huggingface_hub import snapshot_download

        repo = {"seldon": "Neuralk-AI/seldon", "seldon-mini": "Neuralk-AI/seldon-mini"}[cls.model_name]
        return [
            snapshot_download(
                repo, revision=HF_REVISIONS[cls.model_name], allow_patterns=["config.json", "model.safetensors"]
            )
        ]


def _packaged_settings(model_name: str) -> dict:
    """The released model's packaged config (``neuralk_oss/configs/<name>.yaml``), as ``from_pretrained`` reads it.

    Developer fix: neuralk-oss 0.2.0 reads it with the private ``_read_config``; it could expose the settings
    publicly (or take a prebuilt network in ``from_pretrained``). Pinned with the library.
    """
    from neuralk_oss.inference.classifier import _CONFIGS, _read_config

    settings = _read_config(_CONFIGS / f"{model_name}.yaml")
    settings.pop("checkpoint", None)
    return settings


class SeldonMiniModel(SeldonModel):
    """Seldon-Mini: the small released checkpoint, through the same pass-through."""

    ag_key = "TA-SELDON-MINI"
    ag_name = "TA-Seldon-Mini"
    model_name = "seldon-mini"
