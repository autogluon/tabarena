from __future__ import annotations

from typing import ClassVar

from autogluon.core.models.abstract import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel


class CausiloModel(AbstractTorchModel):
    """Pretrained tabular classification and regression by Nums AI Inc.

    Code and documentation: https://github.com/nums-ai/causilo
    Code license: Apache-2.0. Weights have a separate research license:
    https://huggingface.co/nums-ai/causilo/blob/main/LICENSE

    Pinned to causilo 1.0.2: from that release on, a label set wider than the ten-class head is fit
    through the library's own error-correcting output codes (one context per code row over the same
    checkpoint), so the wrapper declares no ``max_classes`` cap. 1.0.0 rejected such datasets.
    """

    ag_key = "TA-CAUSILO"
    warmup_modules: ClassVar[tuple[str, ...]] = (
        "causilo",
        "causilo.engine",
        "causilo.execution.memory",
        "causilo.execution.precision",
        "causilo.serialization",
    )
    ag_name = "Causilo"
    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_resources_physical_cores_only = True
    default_num_gpus = 1
    minimum_num_gpus = 1
    _default_auxiliary_params_extra = {"valid_raw_types": ["int", "float", "category"]}
    _default_ag_args_ensemble_extra = {"fold_fitting_strategy": "sequential_local", "refit_folds": True}
    #: ``Engine.fit`` loads the network through ``causilo.checkpoints.load_pretrained_model(task)`` and
    #: moves it to the engine's device; one build per task and device per process. The loader
    #: has no device input, so ``_fit`` records the device on ``self.device`` before the fit.
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader="causilo.checkpoints:load_pretrained_model", key=("task",)
    )
    #: Knobs that make the warm-up's dummy fit cheap without touching the network.
    cheap_hyperparameters: ClassVar[dict] = {"n_estimators": 1}

    def _fit(self, X, y, X_val=None, y_val=None, time_limit=None, num_cpus=1, num_gpus=0, **kwargs):
        import torch
        from causilo import CausiloClassifier, CausiloRegressor

        # Context fitting has no iterative training, early stopping or validation split.
        params = self._get_model_params().copy()
        requested_device = params.pop("device", "auto")
        if requested_device == "cpu":
            device = "cpu"
        else:
            device = self._resolve_fit_device(
                num_gpus=num_gpus, gpu_device="cuda" if requested_device == "auto" else requested_device
            )
        cls = CausiloRegressor if self.problem_type == "regression" else CausiloClassifier
        self._inference_threads = num_cpus
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(num_cpus)
            self.device = device  # keys the shared network; the library's loader names no device
            self.model = cls(device=device, **params)
            self.model.fit(self.preprocess(X), y)
        finally:
            torch.set_num_threads(previous_threads)

    def _predict_proba(self, X, **kwargs):
        import torch

        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(self._inference_threads)
            return super()._predict_proba(X, **kwargs)
        finally:
            torch.set_num_threads(previous_threads)

    def _set_default_params(self):
        self._set_default_param_value("n_estimators", 8)

    def get_device(self):
        return str(self.model._engine.device)

    def _set_device(self, device):
        from dataclasses import replace

        from causilo.engine import resolve_device
        from causilo.execution.memory import Stage
        from causilo.execution.precision import stage_dtype
        from causilo.serialization import transfer_state

        engine = self.model._engine
        target = resolve_device(str(device))
        engine.model.to(target)
        state = engine.state
        if state.caches is not None or state.code_caches is not None:
            column_dtype = stage_dtype(engine.task, Stage.COLUMN, target)
            prediction_dtype = stage_dtype(engine.task, Stage.PREDICTION, target)

            def moved(caches):
                return tuple(
                    replace(
                        cache,
                        columns=tuple(transfer_state(c, target, dtype=column_dtype) for c in cache.columns),
                        prediction=transfer_state(cache.prediction, target, dtype=prediction_dtype),
                    )
                    for cache in caches
                )

            if state.caches is not None:
                state = replace(state, caches=moved(state.caches))
            if state.code_caches is not None:  # one cache tuple per output-code row of a many-class fit
                state = replace(state, code_caches=tuple(moved(caches) for caches in state.code_caches))
            engine.state = state
        engine.device = target
        engine.device_request = str(target)
        self.model.device = str(target)

    def _more_tags(self):
        return {"can_refit_full": True}


def prefetch_weights() -> str:
    """Download the pinned Causilo checkpoints into the Hugging Face cache; return the snapshot folder.

    The returned path lets the node staging and the SkyPilot seeding copy the repository directory,
    including the hub's ``trees`` listing that ``snapshot_download`` of a pinned commit needs offline.
    """
    from causilo.checkpoints import RELEASE_COMMIT, REPOSITORY
    from huggingface_hub import snapshot_download

    return snapshot_download(
        REPOSITORY,
        revision=RELEASE_COMMIT,
        allow_patterns=[
            "classifier/config.json",
            "classifier/model.safetensors",
            "regressor/config.json",
            "regressor/model.safetensors",
        ],
    )
