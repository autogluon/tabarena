from __future__ import annotations

from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel


class CausiloModel(AbstractTorchModel):
    """Pretrained tabular classification and regression by Nums AI Inc.

    Code and documentation: https://github.com/nums-ai/causilo
    Code license: Apache-2.0. Weights have a separate research license:
    https://huggingface.co/nums-ai/causilo/blob/main/LICENSE
    """

    ag_key = "TA-CAUSILO"
    ag_name = "Causilo"
    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_resources_physical_cores_only = True
    default_num_gpus = 1
    minimum_num_gpus = 1
    _default_auxiliary_params_extra = {"valid_raw_types": ["int", "float", "category"]}
    _default_ag_args_ensemble_extra = {"fold_fitting_strategy": "sequential_local", "refit_folds": True}

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
        if engine.state.caches is not None:
            column_dtype = stage_dtype(engine.task, Stage.COLUMN, target)
            prediction_dtype = stage_dtype(engine.task, Stage.PREDICTION, target)
            caches = tuple(
                replace(
                    cache,
                    columns=tuple(transfer_state(c, target, dtype=column_dtype) for c in cache.columns),
                    prediction=transfer_state(cache.prediction, target, dtype=prediction_dtype),
                )
                for cache in engine.state.caches
            )
            engine.state = replace(engine.state, caches=caches)
        engine.device = target
        engine.device_request = str(target)
        self.model.device = str(target)

    def _more_tags(self):
        return {"can_refit_full": True}


def prefetch_weights():
    from causilo.checkpoints import RELEASE_COMMIT, REPOSITORY
    from huggingface_hub import snapshot_download

    snapshot_download(
        REPOSITORY,
        revision=RELEASE_COMMIT,
        allow_patterns=[
            "classifier/config.json",
            "classifier/model.safetensors",
            "regressor/config.json",
            "regressor/model.safetensors",
        ],
    )
