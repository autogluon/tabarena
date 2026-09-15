from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

from tabarena.models._shared_estimators import check_payload_device
from tabarena.models._shared_weights_model import SharedWeightsModelMixin, SharedWeightsSpec
from tabarena.models._weights import normalize_device

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tabarena.models._shared_weights_model import ResolvedCheckpoint
    from tabarena.models._weights import WeightsKey


def _fit_device_type(hyperparameters: dict, allocated: str) -> str:
    """The key's device type: ``device="cpu"`` in the config forces the CPU, anything else follows the allocation.

    ``"auto"`` (the default) and an explicit CUDA device follow the allocated resources, which is
    what ``_fit`` does through ``_resolve_fit_device``; only the type matters, TabArena runs one GPU
    per job.
    """
    return "cpu" if hyperparameters.get("device", "auto") == "cpu" else allocated


class CausiloModel(SharedWeightsModelMixin, AbstractTorchModel):
    """Pretrained tabular classification and regression by Nums AI Inc.

    Code and documentation: https://github.com/nums-ai/causilo
    Code license: Apache-2.0. Weights have a separate research license:
    https://huggingface.co/nums-ai/causilo/blob/main/LICENSE
    """

    ag_key = "TA-CAUSILO"
    #: Modules the fit, the loader and the restore path import lazily. ``import causilo`` already
    #: pulls in the engine, serialization, execution and checkpoint modules; they are listed so the
    #: declared set is closed. The estimator module is TabArena's own and imports the library.
    warmup_modules: ClassVar[tuple[str, ...]] = (
        "causilo",
        "causilo.checkpoints",
        "causilo.engine",
        "causilo.execution.memory",
        "causilo.execution.precision",
        "causilo.serialization",
        "huggingface_hub",
        "safetensors",
        "tabarena.models.causilo._estimators",
    )
    #: Cheapness knob for the warm-up dummy fit: one ensemble member. ``n_estimators`` only sets how
    #: many permutations the context holds and never touches the network or the primed key.
    warmup_dummy_fit_hyperparameters: ClassVar[dict] = {"n_estimators": 1}
    ag_name = "Causilo"
    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_resources_physical_cores_only = True
    default_num_gpus = 1
    minimum_num_gpus = 1
    _default_auxiliary_params_extra = {"valid_raw_types": ["int", "float", "category"]}
    _default_ag_args_ensemble_extra: ClassVar[dict] = {"refit_folds": True}

    shared_weights_spec: ClassVar[SharedWeightsSpec] = SharedWeightsSpec(
        library="causilo",
        # The coordinates (repository, pinned release commit) live in ``causilo.checkpoints``;
        # ``_resolve_shared_checkpoint`` and ``prefetch_weights`` read them from the library.
        checkpoint=None,
        device_from_params=_fit_device_type,
        # The checkpoint is fp32 and the library never casts its parameters (precision comes from
        # autocast); ``requires_grad`` stays as the library leaves it because frozen parameters
        # select other CPU kernels and move the predictions in the last bits.
        freeze_parameters=False,
        network_attr="_engine.model",
        detach_attr="library",
        device_attrs=(("_engine.device", "torch"), ("_engine.device_request", "str"), ("device", "str")),
    )

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
        from tabarena.models.causilo._estimators import resolve_checkpoint

        return resolve_checkpoint(variant, allow_download=allow_download)

    @classmethod
    def _build_shared_weights(cls, key: WeightsKey):
        from tabarena.models.causilo._estimators import build_network

        return build_network(key)

    @classmethod
    def prefetch_weights(cls) -> list[str]:
        """Make both task checkpoints present at the library's pinned release commit; returns the folder paths."""
        from tabarena.models.causilo._estimators import prefetch_checkpoints

        return prefetch_checkpoints()

    def _attach_shared_weights(self, payload: Any, device: str) -> None:
        """Give the fitted estimator an engine running on ``payload``; the fitted caches follow the device."""
        import torch

        device_type = normalize_device(device)
        check_payload_device(payload, device_type)
        self.model.attach_network(payload, device=torch.device(device_type))
        self._apply_device_bookkeeping(device_type)

    def _move_owned_network(self, device: str) -> None:
        """Move an estimator that loaded its own network in place, its fitted K/V caches included."""
        super()._move_owned_network(device)
        engine = self.model._engine
        state = getattr(engine, "state", None)
        if state is None or getattr(state, "caches", None) is None:
            return
        from dataclasses import replace

        import torch

        from tabarena.models.causilo._estimators import caches_to_device

        target = torch.device(normalize_device(device))
        engine.state = replace(state, caches=caches_to_device(state.caches, engine.task, target))

    def _fit(self, X, y, X_val=None, y_val=None, time_limit=None, num_cpus=1, num_gpus=0, **kwargs):
        import torch

        from tabarena.models.causilo._estimators import estimator_cls

        # Context fitting has no iterative training, early stopping or validation split.
        params = self._get_model_params().copy()
        requested_device = params.pop("device", "auto")
        if requested_device == "cpu":
            device = "cpu"
        else:
            device = self._resolve_fit_device(
                num_gpus=num_gpus, gpu_device="cuda" if requested_device == "auto" else requested_device
            )
        key, payload = self._acquire_shared_weights(device=device)
        cls = estimator_cls(self.shared_weights_spec.variant_for(self.problem_type), shared=key is not None)
        self._inference_threads = num_cpus
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(num_cpus)
            self.model = cls(device=device, **params)
            if key is not None:
                self.model.attach_network(payload)
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

    def _more_tags(self):
        return {"can_refit_full": True}
