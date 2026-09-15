from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

from tabarena.models._shared_estimators import check_payload_device
from tabarena.models._shared_weights_model import CheckpointSpec, SharedWeightsModelMixin, SharedWeightsSpec

if TYPE_CHECKING:
    import pandas as pd

    from tabarena.models._weights import WeightsKey


logger = logging.getLogger(__name__)

#: Hugging Face repository of the TabFM v1.0.0 PyTorch checkpoints (``tabfm.src.pytorch.tabfm_v1_0_0.HF_REPO_ID``).
HF_REPO_ID = "google/tabfm-1.0.0-pytorch"
#: Pinned snapshot commit. It is the commit ``refs/main`` of the repository points at, so the pin
#: changes no bytes; a commit hash also lets huggingface_hub skip its ``repo_info`` request.
HF_REVISION = "77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83"
#: Checkpoint folder inside the repository per AutoGluon problem type.
MODEL_TYPES: dict[str, str] = {"binary": "classification", "multiclass": "classification", "regression": "regression"}
#: File name of the weights inside a checkpoint folder.
WEIGHTS_FILENAME = "model.safetensors"
#: Values of the wrapper-only ``device`` hyperparameter that force a GPU fit.
_GPU_DEVICE_VALUES = ("gpu", "cuda")


def _resolve_device(device: str | None, num_gpus: int, *, cuda_available: bool) -> str:
    """Resolve the torch device a TabFM fit should run on.

    ``device`` is the wrapper-only hyperparameter (``None``, ``"cpu"``, ``"gpu"`` or ``"cuda"``);
    ``num_gpus`` is what AutoGluon allocated for the fit. Returns ``"cuda"`` or ``"cpu"``.

    ``None`` derives the device from ``num_gpus`` (GPU when one was allocated). An explicit GPU
    request (``"gpu"`` or ``"cuda"``), or ``None`` with an allocated GPU, raises ``AssertionError``
    when ``cuda_available`` is False rather than silently falling back to CPU.
    """
    if device is not None:
        device = str(device).lower()
    if device == "cpu":
        return "cpu"
    want_gpu = device in _GPU_DEVICE_VALUES or (device is None and bool(num_gpus))
    if want_gpu and not cuda_available:
        raise AssertionError(
            "TabFM fit requested a GPU, but torch reports no CUDA device. Install a "
            "CUDA-enabled torch build and ensure a GPU is visible, or set device='cpu'.",
        )
    return "cuda" if want_gpu else "cpu"


def _shared_device_rule(hyperparameters: dict, allocated: str) -> str | None:
    """The device type a fit shares its network on, from the wrapper-only ``device`` hyperparameter.

    Mirrors :func:`_resolve_device`: ``"cpu"`` forces the CPU, ``"gpu"`` / ``"cuda"`` force a GPU,
    ``None`` keeps the allocation. A CPU fit shares only when the configuration asks for the CPU:
    without that request a CPU device means the job allocated no GPU, so the child is skipped by
    AutoGluon (``minimum_num_gpus=1``) or the fit is a deliberate CPU debug run, and priming a
    bfloat16 copy of the 6.2 GB checkpoint on the CPU (minutes, about 10 GB of RSS) would be wasted.
    Such a fit loads through the library instead.
    """
    requested = hyperparameters.get("device")
    if requested is not None:
        requested = str(requested).lower()
    if requested == "cpu":
        return "cpu"
    if requested in _GPU_DEVICE_VALUES:
        return "cuda"
    return "cuda" if allocated == "cuda" else None


def _checkpoint(model_type: str) -> CheckpointSpec:
    """The pinned snapshot folder of one checkpoint (``config.json`` plus the weights), which ``tabfm.load`` reads."""
    return CheckpointSpec(
        repo_id=HF_REPO_ID,
        revision=HF_REVISION,
        kind="snapshot",
        subfolder=model_type,
        required_files=(f"{model_type}/config.json", f"{model_type}/{WEIGHTS_FILENAME}"),
    )


def model_type_for(problem_type: str) -> str:
    """The checkpoint folder (``"classification"`` or ``"regression"``) a problem type uses."""
    try:
        return MODEL_TYPES[problem_type]
    except KeyError:
        raise AssertionError(f"Unsupported problem_type: {problem_type}") from None


def _build_tabfm_estimator(*, problem_type: str, device: str, interface: str, base_model=None, **hps):
    """Construct (but do not fit) a TabFM sklearn-style estimator for ``problem_type``.

    The single place both the AutoGluon wrapper (:class:`TabFMModel`) and the system model
    (:class:`~tabarena.systems.tabfm_plus.system.TabFMPlusSystemModel`) build a TabFM estimator, so the
    two never drift. ``interface`` selects the estimator's construction preset:

    * ``"default"``: the plain ``TabFMClassifier`` / ``TabFMRegressor`` constructor.
    * ``"ensemble"``: the ``.ensemble(...)`` preset (square-root feature-cross / SVD schedules,
      NNLS-weighted blending, probability averaging, per-problem calibration).

    ``interface`` is validated before any network is touched, so an invalid value fails fast.
    ``device`` is the resolved torch device (``"cuda"`` / ``"cpu"``, see :func:`_resolve_device`);
    the estimator runs where its network lives. ``base_model`` is a pre-loaded network to wrap
    (the registry payload of a sharing fit), which must already live on ``device``. Without it the
    library loads the network itself (``tabfm.load`` with its own process cache, downloading from
    Hugging Face on a cold cache). The network bounds its own peak activation memory via always-on
    internal chunking, so large tasks need no wrapper-side handling. Remaining ``hps`` are
    forwarded to the estimator (both the plain constructor and ``.ensemble`` accept the same
    keywords, e.g. ``random_state``).
    """
    if interface not in ("default", "ensemble"):
        raise ValueError(f"Unknown TabFM interface {interface!r}; expected 'default' or 'ensemble'.")

    from tabfm import TabFMClassifier, TabFMRegressor, tabfm_v1_0_0_pytorch

    model_type = model_type_for(problem_type)
    model_cls = TabFMRegressor if model_type == "regression" else TabFMClassifier

    if base_model is None:
        base_model = tabfm_v1_0_0_pytorch.load(model_type=model_type, device=device)
    else:
        # The estimator runs where the network's parameters live, so a mismatch would silently run
        # the fit elsewhere than the protocol allocated.
        check_payload_device(base_model, device)

    factory = model_cls.ensemble if interface == "ensemble" else model_cls
    return factory(model=base_model, **hps)


class TabFMModel(SharedWeightsModelMixin, AbstractTorchModel):
    """TabFM: a tabular foundation model that predicts via in-context learning.

    TabFM is a pre-trained PyTorch model: at inference time it is shown the
    training data as context and predicts on the test rows without any per-dataset
    gradient training. It handles mixed numerical/categorical columns and missing
    values natively (via its own internal preprocessing pipeline), so the
    AutoGluon-side preprocessing is left as a no-op and the typed DataFrame is
    passed straight through.

    The bfloat16 network of the pinned checkpoint is shared through the weights registry (see
    :mod:`tabarena.models._shared_weights_model`) and handed to the estimator constructor's
    ``model=`` argument; the estimator reads its device from the network's parameters.

    Accepts an optional ``device`` hyperparameter: ``None`` (default) selects a GPU
    when AutoGluon allocated one and CPU otherwise, ``"cpu"`` forces CPU execution,
    and ``"gpu"``/``"cuda"`` requires a GPU. A CPU fit shares its network only when the
    configuration asks for the CPU explicitly (see :func:`_shared_device_rule`).

    Paper: TabFM (Tabular Foundation Model)
    Authors: Google Research
    Codebase: https://github.com/google-research/tabfm
    License: Apache-2.0

    Install (PyTorch backend):
        pip install "tabfm[pytorch] @ git+https://github.com/google-research/tabfm.git"
    """

    ag_key = "TA-TABFM"
    ag_name = "TA-TabFM"
    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 1
    # refit_folds avoids storing one in-context model per fold (each carries the
    # full training context), refitting a single model on all data instead.
    _default_ag_args_ensemble_extra: ClassVar[dict] = {"refit_folds": True}

    #: Modules the timed fit would otherwise import for the first time: the library (which pulls in
    #: its estimators and the PyTorch loader), the Hub client the resolver uses and the safetensors
    #: reader huggingface_hub imports lazily.
    warmup_modules: ClassVar[tuple[str, ...]] = (
        "tabfm",
        "tabfm.src.pytorch.tabfm_v1_0_0",
        "huggingface_hub",
        "huggingface_hub.errors",
        "safetensors.torch",
    )
    #: Cheapness knob for the warm-up dummy fit; the ensemble size never touches the network, so
    #: the primed key is unaffected.
    warmup_dummy_fit_hyperparameters: ClassVar[dict] = {"n_estimators": 1}

    shared_weights_spec: ClassVar[SharedWeightsSpec] = SharedWeightsSpec(
        library="tabfm",
        checkpoint={model_type: _checkpoint(model_type) for model_type in sorted(set(MODEL_TYPES.values()))},
        variant=MODEL_TYPES,
        dtype="bfloat16",  # ``tabfm.load`` casts the float32 checkpoint to bfloat16
        device_from_params=_shared_device_rule,
        network_attr="model",
    )

    @classmethod
    def _build_shared_weights(cls, key: WeightsKey):
        """The network for ``key`` built from the local checkpoint directory exactly as ``tabfm.load`` does.

        ``tabfm.load`` with a local ``checkpoint_path`` applies ``<dir>/config.json`` to the
        constructor, loads ``model.safetensors`` strictly on the CPU, casts to bfloat16, moves to
        ``key.device`` and calls ``eval()``; the module is bit-identical to the one the library's
        remote branch builds from the same snapshot. ``use_cache=False`` keeps the library's own
        process cache empty so the process holds exactly one copy, in TabArena's registry.
        """
        import torch
        from tabfm import tabfm_v1_0_0_pytorch

        return tabfm_v1_0_0_pytorch.load(
            key.variant,
            checkpoint_path=key.checkpoint,
            device=key.device,
            dtype=torch.bfloat16,
            use_cache=False,
        )

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        import torch

        # `random_state` is injected by AutoGluon via `seed_name`; both the
        # classifier and the regressor accept it (and TabFM's other knobs default
        # sensibly), so the remaining params are forwarded as-is. `device` is a
        # wrapper-only knob (the TabFM estimators take their device from the
        # network's parameters), so it is popped here.
        hps = self._get_model_params()
        device = _resolve_device(
            hps.pop("device", None),
            num_gpus,
            cuda_available=torch.cuda.is_available(),
        )

        key, payload = self._acquire_shared_weights(device=device)
        self.model = _build_tabfm_estimator(
            problem_type=self.problem_type, device=device, interface="default", base_model=payload, **hps
        )

        # Does nothing (TabFM handles categoricals/missing natively); kept for
        # future preprocessing extensions and parity with the other wrappers.
        X = self.preprocess(X, y=y)
        self.model = self.model.fit(X=X, y=y)

    # TODO: support memory estimate! Implementing `_estimate_memory_usage_static` is all it
    #  takes; AutoGluon derives the capability from its presence.

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
