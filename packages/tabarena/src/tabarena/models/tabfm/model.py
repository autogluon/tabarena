from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
from autogluon.core.constants import BINARY, MULTICLASS
from autogluon.core.models.abstract import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

from tabarena.models.cell_budget import gpu_cell_budget, rows_within_budget
from tabarena.utils.logging_utils import import_many_class_classifier, root_handlers_preserved

if TYPE_CHECKING:
    import torch


logger = logging.getLogger(__name__)

#: Peak GPU memory per training cell measured on BeyondArena (about 21M cells on 96 GB), the safety margin that
#: keeps the refit's fragmented reserved memory and the test batches out of trouble, and the library's
#: recommended minimum rows per ensemble member.
TABFM_BYTES_PER_CELL = 4700
TABFM_CELL_SAFETY = 0.7
TABFM_MIN_ROWS_PER_MEMBER = 5000


def _resolve_device(device: str | None, num_gpus: int, *, cuda_available: bool) -> str:
    """Resolve the torch device a TabFM fit should run on.

    ``device`` is the wrapper-only hyperparameter (``None``, ``"cpu"``, ``"gpu"``
    or ``"cuda"``); ``num_gpus`` is what AutoGluon allocated for the fit. Returns
    ``"cuda"`` or ``"cpu"``.

    ``None`` derives the device from ``num_gpus`` (GPU when one was allocated). An
    explicit GPU request (``"gpu"``/``"cuda"``) -- or ``None`` with an allocated
    GPU -- raises ``AssertionError`` when ``cuda_available`` is False rather than
    silently falling back to CPU.
    """
    if device is not None:
        device = str(device).lower()
    if device == "cpu":
        return "cpu"
    want_gpu = device in ("gpu", "cuda") or (device is None and bool(num_gpus))
    if want_gpu and not cuda_available:
        raise AssertionError(
            "TabFM fit requested a GPU, but torch reports no CUDA device. Install a "
            "CUDA-enabled torch build and ensure a GPU is visible, or set device='cpu'.",
        )
    return "cuda" if want_gpu else "cpu"


def _model_type(problem_type: str) -> str:
    """TabFM's checkpoint name for ``problem_type``: ``"classification"`` or ``"regression"``."""
    if problem_type in ["binary", "multiclass"]:
        return "classification"
    if problem_type == "regression":
        return "regression"
    raise AssertionError(f"Unsupported problem_type: {problem_type}")


def _load_tabfm_network(*, problem_type: str, device: str) -> torch.nn.Module:
    """Load the network of ``problem_type``'s TabFM checkpoint onto ``device``.

    Downloads the pre-trained PyTorch checkpoint from Hugging Face on first use (see
    :func:`prefetch_weights`); a no-op once cached. An estimator runs where its network lives. The
    network bounds its own peak activation memory via always-on internal chunking, so large tasks
    need no wrapper-side handling.

    ``load`` reports the download through the root ``logging`` functions, which install a
    ``StreamHandler`` on a root logger that has none; the call runs under
    :func:`~tabarena.utils.logging_utils.root_handlers_preserved`.
    """
    from tabfm import tabfm_v1_0_0_pytorch

    with root_handlers_preserved():
        return tabfm_v1_0_0_pytorch.load(model_type=_model_type(problem_type), device=device)


def _build_tabfm_estimator(*, problem_type: str, device: str, interface: str, network=None, **hps):
    """Construct (but do not fit) a TabFM sklearn-style estimator for ``problem_type``.

    The single place both the AutoGluon wrapper (:class:`TabFMModel`) and the system model
    (:class:`~tabarena.systems.tabfm_plus.system.TabFMPlusSystemModel`) build a TabFM estimator, so the
    two never drift. ``interface`` selects the estimator's construction preset:

    * ``"default"`` — the plain ``TabFMClassifier`` / ``TabFMRegressor`` constructor.
    * ``"ensemble"`` — the ``.ensemble(...)`` preset (square-root feature-cross / SVD schedules,
      NNLS-weighted blending, probability averaging, per-problem calibration).

    ``interface`` is validated before the (cached) checkpoint is loaded, so an invalid value fails
    fast without touching Hugging Face. ``device`` is the resolved torch device (``"cuda"`` /
    ``"cpu"``, see :func:`_resolve_device`); the loaded network is placed there and the estimator
    runs where its network lives. ``network`` is a network already loaded by
    :func:`_load_tabfm_network` to build on; ``None`` loads it. Remaining ``hps`` are forwarded to
    the estimator (both the plain constructor and ``.ensemble`` accept the same keywords, e.g.
    ``random_state``).
    """
    if interface not in ("default", "ensemble"):
        raise ValueError(f"Unknown TabFM interface {interface!r}; expected 'default' or 'ensemble'.")

    from tabfm import TabFMClassifier, TabFMRegressor

    model_cls = TabFMClassifier if _model_type(problem_type) == "classification" else TabFMRegressor
    if network is None:
        network = _load_tabfm_network(problem_type=problem_type, device=device)
    factory = model_cls.ensemble if interface == "ensemble" else model_cls
    return factory(model=network, **hps)


class _TabFMOutputCodeEstimator:
    """The base estimator of the ``ManyClassClassifier`` output coding: one ``TabFMClassifier`` per code row.

    ``ManyClassClassifier`` clones its base estimator for every row of its codebook, fits the clone on
    that row's symbols and reads ``predict_proba`` and ``classes_``. ``TabFMClassifier`` cannot be that
    estimator directly: ``sklearn.base.clone`` deep-copies constructor parameters and the loaded
    network is one, and the wrapper validates the frame into a NumPy array while TabFM's own
    preprocessing reads the column dtypes. This estimator holds the network with the training
    frame's columns and dtypes, clones by sharing them, and rebuilds the frame around every array it
    is given.
    """

    def __init__(self, *, network: torch.nn.Module, columns: pd.Index, dtypes: pd.Series, device: str, hps: dict):
        self._network = network
        self._columns = columns
        self._dtypes = dtypes
        self._device = device
        self._hps = hps

    def __sklearn_clone__(self) -> _TabFMOutputCodeEstimator:
        return _TabFMOutputCodeEstimator(
            network=self._network, columns=self._columns, dtypes=self._dtypes, device=self._device, hps=self._hps
        )

    def _frame(self, X) -> pd.DataFrame:
        return pd.DataFrame(X, columns=self._columns).astype(self._dtypes)

    def fit(self, X, y) -> _TabFMOutputCodeEstimator:
        self._estimator = _build_tabfm_estimator(
            problem_type=MULTICLASS, device=self._device, interface="default", network=self._network, **self._hps
        ).fit(X=self._frame(X), y=np.asarray(y))
        self.classes_ = self._estimator.classes_
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._estimator.predict_proba(self._frame(X))


class TabFMModel(AbstractTorchModel):
    """TabFM: a tabular foundation model that predicts via in-context learning.

    TabFM is a pre-trained PyTorch model: at inference time it is shown the
    training data as context and predicts on the test rows without any per-dataset
    gradient training. It handles mixed numerical/categorical columns and missing
    values natively (via its own internal preprocessing pipeline), so the
    AutoGluon-side preprocessing is left as a no-op and the typed DataFrame is
    passed straight through.

    Wraps ``AbstractTorchModel`` so AutoGluon manages device placement: the network
    is moved to CPU before being pickled and back onto the training device (when
    available) on load, via ``get_device`` / ``_set_device``.

    Accepts an optional ``device`` hyperparameter: ``None`` (default) selects a GPU
    when AutoGluon allocated one and CPU otherwise, ``"cpu"`` forces CPU execution,
    and ``"gpu"``/``"cuda"`` requires a GPU.

    The checkpoint's classification head is ten classes wide and the estimator rejects
    wider label sets. Above ``many_class_threshold`` (an ``ag_args_fit`` parameter, ten by
    default) the fit wraps a :class:`_TabFMOutputCodeEstimator` in the ``ManyClassClassifier``
    of tabpfn-extensions, which codes the labels over that many symbols and fits one TabFM
    estimator per code row on the same network.

    Above a cell budget derived from the GPU (``rows x columns`` of the training table, about 14M cells on a
    96 GB card; :mod:`tabarena.models.cell_budget`) the fit sets TabFM's ``max_num_rows`` so every ensemble
    member sub-samples its rows: the network embeds every training cell and ran out of memory at about 21M
    cells. The ``cell_budget`` hyperparameter overrides the derived budget.

    Paper: TabFM (Tabular Foundation Model)
    Authors: Google Research
    Codebase: https://github.com/google-research/tabfm
    License: Apache-2.0

    Install (PyTorch backend):
        pip install "tabfm[pytorch] @ git+https://github.com/google-research/tabfm.git"
    """

    ag_key = "TA-TABFM"
    warmup_modules: ClassVar[tuple[str, ...]] = ("tabfm", "huggingface_hub", "tabpfn_extensions.many_class")
    ag_name = "TA-TabFM"
    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 1
    #: The estimator takes the network ``tabfm_v1_0_0_pytorch.load`` returns; one build per model
    #: type, checkpoint path, dtype and device per process.
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader="tabfm.src.pytorch.tabfm_v1_0_0:load", key=("model_type", "checkpoint_path", "dtype")
    )
    #: Knobs that make the warm-up's dummy fit cheap without touching the network.
    cheap_hyperparameters: ClassVar[dict] = {"n_estimators": 1}
    # Set fold_fitting_strategy to sequential_local,
    # as parallel folding crashes if model weights aren't pre-downloaded.
    # refit_folds avoids storing one in-context model per fold (each carries the
    # full training context), refitting a single model on all data instead.
    _default_ag_args_ensemble_extra = {
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": True,
    }
    #: No ``max_classes`` cap: above ``many_class_threshold`` (the head's width) the fit output-codes the labels.
    _default_auxiliary_params_extra = {"max_classes": None, "many_class_threshold": 10}
    #: Whether ``self.model`` is the ``ManyClassClassifier`` around a :class:`_TabFMOutputCodeEstimator`.
    _use_many_class: bool = False

    def _ag_params(self) -> set[str]:
        return super()._ag_params() | {"many_class_threshold"}

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
        many_class_threshold = self.params_aux.get("many_class_threshold", 10)

        # Does nothing (TabFM handles categoricals/missing natively); kept for
        # future preprocessing extensions and parity with the other wrappers.
        X = self.preprocess(X, y=y)
        # Cell budget: above it every ensemble member sub-samples its rows (`max_num_rows`, TabFM 1.0.1).
        # `cell_budget` is a wrapper-only hyperparameter that overrides the GPU-derived budget (tests).
        cell_budget = hps.pop("cell_budget", None)
        if cell_budget is None:
            cell_budget = gpu_cell_budget(bytes_per_cell=TABFM_BYTES_PER_CELL, safety=TABFM_CELL_SAFETY)
        if cell_budget is not None and "max_num_rows" not in hps:
            n_rows = rows_within_budget(X.shape[0], X.shape[1], cell_budget, min_rows=TABFM_MIN_ROWS_PER_MEMBER)
            if n_rows < X.shape[0]:
                hps["max_num_rows"] = n_rows
                logger.log(
                    20,
                    f"\tTabFM: {X.shape[0]} x {X.shape[1]} training cells exceed the budget of {cell_budget} cells "
                    f"on this GPU; each ensemble member sub-samples {n_rows} rows (max_num_rows).",
                )
        self._use_many_class = (
            self.problem_type in [BINARY, MULTICLASS]
            and self.num_classes is not None
            and self.num_classes > many_class_threshold
        )
        if self._use_many_class:
            ManyClassClassifier = import_many_class_classifier()

            logger.log(
                20,
                f"\tTabFM: {self.num_classes} classes exceed the checkpoint's {many_class_threshold}-class head, "
                "fitting ManyClassClassifier (output coding) around it.",
            )
            base = _TabFMOutputCodeEstimator(
                network=_load_tabfm_network(problem_type=self.problem_type, device=device),
                columns=X.columns,
                dtypes=X.dtypes,
                device=device,
                hps=hps,
            )
            self.model = ManyClassClassifier(
                estimator=base, alphabet_size=many_class_threshold, random_state=hps.get(self.seed_name, 0)
            ).fit(X, y)
        else:
            self.model = _build_tabfm_estimator(
                problem_type=self.problem_type, device=device, interface="default", **hps
            ).fit(X=X, y=y)

    def _network(self) -> torch.nn.Module:
        """The fitted TabFM network, whichever of the two estimators ``self.model`` is."""
        return self.model.estimator._network if self._use_many_class else self.model.model

    def get_device(self) -> str:
        """Return the torch device of the fitted TabFM network."""
        param = next(self._network().parameters(), None)
        return str(param.device) if param is not None else "cpu"

    def _set_device(self, device: str):
        """Move the fitted TabFM network to ``device`` (the estimators follow it)."""
        self._network().to(device)

    # TODO: support memory estimate! Implementing `_estimate_memory_usage_static` is all it
    #  takes; AutoGluon derives the capability from its presence.

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}


def prefetch_weights() -> None:
    """Pre-download the TabFM v1.0.0 PyTorch checkpoint from Hugging Face.

    Warms the local cache (``google/tabfm-1.0.0-pytorch``) so parallel / offline
    fits do not race on the download.
    """
    from huggingface_hub import snapshot_download
    from tabfm.src.pytorch.tabfm_v1_0_0 import HF_REPO_ID

    snapshot_download(repo_id=HF_REPO_ID)
