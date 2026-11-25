from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models.abstract import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

#: The Hugging Face repository of the TabSil weights (one multitask checkpoint for both tasks).
HF_REPO = "aiden-jeon/tabsil"
#: The commit the Hub tag ``v1`` points to, pinned so the checkpoint never changes with the
#: repository's branches or tags (the tag's own id, 41bf9bd3, stops resolving if the tag is
#: re-created). The weights (``model.safetensors``, LFS sha256 e03c8c32849a...) and ``config.json``
#: were last changed in 677da6a6. Bump deliberately, with a note on what changed.
HF_REVISION = "bc7c0d9b4970e010416408c51f8c4d1a2a66eccd"


class TabSilModel(AbstractTorchModel):
    """TabSil: a tabular foundation model for classification and regression, trained only on synthetic tables.

    One column-row transformer serves both tasks: a Fourier + ECDF cell encoder, six interleaved
    column (ISAB over the training rows) and row stages, and 24 row in-context-learning layers in
    which the test rows attend to the training rows only. Classification reads out through a
    retrieval decoder (up to 160 classes), regression through 999 quantiles. Prediction averages an
    ensemble of 8 preprocessing members (shuffled column and class order, a sample identifier
    column, outlier clipping, one-hot encoding of low-cardinality categoricals); each member sees at
    most 768 original columns, spread evenly over the members.

    ``fit`` trains no weights: every member fits its preprocessing on the training rows, and
    ``predict`` runs the training and test rows through the network together. It has no training
    loop, so ``X_val`` / ``y_val`` and ``time_limit`` are unused, as in the other in-context wrappers.

    Codebase: https://github.com/Aiden-Jeon/tabsil
    Weights: https://huggingface.co/aiden-jeon/tabsil
    License: Apache-2.0 (code and weights)
    """

    ag_key = "TA-TABSIL"
    ag_name = "TA-TabSil"
    ag_priority = 65
    seed_name = "random_state"

    _supported_problem_types: ClassVar[list[str]] = [BINARY, MULTICLASS, REGRESSION]
    #: The retrieval decoder's class cap (``tabsil.models.column_row_transformer.MAX_CLASSES``).
    _default_auxiliary_params_extra: ClassVar[dict] = {"max_classes": 160}
    #: One fold at a time, and a refit on the full data instead of the bag: same quality for an
    #: in-context model, one network at inference.
    _default_ag_args_ensemble_extra: ClassVar[dict] = {
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": True,
    }
    default_resources_physical_cores_only = True
    default_num_gpus = 1
    minimum_num_gpus = 1
    gpu_strongly_recommended = True

    warmup_modules: ClassVar[tuple[str, ...]] = (
        "tabsil",
        "tabsil.hub",
        "tabsil.recipe",
        "tabsil.inference.ensemble",
        "tabsil.models.column_row_transformer",
    )
    #: ``tabsil.hub.from_pretrained`` builds the network (on the CPU) and returns it with the
    #: regression borders; the estimators take the built network. One build per checkpoint and
    #: device per process. The loader names no device, so ``_fit`` records it on ``self.device``.
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader="tabsil.hub:from_pretrained",
        key=("repo_id_or_path", "revision"),
    )
    #: Knobs that make the warm-up's dummy fit cheap without touching the network.
    cheap_hyperparameters: ClassVar[dict] = {"n_estimators": 1}

    _categorical_indices: list[int] | None = None
    _num_cpus: int = 1

    def _preprocess(self, X: pd.DataFrame, *, is_train: bool = False, **kwargs) -> pd.DataFrame:
        """AutoGluon's preprocessing with infinite values as missing; on the training rows, also record the categorical column indices.

        The network treats a non-finite cell as missing, but the members' outlier bounds are the
        mean and standard deviation of the training rows, which one infinite value turns into NaN,
        so the whole column would become missing. Mapping infinities to NaN first follows the
        network's rule on the training and the test rows alike.
        """
        X = super()._preprocess(X, **kwargs)
        numeric = X.select_dtypes(include="number").columns
        if len(numeric) and np.isinf(X[numeric].to_numpy(dtype=np.float64)).any():
            X = X.copy()
            X[numeric] = X[numeric].replace([np.inf, -np.inf], np.nan)
        if is_train:
            categorical = X.select_dtypes(include=["category"]).columns
            self._categorical_indices = [X.columns.get_loc(col) for col in categorical] or None
        return X

    def _fit(self, X: pd.DataFrame, y: pd.Series, num_cpus: int = 1, num_gpus: int = 0, **kwargs):
        import tabsil.hub
        from tabsil import TabSilClassifier, TabSilRegressor

        device = self._resolve_fit_device(num_gpus=num_gpus)
        self.device = device  # keys the shared network; the loader names no device
        self._num_cpus = num_cpus
        X = self.preprocess(X, y=y, is_train=True)

        with self._torch_threads():  # a registry miss builds the network inside the timed fit
            network, borders = tabsil.hub.from_pretrained(HF_REPO, revision=HF_REVISION)
        params = {k: v for k, v in self._get_model_params().items() if v is not None}
        params.update(device=device, categorical_features_indices=self._categorical_indices)
        if self.problem_type == REGRESSION:
            self.model = TabSilRegressor(network, borders, **params)
        else:
            self.model = TabSilClassifier(network, **params)
        self.model.fit(X, y.to_numpy())
        self._compact_training_rows()

    def _compact_training_rows(self) -> None:
        """Developer fix (tabsil 0.1.0): keep the members' encoded training rows in float32.

        ``EnsembleClassifier`` / ``EnsembleRegressor.fit`` keep one float64 copy of the encoded
        training rows per member (``X_trains_``, NumPy, so ``model.pkl`` holds all of them: 8x the
        frame on float64 tables, 64x on int8 tables, about 157x on categorical tables after one-hot),
        and AutoGluon pickles the model inside the timed fit. ``_iter_inputs`` concatenates them with
        the test rows and casts the result to float32, so storing them in float32 halves the pickle and
        leaves every prediction bit-identical. The library could keep them in float32 itself, or store
        the encoded matrix once and build each member's rows in ``_iter_inputs`` (8x smaller).
        """
        self.model.X_trains_ = [x.astype(np.float32, copy=False) for x in self.model.X_trains_]

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)
        with self._torch_threads():
            if self.problem_type == REGRESSION:
                return self.model.predict(X)
            proba = self.model.predict_proba(X)
        # A fold can miss a class; put each column at its label's position.
        out = np.zeros((len(X), self.num_classes), dtype=np.float32)
        out[:, np.asarray(self.model.classes_, dtype=int)] = proba
        return self._convert_proba_to_unified_form(out)

    @contextlib.contextmanager
    def _torch_threads(self):
        """Run torch work (the network build, the forward passes) on the fit's CPU budget.

        Restores torch's thread count afterwards.
        """
        import torch

        previous = torch.get_num_threads()
        torch.set_num_threads(max(1, int(self._num_cpus)))
        try:
            yield
        finally:
            torch.set_num_threads(previous)

    def _set_default_params(self):
        # ``None`` keeps the defaults saved with the weights (``tabsil.recipe.default_inference``):
        # 8 members, at most 768 original columns per member, spread ``balanced``.
        for param in ("n_estimators", "softmax_temperature", "max_features_per_member", "feature_subsampling"):
            self._set_default_param_value(param, None)

    def get_device(self) -> str:
        return self.model.device.type

    def _set_device(self, device: str):
        import torch

        self.model.device = torch.device(device)
        self.model.model.to(device)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    @classmethod
    def _estimate_memory_usage_static(cls, *, X: pd.DataFrame, **kwargs) -> int:
        """The network (about 0.9 GB in float32) and its buffers, plus the members' copies of the training rows.

        Each of the 8 members keeps its preprocessed training rows (float32, see
        ``_compact_training_rows``), and prediction stacks them with the test rows in float64; 20x
        the frame covers both.
        """
        baseline = 4e9
        return int(baseline + 20 * get_approximate_df_mem_usage(X).sum())

    @classmethod
    def prefetch_weights(cls) -> list[Path]:
        """Download the pinned weights and config into the Hugging Face cache and return their paths."""
        from pathlib import Path

        from huggingface_hub import hf_hub_download

        return [
            Path(hf_hub_download(HF_REPO, filename, revision=HF_REVISION))
            for filename in ("model.safetensors", "config.json")
        ]
