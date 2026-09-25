from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, ClassVar

from autogluon.features import LabelEncoderFeatureGenerator
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)

#: The published binary checkpoint, Apache-2.0 and ungated, pinned to a **commit** rather than
#: a branch. A moving reference would let the weights change underneath a recorded leaderboard
#: number, which is the one thing a benchmark entry cannot tolerate.
HF_REPO_ID = "kabartay/fintfm-binary"
HF_REVISION = "f116bfd43a2b15c65ed3551ea8c38e3364629ddc"
HF_FILENAME = "v4-cellattn-labels.pt"


def prefetch_weights() -> str:
    """Download the pinned binary checkpoint into the Hugging Face cache; return its path.

    Declared in ``info.py`` so the benchmark harness can stage weights before a run rather
    than downloading inside the first fit, where the transfer would be charged to that fit's
    time limit.
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        revision=HF_REVISION,
    )


class FinTFMModel(AbstractTorchModel):
    """fintfm: a synthetic-prior tabular foundation model for corporate credit risk.

    An in-context classifier pretrained only on synthetic tasks -- a structural generative
    model of company balance sheets and P&L, plus a generic random-graph structural causal
    model -- and never on real tabular data. Prediction is a single forward pass with the
    user's table supplied as context; there are no gradient updates at fit time.

    Two architectural points differ from the published TFM generation. Cells carry a random
    per-task column identity, and cell-level attention runs across rows *within* a feature
    before pooling, which together let a row representation express column-specific rules
    that a pooled symmetric encoder provably cannot.

    Categorical columns are encoded as out-of-fold smoothed target statistics rather than
    label-encoded, because the model reads every cell as an ordered scalar and an arbitrary
    code order is measurably worse than no order at all (fintfm ``docs/results/FINDINGS.md`` §100).

    Regression works by quantile-binning the target and de-binning the predicted
    distribution: a continuous target cut into K bins is an integer index over K outcomes, so
    the existing classification head does it with no architectural change. The output is
    natively distributional rather than a point estimate.

    **Capability limits are declared rather than worked around.** A checkpoint accepts at most
    ``max_features`` columns and ``max_classes`` classes; a task exceeding either raises
    instead of being silently truncated, so the benchmark records a skip rather than a
    meaningless score.

    Paper: none; the project's measurement log is its public record.
    Authors: Mukharbek Organokov
    Codebase: https://github.com/kabartay/fintfm
    License: Apache-2.0
    """

    #: Two names, two jobs. ``ag_key`` is the registry key and takes the ``TA-`` prefix every
    #: other entrant uses (``TA-ILTM``, ``TA-LIMIX``, ``TA-CAUSILO``); ``ag_name`` is the
    #: public name and does not, which is how ``TA-CAUSILO`` presents itself as ``Causilo``.
    #:
    #: Neither carries a version suffix. The leaderboard lists models rather than releases,
    #: and TabSTAR, OrionMSP, TabFlex and iLTM all appear unversioned. The version belongs in
    #: the changelog, where it can be attached to specific numbers.
    #:
    #: AutoGluon derives the stored config name from ``ag_name``, so the results cache is
    #: keyed by it and a run under a different name is not reused. ``ag_key`` is expected to
    #: match the raw data of the run being submitted (see ``causilo/info.py``).
    #: Imports charged to the untimed warm-up rather than to the first timed fit. ``torch`` and
    #: the fintfm package are the heavy ones; the checkpoint itself is staged separately by
    #: ``prefetch_weights``.
    warmup_modules: ClassVar[tuple[str, ...]] = ("torch", "fintfm")
    ag_key = "TA-FINTFM"
    ag_name = "FinTFM"
    ag_priority = 65
    seed_name = "random_state"

    #: **Binary only, deliberately.** Multiclass and regression are implemented
    #: (``FinancialTFMRegressor``, the ``max_classes=10`` checkpoint) and were scored on real
    #: data: multiclass ranks **93.4 of 95** over 7 datasets, regression **93.1 of 94** over
    #: 12 and is **last on 5 of them**. Both run correctly -- no crashes, no timeouts, correct
    #: units -- and neither is competitive, so declaring them would publish a capability whose
    #: quality the measurement says is not there. Declaring all three would raise coverage from
    #: 26/51 to 46/51; the coverage number is not worth the claim. See fintfm ``docs/results/FINDINGS.md``
    #: §121. Restore them here when the numbers justify it, not before.
    _supported_problem_types: ClassVar[list[str]] = ["binary"]

    #: Not a CUDA-only model: it runs on CPU, MPS and CUDA, and the reference machine has no
    #: CUDA at all. Requiring a GPU here would make every fit fail on that machine rather than
    #: run slowly, so the GPU attributes of the foundation-model template are deliberately
    #: omitted and the inherited default of zero GPUs stands.
    default_resources_physical_cores_only = True

    #: Foundation-model setting: an in-context model has no train loop, so after bagging a
    #: single refit on all the data is at parity with the bagged ensemble and far cheaper.
    _default_ag_args_ensemble_extra: ClassVar[dict] = {
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": True,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator: LabelEncoderFeatureGenerator | None = None
        self._categorical_encoder = None

    def _preprocess(self, X: pd.DataFrame, y: pd.Series | None = None, is_train: bool = False, **kwargs) -> np.ndarray:
        """Encode categoricals as out-of-fold target statistics and emit a dense float array.

        fintfm's cell embedding takes numeric values only: it z-scores each column against the
        context and embeds the scalar, so a string category has no representation. The first
        version of this wrapper label-encoded, which imposes an arbitrary total order on an
        unordered variable, and ``fintfm``'s ``docs/results/FINDINGS.md`` §100 measured what that cost
        — a 0.0894 ROC-AUC deficit on mostly-categorical datasets against 0.0320 on numeric
        ones, correlating -0.668 with log maximum cardinality.

        The replacement gives each level the **smoothed target rate among training rows
        carrying it**, which is ordered on the axis the model reads. Training rows are encoded
        out of fold so no row informs its own statistic; query rows use the full training
        statistics, since their labels cannot leak. See
        ``fintfm.inference.categorical`` for why the out-of-fold step is not optional.
        """
        import numpy as np
        from fintfm.inference.categorical import CategoricalTargetEncoder

        X = super()._preprocess(X, **kwargs)
        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator is not None and self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        arr = X.fillna(0).to_numpy(dtype=np.float32)

        if self._feature_generator is None or not self._feature_generator.features_in:
            return arr
        if os.environ.get("FINTFM_CATEGORICAL", "target") == "label":
            # The pre-§100 path, kept only so the two encodings can be A/B'd on one
            # checkpoint. §98's run did not record which checkpoint it used, so comparing
            # new numbers against its stored ones would confound the encoding change with a
            # possible checkpoint change.
            return arr
        if is_train:
            cat_idx = [X.columns.get_loc(c) for c in self._feature_generator.features_in]
            self._categorical_encoder = CategoricalTargetEncoder(
                categorical_features=cat_idx, random_state=self.random_seed
            )
            return self._categorical_encoder.fit_transform(arr, np.asarray(y))
        if self._categorical_encoder is None:
            raise AssertionError("predict called before fit: no categorical encoder")
        return self._categorical_encoder.transform(arr)

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        """Store the table as in-context evidence. No gradient steps are taken.

        ``X_val``/``y_val`` and ``time_limit`` are ignored, which is the documented behaviour
        for in-context foundation models: there is no train loop to stop early and no
        validation set to stop against.
        """
        import numpy as np
        import torch
        from fintfm.inference.classifier import FinancialTFMClassifier
        from fintfm.modeling.model import FinancialTFM

        params = self._get_model_params()
        # Binary and multiclass take **different checkpoints on purpose.** A multiclass
        # checkpoint is trained at max_classes=10 with the generic SCM prior mixed in, which
        # is a different model from the financial-prior binary one; using it for binary tasks
        # would silently change every binary number this project has published. Selecting by
        # problem type keeps the binary results comparable with §98/§101.
        multiclass_path = params.pop("model_path_multiclass", None) or os.environ.get("FINTFM_CHECKPOINT_MULTICLASS")
        # Regression takes its own checkpoint again: the binned head only means anything if
        # the prior emitted continuous targets during pretraining, which the binary and
        # multiclass checkpoints never did.
        regression_path = params.pop("model_path_regression", None) or os.environ.get("FINTFM_CHECKPOINT_REGRESSION")
        n_bins = int(params.pop("n_bins", 10))
        checkpoint = params.pop("model_path", None) or os.environ.get("FINTFM_CHECKPOINT")
        if self.problem_type == "multiclass" and multiclass_path:
            checkpoint = multiclass_path
        elif self.problem_type == "regression" and regression_path:
            checkpoint = regression_path
        if not checkpoint:
            # Fall back to the published binary weights so the model runs without setup.
            # An explicit ``model_path`` or ``FINTFM_CHECKPOINT`` takes precedence, which is
            # how the private multiclass and regression checkpoints are supplied.
            checkpoint = prefetch_weights()

        # Prefer any real accelerator. AutoGluon's GPU accounting is CUDA-only, so a machine
        # with Apple Silicon reports zero GPUs and would otherwise run this on CPU -- which is
        # 10-100x slower for this model and is the difference between completing a task inside
        # the harness's time budget and hitting TimeLimitExceeded.
        if num_gpus and torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        network = FinancialTFM.load(checkpoint, map_location=device)

        X = self.preprocess(X, y=y, is_train=True)
        if self.problem_type == "regression":
            self._fit_regression(X, np.asarray(y), network, device, n_bins, params)
            return
        n_classes = int(np.unique(np.asarray(y)).size)
        if n_classes > network.cfg.max_classes:
            raise AssertionError(
                f"fintfm checkpoint represents at most {network.cfg.max_classes} classes, "
                f"task has {n_classes}. Set FINTFM_CHECKPOINT_MULTICLASS to a checkpoint "
                f"trained at --max-classes {n_classes} or higher; scoring anyway would "
                f"measure a task the model cannot represent."
            )
        n_features = X.shape[1]
        if n_features > network.cfg.max_features:
            raise AssertionError(
                f"fintfm checkpoint accepts at most {network.cfg.max_features} features, "
                f"task has {n_features}. Truncating would score a different task, so this "
                f"raises instead."
            )

        # ``seed_name = "random_state"`` makes AutoGluon inject the seed into the params dict,
        # so it must not also be passed explicitly.
        params.setdefault("random_state", self.random_seed)
        self.model = FinancialTFMClassifier(network, device=device, **params).fit(X, np.asarray(y))

    def _fit_regression(self, X, y, network, device: str, n_bins: int, params: dict):
        """Store the table as context for the binned-regression wrapper.

        ``n_bins`` is clamped to the checkpoint's ``max_classes`` rather than raising,
        because unlike a class count it is *our* free parameter, not a property of the task:
        a 10-bin request against an 8-logit head is an over-ask by us, and the honest
        response is to use the 8 bins available and say so in the log. A class count that
        exceeds the head is a different thing entirely and still raises.
        """
        from fintfm.inference.regressor import FinancialTFMRegressor

        n_features = X.shape[1]
        if n_features > network.cfg.max_features:
            raise AssertionError(
                f"fintfm checkpoint accepts at most {network.cfg.max_features} features, "
                f"task has {n_features}. Truncating would score a different task, so this "
                f"raises instead."
            )
        bins = min(n_bins, network.cfg.max_classes)
        if bins < n_bins:
            logger.warning(
                "fintfm: n_bins reduced %d -> %d to fit the checkpoint's head; the "
                "prediction grid is coarser than requested.",
                n_bins,
                bins,
            )
        params.setdefault("random_state", self.random_seed)
        self.model = FinancialTFMRegressor(network, n_bins=bins, device=device, **params).fit(X, y)

    def _set_default_params(self):
        default_params = {
            # D12: a single column-identity draw is the configuration that decision's own
            # text calls incomplete, so ensemble over draws by default.
            "n_ensemble": 8,
            # docs/results/FINDINGS.md §83/§84 measured context as nearly flat on a 1M-row panel;
            # 1000 is the largest value that is cheap on every machine this runs on.
            "max_context": 1000,
            # §81: identity-preserving, and materially cheaper wherever attention scores are
            # materialised (CPU and MPS). No effect on CUDA (§95).
            "feature_chunk": 16,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def get_device(self) -> str:
        return str(next(self.model.model.parameters()).device)

    def _set_device(self, device: str):
        self.model.model.to(device)
        self.model.device = device

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
