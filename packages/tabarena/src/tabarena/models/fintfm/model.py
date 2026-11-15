from __future__ import annotations

import copy
import json
import logging
import os
from typing import TYPE_CHECKING, ClassVar

from autogluon.features import LabelEncoderFeatureGenerator
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

from tabarena.models.fintfm.checkpoints import (
    BINARY_CHECKPOINT,
    MULTICLASS_CHECKPOINT_UNPUBLISHED,
    REGRESSION_CHECKPOINT_UNPUBLISHED,
    resolve_checkpoint,
    validate_checkpoint_ref,
)

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)

#: Marks an environment variable that is not set, since ``null`` is a valid JSON value.
_UNSET = object()

#: Hugging Face coordinates of the published binary checkpoint, kept under their earlier names.
HF_REPO_ID = BINARY_CHECKPOINT["repo_id"]
HF_REVISION = BINARY_CHECKPOINT["revision"]
HF_FILENAME = BINARY_CHECKPOINT["filename"]

#: Problem types a checkpoint, partition and probe can be set for.
PROBLEM_TYPES = ("binary", "multiclass", "regression")


def prefetch_weights() -> str:
    """Download the pinned binary checkpoint into the Hugging Face cache; return its path.

    Declared in ``info.py`` so the benchmark harness can stage weights before a run rather
    than downloading inside the first fit, where the transfer would be charged to that fit's
    time limit. The file's SHA-256 is checked against :data:`BINARY_CHECKPOINT`.
    """
    return resolve_checkpoint(BINARY_CHECKPOINT, problem_type="binary")


class FinTFMModel(AbstractTorchModel):
    """fintfm: a synthetic-prior tabular foundation model for corporate credit risk.

    An in-context classifier pretrained only on synthetic tasks and never on real tabular
    data. The published binary checkpoint (``v4-cellattn-labels.pt``) was trained on the
    financial prior alone, a structural generative model of company balance sheets and P&L
    (``--p-financial 1.0``, fintfm ``docs/results/FINDINGS.md`` §91/§104); the generic
    random-graph structural causal model is not in its training mixture. Prediction is a
    single forward pass with the user's table supplied as context; there are no gradient
    updates at fit time.

    Missing values reach the model as NaN. fintfm embeds every cell with a separate
    missingness mask, so imputing here would erase that signal and make a missing cell
    indistinguishable from a real zero. A missing categorical becomes AutoGluon's ``-1``
    category code, which the target encoder treats as a level of its own.

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
    meaningless score. The one opt-in exception is ``max_features_strategy="subsample"``
    (classification only), under which fintfm scores a wider table with an ensemble whose
    members each see a balanced round-robin subset of the columns, so every column is seen by
    at least one member. It is off in the default config and on in every config sampled from
    ``hpo.py``. On TabArena v0.1 it admits the three binary tasks wider than 136 features:
    APSFailure (170), kddcup09_appetency (212) and Bioresponse (1776).

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
    #:
    #: ``FINTFM_ALLOW_MULTICLASS=1`` adds multiclass for a local measurement run (fintfm
    #: DR7.4), with the checkpoint from ``FINTFM_CHECKPOINT_MULTICLASS``. It is read once, at
    #: import, so it must be set before the process starts; unset, the declaration is the
    #: submitted one. ``FINTFM_ALLOW_REGRESSION=1`` adds regression the same way, with the
    #: checkpoint from ``FINTFM_CHECKPOINT_REGRESSION`` (fintfm DR7.5, coverage of all 51
    #: datasets).
    _supported_problem_types: ClassVar[list[str]] = (
        ["binary"]
        + (["multiclass"] if os.environ.get("FINTFM_ALLOW_MULTICLASS") == "1" else [])
        + (["regression"] if os.environ.get("FINTFM_ALLOW_REGRESSION") == "1" else [])
    )

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

    #: The measured entry's settings, which the default config uses when no environment variable
    #: or hyperparameter changes them. A subclass overrides them to be a different entry.
    #:
    #: ``_entry_params`` are ``FinancialTFMClassifier`` arguments. The dicts keyed by problem
    #: type become the hyperparameters ``checkpoint_<type>``, ``partition_<type>`` and (in a
    #: class with a probe) ``probe_<type>``. ``None`` partitions nothing. Precedence: a
    #: hyperparameter passed explicitly, then an environment variable, then these. This class
    #: is the zero-shot head: no partition, and only the binary checkpoint is published.
    _entry_params: ClassVar[dict] = {}
    _entry_n_bins: ClassVar[int] = 10
    _entry_checkpoint: ClassVar[dict] = {
        "binary": BINARY_CHECKPOINT,
        "multiclass": MULTICLASS_CHECKPOINT_UNPUBLISHED,
        "regression": REGRESSION_CHECKPOINT_UNPUBLISHED,
    }
    _entry_partition: ClassVar[dict] = {"binary": None, "multiclass": None, "regression": None}
    #: Names of the per-problem-type options this class has hyperparameters for.
    _per_problem_options: ClassVar[tuple[str, ...]] = ("checkpoint", "partition")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator: LabelEncoderFeatureGenerator | None = None
        self._categorical_encoder = None
        self._minority_flipped = False

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

        Numeric NaN is passed through unchanged for the model's missingness mask. A missing
        categorical arrives as category code ``-1`` from the label encoder, a level distinct
        from every observed category, and is target-encoded like any other level.
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
        arr = X.to_numpy(dtype=np.float32, na_value=np.nan)

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
        entry = self._resolve_entry(params)
        n_bins = entry["n_bins"]
        # Binary and multiclass take **different checkpoints on purpose.** A multiclass
        # checkpoint is trained at max_classes=10 with the generic SCM prior mixed in, which
        # is a different model from the financial-prior binary one; using it for binary tasks
        # would silently change every binary number this project has published. Regression
        # takes its own checkpoint again: the binned head only means anything if the prior
        # emitted continuous targets during pretraining, which the others never did.
        checkpoint = resolve_checkpoint(entry["checkpoint"], problem_type=self.problem_type)

        # Prefer any real accelerator. AutoGluon's GPU accounting is CUDA-only, so a machine
        # with Apple Silicon reports zero GPUs and would otherwise run this on CPU -- which is
        # 10-100x slower for this model and is the difference between completing a task inside
        # the harness's time budget and hitting TimeLimitExceeded.
        forced = self._forced_device()
        if forced is not None:
            device = forced
        elif num_gpus and torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        network = FinancialTFM.load(checkpoint, map_location=device)

        X = self.preprocess(X, y=y, is_train=True)
        if self.problem_type == "regression":
            self._fit_regression(X, np.asarray(y), network, device, n_bins, params, entry["partition"])
            return
        n_classes = int(np.unique(np.asarray(y)).size)
        if n_classes > network.cfg.max_classes:
            raise AssertionError(
                f"fintfm checkpoint represents at most {network.cfg.max_classes} classes, "
                f"task has {n_classes}. Set checkpoint_multiclass (or FINTFM_CHECKPOINT_MULTICLASS) to a checkpoint "
                f"trained at --max-classes {n_classes} or higher; scoring anyway would "
                f"measure a task the model cannot represent."
            )
        n_features = X.shape[1]
        if n_features > network.cfg.max_features and params.get("max_features_strategy", "raise") not in (
            "subsample",
            "select",
        ):
            raise AssertionError(
                f"fintfm checkpoint accepts at most {network.cfg.max_features} features, "
                f"task has {n_features}. Truncating would score a different task, so this "
                f"raises instead. Pass max_features_strategy='subsample' (an ensemble over "
                f"column subsets) or 'select' (the top columns ranked on the training rows, "
                f"fintfm DR8.3) to score it."
            )

        # ``seed_name = "random_state"`` makes AutoGluon inject the seed into the params dict,
        # so it must not also be passed explicitly.
        params.setdefault("random_state", self.random_seed)
        y_fit = np.asarray(y)
        self._minority_flipped = self._flip_to_minority(y_fit)
        if self._minority_flipped:
            y_fit = 1 - y_fit
        classifier = FinancialTFMClassifier(network, device=device, **params)
        partition = entry["partition"]
        if partition is not None:
            from fintfm.inference.large_n import PartitionedTFMClassifier

            classifier = PartitionedTFMClassifier(classifier, random_state=params["random_state"], **partition)
        self.model = classifier.fit(X, y_fit)

    @staticmethod
    def _forced_device() -> str | None:
        """The device named by ``FINTFM_DEVICE``, or ``None`` to choose one automatically.

        Unset, the adapter prefers CUDA, then MPS, then CPU. On Apple Silicon, Metal's memory
        outside live tensors grows over many forward passes, and a fit with many of them (for
        example ``context_strategy="retrieval"``, one context per query group) can exhaust it
        within one dataset. ``FINTFM_DEVICE=cpu`` lets such a dataset rerun on CPU. Like
        ``FINTFM_PARAMS``, it does not change the config name.

        Returns:
            ``"cpu"``, ``"mps"`` or ``"cuda"`` when set, else ``None``.

        Raises:
            ValueError: the variable names another device.
        """
        raw = os.environ.get("FINTFM_DEVICE", "").strip()
        if not raw:
            return None
        if raw not in ("cpu", "mps", "cuda"):
            raise ValueError(f"FINTFM_DEVICE must be cpu, mps or cuda, got {raw!r}")
        logger.warning("fintfm: FINTFM_DEVICE set, running on %s", raw)
        return raw

    def _flip_to_minority(self, y: np.ndarray) -> bool:
        """Whether this fit codes the minority class as 1 (fintfm loss analysis, fix 2).

        AutoGluon codes a binary target's second sorted label as 1, so on some tasks the class
        the model treats as the positive outcome is the majority, while the financial prior
        always made class 1 the rarer one (at most 30% of rows). ``FINTFM_MINORITY_POSITIVE=1``
        swaps the codes when class 1 holds more than half of this fit's training rows, and
        :meth:`_predict_proba` swaps the probability columns back, so AutoGluon reads the same
        class it coded. Unset (the default), nothing changes. Read at fit time, so each bag
        fold decides from its own rows; like ``FINTFM_PARAMS`` it does not change the config
        name, so an arm needs its own results directory.

        Args:
            y: The fit's 0/1 target as AutoGluon coded it.

        Returns:
            ``True`` when the switch is on, the task is binary and class 1 is the majority.
        """
        import numpy as np

        if os.environ.get("FINTFM_MINORITY_POSITIVE") != "1" or self.problem_type != "binary":
            return False
        flip = 2 * int(np.count_nonzero(y == 1)) > y.shape[0]
        logger.warning("fintfm: FINTFM_MINORITY_POSITIVE set, class 1 recoded as 0: %s", flip)
        return flip

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Class probabilities, with the columns swapped back when the fit recoded the target.

        Without a recoding this is AutoGluon's own path unchanged, regression included.
        """
        if not self._minority_flipped:
            return super()._predict_proba(X, **kwargs)
        X = self.preprocess(X, **kwargs)
        proba = self.model.predict_proba(X)
        if self._minority_flipped:
            proba = proba[:, ::-1]
        return self._convert_proba_to_unified_form(proba)

    def _fit_regression(self, X, y, network, device: str, n_bins: int, params: dict, partition: dict | None):
        """Store the table as context for the binned-regression wrapper.

        ``n_bins`` is clamped to the checkpoint's ``max_classes`` rather than raising,
        because unlike a class count it is *our* free parameter, not a property of the task:
        a 10-bin request against an 8-logit head is an over-ask by us, and the honest
        response is to use the 8 bins available and say so in the log. A class count that
        exceeds the head is a different thing entirely and still raises.
        """
        from fintfm.inference.regressor import FinancialTFMRegressor

        n_features = X.shape[1]
        if n_features > network.cfg.max_features and params.get("max_features_strategy", "raise") != "select":
            # "select" ranks the columns against the bin indices on the training rows and is
            # an ordinary narrower table from then on; "subsample" averages member
            # probabilities over column subsets and is left to classification.
            raise AssertionError(
                f"fintfm checkpoint accepts at most {network.cfg.max_features} features, "
                f"task has {n_features}. Truncating would score a different task, so this "
                f"raises instead. Pass max_features_strategy='select' to score it."
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
        regressor = FinancialTFMRegressor(network, n_bins=bins, classifier_params={"device": device, **params})
        thresholds = os.environ.get("FINTFM_REG_THRESHOLDS", "").strip()
        if thresholds:
            # K binary classifiers on y > t_k instead of one classifier over max_classes bins
            # (fintfm ThresholdTFMRegressor, FINDINGS §235): the grid is K, not the head's cap.
            from fintfm.inference.threshold_regressor import ThresholdTFMRegressor

            if partition is not None:
                raise ValueError("FINTFM_REG_THRESHOLDS and a regression partition are not combined yet")
            regressor = ThresholdTFMRegressor(
                network, n_thresholds=int(thresholds), classifier_params={"device": device, **params}
            )
            logger.warning("fintfm: threshold regression, %s thresholds", thresholds)
        if partition is not None:
            # A regression tree on the target; each leaf bins its own targets (fintfm 49.1).
            import inspect

            from fintfm.inference.large_n import PartitionedTFMRegressor

            # missing_classes is a classification setting; the regressor has no classes.
            accepted = set(inspect.signature(PartitionedTFMRegressor.__init__).parameters)
            partition = {k: v for k, v in partition.items() if k in accepted}
            regressor = PartitionedTFMRegressor(regressor, random_state=params["random_state"], **partition)
        self.model = regressor.fit(X, y)

    @staticmethod
    def _default_max_context() -> int:
        """Default ``max_context``: 1000, or ``FINTFM_MAX_CONTEXT`` when set.

        The environment override exists so a TabArena-Lite context sweep (fintfm audit
        AU5.14: 500 and 750 against 1000) runs the default config without a code change. It
        does not change the config name, so each setting needs its own results directory (the
        Lite runner's ``FINTFM_RUN_NAME``), or the results cache returns the earlier numbers.
        A run submitted as the default config leaves it unset. A ``max_context`` passed as a
        hyperparameter takes precedence over both.
        """
        raw = os.environ.get("FINTFM_MAX_CONTEXT")
        if raw is None or raw.strip() == "":
            return 1000
        value = int(raw)
        if value <= 0:
            raise ValueError(f"FINTFM_MAX_CONTEXT must be a positive integer, got {raw!r}")
        logger.warning("fintfm: max_context default overridden to %d by FINTFM_MAX_CONTEXT", value)
        return value

    @staticmethod
    def _env_params() -> dict:
        """Classifier defaults overridden by ``FINTFM_PARAMS``, a JSON object; empty when unset.

        Lets a TabArena-Lite arm run the default config with different inference settings
        and no code change: ``n_ensemble`` 1 (fintfm audit AU5.7), ``ensemble_class_shift``,
        ``ensemble_transforms`` or ``ensemble_svd_components`` (AU5.12, design review DR3.5).
        Same caveat as ``FINTFM_MAX_CONTEXT``: the config name does not change, so each arm
        needs its own results directory (the Lite runner's ``FINTFM_RUN_NAME``). A run
        submitted as the default config leaves it unset, and a hyperparameter passed
        explicitly still takes precedence.

        Every key must be a ``FinancialTFMClassifier`` argument. A misspelled key raises
        rather than being ignored, because an ignored override completes the run and reports
        the default's numbers under the arm's name. ``model``, ``device`` and
        ``random_state`` are set by the adapter and refused here. A JSON list becomes a tuple,
        the type ``ensemble_transforms`` takes.

        Returns:
            The overrides, keyed by classifier argument name.

        Raises:
            ValueError: the variable is not a JSON object, names an unknown or reserved
                argument, or sets ``max_context`` while ``FINTFM_MAX_CONTEXT`` is also set.
        """
        import inspect
        import json

        raw = os.environ.get("FINTFM_PARAMS")
        if raw is None or raw.strip() == "":
            return {}
        try:
            params = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"FINTFM_PARAMS is not valid JSON: {exc}") from exc
        if not isinstance(params, dict):
            raise ValueError(f"FINTFM_PARAMS must be a JSON object, got {type(params).__name__}")

        from fintfm.inference.classifier import FinancialTFMClassifier

        reserved = {"self", "model", "device", "random_state"}
        allowed = set(inspect.signature(FinancialTFMClassifier.__init__).parameters) - reserved
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise ValueError(
                f"FINTFM_PARAMS names {unknown}, which are not FinancialTFMClassifier "
                f"arguments the adapter lets you set; allowed: {sorted(allowed)}"
            )
        if "max_context" in params and os.environ.get("FINTFM_MAX_CONTEXT", "").strip():
            raise ValueError("max_context is set by both FINTFM_PARAMS and FINTFM_MAX_CONTEXT; set one")
        params = {k: tuple(v) if isinstance(v, list) else v for k, v in params.items()}
        logger.warning("fintfm: classifier defaults overridden by FINTFM_PARAMS: %s", params)
        return params

    @staticmethod
    def _env_json(name: str):
        """The JSON value of environment variable ``name``, or ``_UNSET`` when it is unset or blank.

        Raises:
            ValueError: the value is not valid JSON.
        """
        raw = os.environ.get(name)
        if raw is None or raw.strip() == "":
            return _UNSET
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} is not valid JSON: {exc}") from exc

    @staticmethod
    def _validated_partition(value, *, source: str) -> dict | None:
        """Check a partition setting; ``None`` means no partition, ``{}`` the class defaults.

        Fix 4 of fintfm's TabArena loss analysis: wrap the classifier in
        ``fintfm.inference.large_n.PartitionedTFMClassifier``, a shallow decision tree whose
        leaves each carry their own context, with the global context as the fallback for a
        leaf too small or missing a class (fintfm ``docs/results/FINDINGS.md`` §169, +0.0375
        mean AP on V4FinBench at depth 2 with a 5% leaf floor). The value is a dict of that
        class's arguments; ``{}`` takes its defaults, which are §169's. ``estimator`` and
        ``random_state`` are set by the adapter and refused here, and a misspelled key raises
        rather than being ignored, because an ignored setting completes the run and reports
        the default's numbers under the arm's name.

        Args:
            value: ``None`` or a dict of partition arguments.
            source: Where the value came from, for the message.

        Returns:
            ``value``, validated.

        Raises:
            ValueError: ``value`` is neither ``None`` nor a dict, or names an unknown or
                reserved argument.
        """
        import inspect

        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError(f"{source} must be a JSON object or null, got {type(value).__name__}")

        from fintfm.inference.large_n import PartitionedTFMClassifier

        reserved = {"self", "estimator", "random_state"}
        allowed = set(inspect.signature(PartitionedTFMClassifier.__init__).parameters) - reserved
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(
                f"{source} names {unknown}, which are not PartitionedTFMClassifier "
                f"arguments the adapter lets you set; allowed: {sorted(allowed)}"
            )
        return value

    @classmethod
    def _env_entry_overrides(cls) -> dict:
        """Entry hyperparameter defaults set by environment variables, keyed by hyperparameter name.

        The variables set a default, so the precedence is: a hyperparameter passed explicitly,
        then the environment variable, then the entry default. Each variable applies to every
        problem type.

        - ``FINTFM_PARTITION``: ``partition_<type>``; a JSON object, or ``null`` for no partition.
        - ``FINTFM_PROBE``: ``probe_<type>``, for classes that have a probe; a JSON object.
        - ``FINTFM_CHECKPOINT``: ``checkpoint_<type>`` for every type, then
          ``FINTFM_CHECKPOINT_MULTICLASS`` and ``FINTFM_CHECKPOINT_REGRESSION`` for theirs. Local paths.
        - ``FINTFM_REG_BINS``: ``n_bins``.
        - ``FINTFM_REG_THRESHOLDS``: no partition for regression, unless ``FINTFM_PARTITION`` is set,
          because the threshold regressor is not combined with one.

        Returns:
            The overrides; empty when no variable is set.

        Raises:
            ValueError: a JSON variable is malformed, or ``FINTFM_REG_BINS`` is not an integer.
        """
        over: dict = {}
        options = cls._per_problem_options
        partition = cls._env_json("FINTFM_PARTITION")
        if "partition" in options:
            if partition is not _UNSET:
                for pt in PROBLEM_TYPES:
                    over[f"partition_{pt}"] = partition
                logger.warning("fintfm: partition set by FINTFM_PARTITION: %s", partition)
            elif os.environ.get("FINTFM_REG_THRESHOLDS", "").strip():
                over["partition_regression"] = None
        probe = cls._env_json("FINTFM_PROBE")
        if "probe" in options and probe is not _UNSET:
            for pt in PROBLEM_TYPES:
                over[f"probe_{pt}"] = probe
            logger.warning("fintfm: probe options set by FINTFM_PROBE: %s", probe)
        if "checkpoint" in options:
            generic = os.environ.get("FINTFM_CHECKPOINT", "").strip()
            for pt in PROBLEM_TYPES:
                specific = os.environ.get(f"FINTFM_CHECKPOINT_{pt.upper()}", "").strip() if pt != "binary" else ""
                if specific or generic:
                    over[f"checkpoint_{pt}"] = specific or generic
        env_bins = os.environ.get("FINTFM_REG_BINS", "").strip()
        if env_bins:
            try:
                over["n_bins"] = int(env_bins)
            except ValueError:
                raise ValueError(f"FINTFM_REG_BINS must be an integer, got {env_bins!r}") from None
            logger.warning("fintfm: regression n_bins %d (FINTFM_REG_BINS)", over["n_bins"])
        return over

    def _resolve_entry(self, params: dict) -> dict:
        """Take this fit's adapter settings out of ``params``, for the task's problem type.

        Removes every key that is not a ``FinancialTFMClassifier`` argument (the
        per-problem-type ``checkpoint_*``, ``partition_*`` and, where the class has it,
        ``probe_*``; ``n_bins``; ``model_path*``) so the rest can be passed on unchanged, and
        returns the ones that apply to ``self.problem_type``.

        Args:
            params: The model's hyperparameters; modified in place.

        Returns:
            A dict with ``checkpoint`` (a path string or a reference from
            :mod:`tabarena.models.fintfm.checkpoints`, not yet resolved), ``partition``
            (``None`` or a validated dict), ``probe`` (``None`` or a dict; ``None`` for classes
            without a probe) and ``n_bins`` (an int of at least 2).

        Raises:
            ValueError: the problem type is not binary, multiclass or regression, or a value
                is malformed.
        """
        pt = self.problem_type
        if pt not in PROBLEM_TYPES:
            raise ValueError(f"fintfm supports {PROBLEM_TYPES}, not problem type {pt!r}")
        per_type = {
            opt: {t: params.pop(f"{opt}_{t}", None) for t in PROBLEM_TYPES} for opt in self._per_problem_options
        }
        # An explicit local path beats the checkpoint hyperparameters. The type-specific one
        # beats the generic ``model_path``.
        explicit = {t: params.pop(f"model_path_{t}", None) for t in ("multiclass", "regression")}
        generic = params.pop("model_path", None)
        n_bins = int(params.pop("n_bins", self._entry_n_bins))
        if n_bins < 2:
            raise ValueError(f"n_bins must be at least 2, got {n_bins}")

        checkpoint = explicit.get(pt) or generic or per_type.get("checkpoint", {}).get(pt)
        if checkpoint is None:
            raise ValueError(f"no checkpoint is set for problem type {pt!r}")
        validate_checkpoint_ref(checkpoint, source=f"checkpoint_{pt}")
        partition = self._validated_partition(per_type.get("partition", {}).get(pt), source=f"partition_{pt}")
        return {
            "checkpoint": checkpoint,
            "partition": partition,
            "probe": per_type.get("probe", {}).get(pt),
            "n_bins": n_bins,
        }

    def _inner_classifiers(self) -> list:
        """Every fitted ``FinancialTFMClassifier`` behind ``self.model``.

        One for the plain classifier; for a partitioned one, the template, the global
        fallback and each leaf's. They share one network, but each records its own device.

        Returns:
            The classifiers, in no particular order.
        """
        if not hasattr(self.model, "leaf_estimators_"):
            return [self.model]
        inner = [self.model.estimator, self.model.global_estimator_]
        for leaves in self.model.leaf_estimators_:
            inner.extend(est for est in leaves.values() if est is not None)
        return inner

    def _set_default_params(self):
        default_params = {
            # D12: a single column-identity draw is the configuration that decision's own
            # text calls incomplete, so ensemble over draws by default.
            "n_ensemble": 8,
            # docs/results/FINDINGS.md §83/§84 measured context as nearly flat on a 1M-row panel;
            # 1000 is the largest value that is cheap on every machine this runs on.
            "max_context": self._default_max_context(),
            # §81: identity-preserving, and materially cheaper wherever attention scores are
            # materialised (CPU and MPS). No effect on CUDA (§95).
            "feature_chunk": 16,
        }
        default_params.update(copy.deepcopy(self._entry_params))
        default_params.update(self._env_params())
        default_params["n_bins"] = self._entry_n_bins
        for option in self._per_problem_options:
            entry_values = getattr(self, f"_entry_{option}")
            for pt in PROBLEM_TYPES:
                default_params[f"{option}_{pt}"] = copy.deepcopy(entry_values[pt])
        default_params.update(self._env_entry_overrides())
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def get_device(self) -> str:
        return str(next(self._inner_classifiers()[0].model.parameters()).device)

    def _set_device(self, device: str):
        for clf in self._inner_classifiers():
            clf.model.to(device)
            clf.device = device

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
