from __future__ import annotations

import copy
import json
import logging
import math
import os
from typing import TYPE_CHECKING, ClassVar

from tabarena.models.fintfm.checkpoints import (
    BINARY_CHECKPOINT,
    MULTICLASS_CHECKPOINT_UNPUBLISHED,
    REGRESSION_CHECKPOINT_UNPUBLISHED,
)
from tabarena.models.fintfm.model import FinTFMModel, prefetch_weights  # noqa: F401 (re-exported to info.py)
from tabarena.models.fintfm_linprobe.target_transform import TARGET_TRANSFORMS, TargetTransform, smearing_factor

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)

#: Rows given to the downstream classifier's training step. Mirrors
#: ``fintfm.experiments.representation_transfer.TRAIN_SAMPLE_SIZE``: every minority-class row
#: is kept, then padded with a random majority-class sample up to this budget, because a plain
#: random subsample at a low positive rate can miss the minority class outright.
TRAIN_SAMPLE_SIZE = 20_000
#: Fewest probe rows a partition leaf is given, however small its share of the training rows.
LEAF_MIN_SAMPLE = 2_000
#: Keys a probe option dict accepts; see :meth:`FinTFMLinearProbeModel._validated_probe`.
PROBE_OPTIONS = frozenset({"scale", "cv", "raw", "head", "no_rep"})
#: Default of the ``probe_crossfit_folds`` hyperparameter. The measured TabArena-Lite entry
#: (FinTFM-LinProbe, 932 Elo) was produced WITHOUT cross-fitting, so any figure from a run
#: with this default is a new measurement; see :class:`FinTFMLinearProbeModel`.
PROBE_CROSSFIT_FOLDS = 5
#: Deepest tree ``fintfm.inference.large_n.PartitionedTFMClassifier`` and ``PartitionedTFMRegressor``
#: accept (``max_depth`` is checked to lie in [1, 3]); ``partition_depth_cap`` is clipped to it.
PARTITION_LIBRARY_MAX_DEPTH = 3
#: Defaults of ``partition_scale_k`` and ``partition_depth_cap``.
PARTITION_SCALE_K = 4.0
PARTITION_DEPTH_CAP = 5
#: The alpha grid ``reg_ridge_alphas="wide"`` names.
WIDE_RIDGE_ALPHAS = (-3.0, 7.0, 21)


class FinTFMLinearProbeModel(FinTFMModel):
    """fintfm, decided by a linear probe on its representation rather than its own head.

    Sibling of :class:`~tabarena.models.fintfm.model.FinTFMModel` whose default config is the
    measured TabArena-Lite entry (see ``_entry_*`` below). fintfm ``docs/results/FINDINGS.md`` §130/§131 measured that a plain
    ``LogisticRegression`` on the model's own pre-head representation beats its trained
    classification head on V4FinBench, 4 of 5 folds, +0.0214 mean AP, even though the probe
    reads the exact tensor the head itself reads (verified to reproduce the head's own logits
    when the head is applied to it). This class asks whether that finding transfers to
    TabArena's protocol, where the comparison is against 94 other methods on 27 datasets
    rather than two arms of one model on one binary credit task.

    Reuses every step of :class:`FinTFMModel` unchanged -- checkpoint loading, capacity
    checks, categorical preprocessing, context storage -- and changes only the decision step:
    ``_fit`` additionally extracts the pre-head representation on a bounded training
    subsample and fits a ``LogisticRegression`` on it; ``_predict_proba`` reads that
    classifier instead of the network's own head. A separate ``ag_key`` keeps its results out
    of the cache the real entrant's Lite run already populated.

    **Cross-fitting (``probe_crossfit_folds``, default 5).** The rows the readout trains on
    are also rows of the stored in-context set, so a plain ``transform_representation`` call
    lets a context row attend to its own labelled copy, which a test row never can. With
    ``probe_crossfit_folds`` of K >= 2, the context rows among the readout's training rows are
    split into K folds and each fold is represented by the classifier restricted
    (``FinancialTFMClassifier.restrict_context``) to the other K - 1 folds; probe rows that are
    not in the context are represented under the full context, as test rows are. The probe
    rows' categorical encoding is then the out-of-fold encoding the classifier was fitted on,
    not the full-statistics encoding applied in-sample. ``0`` or ``1`` disables both and
    reproduces the earlier readout exactly. **The measured 932 Elo entry was produced without
    cross-fitting. The default here is on because that is the methodologically correct
    setting, and the change must be re-measured before any claim is made about it.**
    Test rows are still represented under the full context, so the readout's training rows
    come from contexts (K - 1)/K the size of the one it is applied under. The ``head`` probe
    option is not cross-fitted and is refused while cross-fitting is on; threshold regression
    (``FINTFM_REG_THRESHOLDS``) is not cross-fitted and keeps the earlier readout with a
    warning.

    **Large-table levers (both off by default; off reproduces the earlier fit exactly).**

    - ``partition_scale_with_n`` (bool): choose the partition depth from the training-row count
      instead of fixing it. With ``n_eff`` the rows one tree is fitted on (all rows for one bag,
      ``bag_frac * n`` above one bag), the depth is
      ``clip(ceil(log2(n_eff / (partition_scale_k * max_context))), base, min(partition_depth_cap, 3))``
      where ``base`` is the partition's own ``max_depth`` (2 when unset). ``partition_scale_k``
      defaults to 4 and ``partition_depth_cap`` to 5, but the fintfm partition classes refuse a
      depth above 3 (8 leaves), so the effective cap is 3 until the library allows more; the
      log line and ``partition_depth_chosen_`` record the requested and the effective depth.
      Only the depth changes: the leaf floor (``min_samples_leaf``), bag count and the rest of the
      partition dict are the caller's. Has no effect without a partition for the problem type.
      Cost: each extra depth level doubles the leaf contexts, so up to 2x the leaf classifiers
      (each a context of ``max_context`` rows) and 2x as many leaf probes per bag.
    - ``probe_budget`` (int or ``"all"``, default 20,000 = ``TRAIN_SAMPLE_SIZE``): the readout's
      training-row budget, in total and split across leaves in proportion to their rows (at
      least ``LEAF_MIN_SAMPLE`` per leaf). ``"all"`` uses every training row. Cost: the
      representation pass over the readout rows is linear in the budget, one pass per leaf
      classifier (and one more per cross-fit fold), and the representation matrix is held in
      memory, so 50,000 rows is about 2.5x and ``"all"`` on a 100k-row table about 5x the default.
    """

    ag_key = "TA-FINTFM-LINPROBE"
    ag_name = "FinTFM-LinProbe"
    ag_priority = 65
    seed_name = "random_state"

    #: All three problem types, because the entry is measured on all three. The multiclass and
    #: regression checkpoints are not published yet, so a fit on either raises
    #: ``CheckpointNotPublishedError`` unless a local checkpoint is supplied.
    _supported_problem_types: ClassVar[list[str]] = ["binary", "multiclass", "regression"]

    #: The measured TabArena-Lite entry (fintfm ``docs/results/FINDINGS.md`` §232, §235, §237).
    #: The default config reproduces it with no environment variable set:
    #:
    #: - binary, run ``fintfm_lp_bin_scalecvraw_1114``: the published checkpoint, partition ``{}``
    #:   (the class defaults), probe ``scale`` + ``cv`` + ``raw``;
    #: - multiclass, run ``fintfm_lp_mc_scr_bag8_1113``: partition ``n_bags`` 8, the same probe;
    #: - regression, run ``fintfm_lp_reg_rawpart4_pure_1114``: partition ``n_bags`` 4, probe ``raw``,
    #:   10 bins.
    #:
    #: All three use ``feature_transform`` ``squash`` and ``max_features_strategy`` ``select``.
    _entry_params: ClassVar[dict] = {"feature_transform": "squash", "max_features_strategy": "select"}
    _entry_checkpoint: ClassVar[dict] = {
        "binary": BINARY_CHECKPOINT,
        "multiclass": MULTICLASS_CHECKPOINT_UNPUBLISHED,
        "regression": REGRESSION_CHECKPOINT_UNPUBLISHED,
    }
    _entry_partition: ClassVar[dict] = {"binary": {}, "multiclass": {"n_bags": 8}, "regression": {"n_bags": 4}}
    _entry_probe: ClassVar[dict] = {
        "binary": {"scale": True, "cv": True, "raw": True},
        "multiclass": {"scale": True, "cv": True, "raw": True},
        "regression": {"raw": True},
    }
    _per_problem_options: ClassVar[tuple[str, ...]] = ("checkpoint", "partition", "probe")

    #: The probe options of the current fit, set by ``_fit``.
    _probe_opts: dict = {}
    #: Cross-fit folds of the current fit (0 or 1: off), set with the probe options.
    _crossfit_folds: int = 0
    #: Whether the readout of the current fit is cross-fitted.
    _crossfit_on: bool = False
    #: The matrix the classifier was fitted on, held only while ``_fit`` builds the readout.
    _train_matrix = None
    #: Counters of the cross-fit work of the current fit, for the log line.
    _crossfit_stats: ClassVar[dict] = {}
    #: Regression levers of the current fit (see :meth:`_parse_reg_levers`).
    _reg_transform_name: str = "none"
    _reg_blend = None
    _reg_alphas = None
    #: Training rows of the current fit, read by :meth:`_resolve_entry` for the depth scaling.
    _n_fit_rows: int | None = None
    #: Large-table levers of the current fit, set by :meth:`_resolve_entry`.
    _scale_with_n: bool = False
    _scale_k: float = PARTITION_SCALE_K
    _depth_cap: int = PARTITION_DEPTH_CAP
    _probe_budget_setting: int | str = TRAIN_SAMPLE_SIZE
    #: What the depth scaling chose for the current fit (``None`` when off or unpartitioned).
    partition_depth_chosen_: dict | None = None
    #: The fitted target transform, or ``None`` when none is applied.
    _target_tf: TargetTransform | None = None
    #: What the transform did: requested name, effective name, and the reason for a fallback.
    _reg_transform_info: ClassVar[dict] = {}
    #: Weight on the ridge prediction in the blend with the binned head, ``None`` for no blend.
    _reg_blend_weight: float | None = None
    #: Duan smearing factor of the log1p back-transform.
    _reg_smear: float = 1.0
    #: Held-out readout-row targets and predictions the blend weight and smearing are chosen on.
    _reg_oof: ClassVar[dict] = {}

    def _set_default_params(self):
        super()._set_default_params()
        self._set_default_param_value("probe_crossfit_folds", PROBE_CROSSFIT_FOLDS)
        self._set_default_param_value("reg_target_transform", "none")
        self._set_default_param_value("reg_readout_blend", None)
        self._set_default_param_value("reg_ridge_alphas", None)
        self._set_default_param_value("partition_scale_with_n", False)
        self._set_default_param_value("partition_scale_k", PARTITION_SCALE_K)
        self._set_default_param_value("partition_depth_cap", PARTITION_DEPTH_CAP)
        self._set_default_param_value("probe_budget", TRAIN_SAMPLE_SIZE)

    @classmethod
    def _env_entry_overrides(cls) -> dict:
        """The base overrides plus the regression levers set by environment variables.

        - ``FINTFM_REG_TARGET_TRANSFORM``: ``reg_target_transform``, ``none``, ``log1p`` or ``quantile``.
        - ``FINTFM_REG_BLEND``: ``reg_readout_blend``, ``cv`` or a number in [0, 1] (the weight on
          the ridge); ``none`` or ``off`` leaves it off.
        - ``FINTFM_REG_ALPHAS``: ``reg_ridge_alphas``, a JSON list of positive numbers, ``wide``, or
          ``logspace:lo,hi,n`` (exponents of 10).
        - ``FINTFM_PARTITION_SCALE_N``: ``partition_scale_with_n``; ``on``/``true`` enables with the default k,
          ``off``/``false`` disables, a positive number enables and sets ``partition_scale_k``.
        - ``FINTFM_PROBE_BUDGET``: ``probe_budget``, a positive integer or ``all``.

        Returns:
            The overrides; empty when no variable is set.

        Raises:
            ValueError: a value is malformed.
        """
        over = super()._env_entry_overrides()
        tf = os.environ.get("FINTFM_REG_TARGET_TRANSFORM", "").strip().lower()
        if tf:
            over["reg_target_transform"] = cls._check_reg_transform(tf, source="FINTFM_REG_TARGET_TRANSFORM")
            logger.warning("fintfm: regression target transform %s (FINTFM_REG_TARGET_TRANSFORM)", tf)
        blend = os.environ.get("FINTFM_REG_BLEND", "").strip().lower()
        if blend and blend not in ("none", "off"):
            value = blend if blend == "cv" else cls._float_or_raise(blend, "FINTFM_REG_BLEND")
            over["reg_readout_blend"] = cls._check_reg_blend(value, source="FINTFM_REG_BLEND")
            logger.warning("fintfm: regression readout blend %s (FINTFM_REG_BLEND)", blend)
        alphas = os.environ.get("FINTFM_REG_ALPHAS", "").strip()
        if alphas:
            if alphas.lower() == "wide":
                value = "wide"
            elif alphas.lower().startswith("logspace:"):
                try:
                    lo, hi, n = alphas.split(":", 1)[1].split(",")
                    value = [float(a) for a in _logspace(float(lo), float(hi), int(n))]
                except ValueError:
                    raise ValueError(f"FINTFM_REG_ALPHAS logspace needs lo,hi,n, got {alphas!r}") from None
            else:
                try:
                    value = json.loads(alphas)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"FINTFM_REG_ALPHAS is not valid JSON: {exc}") from exc
            over["reg_ridge_alphas"] = cls._check_reg_alphas(value, source="FINTFM_REG_ALPHAS")
            logger.warning("fintfm: regression ridge alphas set by FINTFM_REG_ALPHAS")
        scale = os.environ.get("FINTFM_PARTITION_SCALE_N", "").strip().lower()
        if scale:
            if scale in ("on", "true", "yes"):
                over["partition_scale_with_n"] = True
            elif scale in ("off", "false", "no"):
                over["partition_scale_with_n"] = False
            else:
                try:
                    k = float(scale)
                except ValueError:
                    raise ValueError(
                        f"FINTFM_PARTITION_SCALE_N must be on, off or a positive number, got {scale!r}"
                    ) from None
                over["partition_scale_with_n"] = True
                over["partition_scale_k"] = cls._check_scale_k(k, source="FINTFM_PARTITION_SCALE_N")
            logger.warning("fintfm: partition depth scaling %s (FINTFM_PARTITION_SCALE_N)", scale)
        budget = os.environ.get("FINTFM_PROBE_BUDGET", "").strip().lower()
        if budget:
            value = budget if budget == "all" else cls._int_or_raise(budget, "FINTFM_PROBE_BUDGET")
            over["probe_budget"] = cls._check_probe_budget(value, source="FINTFM_PROBE_BUDGET")
            logger.warning("fintfm: probe budget %s (FINTFM_PROBE_BUDGET)", budget)
        return over

    @staticmethod
    def _int_or_raise(text: str, name: str) -> int:
        try:
            return int(text)
        except ValueError:
            raise ValueError(f"{name} must be a positive integer or 'all', got {text!r}") from None

    @staticmethod
    def _check_scale_k(value, *, source: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{source} must be a positive number, got {value!r}")
        return float(value)

    @staticmethod
    def _check_probe_budget(value, *, source: str):
        if value == "all":
            return "all"
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{source} must be a positive integer or 'all', got {value!r}")
        return value

    def _parse_scale_levers(self, params: dict) -> None:
        """Validate ``partition_scale_*``, ``partition_depth_cap`` and ``probe_budget``, popping them.

        Raises:
            ValueError: a lever is malformed.
        """
        on = params.pop("partition_scale_with_n", False)
        if not isinstance(on, bool):
            raise ValueError(f"partition_scale_with_n must be a bool, got {on!r}")
        self._scale_with_n = on
        self._scale_k = self._check_scale_k(
            params.pop("partition_scale_k", PARTITION_SCALE_K), source="partition_scale_k"
        )
        cap = params.pop("partition_depth_cap", PARTITION_DEPTH_CAP)
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise ValueError(f"partition_depth_cap must be a positive integer, got {cap!r}")
        self._depth_cap = cap
        self._probe_budget_setting = self._check_probe_budget(
            params.pop("probe_budget", TRAIN_SAMPLE_SIZE), source="probe_budget"
        )

    def _probe_budget(self, n: int) -> int:
        """Readout training-row budget for ``n`` candidate rows: ``probe_budget``, or ``n`` for ``"all"``."""
        return max(1, n) if self._probe_budget_setting == "all" else int(self._probe_budget_setting)

    def _scaled_partition(self, partition: dict | None) -> dict | None:
        """``partition`` with ``max_depth`` chosen from the training-row count, when scaling is on.

        Args:
            partition: The validated partition dict, or ``None``.

        Returns:
            A copy with ``max_depth`` set, or ``partition`` itself when scaling is off, there is
            no partition, or the row count is not known.
        """
        self.partition_depth_chosen_ = None
        n = self._n_fit_rows
        if not self._scale_with_n or partition is None or not n:
            return partition
        base = int(partition.get("max_depth", 2))
        n_bags = int(partition.get("n_bags", 1))
        n_eff = n if n_bags <= 1 else max(1, round(float(partition.get("bag_frac", 0.6)) * n))
        max_context = int(self._get_model_params().get("max_context", 1000))
        target = n_eff / (self._scale_k * max_context)
        wanted = math.ceil(math.log2(target)) if target > 1 else 0
        cap = min(self._depth_cap, PARTITION_LIBRARY_MAX_DEPTH)
        depth = max(base, min(wanted, cap))
        self.partition_depth_chosen_ = {
            "n_rows": n,
            "n_effective": n_eff,
            "base": base,
            "wanted": max(base, min(wanted, self._depth_cap)),
            "chosen": depth,
            "cap_requested": self._depth_cap,
            "cap_effective": cap,
        }
        logger.warning(
            "FinTFM-LinProbe: partition depth %d for n=%d (n_eff=%d, k=%g, max_context=%d, base %d, "
            "cap %d requested, %d effective).",
            depth,
            n,
            n_eff,
            self._scale_k,
            max_context,
            base,
            self._depth_cap,
            cap,
        )
        return {**partition, "max_depth": depth}

    @staticmethod
    def _float_or_raise(text: str, name: str) -> float:
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"{name} must be 'cv', 'none' or a number in [0, 1], got {text!r}") from None

    @staticmethod
    def _check_reg_transform(value, *, source: str) -> str:
        if value is None:
            return "none"
        if value not in TARGET_TRANSFORMS:
            raise ValueError(f"{source} must be one of {TARGET_TRANSFORMS}, got {value!r}")
        return value

    @staticmethod
    def _check_reg_blend(value, *, source: str):
        if value is None or value == "cv":
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{source} must be None, 'cv' or a number in [0, 1], got {value!r}")
        return float(value)

    @staticmethod
    def _check_reg_alphas(value, *, source: str):
        if value is None or value == "wide":
            return value
        try:
            grid = [float(a) for a in value]
        except (TypeError, ValueError):
            raise ValueError(f"{source} must be None, 'wide' or a list of positive numbers, got {value!r}") from None
        if not grid or any(not math.isfinite(a) or a <= 0 for a in grid):
            raise ValueError(f"{source} must be a non-empty list of positive finite numbers, got {value!r}")
        return grid

    def _parse_reg_levers(self, params: dict, *, pop: bool) -> None:
        """Validate the regression levers in ``params`` and keep them on the model.

        Args:
            params: The model's hyperparameters.
            pop: Remove the keys, so the classifier never sees them.

        Raises:
            ValueError: a lever is malformed, or ``reg_readout_blend='cv'`` is set without
                cross-fitting, which its out-of-fold head predictions need.
        """
        get = params.pop if pop else params.get
        self._reg_transform_name = self._check_reg_transform(
            get("reg_target_transform", "none"), source="reg_target_transform"
        )
        self._reg_blend = self._check_reg_blend(get("reg_readout_blend", None), source="reg_readout_blend")
        self._reg_alphas = self._check_reg_alphas(get("reg_ridge_alphas", None), source="reg_ridge_alphas")
        folds = params.get("probe_crossfit_folds", PROBE_CROSSFIT_FOLDS)
        if self._reg_blend == "cv" and not (isinstance(folds, int) and folds >= 2):
            raise ValueError("reg_readout_blend='cv' needs probe_crossfit_folds >= 2 for out-of-fold head predictions")

    def preprocess(self, X, **kwargs):
        """AutoGluon's preprocess, keeping the training matrix when the readout is cross-fitted.

        The matrix returned for ``is_train=True`` is exactly what the classifier is fitted on,
        with out-of-fold categorical encoding. Cross-fitting needs it to know which rows are in
        which leaf's fit rows and to give the readout the same encoding.
        """
        out = super().preprocess(X, **kwargs)
        if kwargs.get("is_train") and self._crossfit_folds >= 2:
            self._train_matrix = out
        return out

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
        import numpy as np

        self._parse_reg_levers(dict(self._get_model_params()), pop=False)
        self._n_fit_rows = len(X)
        y = self._apply_target_transform(y)
        super()._fit(
            X,
            y,
            X_val=X_val,
            y_val=y_val,
            time_limit=time_limit,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            **kwargs,
        )
        crossfit = self._crossfit_folds >= 2
        if crossfit and hasattr(self.model, "thresholds_"):
            logger.warning(
                "FinTFM-LinProbe: probe_crossfit_folds=%d ignored, threshold regression is not cross-fitted.",
                self._crossfit_folds,
            )
            crossfit = False
        train_matrix, self._train_matrix = self._train_matrix, None
        self._crossfit_stats = {"restricted": 0, "context_rows": 0, "other_rows": 0, "skipped": 0}
        if crossfit:
            # The matrix the classifier was fitted on: out-of-fold categorical encoding, and
            # the routing the partition used at fit time.
            Xp = train_matrix
            if Xp is None:
                raise AssertionError("cross-fitting needs the training matrix, which preprocess did not keep")
        else:
            # The same preprocessing _fit just used, replayed with is_train=False so the already-
            # fitted categorical encoder is reused rather than refit on the same rows twice. The
            # public preprocess, as at predict time: it first drops the columns AutoGluon removed
            # for this model, which the private _preprocess alone keeps, and the fitted classifier
            # then refuses the wider table (Bioresponse on TabArena-Lite, 2026-11-11).
            Xp = self.preprocess(X, is_train=False)
        self._crossfit_on = crossfit
        y_arr = np.asarray(y)
        self._ridge_probe = None
        if self.problem_type == "regression":
            self._leaf_probes = None
            self._linear_probe = None
            if self._reg_blend == "cv" and hasattr(self.model, "thresholds_"):
                raise ValueError("reg_readout_blend='cv' is not supported with threshold regression")
            self._reg_oof = {"y": [], "ridge": [], "head": []}
            if self._reg_blend is None:
                self._reg_blend_weight = None
            else:
                self._reg_blend_weight = 1.0 if self._reg_blend == "cv" else self._reg_blend
            self._reg_smear = 1.0
            self._fit_ridge_probe(np.asarray(Xp, dtype=np.float32), y_arr.astype(np.float64))
            self._finish_reg_levers()
            self._log_crossfit()
            return
        if hasattr(self.model, "trees_"):
            # The partition setting wrapped the classifier: one probe per leaf context.
            self._fit_leaf_probes(np.asarray(Xp, dtype=np.float32), y_arr)
            self._log_crossfit()
            return
        self._leaf_probes = None

        sub = _probe_rows(y_arr, self._probe_budget(len(y_arr)), int(self.random_seed))
        y_sub = y_arr[sub]
        if len(np.unique(y_sub)) < 2:
            # A linear probe cannot be fit on one class, so fall back to the network's own head
            # for this fit rather than let one degenerate call fail the whole run. The sampling
            # in _probe_rows no longer produces a single-class subsample from a two-class y
            # (fintfm's TabArena loss analysis, D1, traced the fallbacks seen on Lite to it);
            # this guard remains for a y that arrives with one class.
            logger.warning(
                "FinTFM-LinProbe: training subsample had %d class(es), not 2; falling back "
                "to the network's own head for this fit.",
                len(np.unique(y_sub)),
            )
            self._linear_probe = None
            return
        Xp = np.asarray(Xp, dtype=np.float32)
        self._fit_raw_conditioner(Xp)
        # The classifier was fitted on every row, so a row's position in its fit rows is its index.
        fit_idx = sub if crossfit else None
        self._linear_probe = self._new_classifier_probe(y_sub).fit(
            self._probe_input(self.model, Xp[sub], fit_idx), y_sub
        )
        self._log_crossfit()

    def _resolve_entry(self, params: dict) -> dict:
        """The base class's entry settings, with the probe validated and kept for fit and predict.

        Also takes ``probe_crossfit_folds`` out of ``params`` so the classifier never sees it.

        Raises:
            ValueError: ``probe_crossfit_folds`` is not a non-negative integer, or is 2 or more
                with the ``head`` probe option, which is not cross-fitted.
        """
        self._parse_reg_levers(params, pop=True)
        self._parse_scale_levers(params)
        folds = params.pop("probe_crossfit_folds", PROBE_CROSSFIT_FOLDS)
        if isinstance(folds, bool) or not isinstance(folds, int) or folds < 0:
            raise ValueError(f"probe_crossfit_folds must be a non-negative integer, got {folds!r}")
        entry = super()._resolve_entry(params)
        entry["partition"] = self._scaled_partition(entry["partition"])
        self._probe_opts = entry["probe"] = self._validated_probe(entry["probe"], source=f"probe_{self.problem_type}")
        if folds >= 2 and self._probe_opts.get("head"):
            raise ValueError(
                "probe option 'head' reads the network's own prediction on the readout rows, which "
                "is not cross-fitted; set probe_crossfit_folds to 0 to use it"
            )
        self._crossfit_folds = folds
        return entry

    def _apply_target_transform(self, y):
        """The training target in the readout's space, and the transform kept for predict.

        Only regression with ``reg_target_transform`` other than ``none`` changes ``y``. The
        transformed target is what everything trained on ``y`` sees: the binned regressor (its
        quantile bin edges are rank-based, so bin membership is the same for a monotone
        transform, and only the bin representatives move into the transformed space), the target
        encoding of categoricals, and the ridge readouts. The binned head's prediction is
        therefore in the transformed space too, and is inverted with the ridge's.

        Args:
            y: Training target.

        Returns:
            ``y``, or a transformed series with the same index.
        """
        import numpy as np
        import pandas as pd

        self._target_tf = None
        self._reg_transform_info = {"requested": self._reg_transform_name, "effective": "none", "note": None}
        if self.problem_type != "regression" or self._reg_transform_name == "none":
            return y
        tf, reason = TargetTransform.fit(self._reg_transform_name, np.asarray(y), seed=int(self.random_seed))
        if tf is None:
            logger.warning("FinTFM-LinProbe: reg_target_transform=%s not applied: %s", self._reg_transform_name, reason)
            self._reg_transform_info["note"] = reason
            return y
        self._target_tf = tf
        self._reg_transform_info["effective"] = tf.name
        z = tf.forward(np.asarray(y))
        return pd.Series(z, index=getattr(y, "index", None), name=getattr(y, "name", None))

    def _finish_reg_levers(self) -> None:
        """Choose the cv blend weight and the smearing factor, on training rows only.

        Both come from held-out predictions of the readout rows collected while the ridge probes
        were fitted (:meth:`_collect_oof`): inner K-fold ridge predictions and, for the cv blend,
        cross-fitted head predictions. No test row is involved.
        """
        import numpy as np

        oof = self._reg_oof
        if not oof["y"]:
            if self._reg_blend == "cv":
                logger.warning("FinTFM-LinProbe: no held-out rows for the cv blend; using the ridge alone.")
            if self._target_tf is not None and self._target_tf.name == "log1p":
                logger.warning("FinTFM-LinProbe: no held-out rows for smearing; using a factor of 1.")
            return
        y = np.concatenate(oof["y"])
        ridge = np.concatenate(oof["ridge"])
        pred = ridge
        if self._reg_blend == "cv":
            head = np.concatenate(oof["head"])
            self._reg_blend_weight = self._blend_weight(y, ridge, head)
            pred = self._reg_blend_weight * ridge + (1.0 - self._reg_blend_weight) * head
            logger.warning(
                "FinTFM-LinProbe: cv blend weight on the ridge %.3f (%d rows).", self._reg_blend_weight, len(y)
            )
        if self._target_tf is not None and self._target_tf.name == "log1p":
            self._reg_smear = smearing_factor(y - pred)
            logger.warning("FinTFM-LinProbe: log1p smearing factor %.4f (%d held-out rows).", self._reg_smear, len(y))

    @staticmethod
    def _blend_weight(y: np.ndarray, ridge: np.ndarray, head: np.ndarray) -> float:
        """The weight ``w`` in [0, 1] minimising ``|w * ridge + (1 - w) * head - y|^2``.

        Args:
            y: Training targets of the held-out rows.
            ridge: Their held-out ridge predictions.
            head: Their held-out head predictions.

        Returns:
            The weight on the ridge; 1.0 when the two predictions coincide.
        """
        import numpy as np

        d = ridge - head
        den = float(d @ d)
        if den <= 0.0:
            return 1.0
        return float(np.clip(float((y - head) @ d) / den, 0.0, 1.0))

    def _alpha_grid(self):
        """The ridge alpha grid: ``reg_ridge_alphas`` when set, else the one chosen by the ``raw`` option."""
        import numpy as np

        if self._reg_alphas == "wide":
            return np.logspace(*WIDE_RIDGE_ALPHAS)
        if self._reg_alphas is not None:
            return np.asarray(self._reg_alphas, dtype=np.float64)
        return np.logspace(-3, 5, 17) if self._probe_opts.get("raw") else np.logspace(-3, 3, 13)

    def _collect_oof(self, pipe, rep: np.ndarray, y_sub: np.ndarray, clf, binner, X_sub: np.ndarray, fit_idx) -> None:
        """Record held-out ridge (and head) predictions for the readout rows of one probe.

        The ridge predictions come from an inner K-fold over the readout rows, at the alpha the
        fitted ``RidgeCV`` chose, on the same cross-fitted representation. The head predictions
        come from the cross-fit machinery, so no row is scored against a context holding it.

        Args:
            pipe: The fitted ridge pipeline.
            rep: The readout's input for the rows.
            y_sub: Their targets (transformed space).
            clf: The classifier whose context conditions the rows.
            binner: Its regressor's target binner.
            X_sub: The rows' preprocessed features.
            fit_idx: Per row, its index among ``clf``'s fit rows, or ``None``.
        """
        import numpy as np
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        k = self._crossfit_folds if self._crossfit_folds >= 2 else 5
        if len(y_sub) < 2 * k:
            return
        alpha = float(pipe[-1].alpha_)
        ridge = np.empty(len(y_sub), dtype=np.float64)
        for tr, te in KFold(k, shuffle=True, random_state=int(self.random_seed)).split(rep):
            ridge[te] = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(rep[tr], y_sub[tr]).predict(rep[te])
        if self._reg_blend == "cv":
            if fit_idx is None or binner is None:
                return
            head = self._crossfit_apply(clf, X_sub, fit_idx, lambda c, Xr: _binned_head(binner, c, Xr))
            self._reg_oof["head"].append(np.asarray(head, dtype=np.float64))
        self._reg_oof["y"].append(np.asarray(y_sub, dtype=np.float64))
        self._reg_oof["ridge"].append(ridge)

    def _log_crossfit(self) -> None:
        """Log what cross-fitting cost in this fit."""
        st = self._crossfit_stats
        if not self._crossfit_on:
            return
        logger.warning(
            "FinTFM-LinProbe: cross-fit K=%d: %d restricted context(s) built, %d context-row and %d "
            "other readout row(s) represented, %d estimator(s) left uncross-fitted.",
            self._crossfit_folds,
            st["restricted"],
            st["context_rows"],
            st["other_rows"],
            st["skipped"],
        )

    def _fit_ridge_probe(self, Xp: np.ndarray, y_arr: np.ndarray) -> None:
        """Fit a ridge readout on the binned regressor's pre-head representation.

        The regression counterpart of the classification probe. ``FinancialTFMRegressor``
        conditions a classifier on quantile-bin indices of the target; its pre-head
        representation of a row is read here and a standardised ``RidgeCV`` maps it to the
        continuous target directly, so the prediction is no longer limited to a weighted
        average of the bin representatives. Up to ``probe_budget`` training rows, drawn
        uniformly at random.

        With a partition the model is a ``PartitionedTFMRegressor``: the global
        fallback regressor gets this probe, and every leaf with its own regressor gets a ridge
        probe of its own on the training rows routed to it, read through that leaf's
        classifier, with the row budget split as in :meth:`_fit_leaf_probes`.

        Args:
            Xp: Preprocessed training features, the rows the regressor was fitted on.
            y_arr: Continuous training targets.
        """
        import numpy as np

        n = len(y_arr)
        seed = int(self.random_seed)
        self._fit_raw_conditioner(Xp)
        self._ridge_leaf_probes = None
        if hasattr(self.model, "thresholds_"):
            # Threshold regression (FINTFM_REG_THRESHOLDS): the readout reads the K threshold logits, not a
            # representation, beside the raw features when the raw option is set.
            self._ridge_probe = self._fit_threshold_ridge(Xp, y_arr, seed)
            return
        part = self.model if hasattr(self.model, "trees_") else None
        glob = part.global_estimator_ if part is not None else self.model
        self._ridge_probe = self._fit_ridge(
            glob.classifier_,
            Xp,
            y_arr,
            np.arange(n),
            self._probe_budget(n),
            seed,
            fit_rows=self._all_rows(n),
            binner=getattr(glob, "binner_", None),
        )
        self._ridge_leaf_probes = None
        if part is None:
            return
        routes = part.route(Xp)
        self._ridge_leaf_probes = []
        for b, leaves in enumerate(part.leaf_estimators_):
            probes = {}
            for leaf, est in leaves.items():
                if est is None:
                    probes[leaf] = None
                    continue
                members = np.flatnonzero(routes[b] == leaf)
                if members.size == 0:
                    # No training row routes here once the replayed preprocessing re-encodes
                    # the categoricals; the leaf then predicts through the global readout.
                    probes[leaf] = None
                    continue
                budget = max(LEAF_MIN_SAMPLE, round(self._probe_budget(n) * len(members) / n))
                probes[leaf] = self._fit_ridge(
                    est.classifier_,
                    Xp,
                    y_arr,
                    members,
                    budget,
                    seed + 1 + b * 1000 + leaf,
                    fit_rows=self._leaf_fit_rows(part, b, members, est.classifier_),
                    binner=getattr(est, "binner_", None),
                )
            self._ridge_leaf_probes.append(probes)
        n_leaf = sum(p is not None for probes in self._ridge_leaf_probes for p in probes.values())
        logger.warning(
            "FinTFM-LinProbe: partitioned regression, %d leaf ridge probe(s) in %d bag(s).",
            n_leaf,
            len(part.leaf_estimators_),
        )

    def _threshold_input(self, X: np.ndarray) -> np.ndarray:
        """The threshold regressor's readout input: its K survival logits, plus raw features.

        Args:
            X: Preprocessed rows.

        Returns:
            ``(n, K)`` logits, widened by the squashed features when the ``raw`` option is set.
        """
        import numpy as np

        blocks = [self.model.readout_features(X)]
        cond = getattr(self, "_raw_conditioner", None)
        if cond is not None:
            blocks.append(np.nan_to_num(cond.transform(X), nan=0.0, posinf=3.0, neginf=-3.0).astype(np.float64))
        return blocks[0] if len(blocks) == 1 else np.hstack(blocks)

    def _fit_threshold_ridge(self, Xp: np.ndarray, y_arr: np.ndarray, seed: int):
        """A standardised ``RidgeCV`` on :meth:`_threshold_input` of up to ``probe_budget`` rows.

        Args:
            Xp: Preprocessed training features.
            y_arr: Continuous training targets.
            seed: Seed of the row draw, the same draw :meth:`_fit_ridge` makes.

        Returns:
            The fitted pipeline.
        """
        import numpy as np
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(seed)
        n = len(y_arr)
        budget = self._probe_budget(n)
        sub = np.sort(rng.choice(n, size=budget, replace=False)) if n > budget else np.arange(n)
        alphas = self._alpha_grid()
        feats = self._threshold_input(Xp[sub])
        return make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)).fit(feats, y_arr[sub])

    def _fit_ridge(
        self,
        clf,
        Xp: np.ndarray,
        y_arr: np.ndarray,
        members: np.ndarray,
        budget: int,
        seed: int,
        fit_rows: np.ndarray | None = None,
        binner=None,
    ):
        """A standardised ``RidgeCV`` on ``clf``'s representation of up to ``budget`` of ``members``.

        Args:
            clf: The fitted classifier inside a binned regressor (a leaf's, or the global one).
            Xp: Preprocessed training features.
            y_arr: Continuous training targets.
            members: Indices of the training rows this probe may learn from.
            budget: Most rows given to the probe, drawn uniformly at random.
            seed: Seed of that draw.
            fit_rows: Sorted indices into ``Xp`` of the rows ``clf`` was fitted on, to cross-fit
                the representation of those among the drawn rows; ``None`` for no cross-fitting.
            binner: ``clf``'s regressor's target binner, for the head predictions of the cv blend.

        Returns:
            The fitted pipeline.
        """
        import numpy as np
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(seed)
        m = len(members)
        sub = members[np.sort(rng.choice(m, size=budget, replace=False))] if m > budget else members
        # A wider grid when raw columns join the representation: a 1,024-column table can need
        # far more shrinkage than the representation alone.
        alphas = self._alpha_grid()
        fit_idx = None if fit_rows is None else _positions_in(fit_rows, sub)
        rep = self._probe_input(clf, Xp[sub], fit_idx)
        pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)).fit(rep, y_arr[sub])
        if self._reg_blend == "cv" or (self._target_tf is not None and self._target_tf.name == "log1p"):
            self._collect_oof(pipe, rep, y_arr[sub], clf, binner, Xp[sub], fit_idx)
        return pipe

    def _fit_leaf_probes(self, Xp: np.ndarray, y_arr: np.ndarray) -> None:
        """Fit one probe per leaf context of a partitioned classifier, plus one global probe.

        A leaf's representation is conditioned on that leaf's own context, so representations
        from two leaves are not in one space and a single probe across leaves would mix them.
        Each leaf with its own context therefore gets a probe trained on the training rows the
        tree routes to it, read through that leaf's classifier. The global fallback context
        gets the plain probe. The probe-row budget is split across leaves in proportion to
        their share of the training rows (at least ``LEAF_MIN_SAMPLE`` each), so the total
        representation cost stays near twice the plain probe's rather than growing with the
        number of leaves.

        Args:
            Xp: Preprocessed training features, the rows the partition was fitted on.
            y_arr: Training labels in the original class order.
        """
        import numpy as np

        part = self.model
        n = len(y_arr)
        seed = int(self.random_seed)
        self._fit_raw_conditioner(Xp)
        self._linear_probe = self._fit_probe(
            part.global_estimator_, Xp, y_arr, np.arange(n), self._probe_budget(n), seed, fit_rows=self._all_rows(n)
        )
        routes = part.route(Xp)
        self._leaf_probes = []
        for b, leaves in enumerate(part.leaf_estimators_):
            probes = {}
            for leaf, est in leaves.items():
                if est is None:
                    probes[leaf] = None
                    continue
                members = np.flatnonzero(routes[b] == leaf)
                budget = max(LEAF_MIN_SAMPLE, round(self._probe_budget(n) * len(members) / n))
                probes[leaf] = self._fit_probe(
                    est,
                    Xp,
                    y_arr,
                    members,
                    budget,
                    seed + 1 + b * 1000 + leaf,
                    fit_rows=self._leaf_fit_rows(part, b, members, est),
                )
            self._leaf_probes.append(probes)
        n_leaf = sum(p is not None for probes in self._leaf_probes for p in probes.values())
        n_ctx = sum(e is not None for leaves in part.leaf_estimators_ for e in leaves.values())
        logger.warning(
            "FinTFM-LinProbe: partitioned, %d leaf probe(s) over %d leaf context(s) in %d bag(s); global probe %s.",
            n_leaf,
            n_ctx,
            len(part.leaf_estimators_),
            "fitted" if self._linear_probe is not None else "absent (head fallback)",
        )

    def _fit_probe(
        self,
        est,
        Xp: np.ndarray,
        y_arr: np.ndarray,
        members: np.ndarray,
        budget: int,
        seed: int,
        fit_rows: np.ndarray | None = None,
    ):
        """A logistic regression on ``est``'s pre-head representation of ``members``, or ``None``.

        Args:
            est: A fitted ``FinancialTFMClassifier`` (a leaf's, or the global fallback).
            Xp: Preprocessed training features.
            y_arr: Training labels in the original class order.
            members: Indices of the training rows this probe may learn from.
            budget: Most rows given to the probe, drawn by :func:`_probe_rows`.
            seed: Seed of that draw.
            fit_rows: Sorted indices into ``Xp`` of the rows ``est`` was fitted on, to cross-fit
                the representation of those among the drawn rows; ``None`` for no cross-fitting.

        Returns:
            The fitted probe, or ``None`` when no training row routes to the leaf or the draw
            misses a class present among ``members``; the caller then scores with ``est``'s own
            head. A leaf fitted under
            ``missing_classes="global"`` can hold fewer classes than the table; its probe
            covers the leaf's classes and the caller widens it with the global probe.
        """
        import numpy as np

        if members.size == 0:
            # A leaf context to which none of the replayed training rows routes: the replay
            # (preprocess with is_train=False) can encode categoricals differently from the
            # fit, so the tree sends them elsewhere (HR_Analytics, 2026-11-14). The caller
            # scores such a leaf with its own head, as when the draw misses a class.
            return None
        sub = members[_probe_rows(y_arr[members], budget, seed)]
        y_sub = y_arr[sub]
        if len(np.unique(y_sub)) < len(np.unique(y_arr[members])):
            return None
        fit_idx = None if fit_rows is None else _positions_in(fit_rows, sub)
        return self._new_classifier_probe(y_sub).fit(self._probe_input(est, Xp[sub], fit_idx), y_sub)

    def _all_rows(self, n: int) -> np.ndarray | None:
        """Fit rows of an estimator fitted on all ``n`` rows when cross-fitting, else ``None``."""
        import numpy as np

        return np.arange(n) if self._crossfit_on else None

    def _leaf_fit_rows(self, part, bag: int, members: np.ndarray, est) -> np.ndarray | None:
        """Indices into the training matrix of the rows a leaf's estimator was fitted on.

        The partition fits a leaf on the bag's rows that its tree routes to the leaf. ``members``
        holds every training row the same tree routes there, bag or not, so the fit rows are
        their intersection with the bag.

        Args:
            part: The fitted partitioned classifier or regressor.
            bag: Bag number.
            members: Training rows routed to the leaf, sorted.
            est: The leaf's fitted classifier.

        Returns:
            Sorted indices, or ``None`` when not cross-fitting.

        Raises:
            RuntimeError: the count disagrees with the rows ``est`` stored, which means the
                matrix routed here is not the one the partition was fitted on.
        """
        import numpy as np

        if not self._crossfit_on:
            return None
        rows = np.intersect1d(members, part.bag_rows_[bag], assume_unique=True)
        stored = getattr(est, "_raw_y", None)
        if stored is not None and len(stored) != rows.size:
            raise RuntimeError(
                f"cross-fit: leaf holds {len(stored)} fitted rows but routing the training matrix "
                f"gives {rows.size}; the readout rows would be matched to the wrong context rows"
            )
        return rows

    def _restricted(self, clf, rows: np.ndarray):
        """``clf`` restricted to ``rows`` of its fit rows, whatever its ensemble setting.

        ``restrict_context`` refuses an estimator whose ``predict_proba`` averages ensemble
        members, as the default ``n_ensemble`` of 8 does. The readout reads
        ``transform_representation``, which uses the one stored context and no members, so the
        restriction is taken on a shallow copy with ``n_ensemble`` 1 and is used for
        representations only.

        Args:
            clf: A fitted ``FinancialTFMClassifier``.
            rows: Indices into the rows its ``fit`` received.

        Returns:
            The restricted copy.
        """
        shallow = copy.copy(clf)
        shallow.n_ensemble = 1
        return shallow.restrict_context(rows)

    def _crossfit_representation(self, clf, X: np.ndarray, fit_idx: np.ndarray) -> np.ndarray:
        """Pre-head representation of ``X`` with no row seeing its own labelled copy.

        Args:
            clf: A fitted ``FinancialTFMClassifier``.
            X: Preprocessed rows.
            fit_idx: Per row, its index among the rows ``clf``'s ``fit`` received, or -1.

        Returns:
            ``(n, d)`` representations.
        """
        import numpy as np

        return self._crossfit_apply(
            clf,
            X,
            fit_idx,
            lambda c, Xr: np.asarray(c.transform_representation(Xr, stage="pre_head"), dtype=np.float64),
        )

    def _crossfit_apply(self, clf, X: np.ndarray, fit_idx: np.ndarray, fn) -> np.ndarray:
        """``fn(estimator, rows)`` for ``X`` with no row seen by a context holding it.

        Rows whose ``fit_idx`` is in ``clf``'s stored context are split into
        ``probe_crossfit_folds`` folds by a seeded shuffle of the context, and each fold is
        evaluated under ``clf`` restricted to the rest of the context. Other rows are evaluated
        under the full context, as test rows are.

        Args:
            clf: A fitted ``FinancialTFMClassifier``.
            X: Preprocessed rows.
            fit_idx: Per row, its index among the rows ``clf``'s ``fit`` received, or -1.
            fn: Maps an estimator and rows to one output row per input row.

        Returns:
            The outputs, stacked in the order of ``X``.
        """
        import numpy as np

        ctx = np.asarray(clf.context_indices_)
        fit_idx = np.asarray(fit_idx)
        stats = self._crossfit_stats
        pos = np.minimum(np.searchsorted(ctx, fit_idx), ctx.size - 1)
        in_ctx = (fit_idx >= 0) & (ctx[pos] == fit_idx)
        k = min(self._crossfit_folds, ctx.size)
        if k < 2 and in_ctx.any():
            stats["skipped"] += 1
            logger.warning("FinTFM-LinProbe: a context of %d row(s) cannot be cross-fitted.", ctx.size)
            in_ctx[:] = False
        blocks: list[tuple[np.ndarray, np.ndarray]] = []
        out = np.flatnonzero(~in_ctx)
        if out.size:
            blocks.append((out, fn(clf, X[out])))
            stats["other_rows"] += int(out.size)
        if in_ctx.any():
            parts = np.array_split(np.random.default_rng(int(self.random_seed) + 7919).permutation(ctx.size), k)
            fold_of = np.empty(ctx.size, dtype=np.int64)
            for j, part in enumerate(parts):
                fold_of[part] = j
            row_fold = fold_of[pos]
            for j, part in enumerate(parts):
                rows = np.flatnonzero(in_ctx & (row_fold == j))
                if rows.size == 0:
                    continue
                twin = self._restricted(clf, np.delete(ctx, part))
                stats["restricted"] += 1
                stats["context_rows"] += int(rows.size)
                blocks.append((rows, fn(twin, X[rows])))
        res = np.empty((len(X), *blocks[0][1].shape[1:]), dtype=np.float64)
        for rows, r in blocks:
            res[rows] = r
        return res

    def _score(self, est, probe, X: np.ndarray) -> np.ndarray:
        """Class probabilities for ``X`` from ``probe`` on ``est``'s representation, or ``est``'s head.

        Args:
            est: The fitted classifier whose context conditions the representation.
            probe: Its probe, or ``None`` to use the network's own head.
            X: Preprocessed query rows.

        Returns:
            ``(n, n_classes)`` probabilities in the original class order.
        """
        if probe is None:
            proba = est.predict_proba(X)
            # The head scores the labels it was fitted on, flipped when FINTFM_MINORITY_POSITIVE
            # flipped them; the probes always see the original labels.
            return proba[:, ::-1] if getattr(self, "_minority_flipped", False) else proba
        return probe.predict_proba(self._probe_input(est, X))

    @staticmethod
    def _validated_probe(opts, *, source: str) -> dict:
        """Check readout options; ``None`` or ``{}`` is the plain readout.

        The plain readout is a ``LogisticRegression`` at ``C=1`` on the raw representation, and
        a standardised ``RidgeCV`` for regression. The keys, each a boolean:

        - ``scale``: standardise the probe's inputs before the logistic regression.
        - ``cv``: choose the logistic regression's ``C`` by 3-fold cross-validated log loss on
          the probe rows, instead of the fixed ``C=1``.
        - ``raw``: append the row's own features to the representation, squashed (median, IQR
          and soft clip, fitted on the training rows) with missing values at the median. The
          readout can then reach a linear model of the features as well as the network's
          representation of them. Applies to the ridge readout too.
        - ``head``: append the network's own output for the row: the log of its head's class
          probabilities, or for regression the binned regressor's own prediction. The readout
          can then keep what the head gets right and correct the rest.
        - ``no_rep``: with ``raw``, drop the representation and read the squashed features
          alone, on the same leaves and rows: the matched raw-only control (fintfm FINDINGS
          §238). The network then only supplies the head for leaves whose probe is ``None``.

        Args:
            opts: The ``probe_<type>`` hyperparameter: ``None`` or a dict of the keys above.
            source: Where the value came from, for the message.

        Returns:
            The options as a dict, empty for the plain readout.

        Raises:
            ValueError: the value is not a dict, names an unknown key, or sets ``no_rep``
                without ``raw``.
        """
        if opts is None:
            return {}
        if not isinstance(opts, dict):
            raise ValueError(f"{source} must be a JSON object, got {opts!r}")
        unknown = set(opts) - PROBE_OPTIONS
        if unknown:
            raise ValueError(f"{source} has unknown key(s) {sorted(unknown)}; known: {sorted(PROBE_OPTIONS)}")
        if opts.get("no_rep") and not opts.get("raw"):
            raise ValueError(f"{source}: no_rep needs raw, the readout would have no input")
        return dict(opts)

    def _new_classifier_probe(self, y: np.ndarray):
        """An unfitted classification readout, as :meth:`_validated_probe` configures it.

        With the ``cv`` option the folds are 3, or fewer when the rarest class in ``y`` has
        fewer than 3 rows (a stratified split needs a row of every class in each fold). With
        fewer than 2 rows of some class there is nothing to select ``C`` on, so the readout is
        the fixed ``C=1`` one. On real data, where every class has at least 3 rows, the readout
        is unchanged.

        Args:
            y: The labels the readout will be fitted on.

        Returns:
            A scikit-learn classifier with ``fit`` and ``predict_proba``.
        """
        import numpy as np
        from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        opts = self._probe_opts
        folds = min(3, int(np.unique(y, return_counts=True)[1].min())) if len(y) else 0
        if opts.get("cv") and folds >= 2:
            probe = LogisticRegressionCV(Cs=np.logspace(-3, 2, 6), cv=folds, scoring="neg_log_loss", max_iter=2000)
        else:
            probe = LogisticRegression(max_iter=1000)
        return make_pipeline(StandardScaler(), probe) if opts.get("scale") else probe

    def _fit_raw_conditioner(self, Xp: np.ndarray) -> None:
        """Fit the squash applied to the raw features the ``raw`` option appends, if set.

        Args:
            Xp: Preprocessed training features, all training rows.
        """
        from fintfm.inference.preprocess import FeatureConditioner

        self._raw_conditioner = None
        if self._probe_opts.get("raw"):
            self._raw_conditioner = FeatureConditioner("squash", random_state=int(self.random_seed)).fit(Xp)

    def _probe_input(self, est, X: np.ndarray, fit_idx: np.ndarray | None = None) -> np.ndarray:
        """The readout's input for rows ``X``: ``est``'s pre-head representation, plus options.

        Args:
            est: The fitted network whose context conditions the representation.
            X: Preprocessed rows.
            fit_idx: Only when fitting a cross-fitted readout: per row, its index among the rows
                ``est`` was fitted on, or -1. ``None`` represents every row under the full
                context, as at predict time.

        Returns:
            ``(n, d)`` representation, widened by the squashed features when the ``raw``
            option is set and by the network's own output when ``head`` is set.
        """
        import numpy as np

        opts = self._probe_opts
        if opts.get("no_rep"):
            blocks = []
        elif fit_idx is not None:
            blocks = [self._crossfit_representation(est, X, fit_idx)]
        else:
            blocks = [np.asarray(est.transform_representation(X, stage="pre_head"), dtype=np.float64)]
        cond = getattr(self, "_raw_conditioner", None)
        if cond is not None:
            blocks.append(np.nan_to_num(cond.transform(X), nan=0.0, posinf=3.0, neginf=-3.0).astype(np.float64))
        if self._probe_opts.get("head"):
            if self.problem_type == "regression":
                blocks.append(np.asarray(self.model.predict(X), dtype=np.float64).reshape(-1, 1))
            else:
                blocks.append(np.log(np.clip(np.asarray(est.predict_proba(X), dtype=np.float64), 1e-6, 1.0)))
        return blocks[0] if len(blocks) == 1 else np.hstack(blocks)

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        import numpy as np

        if getattr(self, "_ridge_probe", None) is not None:
            # Regression: AutoGluon reads this method's output as the predictions themselves.
            Xp = np.asarray(self.preprocess(X, **kwargs), dtype=np.float32)
            pred = self._predict_ridge(Xp)
            if self._reg_blend_weight is not None and self._reg_blend_weight < 1.0:
                head = np.asarray(self.model.predict(Xp), dtype=np.float64)
                pred = self._reg_blend_weight * pred + (1.0 - self._reg_blend_weight) * head
            if self._target_tf is not None:
                pred = self._target_tf.inverse(pred, self._reg_smear)
            return pred
        if getattr(self, "_leaf_probes", None) is not None:
            Xp = np.asarray(self.preprocess(X, **kwargs), dtype=np.float32)
            part = self.model
            routes = part.route(Xp)
            out = np.zeros((Xp.shape[0], len(part.classes_)), dtype=np.float64)
            for b, leaves in enumerate(part.leaf_estimators_):
                for leaf in np.unique(routes[b]):
                    rows = np.flatnonzero(routes[b] == leaf)
                    est, probe = leaves.get(int(leaf)), self._leaf_probes[b].get(int(leaf))
                    if est is None:
                        est, probe = part.global_estimator_, self._linear_probe
                    proba = self._score(est, probe, Xp[rows])
                    if proba.shape[1] < len(part.classes_):
                        # A leaf lacking some classes (missing_classes="global"): the global
                        # probe supplies their probability, the leaf's probe shares the rest.
                        classes = probe.classes_ if probe is not None else est.classes_
                        glob = self._score(part.global_estimator_, self._linear_probe, Xp[rows])
                        proba = part.combine_with_global(proba, classes, glob)
                    out[rows] += proba
            return self._convert_proba_to_unified_form(out / len(part.leaf_estimators_))
        if self._linear_probe is None:
            # Fell back to the network's own head at fit time; delegate to it on the raw
            # frame, since super()._predict_proba does its own preprocess call and would
            # double-preprocess an already-converted array otherwise.
            return super()._predict_proba(X, **kwargs)
        X = np.asarray(self.preprocess(X, **kwargs), dtype=np.float32)
        proba = self._linear_probe.predict_proba(self._probe_input(self.model, X))
        return self._convert_proba_to_unified_form(proba)

    def _predict_ridge(self, Xp: np.ndarray) -> np.ndarray:
        """The ridge readout's prediction for preprocessed rows, in the transformed target space."""
        import numpy as np

        if hasattr(self.model, "thresholds_"):
            return self._ridge_probe.predict(self._threshold_input(Xp))
        if getattr(self, "_ridge_leaf_probes", None) is None:
            return self._ridge_probe.predict(self._probe_input(self.model.classifier_, Xp))
        part = self.model
        routes = part.route(Xp)
        out = np.zeros(Xp.shape[0], dtype=np.float64)
        for b, leaves in enumerate(part.leaf_estimators_):
            for leaf in np.unique(routes[b]):
                rows = np.flatnonzero(routes[b] == leaf)
                est, probe = leaves.get(int(leaf)), self._ridge_leaf_probes[b].get(int(leaf))
                if est is None or probe is None:
                    est, probe = part.global_estimator_, self._ridge_probe
                out[rows] += probe.predict(self._probe_input(est.classifier_, Xp[rows]))
        return out / len(part.leaf_estimators_)


def _binned_head(binner, est, X: np.ndarray) -> np.ndarray:
    """The binned regressor's mean prediction from classifier ``est`` for rows ``X``.

    Args:
        binner: The regressor's fitted target binner.
        est: A fitted classifier over the bin indices, possibly with a restricted context.
        X: Preprocessed rows.

    Returns:
        ``(n,)`` expected values, in the target space the regressor was fitted in.
    """
    import numpy as np

    proba = est.predict_proba(X)
    full = np.zeros((proba.shape[0], binner.n_bins_), dtype=np.float64)
    full[:, np.asarray(est.classes_).astype(int)] = proba
    return binner.expected_value(full)


def _logspace(lo: float, hi: float, n: int) -> list[float]:
    """``n`` values from ``10**lo`` to ``10**hi``, log-spaced."""
    import numpy as np

    return list(np.logspace(lo, hi, n))


def _positions_in(fit_rows: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Position of each of ``rows`` within sorted ``fit_rows``, or -1 where it is absent.

    Args:
        fit_rows: Sorted, distinct indices of the rows an estimator was fitted on.
        rows: Indices to look up.

    Returns:
        Integer array aligned with ``rows``.
    """
    import numpy as np

    if fit_rows.size == 0:
        return np.full(len(rows), -1, dtype=np.int64)
    pos = np.minimum(np.searchsorted(fit_rows, rows), fit_rows.size - 1)
    return np.where(fit_rows[pos] == rows, pos, -1).astype(np.int64)


def _probe_rows(y_arr: np.ndarray, budget: int, seed: int) -> np.ndarray:
    """Positions in ``y_arr`` given to a probe: every minority row, padded with majority rows.

    Args:
        y_arr: Labels of the candidate rows.
        budget: Most rows returned.
        seed: Seed of the random draw.

    Returns:
        Sorted or concatenated positions into ``y_arr``, at most ``budget`` of them.
    """
    import numpy as np

    classes, counts = np.unique(y_arr, return_counts=True)
    if len(classes) > 2 or (len(classes) == 2 and set(classes.tolist()) != {0, 1}):
        return _stratified_rows(y_arr, classes, counts, budget, seed)
    positive = np.where(y_arr == 1)[0]
    negative = np.where(y_arr == 0)[0]
    if len(y_arr) > budget and len(positive) >= budget:
        # Class 1 alone fills the budget (class 1 is the majority on some tasks, e.g.
        # Amazon_employee_access at 0.94). Keeping every class-1 row would leave no room
        # for class 0 and force the head fallback, so cap both classes instead:
        # each gets up to half the budget and the larger one fills what the smaller leaves.
        rng = np.random.default_rng(seed)
        n_negative = min(len(negative), budget // 2)
        n_positive = budget - n_negative
        negative = rng.choice(negative, size=n_negative, replace=False)
        positive = rng.choice(positive, size=n_positive, replace=False)
        return np.sort(np.concatenate([positive, negative]))
    if len(y_arr) > budget:
        rng = np.random.default_rng(seed)
        n_negative = budget - len(positive)
        if n_negative < len(negative):
            negative = rng.choice(negative, size=n_negative, replace=False)
        return np.concatenate([positive, negative])
    return np.arange(len(y_arr))


def _stratified_rows(y_arr: np.ndarray, classes: np.ndarray, counts: np.ndarray, budget: int, seed: int) -> np.ndarray:
    """Positions for a probe on labels other than {0, 1}: every class kept, smallest first.

    The binary rule in :func:`_probe_rows` reads class 1 as the minority and class 0 as the
    rest, so on a multiclass ``y`` it would drop every class above 1. Here the budget is
    water-filled across classes instead: classes are visited from smallest to largest, each
    taking all its rows or an equal share of what is left, whichever is smaller, so a rare
    class is kept whole and the largest classes absorb the cut.

    Args:
        y_arr: Labels of the candidate rows.
        classes: The distinct labels, from ``np.unique``.
        counts: Rows per label, aligned with ``classes``.
        budget: Most rows returned.
        seed: Seed of the random draw.

    Returns:
        Sorted positions into ``y_arr``, at most ``budget`` of them, every class present.
    """
    import numpy as np

    if len(y_arr) <= budget:
        return np.arange(len(y_arr))
    rng = np.random.default_rng(seed)
    order = np.argsort(counts, kind="stable")
    remaining = budget
    picked = []
    for i, c in enumerate(order):
        take = min(int(counts[c]), remaining // (len(order) - i))
        rows = np.flatnonzero(y_arr == classes[c])
        picked.append(rows if take >= len(rows) else rng.choice(rows, size=take, replace=False))
        remaining -= take
    return np.sort(np.concatenate(picked))
