from __future__ import annotations

import contextlib
import logging
import os
from typing import TYPE_CHECKING, ClassVar

from autogluon.features import LabelEncoderFeatureGenerator
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)

#: Problem types a checkpoint, partition and probe can be set for.
PROBLEM_TYPES = ("binary", "multiclass", "regression")
#: The entry options the library takes for one problem type and AutoGluon names per problem type
#: (``checkpoint_binary``, ``partition_multiclass``, ``probe_regression``, ...), so a search space can
#: set each independently.
_PER_PROBLEM_OPTIONS = ("checkpoint", "partition", "probe")
_MISSING = object()


def prefetch_weights() -> str:
    """Download the pinned binary checkpoint into the Hugging Face cache; return its path.

    Declared in ``info.py`` so the benchmark harness can stage weights before a run rather
    than downloading inside the first fit, where the transfer would be charged to that fit's
    time limit. The file's SHA-256 is checked against the pinned reference in ``fintfm``.

    Returns:
        The local path of the binary checkpoint.
    """
    from fintfm.readout import BINARY_CHECKPOINT, resolve_checkpoint

    return resolve_checkpoint(BINARY_CHECKPOINT, problem_type="binary")


class FinTFMLinearProbeModel(AbstractTorchModel):
    """fintfm, decided by a linear probe on its representation rather than its own head.

    A thin AutoGluon adapter over :class:`fintfm.readout.LinearReadoutModel`: this class turns
    AutoGluon's frame into a float32 matrix (categorical columns label-encoded), picks the device and the
    BLAS thread limit, and hands the matrix to the library, which owns the readout, the checkpoints and
    the measured entry (``fintfm.readout.entry_defaults``). The hyperparameters are the library's
    settings; the checkpoint, partition and probe carry the problem type in their name
    (``checkpoint_binary``, ``partition_multiclass``, ``probe_regression``).

    **Inputs it does not use.** ``time_limit`` is accepted for AutoGluon's API and does not bound the
    fit: there is no training loop or early stop to cut short (AutoGluon's own per-fold projection
    still applies). ``X_val`` / ``y_val`` are accepted and ignored. The readout selects its
    regularisation, cross-fit folds and multiclass floor by internal cross-validation on the training
    rows only, inside ``fintfm.readout`` (``LogisticRegressionCV`` and the cross-fit and floor folds for
    classification, the inner K-fold of the ridge readouts for regression); nothing is tuned or selected
    on the validation split.

    **Threads.** ``blas_threads`` (``"auto"``, ``None`` or an int >= 1; default ``"auto"``, that is
    ``min(4, num_cpus)``): fit and predict run with numpy's and scipy's BLAS limited to that many
    threads (``threadpoolctl``); torch's threads are untouched. On an 8-vCPU machine OpenBLAS's default of
    8 threads beside torch's made the readout's ``LogisticRegressionCV`` miss the time limit on large
    tables, and 4 fitted the same table in 412 s (fintfm ``docs/results/FINDINGS.md`` §296, §298).
    ``None`` leaves BLAS as the machine sets it.

    The device is CUDA when AutoGluon grants a GPU and CUDA is present, otherwise CPU; Apple's MPS is
    never chosen automatically. ``FINTFM_DEVICE`` (``cpu``, ``mps`` or ``cuda``) forces it.

    fintfm ``docs/results/FINDINGS.md`` §130/§131 measured that a plain ``LogisticRegression`` on the
    model's own pre-head representation beats its trained classification head on V4FinBench, 4 of 5
    folds, +0.0214 mean AP. This class asks whether that finding transfers to TabArena's protocol.
    A separate ``ag_key`` keeps its results out of any other fintfm entry's cache.

    Paper: none; the project's measurement log is its public record.
    Authors: Mukharbek Organokov
    Codebase: https://github.com/kabartay/fintfm
    License: Apache-2.0
    """

    #: Imports charged to the untimed warm-up rather than to the first timed fit. ``torch`` and
    #: the fintfm package are the heavy ones; the checkpoint itself is staged separately by
    #: ``prefetch_weights``.
    warmup_modules: ClassVar[tuple[str, ...]] = ("torch", "fintfm")
    ag_key = "TA-FINTFM-LINPROBE"
    ag_name = "FinTFM-LinProbe"
    ag_priority = 65
    seed_name = "random_state"

    #: All three problem types, because the entry is measured on all three. All three checkpoints
    #: are published on Hugging Face and pinned by commit and SHA-256 (``fintfm.readout.checkpoints``).
    _supported_problem_types: ClassVar[list[str]] = ["binary", "multiclass", "regression"]
    #: The published multiclass checkpoint represents at most 10 classes and the fit refuses more,
    #: so the limit is declared where AutoGluon reads it (``ag.max_classes``) instead of failing inside
    #: the fit. Every TabArena-Lite multiclass table has at most 10 classes.
    _default_auxiliary_params_extra: ClassVar[dict] = {"max_classes": 10}

    #: Not a CUDA-only model: it runs on CPU and CUDA, and the reference machine has no CUDA at all.
    #: Requiring a GPU (``minimum_num_gpus``) would make every fit fail on that machine rather than run
    #: slowly, so no minimum is set; :meth:`_get_default_resources` asks for one GPU only when CUDA is
    #: present, as ModernNCA does.
    default_resources_physical_cores_only = True

    #: Foundation-model setting: an in-context model has no train loop, so after bagging a
    #: single refit on all the data is at parity with the bagged ensemble and far cheaper.
    _default_ag_args_ensemble_extra: ClassVar[dict] = {
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": True,
    }

    #: BLAS thread limit chosen at fit and reused at predict, or ``None`` (no limit).
    _blas_limit: int | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator: LabelEncoderFeatureGenerator | None = None
        self._categorical_features: list[int] = []
        self._feature_names: list[str] = []

    def _get_default_resources(self) -> tuple[int, int]:
        """CPU cores as the parent decides, plus one GPU when CUDA is available.

        ``_fit`` moves the network to CUDA only when AutoGluon passes ``num_gpus`` > 0, and the
        inherited default is zero GPUs, so without this a CUDA machine fitted on its CPU while
        its GPU sat idle. Machines without CUDA, Apple Silicon included, get zero GPUs and fit on
        CPU (:meth:`_choose_device`).

        Returns:
            ``(num_cpus, num_gpus)``.
        """
        import torch

        num_cpus, _ = super()._get_default_resources()
        return num_cpus, 1 if torch.cuda.is_available() else 0

    def _set_default_params(self):
        """Default hyperparameters: the library's measured entry for this problem type, plus ``blas_threads``."""
        from fintfm.readout import entry_defaults

        defaults = entry_defaults(self.problem_type)
        for option in _PER_PROBLEM_OPTIONS:
            defaults[f"{option}_{self.problem_type}"] = defaults.pop(option)
        defaults["blas_threads"] = "auto"
        for param, val in defaults.items():
            self._set_default_param_value(param, val)

    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, **kwargs) -> np.ndarray:
        """Label-encode the categorical columns and emit a dense float32 matrix, NaN kept.

        fintfm embeds every cell with a separate missingness mask, so imputing here would erase that
        signal and make a missing cell indistinguishable from a real zero. A missing categorical
        becomes AutoGluon's ``-1`` category code. The positions of the categorical columns are kept in
        ``_categorical_features``; the library target-encodes them.

        Args:
            X: The feature frame.
            is_train: Whether these are the training rows (the label encoder is fitted on them).
            **kwargs: Passed on to AutoGluon's own preprocessing.

        Returns:
            The ``(n, d)`` float32 matrix.
        """
        import numpy as np

        X = super()._preprocess(X, **kwargs)
        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator is not None and self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
            self._categorical_features = [X.columns.get_loc(c) for c in self._feature_generator.features_in]
        if is_train:
            self._feature_names = [str(c) for c in X.columns]
        return X.to_numpy(dtype=np.float32, na_value=np.nan)

    @classmethod
    def _estimator_settings(cls, params: dict, problem_type: str) -> dict:
        """The library's keyword settings for one fit, from the model's hyperparameters.

        Takes the per-problem-type hyperparameters (``checkpoint_<type>``, ``partition_<type>``,
        ``probe_<type>``) out of ``params``, keeps those of ``problem_type`` under their plain names and
        drops the others, which belong to a different problem type.

        Args:
            params: The hyperparameters; modified in place.
            problem_type: ``binary``, ``multiclass`` or ``regression``.

        Returns:
            ``params`` with ``checkpoint``, ``partition`` and ``probe`` set when the hyperparameters
            carry them.
        """
        for option in _PER_PROBLEM_OPTIONS:
            per_type = {t: params.pop(f"{option}_{t}", _MISSING) for t in PROBLEM_TYPES}
            if per_type[problem_type] is not _MISSING:
                params[option] = per_type[problem_type]
        return params

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
        """Fit the library's readout under the BLAS thread limit; ``X_val``, ``y_val`` and ``time_limit`` are unused."""
        import numpy as np
        from fintfm.readout import LinearReadoutModel

        params = self._get_model_params()
        setting = self._check_blas_threads(params.pop("blas_threads", None), source="blas_threads")
        self._blas_limit = self._blas_limit_for(setting, num_cpus)
        # ``seed_name = "random_state"`` makes AutoGluon inject the seed into the params dict.
        params.pop("random_state", None)
        settings = self._estimator_settings(params, self.problem_type)
        checkpoint = settings.pop("checkpoint", None)
        device = self._choose_device(num_gpus)
        with self._blas_context():
            matrix = self.preprocess(X, is_train=True)
            self.model = LinearReadoutModel(
                self.problem_type, checkpoint, device=device, random_state=self.random_seed, **settings
            ).fit(
                matrix,
                np.asarray(y),
                categorical_features=self._categorical_features,
                feature_names=self._feature_names,
            )

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Class probabilities (positive class only for binary), or the predictions for regression."""
        with self._blas_context():
            matrix = self.preprocess(X, **kwargs)
            if self.problem_type == "regression":
                return self.model.predict(matrix)
            return self._convert_proba_to_unified_form(self.model.predict_proba(matrix))

    @classmethod
    def _choose_device(cls, num_gpus: int | float | None) -> str:
        """The torch device ``_fit`` loads the network on.

        ``FINTFM_DEVICE`` when set; otherwise CUDA when AutoGluon granted a GPU and CUDA is
        present; otherwise CPU. Apple's MPS is never chosen automatically: every stored
        TabArena-Lite result of this model was fitted on CPU or CUDA, and on MPS the default
        entry's fits exhausted Metal's memory pool (requests above 80 GiB on a 64 GB machine)
        within the first minutes of every dataset tried, 2026-11-18. ``FINTFM_DEVICE=mps``
        still selects it.

        Args:
            num_gpus: GPUs AutoGluon allotted to this fit.

        Returns:
            ``"cpu"``, ``"mps"`` or ``"cuda"``.
        """
        import torch

        forced = cls._forced_device()
        if forced is not None:
            return forced
        if num_gpus and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def _forced_device() -> str | None:
        """The device named by ``FINTFM_DEVICE``, or ``None`` to choose one automatically.

        Unset, the adapter uses CUDA when granted a GPU, else CPU. On Apple Silicon, Metal's memory
        outside live tensors grows over many forward passes and can exhaust within one dataset;
        ``FINTFM_DEVICE=cpu`` lets such a dataset rerun on CPU. It does not change the config name.

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

    @staticmethod
    def _check_blas_threads(value, *, source: str) -> str | int | None:
        """Validate a ``blas_threads`` value.

        Args:
            value: ``"auto"``, ``None`` or an int >= 1.
            source: Where the value came from, for the error message.

        Returns:
            The value.

        Raises:
            ValueError: any other value.
        """
        if value is None or value == "auto":
            return value
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{source} must be 'auto', None or an integer >= 1, got {value!r}")
        return value

    @staticmethod
    def _blas_limit_for(setting: str | int | None, num_cpus) -> int | None:
        """The BLAS thread limit for a fit given AutoGluon's CPUs.

        Args:
            setting: A validated ``blas_threads`` value.
            num_cpus: CPUs AutoGluon gave the fit; anything but a positive int counts as the
                machine's CPU count.

        Returns:
            ``min(4, num_cpus)`` for ``"auto"``, the int itself, or ``None`` for no limit.
        """
        if setting is None:
            return None
        if setting != "auto":
            return int(setting)
        cpus = num_cpus if isinstance(num_cpus, int) and not isinstance(num_cpus, bool) and num_cpus > 0 else None
        return min(4, cpus or os.cpu_count() or 1)

    def _blas_context(self):
        """A context that limits BLAS to ``_blas_limit`` threads, or does nothing when it is ``None``."""
        if self._blas_limit is None:
            return contextlib.nullcontext()
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=self._blas_limit, user_api="blas")

    def get_device(self) -> str:
        """The device the fitted network lives on."""
        return self.model.current_device()

    def _set_device(self, device: str):
        """Move the fitted network to ``device`` (AutoGluon offloads it to CPU around saving)."""
        self.model.to(device)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
