from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pandas as pd

from tabarena.benchmark.exec_models import ExternalSystemModel

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer

    from tabarena.benchmark.task.metadata import ValidationMetadata

# Checkpoints on the public HF repo. The names match the model entry's, but the regression
# checkpoint was replaced on the Hub on 2026-09-28, so the two entries do not load identical
# weights from the same paths.
_HF_REPO = "occams/Xiaomi-TabLDM"
_CLASSIFIER_CHECKPOINT = "checkpoints/clf_default.ckpt"
_REGRESSOR_CHECKPOINT = "checkpoints/reg_default.ckpt"
#: Hub revision holding the checkpoints this system is benchmarked with. The upstream
#: ``_load_model`` downloads from ``main`` with no ``revision``, and the weights are not
#: immutable, so an unpinned fetch resolves to whatever is on the Hub today rather than to the
#: weights the run recorded. Pinning it here is what makes the run reproducible.
_HF_REVISION = "b8364615b202cd823833395ec12c644d85895089"


def _checkpoint_path(filename: str) -> Path:
    """Resolve a checkpoint to a local path, downloading it at the pinned Hub revision.

    The downloaded file is handed to the estimator as ``model_path`` (which its ``_load_model``
    loads directly when it already exists), so the fit never falls back to the unpinned ``main``
    download path inside the upstream estimator.
    """
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=_HF_REPO, filename=filename, revision=_HF_REVISION))


def _checkpoint_name(problem_type: str) -> str:
    """The checkpoint filename for ``problem_type``."""
    return _CLASSIFIER_CHECKPOINT if problem_type in ("binary", "multiclass") else _REGRESSOR_CHECKPOINT


def _log(msg: str) -> None:
    """Emit a flushed, timestamped ``[Xiaomi-TabLDM+]`` progress line.

    ``flush=True`` is deliberate: SLURM block-buffers a job's stdout, so an unflushed ``print``
    inside a long fit/predict is invisible until the buffer fills. Flushing means that if a run
    stalls, the last line written pinpoints the stage it stalled in instead of the log simply
    ending after the Hugging Face "Loading weights" message.
    """
    print(f"[Xiaomi-TabLDM+ {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _import_estimators():
    """Import the newer TabLDM estimators, naming the extra to install when they are missing.

    Xiaomi-TabLDM ships one distribution per pinned commit, and the two commits TabArena pins are
    not API-compatible: the ``tabldm`` extra (the ``Xiaomi-TabLDM`` model entry) exports
    ``TabLDMEnhancedClassifier`` / ``TabLDMEnhancedRegressor``, while this system needs the newer
    ``TabLDMClassifier`` / ``TabLDMRegressor``. Installing both is impossible, so a wrong pin
    shows up as an ``ImportError`` and this message says which extra to swap to.
    """
    try:
        from tabldm import TabLDMClassifier, TabLDMRegressor
    except ImportError as e:
        raise ImportError(
            "Xiaomi-TabLDM+ needs the `tabldm_plus` extra (the newer Xiaomi-TabLDM commit). "
            "Install it with `pip install 'tabarena[tabldm_plus]'`. That extra is mutually "
            "exclusive with `tabldm`, which the Xiaomi-TabLDM model entry uses: the two commits "
            "export different estimator class names and cannot be installed together.",
        ) from e
    return TabLDMClassifier, TabLDMRegressor


def _resolve_device(device: str | None, num_gpus: int | None, *, cuda_available: bool) -> str:
    """Pick the torch device string from the system's ``device`` setting and the split's GPU budget.

    ``device`` is the user-facing override (``None`` = follow the budget, ``"cpu"`` to force CPU,
    ``"gpu"`` / ``"cuda"`` to require a GPU). When it is ``None``, an allocated GPU count above
    zero means CUDA, and an unconstrained budget (``None``) falls back to using a GPU whenever one
    is visible, matching the model wrapper's ``default_num_gpus = 1``.
    """
    if device is not None:
        normalized = device.lower()
        if normalized == "cpu":
            return "cpu"
        if normalized in ("gpu", "cuda"):
            if not cuda_available:
                raise RuntimeError(f"device={device!r} was requested but CUDA is not available.")
            return "cuda"
        return device

    effective_num_gpus = num_gpus if num_gpus is not None else int(cuda_available)
    return "cuda" if (effective_num_gpus > 0 and cuda_available) else "cpu"


class TabLDMPlusSystemModel(ExternalSystemModel):
    """Xiaomi-TabLDM+ — TabLDM run with candidate enhancement, benchmarked as a system.

    The enhanced estimator fits its candidate/ensemble weights on a hold-out split it carves out
    of ``X``/``y`` itself (``validation=True``), which is what makes this a system rather than a
    model: TabArena's model protocol hands the wrapper a validation split of its own and does not
    permit a second internal holdout, while :class:`ExternalSystemModel` documents that a system
    "carves its own internal validation from ``X``/``y`` if it wants one".

    Init hyperparameters (each a per-config knob for the system generator):

    * ``enhance_candidates`` — ``True`` (default) runs the candidate-enhancement path; ``False``
      falls back to upstream's plain single-group inference with no ensembling enhancements and
      no holdout weights.
    * ``validation`` — whether the enhanced path may carve its hold-out split. Only meaningful
      when ``enhance_candidates=True``.
    * ``n_estimators`` — base foundation-model forward passes per group; ``None`` keeps upstream's
      default. Exposed so a smoke fit can drop it to 1.
    * ``device`` — ``None`` (default: follow the split's GPU budget), ``"cpu"`` to force CPU, or
      ``"gpu"`` / ``"cuda"`` to require a GPU.

    The estimator's seed is not an init knob: it is the per-split ``random_state`` the runner
    threads into :meth:`_fit_system` (see the base ``ExternalSystemModel``), so each split gets
    distinct but reproducible randomness.

    Codebase: pip name ``Xiaomi-TabLDM``, import name ``tabldm``, installed from GitHub since it is
    not on PyPI. Runs a newer commit than the ``Xiaomi-TabLDM`` model entry and is installed with
    the mutually exclusive ``tabldm_plus`` extra (see ``_import_estimators``).
    Checkpoints: https://huggingface.co/occams/Xiaomi-TabLDM
    License: Apache-2.0 (Copyright Xiaomi Corporation)
    """

    warmup_modules: ClassVar[tuple[str, ...]] = ("tabldm",)
    warmup_torch_device: ClassVar[bool] = True

    def __init__(
        self,
        *,
        enhance_candidates: bool = True,
        validation: bool = True,
        n_estimators: int | None = None,
        device: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.enhance_candidates = enhance_candidates
        self.validation = validation
        self.n_estimators = n_estimators
        self.device = device
        self._estimator = None

    def _estimator_hps(self, *, num_cpus: int | None, random_state: int | None) -> dict:
        """Build the keyword arguments for the upstream estimator, minus ``device``.

        ``random_state`` is the per-split seed threaded in by the runner; it falls back to ``0``
        when ``None`` (a direct fit outside the runner) so the fit stays deterministic.

        Kept free of any Hub access: the pinned ``model_path`` is resolved in :meth:`_fit_system`,
        where the fit actually needs the weights, so these kwargs can be built and inspected
        without a network.
        """
        hps = {
            "enhance_candidates": self.enhance_candidates,
            "validation": self.validation,
            "random_state": random_state if random_state is not None else 0,
        }
        if self.n_estimators is not None:
            hps["n_estimators"] = self.n_estimators
        if num_cpus is not None:
            hps["n_jobs"] = num_cpus
        return hps

    def _fit_system(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        target_name: str,
        problem_type: str,
        eval_metric: Scorer,
        validation_metadata: ValidationMetadata,
        num_cpus: int | None,
        num_gpus: int | None,
        memory_limit: float | None,
        time_limit: float | None,
        random_state: int | None,
    ):
        """Fit one enhanced TabLDM estimator on all the training data.

        The frames are passed through raw: the estimator's own ``TransformToNumerical``
        preprocessing ordinal-encodes categorical dtypes and mean-imputes numeric NaNs when given
        a DataFrame, and for regression it standardizes/inverse-transforms the target itself. The
        enhanced path then carves its own hold-out split from ``X``/``y`` to fit the blend weights
        (``validation=True``), which is the behaviour that makes this a system.

        The estimator has no early stopping, so ``time_limit`` and ``memory_limit`` cannot be
        pushed into the fit; both are reported here so a run that overruns is visible in the log.
        """
        import torch

        cuda_available = torch.cuda.is_available()
        device = _resolve_device(self.device, num_gpus, cuda_available=cuda_available)

        classifier_cls, regressor_cls = _import_estimators()
        model_cls = classifier_cls if problem_type in ("binary", "multiclass") else regressor_cls
        hps = self._estimator_hps(num_cpus=num_cpus, random_state=random_state)
        # Pinned revision, resolved here rather than in `_estimator_hps` so the kwargs stay
        # network-free. Handed to the estimator as `model_path`, which its `_load_model` loads
        # directly when the file exists, so the fit never falls back to the unpinned `main`
        # download inside the upstream estimator.
        hps["model_path"] = _checkpoint_path(_checkpoint_name(problem_type))

        _log(
            f"fit start: problem_type={problem_type} X={X.shape} device={device} "
            f"enhance_candidates={self.enhance_candidates} validation={self.validation} "
            f"(num_gpus={num_gpus}, num_cpus={num_cpus}, time_limit={time_limit}, "
            f"memory_limit={memory_limit}, cuda_available={cuda_available})",
        )
        start = time.monotonic()
        self._estimator = model_cls(device=device, **hps)
        self._estimator.fit(X, y)
        _log(f"fit done in {time.monotonic() - start:.1f}s")
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        _log(f"predict start: X={X.shape}")
        start = time.monotonic()
        preds = self._estimator.predict(X)
        _log(f"predict done in {time.monotonic() - start:.1f}s")
        return pd.Series(preds, index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        _log(f"predict_proba start: X={X.shape}")
        start = time.monotonic()
        # `classes_` is the original label space (the estimator inverse-transforms its own internal
        # encoding), so the columns line up with the task's labels.
        proba = self._estimator.predict_proba(X)
        _log(f"predict_proba done in {time.monotonic() - start:.1f}s")
        return pd.DataFrame(proba, index=X.index, columns=self._estimator.classes_)

    def cleanup(self):
        """Drop the fitted estimator once the post-evaluate consumers are done with it."""
        self._estimator = None


def prefetch_weights() -> None:
    """Pre-download both checkpoints (classifier and regressor) at the pinned Hub revision.

    The system's own tooling hook (``SystemInfo.prefetch_weights``): the benchmark setup does not
    call it. Resolves both through :func:`_checkpoint_path` so the cached files are the ones the
    run will load, then hands each path to its estimator and calls ``_load_model()`` to let it
    build the network and validate the checkpoint layout.
    """
    classifier_cls, regressor_cls = _import_estimators()
    classifier_cls(model_path=_checkpoint_path(_CLASSIFIER_CHECKPOINT))._load_model()
    regressor_cls(model_path=_checkpoint_path(_REGRESSOR_CHECKPOINT))._load_model()
