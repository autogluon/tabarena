from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pandas as pd

from tabarena.benchmark.exec_models import ExternalSystemModel
from tabarena.models.tabfm.model import _build_tabfm_estimator, _resolve_device

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer

    from tabarena.benchmark.task.metadata import ValidationMetadata


def _log(msg: str) -> None:
    """Emit a flushed, timestamped ``[TabFM+]`` progress line.

    ``flush=True`` is deliberate: SLURM block-buffers a job's stdout, so an
    unflushed ``print`` inside a long fit/predict is invisible until the buffer
    fills. Flushing means that if a run stalls, the last line written pinpoints
    the stage it stalled in (fit vs. predict vs. weight load) instead of the log
    simply ending after the Hugging Face "Loading weights" message.
    """
    print(f"[TabFM+ {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Cap on CPU BLAS threads used during a TabFM+ fit/predict (see ``_blas_thread_limit``).
_FIT_BLAS_THREADS = 1


def _blas_thread_limit():
    """Context manager capping CPU BLAS threads for the duration of the wrapped call.

    TabFM's ``ensemble`` interface (what TabFM+ runs) does multithreaded CPU linear
    algebra the plain TabFM model never does -- ``TruncatedSVD`` over one-hot features,
    NNLS ensemble blending, and Platt/vector calibration. On a many-core node that first
    heavy call makes OpenBLAS spin up its full worker-thread pool, and that pool creation
    (``blas_thread_init``) can deadlock, wedging the fit indefinitely. Capping to a single
    BLAS thread takes OpenBLAS's single-threaded path and sidesteps the deadlock; the real
    compute is the GPU forward passes, so serialising this small CPU BLAS costs ~nothing.

    Applied around predict too: OpenBLAS restores its thread count when the fit's context
    exits, so an uncapped predict could re-trigger the same pool creation.
    """
    from threadpoolctl import threadpool_limits

    return threadpool_limits(limits=_FIT_BLAS_THREADS, user_api="blas")


class TabFMPlusSystemModel(ExternalSystemModel):
    """TabFM+ — TabFM run through its heavier ``ensemble`` interface, benchmarked as a system.

    Init hyperparameters (each a per-config knob for the system generator):

    * ``interface`` — ``"ensemble"`` (default) or ``"default"`` (plain constructor).
    * ``device`` — ``None`` (default: a GPU when available/allocated, else CPU), ``"cpu"`` to force
      CPU, or ``"gpu"`` / ``"cuda"`` to require a GPU.

    The TabFM estimator's ensemble seed is not an init knob: it is the per-split ``random_state``
    the runner threads into :meth:`_fit_system` (see the base ``ExternalSystemModel``), so each split
    gets distinct but reproducible randomness.

    Codebase: https://github.com/google-research/tabfm
    """

    def __init__(
        self,
        *,
        interface: str = "ensemble",
        device: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.interface = interface
        self.device = device
        self._estimator = None

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
        """Fit a single TabFM estimator (``interface`` preset) on all the training data.

        ``random_state`` is the per-split seed threaded in by the runner; it is used as the TabFM
        estimator's ensemble seed, falling back to ``0`` when ``None`` (a direct fit outside the
        runner) so the fit stays deterministic.
        """
        import torch

        cuda_available = torch.cuda.is_available()
        # When the runner leaves the GPU budget unconstrained (``None``), fall back to using a GPU
        # whenever CUDA is present -- TabFM is a GPU foundation model.
        effective_num_gpus = num_gpus if num_gpus is not None else int(cuda_available)
        device = _resolve_device(self.device, effective_num_gpus, cuda_available=cuda_available)

        _log(
            f"fit start: interface={self.interface!r} problem_type={problem_type} "
            f"X={X.shape} device={device} (num_gpus={num_gpus}, cuda_available={cuda_available})",
        )
        estimator = _build_tabfm_estimator(
            problem_type=problem_type,
            device=device,
            interface=self.interface,
            random_state=random_state if random_state is not None else 0,
            verbose=True,
        )
        # Confirm where the network actually lives -- the whole system runs where the model's
        # parameters are, so this is the ground truth for "did the fit use the GPU?".
        param = next(estimator.model.parameters(), None)
        _log(f"weights loaded; model on device={param.device if param is not None else 'unknown'}")

        # TabFM does its own preprocessing/label handling, so the raw frames are passed through.
        # The BLAS-thread cap guards the ensemble interface's CPU linear algebra (see
        # ``_blas_thread_limit``); on many-core nodes an uncapped fit can deadlock in OpenBLAS.
        start = time.monotonic()
        with _blas_thread_limit():
            self._estimator = estimator.fit(X=X, y=y)
        _log(f"fit done in {time.monotonic() - start:.1f}s")
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        _log(f"predict start: X={X.shape}")
        start = time.monotonic()
        with _blas_thread_limit():
            preds = self._estimator.predict(X)
        _log(f"predict done in {time.monotonic() - start:.1f}s")
        return pd.Series(preds, index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        _log(f"predict_proba start: X={X.shape}")
        start = time.monotonic()
        with _blas_thread_limit():
            # `classes_` is the original label space (the estimator inverse-transforms its own
            # internal encoding), so the columns line up with the task's labels.
            proba = self._estimator.predict_proba(X)
        _log(f"predict_proba done in {time.monotonic() - start:.1f}s")
        return pd.DataFrame(proba, index=X.index, columns=self._estimator.classes_)
