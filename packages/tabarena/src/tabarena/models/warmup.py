"""Untimed environment warm-up for benchmarked methods.

A fresh benchmark process pays one-time environment costs that a long-lived real-world deployment
would not pay per fit: importing heavy libraries, JIT and kernel compilation (numba, torch), CUDA
context creation, starting the Ray runtime, and reading pretrained checkpoint weights from disk.
TabArena excludes these from the measured ``time_train_s`` / ``time_infer_s`` (and from fit time
limits) by warming the environment before the timed fit; see
``AbstractExecModel.warmup_fn`` and ``ExperimentRunner.run_warmup``.

Fairness contract. A warm-up may only do data-independent work that is a one-time, per-environment
cost in a real deployment: imports, kernel and JIT compilation, CUDA context initialization,
hardware and shared-library handles (for example the libebm native library), Ray runtime startup,
and reading pretrained checkpoint weights into the process-wide registry (a model class that
declares ``shared_weights`` builds its network once per process; the dummy fit below is what reads
it before the timed fit). It must never touch the
task's data, never carry task- or data-specific state into the fit, and never advance a global
random number generator (kernel probes use ``torch.zeros`` under ``torch.random.fork_rng``, never
``torch.randn``). Data-dependent first-call work stays timed: shape-conditioned autotuning on the
real batch, ``torch.compile`` guards for the real shapes, cuBLASLt heuristics, on-the-fly text
encoding, and the fine-tuning copies of a fine-tuned model.

Dummy fit. A warm-up may fit and predict the model on a small synthetic dataset drawn from a fixed
seed that is independent of the task (:func:`tabarena.utils.synthetic_data.make_synthetic_frames`).
This triggers the lazy imports, kernel and library loads and library caches a first fit pays.
Shape-specific caches built for the dummy shapes (cudnn autotune, einx traces, ``torch.compile``
guards) do not transfer to the real data; everything that does transfer is environment work. The
dummy fit runs under the registry's random-state guard, records the torch globals before and after,
restores the torch thread count, and its outcome is written to the report.

Measured as shipped. Entrants whose wrapper declares no ``shared_weights`` read their checkpoint
per fit as their libraries ship: Mitra v1 (the TabArena ``mitra`` entry uses AutoGluon's
``MitraModel`` directly), the AutoGluon system's own foundation-model children (resolved through
``ag_model_registry``), TabPFN-3-API (no local weights), iLTM (the library caches the network
itself), TabDPT v1.1 and TabDPT-Turbo (their ``tabdpt`` releases load inside the constructor),
TabPFN-Wide and SAP-RPT-OSS. Time comparisons between them and the declaring wrappers must be read
with that in mind.

Scope. Warm-up runs once, in the job's main process, before the timed fit. It warms that process,
anything disk-backed (numba's on-disk kernel cache, the Hugging Face cache), the Ray runtime, and,
when the opt-in worker pool is enabled, an import-only set of Ray worker interpreters. A parallel
fold runs in a Ray worker: AutoGluon recycles the worker after one fold for GPU bags (and, before its
workers were reused for CPU folds, for every bag), so a bagged fit with parallel fold fitting pays
each fresh worker's imports and CUDA context inside the measured fit unless the pool covered that
worker; a reused worker is warm from its first fold on.

Fit and predict. The same process runs the timed inference, so the warm-up de-inflates
``time_infer_s`` too. Untimed inference-side preparation happens in the exec model's
``pre_predict`` / ``post_predict`` hooks; the contract for a model's ``prepare_for_inference`` lives
in ``AbstractExecModel.pre_predict``.

Dispatch. :func:`warmup_model_cls` runs, for one AutoGluon model class, the layers (1) declared
``warmup`` classmethod, (2) torch and CUDA context for ``AbstractTorchModel`` subclasses, (3) the
``warmup_modules`` ClassVar collected over the MRO, (4) :data:`WARMUP_STEPS_BY_AG_KEY` for AutoGluon
built-ins, and (5) the dummy fit, which for a class that declares ``shared_weights`` also builds and
registers the network the timed fit reuses. Every step is recorded in a :class:`WarmupReport`; a failing step is logged and recorded
and never stops the others. :func:`run_warmup_fn` wraps a whole ``warmup_fn`` and the runner stores
the report under ``experiment_metadata["warmup_report"]``.
"""

from __future__ import annotations

import contextlib
import functools
import gc
import importlib
import logging
import os
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from tabarena.utils.timing_audit import EnvironmentSnapshot, take_snapshot

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

logger = logging.getLogger(__name__)

WarmupStatus = Literal["ok", "partial", "failed", "disabled", "none"]

#: Environment flag that opts into the CUDA kernel probe of :func:`warmup_torch`.
KERNEL_PROBE_ENV = "TABARENA_WARMUP_KERNELS"
#: Modules the AutoGluon stack warm-up imports. networkx left the trainer on AutoGluon >= 1.6.2.dev0
#: and ``get_autogluon_metadata`` costs 6 ms, so neither is warmed.
AG_STACK_WARMUP_MODULES: tuple[str, ...] = ("autogluon.tabular",)
#: What ``_ray_fit`` (unpickled by reference) and the pickled ``model_base`` import in every fold worker.
RAY_WORKER_WARMUP_MODULES: tuple[str, ...] = (
    "autogluon.core.models.ensemble.fold_fitting_strategy",
    "autogluon.tabular",
)
PARALLEL_FOLD_FITTING_STRATEGIES = frozenset({"parallel_local", "parallel_distributed"})
#: Default ``time_limit`` of the dummy fit in seconds.
DUMMY_FIT_TIME_LIMIT_S: float = 120.0
#: Set to ``1`` to run the dummy fit on every call even when this process already warmed the same
#: model class, problem type and configuration (see :func:`already_warm`).
WARMUP_ALWAYS_ENV = "TABARENA_WARMUP_ALWAYS"
#: Reason recorded on a dummy-fit record skipped because the process is already warm.
ALREADY_WARM_REASON = "already warmed in this process"
#: Set to ``0`` to keep the warmed heap in the collector's normal generations (see :func:`freeze_warm_heap`).
FREEZE_ENV = "TABARENA_WARMUP_FREEZE"
#: Whether :func:`freeze_warm_heap` already moved the warmed heap to the permanent generation in this process.
_HEAP_FROZEN = False
#: ``(model class, problem type, GPU or not, configuration)`` keys whose dummy fit completed in this process.
_WARMED: set[tuple] = set()
#: Keys a model class may override through ``warmup_dummy_fit_kwargs``.
DUMMY_FIT_KWARGS_KEYS = frozenset({"n_rows", "n_features", "n_categorical", "time_limit"})
_TORCH_GLOBALS = ("num_threads", "default_dtype", "cudnn_benchmark", "cudnn_deterministic", "matmul_allow_tf32")


def warmup_always() -> bool:
    """Whether ``TABARENA_WARMUP_ALWAYS`` disables the already-warm skip of the dummy fit."""
    return os.environ.get(WARMUP_ALWAYS_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


def warm_key(
    model_cls: type, *, problem_type: str | None, num_gpus: float | None, hyperparameters: dict | None
) -> tuple:
    """What one dummy fit warms: the class, the problem type, GPU or CPU, and the configuration.

    The configuration is the model's own hyperparameters (AutoGluon's ``ag_args*`` stripped), since
    they select the checkpoint a shared-weights class loads; two configs of one class warm separately.
    """
    hps = strip_ag_args(hyperparameters) if hyperparameters else {}
    config = tuple(sorted((str(k), repr(v)) for k, v in hps.items()))
    return (f"{model_cls.__module__}.{model_cls.__qualname__}", problem_type, bool(num_gpus), config)


def already_warm(
    model_cls: type, *, problem_type: str | None, num_gpus: float | None, hyperparameters: dict | None
) -> bool:
    """Whether a dummy fit for this key completed in this process (and the skip is not disabled).

    The warm-up runs before every experiment. In a process that runs several items (an in-process
    bundle, a local sweep) everything the dummy fit exists to trigger, imports, the CUDA context, the
    shared network in the weights registry, kernel and library caches, is still in place after the
    first item, so repeating the dummy fit only costs time. A key is recorded only after a dummy fit
    ran without error, so a failed warm-up is retried on the next item.
    """
    if warmup_always():
        return False
    return warm_key(model_cls, problem_type=problem_type, num_gpus=num_gpus, hyperparameters=hyperparameters) in _WARMED


def reset_warm_memo() -> None:
    """Forget which dummy fits completed in this process (tests, or after releasing shared weights)."""
    _WARMED.clear()


def freeze_enabled() -> bool:
    """Whether the warmed heap is frozen after the first warm-up (``TABARENA_WARMUP_FREEZE``, on by default)."""
    return os.environ.get(FREEZE_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def freeze_warm_heap() -> int | None:
    """Once per process, collect garbage and move every surviving object to the collector's permanent generation.

    After a warm-up the heap holds the imported libraries, the CUDA state and the shared networks: a few hundred
    thousand objects that live until the process exits. Every later full collection (the exec model's cleanup runs
    one after each experiment, the dummy fit another) still traverses them, at about 0.1 s per call on a warmed
    process. ``gc.freeze`` takes them out of the collector's generations, so those collections only look at what
    the experiment itself created. The collection right before the freeze makes sure the warm-up's own garbage is
    not frozen alive.

    Returns the number of frozen objects when the freeze happened in this call, ``None`` when it was already done
    in this process or is disabled through ``TABARENA_WARMUP_FREEZE=0``.
    """
    global _HEAP_FROZEN
    if _HEAP_FROZEN or not freeze_enabled():
        return None
    gc.collect()
    gc.freeze()
    _HEAP_FROZEN = True
    return gc.get_freeze_count()


def unfreeze_warm_heap() -> None:
    """Return the frozen objects to the collector and allow :func:`freeze_warm_heap` to run again (tests)."""
    global _HEAP_FROZEN
    gc.unfreeze()
    _HEAP_FROZEN = False


def kernel_probe_enabled() -> bool:
    """Whether ``TABARENA_WARMUP_KERNELS`` opts into the CUDA kernel probe."""
    return os.environ.get(KERNEL_PROBE_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


def warmup_ebm_native() -> None:
    """Load interpret's ``libebm`` native library once per process (otherwise paid at the first predict).

    ``Native.get_native_singleton`` constructs and initializes the ctypes handle exactly once and is
    data-independent; ``import interpret.glassbox`` alone does not trigger it.
    """
    from interpret.utils._native import Native

    Native.get_native_singleton()


#: Warm-up steps for AutoGluon built-in model classes, keyed by their registry ``ag_key``. A ``str``
#: entry is a module to import (``"torch"`` routes through :func:`warmup_torch` and also creates the
#: CUDA context), a callable is a zero-argument idempotent step. TabArena's own wrappers use ``TA-*``
#: keys and declare ``warmup_modules`` on the class instead.
WARMUP_STEPS_BY_AG_KEY: dict[str, tuple[str | Callable[[], None], ...]] = {
    "GBM": ("lightgbm",),
    "CAT": ("catboost",),
    "XGB": ("xgboost",),
    "EBM": ("interpret.glassbox", warmup_ebm_native),
    "FASTAI": ("torch", "fastai.tabular.all"),
    "NN_TORCH": ("torch",),
    "TABICL": ("torch", "tabicl"),
    "TABDPT": ("torch", "tabdpt"),
    "TABDPT-TURBO": ("torch", "tabdpt"),
    "NORI": ("torch", "synthefy_nori", "synthefy_nori.inference.predictor"),
    "REALMLP": ("torch", "pytabkit"),
    "TABPFN-3": ("torch", "tabpfn", "tabpfn.model_loading"),
    "TABPFN-2.6": ("torch", "tabpfn", "tabpfn.model_loading"),
    "REALTABPFN-V2": ("torch", "tabpfn", "tabpfn.model_loading"),
    "REALTABPFN-V2.5": ("torch", "tabpfn", "tabpfn.model_loading"),
}


@dataclass
class WarmupReport:
    """What one warm-up did, stored under ``experiment_metadata["warmup_report"]``.

    Attributes:
        status: ``"ok"`` when every step succeeded, ``"partial"`` when the warm-up returned but at
            least one step failed, ``"failed"`` when the ``warmup_fn`` itself raised, ``"disabled"``
            when the runner was told not to warm up, ``"none"`` when the method has nothing to warm.
        label: The method name the runner passed to :func:`run_warmup_fn`.
        model_classes: Names of the model classes :func:`warmup_model_cls` was called for.
        steps: Every step in order (``import:<module>``, ``torch:cuda``, ``weights:<cls>``, ...);
            a failed step is recorded as ``<name>:failed:<ExceptionType>``.
        failed_steps: The failed entries of ``steps``.
        imported_modules: Non-stdlib top-level packages that appeared in ``sys.modules`` during the
            whole warm-up.
        cuda_initialized: ``torch.cuda.is_initialized()`` after the warm-up, ``None`` without torch.
        weights_preloaded: ``WeightsEntry.to_metadata()`` of every shared-weights entry registered in
            the process after the warm-up (the dummy fit of a ``shared_weights`` class loads it).
        ray: Outcome of the Ray warm-up (``already_initialized``, ``initialized_by_warmup``,
            ``init_args``, ``pool``, or ``skipped`` with a reason).
        dummy_fits: One record per dummy fit (``model_cls``, ``ran``, ``duration_s``, ``n_rows``,
            ``problem_type``, ``num_gpus``, ``skipped_reason``, ``error``, ``torch_globals_changed``).
        torch_globals_before: Torch globals right after the warm-up imported torch (thread count,
            default dtype, cudnn benchmark and deterministic flags, matmul TF32). The torch layer
            and the dummy fit (whose RNG guard imports torch) both record it on first sighting, so it
            is ``None`` only when torch is not installed or neither of those layers ran.
        torch_globals_after: The same globals when the warm-up returned, so the artifact shows the
            state the fit ran under even when nothing changed.
        heap_frozen: Number of objects :func:`freeze_warm_heap` moved to the permanent generation right
            after this warm-up; ``None`` when the heap was frozen by an earlier warm-up of the process,
            when the freeze is disabled, or when the warm-up did not complete.
        gpu_memory_allocated_after_probe_bytes: ``torch.cuda.memory_allocated()`` after the torch
            probe; cuBLAS workspaces owned by the handle stay allocated and raise the GPU memory
            baselines by this amount.
        kernel_probe: Whether the opt-in CUDA kernel probe ran.
        duration_s: Wall-clock seconds of the warm-up; ``None`` for disabled, none and failed, which
            keeps ``time_warmup_s`` meaning what it always meant.
        error: ``"<ExceptionType>: <message>"`` when the ``warmup_fn`` raised.
    """

    heap_frozen: int | None = None
    status: WarmupStatus = "ok"
    label: str | None = None
    model_classes: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    failed_steps: list[str] = field(default_factory=list)
    imported_modules: list[str] = field(default_factory=list)
    cuda_initialized: bool | None = None
    weights_preloaded: list[dict] = field(default_factory=list)
    ray: dict[str, Any] = field(default_factory=dict)
    dummy_fits: list[dict] = field(default_factory=list)
    torch_globals_before: dict[str, Any] | None = None
    torch_globals_after: dict[str, Any] | None = None
    gpu_memory_allocated_after_probe_bytes: int | None = None
    kernel_probe: bool = False
    duration_s: float | None = None
    error: str | None = None

    def step(self, name: str, *, failed: bool = False, error: BaseException | None = None) -> None:
        """Record one step; a failed step is suffixed with ``:failed:<ExceptionType>`` and listed twice."""
        if failed:
            name = f"{name}:failed:{type(error).__name__ if error is not None else 'Exception'}"
            self.failed_steps.append(name)
        self.steps.append(name)

    @property
    def dummy_fit(self) -> dict | None:
        """The record of the most recent dummy fit, or ``None`` when none was attempted."""
        return self.dummy_fits[-1] if self.dummy_fits else None

    def to_dict(self) -> dict[str, Any]:
        """A JSON-friendly dict of every field plus ``dummy_fit`` (the most recent dummy-fit record)."""
        out = asdict(self)
        out["dummy_fit"] = self.dummy_fit
        return out


# --- Passive process-state helpers ---------------------------------------------------------


def cuda_initialized() -> bool | None:
    """Whether the CUDA context exists, read through an already-imported torch; ``None`` without torch."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        return bool(torch.cuda.is_initialized())
    except Exception:
        return None


def _cuda_available() -> bool:
    """Whether CUDA is available; imports torch (allowed at call time, never at module import)."""
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def snapshot_torch_globals() -> dict[str, Any] | None:
    """The torch globals a library may mutate, read through an already-imported torch; ``None`` without it."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        return {
            "num_threads": torch.get_num_threads(),
            "default_dtype": str(torch.get_default_dtype()),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        }
    except Exception:
        return None


def _changed_globals(before: dict | None, after: dict | None) -> dict[str, list]:
    if before is None or after is None:
        return {}
    return {key: [before.get(key), after.get(key)] for key in _TORCH_GLOBALS if before.get(key) != after.get(key)}


def _note_torch_state(report: WarmupReport | None) -> None:
    """Record the torch globals (first sighting) and the post-probe GPU allocation on ``report``."""
    if report is None:
        return
    if report.torch_globals_before is None:
        report.torch_globals_before = snapshot_torch_globals()
    if cuda_initialized():
        with contextlib.suppress(Exception):
            report.gpu_memory_allocated_after_probe_bytes = int(sys.modules["torch"].cuda.memory_allocated())


def _run_step(report: WarmupReport | None, name: str, fn: Callable[[], Any]) -> bool:
    """Run one warm-up step; log and record a failure instead of raising. Returns success."""
    try:
        fn()
    except Exception as exc:
        logger.warning("Warm-up step %s failed: %r", name, exc)
        if report is not None:
            report.step(name, failed=True, error=exc)
        return False
    if report is not None:
        report.step(name)
    return True


# --- Imports and torch ----------------------------------------------------------------------


def warmup_imports(*module_names: str) -> None:
    """Import ``module_names`` so later (timed) imports are cache hits; raises on the first failure."""
    for name in module_names:
        importlib.import_module(name)


def warmup_imports_best_effort(*module_names: str, report: WarmupReport | None = None) -> list[str]:
    """Import each of ``module_names`` in its own try/except; a failure is logged and recorded, never raised.

    Returns:
        The names that imported successfully.
    """
    imported: list[str] = []
    for name in module_names:
        if _run_step(report, f"import:{name}", functools.partial(importlib.import_module, name)):
            imported.append(name)
    return imported


def warmup_torch(*, cuda: bool | None = None, kernels: bool | None = None) -> list[str]:
    """Import torch and initialize the CUDA context (both one-time costs per process); idempotent.

    A tiny ``torch.zeros`` device matmul materializes the CUDA context and the cuBLAS handle; the
    allocator cache is emptied afterwards so no cached block stays reserved (the cuBLAS workspace
    owned by the handle does stay allocated). ``cuda=None`` auto-detects. ``kernels`` enables the
    kernel probe of :func:`_probe_cuda_kernels`; ``None`` reads :func:`kernel_probe_enabled`, which
    keeps the probe opt-in until its magnitude is measured on the cluster.

    Returns:
        The steps taken (``torch:import``, ``torch:cuda``, ``torch:kernels:<name>`` ...).
    """
    import torch

    steps = ["torch:import"]
    if cuda is None:
        cuda = torch.cuda.is_available()
    if cuda and torch.cuda.is_available():
        x = torch.zeros((8, 8), device="cuda")
        (x @ x).sum().item()
        steps.append("torch:cuda")
        if kernels is None:
            kernels = kernel_probe_enabled()
        if kernels:
            steps.extend(_probe_cuda_kernels())
        del x
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return steps


def _probe_cuda_kernels() -> list[str]:
    """Load the CUDA kernels the first real forward pass would otherwise load inside the timer.

    Targets the first-use sites of the foundation models: fp16/bf16 autocast matmuls (tabpfn
    inference, causilo), SDPA flash and memory-efficient attention in both dtypes (tabdpt bf16,
    nori, exaonetabular), layer_norm/softmax/gelu, and ``linalg.svd`` / ``linalg.qr`` /
    ``svd_lowrank`` (tabpfn's torch SVD). Every probe uses ``torch.zeros`` or ``torch.eye`` under
    ``torch.inference_mode`` and ``torch.random.fork_rng`` (``svd_lowrank`` draws random
    projections), so no global RNG advances; cudnn flags and TF32 settings are never touched, and
    shape-conditioned autotuning for the real shapes stays timed by design. Each probe is wrapped in
    its own try/except (unsupported hardware raises inside ``sdpa_kernel``); failures are recorded
    as steps and logged at DEBUG. CUDNN_ATTENTION is deliberately not probed.
    """
    import torch
    import torch.nn.functional as F

    steps: list[str] = []
    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16}

    def probe(name: str, fn: Callable[[], Any]) -> None:
        try:
            fn()
            steps.append(f"torch:kernels:{name}")
        except Exception as exc:
            logger.debug("Kernel probe %s failed: %r", name, exc)
            steps.append(f"torch:kernels:{name}:failed:{type(exc).__name__}")

    with torch.inference_mode(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        a = torch.zeros((16, 16), device="cuda")
        for tag, dt in dtypes.items():

            def _matmul(dt=dt):
                with torch.autocast("cuda", dtype=dt):
                    (a @ a).sum().item()

            probe(f"matmul_{tag}", _matmul)
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            backends = {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION}
        except Exception as exc:
            logger.debug("SDPA backends unavailable: %r", exc)
            backends = {}
        for backend_name, backend in backends.items():
            for tag, dt in dtypes.items():

                def _sdpa(backend=backend, dt=dt):
                    q = torch.zeros((1, 1, 8, 16), dtype=dt, device="cuda")
                    with sdpa_kernel(backend):
                        F.scaled_dot_product_attention(q, q, q)

                probe(f"sdpa_{backend_name}_{tag}", _sdpa)
        z = torch.zeros((8, 16), device="cuda")
        probe("layer_norm", lambda: F.layer_norm(z, (16,)))
        probe("softmax", lambda: torch.softmax(z, -1))
        probe("gelu", lambda: F.gelu(z))
        probe("linalg_svd", lambda: torch.linalg.svd(torch.eye(8, device="cuda")))
        probe("linalg_qr", lambda: torch.linalg.qr(torch.eye(8, device="cuda")))
        probe("svd_lowrank", lambda: torch.svd_lowrank(torch.eye(16, device="cuda"), q=4))
        torch.cuda.synchronize()
    return steps


# --- Class-level declarations ---------------------------------------------------------------


def collect_warmup_modules(cls: type) -> tuple[str, ...]:
    """The ``warmup_modules`` ClassVar entries declared anywhere on ``cls``'s MRO, base first, deduplicated.

    Convention: a class declares ``warmup_modules: ClassVar[tuple[str, ...]] = ("lib", "lib.sub")``
    with the modules its fit and predict import lazily; a ``"torch"`` entry is routed through
    :func:`warmup_torch`. The union over the MRO means a subclass only lists what it adds.
    """
    out: list[str] = []
    for klass in reversed(cls.__mro__):
        for name in klass.__dict__.get("warmup_modules", ()) or ():
            if name not in out:
                out.append(name)
    return tuple(out)


def apply_warmup_entries(
    entries: Iterable[str | Callable[[], None]],
    *,
    cuda: bool | None,
    report: WarmupReport | None,
    torch_done: bool = False,
    label: str = "extra",
) -> bool:
    """Apply module-or-callable warm-up entries; ``"torch"`` routes to :func:`warmup_torch` once.

    Returns:
        Whether torch has been warmed by the end (input ``torch_done`` or a ``"torch"`` entry).
    """
    for entry in entries:
        if callable(entry):
            name = getattr(entry, "__name__", repr(entry))
            _run_step(report, f"{label}:{name}", entry)
        elif entry == "torch":
            if torch_done:
                continue
            torch_done = warmup_torch_step(cuda=cuda, report=report)
        else:
            warmup_imports_best_effort(entry, report=report)
    return torch_done


def warmup_torch_step(*, cuda: bool | None, report: WarmupReport | None) -> bool:
    """Run :func:`warmup_torch` as a recorded step; returns whether it succeeded."""
    try:
        steps = warmup_torch(cuda=cuda)
    except Exception as exc:
        logger.warning("Warm-up step torch failed: %r", exc)
        if report is not None:
            report.step("torch", failed=True, error=exc)
        return False
    if report is not None:
        report.steps.extend(steps or [])
        report.kernel_probe = report.kernel_probe or any(step.startswith("torch:kernels:") for step in steps or [])
        _note_torch_state(report)
    return True


def device_for_num_gpus(num_gpus: float | None, *, default_num_gpus: int = 1) -> str:
    """The device type a fit with ``num_gpus`` runs on: ``"cuda"`` or ``"cpu"``.

    ``"cuda"`` when ``num_gpus > 0``, or when ``num_gpus`` is ``None`` (unknown at warm-up), CUDA is
    available and the class defaults to a GPU (``default_num_gpus > 0``); ``"cpu"`` otherwise. The
    warm-up and ``_fit`` derive the shared-weights key through this one function.
    """
    if num_gpus is not None:
        return "cuda" if num_gpus > 0 else "cpu"
    if default_num_gpus > 0 and _cuda_available():
        return "cuda"
    return "cpu"


def resolve_warmup_num_gpus(hyperparameters: dict | None, num_gpus: float | None) -> float | None:
    """The GPU count a fit of this config uses: ``ag_args_fit.num_gpus`` in the config wins over the job value."""
    ag_args_fit = (hyperparameters or {}).get("ag_args_fit") or {}
    configured = ag_args_fit.get("num_gpus")
    if isinstance(configured, int | float) and not isinstance(configured, bool):
        return configured
    return num_gpus


def strip_ag_args(hyperparameters: dict | None) -> dict:
    """A copy of ``hyperparameters`` without ``ag_args``, ``ag_args_fit`` and ``ag_args_ensemble``.

    This is the shape ``AGSingleWrapper`` passes to the warm-up and the shape a model class sees as its
    own parameters, so key derivation and the dummy fit read the same dict as ``_fit``.
    """
    return {k: v for k, v in (hyperparameters or {}).items() if k not in ("ag_args", "ag_args_fit", "ag_args_ensemble")}


def resolve_fold_fitting_strategy(
    model_cls: type,
    hyperparameters: dict | None = None,
    *,
    problem_type: str | None = None,
    num_gpus: float | None = None,
) -> str:
    """The fold fitting strategy AutoGluon will pick for a bag of ``model_cls`` with ``hyperparameters``.

    Mirrors ``BaggedEnsembleModel._get_fold_fitting_strategy`` over the ``ag_args_ensemble`` that
    ``presets.py`` assembles: the class defaults (``_get_default_ag_args_ensemble(problem_type=...)``)
    merged under the user's ``ag_args_ensemble``; ``fold_fitting_strategy_gpu`` / ``_cpu`` selected by
    the bag's GPU count when it is known (pass ``0`` for CPU models even on a GPU node);
    ``"auto"`` resolves to ``"parallel_local"`` iff AutoGluon's ``try_import_ray`` succeeds
    (TabArena does not use distributed mode); ``_disable_parallel_fitting`` forces
    ``"sequential_local"``. The caller decides whether the fit is a bag at all (``num_bag_folds > 1``).
    """
    try:
        defaults = dict(model_cls._get_default_ag_args_ensemble(problem_type=problem_type) or {})
    except Exception:
        try:
            defaults = dict(model_cls._get_default_ag_args_ensemble() or {})
        except Exception:
            defaults = {}
    user = (hyperparameters or {}).get("ag_args_ensemble") or {}
    params = {**defaults, **user}
    strategy = params.get("fold_fitting_strategy", "auto")
    if isinstance(num_gpus, int | float) and not isinstance(num_gpus, bool):
        key = "fold_fitting_strategy_gpu" if num_gpus > 0 else "fold_fitting_strategy_cpu"
        strategy = params.get(key, strategy)
    if strategy == "auto":
        from tabarena.utils.ray_utils import try_import_ray

        try:
            try_import_ray()
            strategy = "parallel_local"
        except Exception:
            strategy = "sequential_local"
    if params.get("_disable_parallel_fitting") and strategy in PARALLEL_FOLD_FITTING_STRATEGIES:
        strategy = "sequential_local"
    return strategy


def ray_worker_warmup_modules(model_cls: type) -> tuple[str, ...]:
    """Modules an import-only Ray worker warm task imports for a bag of ``model_cls``.

    The fold worker's own imports (:data:`RAY_WORKER_WARMUP_MODULES`), the class's module, its
    ``warmup_modules`` and the module entries of :data:`WARMUP_STEPS_BY_AG_KEY`. ``"torch"`` stays an
    import-only entry here: workers never create a CUDA context.
    """
    names: list[str] = list(RAY_WORKER_WARMUP_MODULES)
    module = getattr(model_cls, "__module__", None)
    if module and module != "__main__":
        names.append(module)
    names.extend(collect_warmup_modules(model_cls))
    names.extend(
        entry for entry in WARMUP_STEPS_BY_AG_KEY.get(getattr(model_cls, "ag_key", None), ()) if isinstance(entry, str)
    )
    return tuple(dict.fromkeys(names))


# --- Layered per-class dispatch -----------------------------------------------------------------


def warmup_ag_stack(*, report: WarmupReport | None = None) -> None:
    """Import the AutoGluon tabular stack and record whether Ray is already imported or initialized.

    Ray itself is neither imported nor started here; ``AGWrapper._warmup_ray`` does that for CPU
    bags that resolve to parallel fold fitting. On SLURM with ``setup_slurm_job`` the runtime is
    already up with the job's resources, which the recorded steps make visible.
    """
    warmup_imports_best_effort(*AG_STACK_WARMUP_MODULES, report=report)
    ray = sys.modules.get("ray")
    if ray is None or report is None:
        return
    report.step("ray:already_imported")
    with contextlib.suppress(Exception):
        if ray.is_initialized():
            report.step("ray:already_initialized")
            report.ray.setdefault("already_initialized", True)


def _resolve_dummy_fit_num_gpus(model_cls: type, hyperparameters: dict | None, num_gpus: float | None) -> float:
    """The GPU count the dummy fit uses: the config or job value, else the class default when CUDA exists."""
    resolved = resolve_warmup_num_gpus(hyperparameters, num_gpus)
    if resolved is not None:
        return resolved
    default = getattr(model_cls, "default_num_gpus", 0) or 0
    if default > 0 and _cuda_available():
        return default
    return 0


def warmup_dummy_fit(
    model_cls: type,
    *,
    problem_type: str | None,
    num_cpus: int | None,
    num_gpus: float | None,
    hyperparameters: dict | None,
    report: WarmupReport,
) -> dict[str, Any]:
    """Fit and predict ``model_cls`` once on a small synthetic dataset (layer 6, best effort).

    Gates: ``model_cls.warmup_dummy_fit`` (default ``True``), a known ``problem_type`` that the
    class's ``_supported_problem_types`` accepts, and a GPU when the class needs one: with
    AutoGluon's semantics ``minimum_num_gpus > 0`` applies when a GPU is available or the class sets
    ``gpu_required``, so a model that merely prefers a GPU (LightGBM, CatBoost, the foundation
    models on a CPU-only node) is still dummy-fitted on the CPU exactly like its real fit. The data comes from
    :func:`tabarena.utils.synthetic_data.make_synthetic_frames` with ``n_rows=96``, ``n_features=6``,
    ``n_categorical=1``, ``seed=0``; a class may override ``n_rows``, ``n_features``,
    ``n_categorical`` and ``time_limit`` through ``warmup_dummy_fit_kwargs`` and merge cheapness knobs
    (``n_estimators=1``, ``fine_tune_steps=1``) over the config through
    ``cheap_hyperparameters``; checkpoint-relevant keys must not be overridden there. The
    model is built in a temporary directory, fitted with ``num_cpus`` (default 1), the resolved GPU
    count and the time limit, asked for ``predict_proba`` (classification) or ``predict``
    (regression) on the prediction frame, then deleted; the directory is removed, ``gc.collect()``
    runs and the CUDA cache is emptied when a context exists. The whole step runs under
    :func:`tabarena.models._weights.rng_guard`, forking the CUDA generators only when the fit uses a
    GPU or the CUDA context already exists (the same rule the registry's loaders follow), so a CPU
    dummy fit on a CUDA host never creates the context. Torch is imported first when it is installed
    (the guard imports it anyway); the torch globals are snapshotted before and after and the thread
    count is restored. Any exception is recorded and never raised.

    Returns:
        The record appended to ``report.dummy_fits``: ``model_cls``, ``ran``, ``duration_s``,
        ``n_rows``, ``problem_type``, ``num_gpus``, ``skipped_reason``, ``error``,
        ``torch_globals_changed``.

    A dummy fit that completed in this process for the same class, problem type, device kind and
    configuration is not repeated: the record carries ``skipped_reason`` :data:`ALREADY_WARM_REASON`
    (``TABARENA_WARMUP_ALWAYS=1`` restores the unconditional fit; see :func:`already_warm`).
    """
    record: dict[str, Any] = {
        "model_cls": model_cls.__name__,
        "ran": False,
        "duration_s": None,
        "n_rows": None,
        "problem_type": problem_type,
        "num_gpus": None,
        "skipped_reason": None,
        "error": None,
        "torch_globals_changed": {},
    }
    report.dummy_fits.append(record)
    name = f"dummy_fit:{model_cls.__name__}"

    def skip(reason: str) -> dict[str, Any]:
        record["skipped_reason"] = reason
        report.step(f"{name}:skipped")
        return record

    if not getattr(model_cls, "warmup_dummy_fit", True):
        return skip("warmup_dummy_fit is False")
    if problem_type is None:
        return skip("problem_type unknown")
    supported = getattr(model_cls, "_supported_problem_types", None)
    if supported is not None and problem_type not in supported:
        return skip(f"problem_type {problem_type!r} not supported")
    resolved_gpus = _resolve_dummy_fit_num_gpus(model_cls, hyperparameters, num_gpus)
    record["num_gpus"] = resolved_gpus
    needs_gpu = (getattr(model_cls, "minimum_num_gpus", 0) or 0) > 0
    if resolved_gpus == 0 and needs_gpu and (getattr(model_cls, "gpu_required", False) or _cuda_available()):
        return skip("model needs a GPU and none is allocated")
    if already_warm(model_cls, problem_type=problem_type, num_gpus=num_gpus, hyperparameters=hyperparameters):
        return skip(ALREADY_WARM_REASON)

    fit_kwargs = dict((getattr(model_cls, "warmup_dummy_fit_kwargs", None) or {}).items())
    unknown = set(fit_kwargs) - DUMMY_FIT_KWARGS_KEYS
    if unknown:
        logger.warning("Ignoring unknown warmup_dummy_fit_kwargs %s on %s", sorted(unknown), model_cls.__name__)
        fit_kwargs = {key: value for key, value in fit_kwargs.items() if key in DUMMY_FIT_KWARGS_KEYS}
    time_limit = fit_kwargs.pop("time_limit", DUMMY_FIT_TIME_LIMIT_S)
    n_rows = int(fit_kwargs.get("n_rows", 96))
    record["n_rows"] = n_rows

    from tabarena.models._weights import rng_guard

    # The RNG guard below imports torch, so import it first (when installed) and snapshot the globals
    # the fit may mutate; otherwise a dummy fit that is the process's first torch import would have
    # nothing to restore or compare against.
    with contextlib.suppress(ImportError):
        import torch
    torch_before = snapshot_torch_globals()
    _note_torch_state(report)
    start = time.perf_counter()
    tmp_dir = tempfile.mkdtemp(prefix="tabarena_warmup_")
    model = None
    try:
        # Fork the CUDA generators only when the fit runs on a GPU or the context already exists;
        # forking them on an idle CUDA host would create the context for a CPU-only fit.
        with rng_guard(cuda=resolved_gpus > 0 or bool(cuda_initialized())):
            from autogluon.core.metrics import get_metric

            from tabarena.benchmark.task.metrics import default_eval_metric
            from tabarena.utils.synthetic_data import make_synthetic_frames

            X, y, X_predict = make_synthetic_frames(
                problem_type,
                n_rows=n_rows,
                n_features=int(fit_kwargs.get("n_features", 6)),
                n_categorical=int(fit_kwargs.get("n_categorical", 1)),
                seed=0,
            )
            hps = strip_ag_args(hyperparameters)
            hps.update(getattr(model_cls, "cheap_hyperparameters", None) or {})
            model = model_cls(
                path=tmp_dir,
                name=f"warmup_{model_cls.__name__}",
                problem_type=problem_type,
                eval_metric=get_metric(default_eval_metric(problem_type), problem_type=problem_type),
                hyperparameters=hps,
            )
            model.fit(X=X, y=y, num_cpus=num_cpus or 1, num_gpus=resolved_gpus, time_limit=time_limit, verbosity=0)
            if problem_type == "regression":
                model.predict(X_predict)
            else:
                model.predict_proba(X_predict)
        record["ran"] = True
        report.step(name)
        _WARMED.add(warm_key(model_cls, problem_type=problem_type, num_gpus=num_gpus, hyperparameters=hyperparameters))
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("Warm-up dummy fit of %s failed: %r", model_cls.__name__, exc)
        report.step(name, failed=True, error=exc)
    finally:
        record["duration_s"] = time.perf_counter() - start
        del model
        shutil.rmtree(tmp_dir, ignore_errors=True)
        gc.collect()
        record["torch_globals_changed"] = _changed_globals(torch_before, snapshot_torch_globals())
        torch = sys.modules.get("torch")
        if torch is not None:
            with contextlib.suppress(Exception):
                if torch_before is not None:
                    torch.set_num_threads(int(torch_before["num_threads"]))
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
    return record


def warmup_model_cls(
    model_cls: type,
    *,
    problem_type: str | None = None,
    num_cpus: int | None = None,
    num_gpus: float | None = None,
    hyperparameters: dict | None = None,
    report: WarmupReport | None = None,
    dummy_fit: bool = True,
) -> WarmupReport:
    """Warm the environment for one AutoGluon model class; every layer is additive and recorded.

    The keyword context is what a real deployment also knows before seeing any data; all fields are
    optional. Layers, in order:

    1. A ``warmup`` classmethod declared by the model class, called as ``warmup(problem_type=...,
       num_cpus=..., num_gpus=..., hyperparameters=...)``. It runs first because it may need to set
       process state before the CUDA context exists (Mitra-v2's allocator configuration) and is
       otherwise reserved for work the generic layers cannot express (numba pre-compilation).
    2. ``AbstractTorchModel`` subclasses: :func:`warmup_torch` (import plus CUDA context; CUDA only
       when ``num_gpus`` does not rule it out).
    3. Best-effort imports of :func:`collect_warmup_modules` (a ``"torch"`` entry routes through
       :func:`warmup_torch`, once).
    4. :data:`WARMUP_STEPS_BY_AG_KEY` for AutoGluon built-ins (applied to torch subclasses too).
    5. The dummy fit via :func:`warmup_dummy_fit`, unless ``dummy_fit`` is ``False`` (the exec
       model's ``warmup_dummy_fit``) or the class opts out. For a class that declares
       ``shared_weights`` the dummy fit builds and registers the network the timed fit reuses;
       :func:`record_shared_weights` then lists the registry's entries on the report.

    A failure in any layer is logged and recorded as a failed step; the remaining layers still run.

    Returns:
        ``report`` (created when ``None``) with this class appended to ``model_classes``.
    """
    if report is None:
        report = WarmupReport()
    report.model_classes.append(model_cls.__name__)
    cuda = None if num_gpus is None else num_gpus > 0

    declared = getattr(model_cls, "warmup", None)
    if declared is not None:
        _run_step(
            report,
            f"warmup:{model_cls.__name__}",
            functools.partial(
                declared,
                problem_type=problem_type,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                hyperparameters=hyperparameters,
            ),
        )

    torch_done = False
    try:
        from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel
    except Exception:
        AbstractTorchModel = None
    if AbstractTorchModel is not None and issubclass(model_cls, AbstractTorchModel):
        torch_done = warmup_torch_step(cuda=cuda, report=report)

    torch_done = apply_warmup_entries(
        collect_warmup_modules(model_cls), cuda=cuda, report=report, torch_done=torch_done
    )
    ag_key = getattr(model_cls, "ag_key", None)
    apply_warmup_entries(
        WARMUP_STEPS_BY_AG_KEY.get(ag_key, ()),
        cuda=cuda,
        report=report,
        torch_done=torch_done,
        label=f"extra:{ag_key}",
    )

    if dummy_fit:
        warmup_dummy_fit(
            model_cls,
            problem_type=problem_type,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            hyperparameters=hyperparameters,
            report=report,
        )
    else:
        report.step(f"dummy_fit:{model_cls.__name__}:skipped")
    record_shared_weights(report)
    return report


def record_shared_weights(report: WarmupReport) -> None:
    """List the shared-weights entries registered in the process on ``report.weights_preloaded``.

    The dummy fit of a class that declares ``shared_weights`` builds its network through the library's
    loader and registers it, so the timed fit takes a hit. This records what is present afterwards
    (``WeightsEntry.to_metadata()`` per entry, without host paths) so a result shows which networks
    the fit found ready.
    """
    from tabarena.models import _weights

    report.weights_preloaded = list(_weights.report()["entries"])


def warmup_model_classes(
    classes: Iterable[tuple[type, dict | None]],
    *,
    problem_type: str | None,
    num_cpus: int | None,
    num_gpus: float | None,
    report: WarmupReport | None = None,
    dummy_fit: bool = True,
) -> WarmupReport:
    """Run :func:`warmup_model_cls` for every ``(model_cls, hyperparameters)`` pair into one report."""
    if report is None:
        report = WarmupReport()
    for model_cls, hyperparameters in classes:
        warmup_model_cls(
            model_cls,
            problem_type=problem_type,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            hyperparameters=hyperparameters,
            report=report,
            dummy_fit=dummy_fit,
        )
    return report


def warmup_feature_generator_cls(
    feature_generator_cls: type | None,
    feature_generator_kwargs: dict | None = None,
    *,
    report: WarmupReport | None = None,
) -> WarmupReport:
    """The preprocessing counterpart of :func:`warmup_model_cls` for a feature generator class.

    Imports :func:`collect_warmup_modules` of the class best effort, then calls an optional
    ``warmup(cls, *, feature_generator_kwargs=None, **kwargs)`` classmethod (which lets
    ``TabArenaModelAgnosticPreprocessing`` import the text encoder stack only when this fit will
    encode text on the fly). ``None`` is a no-op. Dispatched by ``AGWrapper`` from
    ``fit_kwargs["feature_generator_cls"]`` and by ``AGModelWrapper`` from its resolved pipeline.
    """
    if report is None:
        report = WarmupReport()
    if feature_generator_cls is None:
        return report
    apply_warmup_entries(collect_warmup_modules(feature_generator_cls), cuda=None, report=report)
    declared = getattr(feature_generator_cls, "warmup", None)
    if declared is not None:
        _run_step(
            report,
            f"warmup:{feature_generator_cls.__name__}",
            functools.partial(declared, feature_generator_kwargs=feature_generator_kwargs),
        )
    return report


def run_warmup_fn(fn: Callable[[], WarmupReport | None], *, label: str) -> WarmupReport:
    """Run a method's ``warmup_fn`` and describe the outcome; never raises.

    Times ``fn``, wraps a ``None`` return into a fresh report, and fills ``imported_modules``
    (non-stdlib top-level packages that appeared in ``sys.modules``), ``cuda_initialized``,
    ``torch_globals_after`` and ``label``. ``status`` is ``"failed"`` when ``fn`` raised (the
    header line naming ``label`` and the traceback are printed, ``duration_s`` stays ``None``),
    ``"partial"`` when it returned with at least one failed step, ``"ok"`` otherwise. After a warm-up
    that returned, the first call in the process freezes the warmed heap (:func:`freeze_warm_heap`,
    recorded as ``heap_frozen``); the freeze runs after ``duration_s`` is taken.
    """
    before: EnvironmentSnapshot | None = take_snapshot()
    start = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:
        print(f"Warm-up of method {label!r} failed (fitting cold instead):")
        traceback.print_exc()
        report = WarmupReport(status="failed", error=f"{type(exc).__name__}: {exc}")
    else:
        report = result if isinstance(result, WarmupReport) else WarmupReport()
        report.duration_s = time.perf_counter() - start
        report.status = "partial" if report.failed_steps else "ok"
        report.heap_frozen = freeze_warm_heap()
    report.label = label
    if before is not None:
        with contextlib.suppress(Exception):
            report.imported_modules = EnvironmentSnapshot.take().diff(before)["new_packages"]
    report.cuda_initialized = cuda_initialized()
    report.torch_globals_after = snapshot_torch_globals()
    return report
