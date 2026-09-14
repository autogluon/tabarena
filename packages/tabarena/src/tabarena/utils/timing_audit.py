"""Passive snapshots of the process environment around the timed sections of a benchmark fit.

The timed fit and predict of a benchmarked method are supposed to run in a warm process: every
library imported, the CUDA context created, Ray started when the method needs it (see
``tabarena.models.warmup``). This module records whether that held. A snapshot taken right before
a timer and one taken right after it are compared with :meth:`EnvironmentSnapshot.diff`, which
reports how many modules were imported inside the timed section, which non-stdlib top-level
packages appeared for the first time (a cold import the warm-up missed), which already-present
packages lazily loaded submodules, and whether CUDA or Ray changed state.

The snapshot is passive. It reads ``sys.modules`` and asks ``torch.cuda.is_initialized()`` and
``ray.is_initialized()`` only through modules that are already imported; it never imports torch or
ray itself, because an import here would change the process state between the fit and the
predict timers (and pre-warm predict). Taking a snapshot costs well under a millisecond and the
callers take them outside the timers.

Scope limits to keep in mind when reading an audit:

The snapshot sees the main process only. Fold workers of parallel Ray fold fitting import cold in
their own processes and are invisible here, so an empty ``new_packages`` is not proof that a
bagged GBDT fit was warm.

``cuda_initialized_before`` is ``True`` on every CUDA node regardless of the warm-up when the
snapshot is taken after the GPU memory tracker has been entered: the tracker's
``torch.cuda.synchronize`` creates the CUDA context before the fit timer starts.

``new_packages`` lists top-level packages whose top-level module was absent before. Submodules that
an already-imported package loads lazily (AutoGluon, pandas and scikit-learn do this during every
fit) are counted in ``new_modules`` and their package appears in ``new_submodule_packages``, not
in ``new_packages``. Standard-library modules and ``__main__`` are excluded from both lists.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

#: Top-level names never reported as new packages: the standard library and the entry-point module.
_EXCLUDED_TOP_LEVEL: frozenset[str] = frozenset(sys.stdlib_module_names) | {"__main__"}


def _top_level(module_name: str) -> str:
    return module_name.partition(".")[0]


def _cuda_initialized() -> bool | None:
    """Whether the CUDA context exists, read through an already-imported torch; ``None`` without torch."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        return bool(torch.cuda.is_initialized())
    except Exception:
        return None


def _ray_initialized() -> bool | None:
    """Whether Ray is connected, read through an already-imported ray; ``None`` without ray."""
    ray = sys.modules.get("ray")
    if ray is None:
        return None
    try:
        return bool(ray.is_initialized())
    except Exception:
        return None


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """The imported modules and the CUDA and Ray state of this process at one point in time.

    Attributes:
        modules: Names of every entry in ``sys.modules`` when the snapshot was taken.
        cuda_initialized: ``torch.cuda.is_initialized()``, or ``None`` when torch is not imported
            or the query failed.
        ray_initialized: ``ray.is_initialized()``, or ``None`` when ray is not imported or the
            query failed.
    """

    modules: frozenset[str]
    cuda_initialized: bool | None
    ray_initialized: bool | None

    @classmethod
    def take(cls) -> EnvironmentSnapshot:
        """Snapshot the current process without importing anything.

        ``sys.modules`` is copied first (``dict.copy`` is atomic under the GIL, so an import running
        in another thread cannot raise "dictionary changed size during iteration").
        """
        modules = frozenset(sys.modules.copy())
        return cls(modules=modules, cuda_initialized=_cuda_initialized(), ray_initialized=_ray_initialized())

    def diff(self, before: EnvironmentSnapshot) -> dict[str, Any]:
        """Describe what changed between ``before`` and this snapshot.

        Args:
            before: The earlier snapshot.

        Returns:
            A dict with ``new_modules`` (number of ``sys.modules`` entries added), ``new_packages``
            (sorted top-level packages whose top-level module was absent in ``before``, without the
            standard library and ``__main__``), ``new_submodule_packages`` (sorted top-level packages
            that were already present in ``before`` and gained submodules, same exclusions),
            ``cuda_initialized_before`` / ``cuda_initialized_after`` and ``ray_initialized_before``
            / ``ray_initialized_after``.
        """
        new = self.modules - before.modules
        before_top = {_top_level(name) for name in before.modules}
        new_top = {_top_level(name) for name in new}
        return {
            "new_modules": len(new),
            "new_packages": sorted(new_top - before_top - _EXCLUDED_TOP_LEVEL),
            "new_submodule_packages": sorted((new_top & before_top) - _EXCLUDED_TOP_LEVEL),
            "cuda_initialized_before": before.cuda_initialized,
            "cuda_initialized_after": self.cuda_initialized,
            "ray_initialized_before": before.ray_initialized,
            "ray_initialized_after": self.ray_initialized,
        }


def take_snapshot() -> EnvironmentSnapshot | None:
    """Take a snapshot; ``None`` on any failure so a caller inside the benchmark harness never breaks."""
    try:
        return EnvironmentSnapshot.take()
    except Exception:
        return None


def audit_since(before: EnvironmentSnapshot | None) -> dict[str, Any] | None:
    """Take a snapshot now and diff it against ``before``.

    Returns ``None`` when ``before`` is ``None`` (the earlier snapshot failed) or when taking or
    diffing the snapshot fails; it never raises.
    """
    if before is None:
        return None
    try:
        return EnvironmentSnapshot.take().diff(before)
    except Exception:
        return None
