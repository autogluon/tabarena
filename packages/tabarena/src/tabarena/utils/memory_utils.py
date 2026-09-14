from __future__ import annotations

import os
import sys
import threading
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from typing import Literal

#: Mount point of the proc filesystem. Module-level so tests can point the procfs helpers at a fake tree.
_PROCFS_ROOT = "/proc"


def _procfs_children_supported(pid: int) -> bool:
    """Whether the procfs backend can track ``pid`` on this machine.

    Requires Linux, a kernel built with ``CONFIG_PROC_CHILDREN`` (which provides
    ``/proc/<pid>/task/<tid>/children``) and a readable ``/proc/<pid>/statm``.
    """
    if sys.platform != "linux":
        return False
    if not os.path.exists(f"{_PROCFS_ROOT}/{pid}/task/{pid}/children"):
        return False
    try:
        with open(f"{_PROCFS_ROOT}/{pid}/statm", "rb"):
            pass
    except OSError:
        return False
    return True


def _walk_descendants_procfs(root_pid: int) -> list[int]:
    """Return the pids of every live descendant of ``root_pid`` by walking procfs.

    The walk is an iterative depth-first search that, for each visited process, lists its tasks
    (``/proc/<pid>/task``) and reads ``/proc/<pid>/task/<tid>/children`` for every task. All tasks
    are visited because the ``children`` file is per task: a child is listed under the thread that
    forked it, while the child's own parent pid is the thread-group id. The union over all tasks
    therefore equals the view psutil builds from ``ppid == tgid`` for every pid on the node, without
    touching processes outside this subtree.

    Races are tolerated the way psutil tolerates them: a process that exits between discovery and
    read is skipped (``OSError``), and a ``seen`` set guards against a pid being listed twice while
    the tree is changing (the kernel only guarantees a consistent ``children`` file for a frozen
    process). Descendants of an intermediate process that already exited are reparented and lost by
    both approaches.
    """
    descendants: list[int] = []
    seen = {root_pid}
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        task_dir = f"{_PROCFS_ROOT}/{pid}/task"
        try:
            tids = os.listdir(task_dir)
        except OSError:
            continue
        for tid in tids:
            try:
                with open(f"{task_dir}/{tid}/children", "rb") as f:
                    data = f.read()
            except OSError:
                continue
            for token in data.split():
                try:
                    child = int(token)
                except ValueError:
                    continue
                if child not in seen:
                    seen.add(child)
                    descendants.append(child)
                    stack.append(child)
    return descendants


def _read_rss_procfs(pid: int, page_size: int) -> int | None:
    """Return the resident set size of ``pid`` in bytes from ``/proc/<pid>/statm``, or ``None``.

    The second ``statm`` field is the number of resident pages; multiplied by the page size this is
    exactly what psutil's ``memory_info().rss`` reports on Linux (psutil reads the same file and
    multiplies by ``getpagesize()``). A zombie reads as 0 in both implementations. ``None`` is
    returned when the process is gone or the file cannot be parsed, so callers skip it.
    """
    try:
        with open(f"{_PROCFS_ROOT}/{pid}/statm", "rb") as f:
            data = f.read()
        return int(data.split()[1]) * page_size
    except (OSError, ValueError, IndexError):
        return None


class CpuMemoryTracker:
    """Track the minimum and peak CPU RSS of the current process and its live descendants.

    Every sample sums the resident set size of this process and, when ``include_children`` is set,
    of every live descendant process (Ray daemons and workers started from this process, fold
    workers, subprocesses of model libraries). Samples are taken every ``interval`` seconds from a
    background thread, plus once on entering and once on leaving the context, so a change right
    before the exit is still observed. The tracker only observes memory; it never touches task
    data or predictions.

    Two backends compute the same numbers; the choice is made once in ``__init__``:

    procfs: walks ``/proc/<pid>/task/<tid>/children`` over the tracker's own subtree and reads each
    descendant's ``/proc/<pid>/statm``. The cost scales with the number of threads in the subtree
    (microseconds for a process without children, well under a millisecond for a Ray-bagged job)
    and does not depend on how busy the node is. This is the production backend on Linux.

    psutil: ``psutil.Process.children(recursive=True)`` plus ``memory_info().rss`` per process.
    ``children`` scans ``/proc/<pid>/stat`` for every pid on the node on every sample, so the
    sampler's own CPU use grows with the node's process count and perturbs the timed fit it is
    measuring. Kept as the fallback for platforms without the ``children`` procfs file.

    Both backends report identical values per sample: ``statm`` resident pages times the page size
    is what psutil's ``rss`` reads, and the union of the per-task ``children`` files is the same
    descendant set psutil derives from the parent pid of every process on the node. Both lose
    descendants of an intermediate process that already exited (reparented) and both tolerate
    processes that vanish mid-sample. The descendant set is re-walked on every sample rather than on
    a slower cadence, so short-lived fold workers (Ray tasks whose worker exits after one call)
    stay visible for the whole sampling window.

    Attributes:
        min_rss: Minimum observed total RSS in bytes, set once the context is entered.
        peak_rss: Maximum observed total RSS in bytes.
        start_rss: Total RSS (process plus descendants) at ``__enter__``, the pre-fit baseline.
        start_rss_self: RSS of the main process alone at ``__enter__``. ``start_rss - start_rss_self``
            is the descendant baseline at fit start (for example Ray daemons and prestarted workers),
            so analyses can subtract it from the peak and minimum.
        backend: ``"procfs"`` or ``"psutil"``, the backend chosen in ``__init__``.
    """

    def __init__(
        self,
        interval: float = 0.05,
        include_children: bool = True,
        backend: Literal["auto", "procfs", "psutil"] = "auto",
    ):
        """Create a tracker for the current process.

        Args:
            interval: Sampling interval in seconds.
            include_children: Include all live descendant processes of the current process (Ray
                workers, subprocesses) in the RSS total.
            backend: ``"auto"`` picks procfs when this platform supports it and psutil otherwise;
                ``"procfs"`` and ``"psutil"`` force a backend. ``"procfs"`` raises ``ValueError``
                on a platform where the ``children`` procfs file or the page size is unavailable.

        Raises:
            ValueError: If ``backend`` is unknown, or ``"procfs"`` was requested on a platform
                that cannot provide it.
        """
        if backend not in ("auto", "procfs", "psutil"):
            raise ValueError(f"Unknown backend {backend!r}; expected 'auto', 'procfs' or 'psutil'.")
        self.interval = interval
        self.include_children = include_children

        self._pid = os.getpid()
        self._proc = psutil.Process(self._pid)

        try:
            page_size: int | None = os.sysconf("SC_PAGE_SIZE")
        except (AttributeError, ValueError, OSError):
            page_size = None
        procfs_supported = bool(page_size) and _procfs_children_supported(self._pid)
        if backend == "procfs" and not procfs_supported:
            raise ValueError(
                "The 'procfs' backend needs Linux with /proc/<pid>/task/<tid>/children and a readable page size; "
                "use backend='auto' or 'psutil'."
            )
        self.backend: str = "procfs" if backend == "procfs" or (backend == "auto" and procfs_supported) else "psutil"
        self._page_size: int = page_size if self.backend == "procfs" else 0

        # Set by `__exit__`; the sampler waits on it between samples, so stopping does not
        # have to wait out a sleep of `interval`.
        self._stop = threading.Event()
        self._sampler_thread: threading.Thread | None = None

        # Public stats
        self.min_rss: int | None = None
        self.peak_rss: int = 0
        self.start_rss: int | None = None
        self.start_rss_self: int | None = None

    def _descendant_pids(self) -> list[int]:
        """Return the pids of the live descendants seen by the active backend (empty without children tracking)."""
        if not self.include_children:
            return []
        if self.backend == "procfs":
            return _walk_descendants_procfs(self._pid)
        try:
            return [p.pid for p in self._proc.children(recursive=True)]
        except psutil.Error:
            return []

    def _rss_self(self) -> int:
        """Return the RSS of the main process in bytes."""
        if self.backend == "procfs":
            rss = _read_rss_procfs(self._pid, self._page_size)
            if rss is not None:
                return rss
        return self._proc.memory_info().rss

    def _get_current_rss(self) -> int:
        """Return the current total RSS in bytes for this process and, when enabled, its descendants."""
        if self.backend == "procfs":
            total_rss = self._rss_self()
            # The walk starts at this process's own tasks; when they list no children it ends after
            # one directory listing plus one small read per thread, which is the no-children fast path.
            for pid in self._descendant_pids():
                rss = _read_rss_procfs(pid, self._page_size)
                if rss is not None:
                    total_rss += rss
            return total_rss

        total_rss = 0
        procs = [self._proc]

        if self.include_children:
            try:
                # recursive=True to get workers spawned by Ray, etc.
                children = self._proc.children(recursive=True)
                procs.extend(children)
            except psutil.Error:
                # If we can't query children for some reason, just skip them
                pass

        for p in procs:
            try:
                total_rss += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Process may have exited between children() and memory_info()
                continue

        return total_rss

    def _sample(self) -> None:
        rss = self._get_current_rss()
        if self.min_rss is None or rss < self.min_rss:
            self.min_rss = rss
        self.peak_rss = max(self.peak_rss, rss)

    def _sampler(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def __enter__(self):
        # Record the baseline, then initialize the stats with the current value.
        self.start_rss_self = self._rss_self()
        rss = self._get_current_rss()
        self.min_rss = rss
        self.peak_rss = rss
        self.start_rss = rss

        # Start sampling thread
        self._stop.clear()
        self._sampler_thread = threading.Thread(target=self._sampler, daemon=True)
        self._sampler_thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop sampling; the thread returns as soon as it wakes from its wait.
        self._stop.set()
        if self._sampler_thread is not None:
            self._sampler_thread.join()
        # The last periodic sample can be up to `interval` old: read the end state too.
        self._sample()

        return False  # don't suppress exceptions


class GpuMemoryTracker:
    """GPU memory tracker that automatically disables itself when CUDA is not available."""

    def __init__(self, device=0, interval: float = 0.05):
        """Parameters
        ----------
        device : int or torch.device
            CUDA device index or device object. Ignored if CUDA unavailable.
        interval : float
            Sampling interval in seconds.
        """
        self.interval = interval

        # torch is optional (GPU-only): import it lazily so CPU-only installs and
        # CPU model fits don't require it. Without torch, GPU tracking is disabled.
        try:
            import torch
        except ImportError:
            torch = None
        self._torch = torch

        # Detect whether GPU tracking is possible
        self.enabled = torch is not None and torch.cuda.is_available()

        if self.enabled:
            # Validate device index
            if isinstance(device, int):
                if device < 0 or device >= torch.cuda.device_count():
                    # Invalid device → disable tracking
                    self.enabled = False
                else:
                    device = torch.device(f"cuda:{device}")
            elif isinstance(device, torch.device):
                if device.index is None or device.index >= torch.cuda.device_count():
                    self.enabled = False

        self.device = device if self.enabled else None

        # For sampling thread (see `CpuMemoryTracker._stop`)
        self._stop = threading.Event()
        self._sampler_thread: threading.Thread | None = None

        # Public stats (bytes)
        self.min_allocated: int | None = None
        self.peak_allocated: int | None = None
        self.min_reserved: int | None = None
        self.peak_reserved: int | None = None

    # ----------------------------
    # Helpers
    # ----------------------------
    def _sample_gpu_memory(self) -> tuple[int, int]:
        """Return ``(allocated_bytes, reserved_bytes)`` for ``self.device`` with one allocator query.

        ``torch.cuda.memory_stats_as_nested_dict`` is read once per sample and the nested form of
        ``allocated_bytes.all.current`` / ``reserved_bytes.all.current`` is taken from it. These are
        the same counters ``memory_allocated`` / ``memory_reserved`` return, but each of those calls
        flattens and sorts the whole stats dict, so two of them per sample cost more than one nested
        read. Anything unexpected (an empty dict when the allocator is not initialized on the
        device, a missing key, an older torch without the nested API) falls back to the two
        current calls, so the numbers can never differ from theirs.
        """
        torch = self._torch
        try:
            stats = torch.cuda.memory_stats_as_nested_dict(self.device)
            return stats["allocated_bytes"]["all"]["current"], stats["reserved_bytes"]["all"]["current"]
        except Exception:
            return torch.cuda.memory_allocated(self.device), torch.cuda.memory_reserved(self.device)

    def _sampler(self):
        """Background sampler thread."""
        while not self._stop.is_set():
            allocated, reserved = self._sample_gpu_memory()

            # Update min
            if self.min_allocated is None or allocated < self.min_allocated:
                self.min_allocated = allocated
            if self.min_reserved is None or reserved < self.min_reserved:
                self.min_reserved = reserved

            # Update max
            if self.peak_allocated is None or allocated > self.peak_allocated:
                self.peak_allocated = allocated
            if self.peak_reserved is None or reserved > self.peak_reserved:
                self.peak_reserved = reserved

            self._stop.wait(self.interval)

    # ----------------------------
    # Context Manager
    # ----------------------------
    def __enter__(self):
        """Start tracking if enabled."""
        if not self.enabled:
            # If disabled, return a tracker with all fields staying None
            return self

        torch = self._torch
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        # Initialize from current values
        allocated, reserved = self._sample_gpu_memory()
        self.min_allocated = allocated
        self.peak_allocated = allocated
        self.min_reserved = reserved
        self.peak_reserved = reserved

        # Launch sampler thread
        self._stop.clear()
        self._sampler_thread = threading.Thread(target=self._sampler, daemon=True)
        self._sampler_thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop tracking if enabled."""
        if not self.enabled:
            return False  # No suppression

        # Stop sampler
        self._stop.set()
        if self._sampler_thread is not None:
            self._sampler_thread.join()

        torch = self._torch
        torch.cuda.synchronize(self.device)

        # Update peak values from PyTorch internal counters
        peak_alloc_internal = torch.cuda.max_memory_allocated(self.device)
        peak_res_internal = torch.cuda.max_memory_reserved(self.device)

        if self.peak_allocated is None or peak_alloc_internal > self.peak_allocated:
            self.peak_allocated = peak_alloc_internal
        if self.peak_reserved is None or peak_res_internal > self.peak_reserved:
            self.peak_reserved = peak_res_internal

        return False
