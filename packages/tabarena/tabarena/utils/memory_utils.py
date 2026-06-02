import os
import time
import threading
from typing import Optional

import psutil
import torch


class CpuMemoryTracker:
    """
    Tracks min and peak CPU RAM (RSS) for the current process, optionally
    including all child processes (e.g., Ray workers) on the same node.

    Attributes (bytes, set after context finishes):
        min_rss:  minimum observed total RSS
        peak_rss: maximum observed total RSS
    """

    def __init__(self, interval: float = 0.05, include_children: bool = True):
        """
        Parameters
        ----------
        interval : float, default=0.05
            Sampling interval in seconds.
        include_children : bool, default=True
            If True, include all descendant processes of the current process
            (e.g., Ray workers) in the RSS total.
        """
        self.interval = interval
        self.include_children = include_children

        self._proc = psutil.Process(os.getpid())
        self._stop_flag = False
        self._sampler_thread: Optional[threading.Thread] = None

        # Public stats
        self.min_rss: Optional[int] = None
        self.peak_rss: int = 0

    def _get_current_rss(self) -> int:
        """
        Return current total RSS in bytes for this process (and optionally
        all descendant processes).
        """
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

    def _sampler(self):
        while not self._stop_flag:
            rss = self._get_current_rss()
            if self.min_rss is None or rss < self.min_rss:
                self.min_rss = rss
            if rss > self.peak_rss:
                self.peak_rss = rss
            time.sleep(self.interval)

    def __enter__(self):
        # Initialize stats with current value
        rss = self._get_current_rss()
        self.min_rss = rss
        self.peak_rss = rss

        # Start sampling thread
        self._stop_flag = False
        self._sampler_thread = threading.Thread(target=self._sampler, daemon=True)
        self._sampler_thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop sampling
        self._stop_flag = True
        if self._sampler_thread is not None:
            self._sampler_thread.join()

        return False  # don't suppress exceptions


class GpuMemoryTracker:
    """
    GPU memory tracker that automatically disables itself when CUDA is not available.
    """

    def __init__(self, device=0, interval: float = 0.05):
        """
        Parameters
        ----------
        device : int or torch.device
            CUDA device index or device object. Ignored if CUDA unavailable.
        interval : float
            Sampling interval in seconds.
        """
        self.interval = interval

        # Detect whether GPU tracking is possible
        self.enabled = torch.cuda.is_available()

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

        # For sampling thread
        self._stop_flag = False
        self._sampler_thread: Optional[threading.Thread] = None

        # Public stats (bytes)
        self.min_allocated: Optional[int] = None
        self.peak_allocated: Optional[int] = None
        self.min_reserved: Optional[int] = None
        self.peak_reserved: Optional[int] = None

    # ----------------------------
    # Helpers
    # ----------------------------
    def _sample_gpu_memory(self):
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        return allocated, reserved

    def _sampler(self):
        """Background sampler thread."""
        while not self._stop_flag:
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

            time.sleep(self.interval)

    # ----------------------------
    # Context Manager
    # ----------------------------
    def __enter__(self):
        """Start tracking if enabled."""
        if not self.enabled:
            # If disabled, return a tracker with all fields staying None
            return self

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
        self._stop_flag = False
        self._sampler_thread = threading.Thread(target=self._sampler, daemon=True)
        self._sampler_thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop tracking if enabled."""
        if not self.enabled:
            return False  # No suppression

        # Stop sampler
        self._stop_flag = True
        if self._sampler_thread is not None:
            self._sampler_thread.join()

        torch.cuda.synchronize(self.device)

        # Update peak values from PyTorch internal counters
        peak_alloc_internal = torch.cuda.max_memory_allocated(self.device)
        peak_res_internal = torch.cuda.max_memory_reserved(self.device)

        if self.peak_allocated is None or peak_alloc_internal > self.peak_allocated:
            self.peak_allocated = peak_alloc_internal
        if self.peak_reserved is None or peak_res_internal > self.peak_reserved:
            self.peak_reserved = peak_res_internal

        return False
