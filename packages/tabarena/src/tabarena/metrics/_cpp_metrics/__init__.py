from __future__ import annotations

import ctypes
import subprocess
import time
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer


class CppMetrics:
    """A python wrapper class for a C++ library, used to load it once and make fast calls after.
    NB be aware of data types accepted, see method docstrings.
    """

    def __init__(self):
        if not self.plugin_path().exists():
            self._compile()
            assert self.plugin_path().exists(), (
                "Missing cpp_metrics.so compiled file... You must first compile the C++ code to use this metric. "
            )
        self._handle = ctypes.CDLL(self.plugin_path())
        self._handle.cpp_auc_ext.argtypes = [
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_bool, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
        ]
        self._handle.cpp_auc_ext.restype = ctypes.c_double
        self._handle.cpp_rmse_ext.argtypes = [
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
        ]
        self._handle.cpp_rmse_ext.restype = ctypes.c_double

    def roc_auc_score(self, y_true: np.array, y_score: np.array) -> float:
        """A method to calculate AUC via C++ lib.

        Args:
            y_true (np.array): 1D numpy array of dtype=np.bool_ as true labels.
            y_score (np.array): 1D numpy array of dtype=np.float32 as probability predictions.

        Returns:
            float: AUC score.
        """
        n = len(y_true)
        # The C++ kernel does not mutate its inputs, so no defensive copy is needed:
        # this is zero-copy when y_score is already a contiguous float32 array.
        y_score = np.ascontiguousarray(y_score, dtype=np.float32)
        return self._handle.cpp_auc_ext(y_score, y_true, n)

    def rmse(self, y_true: np.array, y_pred: np.array) -> float:
        """A method to calculate root mean squared error via C++ lib.

        Args:
            y_true (np.array): 1D numpy array of regression targets.
            y_pred (np.array): 1D numpy array of predictions.

        Returns:
            float: RMSE.
        """
        # Zero-copy when the inputs are already contiguous float64 arrays.
        y_true = np.ascontiguousarray(y_true, dtype=np.float64)
        y_pred = np.ascontiguousarray(y_pred, dtype=np.float64)
        return self._handle.cpp_rmse_ext(y_true, y_pred, len(y_true))

    def _compile(self):
        # load compilation command
        with open(self.compile_script_path()) as f:
            # remove \n character from the command line
            compile_command = f.readlines()[1].replace("\n", "")
        assert compile_command.startswith("g++")

        # execute compilation command
        print(f'Running "{compile_command}" to compile the c++ metric implementations.')
        # Discard g++ stdout. It was previously redirected to a "std.out" file in the
        # *current working directory* (the `cwd` arg below only applies to the subprocess,
        # not to `open`), which littered the launch dir and crashed on read-only
        # filesystems (e.g. Singularity containers) with OSError(EROFS). g++ emits its
        # diagnostics on stderr, which we leave attached to the parent so genuine compile
        # errors remain visible.
        proc = subprocess.Popen(  # noqa: S603
            compile_command.split(" "),
            shell=False,
            stdout=subprocess.DEVNULL,
            cwd=Path(__file__).parent,
        )

        # wait command completion
        for _max_trials in range(600):
            if proc.poll() is not None:
                break
            time.sleep(0.1)

        # handle potential failure: timeout or error while compiling
        if proc.poll() is None:
            raise ValueError("Could not compile after 60 secs.")
        if proc.poll() != 0:
            raise ValueError(f"Got an error while compiling, you can try to run manually {self.compile_script_path()}")

    @staticmethod
    def compile_script_path() -> Path:
        return Path(__file__).parent / "compile.sh"

    @staticmethod
    def plugin_path() -> Path:
        return Path(__file__).parent / "cpp_metrics.so"

    @staticmethod
    def clean_plugin():
        CppMetrics.plugin_path().unlink(missing_ok=True)
