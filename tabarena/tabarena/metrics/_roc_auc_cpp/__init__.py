from __future__ import annotations

import ctypes
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer


class CppAuc:
    """A python wrapper class for a C++ library, used to load it once and make fast calls after.
    NB be aware of data types accepted, see method docstrings.
    """

    def __init__(self):

        if not self.plugin_path().exists():
            self._compile()
            assert self.plugin_path().exists(), "Missing cpp_auc.so compiled file... " \
                                          "You must first compile the C++ code to use this metric. "
        self._handle = ctypes.CDLL(self.plugin_path())
        self._handle.cpp_auc_ext.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                             ndpointer(ctypes.c_bool, flags="C_CONTIGUOUS"),
                                             ctypes.c_size_t
                                             ]
        self._handle.cpp_auc_ext.restype = ctypes.c_double

    def roc_auc_score(self, y_true: np.array, y_score: np.array) -> float:
        """A method to calculate AUC via C++ lib.

        Args:
            y_true (np.array): 1D numpy array of dtype=np.bool_ as true labels.
            y_score (np.array): 1D numpy array of dtype=np.float32 as probability predictions.

        Returns:
            float: AUC score.
        """
        n = len(y_true)
        return self._handle.cpp_auc_ext(y_score.astype(np.float32), y_true, n)

    def _compile(self):
        # load compilation command
        with open(self.compile_script_path()) as f:
            # remove \n character from the command line
            compile_command = f.readlines()[1].replace("\n", "")
        assert compile_command.startswith("g++")

        # execute compilation command
        print(f'Running "{compile_command}" to compile c++ auc implementation.')
        # Discard g++ stdout. It was previously redirected to a "std.out" file in the
        # *current working directory* (the `cwd` arg below only applies to the subprocess,
        # not to `open`), which littered the launch dir and crashed on read-only
        # filesystems (e.g. Singularity containers) with OSError(EROFS). g++ emits its
        # diagnostics on stderr, which we leave attached to the parent so genuine compile
        # errors remain visible.
        proc = subprocess.Popen(
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
        return Path(__file__).parent / "cpp_auc.so"

    @staticmethod
    def clean_plugin():
        CppAuc.plugin_path().unlink(missing_ok=True)
