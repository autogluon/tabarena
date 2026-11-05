"""Restricted unpickler for third-party ``results.pkl`` files.

A submitted pickle is untrusted: plain ``pickle.load`` runs whatever callables it names. This loader resolves
only the globals a TabArena result needs (numpy arrays and scalars, AutoGluon's params dicts) and raises
``Refused`` on anything else, so a scan over a submission doubles as a check that no file names other code.
"""

from __future__ import annotations

import gzip
import importlib
import pickle
from pathlib import Path

ALLOWED = {
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("autogluon.core.models.abstract._auxiliary_params", "ParamsAuxDict"),
    ("autogluon.core.models.abstract._mutation_deprecated_dict", "ParamsDict"),
    ("collections", "OrderedDict"),
}


class Refused(Exception):  # noqa: N818
    """A pickle referenced a global outside ``ALLOWED``."""


class SafeUnpickler(pickle.Unpickler):
    """Unpickler that resolves only ``ALLOWED`` globals."""

    def find_class(self, module: str, name: str):
        if (module, name) in ALLOWED:
            return getattr(importlib.import_module(module), name)
        raise Refused(f"{module}.{name}")


def safe_load(path: str | Path):
    """Load a gzip or plain ``results.pkl`` with :class:`SafeUnpickler`."""
    with open(path, "rb") as f:
        gz = f.read(2) == b"\x1f\x8b"
    with gzip.open(path) if gz else open(path, "rb") as f:
        return SafeUnpickler(f).load()
