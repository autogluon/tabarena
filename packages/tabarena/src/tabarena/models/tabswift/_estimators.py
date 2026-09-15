"""The vendored TabSwift estimators' seam for the shared-weights registry.

``TabSwiftClassifier`` and ``TabSwiftRegressor`` (``_vendor``) build their network inside
``_load_model``, which ``fit`` calls right after resolving ``device_`` and right before
``self.model_.to(self.device_)`` (a no-op on a module already on that device type); they pickle
the network with the estimator. The derived classes below take that one hook from
:func:`tabarena.models._shared_estimators.derive_shared_estimator`: with a key the network is the
registry's module, without one the vendored build runs. The wrapper's ``__getstate__`` drops the
shared module from the pickle and reattaches it on load.

Imports the vendored estimators (and torch through them) at module level; the wrapper imports this
module from its fit path only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tabarena.models._shared_estimators import derive_shared_estimator
from tabarena.models.tabswift._vendor.classifier import TabSwiftClassifier
from tabarena.models.tabswift._vendor.regressor import TabSwiftRegressor


def attach_network(estimator: Any, network: Any) -> None:
    """Set the post-conditions of the vendored ``_load_model`` from a shared payload.

    The vendored method converts a string ``model_path`` to a ``Path`` and records it as
    ``model_path_``; the wrapper always passes the key's checkpoint as ``model_path``, so the two
    agree.
    """
    if isinstance(estimator.model_path, str):
        estimator.model_path = Path(estimator.model_path)
    estimator.model_path_ = (
        estimator.model_path if estimator.model_path is not None else Path(estimator._tabarena_key.checkpoint)
    )
    estimator.model_ = network


SharedTabSwiftClassifier = derive_shared_estimator(
    TabSwiftClassifier, module=__name__, name="SharedTabSwiftClassifier", apply=attach_network
)
SharedTabSwiftRegressor = derive_shared_estimator(
    TabSwiftRegressor, module=__name__, name="SharedTabSwiftRegressor", apply=attach_network
)


def estimator_cls(problem_type: str) -> type:
    """The registry-aware estimator class for ``problem_type``."""
    return SharedTabSwiftRegressor if problem_type == "regression" else SharedTabSwiftClassifier
