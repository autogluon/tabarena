"""Single-file fit test for the TabSwift wrapper -- a dev convenience for iterating locally.

Fits TabSwift on AutoGluon's toy datasets (binary, multiclass, regression) via
``FitHelper``, using the fast smoke hyperparameters (``n_estimators=1``). TabSwift is a
GPU foundation model but derives its device from the allocated GPUs, so this runs on the
GPU when one is available and otherwise falls back to CPU -- the vendored classifier's
CUDA autocast simply disables itself on a CPU-only machine. So, unlike
``test_all_models.py`` (which *skips* GPU-only models without CUDA), this test runs
everywhere.

It carries the ``models`` + ``network`` markers (a real fit, and TabSwift downloads
``swift.ckpt`` from Hugging Face on first use), which the default
``addopts = -m 'not network and not models'`` skips. Run it explicitly:

    pytest tests/tabarena/models/test_tabswift.py -m "models and network" -s

Only skips when TabSwift's optional dependency is not installed (it is vendored, so that
should not happen in a normal checkout).
"""

from __future__ import annotations

import pytest

from tabarena.models import get_model_registry

from .smoke_configs import smoke_for

MODEL = "TabSwift"

def test_tabswift() -> None:
    registry = get_model_registry()
    assert MODEL in registry, f"{MODEL!r} not in registry. Available: {sorted(registry)}"

    info = registry[MODEL]
    cfg = smoke_for(MODEL)

    try:
        from autogluon.tabular.testing import FitHelper

        kwargs: dict = {
            "model_cls": info.model_cls,
            "model_hyperparameters": dict(cfg.hyperparameters),
        }
        if cfg.problem_types is not None:
            kwargs["problem_types"] = list(cfg.problem_types)
        FitHelper.verify_model(**kwargs)
    except ImportError as err:
        pytest.skip(f"{MODEL}: optional dependency not installed ({err})")
