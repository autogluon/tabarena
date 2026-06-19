"""Tests for the lazy-import surface of `tabarena.models`.

`tabarena/models/__init__.py` exposes model wrapper classes and the
`MethodMetadata` foundation type via a PEP 562 `__getattr__` so that
`import tabarena.models` stays cheap. These tests lock in that behaviour
so a future eager re-export doesn't silently regress the import cost.
"""

from __future__ import annotations

import importlib
import sys


def _purge(prefix: str) -> None:
    """Remove modules from sys.modules so a fresh import is observable."""
    for name in [n for n in sys.modules if n == prefix or n.startswith(prefix + ".")]:
        del sys.modules[name]


def test_method_metadata_lazy_access_works():
    from tabarena.models import MethodMetadata
    from tabarena.models._method_metadata import MethodMetadata as Canonical

    assert MethodMetadata is Canonical


def test_method_metadata_is_same_as_legacy_shim():
    from tabarena.models import MethodMetadata
    from tabarena.nips2025_utils.artifacts.method_metadata import (
        MethodMetadata as Legacy,
    )

    assert MethodMetadata is Legacy


def test_method_metadata_cached_in_module_globals_after_first_access():
    _purge("tabarena.models._method_metadata")
    _purge("tabarena.models")
    pkg = importlib.import_module("tabarena.models")
    assert "MethodMetadata" not in pkg.__dict__
    _ = pkg.MethodMetadata  # triggers lazy load
    assert "MethodMetadata" in pkg.__dict__


def test_method_metadata_listed_in_all():
    import tabarena.models as pkg

    assert "MethodMetadata" in pkg.__all__


def test_all_is_derived_from_lazy_and_eager_sources():
    """`__all__` should be the union of `_LAZY_CLASSES` keys and the
    eager-export tuple. Locked in so a future hand-edit can't drift from
    the single source(s) of truth.
    """
    import tabarena.models as pkg

    expected = sorted({*pkg._LAZY_CLASSES, *pkg._EAGER_EXPORTS})
    assert pkg.__all__ == expected


def test_model_class_module_is_not_loaded_eagerly():
    """The same lazy guarantee for the model wrapper classes."""
    _purge("tabarena.models.ebm")
    _purge("tabarena.models")
    importlib.import_module("tabarena.models")
    assert "tabarena.models.ebm.model" not in sys.modules
    assert "tabarena.models.ebm" not in sys.modules
