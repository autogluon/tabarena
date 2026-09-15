"""Tests for the lazy-import surface of `tabarena.models`.

`tabarena/models/__init__.py` exposes model wrapper classes and the
`MethodMetadata` foundation type via a PEP 562 `__getattr__` so that
`import tabarena.models` stays cheap. These tests lock in that behaviour
so a future eager re-export doesn't silently regress the import cost.

The checks that need a fresh import run in a subprocess. Purging
``tabarena.models`` from ``sys.modules`` inside the test process would leave
every other test module holding stale class objects, which breaks monkeypatches
that target the freshly imported modules.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_in_fresh_interpreter(code: str) -> None:
    """Execute ``code`` in a new interpreter with the cwd kept off ``sys.path``."""
    subprocess.run([sys.executable, "-P", "-c", textwrap.dedent(code)], check=True, timeout=300)  # noqa: S603


def test_method_metadata_lazy_access_works():
    from tabarena.models import MethodMetadata
    from tabarena.models._method_metadata import MethodMetadata as Canonical

    assert MethodMetadata is Canonical


def test_method_metadata_cached_in_module_globals_after_first_access():
    _run_in_fresh_interpreter(
        """
        import importlib
        pkg = importlib.import_module("tabarena.models")
        assert "MethodMetadata" not in pkg.__dict__
        _ = pkg.MethodMetadata  # triggers lazy load
        assert "MethodMetadata" in pkg.__dict__
        """
    )


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
    _run_in_fresh_interpreter(
        """
        import importlib
        import sys
        importlib.import_module("tabarena.models")
        assert "tabarena.models.ebm.model" not in sys.modules
        assert "tabarena.models.ebm" not in sys.modules
        """
    )
