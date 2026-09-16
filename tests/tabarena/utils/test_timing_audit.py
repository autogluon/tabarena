from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from tabarena.utils import timing_audit
from tabarena.utils.timing_audit import EnvironmentSnapshot, audit_since, take_snapshot

_DIFF_KEYS = {
    "new_modules",
    "new_packages",
    "new_submodule_packages",
    "cuda_initialized_before",
    "cuda_initialized_after",
    "ray_initialized_before",
    "ray_initialized_after",
}


def _snapshot(*modules: str) -> EnvironmentSnapshot:
    return EnvironmentSnapshot(modules=frozenset(modules), cuda_initialized=None, ray_initialized=None)


@pytest.fixture
def throwaway_packages(tmp_path, monkeypatch):
    """Write importable throwaway packages under tmp_path and drop them from sys.modules afterwards.

    Stdlib modules are filtered out of the audit, so the tests need real non-stdlib packages.
    """
    created: list[str] = []

    def make(name: str, submodules: tuple[str, ...] = ("sub",)) -> str:
        pkg_dir = tmp_path / name
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("VALUE = 1\n")
        for sub in submodules:
            (pkg_dir / f"{sub}.py").write_text("VALUE = 2\n")
        created.append(name)
        return name

    monkeypatch.syspath_prepend(str(tmp_path))
    yield make
    for name in created:
        for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
            sys.modules.pop(key, None)


def test_diff_reports_new_top_level_non_stdlib_packages(throwaway_packages):
    name = throwaway_packages("fakepkg_audit_new")
    before = EnvironmentSnapshot.take()
    importlib.import_module(f"{name}.sub")
    diff = EnvironmentSnapshot.take().diff(before)

    assert set(diff) == _DIFF_KEYS
    assert diff["new_packages"] == [name]
    assert name not in diff["new_submodule_packages"]
    # The package and its submodule; nothing else may have been imported in between.
    assert diff["new_modules"] == 2


def test_diff_reports_lazily_loaded_submodules_separately(throwaway_packages):
    name = throwaway_packages("fakepkg_audit_lazy", submodules=("sub_a", "sub_b"))
    importlib.import_module(name)  # the top-level module is present before the snapshot
    before = EnvironmentSnapshot.take()
    importlib.import_module(f"{name}.sub_a")
    importlib.import_module(f"{name}.sub_b")
    diff = EnvironmentSnapshot.take().diff(before)

    assert diff["new_packages"] == []
    assert diff["new_submodule_packages"] == [name]
    assert diff["new_modules"] == 2


def test_diff_with_nothing_new():
    before = EnvironmentSnapshot.take()
    diff = EnvironmentSnapshot.take().diff(before)
    assert diff["new_modules"] == 0
    assert diff["new_packages"] == []
    assert diff["new_submodule_packages"] == []


def test_diff_filters_stdlib_and_main():
    stdlib_name = next(iter(sorted(sys.stdlib_module_names)))
    before = _snapshot("already_here")
    after = _snapshot("already_here", "already_here.lazy", "__main__", stdlib_name, f"{stdlib_name}.sub", "newpkg.x")
    diff = after.diff(before)
    assert diff["new_modules"] == 5
    assert diff["new_packages"] == ["newpkg"]
    assert diff["new_submodule_packages"] == ["already_here"]


def test_diff_is_sorted_and_deduplicated():
    before = _snapshot()
    after = _snapshot("zeta", "zeta.a", "zeta.b", "alpha", "alpha.core")
    assert after.diff(before)["new_packages"] == ["alpha", "zeta"]


def test_diff_carries_the_cuda_and_ray_flags():
    before = EnvironmentSnapshot(modules=frozenset(), cuda_initialized=False, ray_initialized=None)
    after = EnvironmentSnapshot(modules=frozenset(), cuda_initialized=True, ray_initialized=True)
    diff = after.diff(before)
    assert diff["cuda_initialized_before"] is False
    assert diff["cuda_initialized_after"] is True
    assert diff["ray_initialized_before"] is None
    assert diff["ray_initialized_after"] is True


def test_cuda_and_ray_state_read_without_importing(monkeypatch):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))
    fake_ray = SimpleNamespace(is_initialized=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    snapshot = EnvironmentSnapshot.take()
    assert snapshot.cuda_initialized is True
    assert snapshot.ray_initialized is False
    assert "torch" in snapshot.modules and "ray" in snapshot.modules
    # take() only looks the modules up; the fakes are still what sys.modules holds.
    assert sys.modules["torch"] is fake_torch
    assert sys.modules["ray"] is fake_ray


def test_failing_state_queries_read_as_none(monkeypatch):
    def boom():
        raise RuntimeError("driver gone")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_initialized=boom)))
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace())  # no is_initialized attribute
    snapshot = EnvironmentSnapshot.take()
    assert snapshot.cuda_initialized is None
    assert snapshot.ray_initialized is None


def test_absent_torch_and_ray_read_as_none(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "ray", raising=False)
    snapshot = EnvironmentSnapshot.take()
    assert snapshot.cuda_initialized is None
    assert snapshot.ray_initialized is None
    # Taking the snapshot must not have imported either library.
    assert "torch" not in sys.modules
    assert "ray" not in sys.modules


def test_snapshot_is_frozen():
    snapshot = EnvironmentSnapshot.take()
    with pytest.raises(AttributeError):
        snapshot.modules = frozenset()


def test_helpers_never_raise(monkeypatch):
    real_snapshot = EnvironmentSnapshot.take()

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(EnvironmentSnapshot, "take", classmethod(boom))
    assert take_snapshot() is None
    assert audit_since(None) is None
    assert audit_since(real_snapshot) is None

    monkeypatch.undo()
    monkeypatch.setattr(EnvironmentSnapshot, "diff", boom)
    assert take_snapshot() is not None
    assert audit_since(real_snapshot) is None


def test_helpers_round_trip(throwaway_packages):
    name = throwaway_packages("fakepkg_audit_helpers")
    before = take_snapshot()
    assert isinstance(before, EnvironmentSnapshot)
    importlib.import_module(name)
    audit = audit_since(before)
    assert set(audit) == _DIFF_KEYS
    assert audit["new_packages"] == [name]


def test_module_import_is_free_of_torch_and_ray(tmp_path):
    """Importing the audit module must not pull in torch or ray (it would pre-warm the timed predict)."""
    code = textwrap.dedent(
        """
        import sys
        import tabarena.utils.timing_audit
        assert "torch" not in sys.modules, "torch imported"
        assert "ray" not in sys.modules, "ray imported"
        """
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_module_has_no_heavy_import_at_load_time():
    assert timing_audit.__name__ == "tabarena.utils.timing_audit"
    assert not hasattr(timing_audit, "torch")
    assert not hasattr(timing_audit, "ray")
