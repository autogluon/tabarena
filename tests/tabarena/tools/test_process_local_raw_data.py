"""Tests for the metadata verification in `tabarena.tools.process_local_raw_data`.

The raw scan is replaced by a hand-built ``inferred`` dict (the output of ``_infer_from_raw``), so
these exercise only the comparison rows and how ``verify_method_metadata`` gates on them.
"""

from __future__ import annotations

import pytest

from tabarena.models._method_metadata import MethodMetadata
from tabarena.tools.process_local_raw_data import (
    RawMethod,
    _compare_with_provided_metadata,
    _comparison_rows,
    verify_method_metadata,
)

_POST_RENAME_DEFAULT = "Dummy_c1_default_BAG_L1"


def _inferred(**overrides) -> dict:
    """What a single-config GPU run of ``TA-Dummy`` infers to (raw ``TA-`` prefix, ``_default`` infix)."""
    inferred = {
        "inferred_method": "TA-Dummy",
        "inferred_method_type": "config",
        "inferred_compute": "gpu",
        "inferred_ag_key": "TA-DUMMY",
        "inferred_can_hpo": False,
        "is_bag_any": False,
        "inferred_config_default": "TA-Dummy_c1_default_BAG_L1",
    }
    inferred.update(overrides)
    return inferred


def _raw_method(**kwargs) -> RawMethod:
    defaults = dict(
        method="Dummy",
        suite="dummy-suite",
        ag_key="TA-DUMMY",
        model_key="DUMMY",
        compute="gpu",
        can_hpo=False,
        is_bag=False,
    )
    defaults.update(kwargs)
    return RawMethod(path_raw="/raw", method_metadata=MethodMetadata.config(**defaults))


def _rows(method: RawMethod) -> dict[str, tuple]:
    return {
        field: (inferred, provided, severity)
        for field, inferred, provided, severity in _comparison_rows(method, _inferred())
    }


class TestConfigDefaultRow:
    def test_declared_is_checked_post_rename(self):
        row = _rows(_raw_method(config_default=_POST_RENAME_DEFAULT))["config_default"]
        assert row == (_POST_RENAME_DEFAULT, _POST_RENAME_DEFAULT, "error")

    def test_undeclared_is_informational(self):
        """A single-config method may leave the field unset; the row is shown but never gates."""
        assert _rows(_raw_method())["config_default"] == (_POST_RENAME_DEFAULT, None, "info")

    def test_raw_prefix_rows_never_gate(self):
        rows = _rows(_raw_method(config_default=_POST_RENAME_DEFAULT))
        assert rows["method"] == ("TA-Dummy", "Dummy", "warn")
        assert rows["model_key"] == ("TA-DUMMY", "DUMMY", "info")


class TestVerifyMethodMetadata:
    def test_declared_match_passes(self, capsys):
        verify_method_metadata(_raw_method(config_default=_POST_RENAME_DEFAULT), inferred=_inferred())
        out = capsys.readouterr().out
        assert "[verify] OK" in out
        # The raw `TA-` prefix on `method` is reported, never fatal.
        assert "WARNING" in out and "method: inferred='TA-Dummy'" in out

    def test_declared_mismatch_raises(self):
        with pytest.raises(ValueError, match="config_default"):
            verify_method_metadata(_raw_method(config_default="Dummy_c1_BAG_L1"), inferred=_inferred())

    def test_undeclared_passes(self, capsys):
        verify_method_metadata(_raw_method(), inferred=_inferred())
        assert "[verify] OK" in capsys.readouterr().out

    def test_inspect_table_labels_undeclared(self, capsys):
        _compare_with_provided_metadata(_raw_method(), _inferred())
        out = capsys.readouterr().out
        assert "not declared (resolved from the processed repo" in out
        assert "All checked fields match" in out
