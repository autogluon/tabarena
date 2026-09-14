"""Tests for the metadata verification in `tabarena.tools.process_local_raw_data`.

The raw scan is replaced by a hand-built ``inferred`` dict (the output of ``_infer_from_raw``), so
these exercise only the comparison rows and how ``verify_method_metadata`` gates on them.
"""

from __future__ import annotations

import pytest

from tabarena.benchmark.validation_protocol import BEYONDARENA_VALIDATION_PROTOCOL, TABARENA_V0PT1_VALIDATION_PROTOCOL
from tabarena.models._method_metadata import MethodMetadata
from tabarena.tools.process_local_raw_data import (
    RawMethod,
    _compare_with_provided_metadata,
    _comparison_rows,
    _fold_histogram,
    _print_method_metadata_snippet,
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
        "validation_protocol_keys": ["8x1"],
        "inferred_validation_protocol": "8x1",
        "validation_flavours": ["bagged"],
        "fold_histogram": [("8x1", 3)],
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
        validation_protocol="8x1",
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


class TestValidationProtocolRow:
    """The recorded protocol must be declared: a raw run with a record fails unless `info.py` says so."""

    def test_declared_match_passes(self, capsys):
        verify_method_metadata(_raw_method(config_default=_POST_RENAME_DEFAULT), inferred=_inferred())
        assert "[verify] OK" in capsys.readouterr().out

    def test_undeclared_with_a_record_fails(self):
        with pytest.raises(ValueError, match="validation_protocol: inferred='8x1' .* provided=None"):
            verify_method_metadata(_raw_method(validation_protocol=None), inferred=_inferred())

    def test_a_wrong_declaration_fails(self):
        with pytest.raises(ValueError, match="validation_protocol"):
            verify_method_metadata(_raw_method(validation_protocol="3x1"), inferred=_inferred())

    def test_legacy_raw_without_record_passes_undeclared(self, capsys):
        legacy = _inferred(validation_protocol_keys=[], inferred_validation_protocol=None, fold_histogram=[])
        verify_method_metadata(_raw_method(validation_protocol=None), inferred=legacy)
        assert "[verify] OK" in capsys.readouterr().out

    def test_row_applies_to_baselines_too(self):
        method = RawMethod(
            path_raw="/raw",
            method_metadata=MethodMetadata.baseline(method="B", suite="s", compute="gpu", validation_protocol="8x1"),
        )
        rows = {f: (i, p, sev) for f, i, p, sev in _comparison_rows(method, _inferred(inferred_method_type="baseline"))}
        assert rows["validation_protocol"] == ("8x1", "8x1", "error")

    def test_snippet_declares_the_recorded_protocol(self, capsys):
        _print_method_metadata_snippet(_raw_method(), _inferred())
        out = capsys.readouterr().out
        assert 'validation_protocol="8x1",' in out

    def test_snippet_omits_the_protocol_for_legacy_raw(self, capsys):
        _print_method_metadata_snippet(_raw_method(), _inferred(inferred_validation_protocol=None))
        assert "validation_protocol" not in capsys.readouterr().out


class TestExpectedValidationProtocol:
    """`--expect-validation-protocol`: the raw results must record the arena's official protocol."""

    def test_matching_expectation_passes(self, capsys):
        verify_method_metadata(
            _raw_method(config_default=_POST_RENAME_DEFAULT),
            inferred=_inferred(),
            expected_validation_protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL,
        )
        assert "[verify] OK" in capsys.readouterr().out

    def test_another_arena_protocol_fails(self):
        with pytest.raises(ValueError, match="cannot be submitted as an official result"):
            verify_method_metadata(
                _raw_method(config_default=_POST_RENAME_DEFAULT),
                inferred=_inferred(),
                expected_validation_protocol=BEYONDARENA_VALIDATION_PROTOCOL,
            )

    def test_a_key_string_is_accepted(self):
        with pytest.raises(ValueError, match="expects '3x1'"):
            verify_method_metadata(
                _raw_method(config_default=_POST_RENAME_DEFAULT),
                inferred=_inferred(),
                expected_validation_protocol="3x1",
            )

    def test_systems_are_exempt(self, capsys):
        method = RawMethod(
            path_raw="/raw",
            method_metadata=MethodMetadata.system(method="Sys", suite="s", compute="gpu"),
        )
        inferred = _inferred(
            inferred_method_type="baseline",
            inferred_method="Sys",
            validation_protocol_keys=["system"],
            inferred_validation_protocol="system",
            validation_flavours=["system"],
            fold_histogram=[],
        )
        verify_method_metadata(
            method, inferred=inferred, expected_validation_protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL
        )
        assert "[verify] OK" in capsys.readouterr().out

    def test_legacy_raw_is_not_checked(self, capsys):
        legacy = _inferred(validation_protocol_keys=[], inferred_validation_protocol=None, fold_histogram=[])
        verify_method_metadata(
            _raw_method(validation_protocol=None),
            inferred=legacy,
            expected_validation_protocol=BEYONDARENA_VALIDATION_PROTOCOL,
        )
        assert "[verify] OK" in capsys.readouterr().out


def test_fold_histogram_labels_the_leave_one_out_child():
    import pandas as pd

    info_df = pd.DataFrame(
        {
            "vp_num_bag_folds": [8, 8, 8, None],
            "vp_num_bag_sets": [1, 1, 1, None],
            "vp_child_oof": [False, False, True, None],
        }
    )
    assert _fold_histogram(info_df) == [("8x1", 2), ("8x1 requested; 1 child via use_child_oof", 1)]
    assert _fold_histogram(pd.DataFrame({"is_bag": [True]})) == []
