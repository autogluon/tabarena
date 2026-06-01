"""Tests for the shared evaluation building blocks in ``tabarena.evaluation._eval_common``."""

from __future__ import annotations

import os

from tabarena.evaluation._eval_common import init_aux_metric_env, subset_label


def test_subset_label_full_and_sorted():
    assert subset_label([]) == "full"
    assert subset_label(["regression"]) == "regression"
    assert subset_label(["b", "a"]) == "a_b"  # sorted + joined


def test_init_aux_metric_env_sets_and_clears():
    from tabarena.utils.aux_metric import AUX_METRIC_ENV_VAR

    original = os.environ.get(AUX_METRIC_ENV_VAR)
    try:
        init_aux_metric_env({"binary": "balanced_accuracy", "regression": "r2"})
        assert AUX_METRIC_ENV_VAR in os.environ
        assert "balanced_accuracy" in os.environ[AUX_METRIC_ENV_VAR]

        init_aux_metric_env(None)
        assert AUX_METRIC_ENV_VAR not in os.environ
    finally:
        if original is None:
            os.environ.pop(AUX_METRIC_ENV_VAR, None)
        else:
            os.environ[AUX_METRIC_ENV_VAR] = original
