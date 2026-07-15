from __future__ import annotations

import numpy as np

from tabarena.metrics._cpp_metrics import CppMetrics


def test_cpp_metrics_compilation():
    CppMetrics.clean_plugin()
    assert not CppMetrics.plugin_path().exists(), "plugin should have been deleted"

    metrics = CppMetrics()
    assert CppMetrics.plugin_path().exists(), "plugin should have been compiled automatically"

    n_samples = 32
    assert np.isclose(
        metrics.roc_auc_score(
            y_true=np.array([i % 2 == 0 for i in range(n_samples)]),
            y_score=np.arange(n_samples) / n_samples + 1,
        ),
        0.5,
    )
    assert np.isclose(
        metrics.rmse(
            y_true=np.zeros(4),
            y_pred=np.array([1.0, -1.0, 1.0, -1.0]),
        ),
        1.0,
    )
