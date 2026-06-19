from __future__ import annotations

import numpy as np

from tabarena.metrics._roc_auc_cpp import CppAuc


def test_cpp_auc_compilation():
    CppAuc.clean_plugin()
    assert not CppAuc.plugin_path().exists(), "plugin should have been deleted"

    auc = CppAuc()
    assert CppAuc.plugin_path().exists(), "plugin should have been compiled automatically"

    n_samples = 32
    assert np.isclose(
        auc.roc_auc_score(
            y_true=np.array([i % 2 == 0 for i in range(n_samples)]),
            y_score=np.arange(n_samples) / n_samples + 1,
        ),
        0.5,
    )
