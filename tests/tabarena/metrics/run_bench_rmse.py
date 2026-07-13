from __future__ import annotations

import numpy as np
from autogluon.core.metrics import root_mean_squared_error
from sklearn.metrics import root_mean_squared_error as sk_root_mean_squared_error

from tabarena.metrics._fast_rmse import fast_rmse
from tabarena.metrics.bench_utils import benchmark_metrics_speed


def benchmark_rmse(num_samples: int, num_repeats: int):
    """Requires compiling C++ code to run `fast_rmse`."""
    print(f"Benchmarking rmse... (num_samples={num_samples}, num_repeats={num_repeats})")
    rng = np.random.default_rng(0)
    y_true = rng.normal(0, 10, num_samples)
    y_pred = y_true + rng.normal(0, 3, num_samples)
    benchmark_metrics = [
        (lambda yt, yp: -sk_root_mean_squared_error(yt, yp), "sk_rmse"),
        (root_mean_squared_error, "ag_rmse"),
        (fast_rmse, "fast_rmse"),
    ]
    benchmark_metrics_speed(
        y_true=y_true,
        y_pred=y_pred,
        benchmark_metrics=benchmark_metrics,
        num_repeats=num_repeats,
        assert_score_isclose=True,
        rtol=1e-9,
    )


if __name__ == "__main__":
    for num_samples, num_repeats in [
        (100, 1000),
        (1000, 1000),
        (10000, 100),
        (100000, 20),
        (1000000, 3),
    ]:
        benchmark_rmse(num_samples=num_samples, num_repeats=num_repeats)
