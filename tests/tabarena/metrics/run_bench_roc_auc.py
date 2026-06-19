from __future__ import annotations

from autogluon.core.metrics import roc_auc
from sklearn.metrics import roc_auc_score

from tabarena.metrics._fast_roc_auc import fast_roc_auc_cpp
from tabarena.metrics.bench_utils import benchmark_metrics_speed, generate_y_true_and_y_pred_binary


def benchmark_root_mean_squared_error(num_samples: int, num_repeats: int):
    """Requires compiling C++ code to run `fast_roc_auc_cpp`."""
    print(f"Benchmarking roc_auc... (num_samples={num_samples}, num_repeats={num_repeats}")
    y_true, y_pred = generate_y_true_and_y_pred_binary(num_samples=num_samples)
    benchmark_metrics = [
        (roc_auc_score, "sk_roc_auc"),
        (roc_auc, "ag_roc_auc"),
        (fast_roc_auc_cpp, "fast_roc_auc_cpp"),
    ]
    benchmark_metrics_speed(
        y_true=y_true,
        y_pred=y_pred,
        benchmark_metrics=benchmark_metrics,
        num_repeats=num_repeats,
        assert_score_isclose=True,
        rtol=5e-7,
    )


if __name__ == "__main__":
    for num_samples, num_repeats in [
        (2, 1000),
        (10, 1000),
        (100, 1000),
        (1000, 1000),
        (2000, 1000),
        (5000, 100),
        (10000, 100),
        (100000, 20),
        (1000000, 3),
    ]:
        benchmark_root_mean_squared_error(num_samples=num_samples, num_repeats=num_repeats)
