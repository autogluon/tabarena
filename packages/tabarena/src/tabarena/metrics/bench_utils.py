"""Utilities for benchmarking the speed of evaluation metrics."""
from typing import List, Tuple
import time

import numpy as np
from sklearn.preprocessing import normalize


def generate_y_true_and_y_pred_binary(num_samples, random_seed=0):
    np.random.seed(seed=random_seed)
    y_true = np.random.randint(0, 2, num_samples).astype(np.bool_)
    y_pred = np.random.rand(num_samples).astype(np.float32)
    return y_true, y_pred


def generate_y_true_and_y_pred_proba(num_samples, num_classes, random_seed=0):
    np.random.seed(seed=random_seed)
    y_true = np.random.randint(0, num_classes, num_samples).astype(np.uint16)
    y_pred = np.random.rand(num_samples, num_classes)
    y_pred = normalize(y_pred, axis=1, norm='l1').astype(np.float32)
    return y_true, y_pred


def generate_y_true_and_y_pred_proba_bulk(num_configs, num_samples, num_classes, random_seed=0):
    np.random.seed(seed=random_seed)
    y_true = np.random.randint(0, num_classes, num_samples)
    if num_classes == 2:
        y_pred_bulk = np.array([np.random.rand(num_samples) for _ in range(num_configs)])
    else:
        y_pred_bulk = [normalize(np.random.rand(num_samples, num_classes), axis=1, norm='l1') for _ in range(num_configs)]
        y_pred_bulk = np.array(y_pred_bulk)
    return y_true, y_pred_bulk


def get_eval_speed(*,
                   eval_metric: callable,
                   y_true: np.array,
                   y_pred: np.array,
                   num_repeats: int) -> Tuple[float, float]:
    score = None
    ts = time.time()
    for _ in range(num_repeats):
        score = eval_metric(y_true, y_pred)
    te = time.time()
    time_average_s = (te - ts) / num_repeats
    return time_average_s, score


def print_benchmark_result(*,
                           baseline_speed: float,
                           time_average_s: float,
                           score: float,
                           func_name: str):
    relative_speedup = baseline_speed / time_average_s
    print(f'\tTime = {time_average_s * 1000:.4f} ms\t'
          f'| Rel Speedup = {relative_speedup:.1f}x\t'
          f'| Score = {score}\t'
          f'| {func_name}')


def benchmark_metrics_speed(y_true: np.array,
                            y_pred: np.array,
                            benchmark_metrics: List[Tuple[callable, str]],
                            num_repeats: int,
                            assert_score_isclose: bool = True,
                            rtol: float = 1e-7) -> Tuple[float, float]:
    baseline_speed = None
    baseline_score = None

    for eval_metric, func_name in benchmark_metrics:
        time_average_s, score = get_eval_speed(
            eval_metric=eval_metric,
            y_true=y_true,
            y_pred=y_pred,
            num_repeats=num_repeats,
        )
        if baseline_speed is None:
            baseline_speed = time_average_s
        if baseline_score is None:
            baseline_score = score

        print_benchmark_result(baseline_speed=baseline_speed,
                               time_average_s=time_average_s,
                               score=score,
                               func_name=func_name)
        if assert_score_isclose:
            np.testing.assert_allclose(baseline_score, score, rtol=rtol)
    return baseline_speed, baseline_score
