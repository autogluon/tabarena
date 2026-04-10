"""Shared infrastructure and entry point for feature selection benchmark evaluation.

Usage:
    python feature_selection_benchmark_runner.py \
        --mode validity \
        --method_name FSBench__RandomFeatureSelector__5__0__lgbm__3600 \
        --data_foundry_task_id "UserTask|1386903908|anneal/019d3f7b-494a-71fa-8eb2-25d01dfb7792|/work/dlclarge1/purucker-fs_benchmark/.openml/tabarena_tasks" \
        --repeat 0 \
        --noise 1.0 \
        --noise_type gaussian

    python feature_selection_benchmark_runner.py \
        --mode stability \
        --method_name FSBench__RandomFeatureSelector__5__0__lgbm__3600 \
        --data_foundry_task_id "UserTask|1386903908|anneal/019d3f7b-494a-71fa-8eb2-25d01dfb7792|/work/dlclarge1/purucker-fs_benchmark/.openml/tabarena_tasks" \
        --repeat 0 \
        --noise 1.0 \
        --noise_type gaussian
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tabarena.benchmark.feature_selection_methods.feature_selection_benchmark_utils import (
    selector_and_config_from_string,
)
from tabarena.benchmark.task.openml import OpenMLTaskWrapper
from tabflow_slurm.run_tabarena_experiment import _parse_task_id


@dataclass
class FeatureSelectionResult:
    """Result object containing feature selection stability metrics from a single repeat.

    Attributes:
        method: Name of the feature selection method evaluated.
        data_foundry_task_id: Task ID that was used

        original_features: Number of features in the original dataset.
        max_features: Maximum number of features requested by the selector.
        repeat: Repeat number for the FS metric.

        selected_features: Names of selected features from the original_features.
        elapsed_time_fs: Runtime measurement (seconds).

        mode: Evaluation mode ("validity" or "stability").
        mode_kwargs: Additional kwargs specific to the evaluation mode (e.g. noise level for validity mode).
    """

    method: str
    data_foundry_task_id: str

    original_features: list[str]
    max_features: int
    repeat: int

    selected_features: list[int]
    elapsed_time_fs: float

    mode: str
    mode_kwargs: dict[str, Any]


def _augment_dataset(
    *, mode: str, X: pd.DataFrame, y: pd.Series, rng: np.random.Generator, **kwargs
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    original_features = list(X.columns)

    if mode == "validity":
        from validity_fs_metric import get_dataset_for_validity  # noqa: PLC0415

        X = get_dataset_for_validity(X=X, rng=rng, **kwargs)
    elif mode == "stability":
        from stability_fs_metric import get_dataset_for_stability  # noqa: PLC0415

        X, y = get_dataset_for_stability(X=X, y=y, rng=rng, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Shuffle feature order to avoid selector bias.°
    shuffled_cols = rng.permutation(X.columns)
    X = X[shuffled_cols]

    return X, y, original_features


def run_benchmark(  # noqa: D417
    *,
    data_foundry_task_id: str,
    mode: str,
    method_name: str,
    repeat: int,
    **kwargs,
) -> FeatureSelectionResult:
    """Run the FS benchmark loop for a given evaluation mode.

    Args:
        data_foundry_task_id: Task to run on
        mode: "validity" or "stability".
        method_name: Name of the feature selection method to evaluate.
        repeat: Repeat number (sets the seed for repeated experiments).
    """
    # Setup random state
    rng = np.random.default_rng(42 + repeat)

    # Parse task and load data
    task_id = _parse_task_id(data_foundry_task_id)
    task = OpenMLTaskWrapper(
        task=task_id.load_local_openml_task(),
    )
    X, y = task.X, task.y

    # Parse method
    feature_selector, config = selector_and_config_from_string(preprocessing_name=method_name)

    # Augment dataset with new feature based on mode.
    X, y, original_features = _augment_dataset(mode=mode, X=X, y=y, rng=rng, **kwargs)

    # Run Feature Selection
    start_time = time.monotonic()
    feature_selector.fit_transform(
        X=X, y=y, time_limit=config.fs_time, eval_metric=task.eval_metric, problem_type=task.problem_type
    )
    elapsed_time = time.monotonic() - start_time

    # Get Results from feature selector object
    selected_features = feature_selector._selected_features
    max_features = feature_selector.max_features

    # Create result object
    return FeatureSelectionResult(
        method=method_name,
        data_foundry_task_id=data_foundry_task_id,
        original_features=original_features,
        max_features=max_features,
        repeat=repeat,
        selected_features=selected_features,
        elapsed_time_fs=elapsed_time,
        mode=mode,
        mode_kwargs=kwargs,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the FS benchmark runner."""
    parser = argparse.ArgumentParser(description="FS Benchmark Runner")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["validity", "stability"],
        required=True,
        help="Benchmark mode: 'validity' or 'stability'",
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default="FSBench__RandomFeatureSelector__5__0__lgbm__3600",
        help="Feature Selection Method name [default: FSBench__AccuracyFeatureSelector__5__0__lgbm__3600]",
    )
    parser.add_argument(
        "--data_foundry_task_id",
        type=str,
        default="UserTask|1386903908|anneal/019d3f7b-494a-71fa-8eb2-25d01dfb7792|/Users/schaefer.bastian/.openml/tabarena_tasks",
        help="TabArena/OpenML task metadata identifier [default: UserTask|1386903908|anneal/019d3f7b-494a-71fa-8eb2-25d"
             "01dfb7792|/Users/schaefer.bastian/.openml/tabarena_tasks]",
    )
    parser.add_argument("--repeat", type=int, default=0, help="Repeat [default: 0]")

    # Mode specific extra kwargs
    parser.add_argument(
        "--noise",
        type=float,
        default=1.0,
        help="Noise features relative to original count (validity mode only) [default: 1.0]",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        choices=["gaussian", "uniform"],
        default="gaussian",
        help="Type of noise features to add (validity mode only) [default: random]",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    result = run_benchmark(
        data_foundry_task_id=args.data_foundry_task_id,
        mode=args.mode,
        method_name=args.method_name,
        repeat=args.repeat,
        noise=args.noise,
        noise_type=args.noise_type,
    )

    print(result)
    result = pd.DataFrame([result.__dict__])
    path = f"results/{args.mode}_{args.method_name}_{args.data_foundry_task_id.split('|')[3].split('/')[0]}_{args.repeat}.csv"
    result.to_csv(Path(__file__).parent / path, index=False)
