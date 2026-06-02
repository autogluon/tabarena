from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabarena.benchmark.result import AGBagResult, BaselineResult, ConfigResult, ExperimentResults

task_metadata = pd.DataFrame(
    {
        "dataset": ["d1"],
        "tid": [0],
    }
)

experiment_batch_runner = ExperimentResults(task_metadata=task_metadata)


def _make_result_baseline():
    return dict(
        framework="m1",
        metric_error=0.5,
        metric="log_loss",
        problem_type="multiclass",
        time_train_s=1.2,
        time_infer_s=1.6,
        task_metadata=dict(
            fold=0,
            repeat=0,
            sample=0,
            split_idx=0,
            name="d1",
        ),
    )


def _make_result_config():
    y_val = np.array([2, 1, 2, 0])
    y_val_idx = np.array([3, 0, 4, 8])

    y_test = np.array([0, 2, 1, 1, 1, 1])
    y_test_idx = np.array([1, 2, 5, 6, 9, 7])

    pred_val = np.array(
        [
            [0.3, 0.2, 0.5],
            [0.3, 0.2, 0.5],
            [0.3, 0.2, 0.5],
            [0.3, 0.2, 0.5],
        ]
    )

    pred_test = np.array(
        [
            [0.3, 0.2, 0.5],
            [0.3, 0.2, 0.5],
            [0.3, 0.2, 0.5],
            [0.3, 0.2, 0.5],
            [0.3, 0.2, 0.5],
            [0.3, 0.2, 0.5],
        ]
    )

    result = _make_result_baseline()
    result["metric_error_val"] = 0.5
    sim_artifacts = dict(
        y_val=y_val,
        y_val_idx=y_val_idx,
        y_test=y_test,
        y_test_idx=y_test_idx,
        pred_val=pred_val,
        pred_test=pred_test,
        ordered_class_labels=["class1", "class2", "class3"],
        ordered_class_labels_transformed=[0, 1, 2],
        num_classes=3,
        label="class",
    )
    method_metadata = dict(
        model_hyperparameters={"param1": 10},
        model_cls="DummyModel",
        model_type="DUMMY",
        name_prefix="Dummy",
    )
    result["simulation_artifacts"] = sim_artifacts
    result["method_metadata"] = method_metadata
    return result


def _make_result_ag_bag():
    result = _make_result_config()

    val_idx_per_child = [
        np.array([0, 2]),
        np.array([1, 3]),
    ]
    pred_val_per_child = [
        np.array(
            [
                [0.3, 0.2, 0.5],
                [0.3, 0.2, 0.5],
            ]
        ),
        np.array(
            [
                [0.3, 0.2, 0.5],
                [0.3, 0.2, 0.5],
            ]
        ),
    ]
    pred_test_per_child = [
        np.array(
            [
                [0.3, 0.2, 0.5],
                [0.3, 0.2, 0.5],
                [0.3, 0.2, 0.5],
                [0.3, 0.2, 0.5],
                [0.3, 0.2, 0.5],
                [0.3, 0.2, 0.5],
            ]
        ),
        np.array(
            [
                [0.3, 0.2, 0.5],
                [0.3, 0.2, 0.5],
                [0.3, 0.2, 0.5],
                [0.3, 0.2, 0.5],
                [0.3, 0.2, 0.5],
                [0.3, 0.2, 0.5],
            ]
        ),
    ]
    bag_info = dict(
        val_idx_per_child=val_idx_per_child,
        pred_val_per_child=pred_val_per_child,
        pred_test_per_child=pred_test_per_child,
    )
    result["simulation_artifacts"]["bag_info"] = bag_info
    return result


def test_result_baseline():
    result = _make_result_baseline()
    BaselineResult(result=result)

    repo = experiment_batch_runner.repo_from_results(results_lst=[result])
    assert repo.baselines() == ["m1"]
    assert repo.configs() == []


def test_result_config():
    result = _make_result_config()
    ConfigResult(result=result)

    repo = experiment_batch_runner.repo_from_results(results_lst=[result])
    assert repo.baselines() == []
    assert repo.configs() == ["m1"]
    assert repo.config_hyperparameters(config="m1") == {"param1": 10}


def test_result_config_calibrate():
    result = _make_result_config()
    result_obj = ConfigResult(result=result)

    try:
        import torch  # noqa: F401
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}",
        )
    result_obj_calibrated = result_obj.generate_calibrated()
    assert result_obj_calibrated.framework == "m1_CAL"


def test_result_ag_bag():
    result = _make_result_ag_bag()
    result_obj = AGBagResult(result=result)

    result_obj_holdout_lst = result_obj.bag_artifacts()
    assert len(result_obj_holdout_lst) == 1
    assert result_obj_holdout_lst[0].framework == "m1_HOLDOUT"

    repo = experiment_batch_runner.repo_from_results(results_lst=[result])
    assert repo.baselines() == []
    assert repo.configs() == ["m1"]
    assert repo.config_hyperparameters(config="m1") == {"param1": 10}


def test_result_ag_bag_calibrate():
    result = _make_result_ag_bag()
    result_obj = AGBagResult(result=result)

    try:
        import torch  # noqa: F401
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}",
        )
    result_obj_calibrated = result_obj.generate_calibrated()
    assert result_obj_calibrated.framework == "m1_CAL"
