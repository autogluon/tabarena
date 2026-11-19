from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabarena.benchmark.result import AGBagResult, BaselineResult, ConfigResult, ExperimentResults
from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.repository.generate_repo import generate_repo_from_results_lst

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


def test_result_baseline_canonicalizes_metric_alias():
    # A no-simulation-artifacts (e.g. outer/no-validation) result must canonicalize the
    # metric alias so it joins the rmse-named baselines in `compare`.
    result = _make_result_baseline()
    result["metric"] = "root_mean_squared_error"
    result["problem_type"] = "regression"
    assert BaselineResult(result=result).result["metric"] == "rmse"


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

    # Use the opt-in torch-free "v2_numpy" temperature scaler so this runs without
    # torch (the default "v2" path requires torch).
    result_obj_calibrated = result_obj.generate_calibrated(method="v2_numpy")
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


def _make_raw_result_ag_bag_per_fold(problem_type: str):
    """A bagged result as the runner writes it for a bag that kept each fold model's validation predictions: 2 folds
    x 2 repeats over 4 rows, no pooled ``pred_proba_dict_val``.
    """
    rng = np.random.default_rng(0)
    shape = {"multiclass": (2, 3), "binary": (2,), "regression": (2,)}[problem_type]
    result = _make_result_baseline()
    result["problem_type"] = problem_type
    result["metric"] = {"multiclass": "log_loss", "binary": "roc_auc", "regression": "rmse"}[problem_type]
    result["metric_error_val"] = 0.5
    y_val = {"multiclass": [2, 1, 2, 0], "binary": [1, 0, 1, 0], "regression": [0.5, 1.5, 2.5, 3.5]}[problem_type]
    pred_test = rng.random((6, 3)).astype(np.float32) if problem_type == "multiclass" else rng.random(6)
    result["simulation_artifacts"] = dict(
        y_val=np.array(y_val),
        y_val_idx=np.array([3, 0, 4, 8]),
        y_test=np.array([0, 2, 1, 1, 1, 1]) if problem_type == "multiclass" else np.array([0, 1, 1, 0, 1, 1]),
        y_test_idx=np.array([1, 2, 5, 6, 9, 7]),
        pred_proba_dict_test={"m1": pred_test},
        label="class",
        bag_info=dict(
            val_idx_per_child=[np.array([0, 2]), np.array([1, 3]), np.array([0, 1]), np.array([2, 3])],
            pred_proba_val_per_child=[rng.random(shape).astype(np.float32) for _ in range(4)],
            pred_proba_test_per_child=[pred_test, pred_test],
        ),
    )
    result["method_metadata"] = dict(
        model_hyperparameters={"param1": 10}, model_cls="DummyModel", model_type="DUMMY", name_prefix="Dummy"
    )
    return result


@pytest.mark.parametrize("problem_type", ["multiclass", "binary", "regression"])
def test_result_ag_bag_pools_the_per_fold_val_predictions(problem_type):
    result = _make_raw_result_ag_bag_per_fold(problem_type)
    per_child = [p.copy() for p in result["simulation_artifacts"]["bag_info"]["pred_proba_val_per_child"]]

    result_obj = BaselineResult.from_dict(result)

    assert isinstance(result_obj, AGBagResult)
    kept = result_obj.y_pred_proba_val_per_child  # the fold models' own predictions, not slices of a pooled array
    assert all(np.array_equal(a, b) for a, b in zip(kept, per_child, strict=True))
    pooled = result_obj.simulation_artifacts["pred_val"]
    assert pooled.dtype == np.float32
    assert pooled.shape == (4, *per_child[0].shape[1:])
    # row 0 is validated by children 0 and 2 (their first rows), row 3 by children 1 and 3 (their second rows)
    np.testing.assert_allclose(pooled[0], (per_child[0][0] + per_child[2][0]) / 2, rtol=1e-6)
    np.testing.assert_allclose(pooled[3], (per_child[1][1] + per_child[3][1]) / 2, rtol=1e-6)
    np.testing.assert_allclose(result_obj.y_pred_proba_val_as_pd.to_numpy(), pooled, rtol=1e-6)


def test_result_ag_bag_per_fold_reaches_the_repo_with_the_pooled_val_predictions():
    result = _make_raw_result_ag_bag_per_fold("multiclass")
    per_child = result["simulation_artifacts"]["bag_info"]["pred_proba_val_per_child"]
    expected_row0 = (per_child[0][0] + per_child[2][0]) / 2

    repo = experiment_batch_runner.repo_from_results(results_lst=[result])

    assert repo.configs() == ["m1"]
    pred_val = repo.predict_val(dataset="d1", fold=0, config="m1")
    np.testing.assert_allclose(pred_val[0], expected_row0, rtol=1e-6)


def test_result_ag_bag_calibrate():
    result = _make_result_ag_bag()
    result_obj = AGBagResult(result=result)

    # See test_result_config_calibrate: torch-free via the opt-in "v2_numpy" scaler.
    result_obj_calibrated = result_obj.generate_calibrated(method="v2_numpy")
    assert result_obj_calibrated.framework == "m1_CAL"


def _legacy_df_d1() -> pd.DataFrame:
    """A complete legacy frame for dataset d1 / tid 0 (all columns from_legacy_df needs)."""
    return pd.DataFrame(
        {
            "dataset": ["d1"],
            "name": ["d1"],
            "tid": [0],
            "problem_type": ["multiclass"],
            "n_folds": [1],
            "n_repeats": [1],
            "n_features": [3],
            "n_classes": [3],
            "NumberOfInstances": [10],
            "n_samples_train_per_fold": [6],
            "n_samples_test_per_fold": [4],
            "target_feature": ["t"],
        }
    )


def _baseline_result_on_tid(tid: int) -> BaselineResult:
    result = _make_result_baseline()
    result["task_metadata"]["tid"] = tid
    return BaselineResult(result=result)


def test_generate_repo_from_results_lst_accepts_collection():
    # A native TaskMetadataCollection produces the same repo as passing its own legacy view.
    coll = TaskMetadataCollection.from_legacy_df(_legacy_df_d1())
    repo_coll = generate_repo_from_results_lst([_baseline_result_on_tid(0)], task_metadata=coll)
    repo_df = generate_repo_from_results_lst([_baseline_result_on_tid(0)], task_metadata=coll.to_legacy_df())
    assert repo_coll.baselines() == repo_df.baselines() == ["m1"]
    assert list(repo_coll.datasets()) == list(repo_df.datasets()) == ["d1"]


def test_generate_repo_from_results_lst_collection_tid_filter_drops_unknown():
    # The tid filter derived from the collection's dataset->tid map drops out-of-suite results.
    coll = TaskMetadataCollection.from_legacy_df(_legacy_df_d1())  # tid 0 only
    with pytest.raises(ValueError, match="No results found after filtering"):
        generate_repo_from_results_lst([_baseline_result_on_tid(999)], task_metadata=coll)


# -- validation_protocol record ----------------------------------------------------------------


def _bagged_record() -> dict:
    return {
        "flavour": "bagged",
        "protocol": {
            "num_bag_folds": 8,
            "num_bag_sets": 1,
            "name": "TabArena-v0.1",
            "arena": "TabArena",
            "enforced": True,
        },
        "key": "8x1",
        "regime": "default",
        "num_bag_folds_resolved": 8,
        "num_bag_sets_resolved": 1,
        "num_bag_folds_fitted": 8,
        "num_bag_sets_fitted": 1,
        "num_child_models": 8,
        "child_oof": False,
    }


def test_df_result_lifts_the_validation_protocol_columns():
    result = _make_result_baseline()
    result["validation_protocol"] = _bagged_record()
    row = BaselineResult(result=result).compute_df_result().iloc[0]
    assert row["vp_key"] == "8x1"
    assert row["vp_flavour"] == "bagged"
    assert row["vp_enforced"] is True or row["vp_enforced"] == True  # noqa: E712 (numpy bool)
    assert (row["vp_num_bag_folds"], row["vp_num_bag_sets"]) == (8, 1)
    assert (row["vp_num_bag_folds_fitted"], row["vp_num_bag_sets_fitted"]) == (8, 1)


def test_df_result_without_record_has_no_vp_columns():
    df = BaselineResult(result=_make_result_baseline()).compute_df_result()
    assert not [c for c in df.columns if c.startswith("vp_")]


def test_info_from_result_carries_the_protocol_key():
    from tabarena.benchmark.result.raw_loading import get_info_from_result

    result = _make_result_baseline()
    assert get_info_from_result(BaselineResult(result=result))["validation_protocol_key"] is None
    result["validation_protocol"] = {**_bagged_record(), "flavour": "system", "protocol": None}
    info = get_info_from_result(BaselineResult(result=result))
    assert info["validation_protocol_key"] == "system"
    assert info["validation_flavour"] == "system"


def test_bag_artifacts_rewrite_the_record_of_the_derived_holdout():
    result = _make_result_ag_bag()
    result["validation_protocol"] = _bagged_record()
    (holdout,) = AGBagResult(result=result).bag_artifacts()
    record = holdout.result["validation_protocol"]
    assert record["flavour"] == "bag-child-holdout"
    assert record["key"] == "bag-child-holdout:8x1"
    assert record["derived_from"] == "m1"
    assert record["child_index"] == 0
    assert (record["num_bag_folds_fitted"], record["num_bag_sets_fitted"], record["num_child_models"]) == (1, 1, 1)
    # The parent keeps its own record.
    assert result["validation_protocol"]["flavour"] == "bagged"
