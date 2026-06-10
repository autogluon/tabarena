from __future__ import annotations

import copy
import shutil
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tabarena import EvaluationRepository, EvaluationRepositoryCollection
from tabarena.contexts.context_artificial import load_repo_artificial

if TYPE_CHECKING:
    from collections.abc import Callable


def verify_equivalent_repository(
    repo1: EvaluationRepository | EvaluationRepositoryCollection,
    repo2: EvaluationRepository | EvaluationRepositoryCollection,
    exact: bool = True,
    verify_metrics: bool = True,
    verify_predictions: bool = True,
    verify_ensemble: bool = False,
    verify_baselines: bool = True,
    verify_metadata: bool = True,
    verify_configs_hyperparameters: bool = True,
    verify_config_fallback: bool = True,
    backend: str = "native",
):
    assert repo1.folds == repo2.folds
    assert repo1.tids() == repo2.tids()
    assert repo1.configs() == repo2.configs()
    assert repo1.datasets() == repo2.datasets()
    assert sorted(repo1.dataset_fold_config_pairs()) == sorted(repo2.dataset_fold_config_pairs())
    if verify_metrics:
        metrics1 = repo1.metrics().sort_index()
        metrics2 = repo2.metrics().sort_index()
        assert metrics1.equals(metrics2)
    if verify_config_fallback:
        assert repo1._config_fallback == repo2._config_fallback
    if verify_predictions:
        for dataset in repo1.datasets():
            for f in repo1.folds:
                configs = repo1.configs(datasets=[dataset])
                for c in configs:
                    repo1_test = repo1.predict_test(dataset=dataset, config=c, fold=f)
                    repo2_test = repo2.predict_test(dataset=dataset, config=c, fold=f)
                    repo1_val = repo1.predict_val(dataset=dataset, config=c, fold=f)
                    repo2_val = repo2.predict_val(dataset=dataset, config=c, fold=f)
                    if exact:
                        assert np.array_equal(repo1_test, repo2_test)
                        assert np.array_equal(repo1_val, repo2_val)
                    else:
                        assert np.isclose(repo1_test, repo2_test).all()
                        assert np.isclose(repo1_val, repo2_val).all()
                if configs:
                    if exact:
                        assert np.array_equal(
                            repo1.labels_test(dataset=dataset, fold=f), repo2.labels_test(dataset=dataset, fold=f)
                        )
                        assert np.array_equal(
                            repo1.labels_val(dataset=dataset, fold=f), repo2.labels_val(dataset=dataset, fold=f)
                        )
                    else:
                        assert np.isclose(
                            repo1.labels_test(dataset=dataset, fold=f), repo2.labels_test(dataset=dataset, fold=f)
                        ).all()
                        assert np.isclose(
                            repo1.labels_val(dataset=dataset, fold=f), repo2.labels_val(dataset=dataset, fold=f)
                        ).all()
    if verify_ensemble:
        datasets1 = [d for d in repo1.datasets() if len(repo1.configs(datasets=[d])) > 0]
        datasets2 = [d for d in repo2.datasets() if len(repo2.configs(datasets=[d])) > 0]
        if len(datasets1) == 0 and len(datasets2) == 0:
            pass
        else:
            df_out_1, df_ensemble_weights_1 = repo1.evaluate_ensembles(
                datasets=datasets1, ensemble_size=10, backend=backend
            )
            df_out_2, df_ensemble_weights_2 = repo2.evaluate_ensembles(
                datasets=datasets2, ensemble_size=10, backend=backend
            )
            assert df_out_1.equals(df_out_2)
            assert df_ensemble_weights_1.equals(df_ensemble_weights_2)
    if verify_baselines:
        baselines1 = repo1._zeroshot_context.df_baselines
        baselines2 = repo2._zeroshot_context.df_baselines
        if baselines1 is not None:
            columns1 = sorted(baselines1.columns)
            columns2 = sorted(baselines2.columns)
            assert columns1 == columns2
            baselines1 = baselines1[columns1].sort_values(by=columns1, ignore_index=True)
            baselines2 = baselines2[columns1].sort_values(by=columns1, ignore_index=True)
            assert baselines1.equals(baselines2)
        else:
            assert baselines1 == baselines2
    if verify_metadata:
        metadata1 = repo1.task_metadata
        metadata2 = repo2.task_metadata
        if metadata1 is None:
            assert metadata1 == metadata2
        else:
            columns1 = sorted(metadata1.columns)
            columns2 = sorted(metadata2.columns)
            assert columns1 == columns2
            metadata1 = metadata1[columns1].sort_values(by=columns1, ignore_index=True)
            metadata2 = metadata2[columns1].sort_values(by=columns1, ignore_index=True)
            assert metadata1.equals(metadata2)
    if verify_configs_hyperparameters:
        assert repo1.configs_hyperparameters() == repo2.configs_hyperparameters()


def test_repository():
    repo = load_repo_artificial()
    dataset = "abalone"
    tid = repo.dataset_to_tid(dataset)
    assert tid == 359946
    config = "NeuralNetFastAI_r1"  # TODO accessor

    assert repo.datasets() == ["abalone", "ada"]
    assert repo.tids() == [359946, 359944]
    assert repo.n_folds() == 3
    assert repo.folds == [0, 1, 2]
    assert repo.dataset_to_tid(dataset) == 359946
    assert repo.configs() == ["NeuralNetFastAI_r1", "NeuralNetFastAI_r2"]
    # TODO check values, something like [{'framework': 'NeuralNetFastAI_r1', 'time_train_s': 0.1965823616800535, 'metric_error': 0.9764594650133958, 'time_infer_s': 0.3687251706609641, 'bestdiff': 0.8209932298479351, 'loss_rescaled': 0.09710127579306127, 'time_train_s_rescaled': 0.8379449074988039, 'time_infer_s_rescaled': 0.09609840789396307, 'rank': 2.345816964276348, 'score_val': 0.4686512016477016}]
    print(repo.metrics(datasets=[dataset], configs=[config], folds=[2]))
    assert repo.predict_val(dataset=dataset, config=config, fold=2).shape == (123, 25)
    assert repo.predict_test(dataset=dataset, config=config, fold=2).shape == (13, 25)
    assert repo.predict_val_multi(dataset=dataset, fold=2, configs=[config]).shape == (1, 123, 25)
    assert repo.predict_test_multi(dataset=dataset, fold=2, configs=[config]).shape == (1, 13, 25)
    assert repo.labels_val(dataset=dataset, fold=2).shape == (123,)
    assert repo.labels_test(dataset=dataset, fold=2).shape == (13,)
    assert repo.dataset_metadata(dataset=dataset) == {
        "dataset": dataset,
        "task_type": "TaskType.SUPERVISED_CLASSIFICATION",
    }
    result_errors, result_ensemble_weights = repo.evaluate_ensembles(
        datasets=[dataset], configs=[config, config], ensemble_size=5, backend="native"
    )
    assert result_errors.shape == (3, 8)
    assert len(result_ensemble_weights) == 3

    dataset_info = repo.dataset_info(dataset=dataset)
    assert dataset_info["metric"] == "root_mean_squared_error"
    assert dataset_info["problem_type"] == "regression"

    # Test ensemble weights are as expected
    repo.task_name(dataset=dataset, fold=0)
    assert np.allclose(result_ensemble_weights.loc[(dataset, 0)], [1.0, 0.0])

    # Test `max_models_per_type`
    result_errors_w_max_models, result_ensemble_weights_w_max_models = repo.evaluate_ensembles(
        datasets=[dataset],
        configs=[config, config],
        ensemble_size=5,
        backend="native",
        ensemble_kwargs={"max_models_per_type": 1},
    )
    assert result_errors_w_max_models.shape == (3, 8)
    assert len(result_ensemble_weights_w_max_models) == 3
    assert np.allclose(result_ensemble_weights_w_max_models.loc[(dataset, 0)], [1.0, 0.0])

    assert repo.evaluate_ensembles(
        datasets=[dataset], configs=[config, config], ensemble_size=5, folds=[2], backend="native"
    )[0].shape == (1, 8)

    repo: EvaluationRepository = repo.subset(folds=[0, 2])
    assert repo.datasets() == ["abalone", "ada"]
    assert repo.n_folds() == 2
    assert repo.folds == [0, 2]
    assert repo.tids() == [359946, 359944]
    assert repo.configs() == ["NeuralNetFastAI_r1", "NeuralNetFastAI_r2"]
    assert repo.predict_val(dataset=dataset, config=config, fold=2).shape == (123, 25)
    assert repo.predict_test(dataset=dataset, config=config, fold=2).shape == (13, 25)
    assert repo.dataset_metadata(dataset=dataset) == {
        "dataset": dataset,
        "task_type": "TaskType.SUPERVISED_CLASSIFICATION",
    }
    # result_errors, result_ensemble_weights = repo.evaluate_ensemble(datasets=[dataset], configs=[config, config], ensemble_size=5, backend="native")[0],
    assert repo.evaluate_ensembles(datasets=[dataset], configs=[config, config], ensemble_size=5, backend="native")[
        0
    ].shape == (2, 8)
    assert repo.evaluate_ensembles(
        datasets=[dataset], configs=[config, config], ensemble_size=5, folds=[2], backend="native"
    )[0].shape == (1, 8)
    assert repo.evaluate_ensemble(dataset=dataset, fold=2, configs=[config, config], ensemble_size=5)[0].shape == (1, 8)

    repo: EvaluationRepository = repo.subset(folds=[2], datasets=[dataset], configs=[config])
    assert repo.datasets() == ["abalone"]
    assert repo.n_folds() == 1
    assert repo.folds == [2]
    assert repo.tids() == [359946]
    assert repo.configs() == [config]
    assert repo.predict_val(dataset=dataset, config=config, fold=2).shape == (123, 25)
    assert repo.predict_test(dataset=dataset, config=config, fold=2).shape == (13, 25)
    assert repo.dataset_metadata(dataset=dataset) == {
        "dataset": dataset,
        "task_type": "TaskType.SUPERVISED_CLASSIFICATION",
    }
    assert repo.evaluate_ensembles(datasets=[dataset], configs=[config, config], ensemble_size=5, backend="native")[
        0
    ].shape == (1, 8)

    assert repo.evaluate_ensembles(
        datasets=[dataset], configs=[config, config], ensemble_size=5, folds=[2], backend="native"
    )[0].shape == (1, 8)


def test_repository_force_to_dense():
    repo1 = load_repo_artificial()

    assert repo1.folds == [0, 1, 2]
    verify_equivalent_repository(repo1, repo1, verify_ensemble=True)

    repo2 = repo1.force_to_dense()  # no-op because already dense

    verify_equivalent_repository(repo1, repo2, verify_ensemble=True)

    repo2 = repo1.subset()  # no-op because already dense

    verify_equivalent_repository(repo1, repo2, verify_ensemble=True)

    repo2._zeroshot_context.subset_folds([1, 2])
    assert repo2.folds == [1, 2]
    with pytest.raises(AssertionError):
        verify_equivalent_repository(repo1, repo2, verify_ensemble=True)

    repo2 = repo2.force_to_dense()
    with pytest.raises(AssertionError):
        verify_equivalent_repository(repo1, repo2, verify_ensemble=True)

    repo3 = repo1.subset(folds=[1, 2])
    verify_equivalent_repository(repo2, repo3, verify_ensemble=True)


def test_repository_predict_binary_as_multiclass():
    """Test to verify that binary_as_multiclass logic works for binary problem_type."""
    dataset = "abalone"
    configs = ["NeuralNetFastAI_r1", "NeuralNetFastAI_r2"]

    for problem_type in ["binary", "multiclass", "regression"]:
        n_classes = 3 if problem_type == "multiclass" else 2
        repo = load_repo_artificial(n_classes=n_classes, problem_type=problem_type)
        assert repo.dataset_info(dataset)["problem_type"] == problem_type
        _assert_predict_multi_binary_as_multiclass(
            repo=repo, fun=repo.predict_val_multi, dataset=dataset, configs=configs, n_rows=123, n_classes=n_classes
        )
        _assert_predict_multi_binary_as_multiclass(
            repo=repo, fun=repo.predict_test_multi, dataset=dataset, configs=configs, n_rows=13, n_classes=n_classes
        )
        _assert_predict_binary_as_multiclass(
            repo=repo, fun=repo.predict_val, dataset=dataset, config=configs[0], n_rows=123, n_classes=n_classes
        )
        _assert_predict_binary_as_multiclass(
            repo=repo, fun=repo.predict_test, dataset=dataset, config=configs[0], n_rows=13, n_classes=n_classes
        )


def test_repository_subset():
    """Verify repo.subset() works as intended and `inplace` argument works as intended."""
    repo = load_repo_artificial()
    assert repo.datasets() == ["abalone", "ada"]

    repo_og = copy.deepcopy(repo)

    repo_subset = repo.subset(datasets=["abalone"])
    assert repo_subset.datasets() == ["abalone"]
    assert repo.datasets() == ["abalone", "ada"]

    verify_equivalent_repository(repo_og, repo, verify_ensemble=True)

    repo_subset_2 = repo.subset(datasets=["abalone"], inplace=True)

    verify_equivalent_repository(repo_subset, repo_subset_2, verify_ensemble=True)
    verify_equivalent_repository(repo, repo_subset_2, verify_ensemble=True)

    assert repo.datasets() == ["abalone"]
    assert repo_og.datasets() == ["abalone", "ada"]


def test_repository_configs_hyperparameters():
    repo1 = load_repo_artificial()
    repo2 = load_repo_artificial(include_hyperparameters=True)
    verify_equivalent_repository(repo1, repo2, verify_ensemble=True, verify_configs_hyperparameters=False)

    with pytest.raises(AssertionError):
        verify_equivalent_repository(repo1, repo2, verify_configs_hyperparameters=True)

    configs = ["NeuralNetFastAI_r1", "NeuralNetFastAI_r2"]

    configs_type_1 = repo1.configs_type()
    for c in configs:
        assert repo1.config_type(c) is None
        assert configs_type_1[c] is None

    configs_hyperparameters_1 = repo1.configs_hyperparameters()
    for c in configs:
        assert repo1.config_hyperparameters(c) is None
        assert configs_hyperparameters_1[c] is None
    with pytest.raises(AssertionError):
        repo1.autogluon_hyperparameters_dict(configs=configs)

    configs_type_2 = repo2.configs_type()
    for c in configs:
        assert repo2.config_type(c) == "FASTAI"
        assert configs_type_2[c] == "FASTAI"

    configs_hyperparameters_2 = repo2.configs_hyperparameters()
    assert repo2.config_hyperparameters("NeuralNetFastAI_r1") == {"foo": 10, "bar": "hello"}
    assert configs_hyperparameters_2["NeuralNetFastAI_r1"] == {"foo": 10, "bar": "hello"}
    assert repo2.config_hyperparameters("NeuralNetFastAI_r2") == {"foo": 15, "x": "y"}
    assert configs_hyperparameters_2["NeuralNetFastAI_r2"] == {"foo": 15, "x": "y"}

    autogluon_hyperparameters_dict = repo2.autogluon_hyperparameters_dict(configs=configs)
    assert autogluon_hyperparameters_dict == {
        "FASTAI": [
            {"ag_args": {"priority": -1}, "bar": "hello", "foo": 10},
            {"ag_args": {"priority": -2}, "foo": 15, "x": "y"},
        ]
    }

    # reverse order
    autogluon_hyperparameters_dict = repo2.autogluon_hyperparameters_dict(
        configs=["NeuralNetFastAI_r2", "NeuralNetFastAI_r1"]
    )
    assert autogluon_hyperparameters_dict == {
        "FASTAI": [
            {"ag_args": {"priority": -1}, "foo": 15, "x": "y"},
            {"ag_args": {"priority": -2}, "bar": "hello", "foo": 10},
        ]
    }

    # no priority
    autogluon_hyperparameters_dict = repo2.autogluon_hyperparameters_dict(configs=configs, ordered=False)
    assert autogluon_hyperparameters_dict == {
        "FASTAI": [
            {"bar": "hello", "foo": 10},
            {"foo": 15, "x": "y"},
        ]
    }

    repo2_subset = repo2.subset(configs=["NeuralNetFastAI_r2"])
    with pytest.raises(ValueError):
        repo2_subset.autogluon_hyperparameters_dict(configs=configs, ordered=False)
    autogluon_hyperparameters_dict = repo2_subset.autogluon_hyperparameters_dict(configs=["NeuralNetFastAI_r2"])
    assert autogluon_hyperparameters_dict == {
        "FASTAI": [
            {"ag_args": {"priority": -1}, "foo": 15, "x": "y"},
        ]
    }


def test_repository_save_load(tmp_path):
    """Test repo save and load work."""
    repo = load_repo_artificial(include_hyperparameters=True)
    save_path = str(tmp_path / "tmp_repo")
    repo.to_dir(path=save_path)
    repo_loaded = EvaluationRepository.from_dir(path=save_path)
    verify_equivalent_repository(repo1=repo, repo2=repo_loaded, verify_ensemble=True, exact=True)

    repo_float64 = load_repo_artificial(include_hyperparameters=True, dtype=np.float64)
    save_path = str(tmp_path / "tmp_repo_from_float64")
    repo_float64.to_dir(path=save_path)
    repo_loaded_float64 = EvaluationRepository.from_dir(path=save_path)
    # exact=False because the loaded version is float32 and the original is float64
    verify_equivalent_repository(repo1=repo_float64, repo2=repo_loaded_float64, verify_ensemble=True, exact=False)


def test_repository_save_load_with_moving_files(tmp_path):
    """Test repo save and load work when moving files to different directories."""
    save_path = str(tmp_path / "tmp_repo")
    copy_path = str(tmp_path / "tmp_repo_copy")

    repo = load_repo_artificial(include_hyperparameters=True)
    repo.set_config_fallback(config_fallback=repo.configs()[0])

    assert repo._config_fallback == repo.configs()[0]
    with pytest.raises(AssertionError):
        repo.predict_test(dataset="abalone", fold=0, config=repo.configs()[0])

    repo.to_dir(path=save_path)
    repo_loaded = EvaluationRepository.from_dir(path=save_path)
    repo_loaded_mem = EvaluationRepository.from_dir(path=save_path, prediction_format="mem")
    repo_loaded_memopt = EvaluationRepository.from_dir(path=save_path, prediction_format="memopt")

    assert repo._config_fallback == repo_loaded_mem._config_fallback
    assert repo._config_fallback == repo_loaded_memopt._config_fallback

    repo_loaded.predict_test(dataset="abalone", fold=0, config=repo_loaded.configs()[0])
    with pytest.raises(AssertionError):
        repo_loaded_mem.predict_test(dataset="abalone", fold=0, config=repo_loaded.configs()[0])
    with pytest.raises(AssertionError):
        repo_loaded_memopt.predict_test(dataset="abalone", fold=0, config=repo_loaded.configs()[0])
    repo.set_config_fallback(None)
    repo_loaded_mem.set_config_fallback(None)
    repo_loaded_memopt.set_config_fallback(None)

    assert repo_loaded_mem._config_fallback is None
    assert repo_loaded_memopt._config_fallback is None

    repo_loaded_mem.predict_test(dataset="abalone", fold=0, config=repo_loaded.configs()[0])
    repo_loaded_memopt.predict_test(dataset="abalone", fold=0, config=repo_loaded.configs()[0])

    verify_equivalent_repository(
        repo1=repo, repo2=repo_loaded, verify_ensemble=True, exact=True, verify_config_fallback=False
    )
    verify_equivalent_repository(repo1=repo, repo2=repo_loaded_mem, verify_ensemble=True, exact=True)
    verify_equivalent_repository(repo1=repo, repo2=repo_loaded_memopt, verify_ensemble=True, exact=True)

    shutil.copytree(save_path, copy_path)

    repo_loaded_copy = EvaluationRepository.from_dir(path=copy_path)
    verify_equivalent_repository(repo1=repo_loaded, repo2=repo_loaded_copy, verify_ensemble=True, exact=True)

    # verify that the original stops working after deleting the original files
    repo_loaded.predict_test(dataset="abalone", fold=0, config=repo_loaded.configs()[0])
    shutil.rmtree(save_path)
    with pytest.raises(FileNotFoundError):
        repo_loaded.predict_test(dataset="abalone", fold=0, config=repo_loaded.configs()[0])

    # verify in-memory repos don't require the original files
    verify_equivalent_repository(repo1=repo, repo2=repo_loaded_mem, verify_ensemble=True, exact=True)
    verify_equivalent_repository(repo1=repo, repo2=repo_loaded_memopt, verify_ensemble=True, exact=True)

    # verify that the copy works even after deleting the original files
    verify_equivalent_repository(
        repo1=repo, repo2=repo_loaded_copy, verify_ensemble=True, exact=True, verify_config_fallback=False
    )

    # verify that the copy stops working after deleting the copied files
    repo_loaded_copy.predict_test(dataset="abalone", fold=0, config=repo_loaded_copy.configs()[0])
    shutil.rmtree(copy_path)
    with pytest.raises(FileNotFoundError):
        repo_loaded_copy.predict_test(dataset="abalone", fold=0, config=repo_loaded_copy.configs()[0])


def test_repository_baselines_differ():
    """Test when baselines exist for datasets without configs.
    Test that folds are properly automatically filtered upon filtering out all tasks using a given fold.
    """
    repo = load_repo_artificial(add_baselines_extra=True)
    dataset = "abalone"
    config = "NeuralNetFastAI_r1"

    assert repo.datasets() == ["a", "abalone", "ada", "b"]
    assert repo.tids() == [5, 359946, 359944, 6]
    assert repo.folds == [0, 1, 2]
    assert repo.configs() == ["NeuralNetFastAI_r1", "NeuralNetFastAI_r2"]
    assert repo.baselines() == ["b1", "b2", "b_e1"]

    repo: EvaluationRepository = repo.subset(folds=[0, 2])
    assert repo.datasets() == ["a", "abalone", "ada", "b"]
    assert repo.folds == [0, 2]
    assert repo.configs() == ["NeuralNetFastAI_r1", "NeuralNetFastAI_r2"]
    assert repo.baselines() == ["b1", "b2", "b_e1"]

    repo_subset1: EvaluationRepository = repo.subset(folds=[2], datasets=[dataset], configs=[config])
    assert repo_subset1.datasets() == ["abalone"]
    assert repo_subset1.folds == [2]
    assert repo_subset1.configs() == [config]
    assert repo_subset1.baselines() == ["b1", "b2"]

    repo_subset2: EvaluationRepository = repo.subset(datasets=["b"])
    assert repo_subset2.datasets() == ["b"]
    assert repo_subset2.folds == [0]
    assert repo_subset2.configs() == []
    assert repo_subset2.baselines() == ["b1", "b_e1"]


def test_repository_only_baselines():
    """Test when only baselines exist and no configs exist."""
    repo = load_repo_artificial(add_baselines_extra=True, include_configs=False)

    assert repo.datasets() == ["a", "abalone", "ada", "b"]
    assert repo.tids() == [5, 359946, 359944, 6]
    assert repo.folds == [0, 1, 2]
    assert repo.configs() == []
    assert repo.baselines() == ["b1", "b2", "b_e1"]

    repo: EvaluationRepository = repo.subset(folds=[0, 2])
    assert repo.datasets() == ["a", "abalone", "ada", "b"]
    assert repo.folds == [0, 2]
    assert repo.configs() == []
    assert repo.baselines() == ["b1", "b2", "b_e1"]

    repo_subset1: EvaluationRepository = repo.subset(folds=[2], datasets=["abalone"])
    assert repo_subset1.datasets() == ["abalone"]
    assert repo_subset1.folds == [2]
    assert repo_subset1.configs() == []
    assert repo_subset1.baselines() == ["b1", "b2"]

    repo_subset2: EvaluationRepository = repo.subset(datasets=["b"])
    assert repo_subset2.datasets() == ["b"]
    assert repo_subset2.folds == [0]
    assert repo_subset2.configs() == []
    assert repo_subset2.baselines() == ["b1", "b_e1"]


def _assert_predict_multi_binary_as_multiclass(repo, fun: Callable, dataset, configs, n_rows, n_classes):
    problem_type = repo.dataset_info(dataset=dataset)["problem_type"]
    predict_multi = fun(dataset=dataset, fold=2, configs=configs)
    predict_multi_as_multiclass = fun(dataset=dataset, fold=2, configs=configs, binary_as_multiclass=True)
    if problem_type in ["binary", "regression"]:
        assert predict_multi.shape == (2, n_rows)
    else:
        assert predict_multi.shape == (2, n_rows, n_classes)
    if problem_type == "binary":
        assert predict_multi_as_multiclass.shape == (2, n_rows, 2)
        predict_multi_as_multiclass_to_binary = predict_multi_as_multiclass[:, :, 1]
        assert (predict_multi == predict_multi_as_multiclass_to_binary).all()
        assert (predict_multi_as_multiclass[:, :, 0] + predict_multi_as_multiclass[:, :, 1] == 1).all()
    else:
        assert (predict_multi == predict_multi_as_multiclass).all()


def _assert_predict_binary_as_multiclass(repo, fun: Callable, dataset, config, n_rows, n_classes):
    problem_type = repo.dataset_info(dataset=dataset)["problem_type"]
    predict = fun(dataset=dataset, fold=2, config=config)
    predict_as_multiclass = fun(dataset=dataset, fold=2, config=config, binary_as_multiclass=True)
    if problem_type in ["binary", "regression"]:
        assert predict.shape == (n_rows,)
    else:
        assert predict.shape == (n_rows, n_classes)
    if problem_type == "binary":
        assert predict_as_multiclass.shape == (n_rows, 2)
        predict_as_multiclass_to_binary = predict_as_multiclass[:, 1]
        assert (predict == predict_as_multiclass_to_binary).all()
        assert (predict_as_multiclass[:, 0] + predict_as_multiclass[:, 1] == 1).all()
    else:
        assert (predict == predict_as_multiclass).all()
