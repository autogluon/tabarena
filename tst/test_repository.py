import copy

from typing import Callable

import numpy as np
import pytest

from tabrepo import EvaluationRepository, EvaluationRepositoryCollection
from tabrepo.contexts.context_artificial import load_repo_artificial


def verify_equivalent_repository(
    repo1: EvaluationRepository | EvaluationRepositoryCollection,
    repo2: EvaluationRepository | EvaluationRepositoryCollection,
    verify_ensemble: bool = False,
    backend: str = "native",
):
    assert repo1.folds == repo2.folds
    assert repo1.tids() == repo2.tids()
    assert repo1.configs() == repo2.configs()
    assert repo1.datasets() == repo2.datasets()
    assert sorted(repo1.dataset_fold_config_pairs()) == sorted(repo2.dataset_fold_config_pairs())
    for dataset in repo1.datasets():
        for f in repo1.folds:
            for c in repo1.configs():
                repo1_test = repo1.predict_test(dataset=dataset, config=c, fold=f)
                repo2_test = repo2.predict_test(dataset=dataset, config=c, fold=f)
                repo1_val = repo1.predict_val(dataset=dataset, config=c, fold=f)
                repo2_val = repo2.predict_val(dataset=dataset, config=c, fold=f)
                assert np.array_equal(repo1_test, repo2_test)
                assert np.array_equal(repo1_val, repo2_val)
            assert np.array_equal(repo1.labels_test(dataset=dataset, fold=f), repo2.labels_test(dataset=dataset, fold=f))
            assert np.array_equal(repo1.labels_val(dataset=dataset, fold=f), repo2.labels_val(dataset=dataset, fold=f))
    if verify_ensemble:
        df_out_1, df_ensemble_weights_1 = repo1.evaluate_ensemble(datasets=repo1.datasets(), ensemble_size=10, backend=backend)
        df_out_2, df_ensemble_weights_2 = repo2.evaluate_ensemble(datasets=repo2.datasets(), ensemble_size=10, backend=backend)
        assert df_out_1.equals(df_out_2)
        assert df_ensemble_weights_1.equals(df_ensemble_weights_2)


def test_repository():
    repo = load_repo_artificial()
    dataset = 'abalone'
    tid = repo.dataset_to_tid(dataset)
    assert tid == 359946
    config = "NeuralNetFastAI_r1"  # TODO accessor

    assert repo.datasets() == ['abalone', 'ada']
    assert repo.tids() == [359946, 359944]
    assert repo.n_folds() == 3
    assert repo.folds == [0, 1, 2]
    assert repo.dataset_to_tid(dataset) == 359946
    assert repo.configs() == ['NeuralNetFastAI_r1', 'NeuralNetFastAI_r2']
    # TODO check values, something like [{'framework': 'NeuralNetFastAI_r1', 'time_train_s': 0.1965823616800535, 'metric_error': 0.9764594650133958, 'time_infer_s': 0.3687251706609641, 'bestdiff': 0.8209932298479351, 'loss_rescaled': 0.09710127579306127, 'time_train_s_rescaled': 0.8379449074988039, 'time_infer_s_rescaled': 0.09609840789396307, 'rank': 2.345816964276348, 'score_val': 0.4686512016477016}]
    print(repo.metrics(datasets=[dataset], configs=[config], folds=[2]))
    assert repo.predict_val(dataset=dataset, config=config, fold=2).shape == (123, 25)
    assert repo.predict_test(dataset=dataset, config=config, fold=2).shape == (13, 25)
    assert repo.predict_val_multi(dataset=dataset, fold=2, configs=[config]).shape == (1, 123, 25)
    assert repo.predict_test_multi(dataset=dataset, fold=2, configs=[config]).shape == (1, 13, 25)
    assert repo.labels_val(dataset=dataset, fold=2).shape == (123,)
    assert repo.labels_test(dataset=dataset, fold=2).shape == (13,)
    assert repo.dataset_metadata(dataset=dataset) == {'dataset': dataset, 'task_type': 'TaskType.SUPERVISED_CLASSIFICATION'}
    result_errors, result_ensemble_weights = repo.evaluate_ensemble(datasets=[dataset], configs=[config, config], ensemble_size=5, backend="native")
    assert result_errors.shape == (3,)
    assert len(result_ensemble_weights) == 3

    dataset_info = repo.dataset_info(dataset=dataset)
    assert dataset_info["metric"] == "root_mean_squared_error"
    assert dataset_info["problem_type"] == "regression"

    # Test ensemble weights are as expected
    task_0 = repo.task_name(dataset=dataset, fold=0)
    assert np.allclose(result_ensemble_weights.loc[(dataset, 0)], [1.0, 0.0])

    # Test `max_models_per_type`
    result_errors_w_max_models, result_ensemble_weights_w_max_models = repo.evaluate_ensemble(
        datasets=[dataset], configs=[config, config], ensemble_size=5, backend="native", ensemble_kwargs={"max_models_per_type": 1}
    )
    assert result_errors_w_max_models.shape == (3,)
    assert len(result_ensemble_weights_w_max_models) == 3
    assert np.allclose(result_ensemble_weights_w_max_models.loc[(dataset, 0)], [1.0, 0.0])

    assert repo.evaluate_ensemble(datasets=[dataset], configs=[config, config],
                                  ensemble_size=5, folds=[2], backend="native")[0].shape == (1,)

    repo: EvaluationRepository = repo.subset(folds=[0, 2])
    assert repo.datasets() == ['abalone', 'ada']
    assert repo.n_folds() == 2
    assert repo.folds == [0, 2]
    assert repo.tids() == [359946, 359944]
    assert repo.configs() == ['NeuralNetFastAI_r1', 'NeuralNetFastAI_r2']
    assert repo.predict_val(dataset=dataset, config=config, fold=2).shape == (123, 25)
    assert repo.predict_test(dataset=dataset, config=config, fold=2).shape == (13, 25)
    assert repo.dataset_metadata(dataset=dataset) == {'dataset': dataset, 'task_type': 'TaskType.SUPERVISED_CLASSIFICATION'}
    # result_errors, result_ensemble_weights = repo.evaluate_ensemble(datasets=[dataset], configs=[config, config], ensemble_size=5, backend="native")[0],
    assert repo.evaluate_ensemble(datasets=[dataset], configs=[config, config], ensemble_size=5, backend="native")[0].shape == (2,)
    assert repo.evaluate_ensemble(datasets=[dataset], configs=[config, config],
                                  ensemble_size=5, folds=[2], backend="native")[0].shape == (1,)

    repo: EvaluationRepository = repo.subset(folds=[2], datasets=[dataset], configs=[config])
    assert repo.datasets() == ['abalone']
    assert repo.n_folds() == 1
    assert repo.folds == [2]
    assert repo.tids() == [359946]
    assert repo.configs() == [config]
    assert repo.predict_val(dataset=dataset, config=config, fold=2).shape == (123, 25)
    assert repo.predict_test(dataset=dataset, config=config, fold=2).shape == (13, 25)
    assert repo.dataset_metadata(dataset=dataset) == {'dataset': dataset, 'task_type': 'TaskType.SUPERVISED_CLASSIFICATION'}
    assert repo.evaluate_ensemble(datasets=[dataset], configs=[config, config], ensemble_size=5, backend="native")[0].shape == (1,)

    assert repo.evaluate_ensemble(datasets=[dataset], configs=[config, config],
                                  ensemble_size=5, folds=[2], backend="native")[0].shape == (1,)


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
    """
    Test to verify that binary_as_multiclass logic works for binary problem_type
    """
    dataset = 'abalone'
    configs = ['NeuralNetFastAI_r1', 'NeuralNetFastAI_r2']

    for problem_type in ["binary", "multiclass", "regression"]:
        if problem_type == "multiclass":
            n_classes = 3
        else:
            n_classes = 2
        repo = load_repo_artificial(n_classes=n_classes, problem_type=problem_type)
        assert repo.dataset_info(dataset)["problem_type"] == problem_type
        _assert_predict_multi_binary_as_multiclass(repo=repo, fun=repo.predict_val_multi, dataset=dataset, configs=configs, n_rows=123, n_classes=n_classes)
        _assert_predict_multi_binary_as_multiclass(repo=repo, fun=repo.predict_test_multi, dataset=dataset, configs=configs, n_rows=13, n_classes=n_classes)
        _assert_predict_binary_as_multiclass(repo=repo, fun=repo.predict_val, dataset=dataset, config=configs[0], n_rows=123, n_classes=n_classes)
        _assert_predict_binary_as_multiclass(repo=repo, fun=repo.predict_test, dataset=dataset, config=configs[0], n_rows=13, n_classes=n_classes)


def test_repository_subset():
    """
    Verify repo.subset() works as intended and `inplace` argument works as intended.
    """
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
    verify_equivalent_repository(repo1, repo2, verify_ensemble=True)

    configs = ['NeuralNetFastAI_r1', 'NeuralNetFastAI_r2']

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
    assert autogluon_hyperparameters_dict == {'FASTAI': [
        {'ag_args': {'priority': -1}, 'bar': 'hello', 'foo': 10},
        {'ag_args': {'priority': -2}, 'foo': 15, 'x': 'y'}
    ]}

    # reverse order
    autogluon_hyperparameters_dict = repo2.autogluon_hyperparameters_dict(configs=['NeuralNetFastAI_r2', 'NeuralNetFastAI_r1'])
    assert autogluon_hyperparameters_dict == {'FASTAI': [
        {'ag_args': {'priority': -1}, 'foo': 15, 'x': 'y'},
        {'ag_args': {'priority': -2}, 'bar': 'hello', 'foo': 10}
    ]}

    # no priority
    autogluon_hyperparameters_dict = repo2.autogluon_hyperparameters_dict(configs=configs, ordered=False)
    assert autogluon_hyperparameters_dict == {'FASTAI': [
        {'bar': 'hello', 'foo': 10},
        {'foo': 15, 'x': 'y'}
    ]}

    repo2_subset = repo2.subset(configs=['NeuralNetFastAI_r2'])
    with pytest.raises(ValueError):
        repo2_subset.autogluon_hyperparameters_dict(configs=configs, ordered=False)
    autogluon_hyperparameters_dict = repo2_subset.autogluon_hyperparameters_dict(configs=['NeuralNetFastAI_r2'])
    assert autogluon_hyperparameters_dict == {'FASTAI': [
        {'ag_args': {'priority': -1}, 'foo': 15, 'x': 'y'}
    ]}


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
