from __future__ import annotations

import numpy as np
import pytest

from tabarena import EvaluationRepositoryCollection
from tabarena.simulation.context_artificial import load_repo_artificial

from .test_repository import verify_equivalent_repository


def test_repository_collection():
    repo = load_repo_artificial()

    assert repo.datasets() == ["abalone", "ada"]
    assert repo.tids() == [359946, 359944]
    assert repo.n_folds() == 3
    assert repo.folds == [0, 1, 2]
    assert repo.configs() == ["NeuralNetFastAI_r1", "NeuralNetFastAI_r2"]

    datasets = repo.datasets()
    folds = repo.folds
    configs = repo.configs()

    repo_1 = repo.subset(configs=["NeuralNetFastAI_r2"])
    repo_2 = repo.subset(configs=["NeuralNetFastAI_r1"])

    repo_collection = EvaluationRepositoryCollection(repos=[repo_1, repo_2])

    verify_equivalent_repository(repo1=repo, repo2=repo_collection, verify_ensemble=True)

    for dataset in datasets:
        for fold in folds:
            assert repo_collection.get_result_to_repo_idx(dataset=dataset, fold=fold, config="NeuralNetFastAI_r2") == 0
            assert repo_collection.get_result_to_repo_idx(dataset=dataset, fold=fold, config="NeuralNetFastAI_r1") == 1

            predict_test = repo.predict_test_multi(dataset=dataset, fold=fold, configs=configs)
            predict_test_collection = repo_collection.predict_test_multi(dataset=dataset, fold=fold, configs=configs)
            assert np.array_equal(predict_test, predict_test_collection)

            predict_val = repo.predict_val_multi(dataset=dataset, fold=fold, configs=configs)
            predict_val_collection = repo_collection.predict_val_multi(dataset=dataset, fold=fold, configs=configs)
            assert np.array_equal(predict_val, predict_val_collection)

    repo_1 = repo.subset(datasets=["abalone"])
    repo_2 = repo.subset(datasets=["ada"])
    repo_collection = EvaluationRepositoryCollection(repos=[repo_1, repo_2])
    verify_equivalent_repository(repo1=repo, repo2=repo_collection, verify_ensemble=True)
    for config in configs:
        for fold in folds:
            assert repo_collection.get_result_to_repo_idx(dataset="abalone", fold=fold, config=config) == 0
            assert repo_collection.get_result_to_repo_idx(dataset="ada", fold=fold, config=config) == 1

    repo_1 = repo.subset(folds=[2, 0])
    repo_2 = repo.subset(folds=[1])
    repo_collection = EvaluationRepositoryCollection(repos=[repo_1, repo_2])
    verify_equivalent_repository(repo1=repo, repo2=repo_collection, verify_ensemble=True)
    for config in configs:
        for dataset in datasets:
            assert repo_collection.get_result_to_repo_idx(dataset=dataset, fold=0, config=config) == 0
            assert repo_collection.get_result_to_repo_idx(dataset=dataset, fold=1, config=config) == 1
            assert repo_collection.get_result_to_repo_idx(dataset=dataset, fold=2, config=config) == 0


def test_repository_collection_concat_only_configs_with_only_baselines():
    """Verifies that merging repos with only baselines and only configs works as intended."""
    repo_configs = load_repo_artificial(include_baselines=False)
    repo_baselines = load_repo_artificial(include_configs=False, add_baselines_extra=True)

    repo_both = load_repo_artificial(add_baselines_extra=True)

    assert repo_configs.datasets() == ["abalone", "ada"]
    assert repo_configs.tids() == [359946, 359944]
    assert repo_configs.n_folds() == 3
    assert repo_configs.folds == [0, 1, 2]
    assert repo_configs.configs() == ["NeuralNetFastAI_r1", "NeuralNetFastAI_r2"]
    assert repo_configs.baselines() == []

    assert repo_baselines.datasets() == ["a", "abalone", "ada", "b"]
    assert repo_baselines.tids() == [5, 359946, 359944, 6]
    assert repo_baselines.n_folds() == 3
    assert repo_baselines.folds == [0, 1, 2]
    assert repo_baselines.configs() == []
    assert repo_baselines.baselines() == ["b1", "b2", "b_e1"]

    repo_collection = EvaluationRepositoryCollection(repos=[repo_configs, repo_baselines])
    verify_equivalent_repository(repo1=repo_both, repo2=repo_collection, verify_ensemble=True)


def test_repository_collection_concat_only_configs():
    """Verifies that merging repos with only configs works as intended."""
    repo_configs = load_repo_artificial(include_baselines=False)
    repo_configs_1 = repo_configs.subset(datasets=["abalone"])
    repo_configs_2 = repo_configs.subset(datasets=["ada"])

    assert repo_configs_1.datasets() == ["abalone"]
    assert repo_configs_1.tids() == [359946]
    assert repo_configs_1.n_folds() == 3
    assert repo_configs_1.folds == [0, 1, 2]
    assert repo_configs_1.configs() == ["NeuralNetFastAI_r1", "NeuralNetFastAI_r2"]
    assert repo_configs_1.baselines() == []

    assert repo_configs_2.datasets() == ["ada"]
    assert repo_configs_2.tids() == [359944]
    assert repo_configs_2.n_folds() == 3
    assert repo_configs_2.folds == [0, 1, 2]
    assert repo_configs_2.configs() == ["NeuralNetFastAI_r1", "NeuralNetFastAI_r2"]
    assert repo_configs_2.baselines() == []

    repo_collection = EvaluationRepositoryCollection(repos=[repo_configs_1, repo_configs_2])
    verify_equivalent_repository(repo1=repo_configs, repo2=repo_collection, verify_ensemble=True)


def test_repository_collection_concat_only_baselines():
    """Verifies that merging repos with only baselines works as intended."""
    repo_baselines = load_repo_artificial(include_configs=False, add_baselines_extra=True)
    repo_baselines_1 = repo_baselines.subset(datasets=["abalone", "ada"])
    repo_baselines_2 = repo_baselines.subset(datasets=["a", "b"])

    assert repo_baselines_1.datasets() == ["abalone", "ada"]
    assert repo_baselines_1.tids() == [359946, 359944]
    assert repo_baselines_1.n_folds() == 3
    assert repo_baselines_1.folds == [0, 1, 2]
    assert repo_baselines_1.configs() == []
    assert repo_baselines_1.baselines() == ["b1", "b2"]

    assert repo_baselines_2.datasets() == ["a", "b"]
    assert repo_baselines_2.tids() == [5, 6]
    assert repo_baselines_2.n_folds() == 1
    assert repo_baselines_2.folds == [0]
    assert repo_baselines_2.configs() == []
    assert repo_baselines_2.baselines() == ["b1", "b_e1"]

    repo_collection = EvaluationRepositoryCollection(repos=[repo_baselines_1, repo_baselines_2])
    verify_equivalent_repository(repo1=repo_baselines, repo2=repo_collection, verify_ensemble=True)


def test_repository_collection_overlap_raise():
    repo = load_repo_artificial()
    with pytest.raises(AssertionError):
        EvaluationRepositoryCollection(repos=[repo, repo])


def test_repository_collection_overlap_first():
    repo = load_repo_artificial()
    repo_collection = EvaluationRepositoryCollection(repos=[repo, repo], overlap="first")
    verify_equivalent_repository(repo1=repo, repo2=repo_collection, verify_ensemble=True)
    for v in repo_collection._mapping.values():
        assert v == 0


def test_repository_collection_overlap_last():
    repo = load_repo_artificial()
    repo_collection = EvaluationRepositoryCollection(repos=[repo, repo, repo], overlap="last")
    verify_equivalent_repository(repo1=repo, repo2=repo_collection, verify_ensemble=True)
    for v in repo_collection._mapping.values():
        assert v == 2
    repo_collection_nested = EvaluationRepositoryCollection(repos=[repo, repo_collection], overlap="last")
    verify_equivalent_repository(repo1=repo, repo2=repo_collection_nested, verify_ensemble=True)
    for v in repo_collection_nested._mapping.values():
        assert v == 1


def test_repository_collection_single():
    repo = load_repo_artificial()
    repo_collection = EvaluationRepositoryCollection(repos=[repo])
    verify_equivalent_repository(repo1=repo, repo2=repo_collection, verify_ensemble=True)
    for v in repo_collection._mapping.values():
        assert v == 0
    repo_collection_nested = EvaluationRepositoryCollection(repos=[repo_collection])
    verify_equivalent_repository(repo1=repo, repo2=repo_collection_nested, verify_ensemble=True)
    for v in repo_collection_nested._mapping.values():
        assert v == 0
