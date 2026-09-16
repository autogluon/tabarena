from __future__ import annotations

import numpy as np
import pytest

from tabarena.repository import EvaluationRepositoryCollection
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


def test_repository_collection_shares_ground_truth():
    """Repos with interchangeable labels share one array per task after merging."""
    import pickle

    repo = load_repo_artificial()
    repo_1 = repo.subset(configs=["NeuralNetFastAI_r2"])
    repo_2 = repo.subset(configs=["NeuralNetFastAI_r1"])
    gt_1, gt_2 = repo_1._ground_truth, repo_2._ground_truth
    task = ("abalone", 0)
    assert gt_1._label_val_dict[task[0]][task[1]] is not gt_2._label_val_dict[task[0]][task[1]]
    size_separate = len(pickle.dumps([repo_1._ground_truth, repo_2._ground_truth]))

    repo_collection = EvaluationRepositoryCollection(repos=[repo_1, repo_2])
    for dataset in repo.datasets():
        for fold in repo.folds:
            for attr in ("_label_val_dict", "_label_test_dict"):
                shared = getattr(repo_collection._ground_truth, attr)[dataset][fold]
                assert getattr(gt_1, attr)[dataset][fold] is shared
                assert getattr(gt_2, attr)[dataset][fold] is shared
    size_shared = len(pickle.dumps([repo_1._ground_truth, repo_2._ground_truth, repo_collection._ground_truth]))
    assert size_shared < size_separate
    verify_equivalent_repository(repo1=repo, repo2=repo_collection, verify_ensemble=True)

    # the merged dicts are the collection's own: growing it must not touch the first repo
    repo_3 = repo.subset(datasets=["ada"])
    repo_4 = repo.subset(datasets=["abalone"])
    EvaluationRepositoryCollection(repos=[repo_3, repo_4])
    assert repo_3._ground_truth.datasets == ["ada"]

    # differing labels are not shared, and the last repo wins as before
    repo_5 = repo.subset(configs=["NeuralNetFastAI_r1"])
    labels = repo_5._ground_truth._label_val_dict["abalone"][0]
    repo_5._ground_truth._label_val_dict["abalone"][0] = labels.copy() + 1
    repo_collection = EvaluationRepositoryCollection(repos=[repo_1, repo_5])
    merged = repo_collection._ground_truth._label_val_dict["abalone"][0]
    assert merged is repo_5._ground_truth._label_val_dict["abalone"][0]
    assert gt_1._label_val_dict["abalone"][0] is not merged
    assert np.array_equal(
        repo_collection.labels_val(dataset="abalone", fold=0), repo_5.labels_val(dataset="abalone", fold=0)
    )


def test_same_labels():
    from tabarena.repository.evaluation_repository_collection import _same_labels

    a = np.array([1.0, np.nan, 3.0])
    assert _same_labels(a, a.copy())
    assert not _same_labels(a, a.copy() + 1)
    assert not _same_labels(a, a.astype("float32"))
    assert not _same_labels(a, a[:2])
    b = np.array([0, 1, 1], dtype=np.int8)
    assert _same_labels(b, b.copy())
    assert not _same_labels(b, b.astype(np.int64))


def test_concat_results_drop_duplicates_matches_pandas():
    import pandas as pd

    from tabarena.repository.evaluation_repository_collection import _concat_results_drop_duplicates

    repo = load_repo_artificial()
    a = repo.subset(configs=["NeuralNetFastAI_r1"])._zeroshot_context.df_configs
    b = repo.subset(configs=["NeuralNetFastAI_r2"])._zeroshot_context.df_configs
    exact_dup = a.iloc[:2]
    conflict = a.iloc[2:3].copy()
    conflict["metric_error"] += 1.0  # same (framework, dataset, fold), different value: both rows stay
    for frames in ([a, b], [a, b, exact_dup], [a, exact_dup, b, conflict]):
        expected = pd.concat(frames, ignore_index=True).drop_duplicates(ignore_index=True)
        got = _concat_results_drop_duplicates(frames)
        assert got.equals(expected) and got.index.equals(expected.index)


def test_result_index_matches_pairs():
    """Every (dataset, fold, config) result maps to the repo holding it; anything else maps to None."""
    repo = load_repo_artificial()
    repos = [repo.subset(configs=["NeuralNetFastAI_r1"]), repo.subset(datasets=["ada"], configs=["NeuralNetFastAI_r2"])]
    collection = EvaluationRepositoryCollection(repos=repos)
    expected = {}
    for idx, r in enumerate(repos):
        for key in r.dataset_fold_config_pairs():
            expected[key] = idx
    assert len(collection._mapping) == len(expected)
    for (dataset, fold, config), idx in expected.items():
        assert collection.get_result_to_repo_idx(dataset=dataset, fold=fold, config=config) == idx
    assert collection.get_result_to_repo_idx(dataset="abalone", fold=0, config="NeuralNetFastAI_r2") is None
    assert collection.get_result_to_repo_idx(dataset="nope", fold=0, config="NeuralNetFastAI_r1") is None
    assert collection.get_result_to_repo_idx(dataset="ada", fold=99, config="NeuralNetFastAI_r1") is None
    assert sorted(collection._mapping.values()) == sorted(expected.values())
