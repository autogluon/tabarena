from __future__ import annotations

import numpy as np
import pandas as pd

from tabarena.repository import EvaluationRepository
from tabarena.simulation.context_artificial import load_repo_artificial
from tabarena.simulation.label_cache import (
    LabelFileCache,
    get_active_label_cache,
    shared_label_files,
    zip_entry_signature,
)


def _write(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def test_label_file_cache_reuses_identical_files(tmp_path):
    labels = pd.DataFrame({"y": [1, 0, 1, 1]}, index=[3, 5, 8, 9])
    other = pd.DataFrame({"y": [1, 0, 0, 1]}, index=[3, 5, 8, 9])
    _write(tmp_path / "a" / "label-val.csv.zip", labels)
    _write(tmp_path / "b" / "label-val.csv.zip", labels)
    _write(tmp_path / "c" / "label-val.csv.zip", other)
    _write(tmp_path / "d" / "label-val.csv", labels)
    assert zip_entry_signature(tmp_path / "a" / "label-val.csv.zip") is not None
    assert zip_entry_signature(tmp_path / "d" / "label-val.csv") is None

    cache = LabelFileCache()
    first = cache.read(tmp_path / "a" / "label-val.csv.zip", dataset="ds", fold=0, split="val")
    second = cache.read(tmp_path / "b" / "label-val.csv.zip", dataset="ds", fold=0, split="val")
    assert second is first
    assert first.equals(labels)
    # same key, different content: parsed fresh, and the first entry stays
    third = cache.read(tmp_path / "c" / "label-val.csv.zip", dataset="ds", fold=0, split="val")
    assert third is not first
    assert third.equals(other)
    assert cache.read(tmp_path / "a" / "label-val.csv.zip", dataset="ds", fold=0, split="val") is first
    # other key or split: no reuse
    assert cache.read(tmp_path / "a" / "label-val.csv.zip", dataset="ds", fold=1, split="val") is not first
    assert cache.read(tmp_path / "a" / "label-val.csv.zip", dataset="ds", fold=0, split="test") is not first
    # non-zip files bypass the cache
    plain = cache.read(tmp_path / "d" / "label-val.csv", dataset="ds", fold=0, split="val")
    assert plain is not first
    assert plain.equals(labels)
    assert cache.hits == 2


def test_shared_label_files_across_repos(tmp_path):
    repo = load_repo_artificial()
    repo_1 = repo.subset(configs=["NeuralNetFastAI_r1"])
    repo_2 = repo.subset(configs=["NeuralNetFastAI_r2"])
    repo_1.to_dir(tmp_path / "m1")
    repo_2.to_dir(tmp_path / "m2")

    assert get_active_label_cache() is None
    with shared_label_files() as cache:
        assert get_active_label_cache() is cache
        loaded_1 = EvaluationRepository.from_dir(tmp_path / "m1", verbose=False)
        loaded_2 = EvaluationRepository.from_dir(tmp_path / "m2", verbose=False)
    assert get_active_label_cache() is None
    assert cache.misses == 2 * len(repo.tasks())
    assert cache.hits == 2 * len(repo.tasks())
    for dataset, fold in repo.tasks():
        assert (
            loaded_1._ground_truth._label_val_dict[dataset][fold]
            is loaded_2._ground_truth._label_val_dict[dataset][fold]
        )
        assert (
            loaded_1._ground_truth._label_test_dict[dataset][fold]
            is loaded_2._ground_truth._label_test_dict[dataset][fold]
        )
        assert np.array_equal(
            loaded_1.labels_val(dataset=dataset, fold=fold), repo.labels_val(dataset=dataset, fold=fold)
        )

    # without the block, each repo parses its own copy
    plain_1 = EvaluationRepository.from_dir(tmp_path / "m1", verbose=False)
    plain_2 = EvaluationRepository.from_dir(tmp_path / "m2", verbose=False)
    dataset, fold = repo.tasks()[0]
    assert (
        plain_1._ground_truth._label_val_dict[dataset][fold] is not plain_2._ground_truth._label_val_dict[dataset][fold]
    )


def test_load_groundtruth_threads_match_sequential(tmp_path):
    import threading

    from tabarena.simulation import simulation_context as sc
    from tabarena.simulation.benchmark_context import BenchmarkContext

    repo = load_repo_artificial()
    repo.to_dir(tmp_path)
    context = BenchmarkContext.from_json(str(tmp_path / "context.json"))
    context.benchmark_paths.relative_path = str(tmp_path)
    paths_gt = context.benchmark_paths.zs_gt_full
    zsc = repo._zeroshot_context

    sequential = zsc.load_groundtruth(paths_gt, n_threads=1)
    threaded = zsc.load_groundtruth(paths_gt, n_threads=8)
    assert sequential.dataset_fold_lst() == threaded.dataset_fold_lst()
    for dataset, fold in sequential.dataset_fold_lst():
        assert np.array_equal(sequential.labels_val(dataset, fold), threaded.labels_val(dataset, fold))
        assert np.array_equal(sequential.labels_test(dataset, fold), threaded.labels_test(dataset, fold))
        assert np.array_equal(threaded.labels_val(dataset, fold), repo.labels_val(dataset=dataset, fold=fold))

    assert sc._default_label_load_threads() == sc.LABEL_LOAD_THREADS
    seen = []
    worker = threading.Thread(target=lambda: seen.append(sc._default_label_load_threads()))
    worker.start()
    worker.join()
    assert seen == [1]
