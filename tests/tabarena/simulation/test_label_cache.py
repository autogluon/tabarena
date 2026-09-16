from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabarena.repository import EvaluationRepository
from tabarena.simulation.context_artificial import load_repo_artificial
from tabarena.simulation.label_cache import (
    LabelFileCache,
    get_active_label_cache,
    shared_label_files,
)
from tabarena.simulation.label_files import LABELS_FILENAME, narrow_int_dtype, read_task_labels, write_labels_dat


def _write_legacy(task_dir: Path, val, test):
    task_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y": val}, index=[10 + i for i in range(len(val))]).to_csv(task_dir / "label-val.csv.zip", index=True)
    pd.DataFrame({"y": test}, index=[50 + i for i in range(len(test))]).to_csv(
        task_dir / "label-test.csv.zip", index=True
    )


def test_labels_dat_round_trip(tmp_path):
    val = np.array([0, 1, 1, 2], dtype=np.int64)
    test = np.array([1.5, np.nan, -2.0])
    write_labels_dat(tmp_path, val, test)
    got_val, got_test = read_task_labels(tmp_path)
    assert got_val.dtype == np.int8  # narrowed on disk
    assert np.array_equal(got_val, val)
    assert got_test.dtype == np.float64
    assert np.array_equal(got_test, test, equal_nan=True)
    assert narrow_int_dtype(np.array([0, 300])) == np.int16
    assert narrow_int_dtype(np.array([-5, 5])) == np.int8
    assert narrow_int_dtype(np.array([2**40])) == np.int64
    with pytest.raises(TypeError):
        write_labels_dat(tmp_path / "bad", np.array(["a", "b"]), test)


def test_read_task_labels_legacy_fallback_generates_dat(tmp_path):
    val, test = [1, 0, 1, 1], [0, 0, 1]
    _write_legacy(tmp_path / "a", val, test)
    assert not (tmp_path / "a" / LABELS_FILENAME).exists()
    got_val, got_test = read_task_labels(tmp_path / "a")
    assert np.array_equal(got_val, val) and np.array_equal(got_test, test)
    assert (tmp_path / "a" / LABELS_FILENAME).exists(), "legacy read should write labels.dat"
    # the generated file is what is read from now on, and matches the legacy content
    (tmp_path / "a" / "label-val.csv.zip").unlink()
    got_val2, got_test2 = read_task_labels(tmp_path / "a")
    assert np.array_equal(got_val2, val) and np.array_equal(got_test2, test)
    # without generation nothing is written
    _write_legacy(tmp_path / "b", val, test)
    read_task_labels(tmp_path / "b", generate=False)
    assert not (tmp_path / "b" / LABELS_FILENAME).exists()
    # neither layout present
    with pytest.raises(FileNotFoundError):
        read_task_labels(tmp_path / "c")


def test_label_file_cache_reuses_identical_tasks(tmp_path):
    val, test = np.array([1, 0, 1, 1]), np.array([0, 0, 1])
    write_labels_dat(tmp_path / "a", val, test)
    write_labels_dat(tmp_path / "b", val, test)
    write_labels_dat(tmp_path / "c", val + 1, test)
    _write_legacy(tmp_path / "d", list(val), list(test))

    cache = LabelFileCache()
    first = cache.read_task(tmp_path / "a", dataset="ds", fold=0)
    second = cache.read_task(tmp_path / "b", dataset="ds", fold=0)
    assert second is first
    assert np.array_equal(first[0], val) and np.array_equal(first[1], test)
    # same key, different content: read fresh, and the first entry stays
    third = cache.read_task(tmp_path / "c", dataset="ds", fold=0)
    assert third is not first
    assert np.array_equal(third[0], val + 1)
    assert cache.read_task(tmp_path / "a", dataset="ds", fold=0) is first
    # other key: no reuse
    assert cache.read_task(tmp_path / "a", dataset="ds", fold=1) is not first
    # legacy dir: read through the fallback (and converted), its own key
    legacy = cache.read_task(tmp_path / "d", dataset="ds", fold=2)
    assert np.array_equal(legacy[0], val)
    assert (tmp_path / "d" / LABELS_FILENAME).exists()
    # a second legacy dir with the same labels hits the cache and is converted from it
    _write_legacy(tmp_path / "e", list(val), list(test))
    assert cache.read_task(tmp_path / "e", dataset="ds", fold=2) is legacy
    assert (tmp_path / "e" / LABELS_FILENAME).exists()
    assert cache.hits == 3


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
    assert cache.misses == len(repo.tasks())
    assert cache.hits == len(repo.tasks())
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


def test_ground_truth_accepts_legacy_frame_entries():
    """A GroundTruth unpickled from before the array storage holds DataFrames; reads still work."""
    from tabarena.simulation.ground_truth import GroundTruth

    gt = GroundTruth(label_val_dict={"d": {0: np.array([1, 0])}}, label_test_dict={"d": {0: np.array([0])}})
    gt._label_val_dict["d"][0] = pd.DataFrame({"y": [1, 0]}, index=[7, 9])  # what old pickles contain
    assert gt.labels_val("d", 0).dtype == np.int64
    assert np.array_equal(gt.labels_val("d", 0), [1, 0])
    assert gt.labels_test("d", 0).dtype == np.int64


def test_ground_truth_normalize_flag():
    from tabarena.simulation.ground_truth import GroundTruth

    wide = np.array([0, 1, 1], dtype=np.int64)
    gt = GroundTruth(label_val_dict={"d": {0: wide}}, label_test_dict={"d": {0: wide}})
    assert gt._label_val_dict["d"][0].dtype == np.int8  # normalized (narrowed)
    raw = GroundTruth(label_val_dict={"d": {0: wide}}, label_test_dict={"d": {0: wide}}, normalize=False)
    assert raw._label_val_dict["d"][0] is wide  # taken as given
    assert raw.labels_val("d", 0).dtype == np.int64
    with pytest.raises(TypeError):
        GroundTruth(label_val_dict={"d": {0: pd.Series(wide)}}, label_test_dict={"d": {0: wide}}, normalize=False)
