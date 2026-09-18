from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tabarena.repository import EvaluationRepository
from tabarena.repository.evaluation_repository import write_processed_context
from tabarena.simulation.context_artificial import load_repo_artificial
from tabarena.simulation.label_cache import shared_label_files
from tabarena.simulation.label_files import LABELS_FILENAME, write_labels_dat
from tabarena.simulation.task_data import (
    METADATA_FILENAME,
    TASKS_FILENAME,
    TaskData,
    decode_dataset_tasks,
    encode_dataset_tasks,
    load_dataset_tasks,
    read_tasks_dat,
)


def _unconsolidate(path: Path) -> None:
    """Turn a consolidated artifact back into the per-task layout (metadata.json + labels.dat)."""
    for dataset_dir in (path / "model_predictions").iterdir():
        dataset, tasks = read_tasks_dat(dataset_dir)
        for fold, task in tasks.items():
            with open(dataset_dir / str(fold) / METADATA_FILENAME, "w") as f:
                json.dump({**task.metadata, "dataset": dataset, "fold": fold}, f)
            write_labels_dat(dataset_dir / str(fold), task.labels_val, task.labels_test)
        (dataset_dir / TASKS_FILENAME).unlink()


def _equal(a: EvaluationRepository, b: EvaluationRepository) -> None:
    assert sorted(a.tasks()) == sorted(b.tasks())
    for dataset, fold in a.tasks():
        assert np.array_equal(a.labels_val(dataset, fold), b.labels_val(dataset, fold))
        assert np.array_equal(a.labels_test(dataset, fold), b.labels_test(dataset, fold))
        assert np.array_equal(
            a.predict_val_multi(dataset, fold, a.configs()), b.predict_val_multi(dataset, fold, a.configs())
        )


def test_encode_decode_round_trip():
    tasks = {
        0: TaskData(
            {"models": ["a", "b"], "pred_val_shape": [2, 3], "pred_test_shape": [2, 2], "dtype": "float32"},
            np.array([0, 1, 1]),
            np.array([1, 0]),
        ),
        2: TaskData(
            {"models": ["a"], "pred_val_shape": [1, 2, 3], "pred_test_shape": [1, 1, 3], "dtype": "float32"},
            np.array([0.5, np.nan]),
            np.array([2.0]),
        ),
    }
    dataset, decoded = decode_dataset_tasks(encode_dataset_tasks("ds", tasks))
    assert dataset == "ds" and sorted(decoded) == [0, 2]
    assert decoded[0].metadata == tasks[0].metadata
    assert decoded[0].labels_val.dtype == np.int8 and np.array_equal(decoded[0].labels_val, [0, 1, 1])
    assert np.array_equal(decoded[2].labels_val, [0.5, np.nan], equal_nan=True)
    with pytest.raises(TypeError):
        encode_dataset_tasks("ds", {0: TaskData(tasks[0].metadata, np.array(["x"]), np.array([1]))})


def test_to_dir_writes_one_task_file_per_dataset(tmp_path):
    repo = load_repo_artificial()
    repo.to_dir(tmp_path)
    root = tmp_path / "model_predictions"
    assert sorted(p.parent.name for p in root.glob(f"*/{TASKS_FILENAME}")) == sorted(repo.datasets())
    assert not list(root.glob(f"*/*/{METADATA_FILENAME}"))
    assert not list(root.glob(f"*/*/{LABELS_FILENAME}"))
    assert len(list(root.glob("*/*/pred-val.dat"))) == len(repo.tasks())
    with open(tmp_path / "context.json") as f:
        ctx = json.load(f)
    assert all(p.endswith(TASKS_FILENAME) for p in ctx["benchmark_paths"]["zs_gt"])
    assert not any(p.endswith(METADATA_FILENAME) for p in ctx["benchmark_paths"]["zs_pp"])
    _equal(EvaluationRepository.from_dir(tmp_path, verbose=False), repo)
    # validate=True checks the per-dataset file
    EvaluationRepository.from_dir(tmp_path, verbose=False, validate=True)


def test_per_task_layout_loads_and_is_consolidated_on_read(tmp_path):
    repo = load_repo_artificial()
    repo.to_dir(tmp_path)
    _unconsolidate(tmp_path)
    root = tmp_path / "model_predictions"
    assert not list(root.glob(f"*/{TASKS_FILENAME}"))
    loaded = EvaluationRepository.from_dir(tmp_path, verbose=False)
    _equal(loaded, repo)
    assert sorted(p.parent.name for p in root.glob(f"*/{TASKS_FILENAME}")) == sorted(repo.datasets())
    # per-task files are left in place by a read; the next load takes the fast path
    assert len(list(root.glob(f"*/*/{METADATA_FILENAME}"))) == len(repo.tasks())
    _equal(EvaluationRepository.from_dir(tmp_path, verbose=False), repo)


def test_incomplete_dataset_is_not_consolidated(tmp_path):
    repo = load_repo_artificial()
    repo.to_dir(tmp_path)
    _unconsolidate(tmp_path)
    dataset, fold = repo.tasks()[0]
    (tmp_path / "model_predictions" / dataset / str(fold) / METADATA_FILENAME).unlink()
    with pytest.raises(FileNotFoundError):
        EvaluationRepository.from_dir(tmp_path, verbose=False)
    assert not (tmp_path / "model_predictions" / dataset / TASKS_FILENAME).exists()


def test_stale_task_file_falls_back_for_missing_folds(tmp_path):
    repo = load_repo_artificial()
    repo.to_dir(tmp_path)
    _unconsolidate(tmp_path)
    dataset_dir = tmp_path / "model_predictions" / repo.datasets()[0]
    folds = sorted(int(p.name) for p in dataset_dir.iterdir() if p.name.isdigit())
    partial = load_dataset_tasks(dataset_dir, folds[:-1], generate=True)  # writes tasks.dat without the last fold
    assert sorted(read_tasks_dat(dataset_dir)[1]) == folds[:-1]
    full = load_dataset_tasks(dataset_dir, folds, generate=True)
    assert sorted(full) == folds and sorted(read_tasks_dat(dataset_dir)[1]) == folds
    assert np.array_equal(full[folds[0]].labels_val, partial[folds[0]].labels_val)


def test_parallel_task_slices_consolidate_at_finalization(tmp_path):
    repo = load_repo_artificial()
    slice_a = repo.subset(folds=[0])
    slice_b = repo.subset(folds=[1, 2])
    slice_a.to_dir_task_data(tmp_path)
    slice_b.to_dir_task_data(tmp_path)
    root = tmp_path / "model_predictions"
    assert len(list(root.glob(f"*/*/{METADATA_FILENAME}"))) == len(repo.tasks())
    write_processed_context(
        path=tmp_path,
        zeroshot_context=repo._zeroshot_context,
        dataset_fold_lst_pp=repo._tabular_predictions.dataset_fold_lst(),
        dataset_fold_lst_gt=repo._ground_truth.dataset_fold_lst(),
    )
    assert not list(root.glob(f"*/*/{METADATA_FILENAME}"))
    assert len(list(root.glob(f"*/{TASKS_FILENAME}"))) == len(repo.datasets())
    _equal(EvaluationRepository.from_dir(tmp_path, verbose=False), repo)


def test_label_cache_shares_arrays_between_task_files(tmp_path):
    repo = load_repo_artificial()
    repo.subset(configs=["NeuralNetFastAI_r1"]).to_dir(tmp_path / "m1")
    repo.subset(configs=["NeuralNetFastAI_r2"]).to_dir(tmp_path / "m2")
    with shared_label_files() as cache:
        a = EvaluationRepository.from_dir(tmp_path / "m1", verbose=False)
        b = EvaluationRepository.from_dir(tmp_path / "m2", verbose=False)
    assert cache.misses == len(repo.tasks()) and cache.hits == len(repo.tasks())
    for dataset, fold in repo.tasks():
        assert a._ground_truth._label_val_dict[dataset][fold] is b._ground_truth._label_val_dict[dataset][fold]
