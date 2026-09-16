from __future__ import annotations

import json

import pytest

from tabarena.repository import EvaluationRepository
from tabarena.simulation.benchmark_context import BenchmarkContext
from tabarena.simulation.context_artificial import load_repo_artificial


def test_from_json_drops_legacy_download_map(tmp_path):
    repo = load_repo_artificial()
    repo.to_dir(tmp_path)
    path = tmp_path / "context.json"
    with open(path) as f:
        payload = json.load(f)
    assert "s3_download_map" not in payload
    payload["s3_download_map"] = None  # what contexts written by earlier versions contain
    with open(path, "w") as f:
        json.dump(payload, f)
    context = BenchmarkContext.from_json(str(path))
    assert not hasattr(context, "s3_download_map")
    assert sorted(EvaluationRepository.from_dir(tmp_path, verbose=False).tasks()) == sorted(repo.tasks())


def test_validate_lists_missing_files(tmp_path):
    repo = load_repo_artificial()
    repo.to_dir(tmp_path)
    dataset, fold = repo.tasks()[0]
    removed = tmp_path / "model_predictions" / dataset / str(fold) / "pred-val.dat"
    removed.unlink()
    with pytest.raises(FileNotFoundError, match="Missing 1 required files") as excinfo:
        EvaluationRepository.from_dir(tmp_path, verbose=False, validate=True)
    assert str(removed) in str(excinfo.value)


def test_default_load_fails_lazily_on_missing_prediction_file(tmp_path):
    """Without validation the load succeeds and the missing memmap fails when first read."""
    repo = load_repo_artificial()
    repo.to_dir(tmp_path)
    dataset, fold = repo.tasks()[0]
    (tmp_path / "model_predictions" / dataset / str(fold) / "pred-val.dat").unlink()
    loaded = EvaluationRepository.from_dir(tmp_path, verbose=False)
    other_dataset = next(d for d in repo.datasets() if d != dataset)
    loaded.predict_val(dataset=other_dataset, fold=fold, config=repo.configs()[0])
    with pytest.raises(FileNotFoundError):
        loaded.predict_val(dataset=dataset, fold=fold, config=repo.configs()[0])


def test_default_load_fails_early_on_missing_task_file(tmp_path):
    repo = load_repo_artificial()
    repo.to_dir(tmp_path)
    dataset, _ = repo.tasks()[0]
    (tmp_path / "model_predictions" / dataset / "tasks.dat").unlink()
    with pytest.raises(FileNotFoundError):
        EvaluationRepository.from_dir(tmp_path, verbose=False)
