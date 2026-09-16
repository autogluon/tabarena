from __future__ import annotations

import json
from pathlib import Path

from tabarena.predictions.tabular_predictions import TabularPredictionsMemmap
from tabarena.repository import EvaluationRepository
from tabarena.simulation.context_artificial import load_repo_artificial


def _metadata_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.rglob("*metadata.json"))


def test_memmap_from_metadata_files_matches_walk(tmp_path):
    repo = load_repo_artificial()
    repo.to_dir_task_data(tmp_path)  # per-task layout (metadata.json per task), before consolidation
    data_dir = tmp_path / "model_predictions"
    metadata_files = _metadata_files(data_dir)
    assert len(metadata_files) == len(repo.tasks())

    walked = TabularPredictionsMemmap.from_data_dir(data_dir)
    listed = TabularPredictionsMemmap.from_data_dir(data_dir, metadata_files=metadata_files)
    assert listed.metadata_dict == walked.metadata_dict
    assert listed.datasets == walked.datasets
    dataset, fold = repo.tasks()[0]
    models = repo.configs()
    assert (listed.predict_val(dataset, fold, models) == walked.predict_val(dataset, fold, models)).all()

    # a task directory on disk that the list does not mention is not loaded
    extra = data_dir / "extra_dataset" / "0"
    extra.mkdir(parents=True)
    with open(extra / "metadata.json", "w") as f:
        json.dump(
            {
                "models": ["m"],
                "dataset": "extra_dataset",
                "fold": 0,
                "pred_val_shape": [1, 2],
                "pred_test_shape": [1, 2],
                "dtype": "float32",
            },
            f,
        )
    assert "extra_dataset" in TabularPredictionsMemmap.from_data_dir(data_dir).metadata_dict
    assert (
        "extra_dataset"
        not in TabularPredictionsMemmap.from_data_dir(data_dir, metadata_files=metadata_files).metadata_dict
    )


def test_from_dir_uses_context_metadata_list(tmp_path, monkeypatch):
    """Loading a processed dir must not walk it: the context lists every task's metadata.json."""
    repo = load_repo_artificial()
    repo.to_dir(tmp_path)

    def no_walk(self, pattern):
        raise AssertionError(f"rglob({pattern!r}) called during from_dir")

    monkeypatch.setattr(Path, "rglob", no_walk)
    loaded = EvaluationRepository.from_dir(tmp_path, verbose=False)
    assert sorted(loaded.tasks()) == sorted(repo.tasks())
    assert sorted(loaded.configs()) == sorted(repo.configs())


def test_memmap_pickle_state_derives_model_indices(tmp_path):
    import pickle

    repo = load_repo_artificial()
    repo.to_dir(tmp_path)
    preds = EvaluationRepository.from_dir(tmp_path, verbose=False)._tabular_predictions
    dataset, fold = repo.tasks()[0]
    task_metadata = preds.metadata_dict[dataset][fold]
    assert task_metadata["models_all"] == repo.configs() and "model_indices" in task_metadata
    state = preds.__getstate__()
    assert "model_indices" not in state["metadata_dict"][dataset][fold]
    # identical model lists are shared across tasks, so they pickle once
    other_dataset, other_fold = repo.tasks()[-1]
    assert preds.metadata_dict[other_dataset][other_fold]["models_all"] is task_metadata["models_all"]

    all_models = repo.configs()
    before = preds.predict_val(dataset, fold, all_models)
    restored = pickle.loads(pickle.dumps(preds))
    assert restored.metadata_dict[dataset][fold]["model_indices"] == {m: i for i, m in enumerate(all_models)}
    assert (restored.predict_val(dataset, fold, all_models) == before).all()

    # restriction keeps the original row indices through a pickle round trip
    preds.restrict_models([all_models[-1]])
    restricted = pickle.loads(pickle.dumps(preds))
    assert restricted.metadata_dict[dataset][fold]["models"] == [all_models[-1]]
    assert (restricted.predict_val(dataset, fold, [all_models[-1]]) == before[-1:]).all()
