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


def test_memmap_pickle_ships_task_table_not_per_task_dicts(tmp_path):
    import pickle

    import numpy as np

    from tabarena.predictions.tabular_predictions import _TaskTable

    repo = load_repo_artificial()
    repo.to_dir(tmp_path)
    preds = EvaluationRepository.from_dir(tmp_path, verbose=False)._tabular_predictions
    dataset, fold = repo.tasks()[0]
    all_models = repo.configs()

    # the legacy view is still available and complete
    task_metadata = preds.metadata_dict[dataset][fold]
    assert task_metadata["models_all"] == all_models
    assert task_metadata["model_indices"] == {m: i for i, m in enumerate(all_models)}
    assert task_metadata["models"] == all_models

    # the pickled state holds the table (arrays + the distinct model lists), no per-task dict
    state = preds.__getstate__()
    assert "metadata_dict" not in state
    assert isinstance(state["_table"], _TaskTable)
    table_state = state["_table"].__getstate__()
    assert "_index" not in table_state
    assert len(table_state["model_lists"]) == 1  # one distinct model list shared by every task
    assert table_state["available"].dtype == bool

    before = preds.predict_val(dataset, fold, all_models)
    restored = pickle.loads(pickle.dumps(preds))
    assert restored.metadata_dict == preds.metadata_dict
    assert (restored.predict_val(dataset, fold, all_models) == before).all()

    # restriction keeps the original row indices through a pickle round trip
    preds.restrict_models([all_models[-1]])
    restricted = pickle.loads(pickle.dumps(preds))
    assert restricted.metadata_dict[dataset][fold]["models"] == [all_models[-1]]
    assert (restricted.predict_val(dataset, fold, [all_models[-1]]) == before[-1:]).all()
    assert restricted.models == [all_models[-1]]
    assert np.array_equal(restricted._table.available.sum(axis=1), np.ones(len(restricted._table), dtype=int))


def test_memmap_restricts_match_in_memory(tmp_path):
    """Dataset, fold and model restrictions on the memmap store give the same availability,
    ordering and predictions as the in-memory store.
    """
    from tabarena.predictions.tabular_predictions import TabularPredictionsInMemory, TabularPredictionsMemmap

    repo = load_repo_artificial()
    pred_dict = repo._tabular_predictions.to_dict()
    memmap = TabularPredictionsMemmap.from_dict(pred_dict, output_dir=str(tmp_path / "mm"))
    memory = TabularPredictionsInMemory.from_dict(pred_dict)
    models = repo.configs()
    for store in (memmap, memory):
        store.restrict_folds([2, 0])
        store.restrict_models(models[::-1][:1])
        store.restrict_datasets(["abalone"])
    assert memmap.model_available_dict() == memory.model_available_dict()
    assert memmap.datasets == memory.datasets
    assert sorted(memmap.folds) == sorted(memory.folds)
    assert sorted(memmap.dataset_fold_lst()) == sorted(memory.dataset_fold_lst())  # memmap rows follow rglob order
    assert memmap.models == memory.models
    for dataset, fold in memmap.dataset_fold_lst():
        for split in ("predict_val", "predict_test"):
            a = getattr(memmap, split)(dataset, fold, memmap.models)
            b = getattr(memory, split)(dataset, fold, memory.models)
            assert (a == b).all()
    # a dataset restriction to nothing leaves an empty store rather than failing
    memmap.restrict_datasets(["not_there"])
    assert memmap.dataset_fold_lst() == []


def test_memmap_unpickles_legacy_per_task_dict_state(tmp_path):
    """Objects pickled with one metadata dict per task load into the task table."""
    import pickle

    from tabarena.predictions.tabular_predictions import TabularPredictionsMemmap

    repo = load_repo_artificial()
    repo.to_dir(tmp_path)
    preds = EvaluationRepository.from_dir(tmp_path, verbose=False)._tabular_predictions
    all_models = repo.configs()
    legacy_dict = preds.metadata_dict
    for fold_dict in legacy_dict.values():
        for task in fold_dict.values():
            task.pop("model_indices")
            task["models"] = all_models[:1]  # a restriction recorded the old way
    legacy = TabularPredictionsMemmap.__new__(TabularPredictionsMemmap)
    legacy.__setstate__({"data_dir": preds.data_dir, "metadata_dict": legacy_dict})
    dataset, fold = repo.tasks()[0]
    assert legacy.metadata_dict[dataset][fold]["models"] == all_models[:1]
    assert legacy.metadata_dict[dataset][fold]["models_all"] == all_models
    assert (legacy.predict_val(dataset, fold, all_models[:1]) == preds.predict_val(dataset, fold, all_models[:1])).all()
    assert pickle.loads(pickle.dumps(legacy)).models == all_models[:1]
