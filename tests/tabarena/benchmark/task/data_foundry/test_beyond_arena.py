from __future__ import annotations

import types

import pytest

from tabarena.benchmark.task.data_foundry.beyond_arena import materialize_task


@pytest.fixture
def local_pickle(tmp_path, monkeypatch):
    """A task whose OpenML pickle already exists, with the text-cache import recorded instead of run."""
    from tabarena.benchmark.task import user_task
    from tabarena.benchmark.task.data_foundry import text_cache

    pickle_path = tmp_path / "ds.pkl"
    pickle_path.write_bytes(b"x")
    stub = types.SimpleNamespace(openml_task_path=pickle_path, slug="ds", task_id_str="UserTask|1|ds")
    monkeypatch.setattr(
        user_task, "UserTask", types.SimpleNamespace(from_task_id_str=staticmethod(lambda task_id_str: stub))
    )
    calls: list[str] = []
    monkeypatch.setattr(text_cache, "ensure_text_cache_for_task", lambda **kw: calls.append(kw["task_key"]))
    return calls


@pytest.mark.parametrize("has_text", [None, True])
def test_local_pickle_imports_the_text_cache_unless_told_the_dataset_has_no_text(local_pickle, has_text):
    task_id = materialize_task(
        collection=object(), task_id_str="UserTask|1|ds", data_foundry_uri="ds/uuid", has_text=has_text
    )
    assert task_id == "UserTask|1|ds"
    assert local_pickle == ["ds"]


def test_local_pickle_of_a_text_free_dataset_does_not_touch_the_container(local_pickle):
    task_id = materialize_task(
        collection=object(), task_id_str="UserTask|1|ds", data_foundry_uri="ds/uuid", has_text=False
    )
    assert task_id == "UserTask|1|ds"
    assert local_pickle == []
