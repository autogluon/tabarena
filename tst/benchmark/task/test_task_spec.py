"""TaskSpec identity contract: cache keys and ``task_id_str`` round-trips.

Cache-key stability is a hard requirement (see ``TaskSpec.cache_key``): the values
pinned here are the per-task component of existing results/text cache paths. If one
of these assertions fails, the change silently invalidates every existing cache.
"""

from __future__ import annotations

import pytest

from tabarena.benchmark.task import (
    TaskSpec,
    UserTask,
    task_spec_from_task_id_str,
)
from tabarena.benchmark.task.openml import OpenMLTaskSpec


def test_openml_task_spec_identity():
    spec = OpenMLTaskSpec(363625)
    assert spec.task_id == 363625
    assert spec.task_id_str == "363625"
    # The bare integer id: the historical results-cache key for OpenML tasks.
    assert spec.cache_key == 363625
    assert isinstance(spec.cache_key, int)


def test_user_task_spec_identity():
    task = UserTask(task_name="toy_classification")
    # The slug: the historical results-cache key for local tasks.
    assert task.cache_key == task.slug
    assert task.cache_key == "toy_classification-b3d50562ac6f"
    assert task.task_id_str == f"UserTask|{task.task_id}|toy_classification"
    assert task.resolve_task_name(None) == task.tabarena_task_name


def test_task_spec_registry_round_trip_openml():
    spec = task_spec_from_task_id_str("363625")
    assert isinstance(spec, OpenMLTaskSpec)
    assert spec == OpenMLTaskSpec(363625)
    assert spec.cache_key == 363625
    # int input is accepted too (legacy collections store ints)
    assert task_spec_from_task_id_str(363625) == spec


def test_task_spec_registry_round_trip_user_task():
    original = UserTask(task_name="blood_transfusion/abc-123")
    spec = task_spec_from_task_id_str(original.task_id_str)
    assert isinstance(spec, UserTask)
    assert spec.task_name == original.task_name
    assert spec.task_id_str == original.task_id_str
    assert spec.cache_key == original.slug


def test_task_spec_registry_round_trip_user_task_with_explicit_path(tmp_path):
    original = UserTask(task_name="custom", task_cache_path=tmp_path)
    spec = task_spec_from_task_id_str(original.task_id_str)
    assert isinstance(spec, UserTask)
    assert spec.task_cache_path == tmp_path
    assert spec.task_id_str == original.task_id_str


def test_task_spec_unknown_prefix_raises():
    with pytest.raises(ValueError, match="no parser"):
        task_spec_from_task_id_str("NotATask|1|whatever")


def test_user_task_is_task_spec():
    assert issubclass(UserTask, TaskSpec)
    assert issubclass(OpenMLTaskSpec, TaskSpec)


def test_cache_key_from_task_id_str_matches_spec_cache_key():
    """Writer (engine, via spec.cache_key) and pre-check (via task_id_str) must agree."""
    from tabarena.benchmark.experiment.experiment_runner_api import task_cache_key_from_task_id_str

    assert task_cache_key_from_task_id_str("363625") == 363625
    user = UserTask(task_name="toy_regression")
    assert task_cache_key_from_task_id_str(user.task_id_str) == user.cache_key
