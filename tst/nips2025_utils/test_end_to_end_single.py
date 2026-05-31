"""Tests for task-directory filtering in end-to-end result processing.

Regression coverage for the UserTask-slug cache-path refactor: task directories
may be OpenML integer ids *or* readable slugs (``tabarena_task_name``), so the
filter must not assume the directory name parses as an ``int``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tabarena.nips2025_utils.end_to_end_single import (
    _filter_file_paths_by_task_metadata,
    _task_dir,
)


def _paths(task_key: str) -> list[Path]:
    return [Path(f"/cache/{task_key}/results.pkl")]


def test_task_dir_extracts_directory_component():
    assert _task_dir("3945/0") == "3945"
    assert _task_dir("emscad-1790bb44ad91/0") == "emscad-1790bb44ad91"


def test_filter_keeps_integer_and_slug_tasks():
    task_metadata = pd.DataFrame(
        {
            "tid": [3945, 1234567890],
            "tabarena_task_name": ["Task-3945", "emscad-1790bb44ad91"],
        }
    )
    all_file_paths = {
        "3945/0": _paths("3945/0"),  # OpenML integer task
        "emscad-1790bb44ad91/0": _paths("emscad-1790bb44ad91/0"),  # UserTask slug
    }

    filtered = _filter_file_paths_by_task_metadata(all_file_paths, task_metadata)

    assert set(filtered) == {"3945/0", "emscad-1790bb44ad91/0"}


def test_filter_drops_unknown_tasks():
    task_metadata = pd.DataFrame(
        {"tid": [3945], "tabarena_task_name": ["emscad-1790bb44ad91"]}
    )
    all_file_paths = {
        "3945/0": _paths("3945/0"),
        "emscad-1790bb44ad91/0": _paths("emscad-1790bb44ad91/0"),
        "9999/0": _paths("9999/0"),  # not present in task_metadata
        "unknown-slug-abc/0": _paths("unknown-slug-abc/0"),  # not present either
    }

    filtered = _filter_file_paths_by_task_metadata(all_file_paths, task_metadata)

    assert set(filtered) == {"3945/0", "emscad-1790bb44ad91/0"}


def test_filter_without_tabarena_task_name_column():
    # Legacy task_metadata lacking the slug column still works for integer tasks.
    task_metadata = pd.DataFrame({"tid": [3945]})
    all_file_paths = {
        "3945/0": _paths("3945/0"),
        "emscad-1790bb44ad91/0": _paths("emscad-1790bb44ad91/0"),
    }

    filtered = _filter_file_paths_by_task_metadata(all_file_paths, task_metadata)

    assert set(filtered) == {"3945/0"}
