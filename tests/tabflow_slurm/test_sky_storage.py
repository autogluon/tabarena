"""Tests for the ``gcloud storage`` wrapper (`tabflow_slurm.setup.sky_storage`), with the CLI stubbed."""

from __future__ import annotations

import subprocess

import pytest

pytest.importorskip("tabflow_slurm.setup", reason="tabflow_slurm is not installed")

from tabflow_slurm.setup.sky_storage import GcsStorage


class _Recorder:
    """Stands in for ``subprocess.run``: records argv and answers from a queue of (returncode, stdout)."""

    def __init__(self, answers=None):
        self.calls: list[list[str]] = []
        self.answers = list(answers or [])

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        rc, out = self.answers.pop(0) if self.answers else (0, "")
        if kwargs.get("check") and rc != 0:
            raise subprocess.CalledProcessError(rc, argv)
        return subprocess.CompletedProcess(argv, rc, stdout=out, stderr="")


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(subprocess, "run", rec)
    return rec


def test_exists_uses_ls_exit_code(recorder):
    recorder.answers = [(0, "gs://b/o\n"), (1, "")]
    storage = GcsStorage()
    assert storage.exists("gs://b/o") is True
    assert storage.exists("gs://b/missing") is False
    assert recorder.calls[0] == ["gcloud", "storage", "ls", "gs://b/o"]


def test_exists_prefix_lists_the_slash_terminated_prefix(recorder):
    recorder.answers = [(1, "")]
    assert GcsStorage().exists_prefix("gs://b/runs/x/output/data") is False
    assert recorder.calls[0][-1] == "gs://b/runs/x/output/data/"


def test_list_names_strips_the_prefix_and_trailing_slashes(recorder):
    recorder.answers = [(0, "gs://b/q/tasks/000000.json\ngs://b/q/tasks/000001.json\ngs://b/q/tasks/sub/\n")]
    assert GcsStorage().list_names("gs://b/q/tasks") == ["000000.json", "000001.json", "sub"]


def test_list_files_is_recursive_and_relative(recorder):
    recorder.answers = [
        (0, "gs://b/c/snapshots/rev1/model.ckpt\ngs://b/c/snapshots/rev1/config.json\ngs://b/c/snapshots/rev1/:\n")
    ]
    assert GcsStorage().list_files("gs://b/c/snapshots") == ["rev1/model.ckpt", "rev1/config.json"]
    assert recorder.calls[0][-1] == "gs://b/c/snapshots/**"


def test_list_names_is_empty_for_a_missing_prefix(recorder):
    recorder.answers = [(1, "")]
    assert GcsStorage().list_names("gs://b/q/claims") == []


def test_create_exclusive_uses_a_generation_precondition(recorder):
    recorder.answers = [(0, ""), (1, "")]
    storage = GcsStorage(gcloud="/opt/gcloud")
    assert storage.create_exclusive("gs://b/q/claims/3", "job-1") is True
    assert storage.create_exclusive("gs://b/q/claims/3", "job-2") is False
    argv = recorder.calls[0]
    assert argv[:4] == ["/opt/gcloud", "storage", "cp", "--if-generation-match=0"]
    assert argv[-1] == "gs://b/q/claims/3"


def test_copy_tree_copies_each_directory_with_explicit_object_names(recorder, tmp_path):
    (tmp_path / "data" / "m" / "k" / "0_1").mkdir(parents=True)
    (tmp_path / "data" / "m" / "k" / "0_1" / "results.pkl").write_bytes(b"x")
    (tmp_path / "data" / "m" / "k" / "0_1" / "validation_protocol.json").write_text("{}")
    (tmp_path / "data" / "m" / "k" / "0_2").mkdir()
    (tmp_path / "data" / "m" / "k" / "0_2" / "results.pkl").write_bytes(b"y")
    GcsStorage().copy_tree(tmp_path / "data", "gs://b/runs/x/output/data")
    targets = [call[-1] for call in recorder.calls]
    assert targets == ["gs://b/runs/x/output/data/m/k/0_1/", "gs://b/runs/x/output/data/m/k/0_2/"]
    assert recorder.calls[0][2] == "cp"
    assert recorder.calls[0][3:5] == [
        str(tmp_path / "data" / "m" / "k" / "0_1" / "results.pkl"),
        str(tmp_path / "data" / "m" / "k" / "0_1" / "validation_protocol.json"),
    ]


def test_upload_dir_rsyncs_the_content_behind_symlinks(recorder, tmp_path):
    GcsStorage().upload_dir(tmp_path / "snapshots", "gs://b/c/snapshots")
    assert recorder.calls[0][2:] == [
        "rsync",
        "-r",
        "--no-ignore-symlinks",
        str(tmp_path / "snapshots"),
        "gs://b/c/snapshots",
    ]


def test_download_dir_creates_the_target_and_rsyncs(recorder, tmp_path):
    GcsStorage().download_dir("gs://b/runs/x/output/data", tmp_path / "out" / "data")
    assert (tmp_path / "out" / "data").is_dir()
    assert recorder.calls[0][2:] == ["rsync", "-r", "gs://b/runs/x/output/data", str(tmp_path / "out" / "data")]
