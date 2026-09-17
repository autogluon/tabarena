"""Tests for the environment staging behind the SkyPilot scheduler (`tabflow_slurm.setup.sky_env`)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("tabflow_slurm.setup", reason="tabflow_slurm is not installed")

from tabflow_slurm.setup import sky_env
from tabflow_slurm.setup.sky_env import (
    EnvSpec,
    RepoArchive,
    WorkingTree,
    archive_repo,
    group_paths_by_repo,
    parse_freeze,
    render_env_setup_script,
    stage_environment,
)


def _run(*args: str, **kwargs) -> subprocess.CompletedProcess:
    """Run a fixed developer tool (git / tar / bash) resolved on PATH."""
    return subprocess.run([shutil.which(args[0]) or args[0], *args[1:]], check=True, **kwargs)  # noqa: S603


_FREEZE = """\
# a comment
aplr @ file:///home/me/code/aplr
-e file:///home/me/code/autogluon/core
-e file:///home/me/code/autogluon/tabular
-e file:///home/me/code/tabarena/packages/tabarena
-e file:///home/me/code/tabarena/packages/tabflow_slurm
numpy==2.3.1
tabpfn-extensions @ git+https://github.com/PriorLabs/tabpfn-extensions@abc123

torch==2.13.0
"""


class TestParseFreeze:
    def test_splits_pins_editables_and_local_installs(self):
        spec = parse_freeze(_FREEZE)
        assert spec.requirements == (
            "numpy==2.3.1",
            "tabpfn-extensions @ git+https://github.com/PriorLabs/tabpfn-extensions@abc123",
            "torch==2.13.0",
        )
        assert spec.editable == (
            Path("/home/me/code/autogluon/core"),
            Path("/home/me/code/autogluon/tabular"),
            Path("/home/me/code/tabarena/packages/tabarena"),
            Path("/home/me/code/tabarena/packages/tabflow_slurm"),
        )
        assert spec.installs == (Path("/home/me/code/aplr"),)

    def test_editable_without_a_local_path_is_refused(self):
        with pytest.raises(ValueError, match="file://"):
            parse_freeze("-e git+https://github.com/org/repo@main#egg=repo")

    def test_percent_encoded_paths_are_decoded(self):
        assert parse_freeze("-e file:///home/me/my%20code/pkg").editable == (Path("/home/me/my code/pkg"),)


class TestGroupPathsByRepo:
    def test_groups_by_repo_root_in_first_seen_order(self):
        roots = {
            Path("/r/autogluon/core"): Path("/r/autogluon"),
            Path("/r/autogluon/tabular"): Path("/r/autogluon"),
            Path("/r/tabarena/packages/tabarena"): Path("/r/tabarena"),
            Path("/r/aplr"): Path("/r/aplr"),
        }
        grouped = group_paths_by_repo(tuple(roots), roots.__getitem__)
        assert list(grouped) == [Path("/r/autogluon"), Path("/r/tabarena"), Path("/r/aplr")]
        assert grouped[Path("/r/autogluon")] == [Path("/r/autogluon/core"), Path("/r/autogluon/tabular")]


def _fake_pack(sha="abc1234", dirty=False, content_hash="deadbeef"):
    def pack(root: Path, out: Path) -> WorkingTree:
        out.write_bytes(b"tar-of-" + root.name.encode())
        return WorkingTree(sha=sha, dirty=dirty, content_hash=content_hash)

    return pack


class TestArchiveRepo:
    def test_uploads_a_content_addressed_tarball_with_relative_install_paths(self, tmp_path, local_storage):
        root = tmp_path / "autogluon"
        (root / "core").mkdir(parents=True)
        (root / "tabular").mkdir()
        archive = archive_repo(
            root,
            storage=local_storage,
            repo_prefix="gs://bucket/me/repo",
            editable=(root / "core", root / "tabular"),
            installs=(),
            pack=_fake_pack(),
        )
        assert archive == RepoArchive(
            name="autogluon",
            sha="abc1234",
            dirty=False,
            uri="gs://bucket/me/repo/autogluon-abc1234-deadbeef.tar.gz",
            editable=("core", "tabular"),
            installs=(),
        )
        assert local_storage.path_for(archive.uri).read_bytes() == b"tar-of-autogluon"

    def test_an_already_staged_tarball_is_not_uploaded_again(self, tmp_path, local_storage):
        root = tmp_path / "aplr"
        root.mkdir()
        for _ in range(2):
            archive_repo(
                root, storage=local_storage, repo_prefix="gs://b/repo", editable=(), installs=(root,), pack=_fake_pack()
            )
        assert local_storage.uploads == ["gs://b/repo/aplr-abc1234-deadbeef.tar.gz"]

    def test_the_repo_root_itself_installs_as_dot(self, tmp_path, local_storage):
        root = tmp_path / "aplr"
        root.mkdir()
        archive = archive_repo(
            root, storage=local_storage, repo_prefix="gs://b/repo", editable=(), installs=(root,), pack=_fake_pack()
        )
        assert archive.installs == (".",)

    def test_a_dirty_tree_is_staged_with_a_warning(self, tmp_path, local_storage):
        root = tmp_path / "tabarena"
        root.mkdir()
        with pytest.warns(UserWarning, match="uncommitted or untracked"):
            archive = archive_repo(
                root,
                storage=local_storage,
                repo_prefix="gs://b/repo",
                editable=(root,),
                installs=(),
                pack=_fake_pack(dirty=True),
            )
        assert archive.dirty is True


class TestPackWorkingTree:
    @pytest.mark.skipif(shutil.which("git") is None or shutil.which("tar") is None, reason="needs git and tar")
    def test_packs_tracked_and_untracked_files_but_not_ignored_ones(self, tmp_path):
        root = tmp_path / "repo"
        root.mkdir()
        _run("git", "-C", str(root), "init", "-q")
        _run("git", "-C", str(root), "config", "user.email", "t@t")
        _run("git", "-C", str(root), "config", "user.name", "t")
        (root / "tracked.py").write_text("print(1)\n")
        (root / ".gitignore").write_text("ignored.txt\n")
        _run("git", "-C", str(root), "add", ".")
        _run("git", "-C", str(root), "commit", "-q", "-m", "init")
        (root / "tracked.py").write_text("print(2)\n")  # modified, on disk
        (root / "untracked.py").write_text("print(3)\n")  # untracked, not ignored
        (root / "ignored.txt").write_text("nope\n")

        out = tmp_path / "repo.tar.gz"
        tree = sky_env.pack_working_tree(root, out)
        listed = _run("tar", "-tzf", str(out), stdout=subprocess.PIPE, text=True).stdout.split()
        assert sorted(listed) == [".gitignore", "tracked.py", "untracked.py"]
        assert tree.dirty is True
        assert len(tree.sha) >= 7
        # The modified content (not HEAD) is what gets shipped.
        extracted = _run("tar", "-xzOf", str(out), "tracked.py", stdout=subprocess.PIPE, text=True).stdout
        assert extracted == "print(2)\n"
        # Deterministic: packing the same tree again gives the same hash.
        assert sky_env.pack_working_tree(root, tmp_path / "again.tar.gz").content_hash == tree.content_hash


class TestStageEnvironment:
    @staticmethod
    def _stage(local_storage, **overrides):
        roots = {
            Path("/home/me/code/autogluon/core"): Path("/home/me/code/autogluon"),
            Path("/home/me/code/autogluon/tabular"): Path("/home/me/code/autogluon"),
            Path("/home/me/code/tabarena/packages/tabarena"): Path("/home/me/code/tabarena"),
            Path("/home/me/code/tabarena/packages/tabflow_slurm"): Path("/home/me/code/tabarena"),
            Path("/home/me/code/aplr"): Path("/home/me/code/aplr"),
        }
        calls: dict = {"freeze": 0}

        def freeze(_python):
            calls["freeze"] += 1
            return _FREEZE

        def archive(root, *, storage, repo_prefix, editable, installs, **_):
            return RepoArchive(
                name=root.name,
                sha="abc1234",
                dirty=False,
                uri=f"{repo_prefix}/{root.name}-abc1234-deadbeef.tar.gz",
                editable=tuple(str(p.relative_to(root)) for p in editable),
                installs=tuple(str(p.relative_to(root)) for p in installs),
            )

        kwargs = {
            "python_path": "/venv/bin/python",
            "storage": local_storage,
            "env_prefix": "gs://bucket/me/env",
            "repo_prefix": "gs://bucket/me/repo",
            "python_version": "3.12",
            "freeze": freeze,
            "repo_root_of": roots.__getitem__,
            "archive": archive,
        }
        kwargs.update(overrides)
        return stage_environment(**kwargs), calls

    def test_manifest_and_requirements_are_staged_under_the_env_hash(self, local_storage):
        sky_env._STAGED.clear()
        env, _ = self._stage(local_storage, extra_requirement_lines=("--extra-index-url https://x/cu128",))
        assert isinstance(env, EnvSpec)
        assert env.manifest_uri == f"gs://bucket/me/env/{env.env_hash}/env.json"
        assert env.requirements_uri == f"gs://bucket/me/env/{env.env_hash}/requirements.txt"
        requirements = local_storage.path_for(env.requirements_uri).read_text().splitlines()
        assert requirements == [
            "numpy==2.3.1",
            "tabpfn-extensions @ git+https://github.com/PriorLabs/tabpfn-extensions@abc123",
            "torch==2.13.0",
            "--extra-index-url https://x/cu128",
        ]
        manifest = json.loads(local_storage.path_for(env.manifest_uri).read_text())
        assert manifest["python"] == "3.12"
        assert manifest["requirements"] == env.requirements_uri
        assert [(r["name"], r["editable"], r["install"]) for r in manifest["repos"]] == [
            ("autogluon", ["core", "tabular"], []),
            ("tabarena", ["packages/tabarena", "packages/tabflow_slurm"], []),
            ("aplr", [], ["."]),
        ]
        assert manifest["repos"][0]["archive"] == "gs://bucket/me/repo/autogluon-abc1234-deadbeef.tar.gz"

    def test_staging_is_memoized_per_process(self, local_storage):
        sky_env._STAGED.clear()
        env_a, calls = self._stage(local_storage)
        env_b, _ = self._stage(local_storage)
        assert env_a is env_b
        assert calls["freeze"] == 1  # the second call reused the first result (a new `calls` dict saw no freeze)

    def test_different_extra_lines_give_a_different_environment(self, local_storage):
        sky_env._STAGED.clear()
        env_a, _ = self._stage(local_storage)
        env_b, _ = self._stage(local_storage, extra_requirement_lines=("--extra-index-url https://x/cu128",))
        assert env_a.env_hash != env_b.env_hash


class TestRenderEnvSetupScript:
    def test_script_builds_the_venv_from_the_manifest_without_jq(self):
        script = render_env_setup_script()
        for needle in (
            'gcloud storage cp "$ENV_SPEC"',
            "uv venv --python",
            '--no-deps -r "$ROOT/requirements.txt"',
            "--no-deps -e",
            "env.done",
            'mkdir -p "$ROOT" "$CACHE_ROOT"',
        ):
            assert needle in script
        assert "jq" not in script

    @pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
    def test_script_is_valid_bash(self, tmp_path):
        path = tmp_path / "setup.sh"
        path.write_text(render_env_setup_script())
        _run("bash", "-n", str(path))
