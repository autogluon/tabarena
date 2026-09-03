"""Tests for ``scripts/release/build_pypi_dists.py``: the pyproject edits and the wheel checks.

Nothing here runs ``uv build``; the full build is exercised by the ``install-from-wheel`` CI job.
"""

from __future__ import annotations

import importlib.util
import tomllib
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "release" / "build_pypi_dists.py"


@pytest.fixture(scope="module")
def build():
    spec = importlib.util.spec_from_file_location("build_pypi_dists", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOML = """[project]
name = "tabarena"
version = "0.0.1"  # keep equal to bencheval
dependencies = [
  "bencheval",
  "numpy",
]

[project.optional-dependencies]
plot = ["bencheval[plot]", "tueplots"]
tabpfn = [
  "tabpfn>=8.0.8",
  "tabpfn-extensions[many_class] @ git+https://github.com/PriorLabs/tabpfn-extensions.git",
]
tabfm = ["tabfm[pytorch] @ git+https://github.com/google-research/tabfm.git@53f3fcf"]  # pinned commit
mixed = ["einops", "sap_rpt_oss @ git+https://github.com/SAP-samples/sap-rpt-1-oss.git@a323a0a"]
tabswift = []
"""


def test_set_version_replaces_only_the_project_version(build):
    out = build.set_version(TOML, "0.1.0a1")
    assert tomllib.loads(out)["project"]["version"] == "0.1.0a1"
    assert "0.0.1" not in out
    assert 'version = "0.1.0a1"  # keep equal to bencheval' in out


def test_pin_bencheval_covers_plain_and_extra_requirements(build):
    project = tomllib.loads(build.pin_bencheval(TOML, "0.1.0a1"))["project"]
    assert "bencheval==0.1.0a1" in project["dependencies"]
    assert "bencheval[plot]==0.1.0a1" in project["optional-dependencies"]["plot"]


def test_strip_direct_url_requirements_empties_extras_but_keeps_them(build):
    text, removed = build.strip_direct_url_requirements(TOML)
    extras = tomllib.loads(text)["project"]["optional-dependencies"]
    assert removed == [
        "tabpfn-extensions[many_class] @ git+https://github.com/PriorLabs/tabpfn-extensions.git",
        "tabfm[pytorch] @ git+https://github.com/google-research/tabfm.git@53f3fcf",
        "sap_rpt_oss @ git+https://github.com/SAP-samples/sap-rpt-1-oss.git@a323a0a",
    ]
    assert extras["tabpfn"] == ["tabpfn>=8.0.8"]
    assert extras["tabfm"] == []
    assert extras["mixed"] == ["einops"]
    assert extras["tabswift"] == []
    assert "# pinned commit" in text


def test_strip_direct_url_requirements_rejects_unhandled_layout(build):
    literal = '[project]\nname = "x"\nversion = "0.0.1"\ndependencies = [\'a @ git+https://e.com/a.git\']\n'
    with pytest.raises(SystemExit, match="survived"):
        build.strip_direct_url_requirements(literal)


def test_compute_dev_version(build):
    stamp = datetime(2026, 9, 3, 10, 2, 33, tzinfo=UTC)
    assert build.compute_dev_version("0.1.0", now=stamp) == "0.1.1.dev20260903100233"
    with pytest.raises(SystemExit, match="plain X.Y.Z"):
        build.compute_dev_version("0.1.0a1")


def test_rewrite_readme_for_pypi(build):
    readme = (
        "See [examples](examples/plots), [docs](./tabrepo.md), [site](https://tabarena.ai/), [top](#-installation).\n"
        "> [!TIP]\n> Tip body\n"
    )
    out = build.rewrite_readme_for_pypi(readme, base_url="https://gh/blob/main/")
    assert "[examples](https://gh/blob/main/examples/plots)" in out
    assert "[docs](https://gh/blob/main/tabrepo.md)" in out
    assert "[site](https://tabarena.ai/)" in out
    assert "[top](#-installation)" in out
    assert "> **Tip**\n> Tip body" in out


def test_repo_pyproject_files_are_publishable_after_preparation(build):
    """Run the real pyproject files through the PyPI edits; this is what the release workflows publish."""
    for package in build.PACKAGES:
        text = (REPO_ROOT / "packages" / package / "pyproject.toml").read_text()
        prepared, _ = build.prepare_pyproject_text(package, text, "0.1.0a1")
        project = tomllib.loads(prepared)["project"]
        assert project["version"] == "0.1.0a1"
        all_reqs = [*project["dependencies"], *(r for reqs in project["optional-dependencies"].values() for r in reqs)]
        assert not [r for r in all_reqs if " @ " in r]
        if package == "tabarena":
            bencheval_reqs = [r for r in all_reqs if r.startswith("bencheval")]
            assert bencheval_reqs
            assert all("==0.1.0a1" in r for r in bencheval_reqs)


def _fake_wheel(path: Path, metadata: str, extra_files: tuple[str, ...] = ()) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("pkg-1.0.dist-info/METADATA", metadata)
        for name in extra_files:
            zf.writestr(name, "")
    return path


def test_check_wheel_accepts_a_good_wheel_and_reports_every_problem(build, tmp_path):
    good_metadata = (
        "Metadata-Version: 2.4\nName: tabarena\nVersion: 0.1.0a1\nLicense-File: LICENSE\n"
        'Requires-Dist: bencheval==0.1.0a1\nRequires-Dist: bencheval[plot]==0.1.0a1; extra == "plot"\n'
        "\nLong description\n"
    )
    good = _fake_wheel(tmp_path / "tabarena-0.1.0a1-py3-none-any.whl", good_metadata, ("tabarena/data/x.csv",))
    build.check_wheel(good, "tabarena", "0.1.0a1", ["tabarena/data/x.csv"])

    bad_metadata = (
        "Metadata-Version: 2.4\nName: tabarena\nVersion: 0.1.0a1\n"
        'Requires-Dist: bencheval\nRequires-Dist: tabfm @ git+https://x/y.git ; extra == "tabfm"\n\n'
    )
    bad = _fake_wheel(tmp_path / "bad.whl", bad_metadata)
    with pytest.raises(SystemExit) as info:
        build.check_wheel(bad, "tabarena", "0.1.0a1", ["tabarena/data/x.csv"])
    message = str(info.value)
    for expected in ("direct-URL", "empty long description", "no License-File", "bencheval not pinned", "missing from"):
        assert expected in message
