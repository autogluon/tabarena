"""``generate_all_figs``'s per-dataset browser, the leaderboard website's "Dataset by dataset" view."""

from __future__ import annotations

import pytest

from tabarena.contexts.abstract_arena_context import AbstractArenaContext


def _subset_folder(path, with_trajectories: bool):
    """A compare pass's per-split results and method info, with one method compared as `new_results` (no
    metadata), and optionally the trajectory pass's per-dataset frame.
    """
    import pandas as pd

    path.mkdir()
    rows = [
        {
            "dataset": dataset,
            "fold": fold,
            "method": method,
            "metric_error": error + 0.01 * fold,
            "time_train_s": 10.0,
            "time_infer_s": 0.5,
            "metric": "log_loss",
            "problem_type": "binary",
            "imputed": False,
        }
        for dataset, errors in (("alpha", (0.3, 0.1)), ("beta", (0.2, 0.4)))
        for method, error in zip(("CAT (default)", "My portfolio"), errors, strict=False)
        for fold in (0, 1)
    ]
    pd.DataFrame(rows).to_csv(path / "results_per_split.csv", index=False)
    pd.DataFrame(
        [
            {
                "method": "CAT (default)",
                "method_type": "config",
                "method_subtype": "default",
                "config_type": "CAT",
                "display_name": "CatBoost",
                "method_class": "model",
            },
            {
                "method": "My portfolio",
                "method_type": "baseline",
                "method_subtype": None,
                "config_type": None,
                "display_name": None,
                "method_class": None,
            },
        ]
    ).to_csv(path / "method_info.csv", index=False)
    if with_trajectories:
        (path / "tuning_trajectories" / "some_subset").mkdir(parents=True)
        pd.DataFrame(
            {
                "dataset": ["alpha"],
                "method": ["CatBoost"],
                "n_configs": [1],
                "train_s": [10.0],
                "err": [0.3],
                "imp": [5.0],
            }
        ).to_csv(path / "tuning_trajectories" / "some_subset" / "tuning_trajectories_per_dataset.csv", index=False)


def _arena(buckets=None):
    from types import SimpleNamespace

    import pandas as pd

    frame = pd.DataFrame(
        {"dataset": ["alpha", "beta"], "dataset_name": ["Alpha", "Beta"], "max_train_rows": [500, 5000]}
    )
    return SimpleNamespace(
        benchmark_name="Arena",
        task_metadata_collection=SimpleNamespace(per_dataset_frame=lambda: frame),
        _per_dataset_size_buckets=buckets,
    )


@pytest.mark.parametrize("with_trajectories", [False, True])
def test_the_per_dataset_explorer_reads_the_subset_folder(tmp_path, with_trajectories: bool):
    """The browser holds every method of the per-split results, a `new_results` one under its own name, and the
    trajectory pass's per-dataset frame when there is one.
    """
    import json
    import re

    from tabarena.plot.interactive.per_dataset_explorer import BEYONDARENA_SIZE_BUCKETS

    _subset_folder(tmp_path / "all", with_trajectories)
    page = AbstractArenaContext._save_per_dataset_explorer(
        _arena(BEYONDARENA_SIZE_BUCKETS), tmp_path / "all", leaderboard=None, subset_label="all"
    )

    assert page == tmp_path / "all" / "per_dataset_explorer.html"
    html = page.read_text(encoding="utf-8")
    config = json.loads(re.search(r"const CONFIG = (\{.*?\});\n", html).group(1))
    assert sorted(m["name"] for m in config["methods"]) == ["CatBoost (default)", "My portfolio"]
    assert [d["name"] for d in config["datasets"]] == ["Alpha", "Beta"]
    assert [b["key"] for b in config["sizeBuckets"]] == ["tiny", "small", "medium", "large"]
    assert config["trajectoryMethods"] == (["CatBoost"] if with_trajectories else [])
    assert "<title>Arena per-dataset results — all</title>" in html


def test_no_per_dataset_explorer_without_the_compare_pass_files(tmp_path):
    (tmp_path / "all").mkdir()
    assert (
        AbstractArenaContext._save_per_dataset_explorer(
            _arena(), tmp_path / "all", leaderboard=None, subset_label="all"
        )
        is None
    )


def test_the_per_dataset_explorer_follows_the_compare_pass(monkeypatch):
    """On by default with the compare pass, which it reads, and it asks the trajectory pass for its per-dataset
    frame; requested without the compare pass it is an error.
    """
    captured = []

    def fake_subset_figs(self, subset, **kwargs):
        captured.append(kwargs)
        return ("", None)

    monkeypatch.setattr(AbstractArenaContext, "_generate_subset_figs", fake_subset_figs)
    monkeypatch.setattr(AbstractArenaContext, "_default_subsets", property(lambda self: [[]]))
    context = AbstractArenaContext.__new__(AbstractArenaContext)
    monkeypatch.setattr(AbstractArenaContext, "_save_composite_pareto_explorer", lambda *args: None)

    AbstractArenaContext.generate_all_figs(context, output_dir="unused", plot_tuning_trajectories=True, max_workers=1)
    assert captured[-1]["save_per_dataset_explorer"] is True
    assert captured[-1]["tuning_trajectory_kwargs"]["per_dataset_trajectories"] is True

    AbstractArenaContext.generate_all_figs(context, output_dir="unused", plot_compare=False, max_workers=1)
    assert captured[-1]["save_per_dataset_explorer"] is False
    with pytest.raises(ValueError, match="requires plot_compare=True"):
        AbstractArenaContext.generate_all_figs(
            context, output_dir="unused", plot_compare=False, save_per_dataset_explorer=True
        )
