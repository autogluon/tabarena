from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from tabarena.plot.interactive.per_dataset_explorer import (
    build_per_dataset_explorer_html,
    dataset_records,
    imputed_counts,
    method_records,
    per_dataset_points,
    tied_with_best,
)

_METHOD_INFO = pd.DataFrame(
    [
        {
            "method": "CAT (default)",
            "method_type": "config",
            "method_subtype": "default",
            "config_type": "CAT",
            "display_name": "CatBoost",
            "method_class": "model",
            "reference_url": "https://catboost.ai",
        },
        {
            "method": "CAT (tuned)",
            "method_type": "hpo",
            "method_subtype": "tuned",
            "config_type": "CAT",
            "display_name": "CatBoost",
            "method_class": "model",
            "reference_url": "https://catboost.ai",
        },
        {
            "method": "AutoGluon 1.6 (extreme, 4h)",
            "method_type": "baseline",
            "method_subtype": None,
            "config_type": None,
            "display_name": "AutoGluon 1.6 (extreme, 4h)",
            "method_class": "system",
            "reference_url": None,
        },
    ],
)


def _results_per_split() -> pd.DataFrame:
    rows = []
    # Two datasets, two splits each. CatBoost (tuned) wins on `alpha`, the system on `beta`.
    errors = {
        ("alpha", "CAT (default)"): [0.30, 0.32],
        ("alpha", "CAT (tuned)"): [0.10, 0.12],
        ("alpha", "AutoGluon 1.6 (extreme, 4h)"): [0.20, 0.22],
        ("beta", "CAT (default)"): [1.00, 1.20],
        ("beta", "CAT (tuned)"): [0.80, 0.90],
        ("beta", "AutoGluon 1.6 (extreme, 4h)"): [0.40, 0.50],
    }
    for (dataset, method), values in errors.items():
        for fold, value in enumerate(values):
            rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "method": method,
                    "metric_error": value,
                    "time_train_s": 10.0,
                    "time_infer_s": 0.5,
                    "metric": "log_loss" if dataset == "alpha" else "rmse",
                    "problem_type": "binary" if dataset == "alpha" else "regression",
                    "imputed": False,
                },
            )
    return pd.DataFrame(rows)


def _results_with_imputation() -> pd.DataFrame:
    """The system could not run on `beta`, so its score there is imputed."""
    df = _results_per_split()
    imputed = (df["dataset"] == "beta") & (df["method"] == "AutoGluon 1.6 (extreme, 4h)")
    df.loc[imputed, "imputed"] = True
    return df


def test_imputed_pairs_are_dropped_for_that_dataset_only():
    points = per_dataset_points(_results_with_imputation()).set_index(["dataset", "method"])
    # Gone where it was imputed...
    assert ("beta", "AutoGluon 1.6 (extreme, 4h)") not in points.index
    # ...and still there where it actually ran.
    assert ("alpha", "AutoGluon 1.6 (extreme, 4h)") in points.index
    # The ranks on `beta` are now over the two methods that ran, not three.
    assert points.loc[("beta", "CAT (tuned)"), "rank"] == pytest.approx(1.0)
    assert points.loc[("beta", "CAT (default)"), "rank"] == pytest.approx(2.0)
    # `alpha` is untouched.
    assert points.loc[("alpha", "AutoGluon 1.6 (extreme, 4h)"), "rank"] == pytest.approx(2.0)


def test_imputed_counts_are_per_dataset():
    counts = imputed_counts(_results_with_imputation())
    assert counts["beta"] == 1
    assert counts["alpha"] == 0


def test_explorer_reports_the_smaller_field(tmp_path):
    out = build_per_dataset_explorer_html(
        results_per_split=_results_with_imputation(),
        method_info=_METHOD_INFO,
        trajectories=None,
        dataset_metadata=None,
        default_contender="CatBoost (tuned)",
        save_path=tmp_path / "per_dataset_explorer.html",
    )
    config = _config(out.read_text(encoding="utf-8"))
    by_key = {d["key"]: d for d in config["datasets"]}
    # The page says why one dataset's field is smaller than the other's.
    assert by_key["beta"]["skipped"] == 1
    assert "skipped" not in by_key["alpha"] or by_key["alpha"]["skipped"] == 0
    # The caller's choice of contender wins over the mean-rank fallback.
    assert config["methods"][config["defaultContender"]]["name"] == "CatBoost (tuned)"


def _trajectories() -> pd.DataFrame:
    rows = []
    for dataset in ("alpha", "beta"):
        for n_configs, err in ((1, 0.30), (5, 0.20), (25, 0.10)):
            rows.append(
                {
                    "dataset": dataset,
                    "method": "CatBoost",
                    "n_configs": n_configs,
                    "err": err,
                    "imp": err * 100,
                    "train_s": 10.0 * n_configs,
                    "infer_s": 0.5,
                    "x_train": n_configs,
                    "x_infer": 0.1,
                    "imputed": 0.0,
                },
            )
    return pd.DataFrame(rows)


def _config(html: str) -> dict:
    return json.loads(re.search(r"const CONFIG = (\{.*?\});\n", html, re.S).group(1))


def _points(html: str) -> list[dict]:
    return json.loads(re.search(r"const POINTS = (\[.*?\]);\n", html, re.S).group(1))


def test_method_records_name_variant_and_family():
    records = method_records(_METHOD_INFO).set_index("method")
    assert records.loc["CAT (tuned)", "name"] == "CatBoost (tuned)"
    # `base` is what the per-dataset trajectory frame is keyed on, so both variants of a model
    # have to collapse onto the same one or picking a contender lights up nothing.
    assert records.loc["CAT (tuned)", "base"] == "CatBoost"
    assert records.loc["CAT (default)", "base"] == "CatBoost"
    assert records.loc["CAT (tuned)", "variant"] == "Tuned"
    assert records.loc["CAT (tuned)", "family"] == "Tree-based"
    # A system is typed from `method_class`, never from its name.
    assert records.loc["AutoGluon 1.6 (extreme, 4h)", "family"] == "System"
    assert records.loc["AutoGluon 1.6 (extreme, 4h)", "variant"] == ""


def test_per_dataset_points_are_the_leaderboard_terms():
    points = per_dataset_points(_results_per_split()).set_index(["dataset", "method"])
    # Improvability is per split against the best on that split, then averaged.
    assert points.loc[("alpha", "CAT (tuned)"), "imp"] == pytest.approx(0.0)
    expected = ((1 - 0.10 / 0.30) + (1 - 0.12 / 0.32)) / 2 * 100
    assert points.loc[("alpha", "CAT (default)"), "imp"] == pytest.approx(expected)
    # Ranks are per split too, so a method that is second on both splits has mean rank 2.
    assert points.loc[("alpha", "AutoGluon 1.6 (extreme, 4h)"), "rank"] == pytest.approx(2.0)
    assert points.loc[("beta", "AutoGluon 1.6 (extreme, 4h)"), "rank"] == pytest.approx(1.0)


def test_dataset_records_fall_back_without_metadata():
    records = dataset_records(_results_per_split(), None)
    assert list(records["dataset"]) == ["alpha", "beta"]
    assert list(records["name"]) == ["alpha", "beta"]
    assert list(records["task"]) == ["binary", "regression"]
    assert list(records["splits"]) == [2, 2]


def test_build_per_dataset_explorer(tmp_path):
    metadata = pd.DataFrame(
        [
            {
                "dataset": "alpha",
                "dataset_name": "Alpha set",
                "num_instances": 500,
                "num_features": 7,
                "num_classes": 2,
                "max_train_rows": 400,
                "domain": "finance",
            },
            {
                "dataset": "beta",
                "dataset_name": "Beta set",
                "num_instances": 200_000,
                "num_features": 3,
                "num_classes": -1,
                "max_train_rows": 160_000,
                "domain": "physics",
            },
        ],
    )
    out = build_per_dataset_explorer_html(
        results_per_split=_results_per_split(),
        method_info=_METHOD_INFO,
        trajectories=_trajectories(),
        dataset_metadata=metadata,
        save_path=tmp_path / "per_dataset_explorer.html",
        title="Per-dataset results",
    )
    assert out is not None
    html = out.read_text(encoding="utf-8")
    config = _config(html)

    assert [d["name"] for d in config["datasets"]] == ["Alpha set", "Beta set"]
    assert config["datasets"][0]["domain"] == "finance"
    # -1 classes is the regression sentinel and must not reach the page as a class count.
    assert config["datasets"][1].get("classes") == -1
    assert config["trajectoryMethods"] == ["CatBoost"]
    assert len(config["trajectory"]["rows"]) == 6
    assert config["trajectory"]["cols"] == ["d", "m", "n", "x", "e", "i"]

    # One record per (dataset, method), indexed into the two lists above.
    points = _points(html)
    assert len(points) == 6
    assert {p["d"] for p in points} == {0, 1}
    assert {p["m"] for p in points} == {0, 1, 2}

    # The page opens on the method with the best mean rank: CatBoost (tuned) is 1st on alpha
    # and 2nd on beta, against the system's 2nd and 1st, and CatBoost (default)'s 3rd and 3rd.
    contender = config["methods"][config["defaultContender"]]
    assert contender["name"] in {"CatBoost (tuned)", "AutoGluon 1.6 (extreme, 4h)"}


def test_build_per_dataset_explorer_without_trajectories(tmp_path):
    """The list half still ships when a cell has no per-dataset trajectory frame."""
    out = build_per_dataset_explorer_html(
        results_per_split=_results_per_split(),
        method_info=_METHOD_INFO,
        trajectories=None,
        dataset_metadata=None,
        save_path=tmp_path / "per_dataset_explorer.html",
    )
    assert out is not None
    config = _config(out.read_text(encoding="utf-8"))
    assert config["trajectoryMethods"] == []
    assert config["trajectory"]["rows"] == []


def _paired_splits(errors_by_method: dict[str, list[float]], dataset: str = "gamma") -> pd.DataFrame:
    rows = []
    for method, errors in errors_by_method.items():
        for fold, error in enumerate(errors):
            rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "method": method,
                    "metric_error": error,
                    "time_train_s": 1.0,
                    "problem_type": "binary",
                    "metric": "roc_auc",
                    "imputed": False,
                },
            )
    return pd.DataFrame(rows)


def test_tied_with_best_separates_a_clear_loser_from_a_near_tie():
    """Six paired splits: a method worse on every split is significantly worse (one-sided
    Wilcoxon, p = 1/64), one that trades wins with the best is tied, and the best is tied with
    itself.
    """
    frame = _paired_splits(
        {
            "best": [0.10, 0.11, 0.12, 0.10, 0.11, 0.12],
            "loser": [0.20, 0.21, 0.22, 0.20, 0.21, 0.22],
            "close": [0.11, 0.10, 0.13, 0.09, 0.12, 0.11],
        },
    )
    tied = tied_with_best(frame).set_index("method")
    assert tied.loc["best", "tied"] is True
    assert tied.loc["best", "p"] is None
    assert tied.loc["loser", "tied"] is False
    assert tied.loc["loser", "p"] == pytest.approx(1 / 64)
    assert tied.loc["close", "tied"] is True


def test_tied_with_best_is_unknown_with_too_few_splits():
    """Below five paired splits the one-sided test cannot reject at 0.05, so no verdict."""
    tied = tied_with_best(_paired_splits({"best": [0.1] * 4, "other": [0.2] * 4})).set_index("method")
    assert tied.loc["best", "tied"] is True
    assert tied.loc["other", "tied"] is None
    tied = tied_with_best(_paired_splits({"best": [0.1] * 5, "other": [0.2] * 5})).set_index("method")
    assert tied.loc["other", "tied"] is False


def test_per_dataset_points_carry_std_and_the_tie_flag():
    points = per_dataset_points(_results_per_split()).set_index(["dataset", "method"])
    assert points.loc[("alpha", "CAT (default)"), "std"] == pytest.approx(pd.Series([0.30, 0.32]).std())
    # Two splits are too few for the test, so the flag is unknown, except for the best method,
    # which is tied with itself.
    assert points.loc[("alpha", "CAT (tuned)"), "tied"] is True
    assert points.loc[("alpha", "CAT (default)"), "tied"] is None


def test_explorer_points_carry_spread_and_tie_fields(tmp_path):
    frame = _paired_splits(
        {
            "CAT (tuned)": [0.10, 0.11, 0.12, 0.10, 0.11, 0.12],
            "CAT (default)": [0.20, 0.21, 0.22, 0.20, 0.21, 0.22],
        },
    )
    method_info = _METHOD_INFO
    path = build_per_dataset_explorer_html(
        results_per_split=frame,
        method_info=method_info,
        dataset_metadata=None,
        save_path=tmp_path / "explorer.html",
    )
    html = path.read_text(encoding="utf-8")
    points = json.loads(re.search(r"const POINTS = (\[.*?\]);\n", html, re.S).group(1))
    by_method = {p["m"]: p for p in points}
    names = list(method_info["method"])
    tuned = by_method[names.index("CAT (tuned)")] if "CAT (tuned)" in names else None
    assert tuned is not None
    assert tuned["b"] == 1
    assert tuned["s"] == pytest.approx(pd.Series([0.10, 0.11, 0.12, 0.10, 0.11, 0.12]).std(), rel=1e-2)
    default = by_method[names.index("CAT (default)")]
    assert default["b"] == 0
    assert '"significance"' in html
