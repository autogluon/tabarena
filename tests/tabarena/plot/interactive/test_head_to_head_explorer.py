from __future__ import annotations

import json
import re

import pandas as pd

from tabarena.plot.interactive.head_to_head_explorer import (
    build_head_to_head_explorer_html,
    column_kinds,
    dataset_axes,
    dataset_facts,
    head_to_head_points,
    round_to_dataset_precision,
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
        },
        {
            "method": "CAT (tuned)",
            "method_type": "hpo",
            "method_subtype": "tuned",
            "config_type": "CAT",
            "display_name": "CatBoost",
            "method_class": "model",
        },
        {
            "method": "My portfolio",
            "method_type": "baseline",
            "method_subtype": None,
            "config_type": None,
            "display_name": "My portfolio",
            "method_class": "model",
        },
    ],
)


def _results_per_split(errors: dict[tuple[str, str], list[float]]) -> pd.DataFrame:
    rows = [
        {
            "dataset": dataset,
            "fold": fold,
            "method": method,
            "metric_error": value,
            "time_train_s": 10.0,
            "time_infer_s": 0.5,
            "metric": "log_loss",
            "problem_type": "binary",
            "imputed": False,
        }
        for (dataset, method), values in errors.items()
        for fold, value in values
    ]
    return pd.DataFrame(rows)


_ERRORS = {
    # `beta`'s splits come in out of order; the per-split errors must still follow the fold order.
    ("alpha", "CAT (default)"): [(0, 0.30), (1, 0.32), (2, 0.31)],
    ("alpha", "CAT (tuned)"): [(0, 0.10), (1, 0.12), (2, 0.11)],
    ("alpha", "My portfolio"): [(0, 0.20), (1, 0.22), (2, 0.21)],
    ("beta", "CAT (default)"): [(2, 1.3), (0, 1.0), (1, 1.2)],
    ("beta", "CAT (tuned)"): [(2, 0.7), (0, 0.8), (1, 0.9)],
    ("beta", "My portfolio"): [(2, 0.6), (0, 0.4), (1, 0.5)],
}


def _config(html: str) -> dict:
    return json.loads(re.search(r"const CONFIG = (\{.*?\});\n", html, re.S).group(1))


def _points(html: str) -> list[dict]:
    return json.loads(re.search(r"const POINTS = (\[.*?\]);\n", html, re.S).group(1))


def test_each_methods_split_errors_follow_the_datasets_fold_order():
    points, folds = head_to_head_points(_results_per_split(_ERRORS))
    assert folds == {"alpha": [0, 1, 2], "beta": [0, 1, 2]}
    by_key = points.set_index(["dataset", "method"])
    assert by_key.loc[("beta", "CAT (default)"), "splits"] == [1.0, 1.2, 1.3]
    assert by_key.loc[("beta", "My portfolio"), "splits"] == [0.4, 0.5, 0.6]
    # The rank is among the whole field of the cell, not between two methods.
    assert by_key.loc[("beta", "My portfolio"), "rank"] == 1.0
    assert by_key.loc[("beta", "CAT (default)"), "rank"] == 3.0


def test_a_method_missing_a_split_of_a_dataset_is_left_out_of_that_dataset():
    errors = dict(_ERRORS)
    errors[("beta", "My portfolio")] = [(0, 0.4), (1, 0.5)]
    points, _ = head_to_head_points(_results_per_split(errors))
    kept = set(zip(points["dataset"], points["method"], strict=True))
    assert ("beta", "My portfolio") not in kept
    assert ("alpha", "My portfolio") in kept


def test_the_page_opens_on_the_named_pair(tmp_path):
    page = build_head_to_head_explorer_html(
        results_per_split=_results_per_split(_ERRORS),
        method_info=_METHOD_INFO,
        method_a="My portfolio",
        method_b="CatBoost (default)",
        save_path=tmp_path / "h2h.html",
    )
    html = page.read_text(encoding="utf-8")
    config = _config(html)
    names = [m["name"] for m in config["methods"]]
    assert names[config["defaultA"]] == "My portfolio"
    assert names[config["defaultB"]] == "CatBoost (default)"
    assert [d["folds"] for d in config["datasets"]] == [[0, 1, 2], [0, 1, 2]]
    # Every method ran every split, so all are in the Elo field and their split errors are its grid rows.
    beta = next(d for d, ds in enumerate(config["datasets"]) if ds["key"] == "beta")
    portfolio = names.index("My portfolio")
    elo = config["elo"]
    assert sorted(elo["field"]) == [0, 1, 2]
    assert elo["grid"][beta][elo["field"].index(portfolio)] == [0.4, 0.5, 0.6]
    assert all(p["f"] is None for p in _points(html))
    assert elo["calibration"] is None  # no RF (default) here


def test_without_a_named_pair_the_page_opens_on_the_two_best_mean_ranks(tmp_path):
    page = build_head_to_head_explorer_html(
        results_per_split=_results_per_split(_ERRORS),
        method_info=_METHOD_INFO,
        method_a="Not a method",
        save_path=tmp_path / "h2h.html",
    )
    config = _config(page.read_text(encoding="utf-8"))
    names = [m["name"] for m in config["methods"]]
    # Mean ranks: CatBoost (tuned) 1.5, My portfolio 1.5, CatBoost (default) 3.
    assert {names[config["defaultA"]], names[config["defaultB"]]} == {"CatBoost (tuned)", "My portfolio"}


def test_a_page_needs_two_methods(tmp_path):
    errors = {key: value for key, value in _ERRORS.items() if key[1] == "CAT (default)"}
    page = build_head_to_head_explorer_html(
        results_per_split=_results_per_split(errors),
        method_info=_METHOD_INFO,
        save_path=tmp_path / "h2h.html",
    )
    assert page is None
    assert not (tmp_path / "h2h.html").exists()


def test_floating_point_noise_in_a_metric_is_a_tie():
    """An AUC one bit short of 1 (1 - AUC = 1.1e-16) against an exact 0 neither wins the split nor the mean, while a
    real difference far below the dataset's typical error still counts.
    """
    errors = {
        ("alpha", "CAT (default)"): [(0, 0.0), (1, 0.0), (2, 1.73e-6)],
        ("alpha", "CAT (tuned)"): [(0, 0.0), (1, 1.1102230246251565e-16), (2, 1.73e-6)],
        ("alpha", "My portfolio"): [(0, 0.03), (1, 0.03), (2, 0.04)],
    }
    points, _ = head_to_head_points(_results_per_split(errors))
    by_key = points.set_index(["dataset", "method"])
    assert by_key.loc[("alpha", "CAT (tuned)"), "splits"] == by_key.loc[("alpha", "CAT (default)"), "splits"]
    assert by_key.loc[("alpha", "CAT (tuned)"), "splits"] == [0.0, 0.0, 1.73e-6]
    assert by_key.loc[("alpha", "CAT (tuned)"), "err"] == by_key.loc[("alpha", "CAT (default)"), "err"]
    assert by_key.loc[("alpha", "CAT (tuned)"), "rank"] == by_key.loc[("alpha", "CAT (default)"), "rank"]


def test_the_rounding_keeps_each_datasets_own_scale():
    frame = _results_per_split(
        {
            ("small", "CAT (default)"): [(0, 0.012345678912)],
            ("small", "CAT (tuned)"): [(0, 0.012345679)],
            ("large", "CAT (default)"): [(0, 12345.678912)],
            ("large", "CAT (tuned)"): [(0, 12345.679)],
        }
    )
    rounded = round_to_dataset_precision(frame).set_index(["dataset", "method"])["metric_error"]
    assert rounded[("small", "CAT (default)")] == rounded[("small", "CAT (tuned)")] == 0.01234568
    assert rounded[("large", "CAT (default)")] == rounded[("large", "CAT (tuned)")] == 12345.68


def test_each_dataset_carries_its_column_kinds_as_the_arena_draws_them(tmp_path):
    metadata = pd.DataFrame(
        {
            "dataset": ["alpha", "beta"],
            "num_text_cols": [2, 0],
            "num_high_cardinality_cats": [1, 0],
            "has_datetime": [True, False],
            "has_categorical": [True, None],
        }
    )
    assert column_kinds(metadata) == {
        "alpha": ["text", "high-cardinality", "datetime", "categorical"],
        # without `has_categorical` neither its kind nor "numerical" can be told
        "beta": [],
    }
    page = build_head_to_head_explorer_html(
        results_per_split=_results_per_split(_ERRORS),
        method_info=_METHOD_INFO,
        dataset_metadata=metadata.assign(has_categorical=[True, False]),
        save_path=tmp_path / "h2h.html",
    )
    config = _config(page.read_text(encoding="utf-8"))
    assert {d["key"]: d["kinds"] for d in config["datasets"]} == {
        "alpha": ["text", "high-cardinality", "datetime", "categorical"],
        "beta": ["numerical"],
    }
    assert column_kinds(metadata[["dataset"]]) == {}


def test_the_dataset_pane_lists_the_metadata_it_has_with_flags_in_place_of_missing_counts():
    metadata = pd.DataFrame(
        {
            "dataset": ["alpha"],
            "num_instances": [12345],
            "num_text_cols": [2],
            "has_categorical": [True],
            "missing_value_fraction": [0.0123],
            "target_imbalance_ratio": [4.0],
            "group_on": [None],
        }
    )
    assert dataset_facts(metadata) == {
        "alpha": [
            ["Rows", "12,345"],
            ["Categorical columns", "yes"],
            ["Text columns", "2"],
            ["Missing cells", "1.23%"],
            ["Class ratio (majority : minority)", "4.0 : 1"],
        ]
    }
    counted = metadata.assign(num_categorical_cols=[7])
    assert ["Categorical columns", "7"] in dataset_facts(counted)["alpha"]


def test_a_method_without_a_result_on_every_split_stays_out_of_the_elo_field(tmp_path):
    errors = dict(_ERRORS)
    errors[("beta", "My portfolio")] = [(0, 0.4), (1, 0.5)]
    page = build_head_to_head_explorer_html(
        results_per_split=_results_per_split(errors),
        method_info=_METHOD_INFO,
        elo_calibration="CAT (default)",
        save_path=tmp_path / "h2h.html",
    )
    html = page.read_text(encoding="utf-8")
    config = _config(html)
    names = [m["name"] for m in config["methods"]]
    portfolio = names.index("My portfolio")
    assert portfolio not in config["elo"]["field"]
    assert config["elo"]["calibration"] == names.index("CatBoost (default)")
    # outside the field, its measured dataset carries its own split errors
    alpha = next(d for d, ds in enumerate(config["datasets"]) if ds["key"] == "alpha")
    assert next(p["f"] for p in _points(html) if p["d"] == alpha and p["m"] == portfolio) == [0.2, 0.22, 0.21]


def test_the_chart_offers_the_metadata_axes_some_dataset_has_a_value_for(tmp_path):
    metadata = pd.DataFrame(
        {
            "dataset": ["alpha", "beta"],
            "max_categorical_levels": [38, None],
            "missing_value_fraction": [0.0, 0.25],
            "num_text_cols": [None, None],
        }
    )
    axes, values = dataset_axes(metadata)
    assert [axis["key"] for axis in axes] == ["lev", "miss"]
    assert values == {"alpha": {"lev": 38.0, "miss": 0.0}, "beta": {"miss": 0.25}}
    page = build_head_to_head_explorer_html(
        results_per_split=_results_per_split(_ERRORS),
        method_info=_METHOD_INFO,
        dataset_metadata=metadata,
        save_path=tmp_path / "h2h.html",
    )
    config = _config(page.read_text(encoding="utf-8"))
    assert [axis["label"] for axis in config["xAxes"]] == ["Most levels of a categorical", "Missing cells"]
    assert {d["key"]: d.get("x") for d in config["datasets"]} == values
    assert all("it" in p for p in _points(page.read_text(encoding="utf-8")))


def test_the_dataset_pane_leaves_out_values_that_say_nothing():
    metadata = pd.DataFrame(
        {
            "dataset": ["reg", "grouped"],
            "num_instances": [100, 100],
            "num_classes": [-1, 3],
            "num_instance_groups": [100, 12],
        }
    )
    facts = dataset_facts(metadata)
    assert facts["reg"] == [["Rows", "100"]]
    assert facts["grouped"] == [["Rows", "100"], ["Classes", "3"], ["Groups", "12"]]
