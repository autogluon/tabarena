from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from tabarena.benchmark.exec_models.grouped_splits import get_recommended_grouped_splits


def _dataset() -> pd.DataFrame:
    """600 rows in 40 groups, a label constant per group and a label per row."""
    rng = np.random.default_rng(0)
    group = rng.integers(0, 40, 600)
    return pd.DataFrame(
        {
            "group": [f"g{i}" for i in group],
            "per_group_label": (np.arange(40) % 3)[group],
            "per_sample_label": rng.integers(0, 2, 600),
        }
    )


def _digest(splits: dict) -> str:
    folds = {
        str(r): {str(f): [list(map(int, tr)), list(map(int, te))] for f, (tr, te) in fs.items()}
        for r, fs in splits.items()
    }
    return hashlib.sha1(json.dumps(folds, sort_keys=True).encode(), usedforsecurity=False).hexdigest()[:16]


# Digests of the splits that data-foundry 0.0.6.dev20261112094419's `curation_recommendations` builds for these
# inputs. A mismatch means the inner validation folds of grouped and temporal tasks changed (in this module or in
# scikit-learn's splitters), so runs before and after the change are no longer on the same folds.
@pytest.mark.parametrize(
    ("group_labels", "stratify_on", "n_repeats", "n_splits", "test_size", "expected"),
    [
        ("per_group", None, 1, 8, None, "212ce833e81d5b17"),
        ("per_group", None, 3, 5, None, "299023b64397ff25"),
        ("per_group", None, 1, 1, 120, "6045779a307a7859"),
        ("per_group", "per_group_label", 1, 8, None, "e81e7131c2f85950"),
        ("per_group", "per_group_label", 3, 5, None, "98d8b69aea993273"),
        ("per_group", "per_group_label", 1, 1, 120, "7f99d1323375ddeb"),
        ("per_sample", None, 1, 8, None, "0e06c15343f4c493"),
        ("per_sample", None, 3, 5, None, "22382f7543197520"),
        ("per_sample", None, 1, 1, 120, "1357b1fb3f3685fe"),
        ("per_sample", "per_sample_label", 1, 8, None, "d02b23d8cf5d6e4d"),
        ("per_sample", "per_sample_label", 3, 5, None, "cd6978e96c918b7d"),
        ("per_sample", "per_sample_label", 1, 1, 120, "cf85201c174ea3b2"),
    ],
)
def test_splits_match_data_foundry(group_labels, stratify_on, n_repeats, n_splits, test_size, expected):
    df = _dataset()
    splits = get_recommended_grouped_splits(
        dataset=df,
        n_repeats=n_repeats,
        n_splits=n_splits,
        group_on="group",
        group_labels=group_labels,
        test_size=test_size,
        stratify_on=stratify_on,
    )
    assert _digest(splits) == expected
    for folds in splits.values():
        for train_idx, test_idx in folds.values():
            assert set(df["group"].iloc[train_idx]).isdisjoint(df["group"].iloc[test_idx])
            assert len(train_idx) + len(test_idx) == len(df)


def test_per_group_rejects_a_label_that_varies_inside_a_group():
    with pytest.raises(ValueError, match="single unique value"):
        get_recommended_grouped_splits(
            dataset=_dataset(),
            n_repeats=1,
            n_splits=5,
            group_on="group",
            group_labels="per_group",
            test_size=None,
            stratify_on="per_sample_label",
        )
