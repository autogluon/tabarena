"""EXAONE-Tabular's support-row cap: engages only above the ``max_support_cells`` budget."""

from __future__ import annotations

from tabarena.models.exaone_tabular.model import EXAONETabularModel

cap = EXAONETabularModel._support_row_limit


def test_cap_is_off_within_budget():
    # california_house_prices_2020 r1 (20,764 x 657, regression feature_limit 1024) fits the default budget.
    assert cap((20_764, 657), 1024, 100_000, EXAONETabularModel.max_support_cells) is None


def test_cap_counts_only_the_columns_the_model_keeps():
    # 96,064 x 1799 classification keeps 100 columns: 9.6M effective cells, within the budget.
    assert cap((96_064, 1799), 100, 100_000, EXAONETabularModel.max_support_cells) is None


def test_cap_subsamples_above_budget():
    # california_house_prices_2020 r2 (31,146 x 657): 20.5M effective cells.
    assert cap((31_146, 657), 1024, 100_000, 14_000_000) == 14_000_000 // 657


def test_cap_never_raises_the_runtime_limit_and_can_be_disabled():
    assert cap((500_000, 100), 100, 100_000, 14_000_000) == 100_000
    assert cap((31_146, 657), 1024, 100_000, None) is None
