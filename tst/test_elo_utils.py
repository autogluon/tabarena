import pytest
from bencheval.elo_utils import EloHelper
import pandas as pd


class TestEloHelper:
    @pytest.mark.parametrize('battles,outcome', [
        (pd.DataFrame({
            "method_1": ["Winner", "Winner"],
            "method_2": ["Loser", "Loser"],
            "winner": ["1", "1"],
            "dataset": ["dataset1", "dataset2"],
        }), -1),
        (pd.DataFrame({
            "method_1": ["Model1", "Model1"],
            "method_2": ["Model2", "Model2"],
            "winner": ["tie", "tie"],
            "dataset": ["dataset1", "dataset2"],
        }), 0),
        (pd.DataFrame({
            "method_1": ["Loser", "Loser"],
            "method_2": ["Winner", "Winner"],
            "winner": ["2", "2"],
            "dataset": ["dataset1", "dataset2"],
        }), 1),
    ])
    def test_compute_iterative_elo_scores(self, battles, outcome):
        elo_helper = EloHelper()

        elo_scores = elo_helper.compute_iterative_elo_scores(battles)
        if outcome == -1:
            assert elo_scores[0] > elo_scores[1]
        elif outcome == 1:
            assert elo_scores[0] < elo_scores[1]
        else:
            assert elo_scores[0] == elo_scores[1]
