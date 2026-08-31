from __future__ import annotations

import pandas as pd
import pytest

from bencheval.elo_utils import EloHelper


class TestEloHelper:
    @pytest.mark.parametrize(
        ("battles", "outcome"),
        [
            (
                pd.DataFrame(
                    {
                        "method_1": ["Winner", "Winner"],
                        "method_2": ["Loser", "Loser"],
                        "winner": ["1", "1"],
                        "dataset": ["dataset1", "dataset2"],
                    }
                ),
                -1,
            ),
            (
                pd.DataFrame(
                    {
                        "method_1": ["Model1", "Model1"],
                        "method_2": ["Model2", "Model2"],
                        "winner": ["tie", "tie"],
                        "dataset": ["dataset1", "dataset2"],
                    }
                ),
                0,
            ),
            (
                pd.DataFrame(
                    {
                        "method_1": ["Loser", "Loser"],
                        "method_2": ["Winner", "Winner"],
                        "winner": ["2", "2"],
                        "dataset": ["dataset1", "dataset2"],
                    }
                ),
                1,
            ),
        ],
    )
    def test_compute_iterative_elo_scores(self, battles, outcome):
        elo_helper = EloHelper()

        elo_scores = elo_helper.compute_iterative_elo_scores(battles)
        if outcome == -1:
            assert elo_scores[0] > elo_scores[1]
        elif outcome == 1:
            assert elo_scores[0] < elo_scores[1]
        else:
            assert elo_scores[0] == elo_scores[1]


def test_elo_solver_tol_defaults_to_the_converged_value():
    """The flag is off by default: a fresh import computes the converged ratings."""
    from bencheval import elo_utils

    assert elo_utils.USE_LEGACY_ELO_SOLVER_TOL is False
    assert elo_utils.elo_solver_tol() == elo_utils.ELO_SOLVER_TOL


def test_legacy_flag_is_read_at_call_time(monkeypatch):
    """Toggling the flag takes effect without re-importing, so it can be flipped per run."""
    from bencheval import elo_utils

    monkeypatch.setattr(elo_utils, "USE_LEGACY_ELO_SOLVER_TOL", True)
    assert elo_utils.elo_solver_tol() == elo_utils.LEGACY_ELO_SOLVER_TOL
    monkeypatch.setattr(elo_utils, "USE_LEGACY_ELO_SOLVER_TOL", False)
    assert elo_utils.elo_solver_tol() == elo_utils.ELO_SOLVER_TOL


def test_legacy_flag_reproduces_the_looser_fit(monkeypatch):
    """With the flag on, ratings are the under-converged ones: shrunk toward the field mean.

    The ladder has to be wide for this to be measurable: the shrinkage grows with distance from
    the field mean, so on a narrow field it is a fraction of an Elo point and the comparison is
    noise. At +/-6 in log-strength the looser fit is a clear ~2 Elo short at the extremes.
    """
    import numpy as np
    import pandas as pd

    from bencheval import elo_utils
    from bencheval.evaluator import BenchmarkEvaluator

    n_methods, n_tasks = 12, 400
    rng = np.random.default_rng(0)
    strengths = np.linspace(6.0, -6.0, n_methods)
    err = -(strengths[:, None] + rng.gumbel(size=(n_methods, n_tasks)))
    df = pd.DataFrame(
        {
            "method": np.repeat([f"m{i}" for i in range(n_methods)], n_tasks),
            "task": np.tile([f"t{j}" for j in range(n_tasks)], n_methods),
            "metric_error": err.ravel(),
        }
    )
    evaluator = BenchmarkEvaluator(task_col="task", error_col="metric_error")

    def elo() -> pd.Series:
        out = evaluator.compute_elo(
            results_per_task=df, BOOTSTRAP_ROUNDS=1, include_quantiles=False, round_decimals=None
        )
        return (out["elo"] if isinstance(out, pd.DataFrame) else out).astype(float)

    converged = elo()
    monkeypatch.setattr(elo_utils, "USE_LEGACY_ELO_SOLVER_TOL", True)
    legacy = elo()

    def spread(s: pd.Series) -> float:
        return float(s.max() - s.min())

    assert spread(legacy) < spread(converged), "the looser fit should report a compressed spread"
    assert not np.allclose(legacy.to_numpy(), converged.reindex(legacy.index).to_numpy())
    # Both still order the field the same way; only the scale differs.
    assert legacy.rank().equals(converged.reindex(legacy.index).rank())


def test_env_var_enables_the_legacy_tolerance(monkeypatch):
    """The env var turns the legacy fit on without touching code, and is read per call."""
    from bencheval import elo_utils

    for truthy in ("1", "true", "TRUE", "yes", " on ".strip(), "Yes"):
        monkeypatch.setenv(elo_utils.LEGACY_ELO_SOLVER_TOL_ENV_VAR, truthy)
        expected = truthy.strip().lower() in ("1", "true", "yes")
        assert elo_utils.use_legacy_elo_solver_tol() is expected, truthy

    for falsey in ("0", "false", "no", ""):
        monkeypatch.setenv(elo_utils.LEGACY_ELO_SOLVER_TOL_ENV_VAR, falsey)
        assert elo_utils.use_legacy_elo_solver_tol() is False, falsey
        assert elo_utils.elo_solver_tol() == elo_utils.ELO_SOLVER_TOL

    monkeypatch.delenv(elo_utils.LEGACY_ELO_SOLVER_TOL_ENV_VAR, raising=False)
    assert elo_utils.use_legacy_elo_solver_tol() is False


def test_module_flag_and_env_var_are_independent(monkeypatch):
    """Either switch alone is enough; neither can turn the other off."""
    from bencheval import elo_utils

    monkeypatch.delenv(elo_utils.LEGACY_ELO_SOLVER_TOL_ENV_VAR, raising=False)
    monkeypatch.setattr(elo_utils, "USE_LEGACY_ELO_SOLVER_TOL", True)
    assert elo_utils.elo_solver_tol() == elo_utils.LEGACY_ELO_SOLVER_TOL

    monkeypatch.setattr(elo_utils, "USE_LEGACY_ELO_SOLVER_TOL", False)
    monkeypatch.setenv(elo_utils.LEGACY_ELO_SOLVER_TOL_ENV_VAR, "1")
    assert elo_utils.elo_solver_tol() == elo_utils.LEGACY_ELO_SOLVER_TOL

    # A falsey env var does not override an explicitly set module flag.
    monkeypatch.setattr(elo_utils, "USE_LEGACY_ELO_SOLVER_TOL", True)
    monkeypatch.setenv(elo_utils.LEGACY_ELO_SOLVER_TOL_ENV_VAR, "0")
    assert elo_utils.elo_solver_tol() == elo_utils.LEGACY_ELO_SOLVER_TOL
