from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bencheval import elo_utils
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


def _ladder_results(n_methods: int = 10, n_tasks: int = 300, seed: int = 0) -> pd.DataFrame:
    """A dense field with a genuine strength ladder, drawn so pairwise outcomes are logistic."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    strengths = np.linspace(3.0, -3.0, n_methods)
    err = -(strengths[:, None] + rng.gumbel(size=(n_methods, n_tasks)))
    return pd.DataFrame(
        {
            "method": np.repeat([f"m{i}" for i in range(n_methods)], n_tasks),
            "task": np.tile([f"t{j}" for j in range(n_tasks)], n_methods),
            "metric_error": err.ravel(),
        }
    )


def test_elo_from_ranks_matches_the_battle_path():
    """The shortcut and the battle path maximise the same likelihood, so they agree."""
    import numpy as np

    from bencheval.elo_utils import EloHelper
    from bencheval.evaluator import BenchmarkEvaluator

    df = _ladder_results()
    evaluator = BenchmarkEvaluator(task_col="task", error_col="metric_error")
    reference = evaluator.compute_elo(
        results_per_task=df, BOOTSTRAP_ROUNDS=1, include_quantiles=False, round_decimals=None
    )
    reference = (reference["elo"] if hasattr(reference, "columns") else reference).astype(float)

    helper = EloHelper(method_col="method", task_col="task", error_col="metric_error")
    fast = helper.compute_mle_elo_from_ranks(results_per_task=df).reindex(reference.index)
    # Bradley-Terry fixes ratings only up to a constant, so compare on a common anchor.
    fast = fast - fast.mean() + reference.mean()

    assert np.abs(fast - reference).max() < 0.05
    assert list(fast.sort_values(ascending=False).index) == list(reference.sort_values(ascending=False).index)


def test_elo_from_ranks_honours_the_calibration_anchor():
    from bencheval.elo_utils import EloHelper

    helper = EloHelper(method_col="method", task_col="task", error_col="metric_error")
    out = helper.compute_mle_elo_from_ranks(
        results_per_task=_ladder_results(), calibration_framework="m3", calibration_elo=1000
    )
    assert out["m3"] == pytest.approx(1000.0)


def test_elo_from_ranks_rejects_a_missing_result():
    """A gap in the schedule must raise, not quietly give a different answer.

    Win totals are a sufficient statistic only when every pair meets on every task.
    """
    from bencheval.elo_utils import EloHelper

    df = _ladder_results()
    helper = EloHelper(method_col="method", task_col="task", error_col="metric_error")
    with pytest.raises(ValueError, match="every method to have every task"):
        helper.compute_mle_elo_from_ranks(results_per_task=df.iloc[:-1])


def test_elo_from_ranks_orders_a_dominant_method_first():
    """A method that wins nearly everything should top the table, ties averaged included."""
    import numpy as np

    from bencheval.elo_utils import EloHelper

    df = _ladder_results(n_methods=5, n_tasks=200)
    wide = df.pivot(index="method", columns="task", values="metric_error")
    wide.loc["m0"] = wide.min(axis=0) - 1.0
    wide.iloc[:, :10] = 0.0  # a block of exact ties, exercising average-rank credit
    df = wide.stack().rename("metric_error").reset_index()

    helper = EloHelper(method_col="method", task_col="task", error_col="metric_error")
    out = helper.compute_mle_elo_from_ranks(results_per_task=df)
    assert out.index[0] == "m0"
    assert np.isfinite(out.to_numpy()).all()


def _ragged_results(n_methods: int = 6, seed: int = 0) -> pd.DataFrame:
    """Results over tasks whose split counts differ, which is where the weighting matters."""
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 1, n_methods)
    rows = []
    for task, n_splits in enumerate([1, 2, 5, 10, 3]):
        for split in range(n_splits):
            for i in range(n_methods):
                rows.append((f"m{i}", f"task{task}", split, -skill[i] + rng.normal(0, 1)))
    return pd.DataFrame(rows, columns=["method", "task", "split", "metric_error"])


def _elo_helper(split_col: str | None = "split") -> EloHelper:
    return EloHelper(method_col="method", task_col="task", error_col="metric_error", split_col=split_col)


def test_rank_path_matches_the_battle_path_on_a_ragged_split_grid():
    """Tasks contribute equally however many splits they ran, as the battle weights specify."""
    results = _ragged_results()
    helper = _elo_helper()

    from_ranks = helper.compute_mle_elo_from_ranks(results_per_task=results, calibration_framework="m0")
    battles = helper.convert_results_to_battles(results_df=results)
    from_battles = helper.compute_mle_elo(battles=battles, calibration_framework="m0")

    pd.testing.assert_series_equal(from_ranks, from_battles.reindex(from_ranks.index), atol=1e-2, rtol=0)


def test_rank_path_matches_the_battle_path_without_splits():
    results = _ragged_results().groupby(["method", "task"], as_index=False)["metric_error"].mean()
    helper = _elo_helper(split_col=None)

    from_ranks = helper.compute_mle_elo_from_ranks(results_per_task=results, calibration_framework="m0")
    from_battles = helper.compute_mle_elo(
        battles=helper.convert_results_to_battles(results_df=results), calibration_framework="m0"
    )

    pd.testing.assert_series_equal(from_ranks, from_battles.reindex(from_ranks.index), atol=1e-2, rtol=0)


def test_rank_path_bootstrap_matches_the_battle_path():
    results = _ragged_results()
    helper = _elo_helper()

    from_ranks = helper.compute_elo_ratings_from_ranks(
        results_per_task=results, BOOTSTRAP_ROUNDS=25, show_process=False
    )
    from_battles = helper.compute_elo_ratings(
        battles=helper.convert_results_to_battles(results_df=results), BOOTSTRAP_ROUNDS=25, show_process=False
    )

    assert from_ranks.shape == from_battles.shape
    for quantile in (0.025, 0.5, 0.975):
        a = from_ranks.quantile(quantile)
        b = from_battles.quantile(quantile).reindex(a.index)
        assert (a - b).abs().max() < 1e-2


def test_ratings_are_centred_on_init_rating():
    """Bradley-Terry fixes strengths only up to a common factor, so the offset needs a convention."""
    elo = _elo_helper().compute_mle_elo_from_ranks(results_per_task=_ragged_results(), INIT_RATING=1000)
    assert elo.mean() == pytest.approx(1000)


def test_rank_path_rejects_an_incomplete_schedule():
    results = _ragged_results()
    helper = _elo_helper()

    assert helper.can_compute_elo_from_ranks(results)
    holey = results.drop(results.index[:1])
    assert not helper.can_compute_elo_from_ranks(holey)
    with pytest.raises(ValueError, match="complete and balanced"):
        helper.compute_mle_elo_from_ranks(results_per_task=holey)


def test_rank_path_rejects_duplicate_rows():
    """The battle path raises on these; averaging them away instead would diverge silently."""
    results = _ragged_results()
    assert not _elo_helper().can_compute_elo_from_ranks(pd.concat([results, results.head(1)]))


def test_fast_elo_is_on_by_default_and_off_under_the_legacy_flag(monkeypatch):
    assert elo_utils.use_fast_elo()

    monkeypatch.setattr(elo_utils, "USE_LEGACY_ELO_SOLVER_TOL", True)
    assert not elo_utils.use_fast_elo(), "the legacy tolerance must reach the solver it names"
    monkeypatch.setattr(elo_utils, "USE_LEGACY_ELO_SOLVER_TOL", False)

    monkeypatch.setattr(elo_utils, "USE_FAST_ELO", False)
    assert not elo_utils.use_fast_elo()
    monkeypatch.setattr(elo_utils, "USE_FAST_ELO", True)

    monkeypatch.setenv(elo_utils.DISABLE_FAST_ELO_ENV_VAR, "1")
    assert not elo_utils.use_fast_elo()
