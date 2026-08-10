#!/usr/bin/env python3
"""Compare Kaggle-style hill climbing vs Caruana greedy ensemble selection (#4505).

#4505 asks: does hill climbing improve ensembling over AutoGluon's current approach?
Verify via TabRepo/TabArena by swapping the ensembler (historical pointer was
``scripts/baseline_comparison/evaluate_baselines.py``; current API is
``repo.evaluate_ensemble`` / ``EnsembleScorer`` with ``ensembler_cls``).

Definitions (see @LennartPurucker on the issue):
  * **GreedyEnsembler** — Caruana et al. 2004 / AG ``EnsembleSelection``
  * **HillClimbingEnsembler** — Kaggle/Matt-OP convex blend hill climb (not continuous BBO)

Usage (synthetic smoke, no dataset download)::

    PYTHONPATH=packages/tabarena/src python scripts/\\!experimental/compare_hill_climbing_vs_greedy.py

Usage with a loaded EvaluationRepository (when you have TabArena caches)::

    from tabarena.simulation.ensemble import GreedyEnsembler, HillClimbingEnsembler

    greedy = repo.evaluate_ensembles(
        configs=configs,
        ensemble_kwargs={"ensembler_cls": GreedyEnsembler, "ensembler_kwargs": {"ensemble_size": 100}},
    )
    hc = repo.evaluate_ensembles(
        configs=configs,
        ensemble_kwargs={
            "ensembler_cls": HillClimbingEnsembler,
            "ensembler_kwargs": {"precision": 0.01, "max_rounds": 50},
        },
    )
    # Compare mean metric_error / rank across tasks.
"""

from __future__ import annotations

import numpy as np
from autogluon.core.metrics import get_metric

from tabarena.simulation.ensemble import GreedyEnsembler, HillClimbingEnsembler, SingleBestEnsembler


def _synthetic_binary(n_models=20, n_samples=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.random(n_samples) < 0.45
    # Diverse strengths + correlated noise
    preds = []
    for i in range(n_models):
        strength = rng.uniform(0.2, 0.85)
        noise = rng.normal(0, 0.15 + 0.05 * (i % 5), n_samples)
        p = np.clip(y.astype(float) * strength + (1 - strength) * 0.5 + noise, 0, 1)
        preds.append(p.astype(np.float32))
    return y.astype(bool), np.stack(preds)


def main():
    y, preds = _synthetic_binary()
    metric = get_metric(metric="roc_auc", problem_type="binary")
    # Hold out last 30% as "test" for a rough generalization check
    n = len(y)
    n_fit = int(0.7 * n)
    y_fit, y_test = y[:n_fit], y[n_fit:]
    preds_fit, preds_test = preds[:, :n_fit], preds[:, n_fit:]

    methods = {
        "single_best": SingleBestEnsembler(problem_type="binary", metric=metric),
        "greedy_caruana": GreedyEnsembler(
            problem_type="binary", metric=metric, ensemble_size=40, random_state=np.random.RandomState(0)
        ),
        "hill_climbing": HillClimbingEnsembler(
            problem_type="binary", metric=metric, precision=0.02, max_rounds=40, random_state=0
        ),
    }

    print(f"{'method':20s}  {'fit_err':>10s}  {'test_err':>10s}  {'n_models':>8s}")
    for name, ens in methods.items():
        ens.fit(predictions=preds_fit, labels=y_fit)
        fit_err = metric.error(y_fit, ens.predict_proba(preds_fit))
        test_err = metric.error(y_test, ens.predict_proba(preds_test))
        n_used = int(ens.models_used().sum())
        print(f"{name:20s}  {fit_err:10.6f}  {test_err:10.6f}  {n_used:8d}")

    print(
        "\nNote: synthetic smoke only. For #4505 evidence, run both ensemblers through "
        "EvaluationRepository.evaluate_ensembles on shared configs/tasks and compare ranks."
    )


if __name__ == "__main__":
    main()
