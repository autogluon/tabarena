#!/usr/bin/env python3
"""Evidence for #4505: hill climbing vs Caruana ES on real AutoGluon bagged OOF.

Import order matters on macOS (libomp): lightgbm before torch.
Does not import tabarena (avoids torch pull-in); uses AG EnsembleSelection + local HC.
"""

from __future__ import annotations

# --- libomp-safe import order ---
import lightgbm  # noqa: F401

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.core.metrics import get_metric
from autogluon.core.models.greedy_ensemble.ensemble_selection import EnsembleSelection
from autogluon.tabular import TabularPredictor

# Local HC implementation (mirror of tabarena.simulation.ensemble.hill_climbing_ensembler)
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages" / "tabarena" / "src"))
# Prefer pure-numpy HC below without importing full tabarena stack.


def hill_climb_weights(
    predictions: list[np.ndarray],
    labels: np.ndarray,
    metric,
    problem_type: str,
    precision: float = 0.02,
    max_rounds: int = 40,
    random_state: int = 0,
) -> np.ndarray:
    """Kaggle-style convex blend HC; returns weight vector (sums to 1)."""
    rng = np.random.RandomState(random_state)
    n_models = len(predictions)
    predictions = [np.asarray(p) for p in predictions]

    def error(pred):
        return metric.error(labels, pred)

    singles = np.array([error(p) for p in predictions])
    best_i = int(rng.choice(np.flatnonzero(np.isclose(singles, singles.min()))))
    weights = np.zeros(n_models)
    weights[best_i] = 1.0
    ens = predictions[best_i].copy()
    best_err = singles[best_i]
    grid = np.arange(precision, 1.0 + precision * 0.5, precision)
    grid = np.unique(np.round(grid / precision) * precision)

    for _ in range(max_rounds):
        improved = False
        for j in rng.permutation(n_models):
            best_local = best_err
            best_w = None
            best_pred = None
            for w in grid:
                trial = (1.0 - w) * ens + w * predictions[j]
                if trial.ndim == 2 and problem_type in ("multiclass", "softclass"):
                    s = trial.sum(axis=1, keepdims=True)
                    s = np.where(s == 0, 1.0, s)
                    trial = trial / s
                err = error(trial)
                if err < best_local - 1e-15:
                    best_local = err
                    best_w = float(w)
                    best_pred = trial
            if best_w is not None:
                weights *= 1.0 - best_w
                weights[j] += best_w
                ens = best_pred
                best_err = best_local
                improved = True
        if not improved:
            break
    weights[np.abs(weights) < 1e-12] = 0.0
    weights = np.maximum(weights, 0.0)
    s = weights.sum()
    return weights / s if s > 0 else weights


def _tasks():
    rng = np.random.default_rng(0)
    n, d = 3000, 16
    X = rng.normal(size=(n, d))
    tasks = []
    # binary
    logits = X[:, 0] * 1.2 + X[:, 2] * 0.8 + X[:, 5] * 0.4 + rng.normal(0, 0.6, n)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(d)])
    df["target"] = (logits > 0).astype(int)
    tasks.append(("synth_binary", df, "binary"))
    # regression
    df2 = pd.DataFrame(X, columns=[f"f{i}" for i in range(d)])
    df2["target"] = X[:, 0] * 1.5 + X[:, 1] ** 2 * 0.3 + rng.normal(0, 0.5, n)
    tasks.append(("synth_reg", df2, "regression"))
    # multiclass
    df3 = pd.DataFrame(X, columns=[f"f{i}" for i in range(d)])
    df3["target"] = X[:, :4].argmax(axis=1)
    tasks.append(("synth_multi", df3, "multiclass"))
    return tasks


def _fit_oof(name, df, problem_type, path: Path, time_limit=180):
    path.mkdir(parents=True, exist_ok=True)
    crit = "squared_error" if problem_type == "regression" else "gini"
    hyperparameters = {
        "GBM": [
            {},
            {"extra_trees": True, "ag_args": {"name_suffix": "XT"}},
            {"learning_rate": 0.03, "num_leaves": 64, "ag_args": {"name_suffix": "Large"}},
        ],
        "CAT": [{}, {"depth": 6, "ag_args": {"name_suffix": "D6"}}],
        "XGB": [{}, {"max_depth": 6, "ag_args": {"name_suffix": "D6"}}],
        "RF": [{"criterion": crit}, {"n_estimators": 100, "ag_args": {"name_suffix": "100"}}],
        "XT": [{"criterion": crit}],
        "LR": [{}],
        "KNN": [{"weights": "uniform"}, {"weights": "distance", "ag_args": {"name_suffix": "Dist"}}],
    }
    # sequential bag folds avoid Ray
    for k, v in list(hyperparameters.items()):
        if isinstance(v, list):
            hyperparameters[k] = [
                {**cfg, "ag_args_ensemble": {"fold_fitting_strategy": "sequential_local"}} for cfg in v
            ]
        else:
            hyperparameters[k] = {**v, "ag_args_ensemble": {"fold_fitting_strategy": "sequential_local"}}

    predictor = TabularPredictor(
        label="target",
        problem_type=problem_type,
        path=str(path),
        verbosity=1,
        eval_metric="roc_auc" if problem_type == "binary" else None,
    )
    predictor.fit(
        df,
        hyperparameters=hyperparameters,
        num_bag_folds=5,
        num_stack_levels=0,
        time_limit=time_limit,
        raise_on_model_failure=False,
    )
    trainer = predictor._trainer
    names = [m for m in trainer.get_model_names(level=1) if "WeightedEnsemble" not in m]
    oofs = []
    keep = []
    for m in names:
        try:
            oof = np.asarray(trainer.get_model_oof(m))
            if oof.ndim == 2 and oof.shape[1] == 2 and problem_type == "binary":
                oof = oof[:, 1]
            oofs.append(oof.astype(np.float32))
            keep.append(m)
        except Exception as e:
            print(f"  skip {m}: {e}")
    y = np.asarray(predictor.transform_labels(df["target"]))
    n = min(len(y), min(len(p) for p in oofs))
    return y[:n], [p[:n] for p in oofs], keep


def _split(y, preds, test_frac=0.3, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    te, tr = idx[:n_test], idx[n_test:]
    return y[tr], [p[tr] for p in preds], y[te], [p[te] for p in preds]


def _combine(preds, weights):
    out = None
    for p, w in zip(preds, weights, strict=True):
        if w == 0:
            continue
        out = p * w if out is None else out + p * w
    return out


def main():
    out = Path("artifacts/hill_climbing_4505/ag_oof")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for name, df, ptype in _tasks():
        print(f"\n=== {name} ({ptype}) ===")
        y, preds, model_names = _fit_oof(name, df, ptype, out / f"ag_{name}", time_limit=200)
        print(f"  models: {len(model_names)}")
        y_fit, p_fit, y_te, p_te = _split(y, preds)
        metric = get_metric(
            "roc_auc" if ptype == "binary" else ("log_loss" if ptype == "multiclass" else "rmse"),
            problem_type=ptype,
        )

        # Single best
        fit_errs = [metric.error(y_fit, p) for p in p_fit]
        bi = int(np.argmin(fit_errs))
        sb_w = np.zeros(len(p_fit))
        sb_w[bi] = 1.0
        sb_test = metric.error(y_te, _combine(p_te, sb_w))

        # Greedy Caruana
        es = EnsembleSelection(
            ensemble_size=40, problem_type=ptype, metric=metric, random_state=np.random.RandomState(0)
        )
        es.fit(predictions=list(p_fit), labels=y_fit)
        g_w = np.asarray(es.weights_)
        g_test = metric.error(y_te, _combine(p_te, g_w))
        g_fit = metric.error(y_fit, _combine(p_fit, g_w))

        # Hill climbing
        hc_w = hill_climb_weights(p_fit, y_fit, metric, ptype)
        hc_test = metric.error(y_te, _combine(p_te, hc_w))
        hc_fit = metric.error(y_fit, _combine(p_fit, hc_w))

        for ens, fit_e, te, w in [
            ("single_best", fit_errs[bi], sb_test, sb_w),
            ("greedy_caruana", g_fit, g_test, g_w),
            ("hill_climbing", hc_fit, hc_test, hc_w),
        ]:
            rows.append(
                {
                    "task": name,
                    "problem_type": ptype,
                    "ensembler": ens,
                    "fit_err": fit_e,
                    "test_err": te,
                    "n_models": int((np.asarray(w) != 0).sum()),
                    "n_pool": len(model_names),
                }
            )
            print(f"  {ens:16s} fit={fit_e:.6f} test={te:.6f} n={(np.asarray(w) != 0).sum()}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    pivot = df.pivot_table(index="task", columns="ensembler", values="test_err")
    delta = float((pivot["hill_climbing"] - pivot["greedy_caruana"]).mean())
    summary = {
        "mean_test_err": pivot.mean().to_dict(),
        "hc_minus_greedy_mean": delta,
        "hc_wins": int((pivot["hill_climbing"] < pivot["greedy_caruana"] - 1e-12).sum()),
        "greedy_wins": int((pivot["greedy_caruana"] < pivot["hill_climbing"] - 1e-12).sum()),
        "wall_time_s": time.time() - t0,
        "note": "Real AG bagged OOF; HC = Kaggle convex blend; Greedy = EnsembleSelection",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if delta < -1e-8:
        conclusion = "HC improves mean test error vs Greedy on this AG OOF suite. Confirm on TabArena before AG default change."
        change_default = False  # still need broader TabArena
    elif abs(delta) <= 1e-8:
        conclusion = "HC ≈ Greedy. Do not change AG default ensemble selection."
        change_default = False
    else:
        conclusion = "Greedy better or HC overfits. Do not change AG default ensemble selection."
        change_default = False
    summary["conclusion"] = conclusion
    summary["change_ag_default"] = change_default
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = f"""# AG OOF: Hill climbing vs Greedy (#4505)

Mean test errors: {json.dumps(summary["mean_test_err"], indent=2)}

Mean (HC − Greedy): **{delta:.6g}** (negative ⇒ HC better)

Task wins HC / Greedy: {summary["hc_wins"]} / {summary["greedy_wins"]}

**{conclusion}**

AG default change: **{change_default}**
"""
    (out / "summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
