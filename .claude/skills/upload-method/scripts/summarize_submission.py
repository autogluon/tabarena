"""Print the validity checks for a scanned submission and write default/tuned rows for ``compare``.

Usage::

    python -I summarize_submission.py <scan.parquet> [--reference ref.parquet] [--expected-configs cfg.json]
        [--new-results new_results.parquet]

``--expected-configs`` is a JSON object ``{"<config dir name>": {<hyperparameter>: <value>, ...}}`` with the
configs the PR's search space generates (re-implement or call the PR's ``hpo.py`` generator). Every key listed
must match the child hyperparameters recorded in the results. ``--new-results`` writes one ``(default)`` and one
``(tuned)`` row per method and split (tuned = the config with the lowest validation error), in the columns
``TabArenaContext.compare(new_results=...)`` takes.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd


def _same(found, expected) -> bool:
    if isinstance(expected, float) and isinstance(found, (int, float)):
        return bool(np.isclose(found, expected, rtol=1e-12, atol=0.0))
    return found == expected


def _section(title: str) -> None:
    print(f"\n== {title}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan")
    ap.add_argument("--reference")
    ap.add_argument("--expected-configs")
    ap.add_argument("--new-results")
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    df = pd.read_parquet(args.scan)
    for col in ("refused", "load_error", "recomp_error", "child_error"):
        if col in df and df[col].notna().any():
            print(f"!! {col}: {int(df[col].notna().sum())} files, e.g. {df[col].dropna().iloc[0]}")
    df = df[df.framework.notna()].copy()
    df["method"] = df.dir_config.str.replace(r"_(c|r)\d+_.*$", "", regex=True)

    _section("Coverage and protocol")
    g = df.groupby("method")
    print(g.agg(results=("path", "size"), configs=("dir_config", "nunique"), datasets=("dataset", "nunique")))
    per_ds = df.groupby(["method", "dataset"]).agg(n=("path", "size"), c=("dir_config", "nunique"))
    print("splits per dataset:", (per_ds.n / per_ds.c).value_counts().to_dict())
    print("framework == config dir:", bool((df.framework == df.dir_config).all()))
    for col in ("n_children", "max_time_limit", "num_cpus", "num_gpus"):
        print(f"{col}:", df[col].unique().tolist())
    print("max time_train_s:", round(float(df.time_train_s.max()), 1))

    if args.reference:
        _section("Splits and labels vs our own run")
        ref = pd.read_parquet(args.reference)
        m = df.merge(ref, on=["tid", "repeat", "fold"], how="left", suffixes=("", "_ref"))
        for h in ("y_test_idx_hash", "y_val_idx_hash", "y_test_hash", "y_val_hash"):
            print(
                f"{h}: match {float((m[h] == m[h + '_ref']).mean()):.4f} (no reference: {int(m[h + '_ref'].isna().sum())})"
            )

    _section("Recomputed metrics (log-loss differs slightly through probability clipping)")
    rel = (df.recomp_test_err - df.metric_error).abs() / df.metric_error.abs().clip(lower=1e-12)
    print(df.assign(rel=rel).groupby("metric").rel.max().round(6).to_dict())

    _section("Prediction sanity")
    print(
        "non-finite:",
        int((~df.finite).sum()),
        " proba out of [0, 1]:",
        int(df.proba_out_of_range.sum()),
        " row sums off by > 1e-3:",
        int((df.rowsum_dev > 1e-3).sum()),
    )
    df["const"] = df.test_n_unique <= 1
    df["all_children_const"] = df.n_children_const_like == df.n_children
    print(
        "hidden failed fits per method (a dataset where every method is near the constant, such as "
        "hiva_agnostic, inflates these; compare with the c1 rows):"
    )
    print(
        df.groupby("method").agg(
            constant_predictions=("const", "sum"),
            all_children_const=("all_children_const", "sum"),
            some_children_const=("n_children_const_like", lambda s: int((s > 0).sum())),
        )
    )
    print("c1 only:", df[df.dir_config.str.contains("_c1_")].groupby("method").all_children_const.sum().to_dict())
    worst = df.groupby("dir_config").all_children_const.mean().sort_values(ascending=False).head(10)
    print("configs with the most failed fits:\n", worst.round(3).to_string())

    if args.expected_configs:
        _section("Hyperparameters vs the search space")
        with open(args.expected_configs) as f:
            exp = json.load(f)
        firsts = df.drop_duplicates("dir_config").set_index("dir_config").hp_json
        bad = 0
        for name, cfg in exp.items():
            hp = json.loads(firsts.get(name, "{}"))
            diff = {k: (hp.get(k), v) for k, v in cfg.items() if not _same(hp.get(k), v)}
            if name not in firsts.index or diff:
                bad += 1
                if bad <= 3:
                    print("mismatch", name, diff or "missing")
        print(
            f"configs checked: {len(exp)}, mismatching: {bad}; variants per config (max):",
            int(df.groupby("dir_config").hp_json.nunique().max()),
        )

    _section("Generalization gap, median per-dataset log(test / val)")
    df["gap"] = np.log(df.metric_error / df.val_err)
    print(df.groupby(["method", "dataset"]).gap.median().groupby(level=0).median().round(4).to_dict())

    if args.new_results:
        df["fold_idx"] = df.repeat * 3 + df.fold
        default = df[df.dir_config.str.contains("_c1_")].assign(kind="default")
        tuned = df.loc[df.groupby(["method", "dataset", "fold_idx"]).val_err.idxmin()].assign(kind="tuned")
        out = pd.concat([default, tuned], ignore_index=True)
        key = out.method.str.upper()
        nr = pd.DataFrame(
            {
                "dataset": out.dataset,
                "fold": out.fold_idx,
                "method": key + " (" + out.kind + ")",
                "metric_error": out.metric_error,
                "time_train_s": out.time_train_s,
                "time_infer_s": out.time_infer_s,
                "metric": out.metric,
                "problem_type": out.problem_type,
                "metric_error_val": out.val_err,
                "method_type": np.where(out.kind == "default", "config", "hpo"),
                "method_subtype": out.kind,
                "config_type": key,
                "ta_name": out.method,
                "ta_suite": "submission",
            }
        )
        nr.to_parquet(args.new_results)
        print("\nwrote", args.new_results, "(tuned time_train_s is the picked config's, not the search's)")


if __name__ == "__main__":
    main()
