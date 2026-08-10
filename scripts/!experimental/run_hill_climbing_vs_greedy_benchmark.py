#!/usr/bin/env python3
"""TabArena post-hoc ensemble benchmark: HillClimbing vs Greedy (Caruana) vs SingleBest.

Addresses autogluon/autogluon#4505 investigation 2 using real TabArena OOF caches.

Definition of hill climbing (issue thread):
  Kaggle / Matt-OP convex blend search — NOT continuous black-box HPO.
  Compared to TabArena default GreedyEnsembler (AG EnsembleSelection / Caruana 2004).

Example::

    export PYTHONPATH=packages/tabarena/src:$PYTHONPATH
    # downloads LightGBM processed (~8.5 GB) if missing
    python scripts/\\!experimental/run_hill_climbing_vs_greedy_benchmark.py \\
        --method LightGBM --max-datasets 15 --max-folds 3 --n-configs 40

Outputs JSON + markdown under ``artifacts/hill_climbing_4505/``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _load_method_repo(method: str):
    from tabarena.contexts.tabarena.methods import tabarena_method_metadata_collection

    meta = tabarena_method_metadata_collection.get_method_metadata(method=method)
    if not meta.path_processed_exists:
        print(f"Downloading processed artifacts for {method} -> {meta.path_processed} ...")
        meta.method_downloader(verbose=True).download_processed()
    return meta.load_processed(), meta


def _evaluate_one(repo, dataset: str, fold: int, configs: list[str], ensembler_cls, ensembler_kwargs: dict):
    df_result, df_weights = repo.evaluate_ensemble(
        dataset=dataset,
        fold=fold,
        configs=configs,
        ensemble_kwargs={
            "ensembler_cls": ensembler_cls,
            "ensembler_kwargs": ensembler_kwargs,
        },
    )
    # df_result is typically multi-index or single row with metric_error etc.
    row = df_result.reset_index(drop=True).iloc[0].to_dict()
    n_used = int((df_weights.iloc[0] != 0).sum()) if len(df_weights) else 0
    row["n_models_used"] = n_used
    return row


def run_benchmark(
    method: str = "LightGBM",
    max_datasets: int | None = 20,
    max_folds: int | None = 3,
    n_configs: int | None = 50,
    ensemble_size: int = 40,
    hc_precision: float = 0.02,
    hc_max_rounds: int = 30,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    from tabarena.simulation.ensemble import GreedyEnsembler, HillClimbingEnsembler, SingleBestEnsembler

    repo, meta = _load_method_repo(method)
    datasets = list(repo.datasets())
    if max_datasets is not None:
        datasets = datasets[:max_datasets]
    configs = list(repo.configs())
    if n_configs is not None:
        configs = configs[:n_configs]

    # Discover folds via first dataset metrics / tasks
    folds_available = sorted({t[1] for t in repo.tasks() if t[0] == datasets[0]}) if hasattr(repo, "tasks") else [0]
    if not folds_available:
        # fallback
        folds_available = list(range(3))
    if max_folds is not None:
        folds_available = folds_available[:max_folds]

    methods = {
        "single_best": (SingleBestEnsembler, {}),
        "greedy_caruana": (
            GreedyEnsembler,
            {"ensemble_size": ensemble_size, "random_state": np.random.RandomState(0)},
        ),
        "hill_climbing": (
            HillClimbingEnsembler,
            {
                "precision": hc_precision,
                "max_rounds": hc_max_rounds,
                "random_state": 0,
            },
        ),
    }

    rows = []
    t0 = time.time()
    for dataset in datasets:
        for fold in folds_available:
            for name, (cls, kwargs) in methods.items():
                try:
                    t1 = time.time()
                    result = _evaluate_one(repo, dataset, fold, configs, cls, kwargs)
                    elapsed = time.time() - t1
                    rows.append(
                        {
                            "method_pool": method,
                            "ensembler": name,
                            "dataset": dataset,
                            "fold": fold,
                            "metric_error": result.get("metric_error"),
                            "metric_error_val": result.get("metric_error_val"),
                            "n_models_used": result.get("n_models_used"),
                            "time_s": elapsed,
                            "n_configs_pool": len(configs),
                        }
                    )
                    print(
                        f"[{len(rows)}] {dataset} fold={fold} {name}: "
                        f"test_err={result.get('metric_error')} val_err={result.get('metric_error_val')} "
                        f"n_used={result.get('n_models_used')} ({elapsed:.2f}s)"
                    )
                except Exception as e:
                    print(f"FAIL {dataset} fold={fold} {name}: {type(e).__name__}: {e}")
                    rows.append(
                        {
                            "method_pool": method,
                            "ensembler": name,
                            "dataset": dataset,
                            "fold": fold,
                            "metric_error": np.nan,
                            "metric_error_val": np.nan,
                            "n_models_used": np.nan,
                            "time_s": np.nan,
                            "n_configs_pool": len(configs),
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )

    df = pd.DataFrame(rows)
    out_dir = out_dir or Path("artifacts/hill_climbing_4505")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"benchmark_{method}_d{len(datasets)}_f{len(folds_available)}_c{len(configs)}.csv"
    df.to_csv(csv_path, index=False)

    summary = _summarize(df)
    md_path = out_dir / f"summary_{method}.md"
    md_path.write_text(summary["markdown"], encoding="utf-8")
    json_path = out_dir / f"summary_{method}.json"
    json_path.write_text(json.dumps(summary["stats"], indent=2, default=str), encoding="utf-8")

    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Total wall time: {time.time() - t0:.1f}s")
    print(summary["markdown"])
    return df


def _summarize(df: pd.DataFrame) -> dict:
    """Compare ensemblers: win rates on test metric_error (lower better)."""
    ok = df.dropna(subset=["metric_error"])
    if ok.empty:
        return {"markdown": "No successful runs.\n", "stats": {}}

    pivot = ok.pivot_table(index=["dataset", "fold"], columns="ensembler", values="metric_error", aggfunc="first")
    stats: dict = {"n_tasks": int(len(pivot)), "mean_error": {}, "wins_vs_greedy": {}, "ties_vs_greedy": {}, "mean_n_models": {}}

    for col in pivot.columns:
        stats["mean_error"][col] = float(pivot[col].mean())

    if "greedy_caruana" in pivot.columns and "hill_climbing" in pivot.columns:
        g, h = pivot["greedy_caruana"], pivot["hill_climbing"]
        stats["wins_vs_greedy"]["hill_climbing"] = int((h < g - 1e-12).sum())
        stats["ties_vs_greedy"]["hill_climbing"] = int(np.isclose(h, g, rtol=0, atol=1e-12).sum())
        stats["wins_vs_greedy"]["greedy_caruana"] = int((g < h - 1e-12).sum())
        stats["mean_delta_hc_minus_greedy"] = float((h - g).mean())  # negative => HC better

    if "single_best" in pivot.columns and "hill_climbing" in pivot.columns:
        s, h = pivot["single_best"], pivot["hill_climbing"]
        stats["wins_vs_single_best"] = {
            "hill_climbing": int((h < s - 1e-12).sum()),
            "single_best": int((s < h - 1e-12).sum()),
        }

    n_models = ok.groupby("ensembler")["n_models_used"].mean()
    stats["mean_n_models"] = {k: float(v) for k, v in n_models.items()}

    lines = [
        "# Hill climbing vs Greedy ensemble selection (TabArena OOF)",
        "",
        "Issue: [autogluon/autogluon#4505](https://github.com/autogluon/autogluon/issues/4505)",
        "",
        "## Definitions",
        "",
        "- **greedy_caruana**: TabArena `GreedyEnsembler` → AutoGluon `EnsembleSelection` (Caruana et al. 2004).",
        "- **hill_climbing**: `HillClimbingEnsembler` — Kaggle/Matt-OP convex blend search (not continuous BBO).",
        "- **single_best**: best validation model only.",
        "",
        f"Tasks (dataset × fold): **{stats['n_tasks']}**",
        "",
        "## Mean test metric_error (lower is better)",
        "",
    ]
    for k, v in sorted(stats["mean_error"].items(), key=lambda x: x[1]):
        lines.append(f"- `{k}`: {v:.6g}")
    lines.append("")
    if "mean_delta_hc_minus_greedy" in stats:
        d = stats["mean_delta_hc_minus_greedy"]
        lines.append(f"Mean (HC − Greedy) test error: **{d:.6g}** (negative means HC wins on average)")
        lines.append(
            f"Task wins: HC {stats['wins_vs_greedy'].get('hill_climbing', 0)} / "
            f"Greedy {stats['wins_vs_greedy'].get('greedy_caruana', 0)} / "
            f"ties {stats['ties_vs_greedy'].get('hill_climbing', 0)}"
        )
        lines.append("")
        if d < -1e-8:
            lines.append("**Conclusion (this slice):** Hill climbing improves mean test error vs Greedy.")
            lines.append("→ Candidate for further portfolio study; AG default change only after broader confirmation.")
        elif abs(d) <= 1e-8:
            lines.append("**Conclusion (this slice):** Hill climbing ≈ Greedy (no meaningful mean difference).")
            lines.append("→ **Do not** change AG default ensemble selection based on this evidence.")
        else:
            lines.append("**Conclusion (this slice):** Greedy is better or HC overfits OOF.")
            lines.append("→ **Do not** change AG default; keep GreedyEnsembler / EnsembleSelection.")
    lines.append("")
    lines.append("## Mean models used")
    for k, v in stats["mean_n_models"].items():
        lines.append(f"- `{k}`: {v:.2f}")
    lines.append("")
    return {"markdown": "\n".join(lines) + "\n", "stats": stats}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="LightGBM")
    p.add_argument("--max-datasets", type=int, default=15)
    p.add_argument("--max-folds", type=int, default=3)
    p.add_argument("--n-configs", type=int, default=40)
    p.add_argument("--ensemble-size", type=int, default=40)
    p.add_argument("--hc-precision", type=float, default=0.02)
    p.add_argument("--hc-max-rounds", type=int, default=30)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/hill_climbing_4505"))
    args = p.parse_args()
    run_benchmark(
        method=args.method,
        max_datasets=args.max_datasets,
        max_folds=args.max_folds,
        n_configs=args.n_configs,
        ensemble_size=args.ensemble_size,
        hc_precision=args.hc_precision,
        hc_max_rounds=args.hc_max_rounds,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
