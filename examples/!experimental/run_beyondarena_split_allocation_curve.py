"""Cost-weighted split allocation curve on BeyondArena, from Lite (or committed core) up to every split.

For each total split count on the path (the start, then every ``--step`` splits, then all splits), and as references
for the committed ``core`` and ``core2k`` subsets and the harmonic-rank stability rule (``stability_rule_counts`` with
reciprocal ranks and Pearson at tau 0.8), writes the following, over the methods that ran on the full benchmark
(``full_grid_methods``):

- the split count per size bucket and the compute (train + infer of every hosted config) relative to committed core
  and to the full benchmark;
- the Elo error against the full benchmark's leaderboard, averaged over ``--draws`` random draws of each dataset's
  allocated number of splits, with its standard error;
- the bootstrap disagreement of the methods' harmonic mean ranks, averaged over datasets with more than one split.

``--suggest`` writes the per-dataset split counts (and disagreement) of the path at those totals next to committed
core and the harmonic-rank rule.

Usage:
    python run_beyondarena_split_allocation_curve.py [--start lite|core] [--step 100] [--draws 32]
        [--workers 24] [--suggest 1300 2400] [--out-dir output_split_allocation]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tabarena.contexts import BeyondArenaContext
from tabarena.contexts.beyondarena.context import CORE2K_TASKS_CSV, CORE_TASKS_CSV

from tabarena.evaluation.split_allocation import (
    available_splits,
    bootstrap_disagreement,
    cost_weighted_path,
    elo_error_vs_full,
    full_grid_methods,
    lite_allocation,
    per_split_ranks,
    runtime_s,
    split_costs,
    split_noise,
    stability_rule_counts,
)

SIZES = ["tiny", "small", "medium", "large"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", choices=["lite", "core"], default="lite")
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--suggest", type=int, nargs="*", default=[])
    parser.add_argument("--out-dir", type=Path, default=Path("output_split_allocation"))
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ctx_all = BeyondArenaContext()
    splits = available_splits(ctx_all)
    methods = full_grid_methods(ctx_all, splits)
    ctx = BeyondArenaContext(methods=[ctx_all.method_metadata(m) for m in methods])
    print(f"methods on the full benchmark: {methods}", flush=True)
    all_tasks = [(d, s) for d, ss in splits.items() for s in ss]
    grid = ctx.task_metadata_collection.task_grid()
    size_of: dict[str, str] = {}
    for s in SIZES:
        for d in grid.loc[ctx.SUBSET_PREDICATES[s](grid).to_numpy(), "dataset"]:
            size_of.setdefault(str(d), s)
    max_train_rows = grid.groupby("dataset")["max_train_rows"].max()

    # full benchmark: its leaderboard and the per-split results compare writes next to it
    lb_full = ctx.compare(output_dir=out / "full", tasks=all_tasks, plot=False)
    lb_full = (lb_full.reset_index() if not isinstance(lb_full.index, pd.RangeIndex) else lb_full).set_index("method")
    ranks = per_split_ranks(pd.read_csv(out / "full" / "results_per_split.csv"))
    repo = ctx.load_repo(methods=list(ctx.methods), config_fallback=None)
    cost = split_costs(repo.metrics().reset_index(), splits)
    del repo
    sigma = split_noise(ranks, splits)
    print(f"inputs ready ({time.time() - t0:.0f}s): {len(splits)} datasets, {len(all_tasks)} splits", flush=True)

    core_counts = pd.read_csv(CORE_TASKS_CSV).groupby("dataset").size()
    core = {d: int(core_counts.get(d, 0)) for d in splits}
    hmr = stability_rule_counts(ranks, splits, value="reciprocal_rank", correlation="pearson")
    start = lite_allocation(splits) if args.start == "lite" else core
    first = sum(start.values())
    targets = sorted(
        {first, *range((first // args.step + 1) * args.step, len(all_tasks), args.step), len(all_tasks), *args.suggest}
    )
    path = cost_weighted_path(start, cost, sigma, targets)

    allocations = {f"cost-weighted {t}": path[t] for t in targets}
    allocations["committed core"] = core
    core2k_counts = pd.read_csv(CORE2K_TASKS_CSV).groupby("dataset").size()
    allocations["committed core2k"] = {d: int(core2k_counts.get(d, 0)) for d in splits}
    allocations["harmonic rank @0.8"] = hmr
    errors = elo_error_vs_full(ctx, splits, allocations, lb_full, seeds=range(args.draws), workers=args.workers)
    print(f"draws done ({time.time() - t0:.0f}s)", flush=True)

    base_rt, full_rt = runtime_s(core, cost), runtime_s(path[targets[-1]], cost)
    rows, per_dataset_dis = [], {}
    for name, k in allocations.items():
        dis = bootstrap_disagreement(ranks, splits, k)
        per_dataset_dis[name] = dis
        e = np.asarray(errors[name])
        row = {"allocation": name, "splits": sum(k.values())}
        row.update({s: sum(n for d, n in k.items() if size_of.get(d) == s) for s in SIZES})
        row.update(
            {
                "runtime vs core": runtime_s(k, cost) / base_rt,
                "% of full runtime": 100 * runtime_s(k, cost) / full_rt,
                "error vs full (Elo)": float(e.mean()),
                "se": float(e.std(ddof=1) / np.sqrt(len(e))) if len(e) > 1 else 0.0,
                "disagreement": float(dis.mean()),
            }
        )
        rows.append(row)
    curve = pd.DataFrame(rows).sort_values(["runtime vs core", "splits"]).reset_index(drop=True)
    curve.to_csv(out / f"curve_from_{args.start}.csv", index=False)
    pd.set_option("display.width", 250)
    print(curve.round(2).to_string(index=False))

    if args.suggest:
        cols = {
            "committed core": core,
            "harmonic rank @0.8": hmr,
            **{f"cost-weighted {t}": path[t] for t in args.suggest},
        }
        table = (
            pd.DataFrame(
                {
                    "size": pd.Series(size_of),
                    "max train rows": max_train_rows,
                    "available": pd.Series({d: len(ss) for d, ss in splits.items()}),
                    **{name: pd.Series(k) for name, k in cols.items()},
                    **{f"disagreement: {name}": per_dataset_dis[name] for name in cols},
                }
            )
            .rename_axis("dataset")
            .sort_values("max train rows")
        )
        table.to_csv(out / f"suggested_splits_from_{args.start}.csv")
        print(f"wrote {out / f'suggested_splits_from_{args.start}.csv'}")
    print(f"total wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
