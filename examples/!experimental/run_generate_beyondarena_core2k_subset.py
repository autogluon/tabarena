"""(Re)generate the committed BeyondArena "core2k" subset.

``core2k`` keeps each dataset's first ``n_d`` splits, where ``n_d`` comes from the cost-weighted split allocation
(:mod:`tabarena.evaluation.split_allocation`): start from Lite (one split per dataset) and repeatedly add the split
with the largest reduction in split-to-split rank noise per second of compute, until the benchmark holds 2,000 splits.
The noise and the compute come from the hosted results on every split of the methods that ran on the full
benchmark (``full_grid_methods``). Running this script rewrites the committed
``BeyondArena_core2k_tasks.csv`` that the ``BeyondArenaContext`` ``"core2k"`` subset predicate reads.

Usage:
    python run_generate_beyondarena_core2k_subset.py [--n-splits 2000] [--out-dir output_beyondarena_core2k]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tabarena.contexts import BeyondArenaContext
from tabarena.contexts.beyondarena.context import CORE2K_TASKS_CSV, CORE_TASKS_CSV
from tabarena.evaluation.split_allocation import (
    available_splits,
    cost_weighted_path,
    full_grid_methods,
    lite_allocation,
    per_split_ranks,
    runtime_s,
    split_costs,
    split_noise,
)

SIZES = ["tiny", "small", "medium", "large"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-splits", type=int, default=2000)
    parser.add_argument("--out-dir", type=Path, default=Path("output_beyondarena_core2k"))
    args = parser.parse_args()

    ctx_all = BeyondArenaContext()
    splits = available_splits(ctx_all)
    methods = full_grid_methods(ctx_all, splits)
    ctx = BeyondArenaContext(methods=[ctx_all.method_metadata(m) for m in methods])
    print(f"methods on the full benchmark: {methods}")

    # 1) the leaderboard on every split; compare writes the per-split results it scored next to it.
    ctx.compare(output_dir=args.out_dir / "full", tasks=[(d, s) for d, ss in splits.items() for s in ss], plot=False)
    ranks = per_split_ranks(pd.read_csv(args.out_dir / "full" / "results_per_split.csv"))

    # 2) per-split compute (train + infer of every hosted config) and per-dataset rank noise.
    repo = ctx.load_repo(methods=list(ctx.methods), config_fallback=None)
    cost = split_costs(repo.metrics().reset_index(), splits)
    sigma = split_noise(ranks, splits)

    # 3) the allocation, as each dataset's first n_d splits.
    allocation = cost_weighted_path(lite_allocation(splits), cost, sigma, [args.n_splits])[args.n_splits]
    rows = [{"dataset": d, "split": s} for d, ss in splits.items() for s in ss[: allocation[d]]]
    tasks = pd.DataFrame(rows).sort_values(["dataset", "split"]).reset_index(drop=True)
    tasks.to_csv(CORE2K_TASKS_CSV, index=False)

    grid = ctx.task_metadata_collection.task_grid()
    size_of = {}
    for s in SIZES:
        for d in grid.loc[ctx.SUBSET_PREDICATES[s](grid).to_numpy(), "dataset"]:
            size_of.setdefault(str(d), s)
    core = pd.read_csv(CORE_TASKS_CSV).groupby("dataset").size().to_dict()
    core = {d: int(core.get(d, 0)) for d in splits}
    by_size = {s: sum(k for d, k in allocation.items() if size_of.get(d) == s) for s in SIZES}
    print(f"Wrote {len(tasks)} core2k (dataset, split) tasks to {CORE2K_TASKS_CSV}")
    print(f"splits per size bucket: {by_size}")
    print(f"compute relative to core: {runtime_s(allocation, cost) / runtime_s(core, cost):.2f}x")
