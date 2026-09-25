"""Cost-weighted split allocation: how many evaluation splits each dataset of an arena gets.

A dataset's score is a mean over its splits, so its noise shrinks as it gets more splits, while one split of a large
dataset can cost as much compute as hundreds of splits of a small one. :func:`cost_weighted_path` starts from a
minimum allocation (Lite: one split per dataset) and repeatedly adds the split with the largest reduction in
split-to-split noise per second of compute,

    gain_d = sigma_d^2 / (n_d (n_d + 1) c_d),

where ``sigma_d^2`` is the split-to-split variance of the methods' ranks on dataset ``d``, ``n_d`` its current split
count and ``c_d`` the compute of its next split. This greedy procedure approximates the optimal allocation for
stratified sampling with unequal costs (Neyman allocation), ``n_d ~ sigma_d / sqrt(c_d)``.

An allocation is scored by :func:`elo_error_vs_full` (how far the leaderboard on the chosen splits lands from the
leaderboard on every split) and :func:`bootstrap_disagreement` (how much two independent draws of the chosen number of
splits disagree on the methods' harmonic mean ranks; it does not treat the full set of splits as the truth).
:func:`stability_rule_counts` gives the split counts of the fold-similarity stability rule that picks BeyondArena's
``core``, for comparison. Every allocation uses each dataset's first ``n_d`` splits. BeyondArena's ``core2k`` subset is
this allocation from Lite at 2,000 splits (``examples/!experimental/run_generate_beyondarena_core2k_subset.py``).
"""

from __future__ import annotations

import contextlib
import heapq
import io
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from tabarena.contexts.abstract_arena_context import AbstractArenaContext

#: dataset -> its split indices, in the order splits are added.
Splits = dict[str, list[int]]
#: dataset -> number of splits it uses (its first ``n``).
Allocation = dict[str, int]


def available_splits(ctx: AbstractArenaContext) -> Splits:
    """Every dataset's split indices in the arena's task definitions, ascending."""
    grid = ctx.task_metadata_collection.task_grid()
    return {str(d): sorted(int(s) for s in g["split"].unique()) for d, g in grid.groupby("dataset")}


def full_grid_methods(ctx: AbstractArenaContext, splits: Splits, *, min_coverage: float = 0.95) -> list[str]:
    """The context's methods whose hosted results cover at least ``min_coverage`` of the arena's splits.

    The allocation reads every split's cost and rank noise, so a method that ran on a subset only (e.g. only the core
    splits) would enter with imputed ranks outside it and with its compute counted on its subset's splits alone.
    """
    grid = {(d, s) for d, ss in splits.items() for s in ss}
    keep = []
    for method in ctx.methods:
        metrics = ctx.load_repo(methods=[method], config_fallback=None).metrics().reset_index()
        covered = {(str(d), int(f)) for d, f in zip(metrics["dataset"], metrics["fold"], strict=True)} & grid
        if len(covered) >= min_coverage * len(grid):
            keep.append(method)
    return keep


def per_split_ranks(results_per_split: pd.DataFrame) -> pd.DataFrame:
    """Each method's rank on each split (1 = lowest ``metric_error``, ties averaged), as bencheval ranks them.

    ``results_per_split`` has one row per ``(dataset, fold, method)`` with a ``metric_error``, e.g. the
    ``results_per_split.csv`` that ``compare`` writes to its ``output_dir``.
    """
    out = results_per_split[["dataset", "fold", "method", "metric_error"]].copy()
    out["rank"] = out.groupby(["dataset", "fold"])["metric_error"].rank(method="average")
    return out


def split_costs(metrics: pd.DataFrame, splits: Splits) -> dict[str, list[float]]:
    """Seconds per split: train + infer time of every config in ``metrics`` (one row per dataset, fold, config)."""
    cost = (
        (metrics["time_train_s"].fillna(0) + metrics["time_infer_s"].fillna(0))
        .groupby([metrics["dataset"], metrics["fold"]])
        .sum()
    )
    return {d: [float(cost.get((d, s), 0.0)) for s in ss] for d, ss in splits.items()}


def split_noise(ranks: pd.DataFrame, splits: Splits) -> dict[str, float]:
    """Per-dataset split-to-split rank noise: the std over splits of each method's rank / number of methods,
    averaged over methods. Datasets with a single split take the median over the others.
    """
    r = ranks.copy()
    r["rank_norm"] = r["rank"] / r.groupby(["dataset", "fold"])["rank"].transform("max")
    sigma = r.groupby(["dataset", "method"])["rank_norm"].std().groupby(level=0).mean()
    sigma = sigma.reindex(list(splits))
    return sigma.fillna(sigma.median()).to_dict()


def lite_allocation(splits: Splits) -> Allocation:
    """One split per dataset."""
    return dict.fromkeys(splits, 1)


def cost_weighted_path(
    start: Allocation,
    cost_s: Mapping[str, list[float]],
    sigma: Mapping[str, float],
    targets: Iterable[int],
) -> dict[int, Allocation]:
    """The greedy allocation at each total split count in ``targets``, starting from ``start``.

    A dataset never goes below its ``start`` count or above its available splits (``len(cost_s[d])``); one that starts
    with no split gets its first before any other split is added. Targets past the total number of splits return the
    full allocation.
    """
    n = dict(start)
    heap: list[tuple[float, str]] = []

    def push(d: str) -> None:
        if n[d] < len(cost_s[d]):
            gain = np.inf if n[d] == 0 else sigma[d] ** 2 / (n[d] * (n[d] + 1)) / max(cost_s[d][n[d]], 1e-9)
            heapq.heappush(heap, (-gain, d))

    for d in n:
        push(d)
    path, total = {}, sum(n.values())
    for target in sorted(targets):
        while total < target and heap:
            _, d = heapq.heappop(heap)
            n[d] += 1
            total += 1
            push(d)
        path[target] = dict(n)
    return path


def runtime_s(allocation: Allocation, cost_s: Mapping[str, list[float]]) -> float:
    """Total compute of an allocation: the costs of every dataset's first ``n_d`` splits."""
    return float(sum(sum(cost_s[d][:k]) for d, k in allocation.items()))


def bootstrap_disagreement(
    ranks: pd.DataFrame,
    splits: Splits,
    allocation: Allocation,
    *,
    pairs: int = 200,
    seed: int = 0,
) -> pd.Series:
    """Per-dataset disagreement of the methods' harmonic mean ranks between two independent draws of the dataset's
    allocated number of splits (with replacement, ``pairs`` draws), averaged over methods, in rank positions.

    Harmonic mean rank = 1 / mean(1 / rank). NaN for datasets with a single available split.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for d, g in ranks.groupby("dataset"):
        rr = (1.0 / g.pivot_table(index="fold", columns="method", values="rank")).reindex(splits[d])
        rr = rr.loc[:, rr.notna().any()].to_numpy()
        n = rr.shape[0]
        if n < 2:
            out[d] = np.nan
            continue
        k = allocation[d]
        acc = 0.0
        for _ in range(pairs):
            with np.errstate(invalid="ignore"):
                h1 = 1.0 / np.nanmean(rr[rng.integers(0, n, size=k)], axis=0)
                h2 = 1.0 / np.nanmean(rr[rng.integers(0, n, size=k)], axis=0)
            acc += float(np.nanmean(np.abs(h1 - h2)))
        out[d] = acc / pairs
    return pd.Series(out, name="disagreement")


def stability_rule_counts(
    ranks: pd.DataFrame,
    splits: Splits,
    *,
    target_reliability: float = 0.8,
    value: Literal["rank", "reciprocal_rank"] = "rank",
    correlation: Literal["spearman", "pearson"] = "spearman",
    cap: int = 100,
) -> Allocation:
    """Split counts of the fold-similarity stability rule (bencheval ``rank_datasets_by_fold_similarity``).

    rho = mean off-diagonal correlation between splits of the methods' per-split ``value``; the Spearman-Brown
    split count for ``target_reliability`` is ``ceil(tau (1 - rho) / (rho (1 - tau)))``, capped at the available
    splits. ``value="rank"`` with Spearman is the rule that picks BeyondArena's committed core at tau 0.8;
    ``value="reciprocal_rank"`` with Pearson weights agreement towards the top of each split's ranking.
    """
    r = ranks.copy()
    r["reciprocal_rank"] = 1.0 / r["rank"]
    tau = float(target_reliability)
    out = {}
    for d, g in r.groupby("dataset"):
        m = g.pivot_table(index="fold", columns="method", values=value)
        if m.shape[0] < 2:
            out[d] = len(splits[d])
            continue
        sim = m.T.corr(method=correlation).to_numpy(copy=True)
        np.fill_diagonal(sim, np.nan)
        rho = min(float(np.nanmean(sim)), 1.0)
        k = cap if rho <= 0 else max(1, min(cap, int(np.ceil(tau * (1 - rho) / (rho * (1 - tau))))))
        out[d] = min(k, len(splits[d]))
    return {d: out.get(d, len(ss)) for d, ss in splits.items()}


# Worker state for the parallel draws; set before the pool forks so every worker inherits it.
_WORKER: dict[str, Any] = {}


def _init_worker() -> None:
    # One thread per worker: with a process per core, multi-threaded BLAS in every worker oversubscribes the
    # machine and makes the draws about 30 times slower.
    from threadpoolctl import threadpool_limits

    _WORKER["thread_limit"] = threadpool_limits(limits=1)


def _one_draw(job: tuple[str, int]) -> tuple[str, int, float]:
    name, seed = job
    ctx, splits, lb_full = _WORKER["ctx"], _WORKER["splits"], _WORKER["lb_full"]
    allocation = _WORKER["allocations"][name]
    rng = np.random.default_rng(seed)
    tasks = [(d, int(s)) for d, ss in splits.items() for s in rng.choice(ss, size=allocation[d], replace=False)]
    with contextlib.redirect_stdout(io.StringIO()):  # compare prints its leaderboard
        lb = ctx.compare(output_dir=None, tasks=tasks, plot=False)
    lb = (lb.reset_index() if not isinstance(lb.index, pd.RangeIndex) else lb).set_index("method")
    common = lb_full.index.intersection(lb.index)
    return name, seed, float((lb_full.loc[common, "elo"] - lb.loc[common, "elo"]).abs().mean())


def elo_error_vs_full(
    ctx: AbstractArenaContext,
    splits: Splits,
    allocations: Mapping[str, Allocation],
    lb_full: pd.DataFrame,
    *,
    seeds: Iterable[int] = range(32),
    workers: int = 8,
) -> dict[str, list[float]]:
    """For each allocation, the mean absolute Elo difference between the leaderboard on a random draw of each
    dataset's allocated number of splits and ``lb_full`` (the leaderboard on every split, indexed by method), one
    value per seed. Draws run in ``workers`` forked processes; ``workers=1`` runs them in this process.
    """
    _WORKER.update(ctx=ctx, splits=splits, lb_full=lb_full, allocations=dict(allocations))
    jobs = [(name, s) for name in allocations for s in seeds]
    out: dict[str, list[float]] = {name: [] for name in allocations}
    if workers <= 1:
        results = map(_one_draw, jobs)
        for name, _, err in results:
            out[name].append(err)
        return out
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("fork"), initializer=_init_worker
    ) as pool:
        for name, _, err in pool.map(_one_draw, jobs, chunksize=4):
            out[name].append(err)
    return out
