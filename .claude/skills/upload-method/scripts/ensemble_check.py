"""Approximate "tuned + ensemble" from a submission's stored predictions, per method and split.

Greedy Caruana selection with replacement over all configs of a method (40 iterations on the out-of-fold
predictions, as TabArena's HPO simulation does), then the weighted test predictions are scored. TabArena's own
implementation differs in details, so expect a few Elo points of difference from the processed artifact; the
check is there to test a claimed tuned + ensembled number before processing, or to ablate configs.

Usage::

    python -I ensemble_check.py <data_dir> <out.parquet> [--exclude cfg_dir_name ...] [--procs 64]

Writes ``dataset, fold, method, metric_error, metric_error_val`` rows (method ``<METHOD> (tuned + ensemble)``)
that can be appended to ``summarize_submission.py --new-results`` for ``compare``.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from safe_load import safe_load
from scan_submission import metric_error

N_ITER = 40


def select(job: tuple[str, list[str]]) -> dict:
    method, paths = job
    pv, pt = [], []
    for p in paths:
        r = safe_load(p)
        sa, fw = r["simulation_artifacts"], r["framework"]
        pv.append(np.asarray(sa["pred_proba_dict_val"][fw], dtype=np.float64))
        pt.append(np.asarray(sa["pred_proba_dict_test"][fw], dtype=np.float64))
    metric, tm = r["metric"], r["task_metadata"]
    yv, yt = np.asarray(sa["y_val"]), np.asarray(sa["y_test"])
    pv, pt = np.stack(pv), np.stack(pt)
    counts, cur = np.zeros(len(paths)), np.zeros_like(pv[0])
    for it in range(N_ITER):
        j = int(np.argmin([metric_error(metric, yv, (cur * it + pv[k]) / (it + 1)) for k in range(len(paths))]))
        counts[j] += 1
        cur = (cur * it + pv[j]) / (it + 1)
    test = np.tensordot(counts / counts.sum(), pt, axes=1)
    return {
        "dataset": tm["name"],
        "fold": tm["repeat"] * 3 + tm["fold"],
        "method": f"{method.upper()} (tuned + ensemble)",
        "metric_error": metric_error(metric, yt, test),
        "metric_error_val": metric_error(metric, yv, cur),
        "metric": metric,
        "problem_type": r["problem_type"],
        "n_configs": len(paths),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("out")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--procs", type=int, default=64)
    args = ap.parse_args()
    groups = defaultdict(list)
    for p in sorted(args.root.rglob("results.pkl")):
        config, tid, split = p.parts[-4], p.parts[-3], p.parts[-2]
        if config in args.exclude:
            continue
        groups[(re.sub(r"_(c|r)\d+_.*$", "", config), tid, split)].append(str(p))
    jobs = [(k[0], v) for k, v in groups.items()]
    print(len(jobs), "method x split jobs", flush=True)
    with Pool(args.procs) as pool:
        pd.DataFrame(pool.map(select, jobs, chunksize=1)).to_parquet(args.out)
    print("written", args.out)


if __name__ == "__main__":
    main()
