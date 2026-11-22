"""Scan every ``results.pkl`` of a submitted run with the restricted unpickler; write one row of checks per file.

Usage (run with ``python -I`` from a directory that is not inside the extracted submission)::

    python -I scan_submission.py <data_dir> <out.parquet> [--procs 64]
    python -I scan_submission.py <our_run_config_dir> <ref.parquet> --reference

``<data_dir>`` is the run's ``data/`` folder (``<config>/<task_id>/<repeat>_<fold>/results.pkl``); several
methods side by side are fine. ``--reference`` hashes the split indices and labels of one of our own runs (pass
the folder of a single config with all 816 splits, e.g. ``.../data/TA-TabDPT-1.3_c1_default_BAG_L1``) so the
summary can check that the submission used the official splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from safe_load import Refused, safe_load

EPS = 1e-15


def metric_error(metric: str, y: np.ndarray, p: np.ndarray) -> float:
    """TabArena's error for ``roc_auc`` (binary, 1-D positive-class proba), ``log_loss`` and ``rmse``."""
    from sklearn.metrics import log_loss, roc_auc_score

    if metric == "rmse":
        return float(np.sqrt(np.mean((y - p) ** 2)))
    if metric == "roc_auc":
        return float(1.0 - roc_auc_score(y, p))
    if metric == "log_loss":
        p = np.clip(p, EPS, 1 - EPS)
        p = p / p.sum(axis=1, keepdims=True)
        return float(log_loss(y, p, labels=list(range(p.shape[1]))))
    raise ValueError(metric)


def constant_error(metric: str, y_fit: np.ndarray, y: np.ndarray, n_classes: int | None) -> float:
    """Error on ``y`` of a constant predictor fit on ``y_fit`` (mean, class prior, or AUC 0.5)."""
    if metric == "rmse":
        return float(np.sqrt(np.mean((y - y_fit.mean()) ** 2)))
    if metric == "roc_auc":
        return 0.5
    counts = np.bincount(y_fit.astype(int), minlength=n_classes).astype(float) + 1.0
    return metric_error("log_loss", y.astype(int), np.tile(counts / counts.sum(), (len(y), 1)))


def _hash(a) -> str:
    return hashlib.blake2b(np.ascontiguousarray(np.asarray(a)).tobytes(), digest_size=8).hexdigest()


def _split_hashes(sa: dict) -> dict:
    return {
        "y_test_hash": _hash(sa["y_test"]),
        "y_test_idx_hash": _hash(sa["y_test_idx"]),
        "y_val_hash": _hash(sa["y_val"]),
        "y_val_idx_hash": _hash(sa["y_val_idx"]),
    }


def scan_one(path: str) -> dict:
    """All per-file checks for one result; errors are recorded, never raised."""
    p = Path(path)
    row = {"path": path, "dir_config": p.parts[-4], "dir_tid": p.parts[-3], "dir_split": p.parts[-2]}
    try:
        r = safe_load(path)
    except Refused as e:
        return {**row, "refused": str(e)}
    except Exception as e:
        return {**row, "load_error": repr(e)[:300]}
    tm, sa, mm = r["task_metadata"], r["simulation_artifacts"], r["method_metadata"]
    fw, metric = r["framework"], r["metric"]
    info = mm["info"]
    bi = info.get("bagged_info") or {}
    hp = dict(bi.get("child_hyperparameters") or info.get("hyperparameters") or {})
    n_classes = sa.get("num_classes")
    y_val, y_test = np.asarray(sa["y_val"]), np.asarray(sa["y_test"])
    pv = np.asarray(sa["pred_proba_dict_val"][fw], dtype=np.float64)
    pt = np.asarray(sa["pred_proba_dict_test"][fw], dtype=np.float64)
    row.update(
        framework=fw,
        tid=tm["tid"],
        dataset=tm["name"],
        repeat=tm["repeat"],
        fold=tm["fold"],
        problem_type=r["problem_type"],
        metric=metric,
        metric_error=float(r["metric_error"]),
        val_err=float(1 - info["val_score"] if metric == "roc_auc" else -info["val_score"]),
        time_train_s=float(r["time_train_s"]),
        time_infer_s=float(r["time_infer_s"]),
        n_children=bi.get("num_child_models"),
        max_time_limit=mm.get("model_hyperparameters", {}).get("ag_args_ensemble", {}).get("ag.max_time_limit"),
        num_cpus=mm.get("fit_kwargs_extra", {}).get("num_cpus"),
        num_gpus=mm.get("fit_kwargs_extra", {}).get("num_gpus"),
        hp_json=json.dumps(
            {k: v if isinstance(v, (str, int, float, bool)) or v is None else str(v) for k, v in sorted(hp.items())}
        ),
        finite=bool(np.isfinite(pv).all() and np.isfinite(pt).all()),
        test_n_unique=len(np.unique(np.round(pt, 7))),
        proba_out_of_range=bool(r["problem_type"] != "regression" and (pt.min() < 0 or pt.max() > 1)),
        rowsum_dev=float(np.abs(pt.sum(axis=1) - 1).max()) if pt.ndim == 2 else 0.0,
        **_split_hashes(sa),
    )
    try:
        row["recomp_test_err"] = metric_error(metric, y_test, pt)
        row["recomp_val_err"] = metric_error(metric, y_val, pv)
        row["const_test_err"] = constant_error(metric, y_val, y_test, n_classes)
    except Exception as e:
        row["recomp_error"] = repr(e)[:300]
    # Bag children no better than a constant predictor on their own holdout rows (hidden failed fits).
    try:
        ch, names = info["children_info"], bi["child_model_names"]
        n_bad = 0
        for name, idx in zip(names, sa["bag_info"]["val_idx_per_child"], strict=True):
            vs = float(ch[name]["val_score"])
            child_err = 1 - vs if metric == "roc_auc" else -vs
            base = constant_error(metric, np.delete(y_val, idx), y_val[idx], n_classes)
            n_bad += base > 0 and child_err / base >= 0.98
        row["n_children_const_like"] = int(n_bad)
    except Exception as e:
        row["child_error"] = repr(e)[:300]
    return row


def reference_one(path: str) -> dict:
    """Split hashes of one of our own (trusted) results."""
    r = safe_load(path)
    tm = r["task_metadata"]
    return {"tid": tm["tid"], "repeat": tm["repeat"], "fold": tm["fold"], **_split_hashes(r["simulation_artifacts"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=64)
    ap.add_argument("--reference", action="store_true", help="hash split indices/labels of our own run")
    args = ap.parse_args()
    paths = sorted(str(p) for p in args.root.rglob("results.pkl"))
    print(f"{len(paths)} files", flush=True)
    with Pool(args.procs) as pool:
        rows = pool.map(reference_one if args.reference else scan_one, paths, chunksize=64)
    pd.DataFrame(rows).to_parquet(args.out)
    print("written", args.out)


if __name__ == "__main__":
    main()
