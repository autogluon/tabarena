"""Run the Chakra-Tab system on TabArena-Lite (or BeyondArena) through the official pipeline and score it on the leaderboard.

The endpoint is read from ``CHAKRA_TAB_URL`` and the key from ``CHAKRA_TAB_KEY``. Results are checkpointed per task to
``<output-dir>/raw_results.pkl`` and a rerun resumes from it; ``--score-only`` registers and scores the checkpoint as is.

    CHAKRA_TAB_KEY=... python examples/advanced/run_chakra_tab_tabarena_lite.py --preset medium --subset lite --output-dir out/lite_medium
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

from tabarena.benchmark.experiment import BeyondArenaExperimentBundle, TabArenaV0pt1ExperimentBundle
from tabarena.contexts import BeyondArenaContext, TabArenaContext
from tabarena.systems.chakra_tab import ChakraTabSystemModel
from tabarena.utils.config_utils import SystemConfigGenerator


def _dataset(r: dict) -> str | None:
    meta = r.get("task_metadata") or {}
    return meta.get("name") or meta.get("dataset") or r.get("dataset")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", nargs="+", default=["lite"])
    ap.add_argument("--preset", default="medium", choices=["medium", "full"])
    ap.add_argument("--arena", default="tabarena", choices=["tabarena", "beyondarena"])
    ap.add_argument("--time-limit", type=int, default=3600, help="fit budget in seconds, forwarded to the API")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()

    context = TabArenaContext() if a.arena == "tabarena" else BeyondArenaContext()
    bundle = TabArenaV0pt1ExperimentBundle if a.arena == "tabarena" else BeyondArenaExperimentBundle
    generator = SystemConfigGenerator(
        model_cls=ChakraTabSystemModel, name="Chakra-Tab", manual_configs=[{"preset": a.preset}]
    )
    experiments = bundle(models=[(generator, 0)], system_experiments=True).build_experiments(time_limit=a.time_limit)
    jobs = context.build_jobs(experiments, subset=list(a.subset), pre_materialize=False)

    os.makedirs(a.output_dir, exist_ok=True)
    ckpt = os.path.join(a.output_dir, "raw_results.pkl")
    raw = []
    if os.path.exists(ckpt):
        with open(ckpt, "rb") as f:
            raw = pickle.load(f)
    done = {_dataset(r) for r in raw}
    jobs = [] if a.score_only else [j for j in jobs if j.task.dataset not in done]
    print(f"{len(jobs)} jobs to run ({len(done)} in checkpoint)", flush=True)

    t0 = time.time()
    for i, job in enumerate(jobs):
        t = time.time()
        try:
            res = context.run_job(job, expname=None, register=False, debug_mode=True)
        except Exception as e:
            print(f"  {job.task.dataset} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        raw.extend(res)
        with open(ckpt, "wb") as f:
            pickle.dump(raw, f)
        err = res[0].get("metric_error", float("nan")) if res else float("nan")
        print(
            f"  [{i + 1}/{len(jobs)}] {job.task.dataset}: metric_error {err:.4f} ({time.time() - t:.0f}s, total {time.time() - t0:.0f}s)",
            flush=True,
        )

    seen, unique = set(), []
    for r in raw:
        meta = r.get("task_metadata") or {}
        key = (_dataset(r), meta.get("fold"), meta.get("repeat"))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    context.register(unique, new_result_prefix="[New] ")
    row, per_split = context.compare(
        output_dir=a.output_dir, subset=list(a.subset), return_results=True, return_single=True, plot=False
    )
    per_split.to_csv(os.path.join(a.output_dir, "per_split.csv"), index=False)
    print("\n=== leaderboard row ===")
    print(row[["elo", "elo+", "elo-", "rank", "winrate", "improvability"]].to_string())


if __name__ == "__main__":
    main()
