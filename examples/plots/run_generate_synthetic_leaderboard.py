"""
TabArena quickstart tutorial (synthetic data)

This script:
1) Creates synthetic long-form results with columns:
   method, task, seed, metric_error, time_train_s, time_infer_s
2) Runs TabArena.leaderboard() with average_seeds=True and False
3) Computes and (optionally) plots a win-rate matrix

Notes:
- TabArena expects DENSE results: every (task, seed, method) combination must exist.
- metric_error must be numeric and >= 0.
- By default, TabArena expects "metric_error" where LOWER is better.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bencheval.tabarena import TabArena


def make_synthetic_results(
    *,
    methods: list[str],
    tasks: list[str],
    seeds: list[int],
    rng_seed: int = 0,
) -> pd.DataFrame:
    """
    Create a dense (task, seed, method) results table with realistic-ish structure:
    - Each task has its own difficulty scale.
    - Each method has a global skill offset.
    - Each seed adds small noise.
    - Includes time_train_s and time_infer_s.

    Returns a long-form DataFrame suitable for TabArena.
    """
    rng = np.random.default_rng(rng_seed)

    # Lower is better: "error"
    method_skill = {
        # Think: A is best, D is worst
        methods[0]: 0.87,
        methods[1]: 0.91,
        methods[2]: 0.92,
        methods[3]: 0.94,
    }

    # Task difficulty multipliers (some tasks are harder => higher error)
    task_difficulty = {t: float(rng.uniform(0.3, 2.5)) for t in tasks}

    rows: list[dict] = []
    for task in tasks:
        for seed in seeds:
            # A bit of per-(task, seed) randomness shared across methods
            shared_noise = float(rng.normal(0.0, 0.02))
            for method in methods:
                base = 0.20  # baseline error floor
                error = base * task_difficulty[task] * method_skill[method]
                error *= float(1.0 + shared_noise + rng.normal(0.0, 0.06))
                error = max(0.0, error)  # must be >= 0 per TabArena.verify_error

                # times: better methods might be slower (arbitrary example)
                time_train_s = float(
                    rng.lognormal(mean=2.5, sigma=0.4) * (1.0 + 0.30 * (methods.index(method)))
                )
                time_infer_s = float(
                    rng.lognormal(mean=0.2, sigma=0.25) * (1.0 + 0.15 * (methods.index(method)))
                )

                rows.append(
                    {
                        "method": method,
                        "task": task,
                        "seed": seed,
                        "metric_error": error,
                        "time_train_s": time_train_s,
                        "time_infer_s": time_infer_s,
                    }
                )

    df = pd.DataFrame(rows)

    # Optional: enforce nice ordering
    df = df.sort_values(["task", "seed", "method"]).reset_index(drop=True)
    return df


def main() -> None:
    # ----------------------------
    # 1) Create synthetic results
    # ----------------------------
    methods = ["ModelA", "ModelB", "ModelC", "ModelD"]
    tasks = [f"task_{i:02d}" for i in range(1, 21)]  # 20 tasks
    seeds = [0, 1, 2]  # 3 repeats per task

    data = make_synthetic_results(methods=methods, tasks=tasks, seeds=seeds, rng_seed=0)

    print("Synthetic results (head):")
    print(data.head(8).to_string(index=False))
    print("\nRow count:", len(data), "(should be tasks * seeds * methods =", len(tasks) * len(seeds) * len(methods), ")")

    # ----------------------------
    # 2) Initialize TabArena
    # ----------------------------
    # Provide seed_column to enable average_seeds=True/False behavior.
    # Keep defaults for:
    #   method_col="method", task_col="task", error_col="metric_error"
    arena = TabArena(seed_column="seed")

    # ----------------------------
    # 3) Compute leaderboard
    # ----------------------------
    leaderboard_avg = arena.leaderboard(
        data=data,
        average_seeds=False,
        include_error=True,
        include_elo=True,
        include_winrate=True,
        include_improvability=True,
        include_mrr=True,
        # If you want Elo anchored to a reference method:
        elo_kwargs=dict(calibration_framework="ModelB", calibration_elo=1000),
        sort_by=["rank"],
    )

    print("\n=== Leaderboard ===")
    print(leaderboard_avg.to_string())

    # ----------------------------
    # 5) (Optional) Win-rate matrix
    # ----------------------------
    # If you want the matrix, TabArena expects results_per_task (the internal per-task df).
    # For a quick demo, we can compute it the same way TabArena does:
    results_per_task = arena.compute_results_per_task(data=data, include_seed_col=True)  # (method, task, seed) rows
    winrate_matrix = arena.compute_winrate_matrix(results_per_task=results_per_task)

    print("\n=== Pairwise win-rate matrix (results_per_task) ===")
    # Pretty-print as percentages
    print((winrate_matrix * 100).round(1).to_string())

    # Plotting requires plotly + kaleido if saving images:
    #   pip install plotly kaleido
    # fig = arena.plot_winrate_matrix(winrate_matrix=winrate_matrix, save_path=None)
    # fig.show()


if __name__ == "__main__":
    main()
