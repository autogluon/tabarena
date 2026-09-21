"""Export the per-split results behind the TabArena leaderboard, without the figures.

The leaderboard aggregates one number per dataset, split and method into Elo, ranks and
win-rates. This script writes those numbers themselves. Every ``compare()`` call with an
``output_dir`` writes them as ``results_per_split.csv`` next to the leaderboard CSVs and the
figures (see ``run_generate_main_leaderboard.py``); this script passes ``plot=False``, which
writes the CSVs and skips the figures, so it finishes much faster.

``results_per_split.csv`` has one row per dataset, split (``fold``) and method with the test
error (``metric_error``), the validation error (``metric_error_val``), ``time_train_s``,
``time_infer_s``, the ``metric`` and ``problem_type``, the auxiliary metric columns, the
method's type, subtype and ``config_type``, and ``imputed`` (``True`` where a missing result
was filled in from the context's ``fillna_method``). Methods carry their leaderboard display
names, so a model's ``(default)``, ``(tuned)`` and ``(tuned + ensemble)`` variants are separate
rows.

The per-split results are not hosted anywhere; this script is how to get them. Results
download to ``~/.cache/tabarena/`` on first run. For BeyondArena, swap in
``BeyondArenaContext`` (also in ``tabarena.contexts``) and pass ``subset=["core"]``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tabarena.contexts import TabArenaContext

if __name__ == "__main__":
    output_dir = Path("output_results_per_split")

    context = TabArenaContext()
    # `plot=False`: write `results_per_split.csv`, `tabarena_leaderboard.csv` and the other CSVs,
    # render no figure.
    context.compare(output_dir=output_dir, plot=False)

    results_per_split = pd.read_csv(output_dir / "results_per_split.csv")
    print(f"Wrote {output_dir / 'results_per_split.csv'}")
    print(
        f"{len(results_per_split)} rows: {results_per_split['dataset'].nunique()} datasets, "
        f"{results_per_split['fold'].nunique()} splits, {results_per_split['method'].nunique()} methods",
    )
    print(results_per_split.head(10).to_markdown(index=False))
