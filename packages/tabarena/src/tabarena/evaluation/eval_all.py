"""Evaluate every cell of the leaderboard subset grid, one figure/table set per cell.

The grid itself (and the folder layout it writes into) lives in
:mod:`tabarena.evaluation.subset_grid`; who competes in each cell lives in
:mod:`tabarena.evaluation.entrants`.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import TYPE_CHECKING

from tabarena.evaluation.entrants import filter_results_to_pool, get_entrant_pool
from tabarena.evaluation.leaderboard_reporter import LeaderboardReporter
from tabarena.evaluation.subset_grid import get_all_subset_combinations, get_website_folder_name
from tabarena.utils.parallel_for import parallel_for

if TYPE_CHECKING:
    import pandas as pd


def evaluate_all(
    tabarena_context,
    df_results: pd.DataFrame,
    eval_save_path: str | Path,
    elo_bootstrap_rounds: int = 200,
    use_latex: bool = False,
    use_website_folder_names: bool = False,
    evaluator_kwargs: dict | None = None,
    engine: str = "auto",
    progress_bar: bool = True,
    website_only: bool = False,
):
    if evaluator_kwargs is None:
        evaluator_kwargs = {}

    # Only the baselines (KNN/LR) are banned from the Pareto figures — they sit
    # far above every real method and stretch the improvability axis. Weak
    # non-baseline methods stay, greyed out by the focus styling.
    evaluator_kwargs_ = {
        "use_latex": use_latex,
        "banned_pareto_methods": ["KNN", "LR"],
    }
    evaluator_kwargs_.update(evaluator_kwargs)
    evaluator_kwargs = evaluator_kwargs_

    eval_save_path = Path(eval_save_path)

    if engine == "auto":
        engine = tabarena_context.engine

    df_results = df_results.copy(deep=True)
    if "imputed" not in df_results.columns:
        df_results["imputed"] = False
    df_results["imputed"] = df_results["imputed"].fillna(0)

    method_metadata_info = tabarena_context.method_metadata_collection.info()

    all_combinations = get_all_subset_combinations()

    # One job per sub-benchmark subset combination. Build the per-combination kwargs up
    # front so the heavy, read-only inputs (`tabarena_context`, `df_results`, ...) can be
    # shared once via `parallel_for`'s `context` (ray's object store) instead of being
    # serialized per job. Each `evaluate_single` writes its own figures/tables to disk.
    inputs = []
    for (
        entrant_pool,
        use_imputation,
        problem_type,
        with_baselines,
        dataset_subset,
        lite,
        average_seeds,
    ) in all_combinations:
        custom_folder_name = None
        if use_website_folder_names:
            custom_folder_name = str(
                get_website_folder_name(
                    entrant_pool=entrant_pool,
                    use_imputation=use_imputation,
                    problem_type=problem_type,
                    dataset_subset=dataset_subset,
                    lite=lite,
                )
            )
        inputs.append(
            {
                "entrant_pool": entrant_pool,
                "use_imputation": use_imputation,
                "problem_type": problem_type,
                "with_baselines": with_baselines,
                "dataset_subset": dataset_subset,
                "lite": lite,
                "average_seeds": average_seeds,
                "custom_folder_name": custom_folder_name,
            }
        )

    parallel_for(
        f=evaluate_single,
        inputs=inputs,
        context={
            "tabarena_context": tabarena_context,
            "df_results": df_results,
            "method_metadata_info": method_metadata_info,
            "eval_save_path": eval_save_path,
            "evaluator_kwargs": evaluator_kwargs,
            "elo_bootstrap_rounds": elo_bootstrap_rounds,
            "website_only": website_only,
        },
        engine=engine,
        progress_bar=progress_bar,
        desc="Generating evaluation figures/tables per subset",
    )


#: Line colors for the systems a pool draws as horizontal references, cycled in collection
#: order. The first two land on AutoGluon 1.4 and 1.5, which is what those lines have always
#: been drawn in.
_REFERENCE_LINE_COLORS: list[str] = ["black", "tab:purple", "tab:blue", "tab:red", "darkgray"]


def get_pool_reference_lines(
    entrant_pool: str,
    method_metadata_info: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """The ``(baselines, baseline_colors)`` for one entrant pool: every system it admits.

    These are the horizontal reference lines the pool's figures draw. They are also, less
    obviously, the systems that reach the pool's leaderboard at all: ``LeaderboardReporter.eval``
    keeps a row only when its method maps to a config framework type *or* is named in
    ``baselines``, and a system is neither. So this has to be the whole admitted set. Naming a
    subset here does not just leave lines off a figure, it deletes those systems from the
    published numbers.

    The models-only pool admits none, and correctly gets no reference lines.
    """
    pool = get_entrant_pool(entrant_pool)
    names = list(
        dict.fromkeys(
            row.display_name
            for row in method_metadata_info.itertuples()
            if getattr(row, "method_class", "model") == "system"
            and pool.admits("system", getattr(row, "tags", ()) or ())
        )
    )
    colors = list(itertools.islice(itertools.cycle(_REFERENCE_LINE_COLORS), len(names)))
    return names, colors


def evaluate_single(
    tabarena_context,
    df_results,
    use_imputation,
    problem_type,
    with_baselines,
    dataset_subset,
    lite,
    average_seeds,
    eval_save_path,
    evaluator_kwargs,
    method_metadata_info: pd.DataFrame,
    entrant_pool: str = "systems_all",
    elo_bootstrap_rounds: int = 200,
    custom_folder_name: str | None = None,
    website_only: bool = False,
):
    from tabarena.nips2025_utils.compare import subset_tasks

    df_results = df_results.copy()

    method_rename_map = tabarena_context.get_method_rename_map()

    # Narrow the field first: Elo, improvability and the ranks are all relative to who
    # competes, so this has to happen before anything is computed.
    df_results = filter_results_to_pool(
        df_results=df_results,
        pool=get_entrant_pool(entrant_pool),
        method_metadata_info=method_metadata_info,
    )
    if len(df_results) == 0:
        print("\tNo results left in this entrant pool, skipping...")
        return
    baselines, baseline_colors = get_pool_reference_lines(entrant_pool, method_metadata_info)

    subset = []
    folder_name = "all"
    if problem_type is not None:
        folder_name = f"{problem_type}"
        if problem_type == "all":
            pass
        else:
            subset.append(problem_type)
    if dataset_subset:
        folder_name_prefix = dataset_subset
        subset.append(dataset_subset)
    else:
        folder_name_prefix = "all"
    if lite:
        subset.append("lite")

    if subset:
        df_results = subset_tasks(
            df_results=df_results,
            subset=subset,
            predicates=tabarena_context.subset_predicates,
        )

    if len(df_results) == 0:
        print("\tNo results after filtering, skipping...")
        return

    folder_name = str(Path(folder_name_prefix) / folder_name)
    if use_imputation:
        folder_name = folder_name + "-imputed"
    if not with_baselines:
        baselines = []
        baseline_colors = []
        folder_name = folder_name + "-nobaselines"

    imputed_freq = df_results.groupby(by=["ta_name", "ta_suite"])["imputed"].transform("mean")
    if not use_imputation:
        df_results = df_results.loc[imputed_freq <= 0]
    else:
        df_results = df_results.loc[imputed_freq < 1]  # always filter out methods that are imputed 100% of the time

    if len(df_results) == 0:
        print("\tNo results after filtering, skipping...")
        return

    if lite:
        folder_name = str(Path("lite") / folder_name)
    if not average_seeds:
        folder_name = str(Path("no_average_seeds") / folder_name)
    # The pool is part of the cell's identity, so it has to be part of the path here too
    # (the website layout carries it as its own `entrants_*` segment).
    folder_name = str(Path(f"entrants_{entrant_pool}") / folder_name)

    if custom_folder_name is not None:
        folder_name = custom_folder_name

    plotter = LeaderboardReporter(
        output_dir=eval_save_path / folder_name,
        task_metadata=tabarena_context.task_metadata_collection,
        elo_bootstrap_rounds=elo_bootstrap_rounds,
        tabarena_context=tabarena_context,
        method_rename_map=method_rename_map,
        **evaluator_kwargs,
    )

    eval_kwargs = {}
    if baselines is not None:
        eval_kwargs["baselines"] = baselines
    if baseline_colors is not None:
        eval_kwargs["baseline_colors"] = baseline_colors

    leaderboard = plotter.eval(
        df_results=df_results,
        plot_extra_barplots=False,
        # The website ships neither the normalized-score bar plots nor the
        # per-method time plots — skip them in website_only mode.
        include_norm_score=not website_only,
        plot_times=not website_only,
        website_only=website_only,
        average_seeds=average_seeds,
        # Draw the systems as points in the Pareto scatter and as rows in the win-rate matrix,
        # not only as horizontal reference lines. This used to be False, which is what produced
        # the inconsistency the entrant pools exist to remove: a system sat in the Elo pool and
        # so moved every other number, while the figures pretended it was not there. A pool that
        # admits no system has no baseline row, so this is a no-op for models-only.
        plot_with_baselines=True,
        **eval_kwargs,
    )

    leaderboard_website_verified = tabarena_context.leaderboard_to_website_format(
        leaderboard=leaderboard,
        include_type=True,
        include_url=True,
    )
    path_website_lb = plotter.output_dir / "website_leaderboard.csv"
    leaderboard_website_verified.to_csv(path_website_lb, index=False)
