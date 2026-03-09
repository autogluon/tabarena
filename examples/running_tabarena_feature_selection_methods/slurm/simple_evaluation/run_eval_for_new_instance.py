from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabarena.benchmark.experiment import ExperimentBatchRunner
from tabarena.benchmark.result import ExperimentResults
from tabarena.nips2025_utils.fetch_metadata import load_task_metadata


@dataclass
class ModelMetadata:
    """Metadata related to the result artifacts for a custom model to be evaluated on TabArena."""

    path_raw: Path
    """Path to the directory containing raw results from the custom model.
    If None, defaults to a predefined path."""
    method: str
    """Name of the custom method to be evaluated. This is the `ag_name` key in the method class."""
    new_result_prefix: str | None = None
    """Optional prefix for the new results. If None, defaults to the method name."""
    only_load_cache: bool = False
    """If False, the results will be computed and cached. If True, only loads the cache."""


@dataclass
class FsMethodMetadata:
    """Metadata related to the result artifacts for a custom model to be evaluated on TabArena."""

    path_raw: Path
    """Path to the directory containing raw results from the custom model.
    If None, defaults to a predefined path."""
    method: str
    """Name of the custom method to be evaluated. This is the `ag_name` key in the method class."""
    new_result_prefix: str | None = None
    """Optional prefix for the new results. If None, defaults to the method name."""
    only_load_cache: bool = False
    """If False, the results will be computed and cached. If True, only loads the cache."""


def run_eval_for_new_models(
    models: list[ModelMetadata],
    *,
    fs_methods: list[FsMethodMetadata],
    fig_output_dir: Path,
    extra_subsets: None | list[list[str]] = None,
    cache_path: str | None = None,
) -> None:
    """Run evaluation for a custom model on TabArena.

    Args:
        models: List of ModelMetadata instances for each custom model to be evaluated.
        fs_methods: List of FsMethodMetadata instances for each custom feature selection method to be evaluated.
        fig_output_dir: Path to the directory where evaluation artifacts will be saved.
        extra_subsets: list of optional subsets of the TabArena dataset to evaluate on.
            Each element is a subset description as a list of strings.
        cache_path: Optional path to the cache directory on the filesystem.

    """
    if cache_path is not None:
        os.environ["TABARENA_CACHE"] = cache_path
        print("Set cache to:", os.getenv("TABARENA_CACHE"))

    # Import here such that env var above is used correctly
    from tabarena.nips2025_utils.end_to_end import EndToEndResults
    from tabarena.nips2025_utils.end_to_end_single import EndToEndSingle
    from tabarena.tabarena.website_format import format_leaderboard

    for model in models:
        if not model.only_load_cache:
            EndToEndSingle.from_path_raw_to_results(
                path_raw=model.path_raw,
                name_suffix=model.new_result_prefix,
                artifact_name=model.new_result_prefix,
                num_cpus=8,
            )

    end_to_end_results = EndToEndResults.from_cache(
        # TODO: check if "+ model.new_result_prefix" is correct here
        # methods=[(m.method + m.new_result_prefix, m.new_result_prefix) for m in models]
        methods=[m.method for m in models]
    )

    def plot_plots(_fig_output_dir, _subset=None):
        leaderboard = end_to_end_results.compare_on_tabarena(
            output_dir=_fig_output_dir,
            subset=_subset,
            tabarena_context_kwargs=dict(include_unverified=True),
        )
        leaderboard_website = format_leaderboard(df_leaderboard=leaderboard)
        print(leaderboard_website.to_markdown(index=False))

    plot_plots(fig_output_dir)
    if extra_subsets is not None:
        for subset in extra_subsets:
            print("\n\n###############")
            print("\t Subset Description:", subset)
            plot_plots(fig_output_dir / "subsets" / "_".join(subset), subset)


"""
if __name__ == "__main__":
    fig_dir = Path(__file__).parent / "evals"
    out_dir = Path("../../results/data")

    run_eval_for_new_models(
        models=[
            ModelMetadata(
                path_raw=out_dir / "LightGBM_c1_BAG_L1_Reproduced-v1",
                method="LightGBM_c1_BAG_L1_Reproduced-v1",
            ),
        ],
        fs_methods=[
            FsMethodMetadata(
                path_raw=out_dir / "Boruta",
                method="Boruta",
            ),
        ],
        extra_subsets=[["boruta"]],
        fig_output_dir=fig_dir / "boruta",
        cache_path=None
    )
 """


if __name__ == "__main__":
    fig_dir = Path(__file__).parent / "evals"
    out_dir = Path("../../results/data")
    repo_dir = out_dir / "processed_repo"  # Add this

    # First, convert raw results to EvaluationRepository and save with context.json
    from tabarena.repository.evaluation_repository import EvaluationRepository
    from tabarena.nips2025_utils.end_to_end_single import EndToEndSingle

    # Process raw results
    for model in [
        ModelMetadata(
            path_raw=out_dir / "LightGBM_c1_BAG_L1_Reproduced-v1",
            method="LightGBM_c1_BAG_L1_Reproduced-v1",
        ),
    ]:
        if not model.only_load_cache:
            EndToEndSingle.from_path_raw_to_results(
                path_raw=model.path_raw / "data",
                name_suffix=model.new_result_prefix,
                artifact_name=model.new_result_prefix,
                num_cpus=8,
            )

    # Create EvaluationRepository and save to disk with context.json
    # You'll need to load your results into an EvaluationRepository first
    # Then call to_dir() which generates context.json
    task_metadata = load_task_metadata()
    exp_batch_runner = ExperimentBatchRunner(expname="Lol", task_metadata=task_metadata)

    experiment_results = ExperimentResults(task_metadata=task_metadata)
    datasets = ["anneal", "credit-g", "diabetes"]  # datasets = list(task_metadata["name"])
    folds = [0]
    results_lst: list[dict[str, Any]] = exp_batch_runner.run(
        datasets=datasets,
        folds=folds,
        methods=model,
        ignore_cache=True,
    )

    repo: EvaluationRepository = experiment_results.repo_from_results(results_lst=results_lst)
    repo.print_info()

    repo.to_dir(path=repo_dir)
    # Now run evaluation pointing to the processed repo
    run_eval_for_new_models(
        models=[
            ModelMetadata(
                path_raw=repo_dir,  # Use the processed repo instead
                method="LightGBM_c1_BAG_L1_Reproduced-v1",
            ),
        ],
        fs_methods=[
            FsMethodMetadata(
                path_raw=out_dir / "Boruta",
                method="Boruta",
            ),
        ],
        extra_subsets=[["boruta"]],
        fig_output_dir=fig_dir / "boruta",
        cache_path=None
    )