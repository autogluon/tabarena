from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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


def run_eval_for_new_models(
    models: list[ModelMetadata],
    *,
    fig_output_dir: Path,
    extra_subsets: None | list[list[str]] = None,
    cache_path: str | None = None,
) -> None:
    """Run evaluation for a custom model on TabArena.

    Args:
        models: List of ModelMetadata instances for each custom model to be evaluated.
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
    from bencheval.website_format import format_leaderboard

    for model in models:
        if not model.only_load_cache:
            EndToEndSingle.from_path_raw_to_results(
                path_raw=model.path_raw / "data",
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


if __name__ == "__main__":
    fig_dir = Path(__file__).parent / "evals"
    out_dir = Path("/work/dlclarge2/purucker-tabarena/output")

    run_eval_for_new_models(
        [
            ModelMetadata(
                path_raw=out_dir / "tabpfnv25_hpo_14112025",
                method="RealTabPFN-v2.5",
            ),
        ],
        extra_subsets=[["tabpfn"]],
        fig_output_dir=fig_dir / "tabpfnv25_submission",
        cache_path="/work/dlclarge2/purucker-tabarena/output/tabarena_cache",
    )
