"""Run the BeyondArena (data-foundry) benchmark on a GCP cluster node: `setup` + `eval` in one file.

ONE file, TWO subcommands, sharing the new run's ``BENCHMARK_NAME`` / ``PathSetup`` /
``CONTENDER_MODEL`` (defined once below)::

    python experiments/run_beyondarena.py setup   # launch the new contender run
    python experiments/run_beyondarena.py eval      # leaderboard: contender vs. the uploaded suite

`setup` launches a fresh run of a single contender (``CONTENDER_MODEL``). The tasks come from the
Data Foundry ``BeyondArena`` collection, which ``BeyondArenaContext`` owns: it loads reference
metadata (no downloads); the benchmark setup later materializes (downloads + converts) only the
surviving datasets on this head node. Scope a subset with ``task_subset=TaskSubset(subset=["core"])``
(``core`` is the recommended default protocol) or ``TaskSubset(dataset_names=[...])``.

`eval` builds the leaderboard the way the TabArena-v0.1 eval and the official BeyondArena leaderboard
do: the **baselines come from the context**, not from hand-wired output dirs. We post-process only the
contender's raw ``results.pkl`` (written by `setup`) and register it via ``extra_methods=`` on a
``BeyondArenaContext``; the context already knows every uploaded baseline
(``beyond_method_metadata_collection`` — the classic + neural + foundation suite, incl. TabPFN-3), so
there is nothing to list by hand. ``only_valid_tasks=True`` scopes every leaderboard to the exact
tasks the contender ran, and ``compare(subset=...)`` slices via BeyondArena's subset predicates.

Note: the illustrative ``CONTENDER_MODEL = "TabPFN-3"`` here is *itself* an uploaded baseline, so we
mark this run a re-run via ``RESULT_SUFFIX`` (rendered ``TabPFN-3 [rerun]``) to keep it distinct from
the cached ``TabPFN-3`` in the leaderboard. A genuinely new method needs no suffix.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tabarena.benchmark.experiment import BeyondArenaExperimentBundle
from tabarena.contexts import BeyondArenaContext
from tabflow_slurm import (
    BeyondArenaResourcesSetup,
    GCPSlurmSetup,
    ModelJob,
    PathSetup,
    TabArenaBenchmarkPlan,
)

# ── Shared identity — the ONE place these live; setup + eval both read them ──
BENCHMARK_NAME = "example_beyondarena_31052026"  # this contender run's cache name (also the cache suite)
WORKSPACE = "/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace"
PYTHON_PATH = "/home/lennart_priorlabs_ai/.venvs/beyondarena_27052026/bin/python"
CONTENDER_MODEL = "TabPFN-3"  # the model this run adds; baselines come from the context
RESULT_SUFFIX = " [rerun]"  # keeps this run distinct from the uploaded TabPFN-3 baseline; None for a new method

# Subset slices to evaluate, anchored on the recommended ``core`` protocol (each dataset's first
# ``folds_to_use`` splits — no need for the full ``["all"]`` split set). ``only_valid_tasks`` already
# restricts the leaderboard to the contender's tasks, so these just carve split-regime / problem-type
# / size / feature slices within it.
SUBSETS = [
    ["core"],
    ["core", "random"],
    ["core", "temporal"],
    ["core", "grouped"],
    ["core", "tiny"],
    ["core", "small"],
    ["core", "medium"],
    ["core", "large"],
    ["core", "low-dim"],
    ["core", "high-dim"],
    ["core", "text"],
    ["core", "high-cardinality"],
]

# Figure format(s) written per subset. PNGs render inline; PDFs are for papers. Each extra format
# re-runs `compare` purely to re-emit the same figures — the leaderboard itself is identical.
FIGURE_FILE_TYPES = ("pdf", "png")


def _path_setup() -> PathSetup:
    return PathSetup(workspace=WORKSPACE, python_path=PYTHON_PATH)


def setup() -> None:
    """Generate the job JSON and emit the ``sbatch`` command(s) for the contender run."""
    plan = TabArenaBenchmarkPlan(
        benchmark_name=BENCHMARK_NAME,
        model_jobs=[
            ModelJob(
                models=(CONTENDER_MODEL, 0),
                name="gpu",
                resources={
                    "num_gpus": 1,
                    "fake_memory_for_estimates": 80,  # we have an 80 GB VRAM GPU
                    "time_limit": 3600 * 12,  # higher time limit for a TFM job, as for TabICL
                },
            ),
        ],
        # No `task_subset` runs the full suite; scope it with e.g. `task_subset=TaskSubset(subset=["core"])`
        # (the recommended protocol) or `TaskSubset(dataset_names=[...])` so only those tasks are fetched.
        context=BeyondArenaContext(),
        experiment_bundle=BeyondArenaExperimentBundle(),
        path_setup=_path_setup(),
        resources_setup=BeyondArenaResourcesSetup(),
        scheduler_setup=GCPSlurmSetup(bundle_size=1),
    )
    plan.setup_jobs()


def evaluate() -> None:
    """Leaderboard per subset: the new contender run vs. the uploaded BeyondArena baselines.

    The baselines are not wired up by hand — they are the methods ``BeyondArenaContext`` already
    knows (the uploaded ``beyond_method_metadata_collection``). We post-process the contender's raw
    results and register them via ``extra_methods=``; ``only_valid_tasks=True`` then scopes every
    leaderboard to the exact tasks the contender ran.
    """
    from tabarena.evaluation._eval_common import (
        MethodArtifact,
        post_process_to_results,
        resolve_ag_name,
        subset_label,
    )
    from tabarena.evaluation.beyond_metadata import load_beyond_task_metadata_collection

    # BeyondArena task metadata (committed CSV) — post-processing the raw run needs it to map the
    # data-foundry task ids to datasets / metrics.
    task_metadata = load_beyond_task_metadata_collection("BeyondArena")

    # Post-process the contender's raw ``results.pkl`` (written by `setup`) into the cache under this
    # run's own suite name, then load them back as an in-memory method.
    contender = post_process_to_results(
        [
            MethodArtifact(
                ag_name=resolve_ag_name(CONTENDER_MODEL),
                path_raw=_path_setup().get_output_path(BENCHMARK_NAME) / "data",
                suite=BENCHMARK_NAME,
                result_suffix=RESULT_SUFFIX,
            ),
        ],
        task_metadata=task_metadata,
    )

    context = BeyondArenaContext(
        extra_methods=contender.to_method_metadata_lst(),
        only_valid_tasks=True,
    )

    figure_output_dir = Path(__file__).parent / "eval_output" / BENCHMARK_NAME
    leaderboards = {}
    for subset in SUBSETS:
        label = subset_label(sorted(subset))
        out_dir = figure_output_dir / "subsets" / label
        leaderboard = None
        for figure_file_type in FIGURE_FILE_TYPES:
            result = context.compare(output_dir=out_dir, subset=subset or None, figure_file_type=figure_file_type)
            leaderboard = leaderboard if leaderboard is not None else result
        print(f"\n##### Leaderboard [{label}]")
        print(leaderboard.to_markdown(index=False))
        leaderboards[label] = leaderboard

    # Cross-subset overview plots (per-family / per-model), the contender highlighted.
    from tabarena.plot.subset_results import plot_subset_results

    contender_label = f"{CONTENDER_MODEL}{RESULT_SUFFIX}" if RESULT_SUFFIX else CONTENDER_MODEL
    plot_subset_results(leaderboards, figure_output_dir / "result_plots", contenders=[contender_label])


MODES = {"setup": setup, "eval": evaluate}
DEFAULT_MODE = "setup"  # bare invocation (no mode arg) runs this

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the setup or eval half of this benchmark.")
    parser.add_argument("mode", nargs="?", default=DEFAULT_MODE, choices=list(MODES))
    MODES[parser.parse_args().mode]()
