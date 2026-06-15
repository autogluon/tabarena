from __future__ import annotations

from pathlib import Path

from tabarena.benchmark.experiment import BeyondArenaExperimentBundle
from tabarena.benchmark.task.metadata import BeyondArenaTaskMetadataCollection
from tabarena.evaluation.context.beyond_arena import BeyondArenaContext

DATASETS = [
    "hepatitis_survival_prediction",
    "cirrhosis_patient_survival_prediction",
    "clock_protein_toxicity",
]

if __name__ == "__main__":
    here = Path(__file__).parent
    run_name = "register_new_methods_beyondarena"
    results_dir = str(here / "experiments" / run_name)
    eval_dir = here / "eval" / run_name

    task_collection = BeyondArenaTaskMetadataCollection().subset_tasks(
        split_indices="lite",
        dataset_names=DATASETS,
    )

    experiments = BeyondArenaExperimentBundle(
        models=[("Linear", 0)],
    ).build_experiments()

    context = BeyondArenaContext(task_metadata=task_collection)
    context.run_experiments(
        experiments,
        expname=results_dir,
        new_result_prefix="[New] ",
        # debug_mode=True,  # <-- For local debugger
    )
    leaderboard = context.compare(output_dir=eval_dir)
    print(leaderboard.to_markdown())
