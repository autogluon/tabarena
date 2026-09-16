"""Run TabArena models under an inner validation protocol other than the official one.

An arena context asserts its official inner validation protocol (TabArena: 8 bagging folds x 1 set)
on every bagged experiment it runs, so a submission cannot deviate by accident. To study another
protocol, opt out once on the context (``official_validation_protocol=False``) and choose the protocol
at any of three levels:

  * the context: ``TabArenaContext(validation_protocol=...)`` applies to every bagged experiment that
    carries none of its own;
  * the bundle: ``TabArenaV0pt1ExperimentBundle(validation_protocol=...)`` for one group of models;
  * a single experiment: ``AGModelBagExperiment(validation_protocol=...)`` for one config.

Every result records the protocol it was fit under (``result["validation_protocol"]``: flavour,
protocol, key, resolved and fitted counts), and the processed method metadata carries its key, so
such runs are never mistaken for official ones.
"""

from __future__ import annotations

from pathlib import Path

from autogluon.tabular.models import LGBModel

from tabarena.benchmark.experiment import AGModelBagExperiment, TabArenaV0pt1ExperimentBundle, ValidationProtocol
from tabarena.contexts import TABARENA_V0PT1_VALIDATION_PROTOCOL, TabArenaContext

DATASETS = ["blood-transfusion-service-center", "anneal"]

if __name__ == "__main__":
    here = Path(__file__).parent
    run_name = "custom_validation_protocol"
    results_dir = str(here / "experiments" / run_name)  # the runner's `expname` (results cache)
    eval_dir = here / "eval" / run_name  # leaderboard / figures `output_dir`

    # Level 1, the context: applies to every bagged experiment without a protocol of its own.
    # Without `official_validation_protocol=False` the context refuses any protocol but its official one.
    context = TabArenaContext(
        validation_protocol=ValidationProtocol.custom(num_bag_folds=3),
        official_validation_protocol=False,
    )

    # Level 2, the bundle: this group of models runs its own protocol instead.
    bundle_experiments = TabArenaV0pt1ExperimentBundle(
        models=[("LightGBM", 0)],
        validation_protocol=ValidationProtocol.custom(num_bag_folds=2, num_bag_sets=2),
    ).build_experiments()

    # Level 3, a single experiment: a hand-built config under the official TabArena protocol, for a
    # side-by-side comparison in the same run.
    official_experiment = AGModelBagExperiment(
        name="LightGBM_official_BAG_L1",
        model_cls=LGBModel,
        model_hyperparameters={},
        time_limit=60,
        validation_protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL,
    )

    # A registry model built without a protocol takes the context's (level 1).
    context_experiments = TabArenaV0pt1ExperimentBundle(models=[("RandomForest", 0)]).build_experiments()

    results = context.build_and_run_jobs(
        [*bundle_experiments, official_experiment, *context_experiments],
        expname=results_dir,
        subset="lite",
        build_kwargs={"dataset_names": DATASETS},
        new_result_prefix="[Custom] ",
        debug_mode=True,  # in-process native backend
    )

    # What each fit ran under, as recorded in its result.
    for result in results:
        record = result["validation_protocol"]
        print(
            f"{result['framework']} on {result['task_metadata']['name']}: {record['key']} "
            f"(enforced={record['protocol']['enforced']}, fitted "
            f"{record.get('num_bag_folds_fitted')} folds x {record.get('num_bag_sets_fitted')} sets)"
        )

    # The leaderboard still works; the registered methods are marked as running a custom protocol.
    leaderboard = context.compare(output_dir=eval_dir)
    print(context.leaderboard_to_website_format(leaderboard=leaderboard).to_markdown(index=False))
