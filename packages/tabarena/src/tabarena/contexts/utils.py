from __future__ import annotations

from typing import TYPE_CHECKING

from tabarena.simulation.dense_utils import intersect_folds_and_datasets, prune_zeroshot_gt

if TYPE_CHECKING:
    from tabarena.predictions.tabular_predictions import TabularModelPredictions
    from tabarena.simulation.ground_truth import GroundTruth
    from tabarena.simulation.simulation_context import ZeroshotSimulatorContext


def load_zeroshot_input(
    path_pred_proba: str,
    paths_gt: list[str],
    datasets: list[str],
    zsc: ZeroshotSimulatorContext,
    prediction_format: str = "memmap",
    verbose: bool = True,
) -> tuple[TabularModelPredictions, GroundTruth, ZeroshotSimulatorContext]:
    if verbose:
        print(
            f"Loading ZS inputs:\n\tpred_proba:  {path_pred_proba}\n",
        )
    zeroshot_gt = zsc.load_groundtruth(paths_gt=paths_gt)
    zeroshot_pred_proba = zsc.load_pred(
        path_pred_proba=path_pred_proba,
        datasets=datasets,
        prediction_format=prediction_format,
    )

    # keep only dataset whose folds are all present
    intersect_folds_and_datasets(zsc, zeroshot_pred_proba, zeroshot_gt=zeroshot_gt)
    zeroshot_pred_proba.restrict_models(zsc.get_configs())
    zeroshot_gt = prune_zeroshot_gt(
        dataset_to_tid_dict=zsc.dataset_to_tid_dict, zeroshot_pred_proba=zeroshot_pred_proba, zeroshot_gt=zeroshot_gt
    )

    return zeroshot_pred_proba, zeroshot_gt, zsc
