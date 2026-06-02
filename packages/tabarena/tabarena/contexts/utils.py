from __future__ import annotations

from typing import List, Tuple

from ..simulation.dense_utils import intersect_folds_and_datasets, prune_zeroshot_gt
from ..simulation.ground_truth import GroundTruth
from ..simulation.simulation_context import ZeroshotSimulatorContext
from ..predictions.tabular_predictions import TabularModelPredictions


def load_zeroshot_input(path_pred_proba: str,
                        paths_gt: List[str],
                        datasets: List[str],
                        zsc: ZeroshotSimulatorContext,
                        prediction_format: str = "memmap",
                        verbose: bool = True,
                        ) -> Tuple[TabularModelPredictions, GroundTruth, ZeroshotSimulatorContext]:
    if verbose:
        print(
            f'Loading ZS inputs:\n'
            f'\tpred_proba:  {path_pred_proba}\n'
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
    zeroshot_gt = prune_zeroshot_gt(dataset_to_tid_dict=zsc.dataset_to_tid_dict, zeroshot_pred_proba=zeroshot_pred_proba, zeroshot_gt=zeroshot_gt)

    return zeroshot_pred_proba, zeroshot_gt, zsc
