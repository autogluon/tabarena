from __future__ import annotations

from typing import List, Tuple

from ..simulation.ground_truth import GroundTruth
from ..simulation.simulation_context import ZeroshotSimulatorContext
from ..predictions.tabular_predictions import TabularModelPredictions


def intersect_folds_and_datasets(zsc: ZeroshotSimulatorContext,
                                 zeroshot_pred_proba: TabularModelPredictions,
                                 zeroshot_gt: GroundTruth):
    zpp_datasets = [dataset for dataset in zeroshot_pred_proba.datasets]
    zsc_datasets = zsc.get_datasets()
    zsc_datasets_set = set(zsc_datasets)
    valid_datasets = [dataset for dataset in zpp_datasets if dataset in zsc_datasets_set]
    if set(zpp_datasets) != set(valid_datasets):
        zeroshot_pred_proba.restrict_datasets(datasets=valid_datasets)
        zpp_datasets = zeroshot_pred_proba.datasets
        gt_datasets = zeroshot_gt.datasets
        for d in gt_datasets:
            if d not in zpp_datasets:
                zeroshot_gt.remove_dataset(dataset=d)
    zsc_datasets = zsc.get_datasets()
    zpp_datasets = [dataset for dataset in zeroshot_pred_proba.datasets if dataset in zsc_datasets]
    zsc.subset_datasets(zpp_datasets, only_configs=True)


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
    # force_to_dense(zeroshot_pred_proba, first_prune_method='task', second_prune_method='dataset')

    # zsc.subset_configs(zeroshot_pred_proba.models)
    # zsc.subset_datasets(zeroshot_pred_proba.datasets)
    zeroshot_pred_proba.restrict_models(zsc.get_configs())
    zeroshot_gt = prune_zeroshot_gt(dataset_to_tid_dict=zsc.dataset_to_tid_dict, zeroshot_pred_proba=zeroshot_pred_proba, zeroshot_gt=zeroshot_gt)

    return zeroshot_pred_proba, zeroshot_gt, zsc


def prune_zeroshot_gt(dataset_to_tid_dict, zeroshot_pred_proba: TabularModelPredictions | None, zeroshot_gt: GroundTruth, verbose: bool = True) -> GroundTruth:

    num_datasets_start = len(zeroshot_gt.datasets)
    if zeroshot_pred_proba is not None:
        dataset_pred = set(dataset for dataset in zeroshot_pred_proba.datasets if dataset in dataset_to_tid_dict)
    else:
        dataset_pred = set(dataset_to_tid_dict.keys())
    for dataset in zeroshot_gt.datasets:
        if dataset not in dataset_pred:
            zeroshot_gt.remove_dataset(dataset=dataset)
    num_datasets_end = len(zeroshot_gt.datasets)
    if verbose:
        print(f'Aligning GroundTruth with TabularPredictions... (Dataset count {num_datasets_start} -> {num_datasets_end})')
    assert len(dataset_pred) == num_datasets_end
    return zeroshot_gt
