import pandas as pd
import numpy as np

from tabarena.predictions import TabularPredictionsInMemory
from tabarena.repository import EvaluationRepository
from tabarena.simulation.ground_truth import GroundTruth
from tabarena.simulation.simulation_context import ZeroshotSimulatorContext


def make_random_metric(model):
    output_cols = ['time_train_s', 'time_infer_s', 'metric_error', 'metric_error_val']
    metric_value_dict = {
        "NeuralNetFastAI_r1": 1.0,
        "NeuralNetFastAI_r2": 2.0,
        "b1": -1.0,
        "b2": -2.0,
        "b_e1": -3.0,
    }
    metric_value = metric_value_dict[model]
    return {output_col: (i + 1) * metric_value for i, output_col in enumerate(output_cols)}


def load_context_artificial(
    n_classes: int = 25,
    problem_type: str = "regression",
    seed=0,
    include_hyperparameters: bool = False,
    add_baselines_extra: bool = False,
    include_configs: bool = True,
    include_baselines: bool = True,
    dtype=np.float32,
    **kwargs,
):
    # TODO write specification of dataframes schema, this code produces a minimal example that enables
    #  to use all the features required in evaluation such as listing datasets, evaluating ensembles or
    #  comparing to baselines
    rng = np.random.default_rng(seed)

    datasets = ["ada", "abalone"]
    tids = [359944, 359946]
    n_folds = 3
    models = ["NeuralNetFastAI_r1", "NeuralNetFastAI_r2"]
    baselines = ["b1", "b2"]
    configs_hyperparameters = None
    if include_hyperparameters:
        configs_hyperparameters = {
            "NeuralNetFastAI_r1": {
                "hyperparameters": {"foo": 10, "bar": "hello"},
                "model_type": "FASTAI",
            },
            "NeuralNetFastAI_r2": {
                "hyperparameters": {"foo": 15, "x": "y"},
                "model_type": "FASTAI",
            },
        }

    configs_full = {model: {} for model in models}

    df_metadata = pd.DataFrame([{
        'dataset': dataset,
        'task_type': "TaskType.SUPERVISED_CLASSIFICATION",
    }
        for tid, dataset in zip(tids, datasets)
    ])

    df_baselines = pd.DataFrame(
        {
            "dataset": dataset,
            "tid": tid,
            "fold": fold,
            "framework": baseline,
            "problem_type": problem_type,
            "metric": "root_mean_squared_error",
            **make_random_metric(baseline),
        } for fold in range(n_folds) for baseline in baselines for (tid, dataset) in zip(tids, datasets)
    )

    if add_baselines_extra:
        baselines_extra = ["b1", "b_e1"]
        datasets_extra = ["a", "b"]
        tids_extra = [5, 6]
        df_baselines_extra = pd.DataFrame(
            {
                "dataset": dataset,
                "tid": tid,
                "fold": fold,
                "framework": baseline,
                "problem_type": problem_type,
                "metric": "root_mean_squared_error",
                **make_random_metric(baseline),
            } for fold in [0] for baseline in baselines_extra for (tid, dataset) in zip(tids_extra, datasets_extra)
        )
        df_baselines = pd.concat([df_baselines, df_baselines_extra], ignore_index=True)
        df_metadata_extra = pd.DataFrame([{
            'dataset': dataset,
            'task_type': "TaskType.SUPERVISED_CLASSIFICATION",
        }
            for tid, dataset in zip(tids_extra, datasets_extra)
        ])
        df_metadata = pd.concat([df_metadata, df_metadata_extra], ignore_index=True)

    df_raw = pd.DataFrame(
        {
            "dataset": dataset,
            "tid": tid,
            "fold": fold,
            "framework": model,
            "problem_type": problem_type,
            "metric": "root_mean_squared_error",
            **make_random_metric(model),
        } for fold in range(n_folds) for model in models for (tid, dataset) in zip(tids, datasets)
    )
    if not include_configs:
        df_raw = None
    if not include_baselines:
        df_baselines = None
    zsc = ZeroshotSimulatorContext(
        df_configs=df_raw,
        df_baselines=df_baselines,
        folds=list(range(n_folds)),
        df_metadata=df_metadata,
        configs_hyperparameters=configs_hyperparameters,
    )
    pred_dict = {
        dataset_name: {
            fold: {
                "pred_proba_dict_val": {
                    m: rng.random((123, n_classes), dtype=dtype) if n_classes > 2 else rng.random(123, dtype=dtype)
                    for m in models
                },
                "pred_proba_dict_test": {
                    m: rng.random((13, n_classes), dtype=dtype) if n_classes > 2 else rng.random(13, dtype=dtype)
                    for m in models
                }
            }
            for fold in range(n_folds)
        }
        for dataset_name in datasets
    }
    zeroshot_pred_proba = TabularPredictionsInMemory.from_dict(pred_dict)

    make_dict = lambda size: {
        dataset: {
            fold: pd.Series(rng.integers(low=0, high=n_classes, size=size))
            for fold in range(n_folds)
        }
        for dataset in datasets
    }

    zeroshot_gt = GroundTruth(label_val_dict=make_dict(123), label_test_dict=make_dict(13))

    if not include_configs:
        zeroshot_pred_proba = None
        zeroshot_gt = None

    return zsc, configs_full, zeroshot_pred_proba, zeroshot_gt


def load_repo_artificial(**kwargs) -> EvaluationRepository:
    zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = load_context_artificial(**kwargs)
    return EvaluationRepository(
        zeroshot_context=zsc,
        tabular_predictions=zeroshot_pred_proba,
        ground_truth=zeroshot_gt,
    )


if __name__ == '__main__':
    load_context_artificial()
