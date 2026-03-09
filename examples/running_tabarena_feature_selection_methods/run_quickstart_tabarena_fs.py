from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Type

import pandas as pd
from autogluon.core.models import AbstractModel

from examples.running_tabarena_feature_selection_methods.run_tabarena_fs import run_experiment
from tabarena.benchmark.experiment import ExperimentBatchRunner, AGModelExperiment
from tabarena.benchmark.feature_selection_methods.ag import Boruta
from tabarena.benchmark.models.wrapper.AutoGluon_class import AGSingleBagWrapper
from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from tabarena.tabarena.website_format import format_leaderboard


class FSModelBagExperiment(AGModelExperiment):
    """
    Simplified Experiment class specifically for fitting a single bagged model using AutoGluon.
    The following arguments are fixed:
        method_cls = AGSingleWrapper
        experiment_cls = OOFExperimentRunner

    All models fit this way will generate out-of-fold predictions on the entire training set,
    and will be compatible with ensemble simulations in TabArena.

    Will fit the model with `num_bag_folds` folds and `num_bag_sets` sets (aka repeats).
    In total will fit `num_bag_folds * num_bag_sets` models in the bag.

    Parameters
    ----------
    name: str
        The name of the experiment / method.
        Should be descriptive and unique compared to other methods.
        For example, `"LightGBM_c1_BAG_L1"`
    model_cls: Type[AbstractModel]
    model_hyperparameters: dict
        Identical to what you would pass to `TabularPredictor.fit(..., hyperparameters={model_cls: [model_hyperparameters]})
    time_limit: float, optional
    num_bag_folds: int, default 8
    num_bag_sets: int, default 1
    method_kwargs: dict, optional
    experiment_kwargs: dict, optional
    """
    _method_cls = AGSingleBagWrapper

    def __init__(
        self,
        name: str,
        model_cls: Type[AbstractModel],
        model_hyperparameters: dict,
        *,
        fs_method,
        time_limit: float | None = None,
        num_bag_folds: int = 8,
        num_bag_sets: int = 1,
        raise_on_model_failure: bool = True,
        method_kwargs: dict = None,
        experiment_kwargs: dict = None,
    ):
        if method_kwargs is None:
            method_kwargs = {}
        method_kwargs = copy.deepcopy(method_kwargs)
        assert isinstance(num_bag_folds, int)
        assert isinstance(num_bag_sets, int)
        assert isinstance(method_kwargs, dict)
        assert num_bag_folds >= 2
        assert num_bag_sets >= 1
        if "fit_kwargs" in method_kwargs:
            assert "num_bag_folds" not in method_kwargs["fit_kwargs"], f"Set `num_bag_folds` directly in {self.__class__.__name__} rather than in `fit_kwargs`"
            assert "num_bag_sets" not in method_kwargs["fit_kwargs"], f"Set `num_bag_sets` directly in {self.__class__.__name__} rather than in `fit_kwargs`"
            method_kwargs["fit_kwargs"] = copy.deepcopy(method_kwargs["fit_kwargs"])
        else:
            method_kwargs["fit_kwargs"] = {}
        method_kwargs["fit_kwargs"]["num_bag_folds"] = num_bag_folds
        method_kwargs["fit_kwargs"]["num_bag_sets"] = num_bag_sets
        super().__init__(
            name=name,
            model_cls=model_cls,
            model_hyperparameters=model_hyperparameters,
            fs_method=fs_method,
            time_limit=time_limit,
            raise_on_model_failure=raise_on_model_failure,
            method_kwargs=method_kwargs,
            experiment_kwargs=experiment_kwargs,
        )

"""
if __name__ == '__main__':
    expname = str(Path(__file__).parent / "experiments" / "quickstart_with_fs")
    eval_dir = Path(__file__).parent / "eval" / "quickstart_with_fs"
    ignore_cache = False

    tabarena_context = TabArenaContext()
    task_metadata = tabarena_context.task_metadata

    datasets = ["anneal", "credit-g", "diabetes"]
    folds = [0]

    from tabarena.benchmark.models.ag import RealMLPModel
    from autogluon.tabular.models import LGBModel

    # Import your feature selection methods
    from sklearn.feature_selection import SelectKBest, f_classif
    import xgboost as xgb

    # Define feature selection methods
    fs_methods = {
        "SelectKBest": SelectKBest(f_classif, k=10),
        "Boruta": Boruta(),
        "None": None,  # Baseline without FS
    }

    # Create method combinations: model + FS
    methods = []

    # LightGBM + SelectKBest (different prefix to avoid conflict)
    methods.append(
        FSModelBagExperiment(
            name="LightGBM_FS_SelectKBest_c1_BAG_L1",  # Changed naming
            model_cls=LGBModel,
            fs_method=fs_methods["Boruta"],
            model_hyperparameters={},
            num_bag_folds=8,
            time_limit=3600,
        )
    )

    exp_batch_runner = ExperimentBatchRunner(expname=expname, task_metadata=task_metadata)

    results_lst: list[dict[str, Any]] = exp_batch_runner.run(
        datasets=datasets,
        folds=folds,
        methods=methods,
        ignore_cache=ignore_cache,
    )

    # Compute results
    end_to_end = EndToEnd.from_raw(
        results_lst=results_lst,
        task_metadata=task_metadata,
        cache=False,
        cache_raw=False
    )
    end_to_end_results = end_to_end.to_results()

    print(f"New Configs Hyperparameters: {end_to_end.configs_hyperparameters()}")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Results:\n{end_to_end_results.model_results.head(100)}")

    leaderboard: pd.DataFrame = end_to_end_results.compare_on_tabarena(
        output_dir=eval_dir,
        only_valid_tasks=True,
        use_model_results=True,
        new_result_prefix="Demo_FS_",
    )
    leaderboard_website = format_leaderboard(df_leaderboard=leaderboard)
    print(leaderboard_website.to_markdown(index=False))
"""

if __name__ == '__main__':
    expname = str(Path(__file__).parent / "experiments" / "quickstart_with_fs")
    eval_dir = Path(__file__).parent / "eval" / "quickstart_with_fs"
    output_dir = Path(__file__).parent / "results"

    tabarena_context = TabArenaContext()
    task_metadata = tabarena_context.task_metadata

    datasets = ["anneal", "credit-g", "diabetes"]
    folds = [0]

    # Your FS methods
    fs_methods = ["Boruta"]

    # Configs
    configs = ["LightGBM_c1_BAG_L1_Reproduced"]

    results_lst = []

    # Run experiments with all combinations of config + FS method
    for config_idx, config in enumerate(configs):
        for task_dataset in datasets:
            for fold in folds:
                for fs_method in fs_methods:
                    print(f"\n{'=' * 60}")
                    print(f"Running: {config} + {fs_method} on {task_dataset} (fold {fold})")
                    print(f"{'=' * 60}")

                    result = run_experiment(
                        config_index=None,
                        task_id=146818,  # Or extract task_id from dataset name
                        fs_method=fs_method,
                        fold=fold,
                        repeat=0,
                        configs_yaml_file="../../tabflow/configs/model_configs.yaml",
                        output_dir=output_dir,
                        ignore_cache=False,
                        num_cpus=8,
                        num_gpus=0,
                        memory_limit=16000,
                        sequential_local_fold_fitting=False,
                    )

                    if result is not None:
                        results_lst.append(result)

    # Now evaluate all results together
    if results_lst:
        end_to_end = EndToEnd.from_raw(
            results_lst=results_lst,
            task_metadata=task_metadata,
            cache=False,
            cache_raw=False
        )
        end_to_end_results = end_to_end.to_results()

        print(f"New Configs Hyperparameters: {end_to_end.configs_hyperparameters()}")
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
            print(f"Results:\n{end_to_end_results.model_results.head(100)}")

        leaderboard: pd.DataFrame = end_to_end_results.compare_on_tabarena(
            output_dir=eval_dir,
            only_valid_tasks=True,
            use_model_results=True,
            new_result_prefix="Demo_FS_",
        )
        leaderboard_website = format_leaderboard(df_leaderboard=leaderboard)
        print(leaderboard_website.to_markdown(index=False))