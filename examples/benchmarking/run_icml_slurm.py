from __future__ import annotations

from typing import Any
import argparse

import pandas as pd

from tabarena.benchmark.experiment import ExperimentBatchRunner
from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from tabarena.website.website_format import format_leaderboard


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run TabArena experiment batch.")
    p.add_argument("--exp-name", type=str, default="icml_prefinal", help="Experiment name (used for output folders).")

    # Parameters you requested
    p.add_argument("--ignore-cache", action="store_true", help="If set, overwrite caches and re-run experiments.")
    p.add_argument("--model-name", type=str, default="LR", choices=["LR", "GBM", "CAT", "TABM", "REALTABPFN-V2.5", "REALMLP", "AutoFeatLinearModel", "BaseLR", "OpenFELGBModel", "GBM-ablation"])
    p.add_argument("--start-config", type=int, default=0)
    p.add_argument("--n-configs", type=int, default=50)
    p.add_argument("--start-dataset", type=int, default=0)
    p.add_argument("--n-datasets", type=int, default=0, help="0 means use all datasets.")
    p.add_argument("--filter-datasets", type=str, default=None, help="Comma-separated list of dataset names to include. If None, use all datasets.")
    p.add_argument("--raise-on-failure", action="store_true", help="If set, raise on failure.")
    p.add_argument("--fold", type=int, default=2)

    # Optional convenience flags (not requested, but often useful)
    p.add_argument("--debug", action="store_true", help="Enable debug mode (sequential_local fold strategy).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    exp_name = args.exp_name

    expname = f"/ceph/atschalz/auto_prep/experiments/{exp_name}"
    eval_dir = f"/ceph/atschalz/auto_prep/eval/{exp_name}"

    ignore_cache: bool = bool(args.ignore_cache)
    debug: bool = bool(args.debug)
    model_name: str = args.model_name
    start_config: int = int(args.start_config)
    n_configs: int = int(args.n_configs)
    start_dataset: int = int(args.start_dataset)
    n_datasets = None if args.n_datasets in (0, None) else int(args.n_datasets)
    filter_datasets = None
    if args.filter_datasets is not None:
        filter_datasets = args.filter_datasets.split(",")
    raise_on_failure: bool = bool(args.raise_on_failure)
    fold: int = int(args.fold)

    tabarena_context = TabArenaContext()
    task_metadata = tabarena_context.task_metadata

    datasets = task_metadata.sort_values("n_samples_train_per_fold").name.tolist()
    if n_datasets is not None:
        datasets = datasets[start_dataset:start_dataset+n_datasets]

    if filter_datasets is not None:
        datasets = [d for d in datasets if d not in filter_datasets]

    folds = [fold]

    if model_name == "GBM":
        from tabarena.models.prep_lgb.generate import gen_lightgbm
        methods = gen_lightgbm.generate_all_bag_experiments(num_random_configs=200)
    elif model_name == "REALTABPFN-V2.5":
        from tabarena.models.prep_tabpfnv2_5.generate import gen_realtabpfnv25
        methods = gen_realtabpfnv25.generate_all_bag_experiments(num_random_configs=200)
    elif model_name == "LR":
        from tabarena.models.prep_lr.generate import gen_linear
        methods = gen_linear.generate_all_bag_experiments(num_random_configs=200)
    elif model_name == "TABM":
        from tabarena.models.prep_tabm.generate import gen_tabm
        methods = gen_tabm.generate_all_bag_experiments(num_random_configs=200)
    elif model_name == "CAT":
        from tabarena.models.prep_catboost.generate import gen_catboost
        methods = gen_catboost.generate_all_bag_experiments(num_random_configs=200)
    elif model_name == "REALMLP":
        from tabarena.models.prep_realmlp.generate import gen_realmlp
        methods = gen_realmlp.generate_all_bag_experiments(num_random_configs=200)
    elif model_name == "AutoFeatLinearModel":
        from tabarena.models.autofeat.generate import gen_autofeatlinear
        methods = gen_autofeatlinear.generate_all_bag_experiments(num_random_configs=200)
    elif model_name=='BaseLR':
        from tabarena.models.lr.generate import gen_linear
        methods = gen_linear.generate_all_bag_experiments(num_random_configs=200)
    elif model_name == "OpenFELGBModel":
        from tabarena.models.openfe.generate import gen_lightgbm
        methods = gen_lightgbm.generate_all_bag_experiments(num_random_configs=200)
    elif model_name == "GBM-ablation":
        from tabarena.models.prep_lgb.generate_ablation import gen_lightgbm
        methods = gen_lightgbm.generate_all_bag_experiments(num_random_configs=200)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    for i in range(len(methods)):
        methods[i].method_kwargs["model_hyperparameters"]["ag_args_ensemble"]["model_random_seed"] = 0
        methods[i].method_kwargs["model_hyperparameters"]["ag_args_ensemble"]["vary_seed_across_folds"] = True

        if model_name == "GBM-ablation":
            methods[i].name = methods[i].name.replace("prep_LightGBM", "prep_LightGBM-ablation")


    exp_batch_runner = ExperimentBatchRunner(expname=expname, task_metadata=task_metadata)

    if debug:
        for i in range(len(methods)):
            methods[i].method_kwargs["model_hyperparameters"]["ag_args_ensemble"]["fold_fitting_strategy"] = "sequential_local"

    methods = methods[start_config:n_configs+start_config]

    for m in methods:
        print(
            f"Method: {m.name}, Hyperparameters: "
            f"{[(k, v) for k, v in m.method_kwargs['model_hyperparameters'].items() if k in ['C', 'C_scale', 'scaler', 'penalty', 'proc.skew_threshold', 'proc.impute_strategy']]}"
        )

    results_lst: list[dict[str, Any]] = exp_batch_runner.run(
        datasets=datasets,
        folds=folds,
        methods=methods,
        ignore_cache=ignore_cache,
        raise_on_failure=raise_on_failure,
    )

    end_to_end = EndToEnd.from_raw(results_lst=results_lst, task_metadata=task_metadata, cache=False, cache_raw=False)
    end_to_end_results = end_to_end.to_results()

    print(f"New Configs Hyperparameters: {end_to_end.configs_hyperparameters()}")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Results:\n{end_to_end_results.model_results.head(100)}")

    from tabarena.nips2025_utils.artifacts._tabarena_method_metadata_misc import gbm_aio_0808_metadata
    extra_methods = [gbm_aio_0808_metadata]

    leaderboard: pd.DataFrame = end_to_end_results.compare_on_tabarena(
        output_dir=eval_dir,
        only_valid_tasks=True,
        use_model_results=True,
        new_result_prefix="Demo_",
        tabarena_context_kwargs={"extra_methods": extra_methods},
    )
    leaderboard_website = format_leaderboard(df_leaderboard=leaderboard)
    print(leaderboard_website.to_markdown(index=False))

    tabarena_context = TabArenaContext(extra_methods=extra_methods)
    hpo_results = pd.concat(
        [
            tabarena_context.load_hpo_results(method=m)
            for m in tabarena_context.methods
            if m not in ["AutoGluon_v140_bq_4h8c", "AutoGluon_v140_eq_4h8c", "AutoGluon_v150_eq_4h8c"]
        ]
    )
    hpo_results = hpo_results.loc[hpo_results.fold == fold].reset_index(drop=True)

    for dat in end_to_end_results.hpo_results.dataset.unique():
        hpo_results_dat = hpo_results.loc[hpo_results.dataset == dat]
        best_model = hpo_results_dat.loc[hpo_results_dat["metric_error"].idxmin(), "config_type"]

        print("---" * 10 + dat + "---" * 10)
        print(
            f"New Model Results: {end_to_end_results.model_results.loc[end_to_end_results.model_results.dataset==dat, ['fold', 'method', 'metric_error', 'metric_error_val']]}"
        )
        print("---" * 20)
        print(
            f"New HPO Results: {end_to_end_results.hpo_results.loc[end_to_end_results.hpo_results.dataset==dat, ['fold', 'method', 'metric_error', 'metric_error_val']]}"
        )
        print("---" * 20)
        print(f"Old {model_name} Results: {hpo_results_dat.loc[hpo_results_dat.method.apply(lambda x: model_name in x), ['method', 'metric_error']]}")
        print("---" * 20)
        print(f"Best Model Results: {hpo_results_dat.loc[hpo_results_dat.method.apply(lambda x: best_model in x), ['method', 'metric_error']]}")
