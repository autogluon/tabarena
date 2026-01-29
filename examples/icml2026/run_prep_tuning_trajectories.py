from tabarena.nips2025_utils.artifacts._tabarena_method_metadata_2026_01_23_tabprep import tabprep_lr_metadata, \
    tabprep_gbm_metadata, tabprep_tabm_metadata, tabprep_realtabpfnv250_metadata

from pathlib import Path
from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from tabarena.nips2025_utils.artifacts import tabarena_method_metadata_collection
from examples.plots.run_plot_pareto_over_tuning_time import plot_tuning_trajectories_all
from tabarena.utils.cache_v2 import cached_parquet_df, CacheMode


if __name__ == '__main__':
    path_prefix = Path("results") / "hpo_trajectory"
    mode = CacheMode.USE_IF_EXISTS
    folds = [0, 1, 2]
    seeds = 20
    n_configs = [
        1,
        2,
        5,
        10,
        25,
        50,
        100,
        150,
        201,
    ]

    ta_context = TabArenaContext(
        extra_methods=[
            tabprep_gbm_metadata,
            tabprep_lr_metadata,
            tabprep_realtabpfnv250_metadata,
            tabprep_tabm_metadata,
        ]
    )

    hpo_trajectory_realtabpfn = cached_parquet_df(
        path=path_prefix / f"PrepRealTabPFN-2.5.parquet",
        mode=mode,
        compute_fn=lambda: ta_context.generate_hpo_trajectories(
            methods=["RealTabPFN-v2.5", tabprep_realtabpfnv250_metadata],
            display_name=tabprep_realtabpfnv250_metadata.display_name,
            n_configs=n_configs,
            seeds=seeds,
            folds=folds,
        ),
    )
    hpo_trajectory_tabm = cached_parquet_df(
        path=path_prefix / f"PrepTabM.parquet",
        mode=mode,
        compute_fn=lambda: ta_context.generate_hpo_trajectories(
            methods=["TabM_GPU", tabprep_tabm_metadata],
            display_name=tabprep_tabm_metadata.display_name,
            n_configs=n_configs,
            seeds=seeds,
            folds=folds,
        ),
    )
    hpo_trajectory_linear = cached_parquet_df(
        path=path_prefix / f"PrepLinear.parquet",
        mode=mode,
        compute_fn=lambda: ta_context.generate_hpo_trajectories(
            methods=[tabprep_lr_metadata, "LinearModel"],
            display_name=tabprep_lr_metadata.display_name,
            n_configs=n_configs,
            seeds=seeds,
            folds=folds,
        ),
    )
    hpo_trajectory_lightgbm = cached_parquet_df(
        path=path_prefix / f"PrepLightGBM.parquet",
        mode=mode,
        compute_fn=lambda: ta_context.generate_hpo_trajectories(
            methods=[tabprep_gbm_metadata, "LightGBM"],
            display_name=tabprep_gbm_metadata.display_name,
            n_configs=n_configs,
            seeds=seeds,
            folds=folds,
        ),
    )

    methods_base = [
        "RealTabPFN-v2.5",
        "TabM_GPU",
        "LightGBM",
        "LinearModel",
    ]

    method_metadatas_base = [tabarena_method_metadata_collection.get_method_metadata(method=m) for m in methods_base]

    ta_context = TabArenaContext()

    plot_kwargs = {
        "method_order": [
            "PrepRealTabPFN-2.5",
            "RealTabPFN-2.5",
            "PrepLightGBM",
            "LightGBM",
            "PrepTabM",
            "TabM",
            "PrepLinear",
            "Linear",
        ],
        "display_names": {
            "PrepRealTabPFN-2.5": "⚙️ PrepTabPFN-2.5",
            "PrepLightGBM": "⚙️ PrepLightGBM",
            "PrepTabM": "⚙️ PrepTabM",
            "PrepLinear": "⚙️ PrepLinear",
        }
    }

    plot_tuning_trajectories_all(
        tabarena_context=ta_context,
        ban_bad_methods=False,
        extra_results=[
            hpo_trajectory_realtabpfn,
            hpo_trajectory_tabm,
            hpo_trajectory_linear,
            hpo_trajectory_lightgbm,
        ],
        folds=folds,
        methods_to_display=[m.method for m in method_metadatas_base],
        plot_kwargs=plot_kwargs,
    )
