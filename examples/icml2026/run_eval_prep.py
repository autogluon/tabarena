import os
import shutil
from pathlib import Path

import pandas as pd

from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from tabarena.nips2025_utils.artifacts import tabarena_method_metadata_collection
from bencheval.website_format import format_leaderboard
from tabarena.nips2025_utils.artifacts._tabarena_method_metadata_2026_01_23_tabprep import tabprep_lr_metadata, \
    tabprep_gbm_metadata, tabprep_tabm_metadata, tabprep_realtabpfnv250_metadata
from tabarena.utils.cache_v2 import cached_parquet_df, CacheMode

from autogluon.common import TabularDataset


# Only methods on live TabArena leaderboard as of Jan 26th, 2026 that have results for all tasks
_methods_icml_paper = [
    # "AutoGluon_v140_bq_4h8c",
    "AutoGluon_v140_eq_4h8c",
    # "AutoGluon_v150_eq_4h8c",

    "CatBoost",
    "ExplainableBM",
    "ExtraTrees",
    "KNeighbors",
    "LightGBM",
    "LinearModel",
    "NeuralNetFastAI",
    "NeuralNetTorch",
    "RandomForest",
    "XGBoost",

    # "Mitra_GPU",
    "ModernNCA_GPU",
    "RealMLP_GPU",
    "TabDPT_GPU",
    # "TabICL_GPU",
    "TabM_GPU",
    # "TabPFNv2_GPU",

    "xRFM_GPU",
    # "LimiX_GPU",
    # "BetaTabPFN_GPU",
    # "TabFlex_GPU",
    "RealTabPFN-v2.5",
    "SAP-RPT-OSS",
]


if __name__ == '__main__':
    fig_output_dir = Path("tabarena_figs") / "icml"
    fig_output_dir_alt = Path("tabarena_figs") / "icml_alt"

    method_metadatas = []
    for method in _methods_icml_paper:
        method_metadata = tabarena_method_metadata_collection.get_method_metadata(method=method)
        method_metadatas.append(method_metadata)

    extra_methods = [
        tabprep_gbm_metadata,
        tabprep_lr_metadata,
        tabprep_realtabpfnv250_metadata,
        tabprep_tabm_metadata,
    ]

    ta_context = TabArenaContext(
        methods=method_metadatas,
        extra_methods=extra_methods,
    )

    calibration_framework = "RF (default)"
    result_baselines = ta_context.load_results_paper()
    result_baselines = TabArenaContext.fillna_metrics(
        df_to_fill=result_baselines,
        df_fillna=result_baselines[result_baselines["method"] == calibration_framework],
    )

    path_hpo_combined = Path("results") / "hpo_combined"

    # results_portfolio = ta_context.simulate_portfolio(
    #     methods=[e.method for e in extra_methods] + ["LightGBM", "RealTabPFN-v2.5"],
    #     # methods=[e.method for e in extra_methods] + ["LightGBM", "TabPFNv2_GPU"],
    #     time_limit=None,
    #     n_portfolios=[2],
    #     # config_fallback="RandomForest_c1_BAG_L1",
    #     config_fallback=None,
    # )
    # results_portfolio["method"] = "PortfolioTabPrep"
    # TabularDataset.save(path="results/result_portfolio_icml.parquet", df=results_portfolio)
    # results_portfolio = TabularDataset.load(path="results/result_portfolio_icml.parquet")
    # results_portfolio["method"] = "Ensemble of 1 TabPrep-LightGBM & 1 RealTabPFN-2.5"

    # results_icml_defaults = ta_context.simulate_portfolio_from_configs(
    #     configs=[
    #         # "PrepLightGBM_c1_BAG_L1",  # FIXME: The correct name
    #         "prep_LightGBM_icml_v3_c1_BAG_L1",
    #         "RealTabPFN-v2.5_c1_BAG_L1",
    #     ],
    # )
    # results_icml_defaults["method"] = "PrepLightGBM + RealTabPFN-2.5 Defaults"
    # TabularDataset.save(path="results/result_portfolio_icml_defaults.parquet", df=results_icml_defaults)
    results_icml_defaults = TabularDataset.load(path="results/result_portfolio_icml_defaults.parquet")

    mode = CacheMode.USE_IF_EXISTS

    extra_kwargs = {
        "n_configs": 201,
        "fit_order": "random",
    }

    extra_results = []

    for name_prefix, fit_kwargs in [
        ("", extra_kwargs),
        # ("Full", {}),
    ]:

        results_hpo_prep_w_tabpfn = cached_parquet_df(
            path=path_hpo_combined / f"{name_prefix}RealTabPFN-2.5.parquet",
            mode=mode,
            compute_fn=lambda: ta_context.combine_hpo(
                methods=["RealTabPFN-v2.5", "PrepRealTabPFN-v2.5"],
                method_default="RealTabPFN-v2.5",
                new_config_type=f"{name_prefix}(Prep)RealTabPFN-2.5",
                ta_name=f"{name_prefix}(Prep)RealTabPFN-2.5",
                ta_suite=f"{name_prefix}(Prep)RealTabPFN-2.5",
                **fit_kwargs,
            ),
        )

        results_hpo_prep_w_lightgbm = cached_parquet_df(
            path=path_hpo_combined / f"{name_prefix}LightGBM.parquet",
            mode=mode,
            compute_fn=lambda: ta_context.combine_hpo(
                methods=["LightGBM", "PrepLightGBM"],
                method_default="PrepLightGBM",
                new_config_type=f"{name_prefix}(Prep)LightGBM",
                ta_name=f"{name_prefix}(Prep)LightGBM",
                ta_suite=f"{name_prefix}(Prep)LightGBM",
                **fit_kwargs,
            ),
        )

        results_hpo_prep_w_tabm = cached_parquet_df(
            path=path_hpo_combined / f"{name_prefix}TabM.parquet",
            mode=mode,
            compute_fn=lambda: ta_context.combine_hpo(
                methods=["TabM_GPU", "PrepTabM"],
                method_default="PrepTabM",
                new_config_type=f"{name_prefix}(Prep)TabM",
                ta_name=f"{name_prefix}(Prep)TabM",
                ta_suite=f"{name_prefix}(Prep)TabM",
                **fit_kwargs,
            ),
        )

        results_hpo_prep_w_lr = cached_parquet_df(
            path=path_hpo_combined / f"{name_prefix}Linear.parquet",
            mode=mode,
            compute_fn=lambda: ta_context.combine_hpo(
                methods=["LinearModel", "PrepLinearModel"],
                method_default="PrepLinearModel",
                new_config_type=f"{name_prefix}(Prep)Linear",
                ta_name=f"{name_prefix}(Prep)Linear",
                ta_suite=f"{name_prefix}(Prep)Linear",
                **fit_kwargs,
            ),
        )

        extra_results += [
            results_hpo_prep_w_tabpfn,
            results_hpo_prep_w_lightgbm,
            results_hpo_prep_w_tabm,
            results_hpo_prep_w_lr,
        ]

    extra_results = pd.concat(extra_results, ignore_index=True)


    # results_hpo_prep_realtabpfn = ta_context.run_hpo(
    #     method=["RealTabPFN-v2.5", "PrepRealTabPFN-v2.5"],
    # )
    # TabularDataset.save(path="results/results_hpo_prep_realtabpfn.parquet", df=results_hpo_prep_realtabpfn)
    # results_hpo_prep_realtabpfn = TabularDataset.load(path="results/results_hpo_prep_realtabpfn.parquet")
    # results_hpo_prep_realtabpfn["method_type"] = "baseline"

    # results_icml_defaults_all = ta_context.simulate_portfolio_from_configs(
    #     configs=[
    #         # "PrepLightGBM_c1_BAG_L1",  # FIXME: The correct name
    #         "prep_LightGBM_icml_v3_c1_BAG_L1",
    #         "LightGBM_c1_BAG_L1",
    #         # "prep_TabM_c1_BAG_L1",
    #         # "TabM_GPU_c1_BAG_L1",
    #         "prep_RealTabPFN-v2.5_c1_BAG_L1",
    #         "RealTabPFN-v2.5_c1_BAG_L1",
    #     ],
    # )
    # results_icml_defaults_all["method"] = "All Prep Defaults"
    # TabularDataset.save(path="results/result_portfolio_icml_defaults_all.parquet", df=results_icml_defaults_all)
    results_icml_defaults_all = TabularDataset.load(path="results/result_portfolio_icml_defaults_all.parquet")
    results_icml_defaults_all["method"] = "(Prep)LightGBM + (Prep)RealTabPFN-2.5 Defaults"


    # results_icml_defaults_no_prep = ta_context.simulate_portfolio_from_configs(
    #     configs=["LightGBM_c1_BAG_L1", "RealTabPFN-v2.5_c1_BAG_L1"],
    # )
    # results_icml_defaults_no_prep["method"] = "LightGBM + RealTabPFN-2.5 Defaults"
    # TabularDataset.save(path="results/result_portfolio_icml_defaults_gbm_tabpfn.parquet", df=results_icml_defaults_no_prep)
    results_icml_defaults_no_prep = TabularDataset.load(path="results/result_portfolio_icml_defaults_gbm_tabpfn.parquet")

    extra_results = pd.concat([
        extra_results,
        # results_icml_defaults_no_prep,
        # results_icml_defaults_all,
    ], ignore_index=True)

    subsets = []

    # #1f77b4  # muted blue
    # #ff7f0e  # muted orange
    # #2ca02c  # muted green
    # #9467bd  # muted purple

    method_style_map = {
        "(Prep)LightGBM": {
            "color": "#1f77b4", "fontweight": "bold",
            "display_name": "PrepLightGBM",
        },
        "PrepLightGBM": {"color": "#1f77b4", "fontstyle": "italic"},
        "LightGBM": {"color": "#1f77b4", "alpha": 0.85},

        "(Prep)RealTabPFN-2.5": {
            "color": "#ff7f0e", "fontweight": "bold",
            "display_name": "PrepTabPFN-2.5",
        },
        "PrepRealTabPFN-2.5": {"color": "#ff7f0e", "fontstyle": "italic"},
        "RealTabPFN-2.5": {
            "color": "#ff7f0e", "alpha": 0.85,
            "display_name": "TabPFN-2.5",
        },

        "(Prep)TabM": {
            "color": "#2ca02c", "fontweight": "bold",
            "display_name": "PrepTabM",
        },
        "PrepTabM": {"color": "#2ca02c", "fontstyle": "italic"},
        "TabM": {"color": "#2ca02c", "alpha": 0.85},

        "(Prep)Linear": {
            "color": "#9467bd", "fontweight": "bold",
            "display_name": "PrepLinear",
        },
        "PrepLinear": {"color": "#9467bd", "fontstyle": "italic"},
        "Linear": {"color": "#9467bd", "alpha": 0.85},

        "(Prep)LightGBM + (Prep)RealTabPFN-2.5 Defaults": {
            "color": "#ff7f0e",
            "text_fontsize": 8,
            "text_fontweight": "bold",
            "line_width": 2,
            "line_ls": ":",
            "display_name": "(Prep)LightGBM +\n(Prep)RealTabPFN-2.5\n(Ens. of Defaults)"
        },

        "LightGBM + RealTabPFN-2.5 Defaults": {
            "color": "black",
            "text_fontsize": 8,
            # "text_fontweight": "bold",
            "line_width": 2,
            "line_ls": ":",
            "display_name": "LightGBM +\nRealTabPFN-2.5\n(Ens. of Defaults)"
        },
        "AutoGluon 1.4 (extreme, 4h)": {
            "color": "black",  # "color": "purple",
            "text_fontsize": 8,
            # "text_fontweight": "bold",
            # "line_width": 2,
            # "line_ls": ":",
            "display_name": "AutoGluon 1.4\n(extreme, 4h)"
        },
    }

    hidden_methods = [
        "xRFM",
        "TabDPT",
        "ModernNCA",
        "SAP-RPT-OSS",
        "EBM",
        "FastaiMLP",
        "ExtraTrees",
        "KNN",
        # "PrepLightGBM",
        # "PrepTabM",
        # "PrepRealTabPFN-2.5",
        # "PrepLinear",
    ]

    kwargs = dict(
        new_results=extra_results,
        # remove_imputed=True,
        average_seeds=False,
        subset=subsets,
        plot_with_baselines=True,
        folds=[0, 1, 2],
        imputed_names=[],  # FIXME: This is to hide imputed
        verbose=False,
        plot_tuning_kwargs={
            "method_style_map": method_style_map,
            "hidden_methods": hidden_methods,
            "baseline_text_y_gap": 1.5,
        },
        baselines=[
            "AutoGluon 1.4 (extreme, 4h)",
            # "LightGBM + RealTabPFN-2.5 Defaults",
            # "(Prep)LightGBM + (Prep)RealTabPFN-2.5 Defaults",
        ],
    )

    ta_context = TabArenaContext(
        methods=method_metadatas,
        # extra_methods=extra_methods,
    )

    leaderboard: pd.DataFrame = ta_context.compare(
        output_dir=fig_output_dir_alt,
        **kwargs,
    )

    # leaderboard_website = format_leaderboard(df_leaderboard=leaderboard)
    leaderboard_website_v2 = ta_context.leaderboard_to_website_format(leaderboard=leaderboard)

    leaderboard_val: pd.DataFrame = ta_context.compare(
        output_dir=fig_output_dir_alt / "val",
        score_on_val=True,
        **kwargs,
    )

    leaderboard_val["elo_val"] = leaderboard_val["elo"]

    leaderboard = pd.merge(leaderboard, leaderboard_val[["method", "elo_val"]], on="method")
    leaderboard["elo_delta"] = leaderboard["elo_val"] - leaderboard["elo"]

    print(leaderboard.to_markdown())

    print(leaderboard[["method", "elo", "elo_val", "elo_delta"]].to_markdown(index=False))

    # print(leaderboard_website.to_markdown(index=False))

    print("website v2")
    print(leaderboard_website_v2.to_markdown(index=False))

    size_subsets_lst = [
        [],
        ["tiny"],
        ["tiny-small"],
        ["small"],
        ["medium"],
    ]

    problem_types_subsets_lst = [
        [],
        ["binary"],
        ["multiclass"],
        ["classification"],
        ["regression"],
    ]

    subsets_lst = []
    for size_subset in size_subsets_lst:
        for pt_subset in problem_types_subsets_lst:
            subsets_lst.append(size_subset + pt_subset)

    kwargs.pop("subset")

    for subset in subsets_lst:
        subset_str = f"{os.path.sep}".join(subset)
        if subset_str:
            fig_output_dir_subset = fig_output_dir / subset_str
        else:
            fig_output_dir_subset = fig_output_dir / "all"
        leaderboard: pd.DataFrame = ta_context.compare(
            output_dir=fig_output_dir_subset,
            subset=subset,
            **kwargs,
        )
        print(f"{subset}")
        print(leaderboard.head(10).to_markdown())
        print()


    zip_results = True
    if zip_results:
        file_prefix = f"prep_paper_results"
        file_name = f"{file_prefix}.zip"
        shutil.make_archive(file_prefix, "zip", root_dir=fig_output_dir)
