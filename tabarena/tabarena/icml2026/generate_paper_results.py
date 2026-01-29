import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tabarena.icml2026.plotting.per_dataset_results import plot_model_performance_across_datasets
from tabarena.icml2026.plotting.new_single_prep_boxplots import compare_methods_via_boxplots
from tabarena.icml2026.plotting.two_figures_boxplots import boxplot_two_dataframes_pubready, boxplot_models_combined_vs_tabprep, boxplot_dataframe_pubready
from tabarena.icml2026.plotting.single_preprocessor_boxplots import ablation_boxplot_colored_by_best

from tabarena.nips2025_utils.tabarena_context import TabArenaContext

from tabarena.nips2025_utils.fetch_metadata import load_task_metadata
datasets_metadata = load_task_metadata()

from tabarena.nips2025_utils.per_dataset_tables import get_per_dataset_tables



dat_map = {
    "HR_Analytics_Job_Change_of_Data_Scientists": "HR_Analytics",
    "students_dropout_and_academic_success": "students_dropout",
    "blood-transfusion-service-center": 'blood-transfusion',
    "Another-Dataset-on-used-Fiat-500": "Fiat-500", 
    "coil2000_insurance_policies": "coil2000",
    "hazelnut-spread-contaminant-detection": "hazelnut-spread",
    "taiwanese_bankruptcy_prediction": "taiwanese_bankruptcy",
    "polish_companies_bankruptcy": "polish_companies",
    "healthcare_insurance_expenses": "healthcare_insurance",
    "in_vehicle_coupon_recommendation": "in_vehicle",
    "Amazon_employee_access": "Amazon_employee",
    "concrete_compressive_strength": "concrete",
    "customer_satisfaction_in_airline": "customer_satisfaction",
    "E-CommereShippingData": "E-Commerce",
    "online_shoppers_intention": "online_shoppers",
    "Is-this-a-good-customer": "Is-good-customer",
    "Food_Delivery_Time": "Food_Delivery",
    "credit_card_clients_default": "credit_card_clients",
    }


ablation_base_path = "/tabarena_figs/icml_ablation"
base_path = "tabarena_figs/icml_final/"
comb_path = "tabarena/examples/icml2026/results/hpo_combined/"
save_path = "tabarena/tabarena/tabarena/icml2026/figures/new"

use_fold = 0

if __name__ == '__main__':
    ### LOAD RESULTS
    ta_context = TabArenaContext()
    # ta_context.load_configs_hyperparameters(methods = ["PrepLightGBM", "PrepLinearModel"], download=True)
    # ta_context.load_results_paper(methods=["PrepLightGBM", "PrepLinearModel"])
    ta_results = pd.concat([ta_context.load_hpo_results(i) for i in ta_context.methods if "AutoGluon" not in i]).reset_index(drop=True)


    results = ta_context.load_config_results("PrepLightGBM")
    hpo_results = ta_context.load_hpo_results("PrepLightGBM")

    # metadata = load_task_metadata()
    # task_map = dict(metadata[["name","tid"]].values)
    # results["task"] = results["dataset"].map(task_map)
    # hpo_results["task"] = hpo_results["dataset"].map(task_map)

    all_model_results = pd.DataFrame()
    all_hpo_results = pd.DataFrame()
    models = ["prep_TabM", "prep_RealTabPFN"] #, "prep_RealMLP"]
    for model_name in models:
        model_results = pd.read_csv(f"{base_path}/{model_name}/model_results.csv")
        model_results["model_name"] = model_name
        all_model_results = pd.concat([all_model_results, model_results]).reset_index(drop=True)

        hpo_results = pd.read_csv(f"{base_path}/{model_name}/hpo_results.csv")
        hpo_results["model_name"] = model_name
        all_hpo_results = pd.concat([all_hpo_results, hpo_results]).reset_index(drop=True)

    all_model_results.ta_name = all_model_results.ta_name.map({"prep_TabM": "PrepTabM", 
                                                    "RealTabPFN-v2.5": "RealTabPFN2.5", 
                                                    "prep_RealTabPFN-v2.5": "PrepRealTabPFN2.5", 
                                                    "TabM_GPU": "TabM"}).fillna(all_model_results.ta_name)

    comb_results = pd.concat([
    all_hpo_results[["dataset", "fold", "ta_name", "metric_error", "metric_error_val", "method_subtype"]], 
    ta_results[["dataset", "fold", "ta_name", "metric_error", "metric_error_val", "method_subtype"]]
    ]).reset_index(drop=True)

    comb_results.ta_name = comb_results.ta_name.map({
        "prep_TabM": "PrepTabM", 
        "RealTabPFN-v2.5": "RealTabPFN2.5", 
        "prep_RealTabPFN-v2.5": "PrepRealTabPFN2.5", 
        "TabM_GPU": "TabM"}).fillna(comb_results.ta_name)

    comb_results_use = comb_results.loc[comb_results['method_subtype']=="tuned_ensemble"]
    comb_results_use = comb_results_use.loc[comb_results_use.fold==0]

    comb_results_use.dataset = comb_results_use.dataset.apply(lambda x: dat_map.get(x, x))

    list(comb_results_use.dataset.unique())

    comb_results_use_bar = comb_results_use.copy()
    for m in ["LightGBM", "Linear", "RealTabPFN-2.5", "TabM"]:
        comb_results_use_bar = pd.concat([comb_results_use_bar, pd.read_parquet(f"{comb_path}/{m}.parquet")]).reset_index(drop=True)
    
    ta_results = ta_results.loc[ta_results.method!='MITRA_GPU (default)']
    get_per_dataset_tables(df_results=pd.concat([
        ta_results,
        comb_results_use_bar.loc[comb_results_use_bar.ta_name=="(Prep)LightGBM"]]).reset_index(drop=True)
        , save_path=f"{save_path}")
    
    
    comb_results_use_bar = comb_results_use_bar.loc[comb_results_use_bar.fold==0]
    comb_results_use_bar = comb_results_use_bar[['dataset', 'fold', 'ta_name', 'metric_error', 'metric_error_val','method_subtype']]
    comb_results_use_bar.dataset = comb_results_use_bar.dataset.apply(lambda x: dat_map.get(x, x))

    # comb_results_use_bar.ta_name = comb_results_use_bar.ta_name.map({
    #     "(Prep)TabM": "PrepTabM", 
    #     "(Prep)RealTabPFN-v2.5": "RealTabPFN2.5", 
    #     "(Prep)RealTabPFN-v2.5": "PrepRealTabPFN2.5", 
    #     "(Prep)TabM": "TabM"}).fillna(comb_results_use_bar.ta_name)
    



    # comb_results_use_bar = comb_results_use.copy()

    # for dataset_name in comb_results_use_bar.dataset.unique():

    #     for model_name in ["TabM", "LinearModel", "LightGBM", "RealTabPFN2.5"]:
    #         prep = comb_results_use_bar.loc[np.logical_and(comb_results_use_bar.dataset==dataset_name, comb_results_use_bar.ta_name==f"Prep{model_name}")]
    #         base = comb_results_use_bar.loc[np.logical_and(comb_results_use_bar.dataset==dataset_name, comb_results_use_bar.ta_name==model_name)]
    #         if prep.shape[0]==0 or base.shape[0]==0:
    #             continue
    #         if base.metric_error_val.values[0] < prep.metric_error_val.values[0]:       
    #             comb_results_use_bar.loc[np.logical_and(comb_results_use_bar.dataset==dataset_name, comb_results_use_bar.ta_name==f"Prep{model_name}"),"metric_error"] = base.metric_error.values[0]
    #             # print(f"{model_name}: {base.metric_error_val.values[0]:.4f}, {prep.metric_error_val.values[0]:.4f}")
    
    comb_results_use.loc[comb_results_use.method_subtype=="default","ta_name"] += "_default"
    comb_results_use.loc[comb_results_use.method_subtype=="tuned","ta_name"] += "_tuned"
    comb_results_use_bar.loc[comb_results_use_bar.method_subtype=="default","ta_name"] += "_default"
    comb_results_use_bar.loc[comb_results_use_bar.method_subtype=="tuned","ta_name"] += "_tuned"

    ### PERFORMANCE ACROSS DATASETS PLOT
    base_marker = "."
    prep_marker = "*"

    fig, ax = plot_model_performance_across_datasets(
        comb_results_use_bar,
        model_col="ta_name",
        mode="median_centered_signed",
        value_label="Quantile-anchored normalized score",
        # normalization_center="third_best",
        # normalization_center_model="RandomForest",
        normalization_reference_models=[i for i in comb_results_use.ta_name.unique() if "prep" not in i and "Prep" not in i],  # define reference
        display_models=["LightGBM", "(Prep)LightGBM", "RealTabPFN-2.5", "(Prep)RealTabPFN-2.5", "LinearModel", "(Prep)Linear", "TabM", "(Prep)TabM"],
        legend_order=["LightGBM", "(Prep)LightGBM", "RealTabPFN2.5", "(Prep)RealTabPFN-2.5", "LinearModel", "(Prep)Linear", "TabM", "(Prep)TabM"],
        title=None,
        sort_direction="worst_to_best",
        # sort_datasets_by_model="prep_LightGBM",
        sort_datasets_by_best_of_models=["(Prep)LightGBM", "(Prep)RealTabPFN-2.5", "(Prep)Linear", "(Prep)TabM"],
        clip_good_side=True,
        bad_side_cap=1,
        good_side_cap=-2,
        show_model_averages=False,
        figsize=(16, 6),
        exclude_marker_groups=[base_marker],
        y_tick_labels={
            -2: "Improves over \n Best as much \n as Best \nover Top 75%",
            -1.0: "Best on\nTabArena",
            0.0: "Top\n75% on\nTabArena",
            1.0: "Top 50%\nor worse on\nTabArena",
        },
        model_color_groups={
        "LightGBM": ["LightGBM", "(Prep)LightGBM"],
        "LinearModel": ["LinearModel", "(Prep)Linear"],
        "TabM": ["TabM", "(Prep)TabM"],
        "RealTabPFN": ["RealTabPFN2.5", "(Prep)RealTabPFN-2.5"],
        },
        model_markers={
        "LightGBM": base_marker,
        "(Prep)LightGBM": prep_marker,
        "LinearModel": base_marker,
        "(Prep)Linear": prep_marker,
        "TabM": base_marker,
        "(Prep)TabM": prep_marker,
        "RealTabPFN2.5": base_marker,
        "(Prep)RealTabPFN-2.5": prep_marker,
        },
        font_size=13,
        title_font_size=13,
        legend_font_size=14,
        tick_font_size=12,    
        save_path=f"{save_path}/final_model_performance_across_datasets_combine.pdf",

        # dataset_order="relative",
    )

    fig, ax = plot_model_performance_across_datasets(
        comb_results_use,
        model_col="ta_name",
        mode="median_centered_signed",
        value_label="Quantile-anchored normalized score",
        # normalization_center="third_best",
        # normalization_center_model="RandomForest",
        normalization_reference_models=[i for i in comb_results_use.ta_name.unique() if "prep" not in i and "Prep" not in i],  # define reference
        display_models=["LightGBM", "PrepLightGBM", "RealTabPFN2.5", "PrepRealTabPFN2.5", "LinearModel", "PrepLinearModel", "TabM", "PrepTabM"],
        legend_order=["LightGBM", "PrepLightGBM", "RealTabPFN2.5", "PrepRealTabPFN2.5", "LinearModel", "PrepLinearModel", "TabM", "PrepTabM"],
        title=None,
        sort_direction="worst_to_best",
        # sort_datasets_by_model="prep_LightGBM",
        sort_datasets_by_best_of_models=["PrepLightGBM", "PrepRealTabPFN2.5", "PrepLinearModel", "PrepTabM"],
        clip_good_side=True,
        bad_side_cap=1,
        good_side_cap=-2,
        show_model_averages=False,
        figsize=(16, 6),
        exclude_marker_groups=[base_marker],
        y_tick_labels={
            -2: "Improves over \n Best as much \n as Best \nover Top 75%",
            -1.0: "Best on\nTabArena",
            0.0: "Top\n75% on\nTabArena",
            1.0: "Top 50%\nor worse on\nTabArena",
        },
        model_color_groups={
        "LightGBM": ["LightGBM", "PrepLightGBM"],
        "LinearModel": ["LinearModel", "PrepLinearModel"],
        "TabM": ["TabM", "PrepTabM"],
        "RealTabPFN": ["RealTabPFN2.5", "PrepRealTabPFN2.5"],
        },
        model_markers={
        "LightGBM": base_marker,
        "PrepLightGBM": prep_marker,
        "LinearModel": base_marker,
        "PrepLinearModel": prep_marker,
        "TabM": base_marker,
        "PrepTabM": prep_marker,
        "RealTabPFN2.5": base_marker,
        "PrepRealTabPFN2.5": prep_marker,
        },
        font_size=13,
        title_font_size=13,
        legend_font_size=14,
        tick_font_size=12,    
        save_path=f"{save_path}/final_model_performance_across_datasets.pdf",

        # dataset_order="relative",
    )

    ### LOAD ABLATION RESULTS
    ablation_model_results = pd.read_csv(f"{ablation_base_path}/model_results.csv")
    ablation_hpo_results = pd.read_csv(f"{ablation_base_path}/hpo_results.csv")

    ablation_model_results = ablation_model_results.loc[ablation_model_results.fold==use_fold]

    ### Prepare AutoFeat results for comparison
    autofeat_comp_df = ablation_model_results.loc[ablation_model_results.method=="AutoFeatLinearModel_c1_BAG_L1",["dataset", "metric_error"]]
    autofeat_comp_df.rename(columns={"metric_error": "AutoFeat"}, inplace=True)
    base_lr_df = ablation_model_results.loc[ablation_model_results.method=="LinearModel_c1_BAG_L1",["dataset", "metric_error"]]
    base_lr_df.rename(columns={"metric_error": "LinearModel"}, inplace=True)

    autofeat_comp_df = autofeat_comp_df.merge(base_lr_df, on="dataset", suffixes=("_autofeat", "_baseLR"), how="outer")
    autofeat_comp_df.rename(columns={"metric_error_baseLR": "LinearModel"}, inplace=True)

    only_order2 = ablation_model_results.loc[ablation_model_results.method=="AutoFeatLinearModel_c2_BAG_L1",["dataset", "metric_error"]]
    # autofeat_comp_df.merge(base_lr_df, on="dataset", suffixes=("_autofeat", "_baseLR"))
    only_order2 = only_order2.rename(columns={"metric_error": "AutoFeat (2-order)"})

    autofeat_comp_df = autofeat_comp_df.merge(only_order2, on="dataset", how="outer")

    prep_lr = ta_results.loc[np.logical_and(
        ta_results.method=="PREP_LR (default)",
        ta_results.fold==use_fold),["dataset", "metric_error"]].rename(columns={"metric_error": "PrepLinearModel"})

    autofeat_comp_df = autofeat_comp_df.merge(prep_lr, on="dataset", suffixes=("", "_prep_lr"), how="outer")


    ### Prepare OpenFE results for comparison
    openfe_comp_df = ablation_model_results.loc[ablation_model_results.method=="OpenFELGBModel_c1_BAG_L1",["dataset", "metric_error"]]
    openfe_comp_df.rename(columns={"metric_error": "OpenFE"}, inplace=True)
    base_lgb_df = ta_results.loc[np.logical_and(
        ta_results.method=="GBM (default)",
        ta_results.fold==use_fold),["dataset", "metric_error"]].rename(columns={"metric_error": "LightGBM"})
    tuned_ensemble_lgb_df = ta_results.loc[np.logical_and(
        ta_results.method=="GBM (tuned + ensemble)",
        ta_results.fold==use_fold),["dataset", "metric_error"]].rename(columns={"metric_error": "LightGBM"})

    openfe_comp_df = openfe_comp_df.merge(base_lgb_df, on="dataset", suffixes=("_openfe", "_baseLGB"), how="outer")
    openfe_comp_df

    prep_lgb_df = ta_results.loc[np.logical_and(
        ta_results.method=="PREP_GBM (default)",
        ta_results.fold==use_fold),["dataset", "metric_error"]].rename(columns={"metric_error": "PrepLightGBM"})


    prep_lgb_tuned_df = ta_results.loc[np.logical_and(
        ta_results.method=="PREP_GBM (tuned)",
        ta_results.fold==use_fold),["dataset", "metric_error"]].rename(columns={"metric_error": "PrepLightGBM"})
    prep_lgb_tuned_ensemble_df = ta_results.loc[np.logical_and(
        ta_results.method=="PREP_GBM (tuned + ensemble)",
        ta_results.fold==use_fold),["dataset", "metric_error"]].rename(columns={"metric_error": "PrepLightGBM"})

    openfe_comp_df = openfe_comp_df.merge(prep_lgb_df, on="dataset", how="outer")

    ### PLOT autoFE RESULTS
    _ = boxplot_two_dataframes_pubready(
        df_left=autofeat_comp_df,
        left_baseline_col="LinearModel",
        left_competitor_cols=["AutoFeat", "PrepLinearModel"],
        df_right=openfe_comp_df,
        right_baseline_col="LightGBM",
        right_competitor_cols=["OpenFE", "PrepLightGBM"],
        left_labels=["Autofeat", "PrepLR"],
        right_labels=["OpenFE", "PrepLGB"],
        mode="relative",
        cap_left=[-0.25,1],
        cap_right=[-0.1,0.25],
        # titles=("Linear Models", "LightGBM"),
        horizontal=True,
        share_scale=False,
        save_path=f"{save_path}/autoFE_boxplots_subset.pdf",
        dpi=300,
        transparent=True,
        font_size=14.0,
        title_size=14.0,
        tick_size=10.0,
    )

    _ = boxplot_two_dataframes_pubready(
        dropna=False,
        df_left=autofeat_comp_df,
        left_baseline_col="LinearModel",
        left_competitor_cols=["AutoFeat", "PrepLinearModel"],
        df_right=openfe_comp_df,
        right_baseline_col="LightGBM",
        right_competitor_cols=["OpenFE", "PrepLightGBM"],
        left_labels=["Autofeat", "PrepLR"],
        right_labels=["OpenFE", "PrepLGB"],
        mode="relative",
        cap_left=[-0.5,1],
        cap_right=[-0.5,0.25],
        # titles=("Linear Models", "LightGBM"),
        horizontal=True,
        share_scale=False,
        save_path=f"{save_path}/autoFE_boxplots_full.pdf",
        dpi=300,
        figsize=(12,6),
        font_size=14.0,
        title_size=14.0,
        tick_size=12.0,
    )

    ### Training time increase analysis OpenFE
    train_time_base = ablation_model_results.loc[ablation_model_results.method=='LightGBM_c1_BAG_L1',["dataset", "time_train_s"]]
    train_time_openfe = ablation_model_results.loc[ablation_model_results.method=='OpenFELGBModel_c1_BAG_L1',["dataset", "time_train_s"]]
    train_time_prep = ta_results.loc[np.logical_and(
            ta_results.method=="PREP_GBM (default)",
            ta_results.fold==use_fold),["dataset", "time_train_s"]].rename(columns={"time_train_s": "time_train_s_prep"})

    train_times = pd.merge(
        train_time_base,
        train_time_openfe,
        on="dataset",
        suffixes=("_base", "_openfe")
    )

    train_times = pd.merge(
        train_times,
        train_time_prep,
        on="dataset"
    )

    increase_prep = (train_times["time_train_s_prep"]/train_times["time_train_s_base"]).describe().loc[["min", "25%", "50%", "75%", "max"]]
    increase_openfe = (train_times["time_train_s_openfe"]/train_times["time_train_s_base"]).describe().loc[["min", "25%", "50%", "75%", "max"]]

    # print("Training time increase PrepLightGBM vs LightGBM:")
    # pd.concat([increase_prep, increase_openfe], axis=1, keys=["PrepLightGBM", "OpenFE_LightGBM"]).to_latex(f"{save_path}/training_time_increase_prep_vs_openfe.tex")

    ### Training time increase analysis autofeat
    train_time_base = ablation_model_results.loc[ablation_model_results.method=='LinearModel_c1_BAG_L1',["dataset", "time_train_s"]]
    train_time_autofeat = ablation_model_results.loc[ablation_model_results.method=='AutoFeatLinearModel_c1_BAG_L1',["dataset", "time_train_s"]]
    train_time_prep_linear = ta_results.loc[np.logical_and(
            ta_results.method=="PREP_LR (default)",
            ta_results.fold==use_fold),["dataset", "time_train_s"]].rename(columns={"time_train_s": "time_train_s_prep"})

    train_times = pd.merge(
        train_time_base,
        train_time_autofeat,
        on="dataset",
        suffixes=("_base", "_autofeat")
    )

    train_times = pd.merge(
        train_times,
        train_time_prep_linear,
        on="dataset"
    )

    increase_prep_linear = (train_times["time_train_s_prep"]/train_times["time_train_s_base"]).describe().loc[["min", "25%", "50%", "75%", "max"]]
    increase_autofeat = (train_times["time_train_s_autofeat"]/train_times["time_train_s_base"]).describe().loc[["min", "25%", "50%", "75%", "max"]]

    # print("Training time increase PrepLinear vs LinearModel:")
    pd.concat([increase_prep_linear, increase_autofeat, increase_prep, increase_openfe], axis=1, keys=["PrepLinearModel", "AutoFeat_LinearModel", "PrepLightGBM", "OpenFE_LightGBM"]).round(1).astype(str).to_latex(f"{save_path}/training_time_increase_prep_vs_autoFE.tex")


    #############################################################################################################################################
    #############################################################################################################################################
    #############################################################################################################################################


    ### Prepare contribution to performance data
    setting_map = {
        "prep_LightGBM-ablation_c1_BAG_L1": "+Arithmetic",
        "prep_LightGBM-ablation_c2_BAG_L1": "RSFC",
        "prep_LightGBM-ablation_c3_BAG_L1": "Combine-TE",
        "prep_LightGBM-ablation_c4_BAG_L1": "OOF-TE",
        "prep_LightGBM-ablation_c5_BAG_L1": "GroupBy",
        "prep_LightGBM-ablation_c6_BAG_L1": "AbsoluteGroupBy",
        "prep_LightGBM-ablation_c7_BAG_L1": "OOF-TE-keepcat",
        "prep_LightGBM-ablation_c8_BAG_L1": "OOF-TE_w_GroupBy",
        "prep_LightGBM-ablation_c9_BAG_L1": "Arithmetic (2-order)",
        "prep_LightGBM-ablation_c10_BAG_L1": "Arithmetic (prod,ratio)",
        "prep_LightGBM-ablation_c11_BAG_L1": "Arithmetic (sum,diff)",
        "prep_LightGBM-ablation_c12_BAG_L1": "Cat-Pipeline",
        "prep_LightGBM-ablation_c13_BAG_L1": "Cat-Pipeline (keepcat)",
        "prep_LightGBM-ablation_c14_BAG_L1": "+OOF-TE",
        "prep_LightGBM-ablation_c15_BAG_L1": "+Combine-TE",
        "prep_LightGBM-ablation_c16_BAG_L1": "+GroupBy",
                }

    prep_lgb_df = ta_results.loc[np.logical_and(
        ta_results.method=="PREP_GBM (default)",
        ta_results.fold==use_fold),["dataset", "metric_error"]]
    prep_lgb_df["method"] = "+RSFC"
    prep_lgb_df.rename(columns={"+RSFC": "metric_error"}, inplace=True)
    prep_lgb_df = prep_lgb_df[["dataset", "method", "metric_error"]]

    prep_ablation_df = ablation_model_results.loc[ablation_model_results.method.apply(lambda x: x.startswith("prep_LightGBM-ablation")),["dataset", "method", "metric_error"]]
    prep_ablation_df["method"] = prep_ablation_df["method"].map(setting_map)
    prep_ablation_df = pd.concat([prep_ablation_df,prep_lgb_df], axis=0).reset_index(drop=True)
    
    df_components = pd.DataFrame(prep_ablation_df.pivot(
            index="dataset",
            columns="method",
            values="metric_error"
        ))
    df_components["Default LightGBM"] = base_lgb_df["LightGBM"].values
    df_components["TunedEnsembleLGB"] = tuned_ensemble_lgb_df["LightGBM"].values
    df_components["dataset"] = df_components.index
    df_components["+HPO"] = prep_lgb_tuned_ensemble_df["PrepLightGBM"].values

    worst_on_ta = comb_results_use.loc[comb_results_use.ta_name.apply(lambda x: "Prep" not in x).values].groupby(["dataset"]).apply(lambda x: x["metric_error"].loc[x["metric_error_val"].idxmax()])
    best_on_ta = comb_results_use.loc[comb_results_use.ta_name.apply(lambda x: "Prep" not in x).values].groupby(["dataset"]).apply(lambda x: x["metric_error"].loc[x["metric_error_val"].idxmin()])
    df_components = df_components.merge(best_on_ta.rename("BestOnTabArena"), left_index=True, right_on="dataset")
    df_components = df_components.merge(worst_on_ta.rename("WorstOnTabArena"), left_index=True, right_on="dataset")

    df_components_best_by_stage = df_components.copy()
    df_components_best_by_stage["+Arithmetic"] = df_components[["Default LightGBM","+Arithmetic"]].min(axis=1)
    df_components_best_by_stage["+OOF-TE"] = df_components[["Default LightGBM","+Arithmetic", "+OOF-TE"]].min(axis=1)
    df_components_best_by_stage["+Combine-TE"] = df_components[["Default LightGBM","+Arithmetic", "+OOF-TE", "+Combine-TE"]].min(axis=1)
    df_components_best_by_stage["+GroupBy"] = df_components[["Default LightGBM","+Arithmetic", "+OOF-TE", "+Combine-TE", "+GroupBy"]].min(axis=1)
    df_components_best_by_stage["+RSFC"] = df_components[["Default LightGBM","+Arithmetic", "+OOF-TE", "+Combine-TE", "+GroupBy", "+RSFC"]].min(axis=1)
    ### VERSION 1
    # boxplot_dataframe_pubready(
    #     df_components, 
    #     baseline_col="Default LightGBM", 
    #     competitor_cols=["+Arithmetic", "+OOF-TE", "+Combine-TE", "+GroupBy", "+RSFC"][::-1],
    #     dpi=300,
    #     transparent=True,
    #     font_size=10.0,          
    #     cap=[-0.1,.25],
    #     figsize = (8, 4),
    #     save_path=f"{save_path}/ablation_contribution_boxplot_v1.pdf",
    #     )

    boxplot_dataframe_pubready(
        df_components, 
        baseline_col="Default LightGBM", 
        competitor_cols=["+Arithmetic", "+OOF-TE", "+Combine-TE", "+GroupBy", "+RSFC", "+HPO"][::-1],
        dpi=300,
        transparent=True,
        font_size=14.0,
        title_size=14.0,
        point_size=14.0,      
        cap=[-0.1,.25],
        figsize = (8, 4),
        save_path=f"{save_path}/ablation_contribution_boxplot_withtuned_v1.pdf",
        )

    # boxplot_dataframe_pubready(
    #     df_components, 
    #     baseline_col="Default LightGBM", 
    #     competitor_cols=["+Arithmetic", "+OOF-TE", "+Combine-TE", "+GroupBy", "+RSFC", "TunedEnsembleLGB", "+HPO"][::-1],
    #     dpi=300,
    #     transparent=True,
    #     font_size=12.0,
    #     point_size=14.0,          
    #     cap=[-0.1,.25],
    #     figsize = (8, 4),
    #     save_path=f"{save_path}/ablation_contribution_boxplot_withbasetuned_v1.pdf",
    #     )

    # boxplot_dataframe_pubready(
    #     df_components_best_by_stage, 
    #     baseline_col="Default LightGBM", 
    #     competitor_cols=["+Arithmetic", "+OOF-TE", "+Combine-TE", "+GroupBy", "+RSFC"][::-1],
    #     dpi=300,
    #     transparent=True,
    #     font_size=10.0,          
    #     cap=[-0.1,.25],
    #     figsize = (8, 4),
    #     save_path=f"{save_path}/ablation_contribution_boxplot_bestbystage_v1.pdf",
    #     )

    # boxplot_dataframe_pubready(
    #     df_components, 
    #     baseline_col="Default LightGBM", 
    #     competitor_cols=["+Arithmetic", "Cat-Pipeline", "+RSFC"][::-1],
    #     dpi=300,
    #     transparent=True,
    #     font_size=10.0,          
    #     cap=[-0.1,.25],
    #     figsize = (8, 4),
    #     save_path=f"{save_path}/ablation_contribution_boxplot_cat_pipeline_v1.pdf",
    #     )

    # fig, ax, df_norm, plot_df, scores_wide = compare_methods_via_boxplots(
    #     df=df_components,
    #     dataset_col="dataset",
    #     lower_bound_col="BestOnTabArena",
    #     upper_bound_col="WorstOnTabArena",
    #     reference_col="Default LightGBM",
    #     # method_cols=["+Arithmetic", "+OOF-TE", "+CombineThenTE", "+GroupBy", "+RSFC", "TunedPrepLightGBM"][::-1],
    #     method_cols=["+Arithmetic", "+OOF-TE", "+Combine-TE", "+GroupBy", "+RSFC"][::-1],
    #     title="Normalized improvement vs lower reference",
    #     horizontal=True,
    #     # clip_norm=None,          # or (0, 1) if you want to clamp
    #     cap=(-0.4,.25),                # or cap=3 to clip extreme outliers in the plot only
    #     dpi=300,
    #     figsize=(8, 4),
    #     save_path=f"{save_path}/ablation_contribution_boxplot_v2.pdf",
    # )


    ### INTITIAL PLOT
    # Adapted call matching your current usage (same args / names reminded).
    # Assumes you already defined ablation_boxplot_colored_by_best exactly as in the last code block.

    # methods_keep = ["RSFC", "OOF-TE", "CombineThenTE", "Arithmetic", "GroupBy", "OOF-TE_w_GroupBy"]

    methods_keep = ["RSFC", "OOF-TE", "Combine-TE", "Arithmetic", "GroupBy"]
    # methods_keep = ["RSFC", "OOF-TE", "CombineThenTE", "Arithmetic", "GroupBy", "Cat-Pipeline", "PrepLightGBM"]
    # methods_keep = ["RSFC", "Arithmetic", "Cat-Pipeline"]
    # methods_keep = ["Cat-Pipeline", "Cat-Pipeline (keepcat)"]

    ### Ablation on decisions we made
    # Arithmetic - prioritize division and products
    # methods_keep = [i for i in prep_ablation_df.method.unique() if "Arithmetic" in i]
    # methods_keep = [i for i in prep_ablation_df.method.unique() if "GroupBy" in i]
    # methods_keep = [i for i in prep_ablation_df.method.unique() if "+" in i]
    methods_keep = ["+Arithmetic", "+OOF-TE", "+Combine-TE", "+GroupBy", "+RSFC"]

    fig, ax, merged_long, best_by_ds, color_by_method = ablation_boxplot_colored_by_best(
        prep_ablation_df=prep_ablation_df.loc[prep_ablation_df.method.isin(methods_keep)],
        base_df=base_lgb_df,
        dataset_col="dataset",
        method_col="method",
        metric_col="metric_error",
        base_metric_col="LightGBM",          # baseline column in base_lgb_df
        baseline_name="Base-LGB",
        winner_rule="methods_or_baseline",   # baseline can win (and wins ties)
        winner_atol=1e-12,
        winner_rtol=0.0,
        order_by="median_score",
        # plot scoring + limits
        mode="relative",
        cap=(-0.1, 0.2),

        # optional tie dropping from plot (off in your example)
        drop_equal_to_baseline=False,

        # style
        figsize=(10, 5),
        jitter=0.10,
        point_size=20.0,
        point_alpha=0.75,
        title=None,

        mean_marker_size_main=80,
        mean_marker_size_other=40,
        xlabel_fontsize=12.0,
        font_size=12,


        # saving
        save_path=f"{save_path}/prep_ablation_colored.pdf",
        dpi=300,
        transparent=True,
    )

    #############################################################################################################################################
    #############################################################################################################################################
    #############################################################################################################################################

    ### COMBINED TRIALS VS. PREP-ONLY TRIALS
    nonprep_res_path = {
        "RealTabPFN2.5": "~/.cache/tabarena/artifacts/tabarena-2025-11-12/methods/RealTabPFN-v2.5/results/model_results.parquet",
        "LightGBM": "~/.cache/tabarena/artifacts/tabarena-2025-06-12/methods/LightGBM/results/model_results.parquet",
        "TabM": "~/.cache/tabarena/artifacts/tabarena-2025-06-12/methods/TabM_GPU/results/model_results.parquet",
        # ""
        }


    def get_metric(df, ta_name, col_name):
        return (
            df.loc[
                (df.method_subtype == "tuned_ensemble") &
                (df.ta_name == ta_name),
                ["dataset", "metric_error"]
            ]
            .rename(columns={"metric_error": col_name})
        )

    dfs = [
        get_metric(comb_results_use_bar, "LightGBM", "LightGBM_baseModel"),
        get_metric(comb_results_use_bar, "PrepLightGBM", "LightGBM_prepModel"),
        get_metric(comb_results_use_bar, "(Prep)LightGBM", "LightGBM_combined"),

        get_metric(comb_results_use_bar, "TabM", "TabM_baseModel"),
        get_metric(comb_results_use_bar, "PrepTabM", "TabM_prepModel"),
        get_metric(comb_results_use_bar, "(Prep)TabM", "TabM_combined"),

        get_metric(comb_results_use_bar, "RealTabPFN2.5", "TabPFN-2.5_baseModel"),
        get_metric(comb_results_use_bar, "PrepRealTabPFN2.5", "TabPFN-2.5_prepModel"),
        get_metric(comb_results_use_bar, "(Prep)RealTabPFN-2.5", "TabPFN-2.5_combined"),
    ]

    df_combined_trials = dfs[0]
    for df in dfs[1:]:
        df_combined_trials = df_combined_trials.merge(
            df, on="dataset", how="inner"   # or "outer" if some datasets are missing
        )



    fig, (axL, axR), (scores_combined, scores_prep) = boxplot_models_combined_vs_tabprep(
        df=df_combined_trials,  # index = dataset names
        models=["LightGBM", "TabM", "TabPFN-2.5"],
        model_labels=["LightGBM", "TabM", "TabPFN-2.5"],
        baseline_cfg="baseModel",
        left_cfg="prepModel",
        right_cfg="combined",
        mode="log_ratio",
        lower_is_better=True,
        cap_left=(-0.3, 0.3),
        cap_right=(-0.15, 0.3),
        horizontal=True,
        share_scale=False,
        titles = (None, None),
        save_path=f"{save_path}/combined_trials_vs_prep_only.pdf",
        dpi=300,
        transparent=True,
        tick_size=10,
        font_size=12,
        title_size=12,
    )

