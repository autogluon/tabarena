import os
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")


def get_significance_dataset(df_use, method="wilcoxon", alpha=0.05, verbose=False, direction="max"):
    ### Get accuracy p-values
    p_values = {}


    df_use_mean = df_use.groupby(["dataset_name", "model_name"]).mean().unstack()[0].loc[df_use.dataset_name.unique()]
    
    dataset_names = list(df_use_mean.index)
    
    for dataset_name in dataset_names:
        # print(dataset_name)
        p_values[dataset_name] = {}
        # try:
        if direction == "min":
            best_model = df_use_mean.columns[df_use_mean.loc[dataset_name].argmin()]
        elif direction == "max":
            best_model = df_use_mean.columns[df_use_mean.loc[dataset_name].argmax()]
        
        for model_name in df_use.model_name.unique():
            if model_name == best_model:
                # print(dataset_name,model_name)
                p_values[dataset_name][model_name] = 1.
            else:
                # print(dataset_name,model_name)
                # Example performance metrics over 10 repeats for two models
                best_results = df_use.loc[np.logical_and(df_use.dataset_name==dataset_name,
                              df_use.model_name==best_model
                             ),0].values
                best_results[best_results<0] = 0
                
                curr_model_results = df_use.loc[np.logical_and(df_use.dataset_name==dataset_name,
                              df_use.model_name==model_name
                             ),0].values
                curr_model_results[curr_model_results<0] = 0
            
                p_values[dataset_name][model_name] = get_significance(best_results, curr_model_results, direction=direction, method=method, verbose=verbose)
        # except:
        #     p_values[dataset_name] = {model_name: 0. for model_name in df_use.model_name.unique()}
    
    significance_df = pd.DataFrame(p_values).transpose()
    return significance_df


def holm_bonferroni_correction(p_values):
    """
    Apply Holm-Bonferroni correction to a list of p-values.
    :param p_values: List of p-values.
    :return: Adjusted p-values and rejection decisions.
    """
    sorted_indices = np.argsort(p_values)
    sorted_p_values = np.array(p_values)[sorted_indices]
    adjusted_p_values = np.zeros_like(sorted_p_values)
    m = len(p_values)
    
    for i, p in enumerate(sorted_p_values):
        adjusted_p_values[i] = min((m - i) * p, 1.0)
    
    # Reorder to original order
    adjusted_p_values = adjusted_p_values[np.argsort(sorted_indices)]
    
    # Determine rejection
    reject = adjusted_p_values < 0.05
    
    return list(zip(p_values, adjusted_p_values, reject))

def get_significance(best_results, curr_model_results, method="wilcoxon", alpha=0.05, verbose=False, direction="max"):

    if direction =="min" and np.mean(best_results)>=np.mean(curr_model_results):
        p_value = 2
    elif direction =="max" and np.mean(best_results)<=np.mean(curr_model_results):
        p_value = 2
    elif direction =="max" and np.all(best_results<=curr_model_results):
        p_value = 2
    
    else:
        # Perform the paired t-test
        if method == "ttest":
            t_statistic, p_value = stats.ttest_rel(best_results, curr_model_results)
        elif method == "wilcoxon":
            t_statistic, p_value = stats.wilcoxon(best_results, curr_model_results) #     w_stat, p_value_w
        # elif method == "wilcoxon-corrected":
        #     t_statistic, p_value = stats.wilcoxon(best_results, curr_model_results) #     w_stat, p_value_w
            # Apply Holm-Bonferroni correction
            # reject, p_value, _, _ = multipletests(p_value, alpha=0.05, method='holm')
        elif method == "kruskal-wallis":
            statistic, p_value = stats.kruskal(best_results, curr_model_results)
        elif method == "ftest":
            statistic, p_value = stats.f_oneway(best_results, curr_model_results)
        elif method == "tabred":
            sub_series_mean = np.mean(best_results)
            sub_series_std = np.std(best_results)
            if direction == "min":
                thresh = sub_series_mean+sub_series_std
                p_value = (np.mean(curr_model_results)<thresh)*2
                
            elif direction == "max":
                thresh = sub_series_mean-sub_series_std
                p_value = (np.mean(curr_model_results)>thresh)*2
            

    p_value_one_tailed = p_value / 2
    criterion = p_value_one_tailed < alpha
            
    if verbose:
        print(f"T-statistic: {t_statistic}")
        print(f"P-value: {p_value}")

    # Interpret the result
    if criterion:
        if verbose:
            print("There is a statistically significant difference between the two models.")
    else:
        if verbose:
            print("There is no statistically significant difference between the two models.")
    if verbose:
        print("----------------------------------------------------------------------------------")

    return p_value
 
def get_per_dataset_tables(df_results: pd.DataFrame, save_path: Path):


    # df_results["method"] = df_results["method"].map({
    #         "AutoGluon_v130_bq_4h8c": "AutoGluon 1.3 (4h)",
    #         "MNCA_GPU (default)": "MNCA (default)",
    #         "MNCA_GPU (tuned)": "MNCA (tuned)",
    #         "MNCA_GPU (tuned + ensemble)": "MNCA (tuned + ensemble)",
    #         "REALMLP_GPU (default)": "REALMLP (default)",
    #         "REALMLP_GPU (tuned)": "REALMLP (tuned)",
    #         "REALMLP_GPU (tuned + ensemble)": "REALMLP (tuned + ensemble)",
    #         "TABM_GPU (default)": "REALMLP (default)",
    #         "REALMLP_GPU (tuned)": "REALMLP (tuned)",
    #         "REALMLP_GPU (tuned + ensemble)": "REALMLP (tuned + ensemble)",
            
            
    #         "TabPFN_c1_BAG_L1": "TABPFN (default)",
    #         "RealMLP_c1_BAG_L1": "REALMLP (default)",
    #         "ExplainableBM_c1_BAG_L1": "EBM (default)",
    #         "FTTransformer_c1_BAG_L1": "FT_TRANSFORMER (default)",
    #         "TabPFNv2_c1_BAG_L1": "TABPFNV2 (default)",
    #         "TabICL_c1_BAG_L1": "TABICL (default)",
    #         'TabDPT_c1_BAG_L1': "TABDPT (default)",
    #         'TabM_c1_BAG_L1': "TABM (default)",
    #         'ModernNCA_c1_BAG_L1': "MNCA (default)",
    #     }).fillna(df_results["method"])


    use_methods_ordered = [   
        # default
        'RF (default)', 'XT (default)','XGB (default)','GBM (default)','CAT (default)','EBM (default)',
        'FASTAI (default)','NN_TORCH (default)','REALMLP_GPU (default)','TABM_GPU (default)','MNCA_GPU (default)',
        'TABPFNV2_GPU (default)','TABDPT_GPU (default)','TABICL_GPU (default)',
        'LR (default)','KNN (default)',
        
        ######### tuned
        'RF (tuned)','XT (tuned)','XGB (tuned)','GBM (tuned)','CAT (tuned)','EBM (tuned)',
        'FASTAI (tuned)','NN_TORCH (tuned)','REALMLP_GPU (tuned)','TABM_GPU (tuned)','MNCA_GPU (tuned)',
        'TABPFNV2_GPU (tuned)',# 'TABDPT (tuned)', 'TABICL (tuned)',
        'LR (tuned)','KNN (tuned)',
        
        ######## tuned  ensemble
        'RF (tuned + ensemble)','XT (tuned + ensemble)','XGB (tuned + ensemble)','GBM (tuned + ensemble)','CAT (tuned + ensemble)','EBM (tuned + ensemble)',
        'FASTAI (tuned + ensemble)','NN_TORCH (tuned + ensemble)','REALMLP_GPU (tuned + ensemble)','TABM_GPU (tuned + ensemble)','MNCA_GPU (tuned + ensemble)',
        'TABPFNV2_GPU (tuned + ensemble)',# 'TABDPT (tuned + ensemble)', 'TABICL (tuned + ensemble)',
        'LR (tuned + ensemble)','KNN (tuned + ensemble)',

        ######### AutoGluon baseline
        'AutoGluon 1.4 (extreme, 4h)',
        ######### AutoGluon baseline
        # "Portfolio-N200 (ensemble) (4h)"
    ]

    df_use = df_results.loc[df_results["method"].apply(lambda x: x in use_methods_ordered)]
    df_use_fold = df_use.loc[df_use["fold"]==0]

    for_sig = df_use[["method", "dataset", "metric_error"]]
    for_sig.columns=["model_name", "dataset_name", 0]

    significance_df = get_significance_dataset(for_sig, method="wilcoxon", alpha=0.05, verbose=False, direction="min")
    significance_default = get_significance_dataset(for_sig.loc[for_sig["model_name"].apply(lambda x: "default" in x)], method="wilcoxon", alpha=0.05, verbose=False, direction="min")
    significance_tuned = get_significance_dataset(for_sig.loc[for_sig["model_name"].apply(lambda x: "(tuned)" in x)], method="wilcoxon", alpha=0.05, verbose=False, direction="min")
    significance_tuned_ensemble = get_significance_dataset(for_sig.loc[for_sig["model_name"].apply(lambda x: "tuned + ensemble" in x)], method="wilcoxon", alpha=0.05, verbose=False, direction="min")

    from tabarena.nips2025_utils.fetch_metadata import load_task_metadata
    df_meta = load_task_metadata()
    df_meta["n_folds"] = 3
    df_meta["n_features"] = (df_meta["NumberOfFeatures"] - 1).astype(int)
    df_meta["n_samples_test_per_fold"] = (df_meta["NumberOfInstances"] / df_meta["n_folds"]).astype(int)
    df_meta["n_samples_train_per_fold"] = (df_meta["NumberOfInstances"] - df_meta["n_samples_test_per_fold"]).astype(int)

    df_meta['can_run_tabpfnv2'] = np.logical_and(np.logical_and(df_meta["n_samples_train_per_fold"] <= 10000, df_meta["n_features"] <= 500), df_meta['NumberOfClasses'] <= 10)
    df_meta['can_run_tabicl'] = np.logical_and(np.logical_and(df_meta["n_samples_train_per_fold"] <= 100000, df_meta["n_features"] <= 500), df_meta['NumberOfClasses'] > 0)

    can_run_tabpfnv2 = dict(df_meta[["dataset", "can_run_tabpfnv2"]].values)
    can_run_tabicl = dict(df_meta[["dataset", "can_run_tabicl"]].values)

    datasets_dict = {}
    for dataset_name in df_use["dataset"].unique():
        df_dat = df_use.loc[df_use["dataset"]==dataset_name]
        imputed_methods = df_dat.loc[df_dat.imputed==True,'method'].unique()

        if np.unique(df_dat["metric"])[0]=="roc_auc":
            df_dat.loc[:, "metric_error"] = 1-df_dat["metric_error"]
            metric_dir = "max"
        else:
            metric_dir = "min"

        df_mean = df_dat[["method", "metric_error"]].groupby("method").mean().sort_values("metric_error").loc[use_methods_ordered]
        df_std = df_dat[["method", "metric_error"]].groupby("method").std().sort_values("metric_error").loc[use_methods_ordered]

        df_mean_raw = df_mean.copy()
        df_std_raw = df_std.copy()

        df_mean = df_mean["metric_error"]
        df_std = df_std["metric_error"]

        if df_mean.apply(lambda x: str(x).find('.')).max()==1:
            df_std = df_std.round(3)
            df_mean = df_mean.round(3)
            df_mean = df_mean.apply(lambda x: format(x, '.3f'))
            df_std = df_std.apply(lambda x: format(x, '.3f'))
        elif df_mean.apply(lambda x: str(x).find('.')).max()==2:
            df_std = df_std.round(2)
            df_mean = df_mean.round(2)
            df_mean = df_mean.apply(lambda x: format(x, '.2f'))
            df_std = df_std.apply(lambda x: format(x, '.2f'))
        elif df_mean.apply(lambda x: str(x).find('.')).max()==3:
            df_std = df_std.round(1)
            df_mean = df_mean.round(1)
            df_mean = df_mean.apply(lambda x: format(x, '.1f'))
            df_std = df_std.apply(lambda x: format(x, '.1f'))
        elif df_mean.apply(lambda x: str(x).find('.')).max()==4:
            df_std = df_std.round(1)
            df_mean = df_mean.round(1)
            df_mean = df_mean.apply(lambda x: format(x, '.1f'))
            df_std = df_std.apply(lambda x: format(x, '.1f'))
        elif df_mean.apply(lambda x: str(x).find('.')).max()==5:
            df_std = df_std.round(0).astype(int)#.astype(str)
            df_mean = df_mean.round(0).astype(int)#.astype(str)
            df_mean = df_mean.apply(lambda x: format(x, '.1f'))
            df_std = df_std.apply(lambda x: format(x, '.1f'))
        elif df_mean.apply(lambda x: str(x).find('.')).max()==6:
            df_std = (df_std/10).round(0).astype(int).astype(str)
            df_mean = (df_mean/10).round(0).astype(int).astype(str)

        df_mean = df_mean.to_frame()
        df_std = df_std.to_frame()

        df_latex = df_mean + " $\\pm$ " + df_std
        df_latex.columns = [dataset_name]
        for method in imputed_methods:
            df_latex.loc[method] = '-'

        if metric_dir == "min":
            df_latex.loc[df_mean_raw.idxmin(),dataset_name] = r"\textcolor{green!50!black}{" + df_latex.loc[df_mean_raw.idxmin(),dataset_name] + "}"
        elif metric_dir == "max":
            df_latex.loc[df_mean_raw.idxmax(),dataset_name] = r"\textcolor{green!50!black}{" + df_latex.loc[df_mean_raw.idxmax(),dataset_name] + "}"

        is_best = df_latex.apply(lambda x: significance_df.loc[dataset_name,x.index]>0.05)
        df_latex.loc[:, dataset_name] = [r"\textbf{"+score+"}" if is_best.loc[name,dataset_name] else score  for name, score in zip(df_latex.index, df_latex[dataset_name])]

        df_latex_def = df_latex.loc[[True  if "default" in i else False for i in df_latex.index]]
        df_latex_tuned = df_latex.loc[[True  if "(tuned)" in i else False for i in df_latex.index]]
        df_latex_tuned_ensemble = df_latex.loc[[True  if "tuned + ensemble" in i else False for i in df_latex.index]]

        is_best_def = df_latex_def.apply(lambda x: significance_default.loc[dataset_name,x.index]>0.05)
        is_best_tuned = df_latex_tuned.apply(lambda x: significance_tuned.loc[dataset_name,x.index]>0.05)
        is_best_tuned_ensemble = df_latex_tuned_ensemble.apply(lambda x: significance_tuned_ensemble.loc[dataset_name,x.index]>0.05)

        df_latex_def.loc[:, dataset_name] = [r"\underline{"+score+"}" if is_best_def.loc[name,dataset_name] else score  for name, score in zip(df_latex_def.index, df_latex_def[dataset_name])]
        df_latex_tuned.loc[:, dataset_name] = [r"\underline{"+score+"}" if is_best_tuned.loc[name,dataset_name] else score  for name, score in zip(df_latex_tuned.index, df_latex_tuned[dataset_name])]
        df_latex_tuned_ensemble.loc[:, dataset_name] = [r"\underline{"+score+"}" if is_best_tuned_ensemble.loc[name,dataset_name] else score  for name, score in zip(df_latex_tuned_ensemble.index, df_latex_tuned_ensemble[dataset_name])]

        # df_latex = pd.concat([df_latex_def, df_latex_tuned, df_latex_tuned_ensemble, df_latex.loc[df_latex.index=="AutoGluon 1.3 (4h)"]], axis=0)
        # df_latex.loc[:, dataset_name] = [r"\textbf{"+score+"}" if is_best_tuned_ensemble.loc[name,dataset_name] else score  for name, score in zip(df_latex.index, df_latex[dataset_name])]

        df_latex_final = pd.merge(df_latex_def.rename(index=lambda s: s.split(" (")[0]), 
                                    df_latex_tuned.rename(index=lambda s: s.split(" (")[0]),
                                    left_index=True, right_index=True, how="left"
                                    ).merge(df_latex_tuned_ensemble.rename(index=lambda s: s.split(" (")[0])
                                    , left_index=True, right_index=True, how="left"
                                    )
        df_latex_final.loc["AutoGluon 1.4 (extreme, 4h)"] = ["-", "-", df_latex.loc["AutoGluon 1.4 (extreme, 4h)", dataset_name]]
        # df_latex_final.loc["Portfolio-N200 (ensemble) (4h)"] = [df_latex.loc["Portfolio-N200 (ensemble) (4h)", dataset_name], "", ""]
        df_latex_final = df_latex_final.fillna("-")
        df_latex_final.columns = ["Default", "Tuned", "Tuned + Ens."]

        # df_latex_final = df_latex_final.replace({"0.": "."}, regex=True)

        replace_dict = {
            "RF": "RF",
            "XT": "ExtraTrees",
            "XGB": "XGBoost",
            "GBM": "LightGBM",
            "CAT": "CatBoost",
            "EBM": "EBM",

            "FASTAI": "FastAIMLP",
            "NN_TORCH": "TorchMLP",
            "REALMLP_GPU": "RealMLP",
            "TABM_GPU": "TabM",
            "MNCA_GPU": "MNCA",

            'TABPFNV2_GPU': 'TabPFNv2',
            'TABDPT_GPU': 'TabDPT',
            'TABICL_GPU': 'TabICL',

            'LR': 'Linear',
            'KNN': 'KNN',
            "AutoGluon 1.4 (extreme, 4h)": "AutoGluon",
            "Portfolio-N200 (ensemble) (4h)": "Portfolio"
        }

        df_latex_final.index = pd.Series(df_latex_final.index).replace(replace_dict)

        # if not can_run_tabpfnv2[dataset_name]:
        #     df_latex_final.loc["TabPFNv2"] = ["-", "-", "-"]    
        # if not can_run_tabicl[dataset_name]:
        #     df_latex_final.loc["TabICL"] = ["-", "-", "-"]    
        
        df_latex_final.index.name = None
        datasets_dict[dataset_name.replace("_", r"\_")] = df_latex_final.copy()

    output_file = str(save_path / "per_dataset_tables.tex")
    per_col    = 2                  # 2 columns per row
    per_page   = 6                  # 3 rows × 2 columns = 6 subtables per figure
    sub_width  = 0.48               # width for each subtable (2 across)
    sub_height = ""  # fixed height for each subtable

    metrics_used = dict(df_use[["dataset", "metric"]].drop_duplicates().values)
    metrics_used = {key.replace("_", r"\_"): "AUC" if value=="roc_auc" else ("logloss" if value=="log_loss" else "rmse") for key, value in metrics_used.items()}

    items = list(datasets_dict.items())

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for page_start in range(0, len(items), per_page):
            page_items = items[page_start:page_start + per_page]

            f.write(r"\begin{table}[htb]" + "\n")
            f.write(r"  \centering" + "\n")
            if page_start == 0:
                f.write(r"\caption{\textbf{Performance Per Dataset.}  \
                    We show the average predictive performance per dataset with the standard deviation over folds. \
                    We show the performance for the default hyperparameter configuration (\texttt{Default}), for the model after tuning (\texttt{Tuned}), and for the ensemble after tuning (\texttt{Tuned + Ens.}). \
                    We highlight the best-performing methods with significance on three levels:  \
                    (1) \textcolor{green!50!black}{Green}: The best performing method on average; \
                    (2) \textbf{Bold}: Methods that are not significantly worse than the best method on average, based on a Wilcoxon Signed-Rank test for paired samples with Holm-Bonferroni correction and $\alpha=0.05$. \
                    (3) \underline{Underlined}: Methods that are not significantly worse than the best method in the same pipeline regime (\texttt{Default}, \texttt{Tuned}, or \texttt{Tuned + Ens.}), based on a Wilcoxon Signed-Rank test for paired samples with Holm-Bonferroni correction and $\alpha=0.05$. We exclude AutoGluon for significance tests in the \texttt{Tuned + Ens.} regime.}" + "\n\n")


            # Build 3 rows of 2 columns each
            for row_start in range(0, len(page_items), per_col):
                row = page_items[row_start:row_start + per_col]
                for idx, (name, df) in enumerate(row):
                    if name == 'HR\\_Analytics\\_Job\\_Change\\_of\\_Data\\_Scientists':
                        print_name = 'HR\\_Analytics\\_Job\\_Change'
                    else:
                        print_name = name
                    # compute column format: index, quad gap, then one 'r' per data column
                    n_data = df.shape[1]
                    col_fmt = "l@{\\quad}" + "l" * n_data

                    f.write(f"  \\begin{{subtable}}[t]{{{sub_width}\\textwidth}}\n")
                    f.write(r"    \centering" + "\n")
                    f.write(r"    \scriptsize" + "\n")

                    # caption at the top
                    if metrics_used[name]=="AUC":
                        arrow = r"$\uparrow$"
                    else:
                        arrow = r"$\downarrow$"
                    f.write(f"    \\caption*{{{print_name} ({metrics_used[name]} {arrow})}}" + "\n")
                    f.write(r"    \vspace{-1ex}")
                    f.write(f"    \\label{{tab:{page_start + row_start + idx + 1}}}\n")

                    # fixed-height minipage, top-aligned
                    f.write(f"    \\begin{{minipage}}[t][{sub_height}][t]{{\\linewidth}}\n")
                    f.write(r"      \vspace{0pt}")

                    # render the table with the index separated by a quad
                    latex_table = df.to_latex(
                        index=True,
                        escape=False,
                        column_format=col_fmt
                    )
                    for line in latex_table.splitlines():
                        f.write("      " + line + "\n")

                    f.write(r"    \end{minipage}" + "\n")
                    f.write(r"  \end{subtable}")
                    # horizontal filler between columns
                    if idx < len(row) - 1:
                        f.write(" \\hfill")
                    f.write("\n")

                # small vertical gap between rows
                f.write(r"  \medskip" + "\n\n")
            
            # overall caption & label for this figure
            # f.write(
            #     rf"  \caption{{Datasets {page_start+1}–{page_start+len(page_items)}}}" + "\n"
            # )
            # f.write(
            #     rf"  \label{{fig:datasets_{page_start+1}_{page_start+len(page_items)}}}" + "\n"
            # )
            f.write(r"\end{table}" + "\n\n")

    print(f"Saved per-dataset tables to {output_file}")