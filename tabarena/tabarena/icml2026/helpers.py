import pandas as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.metrics import log_loss, roc_auc_score, root_mean_squared_error

from autogluon.tabular.models import LGBModel, LinearModel, TabMModel, CatBoostModel
from tabarena.benchmark.models.wrapper.AutoGluon_class import AGSingleBagWrapper
from tabarena.benchmark.models.prep_ag.prep_lr.prep_lr_model import PrepLinearModel
from tabarena.benchmark.models.prep_ag.prep_lgb.prep_lgb_model import PrepLGBModel
from tabarena.benchmark.models.ag.tabpfnv2_5.tabpfnv2_5_model import RealTabPFNv25Model
from tabarena.benchmark.models.prep_ag.prep_tabpfnv2_5.prep_tabpfnv2_5_model import PrepRealTabPFNv25Model
from tabarena.benchmark.models.prep_ag.prep_tabm.prep_tabm_model import PrepTabMModel
from tabarena.benchmark.models.prep_ag.prep_cat.prep_cat_model import PrepCatBoostModel


from sklearn.preprocessing import TargetEncoder
from category_encoders import LeaveOneOutEncoder

from autogluon.features import ArithmeticFeatureGenerator, OOFTargetEncodingFeatureGenerator, CategoricalInteractionFeatureGenerator, GroupByFeatureGenerator, RandomSubsetTAFC

def run_experiment(X, y, X_test, y_test, model_name, prep_type, target_type, verbosity=0, num_bag_folds=0):
    if target_type == 'regression':
        metric = root_mean_squared_error
    elif target_type == 'binary':
        metric = lambda x,y: 1-roc_auc_score(x, y)
    else:
        metric = log_loss

    init_params = {"hyperparameters": {}}
    if model_name == "LR" and prep_type == "None":
        model_cls = LinearModel
    elif model_name == "LR" and prep_type != "None":
        model_cls = PrepLinearModel
    elif model_name == "PFN" and prep_type == "None":
        model_cls = RealTabPFNv25Model
    elif model_name == "PFN" and prep_type != "None":
        model_cls = PrepRealTabPFNv25Model
    elif model_name == "TABM" and prep_type == "None":
        model_cls = TabMModel
    elif model_name == "TABM" and prep_type != "None":
        model_cls = PrepTabMModel
    elif model_name == "CAT" and prep_type == "None":
        model_cls = CatBoostModel
    elif model_name == "CAT" and prep_type != "None":
        model_cls = PrepCatBoostModel
    elif model_name in ["GBM", "GBM-OHE"] and prep_type == "None":
        model_cls = LGBModel
        # init_params['hyperparameters'] = {'n_estimators': 10000}
    elif model_name in ["GBM", "GBM-OHE"] and prep_type != "None":
        model_cls = PrepLGBModel
        # init_params = {"problem_type": target_type, "hyperparameters": {'n_estimators': 10000}}
        # if model_name == "GBM-OHE":
        #     init_params["hyperparameters"].update({'max_cat_to_onehot': 1000000})

    # if prep_type == "OOF-TE":
    #     params = {"ag.prep_params": [['OOFTargetEncodingFeatureGenerator', {}]], 'ag.prep_params.passthrough_types': {"invalid_raw_types": ["category", "object"]}}
    # else:
    #     params = {}
    params = {}
    # X_used = X.copy()
    # X_test_used = X_test.copy()
    if prep_type == "TE":
        agfg = TargetEncoder(random_state=42)
        cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
        X_prep = pd.DataFrame(agfg.fit_transform(X[cat_cols].astype("object"), y), index=X.index, columns=[f'{i}_te' for i in cat_cols])
        X_test_prep = pd.DataFrame(agfg.transform(X_test[cat_cols].astype("object")), index=X_test.index, columns=[f'{i}_te' for i in cat_cols])
        X_prep = pd.concat([X.drop(cat_cols, axis=1), X_prep], axis=1)
        X_test_prep = pd.concat([X_test.drop(cat_cols, axis=1), X_test_prep], axis=1)
        X_used, X_test_used = X_prep, X_test_prep
    elif prep_type == "LOO":
        agfg = LeaveOneOutEncoder(random_state=42)
        cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
        X_prep = pd.DataFrame(agfg.fit_transform(X[cat_cols].astype("object"), y), index=X.index, columns=[f'{i}_loo' for i in cat_cols])
        X_test_prep = pd.DataFrame(agfg.transform(X_test[cat_cols].astype("object")), index=X_test.index, columns=[f'{i}_loo' for i in cat_cols])
        X_prep = pd.concat([X.drop(cat_cols, axis=1), X_prep], axis=1)
        X_test_prep = pd.concat([X_test.drop(cat_cols, axis=1), X_test_prep], axis=1)
        X_used, X_test_used = X_prep, X_test_prep
    elif prep_type == "OOF-TE":
        params = {"ag.prep_params": [['OOFTargetEncodingFeatureGenerator', {}]], 'ag.prep_params.passthrough_types': {"invalid_raw_types": ["category", "object"]}}
        X_used = X.copy()
        X_test_used = X_test.copy()
    elif prep_type == "OOF-TE-Smooth20":
        params = {"ag.prep_params": [['OOFTargetEncodingFeatureGenerator', {"alpha": 20}]], 'ag.prep_params.passthrough_types': {"invalid_raw_types": ["category", "object"]}}
        X_used = X.copy()
        X_test_used = X_test.copy()
    elif prep_type == "OOF-TE-APPEND":
        params = {"ag.prep_params": [['OOFTargetEncodingFeatureGenerator', {"passthrough": True}]]}
        X_used = X.copy()
        X_test_used = X_test.copy()
    elif prep_type in ["CATINT"]:
        params = {"ag.prep_params": [[["CategoricalInteractionFeatureGenerator", {"max_order": 2, "passthrough": True}]]]}
        X_used = X.copy()
        X_test_used = X_test.copy()
    elif prep_type in ["CATINT_OOFTE"]:
        params = {"ag.prep_params": [["CategoricalInteractionFeatureGenerator", {"max_order": 2, "passthrough": True}], ['OOFTargetEncodingFeatureGenerator', {"passthrough": False}]]}
        X_used = X.copy()
        X_test_used = X_test.copy()
    elif prep_type in ["CATINT_OOFTE_APPEND"]:
        params = {"ag.prep_params": [["CategoricalInteractionFeatureGenerator", {"max_order": 2, "passthrough": True}], ['OOFTargetEncodingFeatureGenerator', {"passthrough": True}]]}
        X_used = X.copy()
        X_test_used = X_test.copy()
    elif prep_type in ["CATINT_TE", "CATINT_LOO"]:

        if prep_type == "CATINT_OOFTE":
            pass
        elif prep_type in ["CATINT_TE", "CATINT_LOO"]:
            agfg = CategoricalInteractionFeatureGenerator(target_type=target_type, passthrough=False, max_order=2)
            X_used = pd.concat([X, agfg.fit_transform(X, y)], axis=1)
            X_test_used = pd.concat([X_test, agfg.transform(X_test)], axis=1)
            cat_cols = X_used.select_dtypes(include=['category', 'object']).columns.tolist()
            if prep_type == "CATINT_TE":
                agfg = TargetEncoder(random_state=42)
            elif prep_type == "CATINT_LOO":
                agfg = LeaveOneOutEncoder(random_state=42, handle_missing="value", handle_unknown="value")
            X_prep = pd.DataFrame(agfg.fit_transform(X_used[cat_cols].astype("object"), y), index=X_used.index, columns=[f'{i}_te' for i in cat_cols])
            X_test_prep = pd.DataFrame(agfg.transform(X_test_used[cat_cols].astype("object")), index=X_test_used.index, columns=[f'{i}_te' for i in cat_cols])
            X_used = pd.concat([X_used.drop(cat_cols, axis=1), X_prep], axis=1)
            X_test_used = pd.concat([X_test_used.drop(cat_cols, axis=1), X_test_prep], axis=1)
    elif prep_type == "DROP-CAT":
        cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
        X_used = X.drop(cat_cols, axis=1)
        X_test_used = X_test.drop(cat_cols, axis=1)
    elif prep_type == "DROP-NUM":
        num_cols = X.select_dtypes(include=['number']).columns.tolist()
        X_used = X.drop(num_cols, axis=1)
        X_test_used = X_test.drop(num_cols, axis=1)
    elif prep_type in ["2-ARITHMETIC", "3-ARITHMETIC", "4-ARITHMETIC"]:
        agfg = ArithmeticFeatureGenerator(target_type=target_type, random_state=42, max_order=int(prep_type[0]), passthrough=True, max_new_feats=1500)
        X_used = agfg.fit_transform(X, y)
        X_test_used = agfg.transform(X_test)
    elif prep_type == "RSTAFC-allorder":
        params = {"ag.prep_params": [['RandomSubsetTAFC', {"passthrough": True, "min_subset_size": 1, "max_subset_size": None}]]}
        X_used = X.copy()
        X_test_used = X_test.copy()
    elif prep_type == "RSTAFC-1order":
        params = {"ag.prep_params": [['RandomSubsetTAFC', {"passthrough": True, "min_subset_size": 1, "max_subset_size": 1}]]}
        X_used = X.copy()
        X_test_used = X_test.copy()
    elif prep_type == "RSTAFC":
        params = {"ag.prep_params": [['RandomSubsetTAFC', {"passthrough": True, "min_subset_size": 1, "max_subset_size": 2}]]}
        X_used = X.copy()
        X_test_used = X_test.copy()
    elif prep_type == "RSTAFC-noround":
        params = {"ag.prep_params": [['RandomSubsetTAFC', {"passthrough": True, "min_subset_size": 1, "round_numerical": 10}]]}
        X_used = X.copy()
        X_test_used = X_test.copy()
    elif prep_type == "RSTAFC-noround-1order":
        params = {"ag.prep_params": [['RandomSubsetTAFC', {"passthrough": True, "min_subset_size": 1, "max_subset_size": 1, "round_numerical": 10}]]}
        X_used = X.copy()
        X_test_used = X_test.copy()
    elif prep_type == "REL-GROUPBY":
        params = {"ag.prep_params": [['GroupByFeatureGenerator', {"passthrough": True}]]}
        X_used = X.copy()
        X_test_used = X_test.copy()
    elif prep_type == "REL-GROUPBY-keepMean":
        params = {"ag.prep_params": [['GroupByFeatureGenerator', {"passthrough": True, "aggregations": ("mean","pct_rank"), "drop_basic_groupby_when_relative": False}]]}
        X_used = X.copy()
        X_test_used = X_test.copy()
    elif prep_type == "REL-GROUPBY-noPCT":
        params = {"ag.prep_params": [['GroupByFeatureGenerator', {"passthrough": True, "aggregations": ("mean",)}]]}
        X_used = X.copy()
        X_test_used = X_test.copy()
    elif prep_type == "REL-GROUPBY-meansubtract":
        params = {"ag.prep_params": [['GroupByFeatureGenerator', {"passthrough": True, "aggregations": ("mean",), "relative_ops": ("diff", )}]]}
        X_used = X.copy()
        X_test_used = X_test.copy()
    elif prep_type == "MEAN-GROUPBY":
        params = {"ag.prep_params": [['GroupByFeatureGenerator', {"passthrough": True, 
                                                                  "aggregations": ("mean",), 
                                                                  "relative_to_aggs": (), 
                                                                  "relative_ops": (), 
                                                                  "drop_basic_groupby_when_relative":True
                                                                  }]]}
        X_used = X.copy()
        X_test_used = X_test.copy()

    else:
        X_used, X_test_used = X, X_test

    # model = model_cls(**init_params)
    # print(X_used.iloc[0])
    if model_name == "PFN":
        num_bag_folds = 0  # PFN does refit anyway
    params['ag_args_ensemble'] = {}
    params['ag_args_ensemble']["fold_fitting_strategy"] = "sequential_local"
    model = AGSingleBagWrapper(model_cls,
                                params, 
                                problem_type=target_type, 
                                eval_metric='roc_auc' if target_type=='binary' else ('rmse' if target_type=='regression' else 'log_loss'), 
                                fit_kwargs= {'num_bag_folds': num_bag_folds, 'verbosity': verbosity},
                                )
    model.fit(X=X_used, y=y)
    if target_type == 'regression':
        preds = model.predict(X_test_used)
    elif target_type == 'binary':
        preds = model.predict_proba(X_test_used)
        if isinstance(preds, pd.DataFrame) and preds.shape[1] == 2:
            preds = preds.iloc[:, 1]
        elif isinstance(preds, np.ndarray) and preds.shape[1] == 2:
            preds = preds[:, 1]
    else:
        preds = model.predict_proba(X_test_used)
    score = metric(y_test, preds)

    return preds, score, X_used

def plot_results_heatmap(
    df,
    normalize_per_column=False,
    cmap="RdYlGn_r",
    annotate=True,
    fmt="{:.3f}",
    fontsize=14,          # bigger overall
    label_fontsize=16,    # axis labels
    tick_fontsize=14,     # tick labels
    figsize=None,
    dpi=300,
    savepath=None,
    transparent=False
):
    """
    Publication-ready heatmap:
    - Larger text
    - LaTeX-like fonts (Computer Modern via mathtext; no external LaTeX needed)
    - No title, no colorbar
    """

    encodings = df.index.to_list()
    models = df.columns.to_list()
    data = df.to_numpy(dtype=float)

    if figsize is None:
        figsize = (1.10 * len(models) + 2.0, 0.75 * len(encodings) + 1.4)

    # LaTeX-like typography WITHOUT requiring a LaTeX installation
    plt.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "text.usetex": False,          # critical: avoids your latex crash
        "font.family": "serif",          # let matplotlib choose
        "mathtext.fontset": "cm",        # Computer Modern for math
        "font.size": fontsize,
        "axes.labelsize": label_fontsize,
        "xtick.labelsize": tick_fontsize,
        "ytick.labelsize": tick_fontsize,
        "axes.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Clean spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Heatmap rendering
    if normalize_per_column:
        cmap_obj = plt.get_cmap(cmap)
        rgba = np.zeros((*data.shape, 4))

        for j in range(data.shape[1]):
            col = data[:, j]
            norm = Normalize(vmin=np.nanmin(col), vmax=np.nanmax(col))
            rgba[:, j, :] = cmap_obj(norm(col))

        ax.imshow(rgba, aspect="auto", interpolation="nearest")
    else:
        ax.imshow(data, cmap=cmap, aspect="auto", interpolation="nearest")

    # Axes ticks/labels (use mathtext bold for a LaTeX-like look)
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(encodings)))
    ax.set_xticklabels(models)
    ax.set_yticklabels(encodings)

    ax.set_xlabel(r"$\mathbf{Model}$")
    ax.set_ylabel(r"$\mathbf{Encoding}$")

    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=0, pad=4)

    # Subtle cell grid lines
    ax.set_xticks(np.arange(-.5, len(models), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(encodings), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotations with contrast-aware color (global normalization case)
    if annotate:
        if not normalize_per_column:
            norm = Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
            cmap_obj = plt.get_cmap(cmap)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    val = data[i, j]
                    rgba = cmap_obj(norm(val))
                    lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                    color = "black" if lum > 0.6 else "white"
                    ax.text(j, i, fmt.format(val),
                            ha="center", va="center",
                            fontsize=fontsize, color=color)
        else:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    ax.text(j, i, fmt.format(data[i, j]),
                            ha="center", va="center",
                            fontsize=fontsize, color="black")

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", transparent=transparent)

    plt.show()
    return fig, ax


def plot_fe_paired_dots_oneplot(
    df_delta,
    drop_baseline=True,
    baseline_name="No FE",
    fontsize=14,
    label_fontsize=16,
    tick_fontsize=14,
    figsize=None,
    dpi=300,
    savepath=None,
    legend_ncol=3,
):
    """
    Single combined plot:
    For each FE method (row), show per-model Δ vs baseline on one axis.
    Draw a faint segment from 0 to Δ to emphasize change from baseline.
    """

    if drop_baseline and baseline_name in df_delta.index:
        df_plot = df_delta.drop(index=baseline_name)
    else:
        df_plot = df_delta.copy()

    fe_methods = df_plot.index.to_list()
    models = df_plot.columns.to_list()
    data = df_plot.to_numpy()

    n_rows, n_models = data.shape

    if figsize is None:
        # Good default for papers; adjust if many FE methods
        figsize = (7.2, 0.55 * n_rows + 1.6)

    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.size": fontsize,
        "axes.labelsize": label_fontsize,
        "xtick.labelsize": tick_fontsize,
        "ytick.labelsize": tick_fontsize,
    })

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)

    # y positions (top-to-bottom)
    y = np.arange(n_rows)
    ax.set_yticks(y)
    ax.set_yticklabels(fe_methods)
    ax.invert_yaxis()

    # zero line
    ax.axvline(0, linewidth=1.2, alpha=0.8)

    # nice x-limits with padding
    vmax = np.nanmax(np.abs(data))
    pad = 0.08 * vmax if vmax > 0 else 0.01
    ax.set_xlim(-vmax - pad, vmax + pad)

    # plot each model with a small y-offset so points don't overlap
    # (works well when you have a handful of models)
    offsets = np.linspace(-0.18, 0.18, n_models)

    for j, model in enumerate(models):
        vals = df_plot[model].to_numpy()

        # faint baseline-to-delta segments
        for i, v in enumerate(vals):
            ax.plot([0, v], [y[i] + offsets[j], y[i] + offsets[j]],
                    linewidth=1.2, alpha=0.35)

        # dots at Δ
        ax.scatter(vals, y + offsets[j], s=50, label=model, zorder=3)

    ax.set_xlabel(r"$\Delta$ score vs. baseline (No FE)")
    ax.grid(axis="x", alpha=0.25)

    # clean spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(
        frameon=False,
        ncol=min(legend_ncol, len(models)),
        bbox_to_anchor=(0.5, 1.02),
        loc="lower center",
        columnspacing=1.2,
        handletextpad=0.4
    )

    if savepath:
        fig.savefig(savepath, bbox_inches="tight")

    plt.show()
    return fig, ax
