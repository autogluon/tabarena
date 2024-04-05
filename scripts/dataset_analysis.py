import math
from pathlib import Path
import warnings

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from tabrepo import EvaluationRepository
from scripts import load_context
from scripts.baseline_comparison.plot_utils import figure_path, save_latex_table


def order_clustermap(df, allow_nan=True):
    # TODO we could just call scipy
    if allow_nan:
        df = df.fillna(1)
    cg = sns.clustermap(df)
    row_indices = cg.dendrogram_row.reordered_ind
    col_indices = cg.dendrogram_col.reordered_ind
    plt.close()
    return df.index[row_indices], df.columns[col_indices]


def index(name):
    config_number = name.split("-")[-1]
    if "c" in config_number:
        return None
    else:
        return int(config_number)


def generate_dataset_info_latex(repo: EvaluationRepository, expname_outdir: str):
    metadata = repo._df_metadata.copy()
    assert len(metadata) == len(repo.datasets())

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        metadata['problem_type'] = ''
        metadata['problem_type'][metadata['NumberOfClasses'] == 2] = 'binary'
        metadata['problem_type'][metadata['NumberOfClasses'] > 2] = 'multiclass'
        metadata['problem_type'][metadata['NumberOfClasses'] == 0] = 'regression'

    metadata_min = metadata[["tid", "name", "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses", 'problem_type']]
    metadata_min_sorted = metadata_min.sort_values(by=["name"])
    metadata_latex = metadata_min_sorted.copy()

    max_name_length = 20

    metadata_latex.columns = ['Task ID', 'name', 'n', 'f', 'C', 'Problem Type']
    metadata_latex['Task ID'] = metadata_latex['Task ID'].astype(str)
    metadata_latex['n'] = metadata_latex['n'].astype(int)
    metadata_latex['f'] = metadata_latex['f'].astype(int)
    metadata_latex['f'] = metadata_latex['f'] - 1  # Original counts the label column, remove to get the feature count
    metadata_latex['C'] = metadata_latex['C'].astype(int)

    datasets_statistics = metadata_latex.describe(percentiles=[.05, .1, .25, .5, .75, .9, .95])
    datasets_statistics = datasets_statistics.drop(columns='C')
    datasets_statistics = datasets_statistics.drop(index='count')
    datasets_statistics = datasets_statistics.round()
    datasets_statistics = datasets_statistics.astype(int)

    save_latex_table(df=datasets_statistics, title=f"datasets_statistics", show_table=True, save_prefix=expname_outdir)

    metadata_latex['name'] = metadata_latex['name'].apply(lambda x: x[:max_name_length])

    problem_types = ['binary', 'multiclass', 'regression']
    latex_kwargs = dict(
        index=False,
    )

    # Separate by problem types and into batches to ensure each table fits on one page
    for p in problem_types:
        metadata_latex_p = metadata_latex[metadata_latex['Problem Type'] == p]
        num_datasets = len(metadata_latex_p)

        metadata_latex_v2 = metadata_latex_p.drop(columns='Problem Type')

        vertical = math.ceil(num_datasets / 2)
        metadata_left = metadata_latex_v2.iloc[:vertical].reset_index(drop=True)
        metadata_right = metadata_latex_v2.iloc[vertical:].reset_index(drop=True)
        metadata_left['n'] = metadata_left['n'].astype(str)
        metadata_left['f'] = metadata_left['f'].astype(str)
        metadata_left['C'] = metadata_left['C'].astype(str)

        metadata_right['n'] = metadata_right['n'].astype(str)
        metadata_right['f'] = metadata_right['f'].astype(str)
        metadata_right['C'] = metadata_right['C'].astype(str)
        if p == 'regression':
            metadata_left = metadata_left.drop(columns='C')
            metadata_right = metadata_right.drop(columns='C')

        metadata_combined = pd.concat([metadata_left, metadata_right], axis=1)
        metadata_combined = metadata_combined.fillna('')
        save_latex_table(df=metadata_combined, title=f"datasets_{p}", show_table=True,
                         latex_kwargs=latex_kwargs, save_prefix=expname_outdir)


def generate_dataset_analysis(repo, expname_outdir: str):
    num_models_to_plot = 20
    title_size = 20
    figsize = (20, 7)

    # Fails with: ValueError: Unknown format code 'f' for object of type 'str'
    generate_dataset_info_latex(repo=repo, expname_outdir=expname_outdir)

    zsc = repo._zeroshot_context

    df = zsc.df_configs_ranked.copy()

    df["framework"] = df["framework"].map({
        "FTTransformer_c1_BAG_L1": "FTTransformer_r1_BAG_L1",
        # "FTTransformer_c2_BAG_L1": "FTTransformer_r2_BAG_L1",
        # "FTTransformer_c3_BAG_L1": "FTTransformer_r3_BAG_L1",
        "TabPFN_c1_BAG_L1": "TabPFN_r1_BAG_L1",
        # "TabPFN_c2_BAG_L1": "TabPFN_r2_BAG_L1",
        # "TabPFN_c3_BAG_L1": "TabPFN_r3_BAG_L1",

    }).fillna(df["framework"])
    config_regexp = "(" + "|".join([str(x) for x in range(4)]) + ")"
    df = df[df.framework.str.contains(f"r{config_regexp}_BAG_L1")]
    df.framework = df.framework.str.replace("NeuralNetTorch", "MLP")

    metric = "metric_error"
    df_pivot = df.pivot_table(
        index="framework", columns="tid", values=metric
    )
    df_rank = df_pivot.rank() / len(df_pivot)
    df_rank.index = [x.replace("_BAG_L1", "").replace("_r", "_").replace("_", "-") for x in df_rank.index]
    # shorten framework names
    #df_rank.index = [x.replace("ExtraTrees", "ET").replace("CatBoost", "CB").replace("LightGBM", "LGBM").replace("NeuralNetFastAI", "MLP").replace("RandomForest", "RF").replace("_BAG_L1", "").replace("_r", "_").replace("_", "-") for x in df_rank.index]
    df_rank.index = [
        x
        # .replace("ExtraTrees", "ET")
        # .replace("CatBoost", "CB")
        # .replace("LightGBM", "LGBM")
        .replace("NeuralNetTorch", "MLP")
        # .replace("RandomForest", "_r", "_")
        # .replace("_", "-")
        for x in df_rank.index]

    df_rank = df_rank[[index(name) is not None and index(name) < num_models_to_plot for name in df_rank.index]]

    ordered_rows, ordered_cols = order_clustermap(df_rank)
    df_rank = df_rank.loc[ordered_rows]
    df_rank = df_rank[ordered_cols]
    df_rank.columns.name = "dataset"

    # task-model rank
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=300)
    ax = axes[0]
    cmap = matplotlib.colormaps.get_cmap('RdYlGn_r')
    cmap.set_bad("black")
    sns.heatmap(
        df_rank, cmap=cmap, vmin=0, vmax=1, ax=ax,
    )
    ax.set_xticks([])
    ax.set_xlabel("Datasets", fontdict={'size': title_size})
    ax.set_title("Ranks of models per dataset", fontdict={'size': title_size})

    # model-model correlation
    ax = axes[1]
    sns.heatmap(
        df_rank.T.corr(), cmap="vlag", vmin=-1, vmax=1, ax=ax,
    )
    ax.set_title("Model rank correlation", fontdict={'size': title_size})

    # runtime figure
    df = zsc.df_configs_ranked
    ax = axes[2]
    df['method'] = df.apply(lambda x: x["framework"].split("_")[0], axis=1)
    df['method'] = df['method'].map({"NeuralNetTorch": "MLP"}).fillna(df["method"])
    df_grouped = df[["method", "tid", "time_train_s"]].groupby(["method", "tid"]).max()["time_train_s"].sort_values()
    df_grouped = df_grouped.reset_index(drop=False)
    df_grouped["group_index"] = df_grouped.groupby("method")["time_train_s"].cumcount()
    df_grouped["group_index"] += 1

    sns.lineplot(
        data=df_grouped,
        x="group_index",
        y="time_train_s",
        hue="method",
        hue_order=sorted(list(df_grouped["method"].unique())),
        linewidth=3,
        palette=[  # category10 color palette
            '#1f77b4',
            '#ff7f0e',
            '#2ca02c',
            '#d62728',
            '#9467bd',
            '#8c564b',
            '#e377c2',
            '#7f7f7f',
            '#bcbd22',
            '#17becf',
        ],
        ax=ax,
    )
    ax.set_yscale('log')
    ax.grid()
    ax.legend()
    ax.set_xlabel("Datasets", fontdict={'size': title_size})
    ax.set_ylabel("Training runtime (s)", fontdict={'size': title_size})
    ax.set_title("Training runtime distribution", fontdict={'size': title_size})

    plt.tight_layout()
    fig_save_path = figure_path(prefix=expname_outdir) / f"data-analysis.pdf"
    plt.savefig(fig_save_path)
    plt.show()

    plot_train_time_deep_dive(df, expname_outdir=expname_outdir)


def plot_train_time_deep_dive(df, expname_outdir: str):
    df = df.copy(deep=True)
    title_size = 20
    figsize = (26, 7)
    fig, axes = plt.subplots(1, 4, figsize=figsize, dpi=300)

    # runtime max stats
    index_above_time_limit = df["time_train_s"] >= 2800
    proportion_of_models_reaching_time_limit = index_above_time_limit.mean()
    num_models_reaching_time_limit = index_above_time_limit.sum()
    models_by_family_reaching_time_limit = df.loc[index_above_time_limit].value_counts("method")
    models_by_config_reaching_time_limit = df.loc[index_above_time_limit].value_counts("framework")
    datasets_reaching_time_limit = df.loc[index_above_time_limit].value_counts("dataset")
    datasets_type_reaching_time_limit = df.loc[index_above_time_limit].value_counts(["dataset", "method"])

    latex_kwargs = dict(
        index=False,
    )

    print(f"Percentage of models reaching time limit: {proportion_of_models_reaching_time_limit * 100:.2f}%")
    print(f"Number of models reaching time limit: {num_models_reaching_time_limit} (out of {len(df)})")
    save_latex_table(
        df=datasets_type_reaching_time_limit.reset_index(drop=False),
        title="early_stopping_counts_dataset_family",
        save_prefix=expname_outdir,
        show_table=True,
        latex_kwargs=latex_kwargs,
    )

    save_latex_table(
        df=datasets_reaching_time_limit.reset_index(drop=False),
        title="early_stopping_counts_dataset",
        save_prefix=expname_outdir,
        show_table=True,
        latex_kwargs=latex_kwargs,
    )
    save_latex_table(
        df=models_by_family_reaching_time_limit.reset_index(drop=False),
        title="early_stopping_counts_family",
        save_prefix=expname_outdir,
        show_table=True,
        latex_kwargs=latex_kwargs,
    )

    print(datasets_type_reaching_time_limit.reset_index(drop=False).to_markdown(index=False))
    print(datasets_reaching_time_limit.reset_index(drop=False).to_markdown(index=False))
    print(models_by_family_reaching_time_limit.reset_index(drop=False).to_markdown(index=False))

    df_sorted_by_time = df.sort_values(by=["time_train_s"]).reset_index(drop=True)
    df_sorted_by_time["index"] = df_sorted_by_time.index + 1
    df_sorted_by_time["time_train_s_cumsum"] = df_sorted_by_time["time_train_s"].cumsum()
    # df_sorted_by_time["time_train_s_cumsum"] = df_sorted_by_time["time_train_s_cumsum"] / df_sorted_by_time["time_train_s_cumsum"].max()
    df_sorted_by_time["index"] = df_sorted_by_time["index"] / df_sorted_by_time["index"].max()
    df_sorted_by_time["group_time_train_s_cumsum"] = df_sorted_by_time.groupby("method")["time_train_s"].cumsum()
    df_sorted_by_time["group_index"] = df_sorted_by_time.groupby("method")["time_train_s"].cumcount()
    df_sorted_by_time["group_index_max"] = df_sorted_by_time["method"].map(df_sorted_by_time.value_counts("method"))
    df_sorted_by_time["group_index"] = df_sorted_by_time["group_index"] / df_sorted_by_time["group_index_max"]

    ax = axes[0]
    sns.lineplot(
        data=df_sorted_by_time,
        x="index",
        y="time_train_s",
        linewidth=3,
        ax=ax,
    )
    ax.set_yscale('log')
    ax.grid()
    ax.hlines(3600, xmin=0, xmax=df_sorted_by_time["index"].max(), color="black", label="3600 Seconds", ls="--")
    ax.legend()
    ax.set_xlabel("Configs (Proportion)", fontdict={'size': title_size})
    ax.set_ylabel("Training runtime (s)", fontdict={'size': title_size})
    ax.set_title("Config runtime distribution", fontdict={'size': title_size})
    plt.tight_layout()

    ax = axes[1]
    sns.lineplot(
        data=df_sorted_by_time,
        x="group_index",
        y="time_train_s",
        hue="method",
        hue_order=sorted(list(df["method"].unique())),
        linewidth=3,
        palette=[  # category10 color palette
            '#1f77b4',
            '#ff7f0e',
            '#2ca02c',
            '#d62728',
            '#9467bd',
            '#8c564b',
            '#e377c2',
            '#7f7f7f',
            '#bcbd22',
            '#17becf',
        ],
        ax=ax,
    )
    ax.set_yscale('log')
    ax.grid()
    ax.hlines(3600, xmin=0, xmax=df_sorted_by_time["group_index"].max(), color="black", label="3600 Seconds", ls="--")
    ax.legend()
    ax.set_xlabel("Family configs (Proportion)", fontdict={'size': title_size})
    ax.set_ylabel("Training runtime (s)", fontdict={'size': title_size})
    ax.set_title("Family runtime distribution", fontdict={'size': title_size})
    plt.tight_layout()

    ax = axes[2]
    sns.lineplot(
        data=df_sorted_by_time,
        x="index",
        y="time_train_s_cumsum",
        linewidth=3,
        ax=ax,
    )
    ax.set_yscale('log')
    ax.grid()
    ax.set_xlabel("Configs (Proportion)", fontdict={'size': title_size})
    ax.set_ylabel("Cumulative training runtime (s)", fontdict={'size': title_size})
    ax.set_title("Cumulative config runtime distribution", fontdict={'size': title_size})
    plt.tight_layout()

    ax = axes[3]
    sns.lineplot(
        data=df_sorted_by_time,
        x="group_index",
        y="group_time_train_s_cumsum",
        hue="method",
        hue_order=sorted(list(df["method"].unique())),
        linewidth=3,
        palette=[  # category10 color palette
            '#1f77b4',
            '#ff7f0e',
            '#2ca02c',
            '#d62728',
            '#9467bd',
            '#8c564b',
            '#e377c2',
            '#7f7f7f',
            '#bcbd22',
            '#17becf',
        ],
        ax=ax,
    )
    ax.set_yscale('log')
    ax.grid()
    ax.legend()
    ax.set_xlabel("Family configs (Proportion)", fontdict={'size': title_size})
    ax.set_ylabel("Cumulative training runtime (s)", fontdict={'size': title_size})
    ax.set_title("Cumulative family runtime distribution", fontdict={'size': title_size})
    plt.tight_layout()

    fig_save_path = figure_path(prefix=expname_outdir) / f"data-analysis-runtime.pdf"
    plt.savefig(fig_save_path)
    plt.show()


if __name__ == "__main__":
    repo_version = "D244_F3_C1530_200"
    repo: EvaluationRepository = load_context(version=repo_version)
    expname_outdir = str(Path("output") / repo_version)
    generate_dataset_analysis(repo=repo, expname_outdir=expname_outdir)
