from __future__ import annotations

from ast import literal_eval
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_family_proportion(df, method="Portfolio-N200 (ensemble) (4h)", save_prefix: str = None, show: bool = True, hue_order: list = None):
    df_family = df[df["method"] == method].copy()
    df_family = df_family[df_family["fold"] == 0]
    portfolios = list(df_family["config_selected"].values)
    if len(portfolios) > 0 and not isinstance(portfolios[0], list):
        portfolios_lst = [literal_eval(portfolio) for portfolio in portfolios]
    else:
        portfolios_lst = portfolios

    from collections import defaultdict
    type_count = defaultdict(int)
    type_count_family = defaultdict(int)
    type_count_per_iter = dict()
    type_count_family_per_iter = dict()

    n_iters = 25
    for i in range(n_iters):
        type_count_per_iter[i] = defaultdict(int)
        type_count_family_per_iter[i] = defaultdict(int)
        for portfolio in portfolios_lst:
            if len(portfolio) <= i:
                continue
            name = portfolio[i]
            family = name.split('_', 1)[0]
            type_count[name] += 1
            type_count_family[family] += 1
            type_count_per_iter[i][name] += 1
            type_count_family_per_iter[i][family] += 1

    families = sorted(list(type_count_family.keys()))

    import copy
    type_count_cumulative = dict()
    type_count_family_cumulative = dict()
    type_count_cumulative[0] = copy.deepcopy(type_count_per_iter[0])
    type_count_family_cumulative[0] = copy.deepcopy(type_count_family_per_iter[0])
    for i in range(1, n_iters):
        type_count_cumulative[i] = copy.deepcopy(type_count_cumulative[i-1])
        for k in type_count_per_iter[i].keys():
            type_count_cumulative[i][k] += type_count_per_iter[i][k]
        type_count_family_cumulative[i] = copy.deepcopy(type_count_family_cumulative[i-1])
        for k in type_count_family_per_iter[i].keys():
            type_count_family_cumulative[i][k] += type_count_family_per_iter[i][k]

    data = []
    for i in range(n_iters):
        data.append([type_count_family_per_iter[i][f] for f in families])
    data_cumulative = []
    for i in range(n_iters):
        data_cumulative.append([type_count_family_cumulative[i][f] for f in families])


    data_df = pd.DataFrame(data=data, columns=families)
    data_df = data_df.div(data_df.sum(axis=1), axis=0) * 100
    data_df2 = data_df.stack().reset_index(name='Model Frequency (%) at Position').rename(columns={'level_1': 'Model', 'level_0': 'Portfolio Position'})
    data_df2["Portfolio Position"] += 1

    data_cumulative_df = pd.DataFrame(data=data_cumulative, columns=families)
    data_cumulative_df = data_cumulative_df.div(data_cumulative_df.sum(axis=1), axis=0) * 100
    data_cumulative_df2 = data_cumulative_df.stack().reset_index(name='Cumulative Model Frequency (%)').rename(columns={'level_1': 'Model', 'level_0': 'Portfolio Position'})
    data_cumulative_df2["Portfolio Position"] += 1

    plot_per_position = False
    if plot_per_position:
        nrows = 2
    else:
        nrows = 1

    fig, axes = plt.subplots(nrows, 1, sharey=False, sharex=True, figsize=(3.5, 3), dpi=300, layout="constrained")

    if plot_per_position:
        legend = False
        sns.histplot(
            data_df2,
            x="Portfolio Position",
            weights="Model Frequency (%) at Position",
            # stat="percent",
            hue="Model",
            hue_order=hue_order,
            multiple="stack",
            # palette="light:m_r",
            palette="pastel",
            edgecolor=".3",
            linewidth=.5,
            discrete=True,
            ax=axes[0],
            # legend=False,
        )
        axes[0].set(ylabel="Model Frequency (%) at Position")
        axes[0].set_xlim([0, n_iters+1])
        axes[0].set_ylim([0, 100])
        sns.move_legend(axes[0], "upper left")
    else:
        legend = True
        ax = axes
        axes = [ax, ax]

    sns.histplot(
        data_cumulative_df2,
        x="Portfolio Position",
        weights="Cumulative Model Frequency (%)",
        # stat="percent",
        hue="Model",
        hue_order=hue_order,
        multiple="stack",
        # palette="light:m_r",
        palette="pastel",
        edgecolor=".3",
        linewidth=.5,
        discrete=True,
        ax=axes[1],
        legend=legend,
    )
    axes[1].set(ylabel="Cumulative Model Frequency (%)")
    axes[1].set_xlim([0, n_iters+1])
    axes[1].set_ylim([0, 100])

    if not plot_per_position:
        sns.move_legend(axes[1], "upper right")

    fig.suptitle(f"Model Family Presence in Portfolio by Training Order")

    if save_prefix:
        fig_path = Path(save_prefix)
        fig_path.mkdir(parents=True, exist_ok=True)
        fig_save_path = fig_path / f"portfolio-model-presence.png"
        plt.savefig(fig_save_path)
        fig_save_path = fig_path / f"portfolio-model-presence.pdf"
        plt.savefig(fig_save_path)
    if show:
        plt.show()
