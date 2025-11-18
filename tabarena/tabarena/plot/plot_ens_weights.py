from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# TODO: NaN color as black?
# TODO: Title?
# TODO: return type, don't return `plt`
# TODO: Auto calculate required figsize based on # of rows and # of cols
def create_heatmap(df: pd.DataFrame, xlabel: str = "Config", ylabel: str = "Task", figsize: tuple[float, float] = (12, 10), include_mean: bool = False):
    """
    Creates a heatmap visualizing the ensemble weights across a suite of tasks and configs.

    Parameters
    ----------
    df : pd.DataFrame
        Ensemble weights. Each row represents a task, and each column represents a config.
        Each row should sum to 1.
        Index should be the names of the tasks
        Column names should be the names of the configs
    xlabel : str, default "Config"
    ylabel : str, default "Task"
    figsize : tuple[float, float], default (12, 10)
    include_mean : bool, default False
        If True, will add a row at the bottom with label "mean" representing the mean of the config weights across all tasks.
        NaN values are considered 0 for the purposes of calculating the mean.

    Returns
    -------
    plt

    """
    plt.figure(figsize=figsize)

    if include_mean:
        df_mean = df.sum().to_frame(name="mean").T / len(df)
        df = pd.concat([df, df_mean])

    # Create mask for zero values
    mask = df == 0

    # Create heatmap with specific styling (Blues)
    heatmap = sns.heatmap(
        df,
        cmap="Blues",
        vmin=-0.1,
        vmax=1,
        center=0.5,
        annot=True,
        annot_kws={'size': 20},
        fmt='.2f',
    )

    # Hide text for zero values (but still colored, otherwise fully white)
    for text in heatmap.texts:
        x, y = text.get_position()
        if mask.iloc[int(y), int(x)]:
            text.set_text('')

            # Customize the colorbar
    cbar = heatmap.collections[0].colorbar
    cbar.set_ticks([0, 0.5, 1, ])
    cbar.set_ticklabels(['0', '0.5', '1'])

    # Set labels with specific font sizes
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    # plt.tight_layout()

    # Adjust tick label font sizes
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    return plt


if __name__ == "__main__":
    d_dict = {}
    d_dict["task1"] = {
        "m1": 0.3,
        "m2": 0.5,
        "m3": 0.2,
    }
    d_dict["task2"] = {
        "m1": 0.6,
        "m3": 0.4,
    }
    df_input = pd.DataFrame(d_dict).T
    p = create_heatmap(df=df_input)

    # p.show()
    p.savefig("test_fig_ens_weights")
