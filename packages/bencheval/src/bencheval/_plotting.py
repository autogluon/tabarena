from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


class PlottingMixin:
    """Leaderboard / results plotting helpers (mixed into BenchmarkEvaluator)."""

    @staticmethod
    def plot_winrate_matrix(
        winrate_matrix: pd.DataFrame,
        save_path: str | None,
        title: str | None = None,
    ):
        import matplotlib.pyplot as plt

        z = winrate_matrix.copy()
        z = (z * 100).round().astype("Int64")

        n_rows, n_cols = z.shape

        # Scale figure size with matrix size, while keeping a sensible minimum
        fig_w = max(11.1, 0.55 * n_cols + 3)
        fig_h = max(9.0, 0.45 * n_rows + 2)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

        # PRGn-like colormap
        im = ax.imshow(z.astype(float).to_numpy(), cmap="PRGn", vmin=0, vmax=100)

        # Ticks / labels
        ax.set_xticks(range(n_cols))
        ax.set_yticks(range(n_rows))
        ax.set_xticklabels(
            z.columns,
            fontsize=16,
            rotation=330,
            ha="right",
            va="bottom",
            rotation_mode="anchor",
        )
        ax.set_yticklabels(z.index, fontsize=16)
        ax.tick_params(axis="x", pad=-2)
        ax.tick_params(axis="x", which="both", length=0)
        ax.tick_params(axis="y", which="both", length=0)

        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("Model B: Loser", fontsize=18)
        ax.set_ylabel("Model A: Winner", fontsize=18)

        cmap = plt.get_cmap("PRGn")
        norm = plt.Normalize(vmin=0, vmax=100)

        values = z.to_numpy()
        for i in range(n_rows):
            for j in range(n_cols):
                val = values[i, j]
                if pd.isna(val):
                    text = ""
                    text_color = "black"
                else:
                    text = f"{int(val)}"
                    r, g, b, _ = cmap(norm(float(val)))
                    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    text_color = "black" if luminance > 0.5 else "white"

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=16,
                    color=text_color,
                )

        # Colorbar
        from matplotlib.ticker import FuncFormatter

        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_title("Win Rate", fontsize=18, pad=12, loc="left")
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}%"))

        # Clean look
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")

        # Match imshow orientation expectations
        ax.set_ylim(n_rows - 0.5, -0.5)

        if title is not None:
            # Center horizontally on the *grid* (i.e. the imshow axes), the
            # same way ``ax.set_xlabel("Model B: Loser", ...)`` is centered.
            # ``ax.get_position()`` returns the axes bounding box in figure
            # coordinates, so the midpoint of its x-range is the matrix
            # grid's horizontal center — independent of the colorbar that
            # sits to the right.
            fig.canvas.draw()
            ax_bbox = ax.get_position()
            x_axes_center = (ax_bbox.x0 + ax_bbox.x1) / 2
            fig.suptitle(
                title,
                fontsize=20,
                fontweight="bold",
                x=x_axes_center,
            )

        if save_path is not None:
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    def plot_critical_diagrams(
        self, results_per_task, save_path: str | None = None, show: bool = False, reverse: bool = False
    ):
        import matplotlib.pyplot as plt

        with plt.rc_context({"text.usetex": False}):
            from autorank import autorank
            from autorank._util import cd_diagram

            plt.rcParams.update({"font.size": 12})

            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(1, 1, 1)

            data = results_per_task.pivot_table(index=self.task_col, columns=self.method_col, values="rank")
            result = autorank(data, alpha=0.05, verbose=False, order="ascending", force_mode="nonparametric")

            try:
                _ = cd_diagram(result, reverse=reverse, ax=ax, width=6)
            except KeyError:
                print("Not enough methods to generate cd_diagram, skipping...")
                return

            # plt.tight_layout()  # cuts off text
            if save_path is not None:
                parent_dir = str(Path(save_path).parent)
                os.makedirs(parent_dir, exist_ok=True)
                plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            if show:
                plt.show()

    def plot_dataset_metric_distribution(
        self,
        results_per_task: pd.DataFrame,
        dataset: str,
        metric_col: str | None = None,
        *,
        sort_by: str = "median",  # {"median", "mean"}
        ascending: bool = True,  # lower is better for error-like metrics
        kind: str = "violin",  # {"violin", "box", "bar"}
        show_points: str | bool = "outliers",  # plotly: True, False, "all", "outliers"
        log_y: bool = False,
        max_methods: int | None = 50,  # cap for readability
        save_path: str | None = None,
        title: str | None = None,
    ):
        """Plot a single dataset's metric distribution across methods.

        Parameters
        ----------
        results_per_task : pd.DataFrame
            Output of `self.compute_results_per_task(data=data)` (possibly with seeds).
            Must contain at least: [self.task_col, self.method_col, metric_col].
        dataset : str
            Dataset/task name to filter on (value from `self.task_col`).
        metric_col : str | None, default None
            Column to plot. If None, defaults to `self.error_col`.
            Common choices: self.error_col, "rank", "improvability", "loss_rescaled".
        sort_by : {"median","mean"}, default "median"
            How to sort methods on the x-axis.
        ascending : bool, default True
            Whether to sort ascending (typical for error-like metrics).
        kind : {"violin","box","bar"}, default "violin"
            Plot type. If the dataset has only one value per method, it will fall back to "bar".
        show_points : str | bool, default "outliers"
            Whether to show individual points for violin/box.
        log_y : bool, default False
            Whether to log-scale the y-axis.
        max_methods : int | None, default 50
            If set, keep only the top-N methods by the chosen sorter (post-sort).
        save_path : str | None, default None
            If provided, writes the figure to this path (format inferred by extension).
        title : str | None, default None
            Optional plot title.

        Returns:
        -------
        fig : plotly.graph_objects.Figure
        df_plot : pd.DataFrame
            Filtered dataframe used for plotting (one row per observation).
        """
        import plotly.express as px

        if metric_col is None:
            metric_col = self.error_col

        required = {self.task_col, self.method_col, metric_col}
        missing = [c for c in required if c not in results_per_task.columns]
        if missing:
            raise ValueError(
                f"results_per_task is missing required columns: {missing}\n"
                f"Available columns: {list(results_per_task.columns)}",
            )

        df = results_per_task.loc[results_per_task[self.task_col] == dataset].copy()
        if df.empty:
            raise ValueError(
                f"No rows found for {self.task_col} == {dataset!r}.\n"
                f"Available datasets: {sorted(results_per_task[self.task_col].unique())[:50]}",
            )

        # Drop NaNs in the plotted metric (shouldn’t normally exist, but be safe)
        df = df.dropna(subset=[metric_col])

        # Determine whether we actually have a distribution per method (e.g. seeds)
        # If each method has a single value, violin/box isn't very informative.
        per_method_counts = df.groupby(self.method_col)[metric_col].size()
        has_distribution = bool((per_method_counts > 1).any())

        # Sort methods by mean/median for consistent x-axis ordering
        if sort_by not in {"median", "mean"}:
            raise ValueError(f"Invalid sort_by={sort_by!r}. Expected 'median' or 'mean'.")

        if sort_by == "median":
            sorter = df.groupby(self.method_col)[metric_col].median()
        else:
            sorter = df.groupby(self.method_col)[metric_col].mean()

        sorter = sorter.sort_values(ascending=ascending)

        if max_methods is not None and len(sorter) > max_methods:
            sorter = sorter.iloc[:max_methods]
            df = df[df[self.method_col].isin(sorter.index)].copy()

        method_order = list(sorter.index)

        # Pick plot type (force bar if no distribution)
        plot_kind = kind
        if plot_kind in {"violin", "box"} and not has_distribution:
            plot_kind = "bar"

        if title is None:
            title = f"{dataset}: {metric_col} across methods"

        if plot_kind == "violin":
            fig = px.violin(
                df,
                x=self.method_col,
                y=metric_col,
                category_orders={self.method_col: method_order},
                box=True,
                points=show_points,
                title=title,
            )
        elif plot_kind == "box":
            fig = px.box(
                df,
                x=self.method_col,
                y=metric_col,
                category_orders={self.method_col: method_order},
                points=show_points,
                title=title,
            )
        elif plot_kind == "bar":
            # One value per method (or user forced bar): use median/mean as height
            agg = sorter.rename(metric_col).reset_index()
            fig = px.bar(
                agg,
                x=self.method_col,
                y=metric_col,
                category_orders={self.method_col: method_order},
                title=title,
            )
        else:
            raise ValueError(f"Invalid kind={kind!r}. Expected 'violin', 'box', or 'bar'.")

        fig.update_layout(
            xaxis_title="Method",
            yaxis_title=metric_col,
            xaxis_tickangle=45,
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="white",
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True)

        if log_y:
            fig.update_yaxes(type="log")

        if save_path is not None:
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_image(save_path)

        return fig, df
