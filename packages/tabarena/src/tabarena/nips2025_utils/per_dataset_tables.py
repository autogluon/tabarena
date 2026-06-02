from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning, module="scipy")


NA_STR = "NA"  # change to whatever you want

# Section description shared between the multi-page combined LaTeX file and
# any external aggregator that wants the same leading description (e.g. the
# ``all_per_dataset.tex`` index in nick_scripts). Stored as the *body* of a
# ``\caption``: callers wrap it in ``\caption{...}`` for floating tables, or
# emit it as a plain paragraph (with a leading ``\textbf{}`` heading) for
# inline contexts.
PER_DATASET_TABLE_CAPTION_BODY = (
    r"\textbf{Performance Per Dataset.} "
    r"We show the average predictive performance per dataset with the standard deviation over folds. "
    r"We show the performance for the default hyperparameter configuration (\texttt{Default}), for the model after tuning (\texttt{Tuned}), and for the ensemble after tuning (\texttt{Tuned + Ens.}). "
    r"We highlight the best-performing methods with significance on three levels: "
    r"(1) \textcolor{green!50!black}{Green}: The best performing method on average; "
    r"(2) \textbf{Bold}: Methods that are not significantly worse than the best method on average, based on a Wilcoxon Signed-Rank test for paired samples with Holm-Bonferroni correction and $\alpha=0.05$. "
    r"(3) \underline{Underlined}: Methods that are not significantly worse than the best method in the same pipeline regime (\texttt{Default}, \texttt{Tuned}, or \texttt{Tuned + Ens.}), based on a Wilcoxon Signed-Rank test for paired samples with Holm-Bonferroni correction and $\alpha=0.05$. "
    r"Datasets with only a single split do not receive bold/underline annotations because the significance test degenerates without paired samples."
)

# Companion description of the per-dataset HPO Pareto trajectory plot that
# sits beside each table in ``all_per_dataset.tex``. Kept separate from the
# table caption so callers that emit only tables (e.g. the multi-page
# ``per_dataset_tables.tex``) are not forced to mention a plot they don't
# render. ``run_generate_beyond_leaderboard_minimal.py`` joins both bodies
# in its aggregated index.
PER_DATASET_PARETO_CAPTION_BODY = (
    r"\textbf{HPO Pareto Trajectory.} "
    r"Beside each per-dataset table we plot the metric error ($y$-axis) of each method's tuned configuration against cumulative training time in seconds ($x$-axis), as we increase the number of HPO trials. "
    r"Each curve traces the Pareto frontier of validation error versus training-time budget for a single method, so points further down and to the left dominate. "
    r"Reading the plot together with the table makes it possible to compare the cost (compute budget needed to reach a given error) and the achievable error of each method on that dataset, beyond the single endpoint summary in the table."
)


def _max_dot_pos(s: pd.Series) -> int:
    """Return max position of '.' in stringified values.
    Non-finite values are ignored (so they don't dominate the decision).
    """
    s_num = pd.to_numeric(s, errors="coerce")  # non-numeric -> NaN
    finite = np.isfinite(s_num.to_numpy(dtype="float64", na_value=np.nan))
    if not finite.any():
        return 0
    return s_num[finite].astype(str).str.find(".").max()


def _format_fixed(s: pd.Series, decimals: int) -> pd.Series:
    """Fixed-point formatting with NaN/inf -> NA_STR."""
    s_num = pd.to_numeric(s, errors="coerce")
    finite = np.isfinite(s_num.to_numpy(dtype="float64", na_value=np.nan))
    out = pd.Series(NA_STR, index=s.index, dtype="object")
    out.loc[finite] = s_num.loc[finite].round(decimals).map(lambda v: format(v, f".{decimals}f"))
    return out


def _format_scaled_int_str(s: pd.Series, scale: float, decimals_after: int = 0) -> pd.Series:
    """Scale, round to integer, then string.
    NaN/inf -> NA_STR.
    """
    s_num = pd.to_numeric(s, errors="coerce")
    finite = np.isfinite(s_num.to_numpy(dtype="float64", na_value=np.nan))
    out = pd.Series(NA_STR, index=s.index, dtype="object")
    if finite.any():
        vals = (s_num.loc[finite] / scale).round(0).astype("Int64")  # nullable int
        # If you truly want "int-looking" strings, use .astype(str) here.
        # Your original code sometimes did format(..., '.1f'), which adds .0.
        if decimals_after == 0:
            out.loc[finite] = vals.astype(str)
        else:
            out.loc[finite] = vals.map(lambda v: format(float(v), f".{decimals_after}f"))
    return out


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
                p_values[dataset_name][model_name] = 1.0
            else:
                # print(dataset_name,model_name)
                # Example performance metrics over 10 repeats for two models
                best_results = df_use.loc[
                    np.logical_and(
                        df_use.dataset_name == dataset_name,
                        df_use.model_name == best_model,
                    ),
                    0,
                ].values
                best_results[best_results < 0] = 0

                curr_model_results = df_use.loc[
                    np.logical_and(
                        df_use.dataset_name == dataset_name,
                        df_use.model_name == model_name,
                    ),
                    0,
                ].values
                curr_model_results[curr_model_results < 0] = 0

                p_values[dataset_name][model_name] = get_significance(
                    best_results, curr_model_results, direction=direction, method=method, verbose=verbose
                )
        # except:
        #     p_values[dataset_name] = {model_name: 0. for model_name in df_use.model_name.unique()}

    return pd.DataFrame(p_values).transpose()


def holm_bonferroni_correction(p_values):
    """Apply Holm-Bonferroni correction to a list of p-values.
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

    return list(zip(p_values, adjusted_p_values, reject, strict=False))


def get_significance(best_results, curr_model_results, method="wilcoxon", alpha=0.05, verbose=False, direction="max"):
    if (
        (direction == "min" and np.mean(best_results) >= np.mean(curr_model_results))
        or (direction == "max" and np.mean(best_results) <= np.mean(curr_model_results))
        or (direction == "max" and np.all(best_results <= curr_model_results))
    ):
        p_value = 2

    # Perform the paired t-test
    elif method == "ttest":
        t_statistic, p_value = stats.ttest_rel(best_results, curr_model_results)
    elif method == "wilcoxon":
        t_statistic, p_value = stats.wilcoxon(best_results, curr_model_results)  #     w_stat, p_value_w
    # elif method == "wilcoxon-corrected":
    #     t_statistic, p_value = stats.wilcoxon(best_results, curr_model_results) #     w_stat, p_value_w
    # Apply Holm-Bonferroni correction
    # reject, p_value, _, _ = multipletests(p_value, alpha=0.05, method='holm')
    elif method == "kruskal-wallis":
        statistic, p_value = stats.kruskal(best_results, curr_model_results)
    elif method == "ftest":
        _statistic, p_value = stats.f_oneway(best_results, curr_model_results)
    elif method == "tabred":
        sub_series_mean = np.mean(best_results)
        sub_series_std = np.std(best_results)
        if direction == "min":
            thresh = sub_series_mean + sub_series_std
            p_value = (np.mean(curr_model_results) < thresh) * 2

        elif direction == "max":
            thresh = sub_series_mean - sub_series_std
            p_value = (np.mean(curr_model_results) > thresh) * 2

    p_value_one_tailed = p_value / 2
    criterion = p_value_one_tailed < alpha

    if verbose:
        print(f"T-statistic: {t_statistic}")
        print(f"P-value: {p_value}")

    # Interpret the result
    if criterion:
        if verbose:
            print("There is a statistically significant difference between the two models.")
    elif verbose:
        print("There is no statistically significant difference between the two models.")
    if verbose:
        print("----------------------------------------------------------------------------------")

    return p_value


def _build_dataset_name_map(task_metadata: pd.DataFrame | None) -> dict[str, str]:
    """Map internal ``dataset`` id (e.g. ``Task-7163328506``) to a human-readable
    label drawn from ``task_metadata``.

    Prefers ``dataset_name`` (BeyondArena), falls back to ``name`` (default
    TabArena), and finally to identity if neither column is available.
    """
    if task_metadata is None or "dataset" not in task_metadata.columns:
        return {}
    if "dataset_name" in task_metadata.columns:
        return task_metadata.set_index("dataset")["dataset_name"].astype(str).to_dict()
    if "name" in task_metadata.columns:
        return task_metadata.set_index("dataset")["name"].astype(str).to_dict()
    return {}


_DEFAULT_SUFFIX = " (default)"
_TUNED_SUFFIX = " (tuned)"
_TUNED_ENS_SUFFIX = " (tuned + ensemble)"


def _regime_of(method: str) -> str | None:
    """Classify a method name into one of the three pipeline regimes by its
    suffix, or ``None`` if it doesn't match any (treated as a baseline).
    """
    if method.endswith(_TUNED_ENS_SUFFIX):
        return "tuned_ensemble"
    if method.endswith(_TUNED_SUFFIX):
        return "tuned"
    if method.endswith(_DEFAULT_SUFFIX):
        return "default"
    return None


def _default_method_order(df_results: pd.DataFrame) -> list[str]:
    """Derive a default per-dataset ordering from whatever methods are present
    in ``df_results``: every ``(default)`` method, then every ``(tuned)``,
    then every ``(tuned + ensemble)``, then any baselines, with stable
    alphabetical order within each regime.
    """
    methods = sorted(df_results["method"].dropna().unique().tolist())
    regime_rank = {"default": 0, "tuned": 1, "tuned_ensemble": 2, None: 3}
    return sorted(methods, key=lambda m: (regime_rank[_regime_of(m)], m))


def _safe_significance_dataset(for_sig: pd.DataFrame) -> pd.DataFrame:
    """Run ``get_significance_dataset`` only when the slice has at least two
    methods; otherwise return an empty frame so downstream lookups fall back
    to the ``KeyError`` -> default behavior the caller already handles.
    """
    if for_sig.empty or for_sig["model_name"].nunique() < 2:
        return pd.DataFrame()
    return get_significance_dataset(
        for_sig,
        method="wilcoxon",
        alpha=0.05,
        verbose=False,
        direction="min",
    )


def _is_significance_best(
    significance_df: pd.DataFrame,
    dataset_name: str,
    method_name: str,
) -> bool:
    """Return whether ``method_name`` is "not significantly worse than best"
    on ``dataset_name``. Returns ``False`` when significance was not computed
    (empty frame) or the row/column is missing — which matches what the
    caller wants: no underline/bold when we have no signal.
    """
    if significance_df.empty:
        return False
    if dataset_name not in significance_df.index:
        return False
    if method_name not in significance_df.columns:
        return False
    return significance_df.loc[dataset_name, method_name] > 0.05


def get_per_dataset_tables(
    df_results: pd.DataFrame,
    save_path: Path,
    task_metadata: pd.DataFrame | None = None,
    per_dataset_dir: str | Path | None = None,
    method_order: list[str] | None = None,
):
    """Generate per-dataset performance tables.

    Parameters
    ----------
    df_results :
        Per-method per-fold results (must contain ``dataset``, ``method``,
        ``metric``, ``metric_error``, ``imputed``, ``fold``).
    save_path :
        Directory to write the combined ``per_dataset_tables.tex`` into.
    task_metadata :
        Optional task-metadata frame providing a ``dataset_name`` (or ``name``)
        column to render human-readable dataset labels in captions. If
        omitted, the internal ``dataset`` id is used as the label.
    per_dataset_dir :
        Optional directory; when set, also writes one self-contained
        ``<dataset_id>.tex`` per dataset (a bare ``tabular`` block + label
        comments, no surrounding ``table`` environment) so callers can
        compose tables alongside per-dataset figures.
    method_order :
        Optional ordered list of method names (matching ``df_results["method"]``
        values) controlling which methods appear in the tables and in what row
        order. Methods ending in ``(default)``, ``(tuned)``, or ``(tuned +
        ensemble)`` populate the corresponding regime column; any other
        method is treated as a baseline and rendered in the ``Tuned + Ens.``
        column with ``-`` placeholders for the others. If omitted, every
        method present in ``df_results`` is used, sorted by regime.
    """
    save_path = Path(save_path)
    name_map = _build_dataset_name_map(task_metadata)

    if method_order is None:
        method_order = _default_method_order(df_results)
    use_methods_ordered = list(method_order)
    baseline_methods = [m for m in use_methods_ordered if _regime_of(m) is None]

    df_use = df_results.loc[df_results["method"].isin(use_methods_ordered)]
    df_use.loc[df_use["fold"] == 0]

    for_sig = df_use[["method", "dataset", "metric_error"]]
    for_sig.columns = ["model_name", "dataset_name", 0]

    significance_df = _safe_significance_dataset(for_sig)
    significance_default = _safe_significance_dataset(
        for_sig.loc[for_sig["model_name"].apply(lambda x: _regime_of(x) == "default")],
    )
    significance_tuned = _safe_significance_dataset(
        for_sig.loc[for_sig["model_name"].apply(lambda x: _regime_of(x) == "tuned")],
    )
    significance_tuned_ensemble = _safe_significance_dataset(
        for_sig.loc[for_sig["model_name"].apply(lambda x: _regime_of(x) == "tuned_ensemble")],
    )

    # NOTE: the prior implementation loaded TabArena's global task metadata here
    # only to derive ``can_run_tabpfnv2`` / ``can_run_tabicl`` for blanking
    # incompatible methods — but the blanking lines were already commented out,
    # making the whole load dead code that *also* broke BeyondArena (which has
    # different metadata columns). The dead block has been removed; the
    # ``task_metadata`` parameter on this function takes its place when
    # human-readable labels are needed.

    datasets_dict = {}
    datasets_human_name: dict[str, str] = {}
    for dataset_name in df_use["dataset"].unique():
        df_dat = df_use.loc[df_use["dataset"] == dataset_name]
        imputed_methods = df_dat.loc[df_dat.imputed, "method"].unique()

        if np.unique(df_dat["metric"])[0] == "roc_auc":
            df_dat.loc[:, "metric_error"] = 1 - df_dat["metric_error"]
            metric_dir = "max"
        else:
            metric_dir = "min"

        # Reindex to ``use_methods_ordered`` so missing methods (e.g. a baseline
        # with no result for this dataset) become NaN rows rather than raising.
        df_mean = df_dat[["method", "metric_error"]].groupby("method").mean().reindex(use_methods_ordered)
        df_std = df_dat[["method", "metric_error"]].groupby("method").std().reindex(use_methods_ordered)

        df_mean_raw = df_mean.copy()
        df_std.copy()

        df_mean = df_mean["metric_error"]
        df_std = df_std["metric_error"]

        dot_pos = _max_dot_pos(df_mean)

        if dot_pos == 1:
            df_mean = _format_fixed(df_mean, 3)
            df_std = _format_fixed(df_std, 3)
        elif dot_pos == 2:
            df_mean = _format_fixed(df_mean, 2)
            df_std = _format_fixed(df_std, 2)
        elif dot_pos in (3, 4):
            df_mean = _format_fixed(df_mean, 1)
            df_std = _format_fixed(df_std, 1)
        elif dot_pos == 5:
            # your original: round(0)->int then format(...'.1f') which yields "12.0"
            df_mean = _format_scaled_int_str(df_mean, scale=1, decimals_after=1)
            df_std = _format_scaled_int_str(df_std, scale=1, decimals_after=1)
        elif dot_pos == 6:
            # your original: (x/10)->round(0)->int->str
            df_mean = _format_scaled_int_str(df_mean, scale=10, decimals_after=0)
            df_std = _format_scaled_int_str(df_std, scale=10, decimals_after=0)

        df_mean = df_mean.to_frame()
        df_std = df_std.to_frame()

        # Combine ``mean`` and ``std`` cell-wise: when std is NA (e.g. only one
        # fold available) drop the ``$\pm$ NA`` suffix so the cell shows just
        # the mean instead of a half-empty pair.
        mean_series = df_mean.iloc[:, 0]
        std_series = df_std.iloc[:, 0]

        def _combine_cell(mean_str: str, std_str: str) -> str:
            if mean_str == NA_STR:
                return NA_STR
            if std_str == NA_STR:
                return mean_str
            return f"{mean_str} $\\pm$ {std_str}"

        df_latex = pd.DataFrame(
            {dataset_name: [_combine_cell(m, s) for m, s in zip(mean_series, std_series, strict=False)]},
            index=mean_series.index,
        )
        # ``reindex`` above introduced NaN rows for methods absent from this
        # dataset; render them (and any imputed methods) as the missing-data
        # placeholder so downstream styling can detect and skip them.
        present_methods = set(df_dat["method"].unique())
        missing_methods = [m for m in use_methods_ordered if m not in present_methods]
        for method in list(imputed_methods) + missing_methods:
            df_latex.loc[method] = "-"

        placeholder_methods = set(list(imputed_methods) + missing_methods)

        # ``df_mean_raw`` is reindexed to use_methods_ordered, so absent methods
        # are NaN. Drop imputed and missing rows *before* taking
        # ``idxmin``/``idxmax`` so an imputed score that happens to win the
        # raw ranking (e.g. when the fillna source's value is marginally
        # better than every real result) doesn't suppress the highlight
        # altogether. The cell at ``best_idx`` is then guaranteed to render
        # a real number that we can color.
        highlight_candidates = df_mean_raw["metric_error"].drop(
            labels=list(placeholder_methods),
            errors="ignore",
        )
        if highlight_candidates.notna().any():
            best_idx = highlight_candidates.idxmin() if metric_dir == "min" else highlight_candidates.idxmax()
            df_latex.loc[best_idx, dataset_name] = (
                r"\textcolor{green!50!black}{" + df_latex.loc[best_idx, dataset_name] + "}"
            )

        # With only one split (one fold of data per method) the Wilcoxon
        # significance test degenerates to a comparison of two scalars, which
        # currently produces a ``not significantly worse`` p-value of 2 for
        # every method except the best — i.e. every cell would get bolded /
        # underlined despite there being no statistical evidence. Skip the
        # bold/underline passes entirely for those datasets; the green
        # ``best mean`` highlight stays since it doesn't depend on
        # significance.
        n_splits = df_dat["fold"].nunique()
        single_split = n_splits < 2

        # Bold/underline only cells that have real numbers. The placeholder
        # ``-`` (and any literal ``NA`` that slipped through formatting) gets
        # left alone so we don't end up with ``\textbf{-}`` decorations.
        def _stylize(score: str, name: str, sig_df: pd.DataFrame, prefix: str) -> str:
            if single_split:  # noqa: B023
                return score
            if name in placeholder_methods or score == "-" or NA_STR in score:  # noqa: B023
                return score
            if not _is_significance_best(sig_df, dataset_name, name):  # noqa: B023
                return score
            return prefix + score + "}"

        df_latex.loc[:, dataset_name] = [
            _stylize(score, name, significance_df, r"\textbf{")
            for name, score in zip(df_latex.index, df_latex[dataset_name], strict=False)
        ]

        df_latex_def = df_latex.loc[[_regime_of(i) == "default" for i in df_latex.index]]
        df_latex_tuned = df_latex.loc[[_regime_of(i) == "tuned" for i in df_latex.index]]
        df_latex_tuned_ensemble = df_latex.loc[[_regime_of(i) == "tuned_ensemble" for i in df_latex.index]]

        df_latex_def.loc[:, dataset_name] = [
            _stylize(score, name, significance_default, r"\underline{")
            for name, score in zip(df_latex_def.index, df_latex_def[dataset_name], strict=False)
        ]
        df_latex_tuned.loc[:, dataset_name] = [
            _stylize(score, name, significance_tuned, r"\underline{")
            for name, score in zip(df_latex_tuned.index, df_latex_tuned[dataset_name], strict=False)
        ]
        df_latex_tuned_ensemble.loc[:, dataset_name] = [
            _stylize(score, name, significance_tuned_ensemble, r"\underline{")
            for name, score in zip(df_latex_tuned_ensemble.index, df_latex_tuned_ensemble[dataset_name], strict=False)
        ]

        df_latex_final = pd.merge(
            df_latex_def.rename(index=lambda s: s.split(" (")[0]),
            df_latex_tuned.rename(index=lambda s: s.split(" (")[0]),
            left_index=True,
            right_index=True,
            how="left",
        ).merge(
            df_latex_tuned_ensemble.rename(index=lambda s: s.split(" (")[0]),
            left_index=True,
            right_index=True,
            how="left",
        )
        # Each baseline (a method with no regime suffix) gets its own row in
        # the ``Tuned + Ens.`` column with placeholders for the other two —
        # this generalizes the previous AutoGluon-specific hard-coded line.
        for baseline in baseline_methods:
            if baseline in df_latex.index:
                df_latex_final.loc[baseline] = ["-", "-", df_latex.loc[baseline, dataset_name]]
        df_latex_final = df_latex_final.fillna("-")
        df_latex_final.columns = ["Default", "Tuned", "Tuned + Ens."]

        # if not can_run_tabpfnv2[dataset_name]:
        #     df_latex_final.loc["TabPFNv2"] = ["-", "-", "-"]
        # if not can_run_tabicl[dataset_name]:
        #     df_latex_final.loc["TabICL"] = ["-", "-", "-"]

        df_latex_final.index.name = None
        latex_safe_id = dataset_name.replace("_", r"\_")
        datasets_dict[latex_safe_id] = df_latex_final.copy()
        # Human-readable label (already escaped for LaTeX). Falls back to the
        # latex-safe id when no metadata mapping is available.
        human_label = name_map.get(dataset_name, dataset_name)
        datasets_human_name[latex_safe_id] = human_label.replace("_", r"\_")

    output_file = save_path / "per_dataset_tables.tex"
    per_col = 2  # 2 columns per row
    per_page = 6  # 3 rows × 2 columns = 6 subtables per figure
    sub_width = 0.48  # width for each subtable (2 across)
    sub_height = ""  # fixed height for each subtable

    metrics_used = dict(df_use[["dataset", "metric"]].drop_duplicates().values)
    metrics_used = {
        key.replace("_", r"\_"): "AUC" if value == "roc_auc" else ("logloss" if value == "log_loss" else "rmse")
        for key, value in metrics_used.items()
    }

    items = list(datasets_dict.items())

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for page_start in range(0, len(items), per_page):
            page_items = items[page_start : page_start + per_page]

            f.write(r"\begin{table}[htb]" + "\n")
            f.write(r"  \centering" + "\n")
            if page_start == 0:
                f.write("\\caption{" + PER_DATASET_TABLE_CAPTION_BODY + "}\n\n")

            # Build 3 rows of 2 columns each
            for row_start in range(0, len(page_items), per_col):
                row = page_items[row_start : row_start + per_col]
                for idx, (name, df) in enumerate(row):
                    print_name = datasets_human_name.get(name, name)
                    # Long-name special case from the original tabarena tables.
                    if print_name == "HR\\_Analytics\\_Job\\_Change\\_of\\_Data\\_Scientists":
                        print_name = "HR\\_Analytics\\_Job\\_Change"
                    # compute column format: index, quad gap, then one 'r' per data column
                    n_data = df.shape[1]
                    col_fmt = "l@{\\quad}" + "l" * n_data

                    f.write(f"  \\begin{{subtable}}[t]{{{sub_width}\\textwidth}}\n")
                    f.write(r"    \centering" + "\n")
                    f.write(r"    \scriptsize" + "\n")

                    # caption at the top
                    arrow = r"$\uparrow$" if metrics_used[name] == "AUC" else r"$\downarrow$"
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
                        column_format=col_fmt,
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

    if per_dataset_dir is not None:
        per_dataset_dir = Path(per_dataset_dir)
        per_dataset_dir.mkdir(parents=True, exist_ok=True)
        for latex_safe_id, df in datasets_dict.items():
            metric = metrics_used[latex_safe_id]
            arrow = r"$\uparrow$" if metric == "AUC" else r"$\downarrow$"
            human_label = datasets_human_name.get(latex_safe_id, latex_safe_id)
            n_data = df.shape[1]
            col_fmt = "l@{\\quad}" + "l" * n_data
            tabular = df.to_latex(index=True, escape=False, column_format=col_fmt)

            # File name uses the *raw* dataset id (no LaTeX escaping) so it
            # matches the trajectory-plot folder layout on disk.
            raw_id = latex_safe_id.replace(r"\_", "_")
            tex_path = per_dataset_dir / f"{raw_id}.tex"
            with open(tex_path, "w") as f:
                f.write(f"% Per-dataset performance table for {human_label} ({metric} {arrow}).\n")
                f.write("% Auto-generated; designed to be \\input{} into a minipage / subfigure.\n")
                f.write(r"\scriptsize" + "\n")
                f.write(tabular)

        print(f"Saved {len(datasets_dict)} per-dataset table fragments to {per_dataset_dir}")
