"""Reproduce Figure 6 (rank-1 share) of BeyondArena (arXiv:2606.30410).

The script uses the same frozen camera-ready method collection and all outer splits as
``run_generate_paper_figures_beyondarena2026.py``.  Results are loaded from the normal
TabArena cache, so this is an analysis/plotting pass rather than a new benchmark run.

Outputs are written to ``output_figure6_beyondarena2026/``:

* ``rank1_share.pdf`` / ``rank1_share.png`` — the paper figure;
* ``rank1_share.csv`` — the plotted category counts per sub-benchmark;
* ``significance_categories.csv`` — one auditable category per dataset.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from PIL import Image, ImageChops
from scipy import stats

from tabarena.contexts import BeyondArenaContext
from tabarena.contexts.beyondarena.methods import beyond_method_metadata_2026_camera_ready_collection

plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.labelsize": 18,
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
)


ALPHA = 0.05
OUTPUT_DIR = Path("output_figure6_beyondarena2026")
SUBSETS = {
    "IID": "iid",
    "Grouped": "grouped",
    "Temporal": "temporal",
    "Tiny": "tiny",
    "Small": "small",
    "Medium": "medium",
    "Large": "large",
    "Low-dim": "low-dim",
    "High-dim": "high-dim",
    "Text": "text",
    "High-card.": "high-cardinality",
    "All": "all",
}

# These are the category names used in the paper's legend.  The individual TFM names
# are retained because the paper reports their rank-1 share separately.
TFMS = {"TabICLv2", "TabPFN-2.6", "TabDPT"}
MLPS = {"RealMLP", "TabM"}
GBDTS = {"CatBoost", "LightGBM", "XGBoost"}
PAPER_METHODS = {
    "CatBoost (tuned + ensemble)",
    "ExtraTrees (tuned + ensemble)",
    "LightGBM (tuned + ensemble)",
    "RandomForest (default)",
    "RandomForest (tuned + ensemble)",
    "XGBoost (tuned + ensemble)",
    "TabM (tuned + ensemble)",
    "RealMLP (tuned + ensemble)",
    "Linear (tuned + ensemble)",
    "TabDPT (default)",
    "TabICLv2 (default)",
    "TabPFN-2.6 (default)",
}
COLORS = {
    "TabICLv2": "#8fd3ff",
    "TabPFN-2.6": "#4da3ff",
    "TabDPT": "#1f77b4",
    "TFM": "#0b3c8c",
    "Mixed": "#7f7f7f",
    "non-TFM": "#ff7f0e",
    "MLP": "#ffb347",
    "GBDT": "#ffd27f",
}


def _display_method_names(results: pd.DataFrame, context: BeyondArenaContext) -> pd.Series:
    """Return the paper display name for either raw result schema used by TabArena."""
    # ``method`` carries the tuning variant (the paper roster); ``config_type`` is only the
    # base model name and would collapse default/tuned results into duplicate pivot columns.
    column = "method" if "method" in results else "config_type"
    names = results[column].astype(str)
    rename = context.get_method_rename_map()
    expanded = dict(rename)
    for raw, display in rename.items():
        for suffix in (" (default)", " (tuned)", " (tuned + ensemble)"):
            expanded[f"{raw}{suffix}"] = f"{display}{suffix}"
    return names.map(expanded).fillna(names).map(_canonical_method_name)


def _canonical_method_name(name: str) -> str:
    """Normalize a result name while retaining the paper's default/tuned variant."""
    name = str(name).removeprefix("TA-")
    suffix = ""
    for candidate in (" (tuned + ensemble)", " (tuned)", " (default)"):
        if name.endswith(candidate):
            name, suffix = name[: -len(candidate)], candidate
            break
    return f"{_base_method_name(name)}{suffix}"


def _base_method_name(name: str) -> str:
    """Normalize config variants to the model names shown in Figure 6."""
    name = name.removeprefix("TA-")
    for suffix in (" (default)", " (tuned)", " (tuned + ensemble)"):
        name = name.removesuffix(suffix)
    return {
        "TABICLv2": "TabICLv2",
        "TABPFN-2.6": "TabPFN-2.6",
        "TABDPT": "TabDPT",
        "REALMLP": "RealMLP",
        "TABM": "TabM",
        "CAT": "CatBoost",
        "GBM": "LightGBM",
        "XGB": "XGBoost",
    }.get(name, name)


def _p_value(best: np.ndarray, current: np.ndarray) -> float:
    """Paper test: Wilcoxon for >3 splits, otherwise a wins-all-splits test."""
    if len(best) == 1:
        return 1.0 if best[0] == current[0] else 0.0
    if len(best) <= 3:
        return 0.0 if np.all(best < current) else 1.0
    difference = np.round(best - current, 12)
    if np.all(difference == 0):
        return 1.0
    return float(stats.wilcoxon(difference, alternative="less", zero_method="wilcox").pvalue)


def _dataset_categories(results: pd.DataFrame) -> pd.Series:
    """Choose the paper's significance-aware rank-1 category for every dataset."""
    results = results.copy()
    if "model" not in results:
        method_column = "config_type" if "config_type" in results else "method"
        results["model"] = results[method_column]
    categories: dict[str, str] = {}
    for dataset, frame in results.groupby("dataset", sort=True):
        pivot = frame.pivot_table(index="fold", columns="model", values="metric_error", aggfunc="first")
        full = [model for model in pivot if pivot[model].notna().all()]
        if not full:
            continue
        ordered = pivot[full].mean().sort_values(kind="mergesort").index.tolist()
        selected_p: dict[str, float] = {}
        for candidate in ordered:
            candidate_p = {
                other: 1.0
                if other == candidate
                else _p_value(pivot[candidate].to_numpy(float), pivot[other].to_numpy(float))
                for other in full
            }
            if any(other != candidate and p < ALPHA for other, p in candidate_p.items()):
                selected_p = candidate_p
                break
            if candidate == ordered[-1]:
                selected_p = candidate_p

        tied = [model for model, p in selected_p.items() if p >= ALPHA]
        names = {_base_method_name(str(model)) for model in tied}
        if len(names) == 1 and next(iter(names)) in TFMS:
            label = next(iter(names))
        elif names and names <= TFMS:
            label = "TFM"
        elif names and names <= MLPS:
            label = "MLP"
        elif names and names <= GBDTS:
            label = "GBDT"
        elif names and not (names & TFMS):
            label = "non-TFM"
        else:
            label = "Mixed"
        categories[str(dataset)] = label
    return pd.Series(categories, name="category")


def _dataset_slice_membership(context: BeyondArenaContext) -> pd.DataFrame:
    """Evaluate the official BeyondArena predicates once at dataset level."""
    grid = context.task_metadata_collection.task_grid().groupby("dataset", as_index=False).first()
    membership = pd.DataFrame(index=grid["dataset"].astype(str))
    for label, predicate in SUBSETS.items():
        membership[label] = context.subset_predicates[predicate].evaluate(grid).to_numpy(bool)
    return membership


def rank1_share_table(results: pd.DataFrame, context: BeyondArenaContext) -> pd.DataFrame:
    """Build the long table behind Figure 6."""
    categories = _dataset_categories(results)
    membership = _dataset_slice_membership(context)
    rows = []
    for subset_order, subset in enumerate(SUBSETS):
        datasets = membership.index[membership[subset]].intersection(categories.index)
        counts = categories.loc[datasets].value_counts()
        for category in COLORS:
            rows.append(
                {
                    "subset_order": subset_order,
                    "subset": subset,
                    "category": category,
                    "count": int(counts.get(category, 0)),
                    "n_datasets": len(datasets),
                    "share": float(counts.get(category, 0) / len(datasets)) if len(datasets) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def plot_rank1_share(table: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> tuple[Path, Path]:
    """Render Figure 6 with BeyondBench's exact streamgraph interpolation and styling."""
    output_dir.mkdir(parents=True, exist_ok=True)
    subsets = list(SUBSETS)
    categories = list(COLORS)
    pivot = table.pivot(index="subset", columns="category", values="share").reindex(
        index=subsets, columns=categories, fill_value=0
    )
    totals = table.drop_duplicates("subset").set_index("subset")["n_datasets"].reindex(subsets)
    x = np.arange(len(subsets), dtype=float)
    interpolate_steps = 35
    flat_fraction = 0.5
    dense_x = np.linspace(-0.5, len(subsets) - 0.5, len(subsets) * interpolate_steps + 1)
    flat_half = max(0.0, min(0.5, flat_fraction / 2.0))
    transition_half = max(1e-12, 0.5 - flat_half)
    series = []
    for category in categories:
        values = pivot[category].to_numpy(dtype=float)
        dense_values = np.empty_like(dense_x)
        for pos_idx, x_pos in enumerate(dense_x):
            center = int(np.floor(x_pos + 0.5))
            center = max(0, min(center, len(values) - 1))
            left_boundary = center - 0.5
            right_boundary = center + 0.5
            if x_pos <= left_boundary + transition_half:
                if center == 0:
                    dense_values[pos_idx] = values[0]
                else:
                    blend = (x_pos - (left_boundary - transition_half)) / (2.0 * transition_half)
                    dense_values[pos_idx] = values[center - 1] * (1.0 - blend) + values[center] * blend
            elif x_pos >= right_boundary - transition_half:
                if center == len(values) - 1:
                    dense_values[pos_idx] = values[-1]
                else:
                    blend = (x_pos - (right_boundary - transition_half)) / (2.0 * transition_half)
                    dense_values[pos_idx] = values[center] * (1.0 - blend) + values[center + 1] * blend
            else:
                dense_values[pos_idx] = values[center]
        series.append(dense_values)

    family_colors = {
        "TabDPT": "#1f77b4",
        "TabPFN-2.6": "#4da3ff",
        "TabICLv2": "#8fd3ff",
        "TFM": "#0b3c8c",
        "non-TFM": "#ff7f0e",
        "Mixed": "#7f7f7f",
        "MLP": "#ffb347",
        "GBDT": "#ffd27f",
    }
    fig, ax = plt.subplots(figsize=(12, 5.4))
    stacks = ax.stackplot(
        dense_x,
        *series,
        colors=[family_colors[c] for c in categories],
        baseline="zero",
        alpha=0.96,
        linewidth=0.25,
    )
    stacks[categories.index("Mixed")].set_hatch("/")
    stacks[categories.index("Mixed")].set_edgecolor("0.25")
    stacks[categories.index("Mixed")].set_linewidth(0.4)
    ax.set_xlim(dense_x[0], dense_x[-1])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\nn={int(totals[s])}" for s in subsets])
    ax.set_ylabel("Rank-1-Datasets Share")
    ax.tick_params(axis="x", pad=8)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    for position in x[:-1] + 0.5:
        ax.axvline(position, color="0.5", linestyle="--", linewidth=0.8, alpha=0.45, zorder=4)
    for position in (2.5, 6.5, 8.5, 10.5):
        ax.axvline(position, color="black", linestyle="-", linewidth=1.4, alpha=0.6, zorder=5)
    handles = [
        Patch(
            facecolor=family_colors[c],
            edgecolor="0.25" if c == "Mixed" else "white",
            linewidth=0.4,
            hatch="/" if c == "Mixed" else None,
            label=c,
        )
        for c in categories
    ]
    ax.legend(handles=handles, frameon=False, ncol=len(categories), loc="lower center", bbox_to_anchor=(0.5, 0.968))
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    pdf = output_dir / "rank1_share.pdf"
    png = output_dir / "rank1_share.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=8 / 220)  # vector, with the same 8 px margin as the PNG
    fig.savefig(png, bbox_inches="tight", dpi=220)
    with Image.open(png) as image:
        rgb = image.convert("RGB")
        background = Image.new("RGB", rgb.size, (255, 255, 255))
        bbox = ImageChops.difference(rgb, background).getbbox()
        if bbox is not None:
            left, top, right, bottom = bbox
            bbox = (max(0, left - 8), max(0, top - 8), min(rgb.width, right + 8), min(rgb.height, bottom + 8))
            rgb.crop(bbox).save(png)
    plt.close(fig)
    return pdf, png


def generate_figure6(output_dir: Path = OUTPUT_DIR) -> tuple[Path, Path]:
    """Load the paper results from cache, analyze them, and write Figure 6."""
    context = BeyondArenaContext(methods=beyond_method_metadata_2026_camera_ready_collection.method_metadata_lst)
    _, results = context.compare(output_dir=None, subset=["all"], return_results=True, plot=False)
    results = results.copy()
    results["model"] = _display_method_names(results, context)
    results = results[results["model"].isin(PAPER_METHODS)].copy()
    if set(results["model"].unique()) != PAPER_METHODS:
        missing = sorted(PAPER_METHODS - set(results["model"].unique()))
        raise RuntimeError(f"The paper method roster is incomplete; missing {missing}")
    table = rank1_share_table(results, context)
    categories = _dataset_categories(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "rank1_share.csv", index=False)
    categories.rename_axis("dataset").to_csv(output_dir / "significance_categories.csv")
    return plot_rank1_share(table, output_dir)


if __name__ == "__main__":
    pdf, png = generate_figure6()
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
