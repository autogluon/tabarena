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
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import stats

from tabarena.contexts import BeyondArenaContext
from tabarena.contexts.beyondarena.methods import beyond_method_metadata_2026_camera_ready_collection


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
    column = "config_type" if "config_type" in results else "method"
    names = results[column].astype(str)
    rename = context.get_method_rename_map()
    expanded = dict(rename)
    for raw, display in rename.items():
        for suffix in (" (default)", " (tuned)", " (tuned + ensemble)"):
            expanded[f"{raw}{suffix}"] = f"{display}{suffix}"
    return names.map(expanded).fillna(names).map(_base_method_name)


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
        selected = ordered[0]
        selected_p: dict[str, float] = {}
        for candidate in ordered:
            candidate_p = {
                other: 1.0 if other == candidate else _p_value(
                    pivot[candidate].to_numpy(float), pivot[other].to_numpy(float)
                )
                for other in full
            }
            if any(other != candidate and p < ALPHA for other, p in candidate_p.items()):
                selected, selected_p = candidate, candidate_p
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
            rows.append({
                "subset_order": subset_order,
                "subset": subset,
                "category": category,
                "count": int(counts.get(category, 0)),
                "n_datasets": int(len(datasets)),
                "share": float(counts.get(category, 0) / len(datasets)) if len(datasets) else 0.0,
            })
    return pd.DataFrame(rows)


def plot_rank1_share(table: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> tuple[Path, Path]:
    """Render the flowing stacked area chart used as Figure 6."""
    output_dir.mkdir(parents=True, exist_ok=True)
    subsets = list(SUBSETS)
    categories = list(COLORS)
    pivot = table.pivot(index="subset", columns="category", values="share").reindex(index=subsets, columns=categories, fill_value=0)
    totals = table.drop_duplicates("subset").set_index("subset")["n_datasets"].reindex(subsets)
    x = np.arange(len(subsets), dtype=float)
    dense_x = np.linspace(0, len(subsets) - 1, (len(subsets) - 1) * 25 + 1)
    dense = np.vstack([np.interp(dense_x, x, pivot[category].to_numpy()) for category in categories])

    fig, ax = plt.subplots(figsize=(14.0, 5.2))
    stacks = ax.stackplot(dense_x, *dense, colors=[COLORS[c] for c in categories], alpha=0.96, linewidth=0.25)
    stacks[4].set_hatch("/")
    stacks[4].set_edgecolor("0.25")
    stacks[4].set_linewidth(0.4)
    ax.set_xlim(0, len(subsets) - 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\nn={int(totals[s])}" for s in subsets])
    ax.set_ylabel("Rank-1-Datasets Share")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    for position in x[:-1] + 0.5:
        ax.axvline(position, color="0.5", linestyle="--", linewidth=0.8, alpha=0.45)
    for position in (2.5, 6.5, 8.5, 10.5):
        ax.axvline(position, color="black", linewidth=1.4, alpha=0.6)
    handles = [Patch(facecolor=COLORS[c], edgecolor="0.25" if c == "Mixed" else "white", hatch="/" if c == "Mixed" else None, label=c) for c in categories]
    ax.legend(handles=handles, frameon=False, ncol=len(categories), loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    pdf = output_dir / "rank1_share.pdf"
    png = output_dir / "rank1_share.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=220)
    plt.close(fig)
    return pdf, png


def generate_figure6(output_dir: Path = OUTPUT_DIR) -> tuple[Path, Path]:
    """Load the paper results from cache, analyze them, and write Figure 6."""
    context = BeyondArenaContext(methods=beyond_method_metadata_2026_camera_ready_collection.method_metadata_lst)
    _, results = context.compare(output_dir=None, subset=["all"], return_results=True, plot=False)
    results = results.copy()
    results["model"] = _display_method_names(results, context)
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
