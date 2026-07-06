"""Per-family / per-model subset-overview result plots (BeyondArena flow).

Port of the standalone playground scripts (``result_plot_per_family*.py`` /
``result_plot_per_model*.py``) into one reusable entry point: :func:`plot_subset_results`
consumes the ``{subset_label: leaderboard}`` dict returned by
:func:`tabarena.evaluation.beyond_arena_eval.run_beyond_arena_eval` and renders, per metric
(Elo, improvability, or a custom :class:`MetricSpec`):

* ``per_family`` — one line per model family (TFM / MLP / GBDT / Baseline), aggregated to the
  family's *best* model per subset; whiskers show that best model's CI. *Contender* models are
  drawn as their own standalone line on top of the family lines.
* ``per_model`` — per subset, one translucent bar + hollow marker per model with CI whiskers,
  hatched bars where scores are partially imputed, and a two-row family/model legend.

Both plots share the x-axis scaffolding: one column per subset leaderboard, grouped into
Task / Scale / Dimensionality / Features regions with separators and ``N=<n_datasets>``
annotations per column.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

# --- Subset (x-axis) defaults ----------------------------------------------

#: Canonical x-axis order; ``plot_subset_results`` keeps only the labels present in the input.
DEFAULT_SUBSET_ORDER = [
    "random",
    "grouped",
    "temporal",
    "tiny",
    "small",
    "medium",
    "large",
    "low-dim",
    "high-dim",
    "text",
    "high-cardinality",
    "full",
]

SUBSET_LABELS = {
    "full": "All",
    "random": "IID",
    "iid": "IID",
    "grouped": "Grouped",
    "temporal": "Temporal",
    "tiny": "Tiny",
    "small": "Small",
    "medium": "Medium",
    "large": "Large",
    "high-dim": "High",
    "low-dim": "Low",
    "text": "Text",
    "high-cardinality": "High Card.",
}

#: Region annotation per subset; contiguous runs of the same region get one label + separators.
SUBSET_REGION = {
    "random": "Task",
    "iid": "Task",
    "grouped": "Task",
    "temporal": "Task",
    "tiny": "Scale",
    "small": "Scale",
    "medium": "Scale",
    "large": "Scale",
    "low-dim": "Dimensionality",
    "high-dim": "Dimensionality",
    "text": "Features",
    "high-cardinality": "Features",
}

# --- Method grouping defaults ----------------------------------------------

#: Subtype priority when a method entry does not pin one explicitly.
SUBTYPE_PREFERENCE = ["tuned_ensemble", "tuned", "default"]


def _at(methods: list[str], subtype: str) -> list[tuple[str, str]]:
    return [(m, subtype) for m in methods]


_MLP_METHODS = ["RealMLP", "TabM"]
_GBDT_METHODS = ["CatBoost", "XGBoost", "LightGBM"]
_BASELINE_METHODS = ["RandomForest", "ExtraTrees", "Linear"]

#: Per-family line groups. Methods inside a group are collapsed per subset into the group's
#: best value; the whisker spans the best method's CI. Each method entry is either a bare name
#: (strongest available subtype per ``SUBTYPE_PREFERENCE``) or a ``(method, subtype)`` tuple.
#: ``subtype`` selects the marker/linestyle via ``SUBTYPE_STYLES``; ``ICL`` is the synthetic
#: subtype for foundation models (not tuned; default mode *is* in-context learning).
DEFAULT_FAMILY_GROUPS: dict[str, dict] = {
    "TFM": {"family": "TFM", "subtype": "ICL", "methods": ["TabICLv2", "TabPFN-2.6", "TabDPT", "TabPFN-3"]},
    "MLP (T+E)": {"family": "MLP", "subtype": "T+E", "methods": _at(_MLP_METHODS, "tuned_ensemble")},
    "MLP (D)": {"family": "MLP", "subtype": "D", "methods": _at(_MLP_METHODS, "default")},
    "GBDT (T+E)": {"family": "GBDT", "subtype": "T+E", "methods": _at(_GBDT_METHODS, "tuned_ensemble")},
    "GBDT (D)": {"family": "GBDT", "subtype": "D", "methods": _at(_GBDT_METHODS, "default")},
    "Baseline (T+E)": {"family": "Baseline", "subtype": "T+E", "methods": _at(_BASELINE_METHODS, "tuned_ensemble")},
    "Baseline (D)": {"family": "Baseline", "subtype": "D", "methods": _at(_BASELINE_METHODS, "default")},
}

#: Marker-shape groups for the per-model plot. ``zboost`` lifts a group's markers above all
#: lower-boost groups at every column (contenders get a still-higher boost).
DEFAULT_MARKER_GROUPS: dict[str, dict] = {
    "Foundation Model": {"methods": ["TabPFN-2.6", "TabICLv2", "TabDPT", "TabPFN-3"], "marker": "X", "zboost": 1000},
    "MLP": {"methods": _MLP_METHODS, "marker": "v"},
    "GBDT": {"methods": _GBDT_METHODS, "marker": "^"},
    "Baseline": {"methods": _BASELINE_METHODS, "marker": "o"},
}

#: Synthetic subtype assigned to contender lines in the per-family plot.
CONTENDER_SUBTYPE = "C"
CONTENDER_MARKER = "*"
CONTENDER_ZBOOST = 2000

#: subtype -> (marker, linestyle, legend label). Marker shape carries the subtype in the
#: per-family plot; color carries the family.
SUBTYPE_STYLES: dict[str, tuple[str, object, str]] = {
    "T+E": ("s", "-", "Tuned + Ensembled"),
    "D": ("o", "--", "Default"),
    "ICL": ("X", "-", "ICL"),
    CONTENDER_SUBTYPE: (CONTENDER_MARKER, "-", "Contender"),
}
#: Pipeline-legend display order; anchor priority picks each family's legend-sort value.
SUBTYPE_ORDER = ["ICL", "D", "T+E", CONTENDER_SUBTYPE]
ANCHOR_PRIORITY = ["T+E", "D", "ICL", CONTENDER_SUBTYPE]

#: Manual per-model color overrides (palette default for RandomForest reads too close to
#: the other yellows/oranges in the bright palette).
COLOR_OVERRIDES = {"RandomForest": "darkgreen"}

# --- Metrics -----------------------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    """How to read and render one leaderboard metric.

    ``column`` must exist in every leaderboard; CI whiskers read ``<column>+`` / ``<column>-``
    (asymmetric offsets) when present. ``floor`` clamps the auto-derived y-axis bottom (e.g.
    improvability is a gap-from-best, so it never goes below 0).
    """

    name: str
    column: str
    higher_is_better: bool
    ylabel: str
    ylabel_family: str
    tick_step: float
    floor: float | None = None
    tick_format: str = "{:g}"

    @property
    def err_hi_column(self) -> str:
        return f"{self.column}+"

    @property
    def err_lo_column(self) -> str:
        return f"{self.column}-"

    def sort_value(self, value: float) -> float:
        """Key making *better* values sort first (NaN-safe callers filter beforehand)."""
        return -value if self.higher_is_better else value


METRIC_SPECS: dict[str, MetricSpec] = {
    "elo": MetricSpec(
        name="elo",
        column="elo",
        higher_is_better=True,
        ylabel="Elo",
        ylabel_family="Elo (Best Model)",
        tick_step=100,
    ),
    "improvability": MetricSpec(
        name="improvability",
        column="improvability",
        higher_is_better=False,
        ylabel="Improvability (lower is better)",
        ylabel_family="Improvability (Best Model)",
        tick_step=0.05,
        floor=0.0,
        tick_format="{:.2f}",
    ),
}

# --- Drawing constants (shared by both plot kinds) ---------------------------

_LINE_WIDTH = 1.5
_LINE_ALPHA = 0.85
_MARKER_SIZE = 8
_MARKER_EDGE_WIDTH = 1.8
_ERROR_BAR_WIDTH = 1.4
_ERROR_BAR_ALPHA = 0.7
_ERROR_CAP_WIDTH = 0.08  # x units; T-cap half-width on each side
_WHISKER_ZORDER = 5  # below every marker pass so caps never poke through markers
_CONNECTOR_ZORDER = 2
_MARKER_ZORDER_BASE = 10
_EDGE_ZORDER_OFFSET = 100  # edges above all fills so borders stay visible
_BAR_ALPHA = 0.45
_DODGE_WIDTH = 0.8  # per-model: total x spread of the per-column method slots

_RC_PARAMS = {
    "font.family": "serif",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
    "lines.linewidth": 2.2,
    "lines.markersize": 9,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "hatch.linewidth": 2.0,
}


# --- Public entry point -------------------------------------------------------


def plot_subset_results(
    leaderboards: dict[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    plot_kinds: Sequence[str] = ("per_family", "per_model"),
    metrics: Sequence[str | MetricSpec] = ("elo", "improvability"),
    contenders: Sequence[str | tuple[str, str]] = (),
    subset_order: Sequence[str] | None = None,
    family_groups: dict[str, dict] | None = None,
    marker_groups: dict[str, dict] | None = None,
    models: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Render the per-family / per-model subset-overview plots from eval leaderboards.

    Args:
        leaderboards: ``{subset_label: leaderboard}`` as returned by ``run_beyond_arena_eval``
            (or read back from its ``all_leaderboards`` CSVs).
        output_dir: Directory the figures are written to (one ``<kind>_<metric>.pdf`` + ``.png``
            per combination).
        plot_kinds: Any of ``"per_family"`` / ``"per_model"``.
        metrics: Metric names from :data:`METRIC_SPECS` (``"elo"``, ``"improvability"``) or
            custom :class:`MetricSpec` instances.
        contenders: Methods highlighted as *contenders*: drawn as their own standalone line in
            the per-family plot and star-marked / top-layered in the per-model plot. Entries are
            bare method names (strongest available subtype) or ``(method, subtype)`` tuples.
        subset_order: x-axis order; defaults to :data:`DEFAULT_SUBSET_ORDER` filtered to the
            available labels (+ any unknown labels appended, keeping ``"full"`` last).
        family_groups: Override for :data:`DEFAULT_FAMILY_GROUPS` (per-family lines).
        marker_groups: Override for :data:`DEFAULT_MARKER_GROUPS` (per-model marker shapes).
        models: If given, the per-model plot only shows these methods (default: all available).

    Returns:
        ``{"<kind>_<metric>": <pdf path>}`` for every figure written. A kind is skipped (with a
        printed note) when none of its configured methods appear in the data.
    """
    order = _resolve_subset_order(leaderboards, subset_order)
    df = _prepare_frame(leaderboards, order)
    family_groups = dict(DEFAULT_FAMILY_GROUPS if family_groups is None else family_groups)
    marker_groups = dict(DEFAULT_MARKER_GROUPS if marker_groups is None else marker_groups)
    contender_entries = list(contenders)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}
    with plt.rc_context({**sns.axes_style("whitegrid"), **_RC_PARAMS}):
        for metric in metrics:
            spec = metric if isinstance(metric, MetricSpec) else _lookup_metric(metric)
            _require_columns(df, spec)
            for kind in plot_kinds:
                if kind == "per_family":
                    fig = _per_family_figure(df, order, spec, family_groups, contender_entries)
                elif kind == "per_model":
                    fig = _per_model_figure(df, order, spec, marker_groups, contender_entries, models)
                else:
                    raise ValueError(f"Unknown plot kind {kind!r}; expected 'per_family' or 'per_model'.")
                if fig is None:
                    print(f"Skipping {kind} plot for metric {spec.name!r}: no configured methods in the data.")
                    continue
                path = output_dir / f"{kind}_{spec.name}.pdf"
                fig.savefig(path, bbox_inches="tight")
                fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=220)
                plt.close(fig)
                print(f"Saved: {path}")
                saved[f"{kind}_{spec.name}"] = path
    return saved


# --- Data preparation ----------------------------------------------------------


def _lookup_metric(name: str) -> MetricSpec:
    if name not in METRIC_SPECS:
        raise ValueError(f"Unknown metric {name!r}; known: {sorted(METRIC_SPECS)}. Pass a MetricSpec for others.")
    return METRIC_SPECS[name]


def _require_columns(df: pd.DataFrame, spec: MetricSpec) -> None:
    if spec.column not in df.columns:
        raise ValueError(f"Leaderboards lack metric column {spec.column!r}; available: {sorted(df.columns)}")


def _resolve_subset_order(leaderboards: dict[str, pd.DataFrame], subset_order: Sequence[str] | None) -> list[str]:
    if subset_order is not None:
        missing = [s for s in subset_order if s not in leaderboards]
        if missing:
            raise ValueError(f"subset_order entries missing from leaderboards: {missing}")
        return list(subset_order)
    order = [s for s in DEFAULT_SUBSET_ORDER if s in leaderboards]
    extras = [s for s in leaderboards if s not in order]
    # Unknown labels go between the known ones and the trailing "full" column.
    if order and order[-1] == "full":
        return order[:-1] + extras + ["full"]
    return order + extras


def _prepare_frame(leaderboards: dict[str, pd.DataFrame], order: list[str]) -> pd.DataFrame:
    frames = []
    for label in order:
        sub = leaderboards[label].copy()
        sub["leaderboard"] = label
        frames.append(sub)
    df = pd.concat(frames, ignore_index=True)
    # Strip the "(<subtype>)" display suffix; the method_subtype column already carries it.
    df["method"] = df["method"].astype(str).str.replace(r"\s*\([^)]*\)\s*$", "", regex=True)
    return df


def _series_by(df: pd.DataFrame, order: list[str], key_cols: list[str], col: str) -> dict[tuple, np.ndarray]:
    """Per ``key_cols`` group: the ``col`` series across subsets (NaN where absent)."""
    out: dict[tuple, np.ndarray] = {}
    if col not in df.columns:
        return out
    for key, sub in df.groupby(key_cols):
        key = key if isinstance(key, tuple) else (key,)
        out[key] = sub.set_index("leaderboard").reindex(order)[col].to_numpy(dtype=float)
    return out


def _n_datasets_per_subset(df: pd.DataFrame) -> dict[str, int]:
    if "n_datasets_total" not in df.columns:
        return {}
    return df.groupby("leaderboard")["n_datasets_total"].first().astype(int).to_dict()


def _contender_name(entry: str | tuple[str, str]) -> str:
    return entry[0] if isinstance(entry, tuple) else entry


# --- Shared axes scaffolding ----------------------------------------------------


def _region_spans(order: list[str]) -> list[tuple[str | None, int, int]]:
    """Contiguous ``(region_label, start_idx, end_idx)`` runs over the resolved subset order."""
    spans: list[tuple[str | None, int, int]] = []
    for i, label in enumerate(order):
        region = SUBSET_REGION.get(label)
        if spans and spans[-1][0] == region:
            spans[-1] = (region, spans[-1][1], i)
        else:
            spans.append((region, i, i))
    return spans


def _draw_x_scaffolding(ax: Axes, order: list[str], *, column_guides: bool, left: float, right: float) -> None:
    """Separators, region labels, ticks and x-limits shared by both plot kinds."""
    spans = _region_spans(order)
    boundaries = {spans[i][2] + 0.5 for i in range(len(spans) - 1)}
    if column_guides:  # per-family: a light guide at every column
        for x in range(len(order)):
            ax.axvline(x, color="lightgray", linestyle="--", linewidth=0.9, zorder=0)
        for xb in boundaries:
            ax.axvline(xb, color="dimgray", linestyle=(0, (4, 3)), linewidth=1.4, alpha=0.85, zorder=0)
    else:  # per-model: separators between columns, solid at region boundaries
        for i in range(len(order) - 1):
            xb = i + 0.5
            if xb in boundaries:
                ax.axvline(xb, color="dimgray", linestyle="-", linewidth=1.4, alpha=0.85, zorder=0)
            else:
                ax.axvline(xb, color="gray", linestyle=(0, (4, 4)), linewidth=1.1, alpha=0.7, zorder=0)

    prev_right = left
    for label, _start, end in spans:
        if label is not None:
            ax.text(
                prev_right + 0.1,
                0.965,
                label,
                transform=ax.get_xaxis_transform(),
                ha="left",
                va="top",
                fontsize=12,
                fontweight="bold",
                color="dimgray",
            )
        prev_right = end + 0.5

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([SUBSET_LABELS.get(lb, lb) for lb in order])
    ax.set_xlim(left, len(order) - 1 + right)


def _annotate_n_datasets(ax: Axes, order: list[str], n_per_subset: dict[str, int]) -> None:
    for i, label in enumerate(order):
        if label not in n_per_subset:
            continue
        ax.annotate(
            f"N={n_per_subset[label]}",
            xy=(i, -0.12),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=10,
            color="dimgray",
        )


def _y_axis(ax: Axes, spec: MetricSpec, vmin: float, vmax: float) -> float:
    """Set data-driven y-limits/ticks aligned to ``spec.tick_step``; returns the bottom."""
    step = spec.tick_step
    pad = 0.25 * step
    bottom = np.floor((vmin - pad) / step) * step
    if spec.floor is not None:
        bottom = max(bottom, spec.floor)
    top = np.ceil((vmax + pad) / step) * step
    ticks = np.arange(bottom, top + step / 2, step)
    ax.set_ylim(bottom, top)
    ax.set_yticks(ticks)
    ax.set_yticklabels([spec.tick_format.format(t) for t in ticks])
    return float(bottom)


def _draw_whisker(ax: Axes, x: float, y_low: float, y_high: float, color) -> None:
    """T-shaped CI whisker (vertical line + horizontal caps) below every marker pass."""
    common = dict(
        color=color,
        linewidth=_ERROR_BAR_WIDTH,
        alpha=_ERROR_BAR_ALPHA,
        zorder=_WHISKER_ZORDER,
        solid_capstyle="round",
    )
    ax.plot([x, x], [y_low, y_high], **common)
    for y_cap in (y_low, y_high):
        ax.plot([x - _ERROR_CAP_WIDTH / 2, x + _ERROR_CAP_WIDTH / 2], [y_cap, y_cap], **common)


def _draw_hollow_marker(ax: Axes, x: float, y: float, marker: str, color, zorder: int) -> None:
    """Two-pass hollow marker: white fill low, colored edge high, so borders stay visible."""
    ax.plot(
        [x],
        [y],
        linestyle="none",
        marker=marker,
        markerfacecolor=mcolors.to_rgba("white", 1.0),
        markeredgecolor="none",
        markersize=_MARKER_SIZE,
        zorder=zorder,
    )
    ax.plot(
        [x],
        [y],
        linestyle="none",
        marker=marker,
        markerfacecolor="none",
        markeredgecolor=color,
        markeredgewidth=_MARKER_EDGE_WIDTH,
        markersize=_MARKER_SIZE,
        zorder=zorder + _EDGE_ZORDER_OFFSET,
    )


# --- Per-family plot --------------------------------------------------------------


def _per_family_figure(
    df: pd.DataFrame,
    order: list[str],
    spec: MetricSpec,
    family_groups: dict[str, dict],
    contenders: list[str | tuple[str, str]],
):
    df = df[df["method_subtype"].isin(SUBTYPE_PREFERENCE)]
    val_by_ms = _series_by(df, order, ["method", "method_subtype"], spec.column)
    lo_by_ms = _series_by(df, order, ["method", "method_subtype"], spec.err_lo_column)
    hi_by_ms = _series_by(df, order, ["method", "method_subtype"], spec.err_hi_column)

    def resolve(entry: str | tuple[str, str]) -> tuple[str, str] | None:
        if isinstance(entry, tuple):
            return entry if entry in val_by_ms else None
        for st in SUBTYPE_PREFERENCE:
            if (entry, st) in val_by_ms:
                return (entry, st)
        return None

    # Contenders become standalone single-method line groups with their own family color.
    groups = dict(family_groups)
    for entry in contenders:
        name = _contender_name(entry)
        groups[name] = {"family": name, "subtype": CONTENDER_SUBTYPE, "methods": [entry]}

    keys_by_group = {g: [k for k in (resolve(e) for e in cfg["methods"]) if k is not None] for g, cfg in groups.items()}
    groups = {g: cfg for g, cfg in groups.items() if keys_by_group[g]}
    if not groups:
        return None

    # For each family, the member model with the best metric *averaged across subsets* — named as the
    # family's representative in the legend (rather than whichever member happens to be listed first).
    family_member_keys: dict[str, list] = {}
    for gname, cfg in groups.items():
        family_member_keys.setdefault(cfg["family"], []).extend(keys_by_group[gname])
    best_avg_method: dict[str, str] = {}
    for fam, keys in family_member_keys.items():
        by_method: dict[str, list] = {}
        for method, subtype in keys:
            by_method.setdefault(method, []).append(val_by_ms[(method, subtype)])
        method_avg: dict[str, float] = {}
        for method, arrs in by_method.items():
            # Per subset take the method's best subtype, then average those bests across subsets.
            per_subset_best = (np.nanmax if spec.higher_is_better else np.nanmin)(np.vstack(arrs), axis=0)
            if not np.isnan(per_subset_best).all():
                method_avg[method] = float(np.nanmean(per_subset_best))
        if method_avg:
            best_avg_method[fam] = min(method_avg.items(), key=lambda kv: spec.sort_value(kv[1]))[0]

    # Per group: best value per subset + the best method's CI as the whisker.
    agg: dict[str, np.ndarray] = {}
    whisker_low: dict[str, np.ndarray] = {}
    whisker_high: dict[str, np.ndarray] = {}
    pick_best = np.nanmax if spec.higher_is_better else np.nanmin
    arg_best = np.nanargmax if spec.higher_is_better else np.nanargmin
    for gname in groups:
        val_stack = np.vstack([val_by_ms[k] for k in keys_by_group[gname]])
        lo_stack = np.vstack([lo_by_ms.get(k, np.full(len(order), np.nan)) for k in keys_by_group[gname]])
        hi_stack = np.vstack([hi_by_ms.get(k, np.full(len(order), np.nan)) for k in keys_by_group[gname]])
        n_cols = val_stack.shape[1]
        best = np.full(n_cols, np.nan)
        low = np.full(n_cols, np.nan)
        high = np.full(n_cols, np.nan)
        for i in range(n_cols):
            col = val_stack[:, i]
            if np.isnan(col).all():
                continue
            best[i] = pick_best(col)
            j = int(arg_best(col))
            if not np.isnan(lo_stack[j, i]):
                low[i] = col[j] - lo_stack[j, i]
            if not np.isnan(hi_stack[j, i]):
                high[i] = col[j] + hi_stack[j, i]
        agg[gname], whisker_low[gname], whisker_high[gname] = best, low, high

    families: list[str] = []
    for cfg in groups.values():
        if cfg["family"] not in families:
            families.append(cfg["family"])
    palette = sns.color_palette("bright", n_colors=max(4, len(families)))
    family_color = dict(zip(families, palette, strict=False))  # palette is padded to >= 4 colors

    fig, ax = plt.subplots(figsize=(max(6.0, 1.1 * len(order) + 0.3), 4.2))
    _draw_x_scaffolding(ax, order, column_guides=True, left=-0.4, right=0.15)

    # Per-column z-rank: the better group at a column draws its marker on top.
    rank_at: dict[tuple[str, int], int] = {}
    for i in range(len(order)):
        pairs = [(g, agg[g][i]) for g in groups if not np.isnan(agg[g][i])]
        pairs.sort(key=lambda kv: spec.sort_value(kv[1]), reverse=True)  # worst first
        for rank, (g, _v) in enumerate(pairs):
            rank_at[(g, i)] = rank

    for gname, cfg in groups.items():
        color = family_color[cfg["family"]]
        marker, linestyle, _label = SUBTYPE_STYLES[cfg["subtype"]]
        y = agg[gname]
        ax.plot(
            range(len(order)),
            y,
            color=color,
            linewidth=_LINE_WIDTH,
            alpha=_LINE_ALPHA,
            linestyle=linestyle,
            zorder=_CONNECTOR_ZORDER,
            solid_capstyle="round",
        )
        for i in range(len(order)):
            lo, hi = whisker_low[gname][i], whisker_high[gname][i]
            if not (np.isnan(lo) or np.isnan(hi) or lo == hi):
                _draw_whisker(ax, i, lo, hi, color)
            if not np.isnan(y[i]):
                _draw_hollow_marker(ax, i, y[i], marker, color, _MARKER_ZORDER_BASE + rank_at[(gname, i)])

    all_vals = np.concatenate(
        [agg[g] for g in groups] + [whisker_low[g] for g in groups] + [whisker_high[g] for g in groups],
    )
    _y_axis(ax, spec, np.nanmin(all_vals), np.nanmax(all_vals))
    ax.set_ylabel(spec.ylabel_family)
    _annotate_n_datasets(ax, order, _n_datasets_per_subset(df))

    _per_family_legend(ax, spec, groups, keys_by_group, agg, family_color, best_avg_method)
    fig.tight_layout()
    return fig


def _per_family_legend(ax: Axes, spec: MetricSpec, groups, keys_by_group, agg, family_color, best_avg_method) -> None:
    """Two stacked rows above the axes: pipeline (subtype) styles and family color swatches."""
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker
    from matplotlib.patches import Rectangle

    family_to_groups: dict[str, list[str]] = {}
    family_methods: dict[str, list[str]] = {}
    for gname, cfg in groups.items():
        family_to_groups.setdefault(cfg["family"], []).append(gname)
        seen = family_methods.setdefault(cfg["family"], [])
        for method, _st in keys_by_group[gname]:
            if method not in seen:
                seen.append(method)

    def anchor(family: str) -> float:
        for st in ANCHOR_PRIORITY:
            for gname in family_to_groups[family]:
                if groups[gname]["subtype"] == st and not np.isnan(agg[gname][-1]):
                    return spec.sort_value(float(agg[gname][-1]))
        return float("inf")

    family_order = sorted(family_to_groups, key=anchor)

    def subtype_entry(st: str) -> HPacker:
        marker, linestyle, label = SUBTYPE_STYLES[st]
        da = DrawingArea(30, 14, 0, 0)
        da.add_artist(Line2D([0, 30], [7, 7], color="black", linestyle=linestyle, linewidth=_LINE_WIDTH))
        da.add_artist(
            Line2D(
                [15],
                [7],
                color="black",
                linestyle="none",
                marker=marker,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=_MARKER_EDGE_WIDTH,
                markersize=_MARKER_SIZE + 1,
            ),
        )
        return HPacker(children=[da, TextArea(label, textprops={"fontsize": 11})], pad=0, sep=4, align="center")

    def family_entry(family: str) -> HPacker:
        swatch = DrawingArea(18, 11, 0, 0)
        swatch.add_artist(Rectangle((0, 0), 18, 11, facecolor=family_color[family], edgecolor="none"))
        children = [swatch, TextArea(family, textprops={"fontsize": 11})]
        methods = family_methods.get(family, [])
        if methods != [family]:  # contender lines: the family *is* the method, skip the duplicate
            # Name the family's best-on-average member (+ ellipsis when there are more), not the first listed.
            rep = best_avg_method.get(family, methods[0])
            label = rep if len(methods) == 1 else f"{rep}, ..."
            children.append(TextArea(f"({label})", textprops={"fontsize": 8}))
        return HPacker(children=children, pad=0, sep=4, align="center")

    subtypes_present = [st for st in SUBTYPE_ORDER if any(cfg["subtype"] == st for cfg in groups.values())]
    subtype_row = HPacker(
        children=[
            TextArea("Pipeline:", textprops={"fontsize": 11, "fontweight": "bold"}),
            *[subtype_entry(st) for st in subtypes_present],
        ],
        pad=0,
        sep=14,
        align="center",
    )
    family_row = HPacker(
        children=[
            TextArea("Family:", textprops={"fontsize": 11, "fontweight": "bold"}),
            *[family_entry(fam) for fam in family_order],
        ],
        pad=0,
        sep=14,
        align="center",
    )
    box = AnchoredOffsetbox(
        loc="lower center",
        child=VPacker(children=[subtype_row, family_row], pad=0, sep=6, align="left"),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        bbox_transform=ax.transAxes,
        borderpad=0.4,
    )
    ax.add_artist(box)


# --- Per-model plot -----------------------------------------------------------------


def _per_model_figure(
    df: pd.DataFrame,
    order: list[str],
    spec: MetricSpec,
    marker_groups: dict[str, dict],
    contenders: list[str | tuple[str, str]],
    models: Sequence[str] | None,
):
    # One row per (subset, method): the highest-priority subtype available for that method.
    priority = {st: i for i, st in enumerate(SUBTYPE_PREFERENCE)}
    df = df[df["method_subtype"].isin(priority)].copy()
    df["_pref"] = df["method_subtype"].map(priority)
    df = df.sort_values("_pref").drop_duplicates(subset=["leaderboard", "method"], keep="first")
    if models is not None:
        df = df[df["method"].isin(models)]
    if df.empty:
        return None

    val_by_m = {m: v for (m,), v in _series_by(df, order, ["method"], spec.column).items()}
    lo_by_m = {m: v for (m,), v in _series_by(df, order, ["method"], spec.err_lo_column).items()}
    hi_by_m = {m: v for (m,), v in _series_by(df, order, ["method"], spec.err_hi_column).items()}
    imputed_by_m = {m: v for (m,), v in _series_by(df, order, ["method"], "imputed").items()}

    contender_names = [_contender_name(e) for e in contenders if _contender_name(e) in val_by_m]
    marker_groups = dict(marker_groups)
    if contender_names:
        marker_groups = {
            "Contender": {"methods": contender_names, "marker": CONTENDER_MARKER, "zboost": CONTENDER_ZBOOST},
            **marker_groups,
        }

    def marker_for(method: str) -> str:
        for cfg in marker_groups.values():
            if method in cfg["methods"]:
                return cfg["marker"]
        return "D"

    def zboost_for(method: str) -> int:
        for cfg in marker_groups.values():
            if method in cfg["methods"]:
                return cfg.get("zboost", 0)
        return 0

    # Colors keyed on best-value order so the strongest methods get the leading palette slots.
    pick_best = np.nanmax if spec.higher_is_better else np.nanmin
    method_order = sorted(val_by_m, key=lambda m: spec.sort_value(pick_best(val_by_m[m])))
    palette = sns.color_palette("bright", n_colors=max(10, len(method_order)))
    color_of = {m: palette[i] for i, m in enumerate(method_order)}
    color_of.update({m: c for m, c in COLOR_OVERRIDES.items() if m in color_of})

    # Anchor subset ("full" if present) drives the fixed per-column slot order and the legend.
    anchor_idx = order.index("full") if "full" in order else len(order) - 1
    by_anchor = sorted(
        method_order,
        key=lambda m: (np.isnan(val_by_m[m][anchor_idx]), spec.sort_value(val_by_m[m][anchor_idx])),
    )
    n = len(by_anchor)
    slot = {m: ((i - (n - 1) / 2) / (n - 1) * _DODGE_WIDTH if n > 1 else 0.0) for i, m in enumerate(by_anchor)}

    fig, ax = plt.subplots(figsize=(max(7.0, 1.35 * len(order) - 0.2), 4.2))
    _draw_x_scaffolding(ax, order, column_guides=False, left=-0.52, right=0.45)

    vals = np.concatenate([val_by_m[m] for m in method_order])
    los = np.concatenate([val_by_m[m] - lo_by_m.get(m, np.zeros(len(order))) for m in method_order])
    his = np.concatenate([val_by_m[m] + hi_by_m.get(m, np.zeros(len(order))) for m in method_order])
    baseline = _y_axis(ax, spec, np.nanmin(np.concatenate([vals, los])), np.nanmax(np.concatenate([vals, his])))
    ax.set_ylabel(spec.ylabel)
    _annotate_n_datasets(ax, order, _n_datasets_per_subset(df))

    flagged_imputed = {m for m in method_order if np.nansum(imputed_by_m.get(m, np.zeros(1))) > 0}

    bar_width = _DODGE_WIDTH / max(n - 1, 1) * 0.95
    for col_idx in range(len(order)):
        entries = [(m, val_by_m[m][col_idx]) for m in method_order if not np.isnan(val_by_m[m][col_idx])]
        # Within each zboost tier, the better method draws on top.
        rank_in_tier: dict[str, int] = {}
        for boost in {zboost_for(m) for m, _ in entries}:
            tier = sorted([e for e in entries if zboost_for(e[0]) == boost], key=lambda kv: spec.sort_value(kv[1]))
            for rank, (m, _v) in enumerate(reversed(tier)):
                rank_in_tier[m] = rank
        for method, value in entries:
            x = col_idx + slot[method]
            color = color_of[method]
            ax.bar(
                x,
                value - baseline,
                bottom=baseline,
                width=bar_width,
                color=color,
                alpha=_BAR_ALPHA,
                edgecolor=color,
                linewidth=0.6,
                zorder=4,
            )
            imp = imputed_by_m.get(method, np.full(len(order), np.nan))[col_idx]
            if method in flagged_imputed and not np.isnan(imp) and imp > 0:
                # Separate opaque hatch overlay: the PDF backend applies the bar's alpha to
                # hatch lines too, which would make them nearly invisible.
                ax.bar(
                    x,
                    value - baseline,
                    bottom=baseline,
                    width=bar_width,
                    facecolor="none",
                    edgecolor=color,
                    linewidth=0,
                    hatch="//",
                    zorder=4.1,
                )
            e_lo = lo_by_m.get(method, np.full(len(order), np.nan))[col_idx]
            e_hi = hi_by_m.get(method, np.full(len(order), np.nan))[col_idx]
            if not (np.isnan(e_lo) or np.isnan(e_hi)):
                _draw_whisker(ax, x, value - e_lo, value + e_hi, color)
            zorder = _MARKER_ZORDER_BASE + zboost_for(method) + rank_in_tier[method]
            _draw_hollow_marker(ax, x, value, marker_for(method), color, zorder)

    _per_model_legend(
        fig,
        ax,
        marker_groups,
        method_order=by_anchor,
        marker_for=marker_for,
        color_of=color_of,
        flagged_imputed=flagged_imputed,
    )
    fig.subplots_adjust(top=0.81, bottom=0.13, left=0.05, right=0.995)
    return fig


def _per_model_legend(fig, ax: Axes, marker_groups, *, method_order, marker_for, color_of, flagged_imputed) -> None:
    """Stacked rows above the axes: family marker shapes, then the model entries across two rows."""
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text

    title_fontsize, entry_fontsize = 12, 11
    legend_marker_size = _MARKER_SIZE + 3

    active_groups = [
        (gname, cfg["marker"]) for gname, cfg in marker_groups.items() if any(m in method_order for m in cfg["methods"])
    ]
    active_groups.sort(key=lambda gm: -marker_groups[gm[0]].get("zboost", 0))

    def family_entry(gname: str, marker: str) -> HPacker:
        da = DrawingArea(20, 14, 0, 0)
        da.add_artist(
            Line2D(
                [10],
                [7],
                color="black",
                linestyle="none",
                marker=marker,
                markerfacecolor="none",
                markeredgecolor="black",
                markeredgewidth=_MARKER_EDGE_WIDTH,
                markersize=legend_marker_size,
            ),
        )
        return HPacker(
            children=[da, TextArea(gname, textprops={"fontsize": entry_fontsize})], pad=0, sep=4, align="center"
        )

    def imputed_entry() -> HPacker:
        swatch = DrawingArea(24, 14, 0, 0)
        swatch.add_artist(
            Rectangle((1, 0), 22, 14, facecolor="white", edgecolor="black", hatch="///", linewidth=0.9),
        )
        label = TextArea("Partially imputed scores", textprops={"fontsize": entry_fontsize})
        return HPacker(children=[swatch, label], pad=0, sep=4, align="center")

    def model_entry(method: str) -> HPacker:
        color = color_of[method]
        da = DrawingArea(20, 14, 0, 0)
        da.add_artist(
            Line2D(
                [10],
                [7],
                color=color,
                linestyle="none",
                marker=marker_for(method),
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=_MARKER_EDGE_WIDTH,
                markersize=legend_marker_size,
            ),
        )
        text = f"{method}*" if method in flagged_imputed else method
        label = TextArea(text, textprops={"fontsize": entry_fontsize, "color": color, "fontweight": "semibold"})
        return HPacker(children=[da, label], pad=0, sep=4, align="center")

    # Pad the shorter row title so the first marker in both rows lines up vertically.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def title_width(text: str) -> float:
        t = Text(0, 0, text, fontsize=title_fontsize, fontweight="bold")
        t.set_figure(fig)
        return t.get_window_extent(renderer).width

    max_width = max(title_width("Family:"), title_width("Model:"))

    def padded_title(text: str):
        title = TextArea(text, textprops={"fontsize": title_fontsize, "fontweight": "bold"})
        pad_w = max_width - title_width(text)
        if pad_w <= 0:
            return title
        return HPacker(children=[title, DrawingArea(pad_w, 1, 0, 0)], pad=0, sep=0, align="center")

    family_children = [family_entry(gn, mk) for gn, mk in active_groups]
    if flagged_imputed:
        vsep = DrawingArea(3, 18, 0, 0)
        vsep.add_artist(Line2D([1.5, 1.5], [0, 18], color="black", linewidth=2.2, solid_capstyle="butt"))
        family_children += [vsep, imputed_entry()]

    def titled_row(title_text: str, children: list) -> HPacker:
        inner = HPacker(children=children, pad=0, sep=10, align="center")
        return HPacker(children=[padded_title(title_text), inner], pad=0, sep=10, align="center")

    # Family markers on the first row; the model entries wrap across two rows (too many models to
    # read comfortably in a single row). VPacker stacks them family -> model row 1 -> model row 2.
    model_entries = [model_entry(m) for m in method_order]
    half = (len(model_entries) + 1) // 2
    legend_rows = [
        titled_row("Family:", family_children),
        titled_row("Model:", model_entries[:half]),
        titled_row("", model_entries[half:]),
    ]
    box = AnchoredOffsetbox(
        loc="lower left",
        child=VPacker(children=legend_rows, pad=0, sep=6, align="left"),
        frameon=False,
        bbox_to_anchor=(0.0, 1.02),
        bbox_transform=ax.transAxes,
        borderpad=0.4,
    )
    ax.add_artist(box)
