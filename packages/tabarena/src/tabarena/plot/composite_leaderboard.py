"""Composite leaderboard across evaluation subsets.

Builds a single table from per-subset website-format leaderboards
(the compact output of :func:`tabarena.website.website_format.format_leaderboard`,
i.e. ``compact=True``) and renders it as CSV + color-graded PNGs.

Layout of the composite:
    rows : MultiIndex (method, metric) where metric ∈ {"Elo", "Improv%"}
    cols : one per subset (e.g. ``all``, ``binary``, ``regression``)

Methods are ordered by Elo on the sort-by subset (descending).

Entry points:
    * :func:`generate_composite_leaderboard` — full pipeline from
      ``{subset_name: leaderboard_df}`` to CSV + PNG artifacts. Called by
      :meth:`AbstractArenaContext.generate_all_figs` when
      ``save_composite_leaderboard=True``.
    * :func:`load_subset_leaderboards` — load the per-subset CSVs written
      by ``generate_all_figs(save_website_leaderboard=True)`` from disk,
      for standalone/offline aggregation scripts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

# Source columns to pull from each per-subset leaderboard. The display
# name and scale are handled separately in `METRIC_DISPLAY`. Column names
# match those produced by `format_leaderboard(..., compact=True)`.
METRICS: tuple[str, ...] = ("Elo", "Impro%")

# Per-metric display config:
#   display_name: row label in the composite (substituted for the source name)
#   scale:        multiplier applied to the source values. Both 1.0:
#                 the website-format CSV already scales `Impro%` to percent.
#   higher_better: controls the cmap direction. True = high values colored
#                 toward the "good" end of the cmap; False = inverted.
#   format:       Python format string used to render each cell value in
#                 the PNG (Elo is rendered as an integer; Improv% keeps
#                 one decimal).
#   title_name:   long-form label used in the figure title for per-metric
#                 PNGs (e.g. "Improvability (%)"). Falls back to
#                 ``display_name`` when not set.
METRIC_DISPLAY: dict[str, dict] = {
    "Elo": {"display_name": "Elo", "scale": 1.0, "higher_better": True, "format": "{:.0f}"},
    "Impro%": {
        "display_name": "Improv%",
        "scale": 1.0,
        "higher_better": False,
        "format": "{:.1f}",
        "title_name": "Improvability (%)",
    },
}

LEADERBOARD_FILENAME: str = "leaderboard_website.csv"
# Method column in the website-format leaderboard (renamed from "method"
# by `format_leaderboard`).
METHOD_COL: str = "Model"
# Source column used to order methods on the sort-by subset.
ELO_COL: str = "Elo"

# Preferred column ordering for the composite table. Subsets in this
# tuple appear first (in this order); any remaining subsets are appended
# alphabetically. Matched after `&lite` stripping (see `strip_lite`),
# so list bare names ("all") rather than the on-disk form ("all&lite").
SUBSET_ORDER: tuple[str, ...] = (
    "all",
    "small",
    "medium",
    "binary",
    "multiclass",
    "regression",
)

# Trailing tuning-variant suffixes recognised on method names. Order
# matters: ``(tuned + ensembled)`` must come before ``(tuned)`` so the
# longest match wins inside ``_split_tuning_suffix``.
TUNING_SUFFIXES: tuple[str, ...] = (
    " (tuned + ensembled)",
    " (tuned)",
    " (default)",
)

# Text colors applied to the top-3 cells (per subset, per metric) on
# the PNG output. Indexed by rank: 0 = best. Gold uses ``goldenrod``
# (#DAA520) — yellow enough to read clearly against bronze (#CD7F32,
# orange-brown) while still legible on the dark-green end of the RdYlGn
# background. Silver is the standard medal hue.
MEDAL_COLORS: tuple[str, str, str] = ("#DAA520", "#C0C0C0", "#CD7F32")


def load_subset_leaderboards(
    input_dir: Path,
    filename: str = LEADERBOARD_FILENAME,
) -> dict[str, pd.DataFrame]:
    """Return ``{subset_folder_name: leaderboard_df}``, indexed by method."""
    out: dict[str, pd.DataFrame] = {}
    for subdir in sorted(Path(input_dir).iterdir()):
        if not subdir.is_dir():
            continue
        csv_path = subdir / filename
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path).set_index(METHOD_COL)
        out[subdir.name] = df
    return out


def _split_tuning_suffix(method: str) -> tuple[str, str]:
    """Split ``method`` into ``(base, suffix)`` where ``suffix`` is one of
    ``TUNING_SUFFIXES`` (with its leading space) or ``""`` if none match.
    """
    for suffix in TUNING_SUFFIXES:
        if method.endswith(suffix):
            return method[: -len(suffix)], suffix
    return method, ""


def collapse_tuning_variants(
    leaderboards: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Collapse tuning-variant rows per subset.

    Within each per-subset leaderboard:

    * If ``<base> (tuned + ensembled)`` is present, drop the matching
      ``<base> (tuned)`` and ``<base> (default)`` rows for the same base.
    * Rename surviving ``(tuned + ensembled)`` rows to ``(T+E)``.
    * Strip ``(default)`` from any remaining method names so e.g.
      ``TabPFN-3 (default)`` becomes ``TabPFN-3``.

    Methods without a recognised tuning suffix (or with ``(tuned)`` and no
    sibling T+E) pass through unchanged.
    """
    out: dict[str, pd.DataFrame] = {}
    for subset_name, lb in leaderboards.items():
        parsed = [_split_tuning_suffix(m) for m in lb.index]
        bases_with_te = {base for base, suffix in parsed if suffix == " (tuned + ensembled)"}

        keep_mask: list[bool] = []
        new_names: list[str] = []
        for method, (base, suffix) in zip(lb.index, parsed, strict=True):
            if base in bases_with_te and suffix in (" (default)", " (tuned)"):
                keep_mask.append(False)
                new_names.append(method)
                continue
            keep_mask.append(True)
            if suffix == " (tuned + ensembled)":
                new_names.append(f"{base} (T+E)")
            elif suffix == " (default)":
                new_names.append(base)
            else:
                new_names.append(method)

        filtered = lb[keep_mask].copy()
        filtered.index = pd.Index(
            [n for n, k in zip(new_names, keep_mask, strict=True) if k],
            name=lb.index.name,
        )
        out[subset_name] = filtered
    return out


def filter_excluded_methods(
    leaderboards: dict[str, pd.DataFrame],
    excluded_prefixes: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    """Drop methods whose names start with any entry in ``excluded_prefixes``."""
    if not excluded_prefixes:
        return dict(leaderboards)
    out: dict[str, pd.DataFrame] = {}
    for subset_name, lb in leaderboards.items():
        keep = [not any(str(m).startswith(p) for p in excluded_prefixes) for m in lb.index]
        out[subset_name] = lb[keep]
    return out


def _strip_lite(name: str) -> str:
    """Remove the ``lite`` token from a ``&``-joined subset name. Returns
    ``"lite"`` when the name was exactly that (so we never collapse to "").
    """
    parts = [p for p in name.split("&") if p != "lite"]
    return "&".join(parts) if parts else "lite"


def normalize_subset_names(
    leaderboards: dict[str, pd.DataFrame],
    strip_lite: bool | None = None,
) -> dict[str, pd.DataFrame]:
    """Optionally strip ``&lite`` from each subset name.

    ``strip_lite=None`` auto-detects: strips only when every loaded subset
    contains ``lite`` (i.e. the eval was lite-restricted across the board).
    """
    if strip_lite is None:
        strip_lite = bool(leaderboards) and all("lite" in name.split("&") for name in leaderboards)
    if not strip_lite:
        return dict(leaderboards)

    renamed: dict[str, pd.DataFrame] = {}
    for name, lb in leaderboards.items():
        new_name = _strip_lite(name)
        if new_name in renamed:
            raise ValueError(f"Name collision after stripping &lite: {name!r} collides with existing {new_name!r}")
        renamed[new_name] = lb
    return renamed


def order_subsets(
    present: list[str],
    preferred: tuple[str, ...] = SUBSET_ORDER,
) -> list[str]:
    """Return ``present`` ordered by ``preferred`` first (in that order),
    then any remaining names in alphabetical order.
    """
    present_set = set(present)
    head = [s for s in preferred if s in present_set]
    head_set = set(head)
    tail = sorted(s for s in present if s not in head_set)
    return head + tail


def resolve_sort_subset(leaderboards: dict[str, pd.DataFrame], sort_by: str) -> str:
    """Pick the per-subset name to sort by. Exact match wins; otherwise
    fall back to ``<sort_by>&<extra>`` (handles the lite-restriction
    renaming ``all`` → ``all&lite``).
    """
    if sort_by in leaderboards:
        return sort_by
    candidates = [name for name in leaderboards if name.startswith(f"{sort_by}&")]
    if not candidates:
        raise ValueError(f"No subset matching {sort_by!r} found in {sorted(leaderboards)}")
    # Prefer the shortest match (closest to a bare `sort_by`).
    return min(candidates, key=len)


def build_composite(
    leaderboards: dict[str, pd.DataFrame],
    metrics: tuple[str, ...] = METRICS,
    sort_by_subset: str = "all",
    subset_order: tuple[str, ...] = SUBSET_ORDER,
) -> pd.DataFrame:
    sort_subset = resolve_sort_subset(leaderboards, sort_by_subset)
    method_order = leaderboards[sort_subset][ELO_COL].sort_values(ascending=False).index.tolist()

    # Stack metrics into rows for each subset, then concat horizontally.
    # `lb[metrics].stack()` yields a Series with MultiIndex (method, metric).
    pieces: list[pd.Series] = []
    for subset_name, lb in leaderboards.items():
        missing = [m for m in metrics if m not in lb.columns]
        if missing:
            raise KeyError(
                f"Subset {subset_name!r} is missing metric column(s) {missing}; available columns: {list(lb.columns)}"
            )
        # Apply per-metric scaling before stacking. Both metrics ship at
        # their final scale in the website-format CSV, so the multiplier
        # is currently 1.0; kept as a hook for future per-metric tweaks.
        scaled = lb[list(metrics)].copy()
        for m in metrics:
            scaled[m] = scaled[m] * METRIC_DISPLAY[m]["scale"]
        series = scaled.stack()
        series.name = subset_name
        series.index.names = ["method", "metric"]
        pieces.append(series)
    composite = pd.concat(pieces, axis=1)

    # Rename metric labels to their display form (e.g. Impro% → Improv%).
    rename_map = {m: METRIC_DISPLAY[m]["display_name"] for m in metrics}
    composite.index = composite.index.set_levels(
        composite.index.levels[1].map(lambda m: rename_map.get(m, m)),
        level="metric",
    )

    # Reorder rows: methods follow `method_order`; metrics keep input order
    # but use their display names.
    target_index = pd.MultiIndex.from_product(
        [method_order, [METRIC_DISPLAY[m]["display_name"] for m in metrics]],
        names=["method", "metric"],
    )
    composite = composite.reindex(target_index)

    # Column order: subsets listed in `subset_order` come first (in that
    # order), then any remaining subsets alphabetically. The resolved
    # sort-subset is hoisted to the very front so the column the rows
    # were ranked by is read first regardless of `subset_order`.
    ordered = order_subsets(list(composite.columns), preferred=subset_order)
    if sort_subset in ordered:
        ordered = [sort_subset] + [c for c in ordered if c != sort_subset]
    return composite[ordered]


def select_top_n_methods(composite: pd.DataFrame, top_n: int | None) -> pd.DataFrame:
    """Keep the first ``top_n`` unique methods of ``composite``.

    ``build_composite`` already orders methods by descending Elo on the
    sort-by subset, so the first N unique methods are the top-N. ``None``
    or ``<= 0`` returns the composite unchanged.
    """
    if not top_n or top_n <= 0:
        return composite
    method_order = composite.index.get_level_values("method")
    seen: list[str] = []
    for m in method_order:
        if m not in seen:
            seen.append(m)
        if len(seen) >= top_n:
            break
    return composite.loc[method_order.isin(seen)]


def _group_method_rows(method_per_row: list[str]) -> list[tuple[str, list[int]]]:
    """Group consecutive rows by method name. Returns a list of
    ``(method, [row_indices])`` preserving the input row order.
    """
    groups: list[tuple[str, list[int]]] = []
    for i, m in enumerate(method_per_row):
        if groups and groups[-1][0] == m:
            groups[-1][1].append(i)
        else:
            groups.append((m, [i]))
    return groups


def render_composite_png(
    composite: pd.DataFrame,
    output_path: Path,
    cmap_name: str = "RdYlGn",
    title: str | None = None,
) -> None:
    """Render the composite table to PNG with per-(subset, metric) color gradients.

    Each (subset, metric) cell block is normalized independently — so the
    cmap stretches across each subset's own range rather than sharing a
    scale across subsets. ``higher_better`` from ``METRIC_DISPLAY``
    controls which end of the cmap is "good": for Elo, high → green; for
    Improv%, low → green.

    When the composite carries more than one metric per method, the method
    name is shown once per group and centered vertically between the
    group's rows; the metric label lives in its own narrow first column
    so each data row still self-identifies. With a single metric per
    method the metric column is omitted (it would just repeat one value)
    and the method label sits on its single row.
    """
    cmap = plt.get_cmap(cmap_name)
    n_rows, n_cols = composite.shape

    # Build a per-cell RGBA array shaped like the composite. NaNs render
    # as white so missing cells don't pull a color from the gradient.
    cell_colors = np.full((n_rows, n_cols, 4), 1.0)

    # Map each display-name back to its source spec so we know
    # higher_better and how to format values. Display names are unique.
    display_to_spec = {spec["display_name"]: spec for spec in METRIC_DISPLAY.values()}

    metric_level = composite.index.get_level_values("metric")
    composite_values = composite.to_numpy(dtype=float)
    for j in range(n_cols):
        col_values = composite_values[:, j]
        for display_name, spec in display_to_spec.items():
            mask = metric_level == display_name
            if not mask.any():
                continue
            rows_in_metric = np.where(mask)[0]
            vals = col_values[rows_in_metric]
            finite_mask = np.isfinite(vals)
            if finite_mask.sum() == 0:
                continue
            finite_vals = vals[finite_mask]
            vmin, vmax = float(finite_vals.min()), float(finite_vals.max())
            # Avoid a zero-range Normalize (would map every value to NaN).
            # When all cells share one value, just paint mid-cmap (yellow).
            if vmin == vmax:
                cell_colors[rows_in_metric[finite_mask], j] = cmap(0.5)
                continue
            norm = Normalize(vmin=vmin, vmax=vmax)
            for k, row_idx in enumerate(rows_in_metric):
                val = vals[k]
                if not np.isfinite(val):
                    continue
                t = norm(val)
                if not spec["higher_better"]:
                    t = 1.0 - t
                cell_colors[row_idx, j] = cmap(t)

    metric_per_row = list(metric_level)
    method_per_row = list(composite.index.get_level_values("method"))
    method_groups = _group_method_rows(method_per_row)
    unique_metrics = list(dict.fromkeys(metric_per_row))
    show_metric_column = len(unique_metrics) > 1

    # Build cell text. With a single metric the metric label is redundant
    # so we drop the column entirely.
    cell_text: list[list[str]] = []
    for i, row in enumerate(composite_values):
        met = metric_per_row[i]
        fmt = display_to_spec[met].get("format", "{:.2f}")
        formatted = [fmt.format(v) if np.isfinite(v) else "—" for v in row]
        if show_metric_column:
            cell_text.append([met, *formatted])
        else:
            cell_text.append(formatted)

    if show_metric_column:
        metric_col_colors = np.full((n_rows, 1, 4), 1.0)
        cell_colors_full = np.concatenate([metric_col_colors, cell_colors], axis=1)
        col_labels = ["", *list(composite.columns)]
    else:
        cell_colors_full = cell_colors
        col_labels = list(composite.columns)

    # Show the method name once per group, on the group's first row.
    row_labels = [""] * n_rows
    for method, indices in method_groups:
        row_labels[indices[0]] = method

    # Figure size: with ``bbox=[0,0,1,1]`` the table stretches to fill the
    # axes, so column widths are set as axes fractions. Size the figure from
    # physical (inch) column targets so each column keeps its width at any
    # subset count: the row-label column fits the longest method name (e.g.
    # "AutoGluon 1.5 (extreme, 4h)"), each subset column fits its longest
    # header (e.g. "classification") at font size 8. A small extra slice of
    # height is reserved at the top when a title is provided.
    rowlabel_in = 1.9
    metric_in = 0.6 if show_metric_column else 0.0
    subset_in = 0.85
    title_h = 0.35 if title else 0.0
    fig_w = rowlabel_in + metric_in + subset_in * n_cols
    fig_h = max(2.5, 0.35 + 0.13 * (n_rows + 1)) + title_h
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors_full.tolist(),
        cellLoc="center",
        # ``bbox=[0,0,1,1]`` stretches the table to fill the axes so the
        # right-hand side has no leftover whitespace.
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # Reduce internal cell padding so rows hug their content; default is
    # 0.1 (10% of cell). Applies to every cell.
    for cell in table.get_celld().values():
        cell.PAD = 0.005

    # Override the auto-divided column widths to the physical targets the
    # figure was sized from. Widths are axes-fraction; with ``bbox=[0,0,1,1]``
    # the table fills the axes so fraction x fig_w = the inch targets above.
    rowlabel_w = rowlabel_in / fig_w
    metric_w = metric_in / fig_w
    subset_col_width = subset_in / fig_w
    for (_row, col), cell in table.get_celld().items():
        if col == -1:
            cell.set_width(rowlabel_w)
        elif col == 0 and show_metric_column:
            cell.set_width(metric_w)
        else:
            cell.set_width(subset_col_width)

    # Center each method label vertically across its group of rows. The
    # label is attached to the first row's cell at y=0.5 (cell center).
    # Shifting y to ``0.5 - (n-1)/2`` puts the label at the visual middle
    # of an n-row group (no-op for single-row groups). ``clip_on=False``
    # lets the text extend past the cell boundary.
    for _method, indices in method_groups:
        n = len(indices)
        if n <= 1:
            continue
        cell = table[(indices[0] + 1, -1)]  # +1 to skip the header row.
        txt = cell.get_text()
        txt.set_verticalalignment("center")
        txt.set_y(0.5 - (n - 1) / 2)
        txt.set_clip_on(False)

    # Highlight top-3 cells per (subset, metric) with gold/silver/bronze
    # text. Ranking respects `higher_better`: Elo top-3 are the largest
    # values; Improv% top-3 are the smallest.
    metric_row_indices = np.arange(n_rows)
    data_col_offset = 1 if show_metric_column else 0
    for j in range(n_cols):
        col_values = composite_values[:, j]
        for display_name, spec in display_to_spec.items():
            mask = metric_level == display_name
            if not mask.any():
                continue
            rows_in_metric = metric_row_indices[mask]
            vals = col_values[rows_in_metric]
            finite = np.isfinite(vals)
            if finite.sum() == 0:
                continue
            finite_idx = np.where(finite)[0]
            order = np.argsort(vals[finite_idx])
            if spec["higher_better"]:
                order = order[::-1]
            for rank, k in enumerate(order[: len(MEDAL_COLORS)]):
                actual_row = rows_in_metric[finite_idx[k]]
                # +1 row for the header; +data_col_offset for the metric col.
                cell = table[(actual_row + 1, j + data_col_offset)]
                txt = cell.get_text()
                txt.set_color(MEDAL_COLORS[rank])
                txt.set_fontweight("bold")

    # Drop the axes margins so the table sits flush with the figure
    # edges, reserving the top slice for the title when present.
    if title:
        top_frac = 1.0 - (title_h / fig_h)
        fig.suptitle(title, fontsize=10, fontweight="bold", y=1.0 - 0.4 * (title_h / fig_h))
    else:
        top_frac = 1.0
    fig.subplots_adjust(left=0, right=1, top=top_frac, bottom=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def generate_composite_leaderboard(
    leaderboards: dict[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    filename: str = "composite_leaderboard.csv",
    metrics: tuple[str, ...] = METRICS,
    sort_by: str | None = None,
    strip_lite: bool | None = None,
    collapse_tuning: bool = True,
    excluded_method_prefixes: tuple[str, ...] = (),
    top_n: int | None = None,
    subset_order: tuple[str, ...] = SUBSET_ORDER,
    title: str = "TabArena Leaderboard",
    save_png: bool = True,
    save_per_metric_pngs: bool = True,
) -> pd.DataFrame:
    """Aggregate per-subset website-format leaderboards into one composite.

    Writes ``<output_dir>/<filename>`` (CSV) plus, when enabled, a PNG of
    the full composite and one PNG per metric (suffixed with the metric's
    display name), and returns the composite frame.

    Parameters
    ----------
    leaderboards
        ``{subset_name: leaderboard_df}`` in the compact website format
        (``format_leaderboard(..., compact=True)``); frames may carry the
        method either as the index or as a ``Model`` column.
    sort_by
        Subset whose Elo orders the methods (descending). ``None`` picks
        ``"all"`` when present (including an ``all&<extra>`` variant),
        falling back to the first subset otherwise.
    strip_lite
        Strip the ``&lite`` token from subset names; ``None`` auto-detects
        (strips only when every subset carries it).
    collapse_tuning
        Collapse ``(tuned)``/``(default)`` rows into their sibling
        ``(tuned + ensembled)`` row per :func:`collapse_tuning_variants`.
    excluded_method_prefixes
        Method-name prefixes to drop entirely before ranking / rendering.
    top_n
        Keep only the top-N methods by Elo on the sort-by subset;
        ``None``/``0`` keeps every method.
    """
    leaderboards = {
        name: lb.set_index(METHOD_COL) if METHOD_COL in lb.columns else lb for name, lb in leaderboards.items()
    }
    leaderboards = normalize_subset_names(leaderboards, strip_lite=strip_lite)
    if collapse_tuning:
        leaderboards = collapse_tuning_variants(leaderboards)
    leaderboards = filter_excluded_methods(leaderboards, excluded_prefixes=excluded_method_prefixes)

    if sort_by is None:
        try:
            sort_by = resolve_sort_subset(leaderboards, "all")
        except ValueError:
            sort_by = next(iter(leaderboards))

    composite = build_composite(
        leaderboards,
        metrics=metrics,
        sort_by_subset=sort_by,
        subset_order=subset_order,
    )
    composite = select_top_n_methods(composite, top_n=top_n)

    output_dir = Path(output_dir)
    out_path = output_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composite.to_csv(out_path)

    if save_png:
        png_path = out_path.with_suffix(".png")
        render_composite_png(composite, png_path, title=title)

        if save_per_metric_pngs:
            # Per-metric leaderboard PNGs. Each is a single-metric slice of
            # the composite, rendered through the same pipeline so the
            # colors, medal highlighting, and column widths stay consistent.
            # Filename suffix is derived from the display name (lowercased,
            # ``%`` stripped) so the path is filesystem-friendly.
            metric_level = composite.index.get_level_values("metric")
            display_to_spec = {spec["display_name"]: spec for spec in METRIC_DISPLAY.values()}
            for display_name in dict.fromkeys(metric_level):
                slice_df = composite[metric_level == display_name]
                suffix = display_name.lower().replace("%", "").rstrip("_")
                slice_png_path = png_path.with_name(f"{png_path.stem}_{suffix}.png")
                title_name = display_to_spec.get(display_name, {}).get("title_name", display_name)
                render_composite_png(
                    slice_df,
                    slice_png_path,
                    title=f"{title} — {title_name}",
                )

    return composite
