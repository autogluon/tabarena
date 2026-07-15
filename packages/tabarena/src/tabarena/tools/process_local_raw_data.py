"""Process *already-present* raw TabArena results into method metadata + cached artifacts.

Assumes the raw ``results.pkl`` files are already on local disk (e.g. you unzipped a
submission yourself); there is **no download and no upload**. The three concerns are:

  1. ``inspect``  — scan the raw artifacts and report the fields that drive ``MethodMetadata``
                    construction (method_type, ag_key, compute -> cpu/gpu, is_bag, can_hpo,
                    config_default, ...), plus task/problem-type/metric coverage. Prints a
                    copy-paste ``MethodMetadata.<type>(...)`` snippet built from the inferred values
                    with instructions for turning it into a committed ``info.py``, and — if you
                    already pass an existing ``MethodMetadata`` — an inferred-vs-provided diff.
                    Uses :func:`~tabarena.benchmark.result.raw_loading.scan_raw_info`, which
                    reduces each file to its info row in parallel, so inspection never holds the
                    run's predictions in memory.

  2. ``verify``   — confirm a hand-authored ``MethodMetadata`` is consistent with the raw data
                    (and well-formed) before processing. See :func:`verify_method_metadata`.

  3. ``process``  — build the processed ``EvaluationRepository`` + per-task results and cache
                    them locally (``metadata.yaml`` + ``processed/`` + ``results/`` under the
                    TabArena cache root) via ``EndToEnd.from_path_raw(cache_processed=True)``,
                    which processes and writes each task's slice inside its own worker.

**Processing requires an explicit ``MethodMetadata``.** ``inspect`` can run on raw data alone (and
exists precisely to help you author the metadata), but ``process`` refuses to guess: you must pass a
fully-specified ``MethodMetadata`` (typically imported from a committed ``tabarena.models.<model>``
``info.py``), and :func:`verify_method_metadata` checks it against the raw data before any work is
done — erroring on misalignment unless you opt out with ``ignore_metadata_mismatch=True``.

This module holds the reusable pieces; thin ``scripts/run_process_*`` entry points just build one or
more ``RawMethod`` and call :func:`process_method` (single method) or :func:`process_methods` (a loop over
several).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tabarena.benchmark.result.raw_loading import fetch_raw_result_paths, scan_raw_info
from tabarena.end_to_end import EndToEnd

if TYPE_CHECKING:
    from tabarena.end_to_end import EndToEndResults
    from tabarena.models._method_metadata import MethodMetadata


@dataclass
class RawMethod:
    """A local directory of raw ``results.pkl`` files to inspect, verify and/or process.

    Parameters
    ----------
    path_raw
        Directory containing the method's raw ``results.pkl`` files (recursively). This is the
        already-present raw data; nothing is downloaded.
    method_metadata
        A fully-specified ``MethodMetadata`` (e.g. ``from tabarena.models.tabpfn_3.info import
        tabpfn_3_method_metadata``). **Required for processing** and verified against the raw data
        first; optional for ``inspect`` only. When given, ``name`` / ``suite`` / ``model_key`` below
        are ignored (their values are taken from the metadata).
    name, suite, model_key
        Naming hints used only when ``method_metadata is None`` (i.e. during ``inspect``): they seed
        the suggested-metadata snippet and the inferred-method label. They cannot be used to process
        — author a ``MethodMetadata`` from the snippet and pass it via ``method_metadata`` instead.
    """

    path_raw: Path
    method_metadata: MethodMetadata | None = None
    name: str | None = None
    suite: str | None = None
    model_key: str | None = None

    def __post_init__(self) -> None:
        self.path_raw = Path(self.path_raw)

    # Resolve naming from the metadata when provided, else from the explicit hints.
    @property
    def resolved_name(self) -> str | None:
        if self.name is not None:
            return self.name
        return self.method_metadata.method if self.method_metadata is not None else None

    @property
    def resolved_suite(self) -> str | None:
        if self.suite is not None:
            return self.suite
        return self.method_metadata.suite if self.method_metadata is not None else None

    @property
    def resolved_model_key(self) -> str | None:
        if self.model_key is not None:
            return self.model_key
        return self.method_metadata.model_key if self.method_metadata is not None else None


def _format_py_value(val: object) -> str:
    """Render a Python literal that lints clean under ruff (double-quoted strings)."""
    if isinstance(val, str):
        return json.dumps(val)  # ensures double-quoted, properly escaped
    return repr(val)  # bool/int/float/None render the same under repr and ruff


def _infer_from_raw(method: RawMethod, *, engine: str = "ray", file_paths: list[Path] | None = None) -> dict:
    """Scan the raw artifacts at ``method.path_raw`` and return the values that drive
    ``MethodMetadata`` construction plus task/problem-type/metric coverage.

    The scan reduces each ``results.pkl`` to its small info row and discards the predictions
    (see :func:`~tabarena.benchmark.result.raw_loading.scan_raw_info`), so it stays cheap on
    memory; still, callers that both inspect and verify should call this once and pass the
    result into :func:`log_raw_data_info` / :func:`verify_method_metadata`. ``file_paths``
    are pre-discovered ``results.pkl`` paths, skipping the directory walk.
    """
    info_df = scan_raw_info(path_raw=method.path_raw, file_paths=file_paths, engine=engine)

    def _unique(col: str):
        return sorted(info_df[col].dropna().unique().tolist())

    method_types = _unique("method_type")
    model_types = _unique("model_type")
    ag_keys = _unique("ag_key")
    name_prefixes = _unique("name_prefix")
    problem_types = _unique("problem_type")
    metrics = _unique("metric")
    frameworks = _unique("framework")
    num_gpus_vals = _unique("num_gpus")
    is_bag_any = bool(info_df["is_bag"].any())

    inferred_compute = "gpu" if (num_gpus_vals and max(num_gpus_vals) > 0) else "cpu"
    inferred_can_hpo = len(frameworks) > 1
    inferred_config_default = frameworks[0] if len(frameworks) == 1 else None
    inferred_method_type = method_types[0] if len(method_types) == 1 else None
    inferred_ag_key = ag_keys[0] if len(ag_keys) == 1 else None
    # method is inferred from name_prefix for configs, and from framework for baselines.
    if inferred_method_type == "baseline":
        inferred_method = frameworks[0] if len(frameworks) == 1 else None
    else:
        inferred_method = name_prefixes[0] if len(name_prefixes) == 1 else None

    n_tasks = (
        info_df[["tid", "split_idx"]].drop_duplicates().shape[0]
        if {"tid", "split_idx"}.issubset(info_df.columns)
        else None
    )
    n_datasets = info_df["name"].nunique() if "name" in info_df.columns else None
    n_folds = info_df["split_idx"].nunique() if "split_idx" in info_df.columns else None
    # Per-config (framework) coverage: total occurrences (dataset x fold) each config has results for.
    config_task_counts = info_df.groupby("framework").size().sort_index() if "framework" in info_df.columns else None

    return {
        "n_results": len(info_df),
        "method_types": method_types,
        "model_types": model_types,
        "ag_keys": ag_keys,
        "name_prefixes": name_prefixes,
        "problem_types": problem_types,
        "metrics": metrics,
        "frameworks": frameworks,
        "num_gpus_vals": num_gpus_vals,
        "is_bag_any": is_bag_any,
        "inferred_compute": inferred_compute,
        "inferred_can_hpo": inferred_can_hpo,
        "inferred_config_default": inferred_config_default,
        "inferred_method_type": inferred_method_type,
        "inferred_ag_key": inferred_ag_key,
        "inferred_method": inferred_method,
        "n_tasks": n_tasks,
        "n_datasets": n_datasets,
        "n_folds": n_folds,
        "config_task_counts": config_task_counts,
    }


def _expected_config_default(method: RawMethod, inferred: dict) -> str | None:
    """The ``config_default`` name the processed repo will actually contain (post-rename).

    ``from_raw`` renames every config's name_prefix to ``method.resolved_name`` (the ``name`` it
    forwards to ``from_path_raw``): a config ``{name_prefix}{suffix}`` becomes
    ``{resolved_name}{suffix}``. So a ``config_default`` authored with the *raw* prefix won't match
    the renamed configs (and ``repo.set_config_fallback`` will reject it). Mirror that rename here —
    strip the inferred raw name_prefix and re-prefix with the resolved name — so the comparison sees
    the name that will actually exist. Returns the raw config_default unchanged when the prefix can't
    be determined (e.g. ambiguous name_prefix, or it doesn't start with the inferred prefix).
    """
    raw_default = inferred["inferred_config_default"]
    name_prefix = inferred["inferred_method"]  # for configs this is the config name_prefix
    rename_name = method.resolved_name
    if raw_default is None or not name_prefix or rename_name is None:
        return raw_default
    if not raw_default.startswith(name_prefix):
        return raw_default
    return f"{rename_name}{raw_default.removeprefix(name_prefix)}"


def _comparison_rows(method: RawMethod, inferred: dict) -> list[tuple[str, object, object, str]]:
    """Build the type-aware ``(field, inferred_value, provided_value, severity)`` comparison rows.

    Mirrors the ``MethodMetadata.config`` / ``.baseline`` / ``.portfolio`` constructors: the
    config-only fields are compared only for configs; baselines/portfolios compare ``name`` instead.
    ``config_default`` is compared against its *post-rename* value (see
    :func:`_expected_config_default`), so a ``method`` / ``config_default`` prefix mismatch is caught.
    ``severity`` controls how :func:`verify_method_metadata` treats a mismatch on the row:

      * ``"error"`` — a real raw-signal field; a mismatch fails verification.
      * ``"warn"``  — ``method`` is a naming choice (inferred from name_prefix/framework) the author
        may legitimately override, so a mismatch only warns.
      * ``"info"``  — ``model_key`` is derived (defaults to ``ag_key``) and ``name`` has no raw
        signal; shown in the comparison table for context but never gates verification.
    """
    m = method.method_metadata
    rows: list[tuple[str, object, object, str]] = [
        ("method", inferred["inferred_method"], m.method, "warn"),
        ("method_type", inferred["inferred_method_type"], m.method_type, "error"),
        ("compute", inferred["inferred_compute"], m.compute, "error"),
    ]
    if m.method_type == "config":
        ag_key = inferred["inferred_ag_key"]
        rows += [
            ("ag_key", ag_key, m.ag_key, "error"),
            ("model_key", ag_key, m.model_key, "info"),  # derived from ag_key; never gates
            ("config_default", _expected_config_default(method, inferred), m.config_default, "error"),
            ("can_hpo", inferred["inferred_can_hpo"], m.can_hpo, "error"),
            ("is_bag", inferred["is_bag_any"], m.is_bag, "error"),
        ]
    else:
        # baseline / portfolio: `name` is the type-specific display-name override; no raw signal.
        rows.append(("name", None, m.name, "info"))
    return rows


def _compare_with_provided_metadata(method: RawMethod, inferred: dict) -> None:
    """Print a side-by-side comparison of raw-inferred values against the provided
    ``MethodMetadata`` (only when one was supplied). Useful for eyeballing during ``inspect``;
    :func:`verify_method_metadata` is the function that actually enforces alignment.
    """
    if method.method_metadata is None:
        return
    rows = _comparison_rows(method, inferred)

    field_w = max(len("field"), *(len(f) for f, _, _, _ in rows))
    inf_w = max(len("inferred"), *(len(str(i)) for _, i, _, _ in rows))
    prov_w = max(len("provided"), *(len(str(p)) for _, _, p, _ in rows))

    print("[raw-info]   Comparison: inferred (raw) vs provided MethodMetadata:")
    header = f"{'field'.ljust(field_w)} | {'inferred'.ljust(inf_w)} | {'provided'.ljust(prov_w)} | match"
    print(f"[raw-info]     {header}")
    print(f"[raw-info]     {'-' * len(header)}")

    n_mismatch = 0
    for field, inferred_val, provided_val, _severity in rows:
        if inferred_val is None:
            match = "?"  # not enough info to infer; skip mismatch counting
        elif inferred_val == provided_val:
            match = "yes"
        else:
            match = "NO"
            n_mismatch += 1
        print(
            f"[raw-info]     {field.ljust(field_w)} | {str(inferred_val).ljust(inf_w)} | "
            f"{str(provided_val).ljust(prov_w)} | {match}"
        )

    if n_mismatch == 0:
        print("[raw-info]   All inferable fields match the provided MethodMetadata.")
    else:
        print(
            f"[raw-info]   WARNING: {n_mismatch} field(s) differ between inferred and provided "
            f"MethodMetadata (see 'NO' rows above). Note some differences are intentional "
            f"(e.g. a deliberately-overridden can_hpo / config_default / model_key)."
        )


def _print_method_metadata_snippet(method: RawMethod, inferred: dict) -> None:
    """Print a copy-pasteable ``MethodMetadata.<type>(...)`` snippet from raw-inferred values, plus
    instructions for committing it to a ``tabarena.models.<model>`` ``info.py`` and importing it.

    Emits the type-specific constructor (``MethodMetadata.config`` / ``.baseline`` / ``.portfolio``)
    for the inferred ``method_type``, so the snippet only carries the fields that belong to that type
    (config-only fields are dropped for baselines/portfolios, which expose ``name`` instead). Only
    raw-inferable fields are filled in; ``suite`` (required) and the other manual fields a human owns
    (display_name, reference_url, date, verified, cache_type, cache_kwargs) are placeholders.
    """
    snippet_method = method.resolved_name or inferred["inferred_method"] or "method"
    inferred_method_type = inferred["inferred_method_type"]
    inferred_ag_key = inferred["inferred_ag_key"]

    # Type-specific constructor: each exposes (and validates) only its own fields. Fall back to the
    # bare dataclass with an explicit method_type when the type can't be inferred unambiguously.
    ctor_by_type = {
        "config": "MethodMetadata.config",
        "baseline": "MethodMetadata.baseline",
        "portfolio": "MethodMetadata.portfolio",
    }
    ctor = ctor_by_type.get(inferred_method_type, "MethodMetadata")

    # Active (uncommented) kwargs as (key, value, trailing_comment). `suite` is required and manual,
    # so it goes directly after `method` with a placeholder + a REQUIRED note (it must differ from
    # `method`; see the method != suite check in verify_method_metadata).
    active_fields: list[tuple[str, object, str]] = [
        ("method", snippet_method, ""),
        (
            "suite",
            method.resolved_suite or "<set-me>",
            "  # REQUIRED, set manually: a distinct suite/run id (must differ from method)",
        ),
    ]
    if ctor == "MethodMetadata":
        # Ambiguous type (no single method_type in raw): state it so the snippet stays valid.
        active_fields.append(("method_type", inferred_method_type, ""))
    active_fields.append(("compute", inferred["inferred_compute"], ""))

    if inferred_method_type == "config":
        # Config-only fields (rejected by the baseline/portfolio constructors).
        active_fields.append(("ag_key", inferred_ag_key, ""))
        # model_key defaults to ag_key; only emit when explicitly overridden to something else.
        if method.resolved_model_key is not None and method.resolved_model_key != inferred_ag_key:
            active_fields.append(("model_key", method.resolved_model_key, ""))
        active_fields += [
            # Post-rename name (configs are renamed to the method's prefix during processing), so it
            # stays consistent with `method=` above and matches what the processed repo will contain.
            ("config_default", _expected_config_default(method, inferred), ""),
            ("can_hpo", inferred["inferred_can_hpo"], ""),
            ("is_bag", inferred["is_bag_any"], ""),
        ]

    # has_raw / has_processed / has_results: we read the raw data and will cache processed + results,
    # so all three are True. Emit one only when True differs from this constructor's default —
    # config/baseline default every tier to True (so all are omitted), while portfolio defaults
    # has_raw=False (so a portfolio with raw present surfaces has_raw=True).
    has_defaults = {
        "config": {"has_raw": True, "has_processed": True, "has_results": True},
        "baseline": {"has_raw": True, "has_processed": True, "has_results": True},
        "portfolio": {"has_raw": False, "has_processed": True, "has_results": True},
    }.get(inferred_method_type, {"has_raw": True, "has_processed": True, "has_results": True})
    intended_has = {"has_raw": True, "has_processed": True, "has_results": True}
    for has_field in ("has_raw", "has_processed", "has_results"):
        if intended_has[has_field] != has_defaults[has_field]:
            active_fields.append((has_field, intended_has[has_field], ""))

    slug = re.sub(r"[^a-z0-9_]", "", str(snippet_method).lower().replace("-", "_")) or "method"
    var_name = f"{slug}_method_metadata"
    print("[raw-info]   Suggested MethodMetadata. To use it for processing:")
    print(f"[raw-info]     1. create packages/tabarena/src/tabarena/models/<model>/info.py (e.g. <model>={slug!r}),")
    print("[raw-info]     2. paste the snippet below into it and fill in `suite` + the manual fields,")
    print("[raw-info]     3. import it in your processing script and pass it as method_metadata:")
    print(f"[raw-info]          from tabarena.models.<model>.info import {var_name}")
    print(f"[raw-info]          RawMethod(path_raw=..., method_metadata={var_name})")
    print(f"{var_name} = {ctor}(")
    for key, val, comment in active_fields:
        if val is None:
            continue
        print(f"    {key}={_format_py_value(val)},{comment}")
    # Manual fields, one per line as commented kwargs: uncomment (drop the leading "# ") and fill
    # in the value. Indentation matches the active fields above, so an uncommented line drops
    # straight into the constructor call.
    manual_fields: list[tuple[str, str, str]] = []
    if inferred_method_type in ("baseline", "portfolio"):
        # `name` is the baseline/portfolio display-name override (config uses name_suffix instead).
        manual_fields.append(("name", '"..."', "  # display-name override"))
    manual_fields += [
        ("display_name", '"..."', ""),
        ("reference_url", '"https://..."', ""),
        ("date", '"YYYY-MM-DD"', ""),
        ("verified", "False", ""),
        ("cache_type", '"r2"', '  # one of: "local", "r2", "s3"'),
        (
            "cache_kwargs",
            '{"bucket": "tabarena", "prefix": "cache"}',
            '  # only if uploading (s3 adds "upload_as_public": True)',
        ),
    ]
    print("    # --- Manual fields (not inferable from raw); uncomment and fill in: ---")
    for key, placeholder, note in manual_fields:
        print(f"    # {key}={placeholder},{note}")
    print(")")


def log_raw_data_info(method: RawMethod, *, engine: str = "ray", inferred: dict | None = None) -> None:
    """Inspect the raw artifacts at ``method.path_raw`` and print a summary of the fields that
    drive ``MethodMetadata`` construction (method_type, model_type, ag_key, name_prefix,
    num_gpus -> compute, is_bag, config_default / can_hpo) plus task/problem-type/metric coverage.

    Intended to be run before ``process_raw`` so you can author (or sanity-check) a ``MethodMetadata``
    for this artifact. Pass a precomputed ``inferred`` dict (from :func:`_infer_from_raw`) to avoid
    re-loading the raw data.
    """
    if inferred is None:
        print(f"[raw-info] Loading raw results from '{method.path_raw}' for inspection...")
        inferred = _infer_from_raw(method, engine=engine)

    label = method.resolved_name or str(method.path_raw.name)
    print(f"[raw-info] {label}: {inferred['n_results']} result files")

    print(f"[raw-info]   method_type   = {inferred['method_types']}")
    print(f"[raw-info]   model_type    = {inferred['model_types']}")
    print(f"[raw-info]   ag_key        = {inferred['ag_keys']}")
    print(f"[raw-info]   name_prefix   = {inferred['name_prefixes']} -> method={inferred['inferred_method']!r}")
    print(f"[raw-info]   problem_type  = {inferred['problem_types']}")
    print(f"[raw-info]   metric        = {inferred['metrics']}")
    print(f"[raw-info]   num_gpus      = {inferred['num_gpus_vals']} -> compute='{inferred['inferred_compute']}'")
    print(f"[raw-info]   is_bag (any)  = {inferred['is_bag_any']}")
    print(
        f"[raw-info]   n_frameworks  = {len(inferred['frameworks'])} -> can_hpo={inferred['inferred_can_hpo']}, "
        f"config_default={inferred['inferred_config_default']!r}"
    )
    print(f"[raw-info]   n_datasets    = {inferred['n_datasets']}")
    print(f"[raw-info]   n_tasks (dxf) = {inferred['n_tasks']}")
    print(f"[raw-info]   n_folds       = {inferred['n_folds']}")

    config_task_counts = inferred["config_task_counts"]
    if config_task_counts is not None:
        print(f"[raw-info]   valid configs ({len(config_task_counts)}) and their n_tasks (dataset x fold):")
        name_w = max((len(str(f)) for f in config_task_counts.index), default=0)
        for framework, n in config_task_counts.items():
            print(f"[raw-info]     {str(framework).ljust(name_w)} -> {n}")
    else:
        print(f"[raw-info]   valid configs = {inferred['frameworks']}")

    _compare_with_provided_metadata(method, inferred)
    _print_method_metadata_snippet(method, inferred)


def verify_method_metadata(
    method: RawMethod,
    *,
    engine: str = "ray",
    inferred: dict | None = None,
    check_alignment: bool = True,
    check_method_ne_suite: bool = True,
    ignore_mismatch: bool = False,
) -> None:
    """Verify ``method.method_metadata`` before processing, raising on any failed check.

    Processing requires an explicit ``MethodMetadata``: a missing one always raises (independent of
    ``ignore_mismatch``). The remaining checks are toggleable:

    Parameters
    ----------
    engine, inferred
        ``inferred`` is a precomputed :func:`_infer_from_raw` dict; when ``None`` and an alignment
        check is requested, the raw data is loaded with ``engine``.
    check_alignment
        Verify every raw-inferable ``MethodMetadata`` field matches what the raw data implies
        (method_type, compute, and — for configs — ag_key / config_default / can_hpo / is_bag) and
        fail on a mismatch. ``config_default`` is checked against its *post-rename* value (configs are
        renamed to the method's prefix during processing), so a ``method`` / ``config_default`` prefix
        mismatch is caught up front instead of failing inside ``from_raw``. The ``method`` field is a
        naming choice the author may legitimately override, so a ``method`` mismatch only warns (it
        never fails verification). Derived/signal-free fields (model_key, name) are never checked.
    check_method_ne_suite
        Verify ``method != suite``. ``MethodMetadata.__post_init__`` defaults ``suite`` to ``method``
        when it is left unset, so an equal pair almost always means ``suite`` was forgotten.
    ignore_mismatch
        When ``True``, failed checks are downgraded to a warning instead of raising (the
        always-required explicit-metadata check still raises). ``method`` mismatches warn either way.
    """
    label = method.resolved_name or str(method.path_raw.name)
    m = method.method_metadata
    if m is None:
        raise ValueError(
            f"Processing '{label}' requires an explicit `method_metadata` on the RawMethod, but none "
            f"was given. Run inspect first, author a MethodMetadata from the printed snippet (commit "
            f"it to tabarena.models.<model>/info.py), and pass it via RawMethod(method_metadata=...)."
        )

    problems: list[str] = []  # fatal (raise unless ignore_mismatch)
    warnings: list[str] = []  # non-fatal (always just warn)

    if check_alignment:
        if inferred is None:
            inferred = _infer_from_raw(method, engine=engine)
        for field, inferred_val, provided_val, severity in _comparison_rows(method, inferred):
            if severity == "info" or inferred_val is None or inferred_val == provided_val:
                continue
            msg = f"{field}: inferred={inferred_val!r} (raw) != provided={provided_val!r}"
            (warnings if severity == "warn" else problems).append(msg)

    if check_method_ne_suite and m.method == m.suite:
        problems.append(
            f"method == suite (both {m.method!r}) — set an explicit, distinct `suite` "
            f"(it defaults to `method` when left unset)"
        )

    # Non-fatal warnings (e.g. a `method` name that differs from the inferred one) are always shown.
    for warning in warnings:
        print(f"[verify] WARNING for '{label}': {warning}")

    if not problems:
        suffix = " (otherwise consistent with the raw data)" if warnings else " is consistent with the raw data"
        print(f"[verify] OK: provided MethodMetadata for '{label}'{suffix}.")
        return

    detail = "\n  - ".join(problems)
    if ignore_mismatch:
        print(f"[verify] WARNING (ignored via ignore_metadata_mismatch=True) for '{label}':\n  - {detail}")
        return
    raise ValueError(
        f"MethodMetadata verification failed for '{label}':\n  - {detail}\n\n"
        f"Fix the MethodMetadata to match the raw data, or pass ignore_metadata_mismatch=True to override."
    )


def process_raw(
    method: RawMethod,
    *,
    cache_raw: bool = False,
    cache_hpo_trajectories: bool = False,
    backend: str = "ray",
    file_paths: list[Path] | None = None,
) -> EndToEndResults:
    """Process the already-present raw data into cached method artifacts (no upload).

    Requires an explicit ``method.method_metadata`` (call :func:`verify_method_metadata` first to
    confirm it matches the raw data). Builds the processed ``EvaluationRepository`` and per-task
    results and caches them under the TabArena cache root
    (``~/.cache/tabarena/artifacts/{suite}/methods/{method}/``): ``metadata.yaml`` + ``processed/`` +
    ``results/``. Processing is per-task parallel (each worker writes its task's raw/processed
    slice directly), so memory stays flat even for large runs.

    Parameters
    ----------
    cache_raw
        Whether to also copy the raw ``results.pkl`` files into the TabArena cache. Defaults to
        ``False`` here because the raw data is already present locally at ``method.path_raw`` —
        set ``True`` only if you also want a copy under the cache layout.
    cache_hpo_trajectories
        Whether to also generate and cache HPO trajectories (only applies to ``config`` methods).
    backend
        ``"ray"`` (parallel) or ``"native"`` (sequential) for processing and simulation.
    file_paths
        Pre-discovered ``results.pkl`` paths under ``method.path_raw``; skips the directory walk
        (e.g. reuse the paths the inspect/verify scan already discovered).
    """
    if method.method_metadata is None:
        raise ValueError(
            f"process_raw requires an explicit `method_metadata` (path_raw={method.path_raw}); "
            f"author one from the inspect snippet and pass it via RawMethod(method_metadata=...)."
        )
    return EndToEnd.from_path_raw(
        path_raw=method.path_raw,
        method_metadata=method.method_metadata,
        name=method.resolved_name,
        model_key=method.resolved_model_key,
        suite=method.resolved_suite,
        file_paths=file_paths,
        cache=True,
        cache_raw=cache_raw,
        cache_processed=True,
        cache_hpo_trajectories=cache_hpo_trajectories,
        backend=backend,
    )


def process_method(
    method: RawMethod,
    *,
    inspect: bool = True,
    process: bool = False,
    ignore_metadata_mismatch: bool = False,
    cache_raw: bool = True,
    cache_hpo_trajectories: bool = True,
    backend: str = "ray",
) -> None:
    """Inspect and/or process a single ``RawMethod`` (no upload).

    This is the single-method unit of work; :func:`process_methods` is a thin loop over it.

    Parameters
    ----------
    method
        A local directory of already-present raw ``results.pkl`` files.
    inspect
        Print inferred ``MethodMetadata`` fields + a copy-paste snippet for the method.
    process
        Build + cache the processed repo and results locally (no upload). Requires an explicit,
        verified ``method_metadata`` on the ``RawMethod`` (see :func:`verify_method_metadata`).
    ignore_metadata_mismatch
        Forwarded to :func:`verify_method_metadata` — downgrade alignment/structural failures to a
        warning instead of erroring (the missing-metadata error always raises).
    cache_raw, cache_hpo_trajectories, backend
        Forwarded to :func:`process_raw` (only used when ``process=True``).
    """
    engine = "ray" if backend == "ray" else "sequential"
    label = method.resolved_name or str(method.path_raw.name)
    print(f"{label}  <- {method.path_raw}")
    if not method.path_raw.is_dir():
        raise FileNotFoundError(f"path_raw does not exist or is not a directory: {method.path_raw}")
    ts = time.time()
    # Walk the directory tree once and scan the raw data once; both are reused across
    # inspection, verification, and processing.
    file_paths = None
    inferred = None
    if inspect or process:
        file_paths = fetch_raw_result_paths(
            path_raw=method.path_raw,
            num_workers=len(os.sched_getaffinity(0)) if engine == "ray" else None,
        )
        inferred = _infer_from_raw(method, engine=engine, file_paths=file_paths)
    if inspect:
        log_raw_data_info(method, engine=engine, inferred=inferred)
    if process:
        verify_method_metadata(
            method,
            engine=engine,
            inferred=inferred,
            ignore_mismatch=ignore_metadata_mismatch,
        )
        print(f"[process] Building + caching processed repo and results for '{label}'...")
        process_raw(
            method,
            cache_raw=cache_raw,
            cache_hpo_trajectories=cache_hpo_trajectories,
            backend=backend,
            file_paths=file_paths,
        )
    te = time.time()
    print(f"Finished {label} (duration={te - ts:.1f}s)")


def process_methods(
    methods: list[RawMethod],
    *,
    inspect: bool = True,
    process: bool = False,
    ignore_metadata_mismatch: bool = False,
    cache_raw: bool = True,
    cache_hpo_trajectories: bool = True,
    backend: str = "ray",
) -> None:
    """Run :func:`process_method` over each ``RawMethod`` in ``methods``.

    All keyword options are forwarded unchanged to :func:`process_method` per method.
    """
    if len(methods) == 0:
        raise AssertionError("Populate `methods` with one or more RawMethod entries to run. Currently empty.")

    print(f"Processing {len(methods)} method(s) (inspect={inspect}, process={process})")
    for i, method in enumerate(methods):
        print(f"\n({i + 1}/{len(methods)})")
        process_method(
            method,
            inspect=inspect,
            process=process,
            ignore_metadata_mismatch=ignore_metadata_mismatch,
            cache_raw=cache_raw,
            cache_hpo_trajectories=cache_hpo_trajectories,
            backend=backend,
        )
