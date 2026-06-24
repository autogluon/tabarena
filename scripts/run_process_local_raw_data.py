"""Process *already-present* raw TabArena results into method metadata + cached artifacts.

A download/upload-free counterpart to
``examples/!old/run_download_url_and_cache_to_s3_2025_09_03.py``. That script's job was
URL-download -> local cache -> S3/R2 upload, orchestrated through ``MethodArtifactManager``.
Here we assume the raw ``results.pkl`` files are already on local disk (e.g. you unzipped a
submission yourself), and there is **no download and no upload**. The remaining two concerns are:

  1. ``inspect``  — read the raw artifacts and report the fields that drive ``MethodMetadata``
                    construction (method_type, ag_key, compute -> cpu/gpu, is_bag, can_hpo,
                    config_default, ...), plus task/problem-type/metric coverage. Prints a
                    copy-paste ``MethodMetadata(...)`` snippet built from the inferred values,
                    and — if you pass an existing ``MethodMetadata`` — an inferred-vs-provided
                    diff so you can confirm the hand-authored metadata matches the raw data.

  2. ``process``  — build the processed ``EvaluationRepository`` + per-task results and cache
                    them locally (``metadata.yaml`` + ``processed/`` + ``results/`` under the
                    TabArena cache root) via ``EndToEndSingle.from_path_raw(cache=True)``. This
                    is the same processing the old script ran inside ``MethodArtifactManager.cache``,
                    minus the raw download and the S3/R2 upload.

Two ways to specify each method (see ``RawMethod``):
  * Pass a fully-specified ``method_metadata`` (e.g. imported from a model's ``info.py``), or
  * leave it ``None`` and give ``name`` / ``artifact_name`` / ``model_key`` hints so the
    metadata is inferred from the raw data during processing.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from tabarena.nips2025_utils.end_to_end_single import EndToEndSingle
from tabarena.nips2025_utils.method_processor import get_info_from_result, load_raw

if TYPE_CHECKING:
    from tabarena.models._method_metadata import MethodMetadata


@dataclass
class RawMethod:
    """A local directory of raw ``results.pkl`` files to inspect and/or process.

    Parameters
    ----------
    path_raw
        Directory containing the method's raw ``results.pkl`` files (recursively). This is the
        already-present raw data; nothing is downloaded.
    method_metadata
        A fully-specified ``MethodMetadata`` (e.g. ``from tabarena.models.tabpfn_3.info import
        tabpfn_3_method_metadata``). If given, it is used as-is and ``name`` / ``artifact_name`` /
        ``model_key`` below are ignored (their values are taken from the metadata).
    name, artifact_name, model_key
        Used only when ``method_metadata is None``: naming hints forwarded to
        ``EndToEndSingle.from_path_raw`` so the metadata is inferred from the raw data. Leaving
        all of them ``None`` lets the method name / artifact / model_key be inferred too.
    """

    path_raw: Path
    method_metadata: MethodMetadata | None = None
    name: str | None = None
    artifact_name: str | None = None
    model_key: str | None = None

    def __post_init__(self) -> None:
        self.path_raw = Path(self.path_raw)

    # Resolve naming from the metadata when provided, else from the explicit hints — mirroring
    # what ``MethodArtifactManager.from_method_metadata`` did before forwarding to from_path_raw.
    @property
    def resolved_name(self) -> str | None:
        if self.name is not None:
            return self.name
        return self.method_metadata.method if self.method_metadata is not None else None

    @property
    def resolved_artifact_name(self) -> str | None:
        if self.artifact_name is not None:
            return self.artifact_name
        return self.method_metadata.artifact_name if self.method_metadata is not None else None

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


def _compare_with_provided_metadata(
    method: RawMethod,
    *,
    inferred_method: str | None,
    method_types: list,
    ag_keys: list,
    inferred_compute: str,
    inferred_can_hpo: bool,
    inferred_config_default: str | None,
    is_bag_any: bool,
) -> None:
    """Print a side-by-side comparison of raw-inferred values against the provided
    ``MethodMetadata`` (only when one was supplied). Useful for confirming that hand-authored
    metadata in an ``info.py`` matches what the raw data implies.
    """
    if method.method_metadata is None:
        return
    m = method.method_metadata

    inferred_method_type = method_types[0] if len(method_types) == 1 else None
    inferred_ag_key = ag_keys[0] if len(ag_keys) == 1 else None
    # model_key has no independent raw signal; from_raw defaults it to ag_key.
    inferred_model_key = inferred_ag_key

    rows: list[tuple[str, object, object]] = [
        ("method", inferred_method, m.method),
        ("method_type", inferred_method_type, m.method_type),
        ("compute", inferred_compute, m.compute),
        ("ag_key", inferred_ag_key, m.ag_key),
        ("model_key", inferred_model_key, m.model_key),
        ("config_default", inferred_config_default, m.config_default),
        ("can_hpo", inferred_can_hpo, m.can_hpo),
        ("is_bag", is_bag_any, m.is_bag),
    ]

    field_w = max(len("field"), *(len(f) for f, _, _ in rows))
    inf_w = max(len("inferred"), *(len(str(i)) for _, i, _ in rows))
    prov_w = max(len("provided"), *(len(str(p)) for _, _, p in rows))

    print("[raw-info]   Comparison: inferred (raw) vs provided MethodMetadata:")
    header = f"{'field'.ljust(field_w)} | {'inferred'.ljust(inf_w)} | {'provided'.ljust(prov_w)} | match"
    print(f"[raw-info]     {header}")
    print(f"[raw-info]     {'-' * len(header)}")

    n_mismatch = 0
    for field, inferred, provided in rows:
        if inferred is None:
            match = "?"  # not enough info to infer; skip mismatch counting
        elif inferred == provided:
            match = "yes"
        else:
            match = "NO"
            n_mismatch += 1
        print(
            f"[raw-info]     {field.ljust(field_w)} | {str(inferred).ljust(inf_w)} | {str(provided).ljust(prov_w)} | {match}"
        )

    if n_mismatch == 0:
        print("[raw-info]   All inferable fields match the provided MethodMetadata.")
    else:
        print(
            f"[raw-info]   WARNING: {n_mismatch} field(s) differ between inferred and provided "
            f"MethodMetadata (see 'NO' rows above). Note some differences are intentional "
            f"(e.g. a deliberately-overridden can_hpo / config_default / model_key)."
        )


def _print_method_metadata_snippet(
    method: RawMethod,
    *,
    inferred_method: str | None,
    method_types: list,
    ag_keys: list,
    inferred_compute: str,
    inferred_can_hpo: bool,
    inferred_config_default: str | None,
    is_bag_any: bool,
) -> None:
    """Print a copy-pasteable ``MethodMetadata(...)`` snippet from raw-inferred values.

    Only the fields that can be inferred from the raw data are emitted; the manual fields a
    human still owns (display_name, reference_url, date, verified, cache_type, and — if you
    later upload — cache_kwargs) are listed in a trailing comment.
    """
    snippet_method = method.resolved_name or inferred_method or "method"
    inferred_ag_key = ag_keys[0] if len(ag_keys) == 1 else None

    snippet_fields: list[tuple[str, object]] = [
        ("method", snippet_method),
        ("method_type", method_types[0] if len(method_types) == 1 else None),
        ("compute", inferred_compute),
        ("ag_key", inferred_ag_key),
    ]
    # Only emit model_key when it was explicitly provided and differs from ag_key (it defaults
    # to ag_key otherwise, so emitting it would be noise).
    if method.resolved_model_key is not None and method.resolved_model_key != inferred_ag_key:
        snippet_fields.append(("model_key", method.resolved_model_key))
    snippet_fields += [
        ("config_default", inferred_config_default),
        ("can_hpo", inferred_can_hpo),
        ("is_bag", is_bag_any),
        ("has_raw", True),
        ("has_processed", True),
        ("has_results", True),
    ]
    if method.resolved_artifact_name is not None:
        snippet_fields.append(("artifact_name", method.resolved_artifact_name))

    slug = re.sub(r"[^a-z0-9]", "", str(snippet_method).lower()) or "method"
    print("[raw-info]   Suggested MethodMetadata (copy-paste; fill in the manual fields):")
    print(f"{slug}_metadata = MethodMetadata(")
    for key, val in snippet_fields:
        if val is None:
            continue
        print(f"    {key}={_format_py_value(val)},")
    # Manual fields, one per line as commented kwargs: uncomment (drop the leading "# ") and fill
    # in the value. Indentation matches the inferred fields above, so an uncommented line drops
    # straight into the MethodMetadata(...) call.
    manual_fields: list[tuple[str, str, str]] = [
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


def log_raw_data_info(method: RawMethod, *, engine: str = "ray") -> None:
    """Inspect the raw artifacts at ``method.path_raw`` and print a summary of the fields that
    drive ``MethodMetadata`` construction (method_type, model_type, ag_key, name_prefix,
    num_gpus -> compute, is_bag, config_default / can_hpo) plus task/problem-type/metric coverage.

    Intended to be run before ``process_raw`` so you can sanity-check or build a ``MethodMetadata``
    for this artifact.
    """
    print(f"[raw-info] Loading raw results from '{method.path_raw}' for inspection...")
    results_lst = load_raw(path_raw=method.path_raw, engine=engine)
    info_df = pd.DataFrame([get_info_from_result(r) for r in results_lst])

    label = method.resolved_name or str(method.path_raw.name)
    print(f"[raw-info] {label}: {len(results_lst)} result files")

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
    # method is inferred from name_prefix for configs, and from framework for baselines.
    if len(method_types) == 1 and method_types[0] == "baseline":
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

    # Per-config (framework) coverage: total occurrences (dataset x fold) each config has results
    # for. These frameworks are the valid configs present in the raw data.
    config_task_counts = info_df.groupby("framework").size().sort_index() if "framework" in info_df.columns else None

    print(f"[raw-info]   method_type   = {method_types}")
    print(f"[raw-info]   model_type    = {model_types}")
    print(f"[raw-info]   ag_key        = {ag_keys}")
    print(f"[raw-info]   name_prefix   = {name_prefixes} -> method={inferred_method!r}")
    print(f"[raw-info]   problem_type  = {problem_types}")
    print(f"[raw-info]   metric        = {metrics}")
    print(f"[raw-info]   num_gpus      = {num_gpus_vals} -> compute='{inferred_compute}'")
    print(f"[raw-info]   is_bag (any)  = {is_bag_any}")
    print(
        f"[raw-info]   n_frameworks  = {len(frameworks)} -> can_hpo={inferred_can_hpo}, "
        f"config_default={inferred_config_default!r}"
    )
    print(f"[raw-info]   n_datasets    = {n_datasets}")
    print(f"[raw-info]   n_tasks (dxf) = {n_tasks}")
    print(f"[raw-info]   n_folds       = {n_folds}")

    if config_task_counts is not None:
        print(f"[raw-info]   valid configs ({len(config_task_counts)}) and their n_tasks (dataset x fold):")
        name_w = max((len(str(f)) for f in config_task_counts.index), default=0)
        for framework, n in config_task_counts.items():
            print(f"[raw-info]     {str(framework).ljust(name_w)} -> {n}")
    else:
        print(f"[raw-info]   valid configs = {frameworks}")

    _compare_with_provided_metadata(
        method,
        inferred_method=inferred_method,
        method_types=method_types,
        ag_keys=ag_keys,
        inferred_compute=inferred_compute,
        inferred_can_hpo=inferred_can_hpo,
        inferred_config_default=inferred_config_default,
        is_bag_any=is_bag_any,
    )
    _print_method_metadata_snippet(
        method,
        inferred_method=inferred_method,
        method_types=method_types,
        ag_keys=ag_keys,
        inferred_compute=inferred_compute,
        inferred_can_hpo=inferred_can_hpo,
        inferred_config_default=inferred_config_default,
        is_bag_any=is_bag_any,
    )


def process_raw(
    method: RawMethod,
    *,
    cache_raw: bool = False,
    cache_hpo_trajectories: bool = False,
    backend: str = "ray",
) -> EndToEndSingle:
    """Process the already-present raw data into cached method artifacts (no upload).

    Builds the processed ``EvaluationRepository`` and per-task results and caches them under the
    TabArena cache root (``~/.cache/tabarena/artifacts/{artifact_name}/methods/{method}/``):
    ``metadata.yaml`` + ``processed/`` + ``results/``.

    Parameters
    ----------
    cache_raw
        Whether to also copy the raw ``results.pkl`` files into the TabArena cache. Defaults to
        ``False`` here because the raw data is already present locally at ``method.path_raw`` —
        set ``True`` only if you also want a copy under the cache layout.
    cache_hpo_trajectories
        Whether to also generate and cache HPO trajectories (only applies to ``config`` methods).
    backend
        ``"ray"`` (parallel) or ``"native"`` (sequential) for the HPO/model-result simulation.
    """
    return EndToEndSingle.from_path_raw(
        path_raw=method.path_raw,
        method_metadata=method.method_metadata,
        name=method.resolved_name,
        model_key=method.resolved_model_key,
        artifact_name=method.resolved_artifact_name,
        cache=True,
        cache_raw=cache_raw,
        cache_hpo_trajectories=cache_hpo_trajectories,
        backend=backend,
    )


if __name__ == "__main__":
    # --- What to do ---------------------------------------------------------------------------
    inspect = True  # print inferred MethodMetadata fields + a copy-paste snippet for each method
    process = False  # build + cache processed repo and results locally (no upload)

    # --- Processing options (only used when process=True) -------------------------------------
    cache_raw = False  # raw is already present locally; don't duplicate it into the cache
    cache_hpo_trajectories = False  # also generate+cache HPO trajectories (config methods only)
    backend = "ray"  # "ray" (parallel) or "native" (sequential)

    # --- Methods to process -------------------------------------------------------------------
    # Each entry points at a local directory of already-extracted raw `results.pkl` files.
    #
    # Mode A — infer the metadata from the raw data (optionally give naming hints):
    #     RawMethod(
    #         path_raw=Path("local_data/leaderboard_submissions/data_SomeNewModel"),
    #         name="SomeNewModel_GPU",
    #         artifact_name="tabarena-2026-06-22",
    #         model_key="SOMENEWMODEL_GPU",
    #     )
    #
    # Mode B — reuse a fully-specified MethodMetadata from a model's info.py:
    #     from tabarena.models.tabpfn_3.info import tabpfn_3_method_metadata
    #     RawMethod(
    #         path_raw=Path("local_data/leaderboard_submissions/data_TabPFN_3_12052026"),
    #         method_metadata=tabpfn_3_method_metadata,
    #     )
    # Example (Mode B): point at the raw cache a committed MethodMetadata already resolves to
    # (`.path_raw`), but do NOT pass the metadata — let it be inferred from the raw data so
    # `inspect` exercises the inference path (the printed snippet can be eyeballed against the
    # committed metadata in tabarena/models/tabpfn_3/info.py):
    #     from tabarena.models.tabpfn_3.info import tabpfn_3_method_metadata
    #     RawMethod(path_raw=tabpfn_3_method_metadata.path_raw),
    methods: list[RawMethod] = [
        # RawMethod(path_raw=Path("local_data/.../data_SomeNewModel"), name="SomeNewModel_GPU"),
    ]

    if len(methods) == 0:
        raise AssertionError("Populate `methods` with one or more RawMethod entries to run. Currently empty.")

    engine = "ray" if backend == "ray" else "sequential"
    print(f"Processing {len(methods)} method(s) (inspect={inspect}, process={process})")
    for i, method in enumerate(methods):
        label = method.resolved_name or str(method.path_raw.name)
        print(f"\n({i + 1}/{len(methods)}) {label}  <- {method.path_raw}")
        if not method.path_raw.is_dir():
            raise FileNotFoundError(f"path_raw does not exist or is not a directory: {method.path_raw}")
        ts = time.time()
        if inspect:
            log_raw_data_info(method, engine=engine)
        if process:
            print(f"[process] Building + caching processed repo and results for '{label}'...")
            process_raw(
                method,
                cache_raw=cache_raw,
                cache_hpo_trajectories=cache_hpo_trajectories,
                backend=backend,
            )
        te = time.time()
        print(f"Finished {label} (duration={te - ts:.1f}s)")
