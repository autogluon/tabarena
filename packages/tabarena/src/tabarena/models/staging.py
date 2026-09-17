"""Enumerate the local weight files a benchmark run needs, for node-local staging.

The SLURM submit template copies foundation-model weights from the shared filesystem onto each
compute node before the first fit. :func:`collect_weight_paths` produces the plan it reads from the
job JSON: the Hugging Face ``models--*`` repository directories (copied whole, so ``blobs``,
``refs`` and ``snapshots`` travel together and symlinks stay valid) and the TabPFN checkpoint files.
The paths come from the models' ``prefetch_weights`` callables, which resolve local-first and return
the files they resolved, so calling them after the head-node prefetch is a cache lookup.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tabarena.models.prefetch import PrefetchReport

#: File suffixes of TabPFN checkpoints: the two spellings of the torch pickles up to TabPFN-3 and the
#: safetensors files from TabPFN-3.5 on. A safetensors file inside the Hugging Face hub cache is an HF
#: repo file (:func:`hf_repo_dir` is checked first), so only tabpfn's own cache dir reaches this set.
TABPFN_CHECKPOINT_SUFFIXES = frozenset({".ckpt", ".cpkt", ".safetensors"})


def empty_plan() -> dict:
    """A staging plan with nothing to stage."""
    return {
        "hf_hub_cache_src": None,
        "hf_home_src": None,
        "hf_repo_dirs": [],
        "tabpfn_cache_dir_src": None,
        "tabpfn_files": [],
        "other_paths": [],
        "unresolved": [],
        "complete": True,
    }


def _flatten_paths(returned: object) -> list[Path]:
    """The path-like values in a prefetcher's return value (``None``, one path, iterable or mapping)."""
    if returned is None:
        return []
    if isinstance(returned, Mapping):
        values = list(returned.values())
    elif isinstance(returned, (str, Path)):
        values = [returned]
    else:
        try:
            values = list(returned)  # type: ignore[call-overload]
        except TypeError:
            return []
    return [Path(value) for value in values if isinstance(value, (str, Path))]


def hf_repo_dir(path: Path) -> Path | None:
    """The ``models--*`` ancestor of a Hugging Face cache path (blob or snapshot file); ``None`` otherwise."""
    for candidate in (path, *path.parents):
        if candidate.name.startswith("models--"):
            return candidate
    return None


def classify_paths(paths: Iterable[Path], plan: dict) -> None:
    """Sort ``paths`` into the plan's HF repo dirs, TabPFN files and other paths, in place."""
    for path in paths:
        repo = hf_repo_dir(path)
        if repo is not None:
            _append_unique(plan["hf_repo_dirs"], str(repo))
            plan["hf_hub_cache_src"] = plan["hf_hub_cache_src"] or str(repo.parent)
            plan["hf_home_src"] = plan["hf_home_src"] or str(repo.parent.parent)
        elif path.suffix in TABPFN_CHECKPOINT_SUFFIXES:
            _append_unique(plan["tabpfn_files"], str(path))
            plan["tabpfn_cache_dir_src"] = plan["tabpfn_cache_dir_src"] or str(path.parent)
        else:
            _append_unique(plan["other_paths"], str(path))


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def collect_weight_paths(
    model_names: Iterable[str],
    prefetch_report: PrefetchReport | None = None,
) -> dict:
    """The node-staging plan for ``model_names``: HF repo dirs and TabPFN files their prefetchers resolve.

    Models without a prefetcher contribute nothing. A model whose prefetch did not end ``ok`` in
    ``prefetch_report`` (or whose prefetcher raises or returns no paths here) is listed under
    ``unresolved`` and ``complete`` is False; the template still stages the rest and overlays
    symlinks to the shared cache for everything else, so an unresolved model resolves as before.

    Returns:
        A JSON-able dict with ``hf_hub_cache_src``, ``hf_home_src``, ``hf_repo_dirs``,
        ``tabpfn_cache_dir_src``, ``tabpfn_files``, ``other_paths`` (resolved paths outside both
        caches, not staged), ``unresolved`` and ``complete``.
    """
    from tabarena.models.utils import get_model_info_from_name

    plan = empty_plan()
    status_by_name = {} if prefetch_report is None else {r.name: r.status for r in prefetch_report.results}
    paths_by_prefetcher: dict[object, list[Path]] = {}
    for name in model_names:
        try:
            info = get_model_info_from_name(name)
        except ValueError:
            continue
        prefetcher = info.prefetch_weights
        if prefetcher is None:
            continue
        if status_by_name.get(name, "ok") != "ok":
            plan["unresolved"].append(name)
            continue
        if prefetcher not in paths_by_prefetcher:
            try:
                paths_by_prefetcher[prefetcher] = _flatten_paths(prefetcher())
            except Exception as exc:
                print(f"[staging] {name}: could not enumerate weights ({type(exc).__name__}: {exc})")
                paths_by_prefetcher[prefetcher] = []
        paths = paths_by_prefetcher[prefetcher]
        if not paths:
            plan["unresolved"].append(name)
            continue
        classify_paths(paths, plan)
    plan["complete"] = not plan["unresolved"]
    return plan
