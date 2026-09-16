"""Standardized pre-fetching and local-first resolution of foundation-model weights.

Benchmarking foundation models requires their (often large, Hugging-Face-hosted) weights to be
present locally. Downloading lazily inside ``_fit`` is fine for a single run, but for a benchmark
launched across many parallel compute nodes, some without internet, the weights must be warmed
on the head node before the jobs are dispatched.

Each model declares how to fetch its own weights via ``ModelInfo.prefetch_weights``, a zero-arg
``prefetch_weights`` callable living in the model's ``model.py`` (or, for models supported out of the
box via an external class, its ``info.py``). :func:`prefetch_weights` resolves benchmark model names
to their ``ModelInfo``, calls each one's prefetcher and returns a :class:`PrefetchReport` with one
:class:`PrefetchResult` per name. Models that declare nothing (tree / linear baselines) are skipped.

Resolution on compute nodes
    Wrappers and shared-weights loaders resolve checkpoint files through :func:`resolve_hf_file`
    and :func:`resolve_hf_snapshot`. Both ask the Hugging Face cache first
    (``local_files_only=True``), so a prefetched node never issues the etag HEAD request that
    ``hf_hub_download`` otherwise makes (10 s timeout on firewalled nodes), and fall back to the
    network only when ``allow_download`` is True. With a branch ``revision`` (for example
    ``"main"``) the local branch resolves through the cache's ``refs/<revision>`` pointer, that is
    the snapshot the head node validated during prefetch; online revalidation of a mutable
    revision is what the prefetchers do on the head node. A wrapper that must honor
    ``ag.fetch_pretrained_weights`` decides ``allow_download`` with
    ``autogluon.common.utils.pretrained_weights.fetch_allowed`` and raises AutoGluon's
    ``PretrainedWeightsUnavailableError`` when a fetch is forbidden. ``HF_HUB_OFFLINE=1`` is
    optional hardening for jobs whose bundle prefetched every model and must not be set when any
    model in the bundle has no prefetcher or its prefetch did not complete
    (:attr:`PrefetchReport.complete`).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable


class WeightsUnavailableError(FileNotFoundError):
    """A checkpoint is not in the local Hugging Face cache and downloading it is not allowed."""


def resolve_hf_file(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
    subfolder: str | None = None,
    allow_download: bool = True,
    token: str | bool | None = None,
) -> str:
    """Local path of one Hub file: the cache first, the network only when ``allow_download``.

    Args:
        repo_id: Hugging Face repository id, for example ``"jingang/TabICL"``.
        filename: File name inside the repository (and ``subfolder`` when given).
        revision: Branch, tag or commit; ``None`` is the repository's default branch.
        subfolder: Folder inside the repository holding ``filename``.
        allow_download: Whether a cache miss may download the file.
        token: Hugging Face token for gated repositories.

    Returns:
        The absolute path of the cached blob (symlinks resolved), stable across the snapshot links
        that point at it.

    Raises:
        WeightsUnavailableError: The file is not cached and ``allow_download`` is False.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    kwargs = dict(repo_id=repo_id, filename=filename, revision=revision, subfolder=subfolder, token=token)
    try:
        path = hf_hub_download(**kwargs, local_files_only=True)
    except LocalEntryNotFoundError:
        if not allow_download:
            raise WeightsUnavailableError(
                f"{_describe(repo_id, revision, filename, subfolder)} is not in the local Hugging Face cache and "
                "downloading is not allowed. Prefetch it on a node with network access "
                "(tabarena.models.prefetch.prefetch_weights) or allow the download."
            ) from None
        path = hf_hub_download(**kwargs)
    return str(Path(path).resolve())


def resolve_hf_snapshot(
    repo_id: str,
    *,
    revision: str | None = None,
    allow_patterns: list[str] | str | None = None,
    required_files: Iterable[str] | None = None,
    allow_download: bool = True,
    token: str | bool | None = None,
) -> str:
    """Local snapshot directory of a Hub repository: the cache first, the network when allowed.

    A cached snapshot is accepted only when every path in ``required_files`` exists under it, so a
    partial snapshot (an interrupted download, or a narrower ``allow_patterns`` from an earlier
    call) is completed instead of being served incomplete.

    Args:
        repo_id: Hugging Face repository id.
        revision: Branch, tag or commit; ``None`` is the repository's default branch.
        allow_patterns: Glob patterns of the files to fetch, as ``snapshot_download`` accepts.
        required_files: Paths relative to the snapshot that must exist for it to count as complete.
        allow_download: Whether a cache miss or an incomplete snapshot may be downloaded.
        token: Hugging Face token for gated repositories.

    Returns:
        The absolute path of the snapshot directory (``.../snapshots/<commit sha>``).

    Raises:
        WeightsUnavailableError: The snapshot is missing or incomplete and ``allow_download`` is
            False, or ``required_files`` are still missing after the download.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    required = list(required_files or ())
    kwargs = dict(repo_id=repo_id, revision=revision, allow_patterns=allow_patterns, token=token)
    try:
        path = snapshot_download(**kwargs, local_files_only=True)
    except LocalEntryNotFoundError:
        path = None
    missing = required if path is None else _missing_files(path, required)
    if path is not None and not missing:
        return str(Path(path).resolve())
    if not allow_download:
        what = "is not in the local Hugging Face cache" if path is None else f"is cached without {missing}"
        raise WeightsUnavailableError(
            f"Snapshot of {_describe(repo_id, revision)} {what} and downloading is not allowed. Prefetch it on a "
            "node with network access (tabarena.models.prefetch.prefetch_weights) or allow the download."
        ) from None
    path = snapshot_download(**kwargs)
    missing = _missing_files(path, required)
    if missing:
        raise WeightsUnavailableError(
            f"Snapshot of {_describe(repo_id, revision)} at {path} still lacks {missing} after downloading; "
            "check `allow_patterns` and `required_files`."
        )
    return str(Path(path).resolve())


def _missing_files(snapshot_dir: str, required_files: list[str]) -> list[str]:
    root = Path(snapshot_dir)
    return [name for name in required_files if not (root / name).exists()]


def _describe(repo_id: str, revision: str | None, filename: str | None = None, subfolder: str | None = None) -> str:
    label = repo_id if revision is None else f"{repo_id}@{revision}"
    if filename is not None:
        label += ":" + (f"{subfolder}/{filename}" if subfolder else filename)
    return label


_SNAPSHOT_SHA_RE = re.compile(r"[\\/]snapshots[\\/]([0-9a-fA-F]{7,64})(?=[\\/]|$)")


def commit_from_snapshot_path(path: str | Path) -> str | None:
    """The commit sha in a Hugging Face cache path ``.../snapshots/<sha>/...``; ``None`` otherwise.

    Works on snapshot directories and on unresolved file paths inside one; a blob path returned by
    :func:`resolve_hf_file` no longer carries the sha. Wrappers use it to pin ``revision=<sha>`` for a
    library call that would otherwise revalidate a branch online.
    """
    match = _SNAPSHOT_SHA_RE.search(str(path))
    return None if match is None else match.group(1)


#: Outcome of one name in :func:`prefetch_weights`.
#: ``ok``: prefetcher ran and every path it returned exists (a return value that is not a path,
#: a collection or a mapping of paths is not checked). ``nothing``: the model declares no
#: prefetcher. ``partial``: the prefetcher returned no paths or a path that does not exist.
#: ``unknown``: the name resolved to no model. ``missing_dependency``: the prefetcher raised
#: ``ImportError``. ``failed``: any other exception. ``undeclared``: a method that cannot declare a
#: prefetcher (used by system prefetch).
PrefetchStatus = Literal["ok", "nothing", "partial", "unknown", "missing_dependency", "failed", "undeclared"]

_COMPLETE_STATUSES: frozenset[str] = frozenset({"ok", "nothing"})


@dataclass(frozen=True)
class PrefetchResult:
    """Outcome of prefetching one benchmark method.

    Args:
        name: The requested benchmark model name.
        method: Registry ``method`` of the resolved model, ``None`` when the name was unknown.
        status: One of :data:`PrefetchStatus`.
        detail: Human-readable explanation (error message, missing paths); empty on success.
    """

    name: str
    method: str | None
    status: PrefetchStatus
    detail: str = ""


@dataclass(frozen=True)
class PrefetchReport:
    """Outcomes of one :func:`prefetch_weights` call, one :class:`PrefetchResult` per name."""

    results: tuple[PrefetchResult, ...] = ()

    @property
    def complete(self) -> bool:
        """True when every result is ``ok`` or ``nothing``, so an offline run has all weights."""
        return all(result.status in _COMPLETE_STATUSES for result in self.results)

    def with_status(self, *statuses: PrefetchStatus) -> tuple[PrefetchResult, ...]:
        """The results whose status is one of ``statuses``, in report order."""
        return tuple(result for result in self.results if result.status in statuses)

    @staticmethod
    def merge(*reports: PrefetchReport) -> PrefetchReport:
        """One report holding the results of ``reports`` in order."""
        return PrefetchReport(tuple(result for report in reports for result in report.results))

    def summary(self) -> str:
        """One line per status with the names it applies to, plus a completeness verdict."""
        lines = []
        for status in PrefetchStatus.__args__:
            names = [result.name for result in self.results if result.status == status]
            if names:
                lines.append(f"{status}: {', '.join(names)}")
        verdict = "complete" if self.complete else "incomplete"
        lines.append(f"prefetch {verdict} ({len(self.results)} model(s))")
        return "\n".join(lines)


def prefetch_weights(model_names: Iterable[str], *, raise_on_error: bool = False) -> PrefetchReport:
    """Ensure the weights of every foundation model in ``model_names`` are present locally.

    Resolves each name to its registry ``ModelInfo`` and calls ``info.prefetch_weights()`` if the
    model declares one, de-duplicating models that share a prefetcher (e.g. TabPFN variants) so each
    runs once; the aliases record the same outcome. Models that declare no prefetcher are skipped.

    Args:
        model_names: Benchmark model names (``display_name`` or ``method``), e.g. ``["TabPFN-3"]``.
        raise_on_error: If True, re-raise the first error; otherwise log and continue (a missing
            optional dependency or a single failed download won't abort the benchmark).

    Returns:
        A :class:`PrefetchReport` with one result per requested name.
    """
    from tabarena.models.utils import get_model_info_from_name

    seen: dict[object, PrefetchResult] = {}
    results: list[PrefetchResult] = []
    for model_name in model_names:
        try:
            info = get_model_info_from_name(model_name)
        except ValueError as exc:  # unknown model name
            if raise_on_error:
                raise
            print(f"[prefetch] {model_name}: skipped (could not resolve model: {exc})")
            results.append(PrefetchResult(model_name, None, "unknown", str(exc)))
            continue

        method = info.method_metadata.method
        prefetcher = info.prefetch_weights
        if prefetcher is None:
            print(f"[prefetch] {model_name}: nothing to prefetch (not a foundation model)")
            results.append(PrefetchResult(model_name, method, "nothing"))
            continue
        if prefetcher in seen:
            print(f"[prefetch] {model_name}: weights already warmed by another variant")
            first = seen[prefetcher]
            results.append(PrefetchResult(model_name, method, first.status, first.detail))
            continue

        print(f"[prefetch] {model_name} ({method}): ensuring weights present...")
        try:
            returned = prefetcher()
        except ImportError as exc:
            print(f"[prefetch] {model_name}: skipped (optional dependency missing: {exc})")
            result = PrefetchResult(model_name, method, "missing_dependency", str(exc))
        except Exception as exc:  # one bad download shouldn't abort the whole plan
            if raise_on_error:
                raise
            print(f"[prefetch] {model_name}: FAILED ({type(exc).__name__}: {exc})")
            result = PrefetchResult(model_name, method, "failed", f"{type(exc).__name__}: {exc}")
        else:
            result = _result_from_paths(model_name, method, returned)
        seen[prefetcher] = result
        results.append(result)
    return PrefetchReport(tuple(results))


def _result_from_paths(name: str, method: str, returned: object) -> PrefetchResult:
    """``ok`` unless the prefetcher returned an empty collection or a path that does not exist.

    ``returned`` may be ``None``, one path, an iterable of paths or a mapping whose values are
    paths (for example ``{task: local_path}``). Any other value (a flag, a count) is not checked
    and never raises, so one prefetcher's return convention cannot abort the prefetch loop.
    """
    if returned is None:
        return PrefetchResult(name, method, "ok")
    if isinstance(returned, Mapping):
        paths = list(returned.values())
    elif isinstance(returned, (str, Path)):
        paths = [returned]
    else:
        try:
            paths = list(returned)  # type: ignore[call-overload]
        except TypeError:
            return PrefetchResult(name, method, "ok", "prefetcher returned an unrecognised value; not checked")
    if not paths:
        return PrefetchResult(name, method, "partial", "prefetcher returned no paths")
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        return PrefetchResult(name, method, "partial", f"returned paths do not exist: {missing}")
    return PrefetchResult(name, method, "ok")
