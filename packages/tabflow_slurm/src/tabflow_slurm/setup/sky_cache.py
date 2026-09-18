"""Seed a bucket with the datasets a run needs, so SkyPilot workers never fetch from OpenML or the Hub.

The head node materializes every task of a run into its own caches during ``setup`` (OpenML tasks
into the OpenML cache, data-foundry tasks into ``tabarena_tasks/`` plus their text-embedding
caches). This module copies exactly those files into a static, shared prefix in the run's bucket,
laid out like ``CacheConfig.from_root``::

    <dataset_cache_uri>/openml/org/openml/www/tasks/<task id>/        task.xml, datasplits.*
    <dataset_cache_uri>/openml/org/openml/www/datasets/<dataset id>/  description.xml, features.xml, dataset_<id>.pq, ...
    <dataset_cache_uri>/openml/tabarena_tasks/<slug>.pkl              a portable data-foundry task
    <dataset_cache_uri>/tabarena/text_cache/<embedding>/<slug>_cache.parquet

A ``cache_manifest.json`` in the launch's queue maps each dataset to its entries. The worker pulls
them into its ``CACHE_ROOT`` (the same layout) before an item runs, so the runner's
``--materialize_tasks`` finds everything cached and downloads nothing. Datasets are immutable, so
the prefix is shared across users and runs: seeding uploads only what is missing and then verifies
that every local file is present remotely. Workers only read the prefix. A curated bucket that this
account cannot write to is used the same way with ``upload=False``: the setup then only verifies.

A data-foundry task the head node still holds as a legacy pickle (one that points at this machine's
``local/datasets/``) is upgraded in place: the dataset is re-materialized from its container into a
scratch OpenML root and the portable pickle replaces the legacy one atomically, so a SLURM job reading
the shared cache meanwhile sees one complete file or the other.

Model weights go the same way (:func:`seed_model_weights`): the run's models are prefetched on the
head node into a scratch cache root with ``HF_HOME``, ``XDG_CACHE_HOME`` (tabpfn's cache) and
``TORCH_HOME`` redirected. The files the model's prefetcher resolves in this node's own caches are
mirrored into that root first (hardlinks where possible), so the prefetch downloads only what the
node does not have, and a checkpoint that needs a license token the node no longer has (tabpfn's
gated versions) is still seeded from the warm cache. Whatever lands in the root is uploaded under
``huggingface/``, ``xdg/`` and ``torch/`` (Hugging Face repos as their ``snapshots``, ``refs`` and ``trees``,
so no blob is stored twice), and a per-model manifest ``weights/<model>.json`` makes the next setup skip
the prefetch. Workers pull the entries at start and, when every model's weights are present, load
them with ``HF_HUB_OFFLINE=1``, so no token is needed on the VMs.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from tabarena.benchmark.task.metadata.schema import tid_from_task_id_str

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from tabarena.benchmark.task.metadata import TaskMetadataCollection
    from tabflow_slurm.setup.sky_storage import Storage

_USER_TASK_PREFIX = "UserTask|"
_PORTABLE_USER_TASK_MARKER = b"tabarena-user-task-v1"
_OPENML_WWW = Path("org/openml/www")


@dataclass(frozen=True)
class CacheEntry:
    """One file or directory of a task, as it lives locally and where it goes in the cache prefix."""

    rel: str
    """Path relative to the ``CacheConfig.from_root`` layout (``openml/...`` or ``tabarena/...``)."""
    local: Path
    is_dir: bool


def local_cache_roots() -> tuple[Path, Path]:
    """The head node's ``(openml root, tabarena root)`` as configured in this process."""
    import openml

    from tabarena.loaders import get_tabarena_cache_root

    return Path(openml.config._root_cache_directory).expanduser(), Path(get_tabarena_cache_root()).expanduser()


def dataset_id_from_task_xml(path: Path) -> int:
    """The ``data_set_id`` an OpenML ``task.xml`` names."""
    match = re.search(r"<oml:data_set_id>\s*(\d+)\s*</oml:data_set_id>", path.read_text())
    if match is None:
        raise ValueError(f"{path} names no data_set_id.")
    return int(match.group(1))


def openml_task_entries(tid: int, *, openml_root: Path) -> list[CacheEntry]:
    """The task and dataset directories of OpenML task ``tid`` in the local cache (empty when not cached)."""
    task_dir = openml_root / _OPENML_WWW / "tasks" / str(tid)
    if not (task_dir / "task.xml").exists():
        return []
    entries = [CacheEntry(rel=f"openml/{_OPENML_WWW}/tasks/{tid}", local=task_dir, is_dir=True)]
    dataset_dir = openml_root / _OPENML_WWW / "datasets" / str(dataset_id_from_task_xml(task_dir / "task.xml"))
    if dataset_dir.is_dir():
        entries.append(
            CacheEntry(rel=f"openml/{_OPENML_WWW}/datasets/{dataset_dir.name}", local=dataset_dir, is_dir=True)
        )
    return entries


def is_portable_user_task(path: Path) -> bool:
    """Whether a ``tabarena_tasks/<slug>.pkl`` is the self-contained format (a legacy pickle names head-node paths)."""
    with path.open("rb") as fh:
        return _PORTABLE_USER_TASK_MARKER in fh.read(4096)


def user_task_entries(task_id_str: str, *, openml_root: Path, tabarena_root: Path) -> list[CacheEntry]:
    """The task pickle and text cache of a data-foundry task in the local caches (empty when not cached).

    A legacy pickle (one that points at ``local/datasets/`` of this machine) is skipped with a
    warning; the worker then materializes that dataset from its source as it would without a cache.
    """
    from tabarena.benchmark.preprocessing.text_cache import embedding_id
    from tabarena.benchmark.task.user_task import UserTask

    slug = UserTask.from_task_id_str(task_id_str).slug
    task_path = openml_root / "tabarena_tasks" / f"{slug}.pkl"
    if not task_path.exists():
        return []
    if not is_portable_user_task(task_path):
        warnings.warn(
            f"{task_path} is a legacy task pickle bound to this machine's cache; not seeded. Re-materialize it to "
            "get the portable format, or let the workers download the dataset.",
            stacklevel=2,
        )
        return []
    entries = [CacheEntry(rel=f"openml/tabarena_tasks/{slug}.pkl", local=task_path, is_dir=False)]
    text_cache = tabarena_root / "text_cache" / embedding_id() / f"{slug}_cache.parquet"
    if text_cache.exists():
        entries.append(
            CacheEntry(rel=f"tabarena/text_cache/{embedding_id()}/{text_cache.name}", local=text_cache, is_dir=False)
        )
    return entries


def upgrade_legacy_user_task(collection: TaskMetadataCollection, dataset: str, *, openml_root: Path) -> bool:
    """Replace ``dataset``'s legacy task pickle by the portable format; ``True`` when it was upgraded.

    Re-materializes the dataset through the collection's source (its recorded suite) into a scratch
    OpenML root next to the real one, then moves the new ``tabarena_tasks/<slug>.pkl`` over the legacy
    file with ``os.replace`` (atomic on one filesystem). The text cache lands in the TabArena cache as
    usual. Nothing changes when the collection has no materializing source or the conversion fails.
    """
    import openml

    scoped = collection.subset_tasks(dataset_names=[dataset])
    if scoped.preset is None or len(scoped) == 0:
        return False
    slug = dataset  # a data-foundry task's tabarena_task_name is its slug
    target = openml_root / "tabarena_tasks" / f"{slug}.pkl"
    scratch_root = openml_root / f".upgrade_{os.getpid()}"
    scratch_root.mkdir(parents=True, exist_ok=True)
    saved_root = openml.config._root_cache_directory
    try:
        openml.config.set_root_cache_directory(str(scratch_root))
        scoped.materialize()
    except Exception as exc:
        warnings.warn(f"Could not re-materialize {dataset!r} to upgrade its legacy task pickle: {exc!r}", stacklevel=2)
        shutil.rmtree(scratch_root, ignore_errors=True)
        return False
    finally:
        openml.config.set_root_cache_directory(str(saved_root))
    fresh = scratch_root / "tabarena_tasks" / f"{slug}.pkl"
    ok = fresh.exists() and is_portable_user_task(fresh)
    if ok:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(fresh, target)
        print(f"upgraded the legacy task pickle of {dataset!r} to the portable format")
    else:
        warnings.warn(
            f"Re-materializing {dataset!r} produced no portable task pickle; keeping the legacy one.", stacklevel=2
        )
    shutil.rmtree(scratch_root, ignore_errors=True)
    return ok


def collect_cache_entries(
    collection: TaskMetadataCollection,
    *,
    openml_root: Path,
    tabarena_root: Path,
    upgrade_legacy: bool = True,
) -> dict[str, list[CacheEntry]]:
    """Per dataset (``tabarena_task_name``), the cache entries the head node holds for it.

    With ``upgrade_legacy`` a data-foundry task held as a legacy pickle is converted first (see
    :func:`upgrade_legacy_user_task`) so it can be seeded; without it, or when the conversion is not
    possible, that dataset has no entries and the workers download it.
    """
    by_dataset: dict[str, list[CacheEntry]] = {}
    for ttm in collection:
        dataset = ttm.tabarena_task_name
        if dataset in by_dataset or ttm.task_id_str is None:
            continue
        task_id_str = str(ttm.task_id_str)
        if task_id_str.startswith(_USER_TASK_PREFIX):
            legacy = openml_root / "tabarena_tasks" / f"{dataset}.pkl"
            if upgrade_legacy and legacy.exists() and not is_portable_user_task(legacy):
                upgrade_legacy_user_task(collection, dataset, openml_root=openml_root)
            entries = user_task_entries(task_id_str, openml_root=openml_root, tabarena_root=tabarena_root)
        else:
            entries = openml_task_entries(tid_from_task_id_str(task_id_str), openml_root=openml_root)
        by_dataset[dataset] = entries
    return by_dataset


@dataclass
class CacheSeedReport:
    """What seeding did and found."""

    cache_uri: str
    datasets: int = 0
    files: int = 0
    total_bytes: int = 0
    uploaded_files: int = 0
    not_cached_locally: list[str] = field(default_factory=list)
    """Datasets with nothing to seed (not materialized here, or a legacy pickle); their workers download."""
    unverified: list[str] = field(default_factory=list)
    """Entries still missing remotely after seeding (read-only prefix, or a failed upload)."""

    def summary(self) -> str:
        parts = [
            f"dataset cache {self.cache_uri}: {self.datasets} dataset(s), {self.files} file(s), "
            f"{self.total_bytes / 1e6:.0f} MB, {self.uploaded_files} file(s) uploaded now"
        ]
        if self.not_cached_locally:
            parts.append(f"not cached on this node (workers download them): {self.not_cached_locally}")
        if self.unverified:
            parts.append(f"MISSING remotely (workers download them): {self.unverified}")
        return "\n".join(parts)


def seed_dataset_cache(
    entries_by_dataset: dict[str, list[CacheEntry]],
    *,
    storage: Storage,
    cache_uri: str,
    upload: bool = True,
) -> tuple[dict, CacheSeedReport]:
    """Upload what is missing under ``cache_uri`` (when ``upload``), verify, and build the worker manifest.

    Returns ``(manifest, report)``. The manifest lists, per dataset, the entries that are verified
    present remotely: ``{"cache_uri": ..., "datasets": {"<dataset>": ["openml/...", ...]}}``. An entry
    that could not be verified is left out, so the worker falls back to downloading that dataset from
    its source instead of failing on a half-seeded cache.
    """
    cache_uri = cache_uri.rstrip("/")
    report = CacheSeedReport(cache_uri=cache_uri)
    manifest: dict = {"cache_uri": cache_uri, "datasets": {}}
    for dataset, entries in entries_by_dataset.items():
        if not entries:
            report.not_cached_locally.append(dataset)
            continue
        report.datasets += 1
        verified: list[str] = []
        for entry in entries:
            remote = f"{cache_uri}/{entry.rel}"
            if entry.is_dir:
                files = sorted(p for p in entry.local.rglob("*") if p.is_file())
                names = {p.relative_to(entry.local).as_posix() for p in files}
                present = set(_remote_files(storage, remote))
                missing = names - present
                if missing and upload:
                    storage.upload_dir(entry.local, remote)
                    report.uploaded_files += len(missing)
                    present = set(_remote_files(storage, remote))
                report.files += len(files)
                report.total_bytes += sum(p.stat().st_size for p in files)
                ok = names <= present
            else:
                exists = storage.exists(remote)
                if not exists and upload:
                    storage.upload_file(entry.local, remote)
                    report.uploaded_files += 1
                    exists = storage.exists(remote)
                report.files += 1
                report.total_bytes += entry.local.stat().st_size
                ok = exists
            if ok:
                verified.append(entry.rel)
            else:
                report.unverified.append(entry.rel)
        if verified:
            manifest["datasets"][dataset] = verified
    return manifest, report


def _remote_files(storage: Storage, uri: str) -> list[str]:
    """Relative paths of the objects under ``uri`` at any depth (Hugging Face snapshots nest one level)."""
    return storage.list_files(uri)


# ---------------------------------------------------------------------------- model weights

#: Environment variable -> sub-directory of a cache root; the libraries that read them: huggingface_hub
#: (``HF_HOME``), tabpfn and other platformdirs users (``XDG_CACHE_HOME``), torch hub (``TORCH_HOME``).
WEIGHT_CACHE_ENV: dict[str, str] = {"HF_HOME": "huggingface", "XDG_CACHE_HOME": "xdg", "TORCH_HOME": "torch"}


def weight_cache_env(cache_root: Path) -> dict[str, str]:
    """The environment that points every weight cache under ``cache_root``."""
    return {var: str(Path(cache_root) / sub) for var, sub in WEIGHT_CACHE_ENV.items()}


def head_hf_token() -> str | None:
    """The Hugging Face token of this node: ``HF_TOKEN``, else the token file of the ambient ``HF_HOME``."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    try:
        return (home / "token").read_text().strip() or None
    except OSError:
        return None


_WEIGHT_PLAN_MARKER = "TABARENA_WEIGHT_PLAN="


def resolve_local_weights(model_names: Iterable[str], *, python: str) -> dict:
    """The weight files of ``model_names`` as this node's caches hold them, via a subprocess.

    Runs ``tabarena.models.staging.collect_weight_paths`` in ``python`` with the ambient caches (the
    prefetchers resolve local-first, so on a node that ran these models before this is a cache
    lookup) and returns its plan: ``hf_repo_dirs`` and ``tabpfn_files`` are what
    :func:`mirror_local_weights` copies. A failure returns an empty plan; the scratch prefetch then
    downloads everything as before.
    """
    from tabarena.models.staging import empty_plan

    names = list(model_names)
    code = (
        "import json; from tabarena.models.staging import collect_weight_paths; "
        f"print({_WEIGHT_PLAN_MARKER!r} + json.dumps(collect_weight_paths({names!r})))"
    )
    result = subprocess.run([python, "-P", "-c", code], check=False, capture_output=True, text=True)  # noqa: S603
    lines = [line for line in result.stdout.splitlines() if line.startswith(_WEIGHT_PLAN_MARKER)]
    if result.returncode != 0 or not lines:
        print(f"[seed] {', '.join(names)}: could not resolve this node's cached weights; the prefetch downloads them")
        return empty_plan()
    return json.loads(lines[-1][len(_WEIGHT_PLAN_MARKER) :])


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink ``src`` at ``dst`` (same filesystem), else copy it; an existing ``dst`` is kept."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _mirror_tree(src: Path, dst: Path) -> None:
    """Recreate ``src`` under ``dst``: symlinks with the same target, files hardlinked or copied; existing entries are kept."""
    for path in sorted(src.rglob("*")):
        target = dst / path.relative_to(src)
        if path.is_symlink():
            if not os.path.lexists(target):
                target.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(os.readlink(path), target)
        elif path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            _link_or_copy(path, target)


def mirror_local_weights(plan: dict, cache_root: Path) -> int:
    """Copy the weights of ``plan`` (see :func:`resolve_local_weights`) into ``cache_root``; return the file count.

    A Hugging Face repository directory contributes its ``snapshots``, ``refs`` and ``trees`` directories
    with their symlinks intact and its ``blobs`` as hardlinks, so the snapshot links resolve inside the root and
    :func:`collect_weight_entries` sees the usual layout. TabPFN checkpoints go to ``xdg/tabpfn/``,
    where the redirected ``XDG_CACHE_HOME`` makes the tabpfn loader look for them.
    """
    cache_root = Path(cache_root)
    count = 0
    for repo in plan.get("hf_repo_dirs", []):
        src = Path(repo)
        dst = cache_root / "huggingface" / "hub" / src.name
        for sub in ("snapshots", "refs", "trees"):
            if (src / sub).is_dir():
                _mirror_tree(src / sub, dst / sub)
        blobs = src / "blobs"
        if blobs.is_dir():
            for blob in sorted(blobs.iterdir()):
                if blob.is_file():
                    _link_or_copy(blob, dst / "blobs" / blob.name)
                    count += 1
    for file in plan.get("tabpfn_files", []):
        src = Path(file)
        if src.is_file():
            _link_or_copy(src, cache_root / "xdg" / "tabpfn" / src.name)
            count += 1
    return count


def prefetch_weights_into(
    model_names: Iterable[str],
    *,
    python: str,
    cache_root: Path,
    resolve: Callable[..., dict] = resolve_local_weights,
) -> None:
    """Run tabarena's weight prefetch for ``model_names`` in a subprocess whose caches live under ``cache_root``.

    The files the prefetchers resolve in this node's own caches are mirrored into ``cache_root`` first
    (``resolve`` + :func:`mirror_local_weights`), so the prefetch downloads only what the node lacks;
    the head node's own token is passed as ``HF_TOKEN`` so gated repositories download. The redirected
    caches never see the node's other weights, so what lands under ``cache_root`` is exactly this run's.
    """
    names = list(model_names)
    for path in weight_cache_env(cache_root).values():
        Path(path).mkdir(parents=True, exist_ok=True)
    mirrored = mirror_local_weights(resolve(names, python=python), cache_root)
    if mirrored:
        print(f"[seed] {', '.join(names)}: {mirrored} file(s) mirrored from this node's caches")
    env = {k: v for k, v in os.environ.items() if k not in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HUB_OFFLINE")}
    env.update(weight_cache_env(cache_root))
    token = head_hf_token()
    if token:
        env["HF_TOKEN"] = token
    code = f"from tabarena.models.prefetch import prefetch_weights; prefetch_weights({names!r}, raise_on_error=True)"
    subprocess.run([python, "-P", "-c", code], env=env, check=True)  # noqa: S603


def _is_hidden(path: Path, root: Path) -> bool:
    return any(part.startswith(".") for part in path.relative_to(root).parts)


def collect_weight_entries(cache_root: Path) -> list[CacheEntry]:
    """The seedable files a prefetch left under ``cache_root``.

    Hugging Face repositories contribute their ``snapshots``, ``refs`` and ``trees`` directories (the
    snapshot files are symlinks into ``blobs``; the upload follows them, so the blobs are not stored twice;
    ``trees`` holds the hub's cached tree listing per commit, which ``snapshot_download`` of a pinned commit
    needs offline, or it asks the Hub for the listing). Everything
    else (``xdg/tabpfn/*.ckpt``, torch hub checkpoints) is taken file by file; lock files and hidden
    directories are skipped.
    """
    cache_root = Path(cache_root)
    entries: list[CacheEntry] = []
    hub = cache_root / "huggingface" / "hub"
    if hub.is_dir():
        for repo in sorted(hub.iterdir()):
            if not repo.is_dir() or not repo.name.startswith("models--"):
                continue
            for sub in ("snapshots", "refs", "trees"):
                if (repo / sub).is_dir():
                    entries.append(CacheEntry(rel=f"huggingface/hub/{repo.name}/{sub}", local=repo / sub, is_dir=True))
    for top in ("xdg", "torch"):
        root = cache_root / top
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and not _is_hidden(path, cache_root) and not path.name.endswith(".lock"):
                entries.append(CacheEntry(rel=path.relative_to(cache_root).as_posix(), local=path, is_dir=False))
    return entries


def model_has_prefetcher(model_name: str) -> bool:
    """Whether the registry model declares weights to prefetch (foundation models do, trees do not)."""
    from tabarena.models.utils import get_model_info_from_name

    try:
        return get_model_info_from_name(model_name).prefetch_weights is not None
    except ValueError:
        return False


def seed_model_weights(
    model_names: Iterable[str],
    *,
    python: str,
    storage: Storage,
    cache_uri: str,
    upload: bool = True,
    scratch_dir: Path | None = None,
    prefetch: Callable[..., None] = prefetch_weights_into,
    has_prefetcher: Callable[[str], bool] = model_has_prefetcher,
) -> tuple[dict, CacheSeedReport]:
    """Make the weights of ``model_names`` available under ``cache_uri``; return ``(manifest, report)``.

    A model whose ``weights/<model>.json`` exists remotely with entries is taken from there. Otherwise
    (and only when ``upload``) it is prefetched into a fresh scratch root, its entries are seeded and
    verified, and the remote manifest is written. The returned manifest is
    ``{"weights": {"<model>": [rels]}, "offline_weights": bool}``; ``offline_weights`` is True when every
    model that declares weights has verified entries, which lets the workers load with ``HF_HUB_OFFLINE=1``.
    """
    cache_uri = cache_uri.rstrip("/")
    report = CacheSeedReport(cache_uri=cache_uri)
    weights: dict[str, list[str]] = {}
    complete = True
    scratch = Path(scratch_dir) if scratch_dir is not None else Path.home() / ".cache" / "tabarena_sky_seed"
    for model in dict.fromkeys(model_names):
        remote_manifest = f"{cache_uri}/weights/{model}.json"
        if storage.exists(remote_manifest):
            rels = list(json.loads(storage.read_text(remote_manifest))["entries"])
            if rels:
                weights[model] = rels
                continue
            # An empty manifest records a seeding whose prefetch produced nothing; seed again.
        if not has_prefetcher(model):
            weights[model] = []
            continue
        if not upload:
            report.unverified.append(f"weights of {model}")
            complete = False
            continue
        root = scratch / model
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        try:
            prefetch([model], python=python, cache_root=root)
            entries = collect_weight_entries(root)
            seeded, part = seed_dataset_cache({model: entries}, storage=storage, cache_uri=cache_uri, upload=True)
            report.files += part.files
            report.total_bytes += part.total_bytes
            report.uploaded_files += part.uploaded_files
            report.unverified.extend(part.unverified)
            rels = seeded["datasets"].get(model, [])
            if not entries or part.unverified:
                complete = False
            storage.write_text(remote_manifest, json.dumps({"model": model, "entries": rels}))
            weights[model] = rels
        finally:
            shutil.rmtree(root, ignore_errors=True)
    report.datasets = len(weights)
    return {"weights": weights, "offline_weights": complete}, report
