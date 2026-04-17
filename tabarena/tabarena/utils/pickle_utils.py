from __future__ import annotations

import fnmatch
import os
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path
from typing import Any

import tqdm


def fetch_all_pickles_new(
    dir_path: str | Path | list[str | Path],
    suffix: str | tuple[str, ...] = ".pkl",
    *,
    follow_symlinks: bool = False,
    max_workers: int = 0,  # 0 or 1 -> single-threaded; >1 -> thread pool across roots
    max_files: int | None = None,
) -> list[Path]:
    """Recursively find files ending in `suffix` under *dir_path* and return their paths.

    Notes:
    -----
    - Threading helps most when scanning multiple roots and/or network filesystems.
    - If you'll unpickle the files later, that is CPU-bound; prefer ProcessPool for that step.
    """
    # Normalize inputs once; avoid .resolve() which can be slow and change semantics
    roots: list[Path] = list(dir_path) if isinstance(dir_path, list) else [dir_path]  # type: ignore[arg-type]
    roots = [Path(p).expanduser() for p in roots]

    # Fast path: handle single-file inputs
    out: list[Path] = []
    for r in roots:
        if r.is_file():
            if isinstance(suffix, tuple):
                ok = any(str(r).endswith(s) for s in suffix)
            else:
                ok = str(r).endswith(suffix)
            if not ok:
                raise AssertionError(f"{r} is a file that does not end in `{suffix}`.")
            out.append(r)
    # Keep only directory roots for walking
    dir_roots: list[Path] = [r for r in roots if r.exists() and r.is_dir()]
    bad_roots = [
        r for r in roots if not r.exists() or (not r.is_dir() and not r.is_file())
    ]
    if bad_roots:
        raise NotADirectoryError(
            f"{', '.join(map(str, bad_roots))} not found or not directories"
        )

    def scan(root: Path) -> list[Path]:
        matches: list[Path] = []
        # os.walk + scandir is fast; filter by suffix in Python
        for dirpath, _dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
            # local bind for speed
            dp = Path(dirpath)
            if isinstance(suffix, tuple):
                matches.extend(dp / fn for fn in filenames if fn.endswith(suffix))
            else:
                sfx = suffix
                matches.extend(dp / fn for fn in filenames if fn.endswith(sfx))
            if max_files is not None and len(matches) >= max_files:
                return matches
        return matches

    if max_workers and len(dir_roots) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            lists = list(ex.map(scan, dir_roots))
    else:
        lists = [scan(r) for r in dir_roots]

    out.extend(chain.from_iterable(lists))
    if max_files is not None and len(out) >= max_files:
        return out[:max_files]
    return out


def fetch_all_pickles(
    dir_path: str | Path | list[str | Path],
    suffix: str = ".pkl",
    max_files: int | None = None,
    name_pattern: str | None = None,
    num_workers: int | None = None,
    **kwargs,
) -> list[Path]:
    """Recursively find every file ending in “.pkl” under *dir_path*
    and un‑pickle its contents.

    Parameters
    ----------
    dir_path : str | Path | list[str | Path]
        Root directory to search.
        If a list of directories, will search over all directories.
    name_pattern: str | None
        If provided, only files in subdirectories matching this pattern will be considered.
    num_workers: int | None
        If set to an int > 1, use ray to parallelize the directory walk across
        top-level and next-level subdirectories. Default (None or 1) is sequential.

    Returns:
    -------
    list[Path]
        A list of paths to .pkl files.

    Notes:
    -----
    Never un‑pickle data you do not trust.
    Malicious pickle data can execute arbitrary code.
    """
    if num_workers is not None and num_workers > 1:
        if max_files is not None:
            raise NotImplementedError(
                "Limiting max_files is not currently implemented for num_workers > 1."
            )
        return _fetch_all_pickles_parallel(
            dir_path=dir_path,
            suffix=suffix,
            name_pattern=name_pattern,
            num_workers=num_workers,
        )

    if not isinstance(dir_path, list):
        dir_path = [dir_path]

    file_paths: list[Path] = []
    for cur_dir_path in dir_path:
        root = Path(cur_dir_path).expanduser()
        if not root.is_dir():
            if root.is_file():
                assert str(root).endswith(suffix), (
                    f"{root} is a file that does not end in `{suffix}`."
                )
                file_paths.append(root)
            else:
                raise NotADirectoryError(f"{root} is not a directory")
        else:
            if name_pattern is not None:
                prefix_glob = f"{name_pattern}*"
                top_dirs = [
                    Path(e.path)
                    for e in os.scandir(root)
                    if e.is_dir(follow_symlinks=False)
                    and fnmatch.fnmatch(e.name, prefix_glob)
                ]
            else:
                top_dirs = [root]

            pbar = tqdm.tqdm(
                desc=f"Searching for pickles in {cur_dir_path} (suffix={suffix}, name_pattern={name_pattern})",
                unit=" files",
            )
            try:
                for top in top_dirs:
                    for dirpath, _dirnames, filenames in os.walk(top):
                        dp = Path(dirpath)
                        for fn in filenames:
                            if fn.endswith(suffix):
                                file_paths.append(dp / fn)
                                pbar.update(1)
                                if (
                                    max_files is not None
                                    and len(file_paths) == max_files
                                ):
                                    return file_paths
            finally:
                pbar.close()

    return file_paths


def _scan_dirs_for_pickles(dir_paths: list[Path], suffix: str) -> list[Path]:
    """Scan a batch of directories for files ending in `suffix` (ray worker)."""
    out: list[Path] = []
    for d in dir_paths:
        for dirpath, _dirnames, filenames in os.walk(d):
            dp = Path(dirpath)
            out.extend(dp / fn for fn in filenames if fn.endswith(suffix))
    return out


def _fetch_all_pickles_parallel(
    dir_path: str | Path | list[str | Path],
    *,
    suffix: str,
    name_pattern: str | None,
    num_workers: int,
) -> list[Path]:
    """Parallel variant of `fetch_all_pickles` using ray.

    Expands search roots to their immediate subdirectories (and, when
    `name_pattern` is set, filters top-level children first), batches the
    resulting work units, and maps `_scan_dirs_for_pickles` over them.
    """
    from tabarena.utils.ray_utils import ray_map_list, to_batch_list

    if not isinstance(dir_path, list):
        dir_path = [dir_path]

    file_paths: list[Path] = []
    work_units: list[Path] = []

    for cur_dir_path in dir_path:
        root = Path(cur_dir_path).expanduser()
        if not root.is_dir():
            raise NotADirectoryError(f"{root} is not a directory")

        if name_pattern is not None:
            prefix_glob = f"{name_pattern}*"
            top_dirs = [
                Path(e.path)
                for e in os.scandir(root)
                if e.is_dir(follow_symlinks=False)
                and fnmatch.fnmatch(e.name, prefix_glob)
            ]
        else:
            top_dirs = [root]

        # Expand one level deeper for better parallelism across ray workers.
        for top in top_dirs:
            subs = [
                Path(e.path)
                for e in os.scandir(top)
                if e.is_dir(follow_symlinks=False)
            ]
            if not subs:
                warnings.warn(f"{top} does not contain results!")
                continue
            work_units.extend(subs)

    if work_units:
        batch_size = max(len(work_units) // (num_workers * 4), 1)
        batch_size = min(batch_size, 500)
        batches = list(to_batch_list(work_units, batch_size))

        import ray

        if not ray.is_initialized():
            ray.init(num_cpus=num_workers)

        results = ray_map_list(
            list_to_map=batches,
            func=_scan_dirs_for_pickles,
            func_element_key_string="dir_paths",
            num_workers=num_workers,
            num_cpus_per_worker=1,
            func_kwargs={"suffix": suffix},
            track_progress=True,
            tqdm_kwargs={
                "desc": (
                    f"Searching for pickles (suffix={suffix}, "
                    f"name_pattern={name_pattern}, workers={num_workers})"
                ),
            },
            ray_remote_kwargs={"max_calls": 0},
        )
        file_paths.extend(chain.from_iterable(results))

    return file_paths


def load_all_pickles(dir_path: str | Path) -> list[Any]:
    """Recursively find every file ending in “.pkl” or “.pickle” under *dir_path*
    and un‑pickle its contents.

    Parameters
    ----------
    dir_path : str | pathlib.Path
        Root directory to search.

    Returns:
    -------
    List[Any]
        A list whose elements are the Python objects obtained from each
        successfully un‑pickled file, in depth‑first lexical order.

    Notes:
    -----
    Never un‑pickle data you do not trust.
    Malicious pickle data can execute arbitrary code.
    """
    file_paths = fetch_all_pickles(dir_path=dir_path)
    loaded_objects = []

    print("Load results...")
    for file_path in file_paths:
        with file_path.open("rb") as f:
            loaded_objects.append(pickle.load(f))
    return loaded_objects
