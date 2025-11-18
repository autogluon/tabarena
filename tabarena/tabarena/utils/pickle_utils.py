from __future__ import annotations

import os
import pickle
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
    **kwargs,
) -> list[Path]:
    """Recursively find every file ending in “.pkl” under *dir_path*
    and un‑pickle its contents.

    Parameters
    ----------
    dir_path : str | Path | list[str | Path]
        Root directory to search.
        If a list of directories, will search over all directories.

    Returns:
    -------
    list[Path]
        A list of paths to .pkl files.

    Notes:
    -----
    Never un‑pickle data you do not trust.
    Malicious pickle data can execute arbitrary code.
    """
    if not isinstance(dir_path, list):
        dir_path = [dir_path]

    file_paths: list[Path] = []
    for cur_dir_path in dir_path:
        root = Path(cur_dir_path).expanduser().resolve()
        if not root.is_dir():
            if root.is_file():
                assert str(root).endswith(suffix), (
                    f"{root} is a file that does not end in `{suffix}`."
                )
                file_paths.append(root)
            else:
                raise NotADirectoryError(f"{root} is not a directory")
        else:
            # Look for *.pkl
            pattern = f"*{suffix}"
            for file_path in tqdm.tqdm(
                root.rglob(pattern), desc=f"Searching for pickles in {cur_dir_path}"
            ):
                file_paths.append(file_path)

                if (max_files is not None) and (len(file_paths) == max_files):
                    return file_paths

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
