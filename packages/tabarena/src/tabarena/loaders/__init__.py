from pathlib import Path
import os
from typing import Dict, List, Tuple

from ._configs import load_configs
from ._results import load_results
from ._download import download_zs_metadata


# One-element holder for the runtime TabArena cache-root override. Resolved
# lazily (see `get_tabarena_cache_root`) so it can be set *after* this module is
# imported — avoiding the need to set `$TABARENA_CACHE` before importing tabarena.
_tabarena_cache_override = [None]


def get_tabarena_cache_root() -> Path:
    """Resolve the TabArena cache root.

    Preference order, evaluated on every call (lazy): a runtime override set via
    `set_tabarena_cache_root`, then `$TABARENA_CACHE`, then `~/.cache/tabarena`.
    """
    if _tabarena_cache_override[0] is not None:
        return _tabarena_cache_override[0]
    env = os.getenv("TABARENA_CACHE")
    return Path(env) if env else Path.home() / ".cache" / "tabarena"


def set_tabarena_cache_root(path: str | Path | None) -> None:
    """Override the TabArena cache root at runtime (pass None to clear).

    Lets callers point TabArena at a cache directory programmatically instead of
    relying on `$TABARENA_CACHE` being set before import.
    """
    _tabarena_cache_override[0] = Path(path) if path is not None else None


class Paths:
    # ``packages/tabarena/`` — the package dir one level above ``src/`` — so
    # ``data_root`` resolves to ``packages/tabarena/data`` as it did before the
    # src/ layout move (``__file__`` is ``.../src/tabarena/loaders/__init__.py``).
    project_root: Path = Path(__file__).resolve().parents[3]
    data_root: Path = project_root / 'data'

    # default path is ~/.cache/tabrepo/data, can be overriden by setting the environment variable TABREPO_CACHE
    data_root_cache: Path = Path.home() / ".cache" / "tabrepo" / "data" if os.getenv("TABREPO_CACHE") is None else Path(os.getenv("TABREPO_CACHE"))
    results_root_cache: Path = data_root_cache / "results"

    #####
    # TODO: Clean these up and move it to its own class
    _tabarena_root_cache = Path.home() / ".cache" / "tabarena" if os.getenv("TABARENA_CACHE") is None else Path(os.getenv("TABARENA_CACHE"))
    _data_root_cache_tabarena = _tabarena_root_cache / "data"
    data_root_cache_raw: Path = _data_root_cache_tabarena / "raw"
    data_root_cache_processed: Path = _data_root_cache_tabarena / "processed"
    results_root_cache_tabarena: Path = _tabarena_root_cache / "results"
    artifacts_root_cache_tabarena = _tabarena_root_cache / "artifacts"
    #####

    @staticmethod
    def abs_to_rel(path: str, relative_to: Path = project_root) -> str:
        relative_to_split = str(relative_to)
        if relative_to_split[-1] != os.sep:
            relative_to_split += os.sep
        assert path.startswith(relative_to_split)
        path_relative = path.split(relative_to_split, 1)[1]
        return path_relative

    @staticmethod
    def rel_to_abs(path: str, relative_to: Path = project_root) -> str:
        return str(relative_to / Path(path))

    @staticmethod
    def s3_to_local_tuple_list_to_dict(
            s3_to_local_tuple_list: List[Tuple[str, str]],
            relative_to: Path = data_root,
    ) -> Dict[str, str]:
        return {Paths.abs_to_rel(val, relative_to=relative_to): key for key, val in s3_to_local_tuple_list}
