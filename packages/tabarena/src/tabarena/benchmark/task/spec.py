"""Task vending: lightweight handles that identify a task and know how to load it.

``TaskSpec`` is the source-agnostic counterpart of :class:`~tabarena.benchmark.task.wrapper.TaskWrapper`:
where the wrapper is the *loaded* task the run engine fits on, a spec is the cheap,
picklable handle the engine passes around *before* loading — its identity
(``task_id_str``, ``cache_key``) plus its vending logic (``load``). Each task type
defines its own vending: an OpenML download
(:class:`~tabarena.benchmark.task.openml.OpenMLTaskSpec`), a local cache
(:class:`~tabarena.benchmark.task.user_task.UserTask`), or anything else a subclass
implements.

Serialized ids round-trip through :func:`task_spec_from_task_id_str`, which dispatches
on the id's ``"{Prefix}|..."`` form via a parser registry — a new task type registers
its parser with :func:`register_task_spec_parser` instead of patching the engine. Ids
with no registered prefix must be plain integer OpenML task ids.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self

    from tabarena.benchmark.task.metadata import TabArenaTaskMetadata
    from tabarena.benchmark.task.wrapper import TaskWrapper


class TaskSpec(ABC):
    """Identity and vending logic for one benchmark task.

    Specs are lightweight and picklable: loading the (heavy) task data happens only
    in :meth:`load`, so the engine can defer/skip it for fully-cached jobs.
    """

    #: The task's known metadata, when attached (see :meth:`with_task_metadata`).
    task_metadata: TabArenaTaskMetadata | None = None

    def with_task_metadata(self, task_metadata: TabArenaTaskMetadata | None) -> Self:
        """Attach the task's ``TabArenaTaskMetadata`` (returning ``self``).

        ``load`` implementations hand the attached metadata to the wrapper they
        build, making it the run-time source of truth for ``problem_type`` /
        ``label`` / ``eval_metric`` / dtype flags (see ``TaskWrapper``).
        ``ExperimentBatchRunner`` attaches each task's collection entry when
        resolving datasets, so runs cannot diverge from the collection that
        scheduled them.
        """
        self.task_metadata = task_metadata
        return self

    @property
    @abstractmethod
    def task_id_str(self) -> str:
        """Portable serialized identifier of this task.

        Either a plain integer OpenML task id, or ``"{Prefix}|{task_id}|..."`` with
        the prefix registered via :func:`register_task_spec_parser`, so the id
        reconstructs this spec through :func:`task_spec_from_task_id_str`. The second
        segment must be the integer :attr:`task_id` — consumers that only need the
        legacy ``tid`` parse it out without reconstructing the spec (see
        ``tid_from_task_id_str``).
        """

    @property
    @abstractmethod
    def task_id(self) -> int:
        """Integer identifier of the task, recorded as the legacy ``tid``.

        For OpenML tasks this is the OpenML task id; other task types derive a
        stable integer (e.g. ``UserTask`` hashes its name). Used where an integer id
        is structurally required (results ``tid`` column, legacy frames).
        """

    @property
    @abstractmethod
    def cache_key(self) -> int | str:
        """Canonical, filesystem-safe identifier keying the task's caches.

        This is the per-task component of the results cache path (see
        ``experiment_runner_api._build_cache_prefix``) and the key of the task's
        text-embedding cache, so a task's results and text caches stay consistently
        keyed off one identifier. Stability is a hard requirement: changing the value
        for an existing task silently invalidates its caches.
        """

    @abstractmethod
    def load(self) -> TaskWrapper:
        """Load (vend) the task, returning its runtime wrapper.

        Where the task comes from — an OpenML download, a local cache, in-memory
        frames — is entirely the spec's business; the run engine only ever sees the
        returned :class:`TaskWrapper`.
        """

    @abstractmethod
    def resolve_task_name(self, task: TaskWrapper) -> str:
        """Display name recorded as the results ``dataset`` key.

        Receives the loaded ``task`` (the return of :meth:`load`) for specs whose
        name lives in the loaded source; specs that know their name up front may
        ignore it.
        """


#: ``task_id_str`` prefix -> parser reconstructing the spec from the full id string.
_TASK_ID_STR_PARSERS: dict[str, Callable[[str], TaskSpec]] = {}


def register_task_spec_parser(prefix: str, parser: Callable[[str], TaskSpec]) -> None:
    """Register a parser for ``task_id_str`` values of the form ``"{prefix}|..."``.

    The prefix is the id's first ``|``-segment (e.g. ``"UserTask"``). Registering an
    already-registered prefix overwrites the previous parser.
    """
    _TASK_ID_STR_PARSERS[prefix] = parser


def task_spec_from_task_id_str(task_id_str: str | int) -> TaskSpec:
    """Reconstruct a :class:`TaskSpec` from its serialized ``task_id_str``.

    Dispatches on the id's ``"{Prefix}|..."`` form via the parser registry; an id
    with no registered prefix must be a plain integer OpenML task id and yields an
    :class:`~tabarena.benchmark.task.openml.OpenMLTaskSpec`.
    """
    s = str(task_id_str)
    parser = _TASK_ID_STR_PARSERS.get(s.split("|", 1)[0])
    if parser is not None:
        return parser(s)
    try:
        tid = int(s)
    except ValueError:
        registered = sorted(_TASK_ID_STR_PARSERS)
        raise ValueError(
            f"Unrecognized task_id_str {s!r}: not an integer OpenML task id and no parser "
            f"is registered for its prefix (registered prefixes: {registered}).",
        ) from None
    from tabarena.benchmark.task.openml.spec import OpenMLTaskSpec

    return OpenMLTaskSpec(tid)
