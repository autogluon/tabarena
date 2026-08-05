"""Who competes in a leaderboard, and why that has to be a recompute.

Every headline TabArena number is *pool-relative*. Elo is pairwise over the participants,
improvability is ``1 - best_error_in_pool / error``, and rank / harmonic rank are positions
within the field. Change who competes and everyone's numbers change, so "hide the systems"
cannot be a client-side filter over one published table: it has to be its own evaluation,
exactly like ``imputation_yes`` / ``imputation_no``.

An :class:`EntrantPool` names one such field. The website ships one artifact tree per pool
(``entrants_<key>/``), and the leaderboard's "Who's competing?" selector switches between
them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tabarena.models._method_metadata import MethodTag

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

    from tabarena.models._method_metadata import MethodMetadata


@dataclass(frozen=True)
class EntrantPool:
    """One field of competitors, evaluated together.

    Attributes:
    ----------
    key
        Folder segment for this pool's artifacts (``entrants_<key>/``) and the value the
        leaderboard's selector round-trips.
    label
        What the selector shows. The pools are cumulative, so all but the first read as
        additions to the one before.
    include_systems
        Whether systems (``method_class="system"``) compete at all. Models always do.
    allow_tags
        Which :class:`~tabarena.models._method_metadata.MethodTag` values a competing system
        may carry. A system is admitted only when *all* of its tags are in here, so an
        untagged open system is in every systems pool while a closed-API one is in only the
        pools that name that tag.
    """

    key: str
    label: str
    include_systems: bool
    allow_tags: frozenset[str]

    def admits(self, method_class: str, tags: Iterable[str]) -> bool:
        """Whether a method with this class and these tags competes in this pool."""
        if method_class != "system":
            return True
        if not self.include_systems:
            return False
        return set(tags) <= self.allow_tags

    def admits_metadata(self, method_metadata: MethodMetadata) -> bool:
        """:meth:`admits` for a :class:`MethodMetadata`."""
        return self.admits(method_metadata.method_class, method_metadata.tags)


#: The published pools, in selector order. Cumulative: each admits everything the previous one
#: does plus one more group. First entry is the default the website opens on.
#:
#: Note the shape. Two independent checkboxes ("include closed-source", "include LLMs") would
#: imply five fields, and "closed-source API without an LLM" is not among the four below, so the
#: leaderboard renders this as a single four-stop selector. Adding that fifth field later is one
#: entry here plus one more generation pass.
ENTRANT_POOLS: list[EntrantPool] = [
    EntrantPool(
        key="models",
        label="Models only",
        include_systems=False,
        allow_tags=frozenset(),
    ),
    EntrantPool(
        key="systems_open",
        label="+ Open-source systems",
        include_systems=True,
        allow_tags=frozenset(),
    ),
    EntrantPool(
        key="systems_llm",
        label="+ Systems with LLMs",
        include_systems=True,
        allow_tags=frozenset({MethodTag.WITH_LLM}),
    ),
    EntrantPool(
        key="systems_all",
        label="+ Closed-source API systems",
        include_systems=True,
        allow_tags=frozenset({MethodTag.WITH_LLM, MethodTag.CLOSED_SOURCE_API}),
    ),
]

#: Pool key -> pool, for lookups by folder segment.
ENTRANT_POOLS_BY_KEY: dict[str, EntrantPool] = {pool.key: pool for pool in ENTRANT_POOLS}

#: The pool the website opens on: individual models, with no system in the field.
DEFAULT_ENTRANT_POOL = ENTRANT_POOLS[0]


def get_entrant_pool(key: str) -> EntrantPool:
    """Look a pool up by its folder-segment key."""
    try:
        return ENTRANT_POOLS_BY_KEY[key]
    except KeyError:
        raise ValueError(f"Unknown entrant pool {key!r}. Valid keys: {list(ENTRANT_POOLS_BY_KEY)}") from None


def filter_results_to_pool(
    df_results: pd.DataFrame,
    pool: EntrantPool,
    method_metadata_info: pd.DataFrame,
) -> pd.DataFrame:
    """Drop the rows of ``df_results`` whose method does not compete in ``pool``.

    ``method_metadata_info`` is a :meth:`MethodMetadataCollection.info` frame carrying
    ``method_class`` and ``tags``, joined on ``(ta_name, ta_suite)`` the same way
    ``website_format.format_leaderboard`` joins it. A results row whose method is missing from
    the info frame is kept: an unknown method defaults to being a model, which is what every
    pre-existing result is.
    """
    if pool.include_systems and pool.allow_tags >= frozenset(MethodTag.values()):
        return df_results  # admits everything; skip the join

    info = method_metadata_info.rename(columns={"method": "ta_name", "suite": "ta_suite"})
    admitted = {
        (row.ta_name, row.ta_suite)
        for row in info.itertuples()
        if pool.admits(getattr(row, "method_class", "model"), getattr(row, "tags", ()) or ())
    }
    known = set(zip(info["ta_name"], info["ta_suite"], strict=True))
    keys = list(zip(df_results["ta_name"], df_results["ta_suite"], strict=True))
    keep = [key in admitted or key not in known for key in keys]
    return df_results.loc[keep]
