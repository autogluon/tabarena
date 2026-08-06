"""Who competes in a leaderboard, and why that has to be a recompute.

Every headline TabArena number is *pool-relative*. Elo is pairwise over the participants,
improvability is ``1 - best_error_in_pool / error``, and rank / harmonic rank are positions
within the field. Change who competes and everyone's numbers change, so "hide the systems"
cannot be a client-side filter over one published table: it has to be its own evaluation,
exactly like ``imputation_yes`` / ``imputation_no``.

Models always compete. Systems are grouped into :data:`SYSTEM_CATEGORIES`, each independently
selectable, and every combination of them is published as its own :class:`EntrantPool` under
``entrants_<key>/``. Independent rather than cumulative on purpose: "LLM-based systems but not
the plain open-source ones" is a question someone actually has, and a ladder cannot express it.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

from tabarena.models._method_metadata import MethodTag

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

    from tabarena.models._method_metadata import MethodMetadata


@dataclass(frozen=True)
class SystemCategory:
    """One independently selectable group of systems.

    Attributes:
    ----------
    key
        Short id used in the pool's folder segment and by the leaderboard's toggles.
    label
        What the toggle shows.
    tag
        The :class:`~tabarena.models._method_metadata.MethodTag` this category covers, or
        ``None`` for the untagged systems (open-source, local, no LLM). A system belongs to
        every category whose tag it carries, and to ``None``'s category when it carries none.
    note
        One line explaining what selecting it adds.
    """

    key: str
    label: str
    tag: str | None
    note: str


#: The selectable groups, in toggle order. A system with several tags belongs to several
#: categories and needs all of them selected, so TabPFN-Thinking (LLM behind a closed API)
#: appears only when both `llm` and `api` are on.
SYSTEM_CATEGORIES: list[SystemCategory] = [
    SystemCategory(
        key="open",
        label="📊 Open-source systems",
        tag=None,
        note="Systems you can inspect and run yourself, such as AutoGluon.",
    ),
    SystemCategory(
        key="llm",
        label="🤖 Systems with LLMs",
        tag=MethodTag.WITH_LLM,
        note="Systems with an LLM in the loop, including agents.",
    ),
    SystemCategory(
        key="api",
        label="🔒 Closed-source API systems",
        tag=MethodTag.CLOSED_SOURCE_API,
        note="Systems behind a remote API whose internals we cannot inspect.",
    ),
]

SYSTEM_CATEGORIES_BY_KEY: dict[str, SystemCategory] = {c.key: c for c in SYSTEM_CATEGORIES}

#: Folder segment for the pool where no system competes.
MODELS_ONLY_KEY = "models"


def pool_key(categories: Iterable[str]) -> str:
    """Folder segment for a set of selected categories, in :data:`SYSTEM_CATEGORIES` order."""
    selected = set(categories)
    ordered = [c.key for c in SYSTEM_CATEGORIES if c.key in selected]
    return "_".join(ordered) if ordered else MODELS_ONLY_KEY


@dataclass(frozen=True)
class EntrantPool:
    """One field of competitors, evaluated together.

    Attributes:
    ----------
    key
        Folder segment for this pool's artifacts (``entrants_<key>/``).
    categories
        The selected :data:`SYSTEM_CATEGORIES` keys. Empty means models only.
    """

    key: str
    categories: frozenset[str]

    @property
    def label(self) -> str:
        """What the leaderboard calls this pool."""
        if not self.categories:
            return "Models only"
        names = [c.label for c in SYSTEM_CATEGORIES if c.key in self.categories]
        return "Models + " + ", ".join(names)

    @property
    def allowed_tags(self) -> frozenset[str]:
        """Tags a competing system may carry, from the selected categories."""
        return frozenset(c.tag for c in SYSTEM_CATEGORIES if c.key in self.categories and c.tag is not None)

    @property
    def includes_untagged_systems(self) -> bool:
        """Whether the untagged (plain open-source) systems compete."""
        return any(c.key in self.categories and c.tag is None for c in SYSTEM_CATEGORIES)

    def admits(self, method_class: str, tags: Iterable[str]) -> bool:
        """Whether a method with this class and these tags competes in this pool.

        Models always compete. An untagged system needs its own category selected; a tagged
        one needs *every* tag it carries to be selected, so a method is never admitted on the
        strength of a property the reader excluded.
        """
        if method_class != "system":
            return True
        tags = set(tags)
        if not tags:
            return self.includes_untagged_systems
        return tags <= self.allowed_tags

    def admits_metadata(self, method_metadata: MethodMetadata) -> bool:
        """:meth:`admits` for a :class:`MethodMetadata`."""
        return self.admits(method_metadata.method_class, method_metadata.tags)


def _all_pools() -> list[EntrantPool]:
    """Every combination of the categories, smallest field first."""
    keys = [c.key for c in SYSTEM_CATEGORIES]
    pools = []
    for size in range(len(keys) + 1):
        for combo in combinations(keys, size):
            pools.append(EntrantPool(key=pool_key(combo), categories=frozenset(combo)))
    return pools


#: Every published pool. ``2 ** len(SYSTEM_CATEGORIES)`` of them, so adding a category doubles
#: the artifact count and the generation time; that is the price of independent toggles.
ENTRANT_POOLS: list[EntrantPool] = _all_pools()

#: Pool key -> pool, for lookups by folder segment.
ENTRANT_POOLS_BY_KEY: dict[str, EntrantPool] = {pool.key: pool for pool in ENTRANT_POOLS}

#: The pool the website opens on: individual models, with no system in the field.
DEFAULT_ENTRANT_POOL = ENTRANT_POOLS_BY_KEY[MODELS_ONLY_KEY]


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
