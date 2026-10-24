"""TabSTAR (v1), suite tabarena-2026-03-18: hosted results of a model with no wrapper in this package.

The installable TabSTAR is TabSTAR-v1.1 (`tabarena.models.tabstar_v1p1`), which holds the TabSTAR slot on the
leaderboard. These results stay reachable through the complete method collection (`methods_superseded`).
"""

from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata

__all__ = ["tabstar_metadata"]

tabstar_metadata = MethodMetadata.config(
    method="TabSTAR",
    suite="tabarena-2026-03-18",
    ag_key="TABSTAR",
    config_default="TabSTAR_c1_BAG_L1",
    compute="gpu",
    is_bag=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
    date="2026-03-02",
    date_introduced="2025-05",
    reference_url="https://arxiv.org/abs/2505.18125",
    organization="Technion",
    license="MIT (code); CC-BY-4.0 (weights)",
    display_name="TabSTAR",
)
