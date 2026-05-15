"""Back-compat shim: `gen_tabstar` now lives in `tabarena.models.tabstar.hpo`.

`tabarena.models.utils:name_to_import_map` references this module by path
(`tabarena.models.tabstar.generate.gen_tabstar`); keep this thin re-export
so the legacy dispatch continues to work during the per-model-folder migration.
"""

from __future__ import annotations

from tabarena.models.tabstar.hpo import gen_tabstar

__all__ = ["gen_tabstar"]
