"""Re-export of the model class for this folder.

The wrapper class itself currently lives at
`tabarena.benchmark.models.ag.ebm.ebm_model.ExplainableBoostingMachineModel`.
This module makes it available as `tabarena.models.ebm.model.ExplainableBoostingMachineModel`,
giving per-model folders a uniform `model.py` entry point alongside
`hpo.py` and `info.py`. A future cleanup can flip the direction —
physically relocate the wrapper here and make the legacy path the shim.
"""

from __future__ import annotations

from tabarena.benchmark.models.ag.ebm.ebm_model import ExplainableBoostingMachineModel

__all__ = ["ExplainableBoostingMachineModel"]
