"""Re-export of the model class(es) for this folder.

The wrapper class(es) currently live at `tabarena.benchmark.models.ag.tabpfnv3.tabpfn_3_model`. This module
makes them available as `tabarena.models.tabpfn_3.model`, giving per-model
folders a uniform `model.py` entry point alongside `hpo.py` and `info.py`.
A future cleanup can flip the direction — physically relocate the wrapper
here and make the legacy path the shim.
"""

from __future__ import annotations

from tabarena.benchmark.models.ag.tabpfnv3.tabpfn_3_model import TabPFN3Model

__all__ = ["TabPFN3Model"]
