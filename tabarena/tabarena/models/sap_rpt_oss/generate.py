"""Back-compat shim: `gen_sap_rpt_oss` now lives in `tabarena.models.sap_rpt_oss.hpo`."""

from __future__ import annotations

from tabarena.models.sap_rpt_oss.hpo import gen_sap_rpt_oss

__all__ = ["gen_sap_rpt_oss"]
