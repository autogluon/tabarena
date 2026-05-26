"""Shim for the relocated SAP-RPT-OSS model wrapper.

The implementation now lives at `tabarena.models.sap_rpt_oss.model`. This
module re-exports it so existing imports of
`tabarena.benchmark.models.ag.sap_rpt_oss.sap_rpt_oss_model` continue to work.
"""

from __future__ import annotations

from tabarena.models.sap_rpt_oss.model import SAPRPTOSSModel, pre_download_model

__all__ = ["SAPRPTOSSModel", "pre_download_model"]
