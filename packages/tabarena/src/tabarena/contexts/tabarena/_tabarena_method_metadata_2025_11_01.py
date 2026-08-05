from __future__ import annotations

from typing import TYPE_CHECKING

from tabarena.systems.autogluon.info import (
    ag_140_bq_1h8c_metadata,
    ag_140_bq_4h8c_metadata,
    ag_140_bq_5m8c_metadata,
    ag_140_eq_1h8c_metadata,
    ag_140_eq_4h8c_metadata,
    ag_140_eq_5m8c_metadata,
    ag_140_hq_4h8c_metadata,
    ag_140_hq_5m8c_metadata,
    ag_140_hqil_4h8c_metadata,
    ag_140_hqil_5m8c_metadata,
)

if TYPE_CHECKING:
    from tabarena.models._method_metadata import MethodMetadata

methods_2025_11_01_ag: list[MethodMetadata] = [
    ag_140_eq_4h8c_metadata,
    ag_140_eq_1h8c_metadata,
    ag_140_eq_5m8c_metadata,
    ag_140_bq_4h8c_metadata,
    ag_140_bq_1h8c_metadata,
    ag_140_bq_5m8c_metadata,
    ag_140_hq_4h8c_metadata,
    ag_140_hq_5m8c_metadata,
    ag_140_hqil_4h8c_metadata,
    ag_140_hqil_5m8c_metadata,
]
