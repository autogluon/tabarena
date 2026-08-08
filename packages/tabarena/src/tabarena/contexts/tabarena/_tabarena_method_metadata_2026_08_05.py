from __future__ import annotations

from typing import TYPE_CHECKING

from tabarena.systems.autogluon.info import ag_160_eq_4h_metadata, ag_160_noncomm_4h_metadata

if TYPE_CHECKING:
    from tabarena.models._method_metadata import MethodMetadata

methods_2026_08_05_ag: list[MethodMetadata] = [
    ag_160_eq_4h_metadata,
    ag_160_noncomm_4h_metadata,
]
