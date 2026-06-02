"""Back-compat shim for the relocated MethodMetadataCollection.

`MethodMetadataCollection` now lives at
`tabarena.models._method_metadata_collection`, co-located with
`MethodMetadata`. Keeping a re-export here preserves external imports of
`tabarena.nips2025_utils.artifacts.method_metadata_collection`.
"""

from __future__ import annotations

from tabarena.models._method_metadata_collection import MethodMetadataCollection

__all__ = ["MethodMetadataCollection"]
