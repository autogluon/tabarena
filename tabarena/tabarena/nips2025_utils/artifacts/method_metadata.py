"""Back-compat shim for the relocated MethodMetadata.

`MethodMetadata` now lives at `tabarena.models._method_metadata`. Moving it
out of `tabarena.nips2025_utils.artifacts` breaks the latent circular
import where per-model `info.py` files needed `MethodMetadata` but the
artifacts package's `__init__.py` eagerly loads `_tabarena_method_metadata`
— which back-imports every per-model `info.py`.

External imports of `tabarena.nips2025_utils.artifacts.method_metadata`
keep working via this re-export.
"""

from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata

__all__ = ["MethodMetadata"]
