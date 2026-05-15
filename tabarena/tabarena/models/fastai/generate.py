"""Back-compat shim: `gen_fastai` now lives in `tabarena.models.fastai.hpo`."""

from __future__ import annotations

from tabarena.models.fastai.hpo import gen_fastai, generate_configs_fastai

__all__ = ["gen_fastai", "generate_configs_fastai"]
