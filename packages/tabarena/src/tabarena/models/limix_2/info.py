from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.limix_2.hpo import gen_limix_2
from tabarena.models.limix_2.model import LimiX2Model

limix_2_method_metadata = MethodMetadata.config(
    method="LimiX-2",
    display_name="LimiX-2",
    compute="gpu",
    date="2026-09-16",
    date_introduced="2026-09",
    ag_key="TA-LIMIX-2",
    model_key="LIMIX-2",
    can_hpo=False,
    is_bag=False,
    suite="tabarena-2026-09-16",
    verified=False,
    reference_url="https://arxiv.org/abs/2609.17488",
)


limix_2_info = ModelInfo(
    model_cls=LimiX2Model,
    search_space=gen_limix_2,
    method_metadata=limix_2_method_metadata,
    # Official inference package is not on PyPI; pin the git commit so the
    # imported LimiXPredictor matches the checkpoint under $LIMIX_MODEL_DIR.
    pip_extra=("LimiX @ git+https://github.com/limix-ldm-ai/LimiX.git@89ee0093ac35c791974dc3e8041e4a297fa03c6a",),
    prefetch_weights=LimiX2Model.prefetch_weights,
)
