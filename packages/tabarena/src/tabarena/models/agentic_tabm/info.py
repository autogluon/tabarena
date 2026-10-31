from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.agentic_tabm.hpo import gen_agentic_tabm
from tabarena.models.agentic_tabm.model import AgenticTabMModel

agentic_tabm_method_metadata = MethodMetadata.config(
    method="TabM-A",
    validation_protocol="8x1",
    suite="tabarena-agentic-2026-10",
    ag_key="TA-TABM-A",
    model_key="TABM-A",
    config_default="TabM-A_c1_default_BAG_L1",
    can_hpo=True,
    compute="gpu",
    is_bag=True,
    date="2026-10-14",
    date_introduced="2026-09",
    reference_url="https://arxiv.org/abs/2609.16309",
    license="MIT",
    display_name="TabM-A",
    verified=False,
)

agentic_tabm_info = ModelInfo(
    model_cls=AgenticTabMModel,
    search_space=gen_agentic_tabm,
    method_metadata=agentic_tabm_method_metadata,
    pip_extra=(
        "agentic-tabular[torch] @ git+https://github.com/yandex-research/agentic-hpset.git"
        "@3d7349b36df9bea48a5c236830a8f313ba1819c9",
        # bin/tabm/_util.py logs through loguru, which the torch extra does not install.
        "loguru>=0.7.0",
    ),
)
