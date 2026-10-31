from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.agentic_realmlp.hpo import gen_agentic_realmlp
from tabarena.models.agentic_realmlp.model import AgenticRealMLPModel

agentic_realmlp_method_metadata = MethodMetadata.config(
    method="RealMLP-A",
    validation_protocol="8x1",
    suite="tabarena-agentic-2026-10",
    ag_key="TA-REALMLP-A",
    model_key="REALMLP-A",
    config_default="RealMLP-A_c1_default_BAG_L1",
    can_hpo=True,
    compute="gpu",
    is_bag=True,
    date="2026-10-14",
    date_introduced="2026-09",
    reference_url="https://arxiv.org/abs/2609.16309",
    license="MIT",
    display_name="RealMLP-A",
    verified=False,
)

agentic_realmlp_info = ModelInfo(
    model_cls=AgenticRealMLPModel,
    search_space=gen_agentic_realmlp,
    method_metadata=agentic_realmlp_method_metadata,
    pip_extra=(
        "agentic-tabular[realmlp] @ git+https://github.com/yandex-research/agentic-hpset.git"
        "@3d7349b36df9bea48a5c236830a8f313ba1819c9",
    ),
)
