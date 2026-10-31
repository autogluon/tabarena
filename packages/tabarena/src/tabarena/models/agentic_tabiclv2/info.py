from __future__ import annotations

from pathlib import Path

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.agentic_tabiclv2.hpo import gen_agentic_tabiclv2
from tabarena.models.agentic_tabiclv2.model import AgenticTabICLv2Model

agentic_tabiclv2_method_metadata = MethodMetadata.config(
    method="TabICLv2-A",
    validation_protocol="8x1",
    suite="tabarena-agentic-2026-10",
    ag_key="TA-TABICLV2-A",
    model_key="TABICLV2-A",
    config_default="TabICLv2-A_c1_default_BAG_L1",
    can_hpo=True,
    compute="gpu",
    is_bag=False,
    date="2026-10-16",
    date_introduced="2026-09",
    reference_url="https://arxiv.org/abs/2609.16309",
    license="MIT",
    display_name="TabICLv2-A",
    verified=False,
)


def prefetch_weights() -> list[Path]:
    """Download the classifier and regressor checkpoints and return their paths in the Hugging Face cache."""
    from bin.tabicl.modules.nanotabicl.model import (
        DEFAULT_CLASSIFIER_CHECKPOINT_VERSION,
        DEFAULT_REGRESSOR_CHECKPOINT_VERSION,
        HF_REPO_ID,
    )
    from huggingface_hub import hf_hub_download

    return [
        Path(hf_hub_download(HF_REPO_ID, name))
        for name in (DEFAULT_CLASSIFIER_CHECKPOINT_VERSION, DEFAULT_REGRESSOR_CHECKPOINT_VERSION)
    ]


agentic_tabiclv2_info = ModelInfo(
    model_cls=AgenticTabICLv2Model,
    search_space=gen_agentic_tabiclv2,
    method_metadata=agentic_tabiclv2_method_metadata,
    pip_extra=(
        "agentic-tabular[tabicl] @ git+https://github.com/yandex-research/agentic-hpset.git"
        "@3d7349b36df9bea48a5c236830a8f313ba1819c9",
    ),
    prefetch_weights=prefetch_weights,
)
