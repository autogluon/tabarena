from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.limix_2.hpo import gen_limix_2
from tabarena.models.limix_2.model import LimiX2Model

_limix_2_common = {
    "compute": "gpu",
    "is_bag": False,
    "reference_url": "https://arxiv.org/abs/2609.17488",
    "organization": "Stable AI & Tsinghua University",
    "commercial_use": False,
    "license": "StableAI LimiX Non-Commercial License v1.0",
}

#: The checkpoint the wrapper loads, ``stable-ai/LimiX-2-1016``: LimiX-2 retrained without the tables that
#: autogluon/tabarena#648 found in the first checkpoint's training data.
limix_2_1016_descriptor = ModelDescriptor(
    display_name="LimiX-2-1016",
    date_introduced="2026-11-11",
    **_limix_2_common,
)

#: The first checkpoint, ``stable-ai/LimiX-2`` (revision ``20c07a07``), which only the hosted LimiX-2 results
#: use. Stripped down to the ensemble members that keep the column order, it reaches a test 1-AUC of 0.0066 on
#: diabetes, where every other method is at 0.15 or worse, and reversing the column order brings it back: it saw
#: these tables with their labels during training.
limix_2_descriptor = ModelDescriptor(
    display_name="LimiX-2",
    date_introduced="2026-09-15",
    leakage_evidence="https://github.com/autogluon/tabarena/issues/648",
    leakage_datasets=("blood-transfusion-service-center", "diabetes"),
    **_limix_2_common,
)

# Run limix21016_12112026 (SkyPilot pool, 32 RTX PRO 6000, BENCHMARK_LOG.md): LimiX a917fa0 with
# stable-ai/LimiX-2-1016 at bbbf26c5, time limits of 2 h, and 8 h on the nine large or wide datasets, as for LimiX-2.
limix_2_1016_method_metadata = limix_2_1016_descriptor.method_metadata(
    method="LimiX-2-1016",
    ag_key="TA-LIMIX-2",
    config_default="LimiX-2-1016_c1_default_BAG_L1",
    can_hpo=False,
    date="2026-11-12",
    verified=True,
    cache_type="r2",
    validation_protocol="8x1",
    suite="tabarena-2026-11-12",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)

# Run limix2_17092026 (SkyPilot pool, 64 RTX PRO 6000, BENCHMARK_LOG.md): time limits of 2 h, and 8 h on the
# nine large or wide datasets, instead of the 1 h default (a recorded protocol deviation). LimiX-2-1016 above
# replaced it on the leaderboard.
limix_2_method_metadata = limix_2_descriptor.method_metadata(
    method="LimiX-2",
    ag_key="TA-LIMIX-2",
    config_default="LimiX-2_c1_default_BAG_L1",
    can_hpo=False,
    date="2026-09-17",
    verified=True,
    cache_type="r2",
    validation_protocol="8x1",
    suite="tabarena-2026-09-17",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)


limix_2_info = ModelInfo(
    model_cls=LimiX2Model,
    search_space=gen_limix_2,
    method_metadata=limix_2_1016_method_metadata,
    # The inference package is not on PyPI; pinned to the commit the wrapper was written against.
    # Keep in sync with the `limix_2` extra in pyproject.toml, which is installed with `--no-deps`
    # (the package pins torch==2.9.1); see the wrapper docstring.
    pip_extra=(
        "LimiX @ git+https://github.com/limix-ldm-ai/LimiX.git@a917fa052eb637db98318b65712def280055ec34",
        "nvtx",
    ),
    prefetch_weights=LimiX2Model.prefetch_weights,
)
