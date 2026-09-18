from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.limix_2.hpo import gen_limix_2
from tabarena.models.limix_2.model import LimiX2Model

limix_2_descriptor = ModelDescriptor(
    display_name="LimiX-2",
    compute="gpu",
    is_bag=False,
    reference_url="https://arxiv.org/abs/2609.17488",
    commercial_use=False,
    license="StableAI LimiX Non-Commercial License v1.0",
    date_introduced="2026-09-15",
)

# Run limix2_17092026 (SkyPilot pool, 64 RTX PRO 6000, BENCHMARK_LOG.md): time limits of 2 h, and 8 h on the
# nine large or wide datasets, instead of the 1 h default (a recorded protocol deviation).
limix_2_method_metadata = limix_2_descriptor.method_metadata(
    method="LimiX-2",
    ag_key="TA-LIMIX-2",
    config_default="LimiX-2_c1_default_BAG_L1",
    can_hpo=False,
    date="2026-09-17",
    verified=False,
    cache_type="r2",
    validation_protocol="8x1",
    suite="tabarena-2026-09-17",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)


limix_2_info = ModelInfo(
    model_cls=LimiX2Model,
    search_space=gen_limix_2,
    method_metadata=limix_2_method_metadata,
    # The inference package is not on PyPI; pinned to the commit the wrapper was written against.
    # Keep in sync with the `limix_2` extra in pyproject.toml, which is installed with `--no-deps`
    # (the package pins torch==2.9.1); see the wrapper docstring.
    pip_extra=(
        "LimiX @ git+https://github.com/limix-ldm-ai/LimiX.git@774aa3e1a994cbe38f33758e3d663e9951855554",
        "nvtx",
    ),
    prefetch_weights=LimiX2Model.prefetch_weights,
)
