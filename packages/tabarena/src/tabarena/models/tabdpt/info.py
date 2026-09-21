from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabdpt.hpo import gen_tabdpt, gen_tabdpt_turbo, gen_tabdpt_v13
from tabarena.models.tabdpt.model import TabDPTModel, TabDPTTurboModel, TabDPTv13Model

tabdpt_descriptor = ModelDescriptor(
    display_name="TabDPT",
    compute="gpu",
    is_bag=False,
    reference_url="https://arxiv.org/abs/2410.18164",
    license="Apache-2.0",
    date_introduced="2024-10",
)

tabdpt_turbo_descriptor = ModelDescriptor(
    display_name="TabDPT-Turbo",
    compute="gpu",
    is_bag=False,
    reference_url="https://openreview.net/pdf?id=Y00pwFyrHR",
    license="Apache-2.0",
    date_introduced="2026-06-05",  # tabdpt1_2.safetensors upload to Layer6/TabDPT
)

tabdpt_v13_descriptor = ModelDescriptor(
    display_name="TabDPT-1.3",
    compute="gpu",
    is_bag=False,
    reference_url="https://github.com/layer6ai-labs/TabDPT-inference/releases/tag/v1.3.0",
    license="Apache-2.0",
    date_introduced="2026-09-08",  # tabdpt 1.3.0 on PyPI and tabdpt1_3.safetensors upload to Layer6/TabDPT
)

tabdpt_method_metadata = tabdpt_descriptor.method_metadata(
    method="TabDPT_GPU",
    suite="tabarena-2025-10-20",
    ag_key="TABDPT",
    model_key="TABDPT_GPU",
    config_default="TabDPT_GPU_c1_BAG_L1",
    cache_type="s3",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache", "upload_as_public": True},
    date="2025-10-20",
)

# A distinct `model_key`/`ag_key` keeps TabDPT-Turbo (v1.2) a separate leaderboard method from TabDPT above.
tabdpt_turbo_method_metadata = tabdpt_turbo_descriptor.method_metadata(
    method="TabDPT-Turbo",
    suite="tabarena-2026-07-13",
    ag_key="TA-TABDPT-TURBO",
    model_key="TABDPT_TURBO",
    config_default="TabDPT-Turbo_c1_default_BAG_L1",
    can_hpo=False,
    date="2026-07-15",
    verified=False,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)

# A distinct `model_key`/`ag_key` keeps TabDPT-1.3 a separate leaderboard method from the two above.
# Run `tabdpt13_21092026` (BENCHMARK_LOG.md), hosted as suite `tabarena-2026-09-21`.
tabdpt_v13_method_metadata = tabdpt_v13_descriptor.method_metadata(
    method="TabDPT-1.3",
    suite="tabarena-2026-09-21",
    ag_key="TA-TABDPT-1.3",
    model_key="TABDPT-1.3",
    config_default="TabDPT-1.3_c1_default_BAG_L1",
    can_hpo=False,
    validation_protocol="8x1",
    date="2026-09-21",
    verified=False,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)


# Pinned below 1.2: this entry runs the `tabdpt1_1` checkpoint, whose architecture config
# (8 keys, no `enc_cell_dim`) the 1.2 loader cannot read — `TabDPTModel.load` reads v1.2-only keys
# with no legacy branch. Every later entry needs a newer `tabdpt`, so this one cannot be installed
# together with them, which is what `superseded` records.
tabdpt_info = ModelInfo(
    model_cls=TabDPTModel,
    search_space=gen_tabdpt,
    method_metadata=tabdpt_method_metadata,
    pip_extra=("tabdpt<1.2",),
    prefetch_weights=TabDPTModel.prefetch_weights,
    superseded=True,
)


# Pinned below 1.3: the 1.3 release renamed the network's label encoders, so the 1.3 loader cannot
# read `tabdpt1_2.safetensors` (nor the 1.2 loader `tabdpt1_3.safetensors`). TabDPT-1.3 below is
# the installable entry.
tabdpt_turbo_info = ModelInfo(
    model_cls=TabDPTTurboModel,
    search_space=gen_tabdpt_turbo,
    method_metadata=tabdpt_turbo_method_metadata,
    pip_extra=("tabdpt>=1.2.0,<1.3",),
    prefetch_weights=TabDPTTurboModel.prefetch_weights,
    superseded=True,
)


# Pinned to the merge of layer6ai-labs/TabDPT-inference#79 (2026-09-21), which added
# `TabDPTEstimator._load_model`, the shared-weights loader `TabDPTv13Model` declares, after the 1.3.0
# release. Move to `tabdpt>=<version>` once a release ships it. Keep in sync with the `tabdpt` extra
# in pyproject.toml.
tabdpt_v13_info = ModelInfo(
    model_cls=TabDPTv13Model,
    search_space=gen_tabdpt_v13,
    method_metadata=tabdpt_v13_method_metadata,
    pip_extra=(
        "tabdpt @ git+https://github.com/layer6ai-labs/TabDPT-inference.git@336ed08fd38ebdd63fbe5b345734f7df287aa5d6",
    ),
    prefetch_weights=TabDPTv13Model.prefetch_weights,
)
