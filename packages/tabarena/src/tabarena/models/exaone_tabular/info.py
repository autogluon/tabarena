from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.exaone_tabular.hpo import gen_exaone_tabular
from tabarena.models.exaone_tabular.model import EXAONETabularModel

# Superseded by the rerun below; kept so the hosted artifacts stay loadable.
exaone_tabular_method_metadata = MethodMetadata.config(
    method="EXAONE-Tabular",
    suite="tabarena-2026-08-06",
    ag_key="TA-EXAONE-TABULAR",
    config_default="EXAONE-Tabular_c1_default_BAG_L1",
    can_hpo=False,
    compute="gpu",
    is_bag=False,
    date="2026-08-06",
    date_introduced="2026-07-31",
    reference_url="https://github.com/LGAI-Research/EXAONE-Tabular",
    commercial_use=False,
    license="EXAONE AI Model License 1.2 - NC",
    display_name="EXAONE-Tabular",
    verified=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)

# Rerun on the timing and warm-up pipeline (PR #584).
exaone_tabular_new_method_metadata = MethodMetadata.config(
    method="EXAONE-Tabular",
    validation_protocol="8x1",
    suite="tabarena-2026-09-16",
    ag_key="TA-EXAONE-TABULAR",
    config_default="EXAONE-Tabular_c1_default_BAG_L1",
    can_hpo=False,
    compute="gpu",
    is_bag=False,
    date="2026-09-16",
    date_introduced="2026-07-31",
    reference_url="https://github.com/LGAI-Research/EXAONE-Tabular",
    display_name="EXAONE-Tabular",
    verified=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
    commercial_use=False,
    license="EXAONE AI Model License 1.2 - NC",
)


exaone_tabular_info = ModelInfo(
    model_cls=EXAONETabularModel,
    search_space=gen_exaone_tabular,
    method_metadata=exaone_tabular_new_method_metadata,
    pip_extra=(
        "exaonetabular @ git+https://github.com/LGAI-Research/EXAONE-Tabular.git@8638e07d09fad154249bd75ba4786181491f7025",
    ),
    prefetch_weights=EXAONETabularModel.prefetch_weights,
)
