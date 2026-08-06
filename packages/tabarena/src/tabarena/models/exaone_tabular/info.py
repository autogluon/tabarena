from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.exaone_tabular.hpo import gen_exaone_tabular
from tabarena.models.exaone_tabular.model import EXAONETabularModel

exaone_tabular_method_metadata = MethodMetadata.config(
    method="EXAONE-Tabular",
    suite="tabarena-2026-07-31",
    ag_key="TA-EXAONE-TABULAR",
    config_default="EXAONE-Tabular_c1_default_BAG_L1",
    can_hpo=False,
    compute="gpu",
    is_bag=False,
    date="2026-07-31",
    date_introduced="2026-07-31",
    reference_url="https://github.com/LGAI-Research/EXAONE-Tabular",
    display_name="EXAONE-Tabular",
    verified=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)


exaone_tabular_info = ModelInfo(
    model_cls=EXAONETabularModel,
    search_space=gen_exaone_tabular,
    method_metadata=exaone_tabular_method_metadata,
    pip_extra=(
        "exaonetabular @ git+https://github.com/LGAI-Research/EXAONE-Tabular.git@6cca1af2395663837e104d2efd8d37fea89fe688",
    ),
    prefetch_weights=EXAONETabularModel.prefetch_weights,
)
