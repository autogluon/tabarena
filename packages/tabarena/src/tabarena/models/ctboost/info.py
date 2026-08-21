from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.ctboost.hpo import gen_ctboost
from tabarena.models.ctboost.model import CTBoostModel

ctboost_method_metadata = MethodMetadata.config(
    method="CTBoost",
    display_name="CTBoost",
    compute="cpu",
    date="2026-08-21",
    date_introduced="2026-04-10",
    ag_key="CTB",
    model_key="CTB",
    config_default="CTBoost_c1_default_BAG_L1",
    can_hpo=True,
    is_bag=True,
    has_raw=True,
    has_processed=True,
    has_results=True,
    suite="tabarena-2026-08-21",
    verified=False,
    reference_url="https://github.com/captnmarkus/ctboost",
)


ctboost_info = ModelInfo(
    model_cls=CTBoostModel,
    search_space=gen_ctboost,
    method_metadata=ctboost_method_metadata,
    pip_extra=("ctboost>=0.1.56",),
)
