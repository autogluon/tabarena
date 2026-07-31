from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.exaone_tabular.hpo import gen_exaone_tabular
from tabarena.models.exaone_tabular.model import EXAONETabularModel

exaone_tabular_method_metadata = MethodMetadata.config(
    method="EXAONE-Tabular",
    ag_key="TA-EXAONE-TABULAR",
    config_default="EXAONE-Tabular_c1_default_BAG_L1",
    can_hpo=False,
    compute="gpu",
    is_bag=False,
    date="2026-07-31",
    date_introduced="2026-07-31",
    reference_url="https://github.com/LGAI-Research/EXAONE-Tabular",
    display_name="EXAONE-Tabular",
    verified=False,
    # No `suite` / `cache_kwargs` yet: not benchmarked, so the artifacts are local-only. Both are
    # filled in by the upload flow once a run exists.
)


exaone_tabular_info = ModelInfo(
    model_cls=EXAONETabularModel,
    search_space=gen_exaone_tabular,
    method_metadata=exaone_tabular_method_metadata,
    # Pinned to a commit: the package is not on PyPI and the repository was published on
    # 2026-07-31, so an unpinned `git+...` would silently change what gets benchmarked.
    pip_extra=(
        "exaonetabular @ git+https://github.com/LGAI-Research/EXAONE-Tabular.git@f8e9cc5a8befa432b2d783bdaffc413384fa2263",
    ),
    prefetch_weights=EXAONETabularModel.prefetch_weights,
)
