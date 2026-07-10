from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabswift.hpo import gen_tabswift
from tabarena.models.tabswift.model import TabSwiftModel

tabswift_method_metadata = MethodMetadata.config(
    method="TabSwift",
    suite="tabarena-2026-07-06",
    ag_key="TA-TABSWIFT",
    config_default="TabSwift_c1_default_BAG_L1",
    can_hpo=False,
    compute="gpu",
    is_bag=False,
    date="2026-07-06",
    reference_url="https://github.com/LAMDA-Tabular/TabSwift",
    display_name="TabSwift",
    verified=False,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)


tabswift_info = ModelInfo(
    model_cls=TabSwiftModel,
    search_space=gen_tabswift,
    method_metadata=tabswift_method_metadata,
    # Vendored under `_vendor/`; all runtime deps (torch, numpy, scikit-learn, scipy, psutil,
    # tqdm, huggingface_hub, packaging) are already in TabArena's base tree, so no pip extra.
    pip_extra=(),
    prefetch_weights=TabSwiftModel.prefetch_weights,
)
