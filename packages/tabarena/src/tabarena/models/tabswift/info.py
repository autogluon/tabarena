from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata, ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabswift.hpo import gen_tabswift
from tabarena.models.tabswift.model import TabSwiftModel

#: Intrinsic facts shared by every run of the model (the BeyondArena entries build on it).
tabswift_descriptor = ModelDescriptor(
    display_name="TabSwift",
    compute="gpu",
    is_bag=False,
    reference_url="https://github.com/LAMDA-Tabular/TabSwift",
    license="Apache-2.0 (weights); MIT (code)",
    date_introduced="2026-06-05",
)

# Superseded by the rerun below; kept so the hosted artifacts stay loadable.
tabswift_method_metadata = MethodMetadata.config(
    method="TabSwift",
    suite="tabarena-2026-07-06",
    ag_key="TA-TABSWIFT",
    config_default="TabSwift_c1_default_BAG_L1",
    can_hpo=False,
    compute="gpu",
    is_bag=False,
    date="2026-07-06",
    date_introduced="2026-06-05",
    reference_url="https://github.com/LAMDA-Tabular/TabSwift",
    license="Apache-2.0 (weights); MIT (code)",
    display_name="TabSwift",
    verified=False,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)

# Pareto-front rerun with the improved train/infer time measurement; superseded by the rerun below.
tabswift_new_method_metadata = MethodMetadata.config(
    method="TabSwift",
    suite="tabarena-2026-07-13",
    ag_key="TA-TABSWIFT",
    config_default="TabSwift_c1_default_BAG_L1",
    can_hpo=False,
    compute="gpu",
    is_bag=False,
    date="2026-07-15",
    date_introduced="2026-06-05",
    reference_url="https://github.com/LAMDA-Tabular/TabSwift",
    license="Apache-2.0 (weights); MIT (code)",
    display_name="TabSwift",
    verified=False,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)

# Rerun on the timing and warm-up pipeline (PR #584).
tabswift_2026_09_method_metadata = MethodMetadata.config(
    method="TabSwift",
    validation_protocol="8x1",
    suite="tabarena-2026-09-16",
    ag_key="TA-TABSWIFT",
    config_default="TabSwift_c1_default_BAG_L1",
    can_hpo=False,
    compute="gpu",
    is_bag=False,
    date="2026-09-16",
    date_introduced="2026-06-05",
    reference_url="https://github.com/LAMDA-Tabular/TabSwift",
    display_name="TabSwift",
    verified=False,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
    license="Apache-2.0 (weights); MIT (code)",
)


tabswift_info = ModelInfo(
    model_cls=TabSwiftModel,
    search_space=gen_tabswift,
    method_metadata=tabswift_2026_09_method_metadata,
    # Vendored under `_vendor/`; all runtime deps (torch, numpy, scikit-learn, scipy, psutil,
    # tqdm, huggingface_hub, packaging) are already in TabArena's base tree, so no pip extra.
    pip_extra=(),
    prefetch_weights=TabSwiftModel.prefetch_weights,
)
