from __future__ import annotations

from autogluon.tabular.models import XTModel

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.extra_trees.hpo import gen_extratrees

extra_trees_descriptor = ModelDescriptor(
    display_name="ExtraTrees",
    compute="cpu",
    is_bag=True,
    reference_url="https://link.springer.com/article/10.1007/s10994-006-6226-1",
    date_introduced="2006",
)

# Superseded by the rerun below; kept so the hosted artifacts stay loadable.
extra_trees_method_metadata = extra_trees_descriptor.method_metadata(
    method="ExtraTrees",
    suite="tabarena-2025-06-12",
    ag_key="XT",
    config_default="ExtraTrees_c1_BAG_L1",
    cache_type="s3",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache", "upload_as_public": True},
    date="2025-06-12",
    # FIXME: technically baselines are not verified
)

# Pareto-front rerun with the improved train/infer time measurement.
extra_trees_new_method_metadata = extra_trees_descriptor.method_metadata(
    method="ExtraTrees",
    suite="tabarena-2026-07-13",
    ag_key="XT",
    config_default="ExtraTrees_c1_default_BAG_L1",
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
    date="2026-07-15",
)


extra_trees_info = ModelInfo(
    model_cls=XTModel,
    search_space=gen_extratrees,
    method_metadata=extra_trees_new_method_metadata,
)
