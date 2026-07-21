from __future__ import annotations

from autogluon.tabular.models import EBMModel

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.ebm.hpo import gen_ebm

# Superseded by the rerun below; kept so the hosted artifacts stay loadable.
ebm_method_metadata = MethodMetadata.config(
    method="ExplainableBM",
    suite="tabarena-2025-09-03",
    ag_key="EBM",
    config_default="ExplainableBM_c1_BAG_L1",
    compute="cpu",
    is_bag=True,
    cache_type="s3",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache", "upload_as_public": True},
    date="2025-09-03",
    date_introduced="2019-09",
    reference_url="https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf",
    display_name="EBM",
)

# Pareto-front rerun with the improved train/infer time measurement.
ebm_new_method_metadata = MethodMetadata.config(
    method="ExplainableBM",
    suite="tabarena-2026-07-13",
    ag_key="EBM",
    config_default="ExplainableBM_c1_default_BAG_L1",
    compute="cpu",
    is_bag=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
    date="2026-07-15",
    date_introduced="2019-09",
    reference_url="https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf",
    display_name="EBM",
)


ebm_info = ModelInfo(
    model_cls=EBMModel,
    search_space=gen_ebm,
    method_metadata=ebm_new_method_metadata,
    pip_extra=("autogluon.tabular[interpret]>=1.5,<1.6",),
)
