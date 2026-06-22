from __future__ import annotations

from autogluon.tabular.models import EBMModel

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.ebm.hpo import gen_ebm

ebm_method_metadata = MethodMetadata(
    method="ExplainableBM",
    method_type="config",
    display_name="EBM",
    compute="cpu",
    date="2025-09-03",
    ag_key="EBM",
    config_default="ExplainableBM_c1_BAG_L1",
    can_hpo=True,
    is_bag=True,
    verified=True,
    reference_url="https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf",
    artifact_name="tabarena-2025-09-03",
    cache_type="s3",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    has_raw=True,
    has_processed=True,
    has_results=True,
    name_suffix=None,
)


ebm_info = ModelInfo(
    model_cls=EBMModel,
    search_space=gen_ebm,
    method_metadata=ebm_method_metadata,
    pip_extra=("autogluon.tabular[interpret]>=1.5,<1.6",),
)
