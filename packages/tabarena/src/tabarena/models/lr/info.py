from __future__ import annotations

from autogluon.tabular.models import LinearModel

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.lr.hpo import gen_linear

lr_descriptor = ModelDescriptor(
    display_name="Linear",
    compute="cpu",
    is_bag=True,
    reference_url="https://scikit-learn.org/stable/modules/linear_model.html",
)

lr_method_metadata = lr_descriptor.method_metadata(
    method="LinearModel",
    date="2025-10-20",
    ag_key="LR",
    config_default="LinearModel_c1_BAG_L1",
    can_hpo=True,
    has_raw=True,
    has_processed=True,
    artifact_name="tabarena-2025-10-20",
    cache_type="s3",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache", "upload_as_public": True},
    has_results=True,
    name_suffix=None,
    # FIXME: technically LR is not verified
    verified=True,
)


lr_info = ModelInfo(
    model_cls=LinearModel,
    search_space=gen_linear,
    method_metadata=lr_method_metadata,
)
