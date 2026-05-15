from __future__ import annotations

from tabarena.models.mitra.info import mitra_method_metadata
from tabarena.models.realmlp.info import realmlp_method_metadata
from tabarena.models.xrfm.info import xrfm_method_metadata
from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata

# Legacy aliases — preserved for any external code still importing the older names.
mitra_metadata = mitra_method_metadata
realmlp_gpu_metadata = realmlp_method_metadata
xrfm_metadata = xrfm_method_metadata

_common_kwargs = dict(
    artifact_name="tabarena-2025-09-03",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    has_raw=True,
    has_processed=True,
    has_results=True,
    name_suffix=None,
)

# New methods (tabarena-2025-09-03)
ag_140_metadata = MethodMetadata(
    method="AutoGluon_v140",
    method_type="baseline",
    display_name="AutoGluon 1.4 (4h)",
    compute="gpu",
    date="2025-09-03",
    verified=True,
    reference_url="https://arxiv.org/abs/2003.06505",
    **_common_kwargs,
)
limix_metadata = MethodMetadata(
    method="LimiX_GPU",
    method_type="config",
    display_name="LimiX",
    compute="gpu",
    date="2025-09-03",
    ag_key="LIMIX",
    model_key="LIMIX_GPU",
    config_default="LimiX_GPU_c1_BAG_L1",
    can_hpo=False,
    is_bag=False,
    verified=False,
    reference_url="https://arxiv.org/abs/2509.03505",
    **_common_kwargs,
)
ebm_metadata = MethodMetadata(
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
    **_common_kwargs,
)
tabflex_metadata = MethodMetadata(
    method="TabFlex_GPU",
    method_type="config",
    display_name="TabFlex",
    compute="gpu",
    date="2025-09-03",
    ag_key="TABFLEX",
    model_key="TABFLEX_GPU",
    config_default="TabFlex_GPU_c1_BAG_L1",
    can_hpo=False,
    is_bag=False,
    verified=False,
    reference_url="https://arxiv.org/abs/2506.05584",
    **_common_kwargs,
)
betatabpfn_metadata = MethodMetadata(
    method="BetaTabPFN_GPU",
    method_type="config",
    display_name="BetaTabPFN",
    compute="gpu",
    date="2025-09-03",
    ag_key="BETA",
    model_key="BETA_GPU",
    config_default="BetaTabPFN_GPU_c1_BAG_L1",
    can_hpo=False,
    is_bag=True,
    verified=False,
    reference_url="https://arxiv.org/abs/2502.02527",
    **_common_kwargs,
)
