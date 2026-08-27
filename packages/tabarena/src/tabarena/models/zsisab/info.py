from __future__ import annotations

try:
    from tabarena.models._method_metadata import MethodMetadata
    from tabarena.models._model_info import ModelInfo
    from tabarena.models.zsisab.hpo import gen_zsisab
    from tabarena.models.zsisab.model import ZSISABModel

    zsisab_method_metadata = MethodMetadata.config(
        method="ZS-ISAB",
        ag_key="ZSISAB",
        config_default="ZS-ISAB_c1_BAG_L1",
        can_hpo=True,
        compute="gpu",
        is_bag=False,
        date="2026-08-28",
        date_introduced="2026-08",
        reference_url="https://github.com/iam-saiteja/Zero-Shot-TabPFN",
        display_name="Zero-Shot ISAB",
        verified=False,
    )

    zsisab_info = ModelInfo(
        model_cls=ZSISABModel,
        search_space=gen_zsisab,
        method_metadata=zsisab_method_metadata,
    )
except ImportError:
    zsisab_method_metadata = None
    zsisab_info = None
