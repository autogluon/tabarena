from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabpfn_3_5.hpo import gen_tabpfn_3_5, gen_tabpfn_3_5_fast
from tabarena.models.tabpfn_3_5.model import TabPFN35FastModel, TabPFN35Model

tabpfn_3_5_descriptor = ModelDescriptor(
    display_name="TabPFN-3.5",
    compute="gpu",
    is_bag=False,
    reference_url="https://arxiv.org/abs/2609.17895",
    date_introduced="2026-09-15",
)

tabpfn_3_5_fast_descriptor = ModelDescriptor(
    display_name="TabPFN-3.5-Fast",
    compute="gpu",
    is_bag=False,
    reference_url="https://arxiv.org/abs/2609.17895",
    date_introduced="2026-09-15",
)

# Not benchmarked yet: no suite and no hosted artifacts until a run exists (the upload step fills
# them in and confirms the config name).
tabpfn_3_5_method_metadata = tabpfn_3_5_descriptor.method_metadata(
    method="TabPFN-3.5",
    ag_key="TA-TABPFN-3.5",
    can_hpo=False,
    date="2026-09-16",
    verified=False,
)

tabpfn_3_5_fast_method_metadata = tabpfn_3_5_fast_descriptor.method_metadata(
    method="TabPFN-3.5-Fast",
    ag_key="TA-TABPFN-3.5-FAST",
    can_hpo=False,
    date="2026-09-16",
    verified=False,
)


tabpfn_3_5_info = ModelInfo(
    model_cls=TabPFN35Model,
    search_space=gen_tabpfn_3_5,
    method_metadata=tabpfn_3_5_method_metadata,
    pip_extra=("tabpfn>=9.0.0",),
    prefetch_weights=TabPFN35Model.prefetch_weights,
)

tabpfn_3_5_fast_info = ModelInfo(
    model_cls=TabPFN35FastModel,
    search_space=gen_tabpfn_3_5_fast,
    method_metadata=tabpfn_3_5_fast_method_metadata,
    pip_extra=("tabpfn>=9.0.0",),
    prefetch_weights=TabPFN35FastModel.prefetch_weights,
)
