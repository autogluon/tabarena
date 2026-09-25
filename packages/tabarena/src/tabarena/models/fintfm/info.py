from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.fintfm.hpo import gen_fintfm
from tabarena.models.fintfm.model import FinTFMModel, prefetch_weights

fintfm_method_metadata = MethodMetadata.config(
    method="FinTFM",
    display_name="FinTFM",
    #: ``TA-`` prefixed like every other entrant; the unprefixed public name lives in
    #: ``display_name`` and the wrapper's ``ag_name``.
    ag_key="TA-FINTFM",
    model_key="FINTFM",
    config_default="FinTFM_c1_default_BAG_L1",
    #: HPO is off because ``hpo.py`` declares no search space. Each inference-time knob was
    #: measured and found flat or harmful: context size is nearly flat (fintfm
    #: ``docs/results/FINDINGS.md`` S83/S84), retrieval harms at low prevalence (S70), and
    #: ensembling over column-identity draws is a correctness setting rather than a
    #: hyperparameter (``docs/design/DECISIONS.md`` D12).
    can_hpo=False,
    #: Runs on CPU, MPS and CUDA; the reference machine has no CUDA at all.
    compute="cpu",
    is_bag=True,
    date_introduced="2026-09",
    reference_url="https://github.com/kabartay/fintfm",
    license="Apache-2.0",
    commercial_use=True,
)

fintfm_info = ModelInfo(
    model_cls=FinTFMModel,
    search_space=gen_fintfm,
    method_metadata=fintfm_method_metadata,
    pip_extra=("fintfm==0.5.5",),
    #: The binary checkpoint is public, ungated and Apache-2.0, pinned to a commit so the
    #: weights behind a recorded leaderboard number cannot change. Staged before a run rather
    #: than downloaded inside the first fit, where the transfer would be charged to that fit's
    #: time limit.
    prefetch_weights=prefetch_weights,
)
