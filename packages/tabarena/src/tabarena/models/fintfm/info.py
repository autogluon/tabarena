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
    #: ``can_hpo`` records whether the raw results hold more than one config, and the results
    #: produced so far hold only the default. ``hpo.py`` now declares a search space; set this
    #: to ``True`` once a run that includes sampled configs is the one being registered.
    can_hpo=False,
    #: Runs on CPU, MPS and CUDA; the reference machine has no CUDA at all.
    compute="cpu",
    is_bag=True,
    date_introduced="2026-09",
    reference_url="https://github.com/kabartay/fintfm",
    license="Apache-2.0",
    commercial_use=True,
    #: Not signed off by the TabArena maintainers (no merged submission), so no verified mark.
    verified=False,
)

fintfm_info = ModelInfo(
    model_cls=FinTFMModel,
    search_space=gen_fintfm,
    method_metadata=fintfm_method_metadata,
    pip_extra=("fintfm==0.10.1",),
    #: The binary checkpoint is public, ungated and Apache-2.0, pinned to a commit so the
    #: weights behind a recorded leaderboard number cannot change. Staged before a run rather
    #: than downloaded inside the first fit, where the transfer would be charged to that fit's
    #: time limit.
    prefetch_weights=prefetch_weights,
)
