from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.fintfm_linprobe.hpo import gen_fintfm_linprobe
from tabarena.models.fintfm_linprobe.model import FinTFMLinearProbeModel, prefetch_weights

fintfm_linprobe_method_metadata = MethodMetadata.config(
    method="FinTFM-LinProbe",
    display_name="FinTFM with a trained linear readout",
    #: Distinct from ``TA-FINTFM`` so this exploratory variant's results never share a cache
    #: entry, a config-name collision, or a leaderboard row with the real entrant.
    ag_key="TA-FINTFM-LINPROBE",
    model_key="FINTFM-LINPROBE",
    config_default="FinTFM-LinProbe_c1_default_BAG_L1",
    #: The raw results hold only the default config. ``hpo.py`` holds a proposed space that has
    #: not been run; set this to ``True`` once a run with sampled configs is registered.
    can_hpo=False,
    compute="cpu",
    is_bag=True,
    date_introduced="2026-09",
    reference_url="https://github.com/kabartay/fintfm",
    license="Apache-2.0",
    commercial_use=True,
    #: Not signed off by the TabArena maintainers: the entry is Lite-only and its pull request is
    #: on hold, so it carries no verified mark.
    verified=False,
)

fintfm_linprobe_info = ModelInfo(
    model_cls=FinTFMLinearProbeModel,
    search_space=gen_fintfm_linprobe,
    method_metadata=fintfm_linprobe_method_metadata,
    pip_extra=("fintfm==0.11.0",),
    prefetch_weights=prefetch_weights,
)
