from __future__ import annotations

from tabarena.models.fintfm.model import FinTFMModel
from tabarena.utils.config_utils import ConfigGenerator

#: Empty by design. fintfm's inference-time knobs are measured rather than tuned: context size
#: is nearly flat (docs/results/FINDINGS.md S83/S84), retrieval actively harms at low prevalence (S70),
#: and ensembling over column-identity draws is a correctness setting rather than a
#: hyperparameter (docs/design/DECISIONS.md D12). Tuning them here would search a space this project
#: has already measured as flat.
search_space: dict = {}

gen_fintfm = ConfigGenerator(
    model_cls=FinTFMModel,
    search_space=search_space,
    manual_configs=[{}],
)
