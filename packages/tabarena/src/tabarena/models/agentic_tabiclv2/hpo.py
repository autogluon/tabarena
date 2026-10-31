from __future__ import annotations

import numpy as np

from tabarena.models.agentic_tabiclv2.model import AgenticTabICLv2Model
from tabarena.models.utils import convert_numpy_dtypes
from tabarena.utils.config_utils import CustomAGConfigGenerator

#: Module index choices per stage, as in the package's ``bin.tabicl.pipeline.implementation_choices``. The fit
#: keeps the ``_reg`` or ``_clf`` draw of its task.
_INDEX_CHOICES = {
    "model_idx": (0,),
    "feature_preproc_basket_idx": (0, 1, 2),
    "augmentation_basket_idx": (0, 1),
    "postproc_basket_idx_reg": (0, 1),
    "postproc_basket_idx_clf": (0,),
    "feature_permutation_idx": (0, 1, 2, 3),
    "ensemble_idx_reg": (0, 1, 5),
    "ensemble_idx_clf": (0, 1, 2, 5),
}


def generate_single_config_agentic_tabiclv2(rng: np.random.Generator) -> dict:
    return convert_numpy_dtypes({key: rng.choice(choices) for key, choices in _INDEX_CHOICES.items()})


def generate_configs_agentic_tabiclv2(num_random_configs: int = 200, seed: int = 1234) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [generate_single_config_agentic_tabiclv2(rng) for _ in range(num_random_configs)]


gen_agentic_tabiclv2 = CustomAGConfigGenerator(
    model_cls=AgenticTabICLv2Model,
    search_space_func=generate_configs_agentic_tabiclv2,
    manual_configs=[{}],
)
