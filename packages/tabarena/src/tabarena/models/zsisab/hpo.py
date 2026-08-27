from __future__ import annotations

from tabarena.models.zsisab.model import ZSISABModel
from tabarena.utils.config_utils import ConfigGenerator

gen_zsisab = ConfigGenerator(
    model_cls=ZSISABModel,
    search_space={
        "num_prototypes": [64, 128, 256, 512, 1024],
        "chunk_size": [4096, 8192, 16384, 32768],
        "n_ensemble": [1, 4, 8, 16, 32],
    },
    manual_configs=[
        {
            "num_prototypes": 512,
            "chunk_size": 16384,
            "n_ensemble": 32,
        }
    ],
)
