from __future__ import annotations

try:
    from tabarena.models.zsisab.model import ZSISABModel
    from tabarena.utils.config_utils import ConfigGenerator

    gen_zsisab = ConfigGenerator(
        model_cls=ZSISABModel,
        search_space={
            "num_prototypes": [128, 256, 512, 1024],
            "chunk_size": [8192, 16384, 32768],
        },
        manual_configs=[
            {
                "num_prototypes": 512,
                "chunk_size": 16384,
                "n_ensemble": 32,
            }
        ],
    )
except ImportError:
    gen_zsisab = None
