from __future__ import annotations

import numpy as np


def convert_numpy_dtypes(data: dict) -> dict:
    """Converts NumPy dtypes in a dictionary to Python dtypes.
    Some hyperparameter search space's generate configs with
    numpy dtypes which aren't serializable to yaml. This fixes that.
    """
    converted_data = {}
    for key, value in data.items():
        if isinstance(value, np.generic):
            converted_data[key] = value.item()
        elif isinstance(value, dict):
            converted_data[key] = convert_numpy_dtypes(value)
        elif isinstance(value, list):
            converted_data[key] = [
                convert_numpy_dtypes({i: v})[i] if isinstance(v, (dict, np.generic)) else v for i, v in enumerate(value)
            ]
        else:
            converted_data[key] = value
    return converted_data


def get_configs_generator_from_name(model_name: str):
    """Map a friendly model name to its search-space generator (`gen_<key>`).

    Resolves against the auto-discovered `MODEL_REGISTRY`: the first
    `ModelInfo` whose `method_metadata.display_name` or
    `method_metadata.method` equals `model_name` wins. CPU and GPU variants
    of the same model share the same `ConfigGenerator` instance, so any tie
    between compute variants returns the same generator either way.
    """
    from tabarena.models import get_model_registry

    registry = get_model_registry()
    for info in registry.values():
        md = info.method_metadata
        if model_name in (md.display_name, md.method):
            return info.search_space

    available = sorted({
        name
        for info in registry.values()
        for name in (info.method_metadata.display_name, info.method_metadata.method)
        if name
    })
    raise ValueError(
        f"Model name {model_name!r} is not recognized. Options are: {available}"
    )
