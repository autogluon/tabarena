import numpy as np


def convert_numpy_dtypes(data: dict) -> dict:
    """
    Converts NumPy dtypes in a dictionary to Python dtypes.
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
            converted_data[key] = [convert_numpy_dtypes({i: v})[i] if isinstance(v, (dict, np.generic)) else v for i, v in enumerate(value)]
        else:
            converted_data[key] = value
    return converted_data
