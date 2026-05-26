# Model Implementation Patterns for TabArena

Reference patterns for the `add-model` skill. These are annotated templates based on real implementations in the codebase.

---

## Model wrapper template

Full template for `{ModelKey}_model.py`. Adapt based on model type.

```python
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel
# For CPU/sklearn models use instead:
# from autogluon.core.models import AbstractModel

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class {ClassName}Model(AbstractTorchModel):
    """{ModelName}: {doc_url}.

    Paper: {paper_title}
    Authors: {authors}
    Codebase: {github_url}
    License: {license}
    """

    ag_key = "TA-{MODEL_KEY_UPPER}"   # e.g. "TA-TABPFN-2.6"
    ag_name = "TA-{ModelName}"         # e.g. "TA-TabPFN-2.6"
    ag_priority = 65
    seed_name = "random_state"

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        import torch

        available_num_gpus = ResourceManager.get_gpu_count_torch(cuda_only=True)
        if num_gpus > available_num_gpus:
            raise AssertionError(
                f"Fit specified to use {num_gpus} GPU, but only {available_num_gpus} "
                "CUDA GPUs are available. Please activate CUDA or switch to CPU usage.",
            )
        device = "cuda" if num_gpus != 0 else "cpu"
        if (device == "cuda") and (not torch.cuda.is_available()):
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead.",
            )

        from {pip_module} import {Classifier}, {Regressor}

        if self.problem_type in ["binary", "multiclass"]:
            model_cls = {Classifier}
        elif self.problem_type == "regression":
            model_cls = {Regressor}
        else:
            raise AssertionError(f"Unsupported problem_type: {self.problem_type}")

        hps = self._get_model_params()

        self.model = model_cls(
            **hps,
            device=device,
        )

        # If model needs a validation split and none is provided:
        if X_val is None:
            from autogluon.core.utils import generate_train_test_split
            X, X_val, y, y_val = generate_train_test_split(
                X=X, y=y, problem_type=self.problem_type,
                test_size=0.33, random_state=0,
            )

        X = self.preprocess(X, y=y)
        if X_val is not None:
            X_val = self.preprocess(X_val)

        self.model.fit(X=X, y=y, X_val=X_val, y_val=y_val)

    def _set_default_params(self):
        default_params = {
            # Add model-specific defaults here
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def get_device(self) -> str:
        return self.model.device

    def _set_device(self, device: str):
        self.model.to(device)

    def _get_default_resources(self) -> tuple[int, int]:
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))
        return num_cpus, num_gpus

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, int | float]:
        return {
            "num_cpus": 1,
            "num_gpus": 1 if is_gpu_available else 0,
        }

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        """Set fold_fitting_strategy to sequential_local to avoid crashes
        if model weights aren't pre-downloaded when fitting in parallel.
        """
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        default_ag_args_ensemble.update({"fold_fitting_strategy": "sequential_local"})
        return default_ag_args_ensemble

    @classmethod
    def _class_tags(cls) -> dict:
        # TODO: implement memory estimation and set to True
        return {"can_estimate_memory_usage_static": False}

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
```

### CPU/sklearn model variant

For models without GPU support, use `AbstractModel` instead and remove GPU-related methods:

```python
from autogluon.core.models import AbstractModel

class {ClassName}Model(AbstractModel):
    ag_key = "TA-{MODEL_KEY_UPPER}"
    ag_name = "TA-{ModelName}"
    ag_priority = 65
    seed_name = "random_state"

    def _fit(self, X, y, num_cpus=1, num_gpus=0, **kwargs):
        from {pip_module} import {ModelClass}
        hps = self._get_model_params()
        self.model = {ModelClass}(**hps)
        X = self.preprocess(X, y=y)
        self.model.fit(X=X, y=y)

    def _set_default_params(self):
        default_params = {}
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    @classmethod
    def _class_tags(cls) -> dict:
        return {"can_estimate_memory_usage_static": False}

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
```

---

## generate.py template

Default: empty search space (fine for foundation models with a single checkpoint). Add hyperparameters only when explicitly requested.

```python
from __future__ import annotations

from tabarena.models.{ModelKey}.model import {ClassName}Model
from tabarena.utils.config_utils import ConfigGenerator

gen_{ModelKey} = ConfigGenerator(
    model_cls={ClassName}Model,
    search_space={},
    manual_configs=[{}],
)

if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_{ModelKey}.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
```

### With hyperparameter search space (optional)

```python
from autogluon.common.space import Categorical, Real, Int

search_space = {
    "learning_rate": Real(1e-4, 1e-2, log=True),
    "n_layers": Int(2, 6),
    "dropout": Categorical(0.0, 0.1, 0.2, 0.3),
}

gen_{ModelKey} = ConfigGenerator(
    model_cls={ClassName}Model,
    search_space=search_space,
    manual_configs=[{}],
)
```

---

## Test template

```python
from __future__ import annotations

import pytest


def test_{ModelKey}():
    try:
        from autogluon.tabular.testing import FitHelper
        from tabarena.models.{ModelKey}.model import {ClassName}Model

        model_cls = {ClassName}Model
        # Add a speed-up hyperparameter if the model supports one (e.g. max_epochs=1)
        FitHelper.verify_model(model_cls=model_cls, model_hyperparameters={})
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
```

---

## Registry update snippets

### `benchmark/models/ag/__init__.py` — sorted import + __all__

```python
# Add import (keep sorted by module path):
from tabarena.models.{ModelKey}.model import {ClassName}Model

# Add to __all__ (keep sorted):
__all__ = [
    ...
    "{ClassName}Model",
    ...
]
```

### `benchmark/models/model_registry.py`

```python
# In the import block:
from tabarena.benchmark.models.ag import (
    ...
    {ClassName}Model,
)

# In _models_to_add list:
_models_to_add = [
    ...
    {ClassName}Model,
]
```

### `models/utils.py` — name_to_import_map entry

The key is the `ag_name` string (e.g. `"TA-TabPFN-2.6"`):

```python
"TA-{ModelName}": lambda: importlib.import_module("tabarena.models.{ModelKey}.generate").gen_{ModelKey},
```

### `pyproject.toml`

```toml
# In [project.optional-dependencies]:
{ModelKey} = ["{pip_package}"]

# In the benchmark extra (append to the list):
benchmark = [
  ...
  "{pip_package}",
]
```

---

## Metadata artifacts

Models that are fully benchmarked in TabArena may also have entries in:
- `tabarena/tabarena/nips2025_utils/artifacts/` — check what files exist there for naming/metadata conventions

You do NOT need to add metadata artifacts when first adding a model — those are generated after benchmarking runs.

---

## Common patterns from existing models

### Handling missing validation split
```python
if X_val is None:
    from autogluon.core.utils import generate_train_test_split
    X, X_val, y, y_val = generate_train_test_split(
        X=X, y=y, problem_type=self.problem_type, test_size=0.33, random_state=0,
    )
```

### Inverse label transform (needed if passing original labels to external model)
```python
if self.problem_type in ["binary", "multiclass"]:
    y = self.label_cleaner.inverse_transform(y)
    if y_val is not None:
        y_val = self.label_cleaner.inverse_transform(y_val)
```

### fixed_random_state (for models where random state affects preprocessing)
```python
fixed_random_state: int = 0
# In _fit():
if self.fixed_random_state is not None:
    hps[self.seed_name] = self.fixed_random_state
```

### max_rows / max_features limits
```python
def _get_default_auxiliary_params(self) -> dict:
    default_auxiliary_params = super()._get_default_auxiliary_params()
    default_auxiliary_params.update({
        "max_rows": 100_000,
        "max_features": 2000,
    })
    return default_auxiliary_params
```
