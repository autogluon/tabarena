# Model Implementation Patterns for TabArena

Reference patterns for the `add-model` skill. These are annotated templates based on real implementations in the codebase.

Every model lives in a single folder at `tabarena/tabarena/models/{ModelKey}/` with this layout:

```
tabarena/models/{ModelKey}/
  __init__.py    # re-exports gen_{ModelKey}, {ModelKey}_info, {ModelKey}_method_metadata
  model.py       # the AutoGluon wrapper class
  hpo.py         # ConfigGenerator + search space
  info.py        # ModelInfo + MethodMetadata
  _internal/     # (optional) hand-written helper modules
  _vendor/       # (optional) verbatim upstream code
```

---

## Model wrapper template (`model.py`)

Full template for the AutoGluon wrapper. Adapt based on model type.

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

## hpo.py template

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

## info.py template

Defines the per-model `MethodMetadata` + `ModelInfo`. `info.py` is the file `discover_models()` walks — populating it is what makes the model visible to the registry.

```python
from __future__ import annotations

from tabarena.models._model_info import ModelInfo
from tabarena.models.{ModelKey}.hpo import gen_{ModelKey}
from tabarena.models.{ModelKey}.model import {ClassName}Model
from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata


{ModelKey}_method_metadata = MethodMetadata(
    method="{ModelName}",                  # e.g. "TabSTAR"
    method_type="config",
    display_name="{ModelName}",
    compute="gpu",                          # or "cpu"
    date="YYYY-MM-DD",                     # date of the benchmarking run (or planning date if unbenchmarked)
    ag_key="{ag_key_without_TA}",          # e.g. "TABSTAR" (matches {ClassName}Model.ag_key without the TA- prefix)
    model_key="{ag_key_without_TA}",
    config_default="{ModelName}_c1_BAG_L1",
    can_hpo=True,
    is_bag=True,
    has_raw=True,
    has_processed=True,
    has_results=True,
    artifact_name="tabarena-YYYY-MM-DD",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    name_suffix=None,
    verified=False,                         # flip to True once benchmark run is signed off
    reference_url="{doc_url}",
    cache_type="r2",
)


{ModelKey}_info = ModelInfo(
    model_cls={ClassName}Model,
    search_space=gen_{ModelKey},
    method_metadata={ModelKey}_method_metadata,
    pip_extra=("{pip_package}",),
)
```

`pip_extra` is the tuple of pip specs the auto-discovery uses when computing what extras to install for this model — list every dependency the wrapper imports lazily.

---

## __init__.py template

```python
from __future__ import annotations

from tabarena.models.{ModelKey}.hpo import gen_{ModelKey}
from tabarena.models.{ModelKey}.info import {ModelKey}_info, {ModelKey}_method_metadata

__all__ = ["gen_{ModelKey}", "{ModelKey}_info", "{ModelKey}_method_metadata"]
```

---

## Test template

Tests for `tabarena/tabarena/models/{ModelKey}/model.py` live at `tst/models/test_{ModelKey}.py`.

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

### `tabarena/tabarena/models/__init__.py` — lazy class entry

```python
# Add to _LAZY_CLASSES (keep alphabetised by class name):
_LAZY_CLASSES = {
    ...
    "{ClassName}Model": "tabarena.models.{ModelKey}.model",
    ...
}

# Add to __all__ (keep sorted):
__all__ = [
    ...
    "{ClassName}Model",
    ...
]

# Add to the TYPE_CHECKING block (keep sorted):
if TYPE_CHECKING:
    ...
    from tabarena.models.{ModelKey}.model import {ClassName}Model
```

### `tabarena/tabarena/models/utils.py` — `name_to_import_map` entry

Used by `get_configs_generator_from_name()`. The key is the friendly model name (typically same as `ModelName`):

```python
"{ModelName}": lambda: importlib.import_module("tabarena.models.{ModelKey}.hpo").gen_{ModelKey},
```

### `tabarena/pyproject.toml`

```toml
# In [project.optional-dependencies]:
{ModelKey} = ["{pip_package}"]

# In the extended extra (append "tabarena[{ModelKey}]" — keep the list sorted):
extended = [
  ...
  "tabarena[{ModelKey}]",
]
```

---

## Multi-file models (optional)

If the wrapper needs supporting modules, organise them under a private subfolder of `tabarena/tabarena/models/{ModelKey}/`:

```
tabarena/models/modernnca/        # example of a multi-file model
  __init__.py
  hpo.py
  info.py
  model.py
  _internal/
    __init__.py
    base.py
    data.py
    modernnca_method.py
    num_embeddings.py
    ...

tabarena/models/limix/             # example with vendored upstream code
  __init__.py
  hpo.py
  info.py
  model.py
  _vendor/
    __init__.py
    LICENSE.txt
    inference/
    model/
    utils/
```

Conventions:
- **`_internal/`** is the default for hand-written helpers (preprocessors, adapters, glue).
- **`_vendor/`** is reserved for code copied verbatim from an upstream project — keep the original layout and ship the license file.
- Both subfolders are private to the model; everything imports through absolute paths like `tabarena.models.{ModelKey}._internal.<submodule>`.
- A single model folder may legitimately contain **both** `_internal/` and `_vendor/` if it has hand-written wrapper helpers around a vendored library.

---

## Metadata artifacts

`info.py` is now the single source of truth for `MethodMetadata`. There's no separate "add a metadata entry" step when first introducing a model.

If/when the model has been benchmarked and the results are ready to be registered in TabArena's downstream artifact-aggregation files, also import the metadata you defined in `info.py` into the dated batch file under `tabarena/tabarena/nips2025_utils/artifacts/_tabarena_method_metadata_YYYY_MM_DD.py`. That step is for downstream artifact handling only — it is not required for the model to work in the registry.

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
