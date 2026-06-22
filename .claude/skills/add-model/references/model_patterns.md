# Model Implementation Patterns for TabArena

Reference patterns for the `add-model` skill. These are annotated templates based on real implementations in the codebase.

Every model lives in a single folder at `packages/tabarena/src/tabarena/models/{ModelKey}/` with this layout:

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

    def _fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        time_limit=None,
        num_cpus=1,
        num_gpus=0,
        **kwargs,
    ):
        # See "The _fit contract" section below — use X_val/y_val, time_limit, and
        # num_cpus rather than ignoring them. This skeleton only shows the minimum.
        from {pip_module} import {ModelClass}
        hps = self._get_model_params()
        hps["n_jobs"] = num_cpus  # wire the CPU budget to the library's thread arg
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

## The `_fit` contract: use what TabArena hands you

> These rules come straight from real PR review feedback. They are the
> mistakes new wrappers most often make. `_fit` receives `X_val`, `y_val`, `time_limit`,
> `num_cpus`, and `num_gpus` — **use them**, don't ignore them and don't re-derive them. Read
> `models/realmlp/model.py` as the canonical GPU/CPU reference for all four.

### 1. Validation split — use the provided `X_val`/`y_val`, don't carve your own

TabArena has already split off a validation set for early stopping. If the wrapped library supports
early stopping with an eval set, pass `X_val`/`y_val` straight through. **Do not** let the library
auto-split a second holdout out of the training data, and **do not** call `generate_train_test_split`
when a val set was provided — both shrink the training data and hurt performance.

```python
eval_set = None
if X_val is not None and y_val is not None:
    X_val = self.preprocess(X_val)
    eval_set = (X_val, y_val)
# ... model.fit(X, y, eval_set=eval_set, ...)
```

Only generate a split yourself when `X_val is None` (see "Handling missing validation split" below).

### 2. `num_cpus` / `num_gpus` — wire them to the library, never hardcode a default

The scheduler allocates a CPU/GPU budget and passes it into `_fit`. Route it to the library's
thread/device argument (e.g. RealMLP does `n_threads=num_cpus`). **Do not** set the thread count
(`thread_count=-1`, `n_jobs=-1`, …) as a default in `_set_default_params()` — that ignores the
budget and oversubscribes when folds run in parallel.

```python
params["thread_count"] = num_cpus   # set from the _fit arg, not _set_default_params
```

### 3. `time_limit` — honor it, with a little headroom

A wrapper that ignores `time_limit` is not fully TabArena-compatible. Pass the remaining budget to
the library (a `time_to_fit_in_seconds=...` argument, or a wall-clock early-stop callback) and
subtract the time already spent. Leave **~5% headroom** so prediction/cleanup finishes inside the
budget. RealMLP (`realmlp/model.py`, the `time_to_fit_in_seconds=time_limit - (time.time() - start_time)`
line) is the reference.

```python
start_time = time.time()
...
remaining = (time_limit - (time.time() - start_time)) * 0.95 if time_limit is not None else None
```

### 4. `random_state` — set `seed_name`, don't hardcode the seed

Set the class attribute `seed_name = "random_state"` (or whatever the library's seed kwarg is).
AutoGluon then injects the *framework* seed there so every model uses the same seeding strategy.
**Do not** hardcode `random_state=0` in `_set_default_params()`.

### 5. Leave *global* torch state unchanged — snapshot & restore in a `finally`

The model fit-test (`autogluon.core.testing.global_context_snapshot.GlobalContextSnapshot`) asserts
that `_fit` does **not** leak changes into global torch state — it guards
`torch.get_num_threads()`, `torch.backends.cudnn.{benchmark,deterministic,enabled}`, the TF32 flags,
and the default dtype. Many libraries mutate these as a side effect: `torch.set_num_threads(...)`,
or seeding helpers that set `torch.backends.cudnn.deterministic = True` (LightAutoML's
`seed_everything` does exactly this). If your wrapper (or the lib it calls) touches any guarded
field, snapshot it before fitting and restore it in a `finally`:

```python
original_num_threads = torch.get_num_threads()
original_cudnn_deterministic = torch.backends.cudnn.deterministic
try:
    ...  # build + fit the inner model (may call set_num_threads / seed_everything)
finally:
    torch.set_num_threads(original_num_threads)
    torch.backends.cudnn.deterministic = original_cudnn_deterministic
```

`models/denselight/model.py` is the reference. Symptom if you forget: the smoke test fails with
`AssertionError: Global context changed across operation: - torch_cudnn_deterministic changed`.

---

## Categorical & missing-value handling — prefer the library's native path

A frequent review finding: wrappers needlessly label-encode categoricals, impute with `fillna(0)`,
and cast to a NumPy object array — all of which **destroy signal** when the library has native
categorical/missing handling (CatBoost-style models, EBM, RealMLP, etc.).

Decision order:

1. **Does the library accept a DataFrame and handle categoricals/NaN natively?** Then pass the
   frame through unchanged. Read the categorical columns from the dtypes (AutoGluon keeps them as
   `category` when `valid_raw_types` allows) and pass their names — don't re-encode:
   ```python
   def _preprocess(self, X, is_train=False, **kwargs):
       X = super()._preprocess(X, **kwargs)
       if is_train:
           self._cat_col_names = list(X.select_dtypes(include="category").columns)
       return X
   # in _fit: model.fit(X, y, cat_features=self._cat_col_names or None, ...)
   ```
   Let the library route NaN to its own missing bin — **do not** `fillna(0)` (0 collides with a real
   value and is not "missing").
2. **Only if the library needs purely numeric input** should you label-encode (e.g. via
   `LabelEncoderFeatureGenerator`) and impute — and then impute deliberately, not blindly with 0.

---

## Memory estimation — implement it for CPU models that fan out across folds

`can_estimate_memory_usage_static: False` with a `# TODO` is fine to *ship*, but for CPU models a
real estimate is what lets the scheduler safely fit cross-validation folds in parallel — a big
usability win that reviewers will ask for. When you can estimate peak memory from
`(n_rows, n_features, n_classes, …)`, implement `_estimate_memory_usage` / a static
`_estimate_memory_usage_static` and flip the tag to `True`. Reference:
`autogluon/tabular/src/autogluon/tabular/models/ebm/ebm_model.py` (`_estimate_memory_usage_static`).

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

### Iterative / boosting models (n_estimators)

For gradient-boosting-style models with early stopping (and a provided val set — see the `_fit`
contract above):

- Set a **high** `n_estimators` cap in `_set_default_params()` — other boosting models use ~10000,
  not a few hundred. Early stopping picks the real count; the cap is just headroom.
- **Don't also put `n_estimators` in the HPO search space** once it's a fixed high cap in the
  defaults. Searching a budget that early stopping already controls only adds noise and duplicates a
  value that's pinned elsewhere.

---

## info.py template

Defines the per-model `MethodMetadata` + `ModelInfo`. `info.py` is the file `discover_models()` walks — populating it is what makes the model visible to the registry.

```python
from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.{ModelKey}.hpo import gen_{ModelKey}
from tabarena.models.{ModelKey}.model import {ClassName}Model


{ModelKey}_method_metadata = MethodMetadata(
    method="{ModelName}",                  # e.g. "TabSTAR"
    method_type="config",
    display_name="{ModelName}",
    compute="gpu",                          # or "cpu"
    date="YYYY-MM-DD",                     # date of the benchmarking run (or planning date if unbenchmarked)
    ag_key="{ag_key}",                     # MUST equal {ClassName}Model.ag_key EXACTLY, incl. any "TA-" prefix (e.g. "TA-DENSELIGHT")
    model_key="{MODEL_KEY_UPPER}",         # short upper-case key (e.g. "DENSELIGHT"), commonly ag_key without the "TA-" prefix
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

## Test config (no per-model test file)

Models are fit-tested by the single registry-driven `tests/tabarena/models/test_all_models.py`,
which parametrizes over `get_model_registry()` and calls `FitHelper.verify_model(...)` per model.
A new model is picked up automatically once its `info.py` is discoverable — **do not write a
`test_{ModelKey}.py`**.

Only add a speed-up override to `tests/tabarena/models/smoke_configs.py` if the default
(empty hyperparameters, all problem types) is too slow or unsupported. Key by the model's
`MethodMetadata.method` (the registry key):

```python
# tests/tabarena/models/smoke_configs.py -> SMOKE_OVERRIDES
SMOKE_OVERRIDES: dict[str, ModelSmokeTest] = {
    ...
    "{ModelName}": ModelSmokeTest({"max_epochs": 1}),          # faster toy fit
    # or, for a regression-only model:
    # "{ModelName}": ModelSmokeTest(problem_types=("regression",)),
}
```

GPU-only models (`MethodMetadata.compute == "gpu"`) are skipped automatically when no CUDA
device is available, and any model is skipped when its optional dependency isn't installed.

---

## Registry update snippets

### `packages/tabarena/src/tabarena/models/__init__.py` — lazy class entry

```python
# Add to _LAZY_CLASSES (keep alphabetised by class name):
_LAZY_CLASSES = {
    ...
    "{ClassName}Model": "tabarena.models.{ModelKey}.model",
    ...
}

# `__all__` is auto-derived from `_LAZY_CLASSES` + `_EAGER_EXPORTS` — do NOT edit it by hand.

# Add to the TYPE_CHECKING block (keep sorted):
if TYPE_CHECKING:
    ...
    from tabarena.models.{ModelKey}.model import {ClassName}Model
```

`utils.py` needs no edit: `get_configs_generator_from_name()` resolves the search space from the
auto-discovered `MODEL_REGISTRY`, so there is no `name_to_import_map` to update.

### `packages/tabarena/pyproject.toml`

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

If the wrapper needs supporting modules, organise them under a private subfolder of `packages/tabarena/src/tabarena/models/{ModelKey}/`:

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

If/when the model has been benchmarked and the results are ready to be registered in TabArena's downstream artifact-aggregation files, also import the metadata you defined in `info.py` into the dated batch file under `packages/tabarena/src/tabarena/nips2025_utils/artifacts/_tabarena_method_metadata_YYYY_MM_DD.py`. That step is for downstream artifact handling only — it is not required for the model to work in the registry.

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
