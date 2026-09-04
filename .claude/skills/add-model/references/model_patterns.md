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

    # --- AutoGluon 1.6 declarative config: attributes, not method overrides ---
    _supported_problem_types = ["binary", "multiclass", "regression"]
    # GPU models only: count physical cores, take one CUDA GPU, and require a whole GPU
    # per fit. Drop all three for a CPU model (0 is the inherited default).
    default_resources_physical_cores_only = True
    default_num_gpus = 1
    minimum_num_gpus = 1
    # sequential_local avoids crashes when weights are not pre-downloaded and folds fit in
    # parallel. Foundation / pre-trained models ALSO set refit_folds (see the note below);
    # from-scratch NNs (TabM, RealMLP) omit it.
    _default_ag_args_ensemble_extra = {
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": True,
    }

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

    def get_device(self) -> str:
        return self.model.device

    def _set_device(self, device: str):
        self.model.to(device)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
```

Note what is *not* in that body. Problem types, resources, minimum resources and ensemble args
are declared as class attributes at the top of the class (see the next section), and the
memory-estimate capability is derived rather than declared.

> **Foundation models: set `refit_folds=True` in `_default_ag_args_ensemble_extra`.** Every
> pre-trained / in-context-learning wrapper (TabPFN, TabICL, LimiX, TabDPT, SAP-RPT-OSS,
> OrionMSP, TabSwift, ...) sets `refit_folds=True` *alongside*
> `fold_fitting_strategy: "sequential_local"`. A TFM has no train loop, so after bagging it
> refits a single model on all the data — much faster to score, at parity with the bagged
> ensemble. **Do not ship a TFM wrapper with only `sequential_local`** (a recurring miss).
> From-scratch NNs (TabM, RealMLP) intentionally omit it and set `can_refit_full=False`.

### Declare config as class attributes (AutoGluon 1.6)

AutoGluon 1.6 replaced a set of override methods with class attributes. Declare the attribute;
do not override the method. Only `_supported_problem_types` is enforced (AutoGluon's
`FitHelper.verify_model` raises on the old override, so `pytest -m models -k <Model>` fails),
but the whole table is the current convention and a new wrapper should follow all of it.

| Do not override | Declare instead |
|---|---|
| `supported_problem_types()` | `_supported_problem_types = [...]` |
| `_get_default_auxiliary_params()` | `_default_auxiliary_params_extra = {...}` |
| `_get_default_ag_args_ensemble()` | `_default_ag_args_ensemble_extra = {...}` |
| `_get_default_resources()` | `default_resources_physical_cores_only` + `default_num_gpus` |
| `get_minimum_resources()` | `minimum_num_gpus` (+ `gpu_required` if the model cannot run on CPU) |

The two `_extra` dicts are merged base-most class first, so a subclass wins over its parent.
That covers the common `super()` + `.update({...})` shape; keep the method only when the body
genuinely needs the parent's resolved value (for example
`refit_folds=parent.pop("refit_folds", True)`) or branches on state. Overriding still works at
runtime for every row except the first, so an inherited wrapper you have not converted is not
broken, just old.

`_default_auxiliary_params_extra` gains a typo guard the override never had: `verify_model`
checks every declared key against the known auxiliary params and fails on an unknown one. A
misspelled key in an overridden `_get_default_auxiliary_params` is silently ignored instead.

**Memory estimation is derived, not declared.** Do not write
`_class_tags() -> {"can_estimate_memory_usage_static": ...}`; AutoGluon reads whether the class
implements `_estimate_memory_usage_static`. And do not write an `_estimate_memory_usage` that
just forwards to the static estimate — that is the base-class default. So the whole memory story
for a new model is: implement `_estimate_memory_usage_static` (and it is on), or don't (and it is
off). Keep `_class_tags` only for other tags, e.g. TabM's `reset_torch_threads`.

**Never mutate `self.params` or `self.params_aux` after construction.** They are resolved
configuration; mutation warns in AutoGluon 1.6 and raises in 1.7. If `_fit` computes a value that
a later call needs, store it on the instance and override the getter. The
`TabPFNv26Model._max_batch_size_resolved` + `_get_max_batch_size()` pair in
`models/tabpfnv2_5/model.py` is the in-repo example; AutoGluon's own pattern references are
`AbstractModel.temperature_scalar` and `AbstractModel._get_max_batch_size`.

### Choosing `AbstractTorchModel` vs `AbstractModel`

`AbstractTorchModel` exists **only** to provide torch device management — `get_device()` /
`_set_device()` are abstract, and its load path calls `torch.cuda.is_available()` / builds a
`torch.device(...)` (so a non-torch device string like `"gpu"` crashes it). Pick the base class by
*framework*, not by whether the model uses a GPU:

- **Torch model (GPU or CPU NN):** `AbstractTorchModel`. Implement `get_device()` + `_set_device()`.
- **Non-torch model that still runs on GPU** (JAX/Flax like TabFM, or any library that places itself
  on the device via `CUDA_VISIBLE_DEVICES` / `jax.devices()`): use **`AbstractModel`**, add the GPU
  *resource* methods (below), and do **not** implement `get_device`/`_set_device`. `tabstar/model.py`
  is a GPU foundation model on `AbstractModel`; `tabfm/model.py` is the JAX example.
- **Pure CPU / sklearn-like:** `AbstractModel`.

### GPU model on `AbstractModel` (non-torch) variant

A GPU model that is **not** torch-based inherits `AbstractModel` and keeps the GPU *resource* methods
but omits `get_device`/`_set_device`. The compute device is selected by the library / the process
environment (e.g. `jax.devices()` honoring `CUDA_VISIBLE_DEVICES`), not by AutoGluon:

```python
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel

class {ClassName}Model(AbstractModel):
    ag_key = "TA-{MODEL_KEY_UPPER}"
    ag_name = "TA-{ModelName}"
    ag_priority = 65
    seed_name = "random_state"

    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_resources_physical_cores_only = True
    default_num_gpus = 1
    minimum_num_gpus = 1
    # refit_folds=True for foundation models (see note above); drop it for from-scratch NNs.
    _default_ag_args_ensemble_extra = {
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": True,
    }

    def _fit(self, X, y, num_cpus=1, num_gpus=0, **kwargs):
        # Validate GPU availability against the actual backend (e.g. jax), not torch.
        # Load the (pre-trained) model, build the sklearn-style wrapper, fit.
        ...

    def _more_tags(self): return {"can_refit_full": True}
    # NOTE: no get_device / _set_device — those are AbstractTorchModel-only.
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
    _supported_problem_types = ["binary", "multiclass", "regression"]
    # CPU model: no GPU attributes. Set this only if the library benchmarks better on
    # physical cores (most GBDTs and NNs do); leave it off to count logical cores.
    default_resources_physical_cores_only = True

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

## Warm-up classmethod (untimed environment warm-up)

TabArena runs an untimed warm-up before the timed fit (`tabarena.models.warmup.warmup_model_cls`
dispatches it), so one-time per-environment costs (heavy imports, JIT/kernel compilation, CUDA
context) don't inflate the measured fit/inference times. `AbstractTorchModel` subclasses are
covered automatically (generic torch + CUDA warm-up) — declare a classmethod only when the model
is torch-backed on plain `AbstractModel`, has a heavy extra import, or its library pre-compiles
kernels. Warm-up must be data-independent (never touch task data), and it only warms the main job
process + disk-backed caches — parallel-fold (Ray) workers are fresh processes, so prefer library
warm-ups whose compile cache persists to disk.

```python
# Torch-backed model on AbstractModel (generic fallback doesn't reach these).
# References: modernnca, xrfm, tabstar.
@classmethod
def warmup(cls, *, num_gpus: float | None = None, **kwargs) -> None:
    """Warm torch (+ CUDA context) and the library import (untimed, data-independent)."""
    from tabarena.models.warmup import warmup_imports, warmup_torch

    warmup_torch(cuda=None if num_gpus is None else num_gpus > 0)
    warmup_imports("somelib.model")  # only if the import is heavy (e.g. pulls transformers)
```

```python
# Library with its own kernel pre-compilation (numba / JAX / custom kernels).
# Reference: chimeraboost (warmup() exists from chimeraboost>=0.14.1 — pin the pip extra
# accordingly). Ask the user for the entry point + minimum version if the docs don't say.
@classmethod
def warmup(cls, **kwargs) -> None:
    """Pre-compile the library's kernels (disk-cached per environment)."""
    import somelib

    somelib.warmup()
```

The dispatch always passes `problem_type` / `num_cpus` / `num_gpus` / `hyperparameters` as
keyword arguments — declare the ones you read, keep `**kwargs` for the rest.

Inference side: the exec model persists the fitted model in memory around the inference timer by
default (`AGWrapper.persist`, memory-guarded), and calls an optional **instance** method
`prepare_for_inference(self) -> None` on every persisted model object (incl. bagged children) —
untimed, for model-only prep like moving weights offloaded at the end of `_fit` back to the
inference device; never touch test data there. Outer/direct fits (`AGModelWrapper`) dispatch the
same hook on their in-memory model, so declaring it covers every fit path. Avoid deferring other
one-time work to the first `_predict` (put it in `_fit` or `warmup`).

---

## Foundation-model weights: always pin the HF checkpoint revision

Any foundation model whose weights come from Hugging Face (`hf_hub_download` /
`snapshot_download`) must pin `revision=` to a specific commit — never resolve against the repo's
current default branch. Without it, a push to the HF repo silently changes the weights every
subsequent fit uses, with nothing in this repo recording that it happened (see
autogluon/tabarena#510 for the incident that prompted this).

```python
_DEFAULT_HF_REPO = "SomeOrg/SomeModel"
_DEFAULT_HF_FILENAME = "checkpoint.safetensors"
#: Commit pinned so the checkpoint fetched here never silently changes if the
#: repo's default branch moves. Bump deliberately (with a note on what changed)
#: when picking up a newer checkpoint.
_DEFAULT_HF_REVISION = "<commit-sha-from-huggingface.co/api/models/{repo}>"


@classmethod
def prefetch_weights(cls) -> str:
    """Pre-download the checkpoint from Hugging Face and return its local path.

    Tries the local cache first so offline compute nodes skip the etag
    HEAD-request that ``hf_hub_download`` performs by default.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return hf_hub_download(
            repo_id=_DEFAULT_HF_REPO,
            filename=_DEFAULT_HF_FILENAME,
            revision=_DEFAULT_HF_REVISION,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        return hf_hub_download(
            repo_id=_DEFAULT_HF_REPO,
            filename=_DEFAULT_HF_FILENAME,
            revision=_DEFAULT_HF_REVISION,
        )
```

Resolve `_DEFAULT_HF_REVISION` from `https://huggingface.co/api/models/{repo_id}` (the `sha`
field) at the time you write the wrapper, not by hand-guessing a commit. Wire this same
classmethod into `ModelInfo(prefetch_weights=...)` in `info.py` so the foundation-model
pre-download scripts (see the `benchmark-model` skill) pick it up.

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

Shipping without an estimate is fine (leave `_estimate_memory_usage_static` unimplemented and add
a `# TODO`), but for CPU models a real estimate is what lets the scheduler safely fit
cross-validation folds in parallel — a big usability win that reviewers will ask for. When you can
estimate peak memory from `(n_rows, n_features, n_classes, …)`, implement the classmethod
`_estimate_memory_usage_static`. That single method is the whole opt-in: AutoGluon 1.6 derives
`can_estimate_memory_usage_static` from its presence and the base `_estimate_memory_usage` already
forwards to it, so there is no tag to flip and no instance wrapper to write. Reference:
`autogluon/tabular/src/autogluon/tabular/models/ebm/ebm_model.py` (`_estimate_memory_usage_static`).

GPU models have a parallel hook, `_estimate_gpu_memory_usage_static`, which enables VRAM safety
checks the same way. Without it AutoGluon budgets parallel folds against node RAM, which is why
benchmark runs pass `fake_memory_for_estimates` (see the `benchmark-model` skill).

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


{ModelKey}_method_metadata = MethodMetadata.config(
    method="{ModelName}",                  # e.g. "TabSTAR"
    display_name="{ModelName}",
    compute="gpu",                          # or "cpu"
    date="YYYY-MM-DD",                     # date of the benchmarking run (or planning date if unbenchmarked)
    ag_key="{ag_key}",                     # MUST equal {ClassName}Model.ag_key EXACTLY, incl. any "TA-" prefix (e.g. "TA-DENSELIGHT")
    model_key="{MODEL_KEY_UPPER}",         # short upper-case key (e.g. "DENSELIGHT"), commonly ag_key without the "TA-" prefix. This value is the model's config_type — add it to get_model_family() (SKILL Step 4d) or the leaderboard shows ❓ Other
    config_default="{ModelName}_c1_BAG_L1",
    can_hpo=True,
    is_bag=True,
    has_raw=True,
    has_processed=True,
    has_results=True,
    suite="tabarena-YYYY-MM-DD",
    # Storage backend is inferred from the cache location in cache_kwargs: once results are hosted
    # in the official pool, set cache_kwargs={"bucket": ..., "prefix": ...} and cache_type infers
    # "r2" (the default public backend). Omit cache_kwargs until hosted -> cache_type infers
    # "local". (s3/r2 require bucket+prefix in cache_kwargs; "local" forbids them — construction
    # raises on a mismatch.)
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
    verified=False,                         # flip to True once benchmark run is signed off
    reference_url="{doc_url}",
)


{ModelKey}_info = ModelInfo(
    model_cls={ClassName}Model,
    search_space=gen_{ModelKey},
    method_metadata={ModelKey}_method_metadata,
    pip_extra=("{pip_package}",),
)
```

`pip_extra` is the tuple of pip specs the auto-discovery uses when computing what extras to install for this model — list every dependency the wrapper imports lazily.

**Storage conventions:**
- Use `MethodMetadata.config(...)` (the config-method constructor — it sets `method_type="config"` and exposes the config-only fields `ag_key` / `model_key` / `config_default` / `name_suffix` / `can_hpo` / `is_bag`). Baseline/portfolio methods use `MethodMetadata.baseline(...)` / `MethodMetadata.portfolio(...)`.
- **Don't set `cache_type` by hand** — it's inferred: `"r2"` when `cache_kwargs` carries a remote location (`{"bucket": ..., "prefix": ...}`), else `"local"`. Set it explicitly only to force `"s3"`.
- **All cache config lives in `cache_kwargs`** (a dict), not top-level fields — so a casual (local) model isn't shown remote-storage knobs and future backends can add their own keys. The remote location is `cache_kwargs={"bucket": ..., "prefix": ...}`; for s3 add `"upload_as_public": True` for a public-read ACL. A local (unhosted) model sets no `cache_kwargs`. (`upload_as_public`, `s3_bucket`/`s3_prefix`, and `bucket`/`prefix` are no longer top-level args.)
- For artifacts hosted in the **legacy public-S3 pool**, prefer the `MethodMetadata.tabarena_legacy_s3(...)` preset, which fixes `cache_type="s3"` and the public-read ACL (`cache_kwargs={"upload_as_public": True}`) for you.

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

If/when the model has been benchmarked and its results processed + uploaded, also import the metadata you defined in `info.py` into the arena collection in `packages/tabarena/src/tabarena/contexts/tabarena/methods.py` (add it to `tabarena_method_metadata_collection.method_metadata_lst`; see the `upload-method` skill). That step is for downstream artifact handling only — it is not required for the model to work in the registry.

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
_default_auxiliary_params_extra = {
    "max_rows": 100_000,
    "max_features": 2000,
}
```
AutoGluon 1.6 also offers `min_features` / `min_cells` / `max_cells`, and reports a constraint
miss as a skip rather than a failure. Keys are validated, so a typo fails `verify_model`
instead of being silently ignored.
