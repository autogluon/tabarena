# Vendored LimiX

Upstream: https://github.com/limix-ldm-ai/LimiX (Apache-2.0)

LimiX is not published as a pip-installable package. This directory vendors
the inference-time sources required by `LimiXModel`. Files were copied from
upstream `main` and adjusted as described below.

See `LICENSE.txt` for the upstream Apache-2.0 license.

## What was vendored

| Upstream path | Vendored path |
| --- | --- |
| `inference/inference_method.py` | `inference/inference_method.py` |
| `inference/predictor.py` | `inference/predictor.py` |
| `inference/preprocess.py` | `inference/preprocess.py` |
| `model/encoders.py` | `model/encoders.py` |
| `model/layer.py` | `model/layer.py` |
| `model/transformer.py` | `model/transformer.py` |
| `utils/data_utils.py` | `utils/data_utils.py` |
| `utils/inference_utils.py` | `utils/inference_utils.py` |
| `utils/loading.py` | `utils/loading.py` |
| `utils/retrieval_utils.py` | `utils/retrieval_utils.py` |
| `config/*.json` (7 files) | `config/*.json` |
| `LICENSE.txt` | `LICENSE.txt` |

`__init__.py` files were added to each Python package so the tree imports
cleanly under the new namespace.

## What was NOT vendored

- `retrieval_extension/` — only used by the optional hyperparameter-search
  loop, not by the predict path. Avoids an `optuna` runtime dependency.
- `utils/utils.py` — only contains dataset/checkpoint download helpers used
  by the upstream example scripts. The wrapper downloads the checkpoint
  directly via `huggingface_hub.hf_hub_download`.
- `inference_classifier.py` / `inference_regression.py` — example scripts at
  the repo root.

## Modifications to the vendored source

Every change below is also marked with a `# NOTE (tabarena vendor):` comment
in the source where applicable, to make diffs against upstream obvious.

### 1. Namespace rewrite (all Python files)

Upstream uses bare top-level package names (`inference.*`, `model.*`,
`utils.*`). To avoid clashing with anything else on `sys.path`, every
`from inference.X import …`, `from model.X import …`, and
`from utils.X import …` was rewritten to
`from tabarena.benchmark.models.ag.limix._vendor.{inference|model|utils}.X
import …`. No other code was changed by this pass.

### 2. Make fitted pipelines picklable — `inference/preprocess.py`

**Why:** AutoGluon pickles the fitted model after `_fit` returns
(`save_pkl.pickle_fn`). The fitted `LimiXPredictor` holds the sklearn
preprocessing pipelines built in `RebalanceFeatureDistribution._set` and the
top-level SVD branch, several of which wrap inline `lambda`s inside
`FunctionTransformer`. Pickling fails with
`AttributeError: Can't get local object 'RebalanceFeatureDistribution._set.<locals>.<lambda>'`.

**Change:** added four module-level helpers near the top of `preprocess.py`:

```python
def _identity(x): return x
def _nan_inf_to_nan(x): return np.nan_to_num(x, nan=np.nan, neginf=np.nan, posinf=np.nan)
def _shift_by_nanmin(x): return x + np.abs(np.nanmin(x))
def _add_epsilon(x): return x + 1e-10
```

Then replaced every lambda used inside a `FunctionTransformer` step in
`RebalanceFeatureDistribution._set` and the SVD branch with the matching
named helper. Concretely, the following lambdas were substituted:

| Lambda | Helper |
| --- | --- |
| `lambda x: np.nan_to_num(x, nan=np.nan, neginf=np.nan, posinf=np.nan)` | `_nan_inf_to_nan` |
| `lambda x: x` | `_identity` |
| `lambda x: x + np.abs(np.nanmin(x))` | `_shift_by_nanmin` |
| `lambda x: x + 1e-10` | `_add_epsilon` |

Eight separate `FunctionTransformer(...)` call sites are affected (the
`logNormal` Pipeline, the `power` SelectiveInversePipeline, the
`worker_tag is None` / fallback branches, and the SVD `default` /
`save_standard` pipelines).

The remaining `lambda` in the vendor — `lambda: EncoderBaseLayer(...)` in
`model/transformer.py` — is an unfitted layer-factory used only during
model construction, never stored in pickled state, so it is left as-is.

### 3. Single-feature `.squeeze()` bug — `inference/preprocess.py`

**Why:** `SubSampleData.fit` collapses `feature_attention_score[:, -1, :]`
with a bare `.squeeze()` before either a `.permute(1, 0)` or a
`torch.mean(..., dim=0)`. `[:, -1, :]` already returns a 2-D
`(test_sample_lens, n_features)` tensor; the extra `.squeeze()` then drops
the feature dimension whenever a dataset has exactly one feature,
producing a 1-D tensor and crashing the subsequent `.permute(1, 0)` /
shape-dependent mean. The squeeze appears to be a leftover from an earlier
shape contract.

**Change:** removed the trailing `.squeeze()` in both branches of
`SubSampleData.fit` (the `use_type == "mixed"` branch and the `else`
branch). The downstream ops already expect 2-D input, so removing the
squeeze is a no-op for the multi-feature case and fixes the one-feature
case.

## Known issue not patched: DDP path

`InferenceResultWithRetrieval.inference` in
`inference/inference_method.py` has an `else` branch (taken when the
inference config sets `retrieval_config.use_cluster = false`) that
unconditionally calls `setup()` → `dist.init_process_group("nccl")` and
wraps the model in `DistributedDataParallel`, then uses
`dist.all_gather_object` to collect outputs. There is no flag to disable
this path without rewriting it; even setting `inference_with_DDP=False` on
`LimiXPredictor` does not skip it. The only way to run "single-threaded"
is to enter via the `use_cluster=true` branch.

The default configs shipped here (`cls_default_16M_retrieval.json` and
`reg_default_16M_retrieval.json`) both set `use_cluster: true`, so the
default wrapper flow never hits the DDP code. The other shipped configs
(`*_noretrieval.json`, `*_2M_retrieval.json`, etc.) are unused by the
wrapper but should not be selected without first patching this path.

A proper fix needs a `world_size == 1` shortcut that skips
`init_process_group`, the `DDP(...)` wrap, the
`NonPaddingDistributedSampler`, and the `dist.all_gather_object` calls,
plus clean teardown of the process group. This is a TODO for a follow-up.
