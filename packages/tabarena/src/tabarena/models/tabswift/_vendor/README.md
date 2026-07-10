# Vendored TabSwift

Upstream: https://github.com/LAMDA-Tabular/TabSwift (MIT)
Vendored from commit `97cb58525355eb8cad9cf320f94b94ca04917ccb`, the
`TALENT/model/lib/tabswift` package.

TabSwift is a tabular in-context-learning foundation model (like TabPFN / TabICL). It
is not published as a pip-installable package — it only ships inside the TALENT
benchmark framework — so the inference-time sources are vendored here. The vendored
`tabswift` library is fully self-contained: it has **no** imports from `TALENT` and only
depends on packages already in TabArena's dependency tree (`torch`, `numpy`,
`scikit-learn`, `scipy`, `psutil`, `tqdm`, `huggingface_hub`, `packaging`).

See `LICENSE` for the upstream MIT license.

## What was vendored

| Upstream path (`TALENT/model/lib/tabswift/`) | Vendored path |
| --- | --- |
| `classifier.py`     | `classifier.py` |
| `regressor.py`      | `regressor.py` |
| `preprocessing.py`  | `preprocessing.py` |
| `__about__.py`      | `__about__.py` |
| `model/__init__.py` | `model/__init__.py` |
| `model/tabswift.py` | `model/tabswift.py` |
| `model/learning.py` | `model/learning.py` |
| `model/encoders.py` | `model/encoders.py` |
| `model/attention.py`| `model/attention.py` |
| `model/layers.py`   | `model/layers.py` |
| `model/inference.py`| `model/inference.py` |
| `LICENSE`           | `LICENSE` |

All internal imports upstream are **relative** (`from .preprocessing import ...`,
`from .model.tabswift import ...`, `from .layers import ...`), so they resolve unchanged
under the `tabarena.models.tabswift._vendor` namespace — no import rewriting was needed.

## What was NOT vendored

- `TALENT/model/methods/tabswift.py` — the TALENT `Method` adapter (depends on the rest of
  TALENT). Its intended workflow (NaN handling, ordinal categorical encoding, target
  standardization for regression, and the `n_estimators=16 / norm_methods=["none","power"]`
  defaults) is reproduced in the TabArena wrapper (`../model.py`) instead.
- `model/__pycache__` and a stray `.lnk` shortcut file present in the upstream tree.

## Modifications to the vendored source

Every change is also marked with a `# NOTE (tabarena vendor):` comment at the site.

### 1. Corrected the checkpoint download target — `classifier.py`, `regressor.py`

`_load_model` upstream auto-downloads from the private/broken repo
`pretrain-models/tabswift` (filenames `tabswift-classifier.ckpt` /
`tabswift-regressor.ckpt`). The public checkpoint is a **single** `swift.ckpt` hosted at
[`LAMDA-Tabular/TabSwift`](https://huggingface.co/LAMDA-Tabular/TabSwift) and shared by
both the classifier and the regressor (per the upstream README). The default `repo_id` /
`filename` were retargeted accordingly. In practice the TabArena wrapper always passes an
explicit `model_path` (the prefetched `swift.ckpt`), so this download fallback is rarely
exercised — but the default is now correct.

### 2. Removed stray debug prints — `classifier.py`, `regressor.py`

Both `predict_proba` / `predict` ended with an unconditional `print(avg.shape)` that fired
on every prediction call. Removed (all other prints in the tree are `self.verbose`-gated
or only run on the DataFrame input path, which the wrapper never uses — it passes NumPy).

### 3. `_vendor/__init__.py`

Upstream `__init__.py` is `from .classifier import TabSwiftClassifier`, which would import
torch at package-import time. Replaced with a docstring-only module so importing the
`_vendor` package stays lazy; the wrapper imports `classifier` / `regressor` submodules
directly inside `_fit`.

### 4. scikit-learn ≥ 1.6 `validate_data` compat — `classifier.py`, `regressor.py`, `preprocessing.py`

sklearn 1.6 removed the `BaseEstimator._validate_data` **method** in favor of the
module-level `sklearn.utils.validation.validate_data(estimator, ...)` **function** (fully
gone by 1.7). Upstream called `self._validate_data(...)` directly (and its `OLD_SKLEARN`
branches were inverted — the "old" branch passed a `cast_to_ndarray` kwarg that neither the
method nor the function accepts), so every fit/predict raised
`AttributeError: 'TabSwift...' object has no attribute '_validate_data'` on modern sklearn.

Fix: import `validate_data as _validate_data_func` (guarded by `try/except ImportError` for
sklearn < 1.6) and route the new-sklearn path through it. In `classifier.py` / `regressor.py`
the two `if OLD_SKLEARN` blocks now call `self._validate_data(...)` only on old sklearn and
`_validate_data_func(self, ...)` otherwise (and the bogus `cast_to_ndarray` kwarg was
dropped). In `preprocessing.py` a small module-level `_validate_data(estimator, *args,
**kwargs)` shim dispatches the same way, and all 10 `self._validate_data(...)` transformer
calls were rewritten to `_validate_data(self, ...)`. TabArena always feeds these estimators a
dense NumPy array, so no DataFrame-preservation behavior is lost.
