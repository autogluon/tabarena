"""ValSelect — a two-member validation-selection system over public tabular foundation models.

The system fits LimiX-2-1016 and Kumo-Tabular (large) on an internal holdout split of the training
data, scores both on the held-out rows with the task's metric, and refits the winner on the full
training data. Selection is per split and uses only training data: test rows never influence the
choice, and every prediction is inductive (each member predicts query rows from its stored context
alone). The members run through TabArena's own model wrappers, so their preprocessing and inference
match the leaderboard entries bit-for-bit; the system layer adds only the validation carve, the
selection, and the refit.

Rationale (offline simulation on the official 118-method cache, bencheval Elo): LimiX dominates
binary classification while Kumo wins several regression tables; per-dataset validation selection
between just these two public-weight models scored Elo ~2072 (#1, +62 over LimiX alone), with
15 dataset wins / 34 ties / 2 losses. Small shortlists keep selection noise low.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd

from tabarena.benchmark.exec_models import ExternalSystemModel
from tabarena.models.kumo_tabular.model import KumoTabularModel
from tabarena.models.limix_2.model import LIMIX2_MAX_CELLS, LimiX2Model

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer

    from tabarena.benchmark.task.metadata import ValidationMetadata


def _log(msg: str) -> None:
    print(f"[ValSelect {time.strftime('%H:%M:%S')}] {msg}", flush=True)


#: LimiX context cap applied on GPUs with less than this much VRAM (a 24 GB card cannot hold the
#: encoded 32-member ensemble above ~2M cells; the official 10M cap assumes larger devices).
LOW_VRAM_CUTOFF_GIB = 32.0
LOW_VRAM_LIMIX_MAX_CELLS = 2_000_000


def _cap_for_vram_gib(total_gib: float | None) -> int:
    """Cell cap for a device with ``total_gib`` GiB of VRAM (None = no CUDA device detected).

    Small cards get the 2M cap that keeps LimiX's autobatch on its fast inplace path; everything
    else (large-VRAM GPUs, or CPU-only where LimiX would run offline anyway) keeps the wrapper's
    official cap, so a re-run on benchmark hardware matches the official LimiX-2 entry's context.
    """
    if total_gib is not None and total_gib < LOW_VRAM_CUTOFF_GIB:
        return LOW_VRAM_LIMIX_MAX_CELLS
    return LIMIX2_MAX_CELLS


def _device_vram_gib() -> float | None:
    """VRAM of the current CUDA device in GiB, or None when no CUDA device is available."""
    import torch

    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2**30
    except Exception:  # a driver quirk must not break the fit; fall back to the official cap
        return None


class ValSelectSystemModel(ExternalSystemModel):
    """Validation selection between LimiX-2-1016 and Kumo-Tabular (large).

    Init hyperparameters:

    * ``val_frac`` — fraction of the training rows held out for the selection score (default 0.2,
      stratified on the label for classification when every class has at least two rows).
    * ``val_repeats`` — number of independent holdout splits whose per-candidate errors are averaged
      before the selection (default 1). More repeats trade fit time for a less noisy choice.
    * ``limix_max_cells`` — cell cap for LimiX's context in both fits (default None = auto).
      The wrapper's official 10M cap assumes more VRAM than a 24 GB card: the 32-member ensemble's
      encoded context fills it, and LimiX's adaptive autobatch then degrades to tiny batches with
      an O(n^2) chunk-concat (hours per fit). At 2M cells the fast inplace path stays available.
      Auto mode reads the fitting device at fit time and applies 2M below 32 GiB, the official cap
      otherwise, so a re-run on large-VRAM hardware matches the official entries' context. Pass an
      integer to pin the cap (a declared configuration choice, applied identically to val and refit;
      tables below the cap are untouched).
    """

    warmup_modules: ClassVar[tuple[str, ...]] = (
        "limix",
        "inference.v2_0.predictor",
        "sdm",
        "sdm.models.kumo.tabular",
    )
    warmup_torch_device: ClassVar[bool] = True
    # The system keeps its holdout predictions so the runner records the simulation artifact
    # (validation slot from get_oof, timed test predictions added by the runner) — the raw
    # predictions a submission is verified against.
    can_get_oof: ClassVar[bool] = True

    def __init__(self, *, val_frac: float = 0.2, val_repeats: int = 1, limix_max_cells: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.val_frac = val_frac
        self.val_repeats = val_repeats
        self.limix_max_cells = limix_max_cells  # None = resolve per fit from the device's VRAM
        self._limix_cap = limix_max_cells if limix_max_cells is not None else LOW_VRAM_LIMIX_MAX_CELLS
        self._limix_cap_source = "config" if limix_max_cells is not None else "unset"
        self._winner = None
        self._winner_name: str | None = None
        self._classes: np.ndarray | None = None
        self._problem_type: str | None = None
        self._target_name: str | None = None
        self._selection: dict = {}
        self._oof: dict = {}

    def _candidates(self) -> list[tuple[str, type, dict, dict]]:
        """(name, class, refit hyperparameters, extra validation-fit hyperparameters).

        LimiX keeps its packaged 32-member preprocessing ensemble (the default config) with the
        system's own cell cap (see ``limix_max_cells``): the ensemble's GPU footprint (transformed
        context copies plus the encoder's activations) grows with the cell count, and above ~2M
        cells on a 24 GB card free VRAM drops so low that LimiX's adaptive autobatch degrades to
        tiny batches with an O(n^2) chunk-concat — hours per fit instead of minutes. Kumo runs
        without the context KV cache: each fit only stores the context rows and each predict is one
        stateless forward, which is all a single-fit member needs (predictions match the cached
        path up to floating-point rounding).
        """
        return [
            ("LimiX-2-1016", LimiX2Model, {"max_cells": self._limix_cap}, {}),
            ("Kumo-Tabular", KumoTabularModel, {"cache_context": False}, {}),
        ]

    def _fit_system(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        target_name: str,
        problem_type: str,
        eval_metric: Scorer,
        validation_metadata: ValidationMetadata,
        num_cpus: int | None,
        num_gpus: int | None,
        memory_limit: float | None,
        time_limit: float | None,
        random_state: int | None,
    ):
        import torch
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        seed = random_state if random_state is not None else 0
        if self.limix_max_cells is None:
            vram_gib = _device_vram_gib()
            self._limix_cap = _cap_for_vram_gib(vram_gib)
            self._limix_cap_source = f"auto(vram_gib={None if vram_gib is None else round(vram_gib, 1)})"
        else:
            self._limix_cap = self.limix_max_cells
            self._limix_cap_source = "config"
        _log(f"limix_max_cells={self._limix_cap} [{self._limix_cap_source}]")
        self._problem_type = problem_type
        self._target_name = target_name
        classification = problem_type in ("binary", "multiclass")
        if classification:
            encoder = LabelEncoder()
            y_enc = pd.Series(encoder.fit_transform(y), index=y.index, name=y.name)
            self._classes = encoder.classes_
        else:
            y_enc = y
            self._classes = None

        # Internal holdout(s) for the selection score; the members never see these rows in fit.
        # Several repeats average out the split noise of a single holdout.
        cuda_available = torch.cuda.is_available()
        fit_gpus = num_gpus if num_gpus is not None else int(cuda_available)
        fit_cpus = num_cpus if num_cpus is not None else 8
        _log(
            f"fit start: problem_type={problem_type} X={X.shape} "
            f"val_frac={self.val_frac} repeats={self.val_repeats} seed={seed} gpus={fit_gpus}",
        )

        # Score each candidate on the holdout(s), one at a time so only one network is resident.
        # The last holdout's predictions are kept for the simulation artifact (get_oof); with
        # several repeats the splits overlap, so only the final one can be stored row-consistently.
        val_scores: dict[str, list[float]] = {name: [] for name, _, _, _ in self._candidates()}
        val_preds: dict[str, pd.Series | pd.DataFrame] = {}
        y_val: pd.Series | None = None
        for repeat in range(self.val_repeats):
            stratify = y_enc if classification and y_enc.value_counts().min() >= 2 else None
            X_fit, X_val, y_fit, y_val = train_test_split(
                X, y_enc, test_size=self.val_frac, random_state=seed + repeat, stratify=stratify
            )
            for name, cls, hps, val_hps in self._candidates():
                start = time.monotonic()
                model = None
                try:
                    model = cls(
                        name=f"ValSelect_{name}",
                        problem_type=problem_type,
                        eval_metric=eval_metric,
                        hyperparameters={**hps, **val_hps},
                    )
                    model.fit(X=X_fit, y=y_fit, num_cpus=fit_cpus, num_gpus=fit_gpus, random_seed=seed, verbosity=0)
                    preds = model.predict(X_val) if problem_type == "regression" else model.predict_proba(X_val)
                    preds = self._as_pandas_val(preds, X_val.index, problem_type)
                    err = self._score_val(preds, y_val, eval_metric)
                    val_preds[name] = preds
                except Exception as e:  # a member that cannot fit/score this table must not kill the run
                    _log(f"repeat {repeat}: {name} failed ({type(e).__name__}: {e}); assigning inf")
                    err = float("inf")
                    val_preds.pop(name, None)
                finally:
                    del model
                val_scores[name].append(err)
                _log(f"repeat {repeat}: {name} val_error={err:.6f} ({time.monotonic() - start:.1f}s)")
                if cuda_available:
                    torch.cuda.empty_cache()

        val_errors = {name: float(np.mean(errs)) for name, errs in val_scores.items()}
        winner_name = min(val_errors, key=val_errors.get)
        self._selection = {
            "val_frac": self.val_frac,
            "val_repeats": self.val_repeats,
            "limix_max_cells": self.limix_max_cells,
            "limix_max_cells_used": self._limix_cap,
            "limix_max_cells_source": self._limix_cap_source,
            "val_errors": val_errors,
            "val_errors_per_repeat": val_scores,
            "winner": winner_name,
            "n_train": len(X),
        }
        self._oof = {
            "pred_proba_dict_val": val_preds.get(winner_name, self._empty_val_pred(problem_type)),
            "y_val": y_val if winner_name in val_preds else y_val.iloc[:0],
            "eval_metric": eval_metric.name,
            "problem_type": problem_type,
        }
        _log(f"selected {winner_name}; refitting on all {len(X)} training rows")

        # Refit the winner on the full training data (the serving model, like the members' refit_full).
        cls, hps = next((c, h) for n, c, h, _ in self._candidates() if n == winner_name)
        self._winner_name = winner_name
        self._winner = cls(
            name=f"ValSelect_{winner_name}_refit",
            problem_type=problem_type,
            eval_metric=eval_metric,
            hyperparameters=dict(hps),
        )
        self._winner.fit(X=X, y=y_enc, num_cpus=fit_cpus, num_gpus=fit_gpus, random_seed=seed, verbosity=0)
        _log("refit done")
        return self

    @staticmethod
    def _as_pandas_val(preds, index: pd.Index, problem_type: str) -> pd.Series | pd.DataFrame:
        """Wrap raw member predictions as pandas in the internal label space (the artifact format:
        binary = positive-class series, multiclass = 0..k-1 columns, regression = value series).
        """
        arr = np.asarray(preds)
        if problem_type == "multiclass":
            return pd.DataFrame(arr, index=index, columns=list(range(arr.shape[1])))
        return pd.Series(arr.reshape(-1), index=index)

    def _empty_val_pred(self, problem_type: str) -> pd.Series | pd.DataFrame:
        """Zero-row validation slot for the degenerate case where no member scored the holdout."""
        if problem_type == "multiclass":
            k = len(self._classes) if self._classes is not None else 0
            return pd.DataFrame(np.empty((0, k), dtype=np.float32), columns=list(range(k)))
        return pd.Series(np.empty(0, dtype=np.float32))

    def _score_val(self, preds, y_val: pd.Series, eval_metric: Scorer) -> float:
        """Task-metric error of holdout predictions (lower is better).

        AutoGluon's error scorers (log_loss, rmse) already return the negated error, while ``roc_auc``
        returns the AUC itself, whose TabArena error is ``1 - AUC``. Classification labels are the
        system's 0..k-1 encoding, which lines up with the wrappers' probability columns; binary
        probabilities are the positive-class vector.
        """
        try:
            score = float(eval_metric(y_val, preds))
            return 1.0 - score if eval_metric.name == "roc_auc" else -score
        except Exception as e:  # a degenerate holdout (one class present) must not kill the fit
            _log(f"val scoring failed ({type(e).__name__}: {e}); assigning inf")
            return float("inf")

    def get_oof(self) -> dict:
        """The winner's holdout predictions in the system's internal label space.

        The runner adds the timed test predictions (label-cleaned) and wraps both slots by method;
        binary holdout probabilities are the positive-class vector, multiclass columns are 0..k-1,
        matching the wrappers' internal space.
        """
        return dict(self._oof)

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        preds = self._winner.predict(X)
        if self._problem_type != "regression":
            preds = self._classes[np.asarray(preds, dtype=int)]
        return pd.Series(preds, index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._problem_type == "regression":
            preds = self._winner.predict(X)
            return pd.DataFrame({self._target_name: np.asarray(preds)}, index=X.index)
        proba = np.asarray(self._winner.predict_proba(X))
        if self._problem_type == "binary":
            proba = np.column_stack([1.0 - proba, proba])
        return pd.DataFrame(proba, index=X.index, columns=self._classes)

    def get_metadata(self) -> dict:
        # The standard config-metadata fields (as the AutoGluon exec model's get_metadata_init):
        # results with simulation artifacts are read as ConfigResult, whose method inference needs
        # model_type/name_prefix/model_hyperparameters. num_gpus marks the run as GPU compute.
        return {
            "model_cls": type(self).__name__,
            "model_type": "ValSelect",
            "ag_key": "ValSelect",
            "name_prefix": "ValSelect",
            "model_hyperparameters": {
                "val_frac": self.val_frac,
                "val_repeats": self.val_repeats,
                "limix_max_cells": self.limix_max_cells,
            },
            "num_gpus": 1,
            "val_select": self._selection,
        }

    def cleanup(self):
        import torch

        self._winner = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
