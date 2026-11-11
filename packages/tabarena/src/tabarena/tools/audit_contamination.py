"""Audit a model for memorized TabArena test tables: does it predict the published test rows better than
the same rows with the columns in another order?

Usage::

    python -P -m tabarena.tools.audit_contamination --model NAME [--datasets NAME ...] [--checks CHECK ...]
        [--splits 0 | all | 0 1 2] [--random-orders K] [--confirm] [--config off | released]
        [--seed N] [--num-gpus N] [--json PATH]
    python -P -m tabarena.tools.audit_contamination --list-models

A model pretrained on real tables may have seen a benchmark's test rows with their labels. The TabArena
datasets are public, their copies on OpenML, Kaggle and UCI are many, and a dedup filter that compares names
and shapes misses a copy under another name (LimiX-2, autogluon/tabarena#648: an overlap filter missed
``diabetes`` and ``blood-transfusion-service-center``, and the rows entered training with a fixed feature
order). This tool fits a model from the registry on one split of each dataset, under several column orders,
and compares what the published order buys. The checks:

``order``
    The primary probe. The training and test columns are permuted together: published order, reversed, and
    ``--random-orders`` random permutations (seeded, recorded). Permuting the columns gives an honest model no
    information, so its error moves by noise only; a model that memorized the table in its published order
    loses the memory under every permutation. The statistic per permutation is the log ratio of the errors,
    ``d = log(err_permuted / err_published)`` (1 - AUC, log loss or RMSE, floored at ``ERR_EPS``); ``d_mean``
    averages the permutations and ``d_min`` is the weakest one. The published order's rank among all orders
    is reported as ``rank_p`` (1 / n_orders when it beats every permutation on every split): under the null
    that order carries no information, the rank is uniform. Honest models are not all order-invariant (an
    ensemble member can read a column position; a hand-chosen column order carries a little information),
    so the verdict is an across-dataset outlier rule on top of a fixed floor: ``d_mean`` is compared with
    the median and MAD of the same model over every dataset of the run, and a dataset is suspect when it
    stands ``ORDER_MAD_MULTIPLIER`` MADs above the median and above ``ORDER_LOG_RATIO_WARN``. FAIL needs the
    effect on every permutation (``d_min`` above ``ORDER_LOG_RATIO_FAIL``) and the confirmation stage:
    ``--confirm`` reruns every suspect dataset with ``ORDER_N_RANDOM_CONFIRM`` random orders on every split,
    and a FAIL needs ``rank_p`` at or below ``ORDER_RANK_P_CONFIRM`` there (or a positive ``d_min`` on every
    one of several splits). A ``d_min`` above ``ORDER_LOG_RATIO_FAIL`` is at least a WARN whatever the run's
    median, and a run whose median ``d_mean`` is itself above ``ORDER_LOG_RATIO_WARN`` cannot calibrate (the
    model prefers the published order on most of its datasets): it gets a run-level WARN row and the floors
    apply alone. Fewer than ``CALIBRATION_MIN_DATASETS`` datasets give no calibration: the probe then reports
    SKIP, unless ``--no-calibration`` applies the fixed floors alone.
``ensemble``
    One extra fit with the released configuration on the published order, on the datasets the ``order``
    probe flagged (WARN or FAIL), after the calibration. Stripping a model's test-time ensemble (the
    augmentation-off configuration below) makes an honest model equal or worse; a model that memorized the
    published table gets better, because the released members that reorder columns, rescale values or add
    features hide what it remembers (LimiX-2 on ``diabetes``: 1 - AUC 0.1249 as released, 0.0066 stripped).
    ``d = log(err_released / err_off)`` above ``ENSEMBLE_LOG_RATIO_WARN`` warns; the probe never fails a
    dataset on its own. A released fit is the expensive one (32 members for LimiX-2), which is why it is not
    run on every dataset; on 51 datasets TabPFN-3's released ensemble was better than its single member by
    a median d of -0.02 and never worse than +0.05.
``labels``, ``shuffled``, ``jitter``
    The behaviour probes of :mod:`tabarena.tools.audit_system`, run on the published order: relabeled
    classes, shuffled training labels, and 1% feature noise against a local reference model. They were
    written for a system that looks test labels up at prediction time; whether a pretrained memory shows
    the same way is not established, so their verdicts raise a dataset to WARN at most.

The probes run with the model's test-time augmentation turned off (``--config off``, the default): one
ensemble member, no feature shift, for the registry names in ``AUGMENTATION_OFF`` (LimiX-2: the members that
keep the column order, without their polynomial and rebalancing features, the recipe of #648). Other models
fall back to their ``cheap_hyperparameters``, which the report says. ``--config released`` runs every probe
with the registry default instead, which is what the leaderboard saw and where the ensemble hides most of
the signal.

The datasets are TabArena-v0.1 names (``all`` for every one of them), ``openml:<dataset id>[:<problem type>]``
for a public table that is not on the benchmark (a stratified 3 x 3 split is drawn), ``csv:<path>:<target>``
for a private table, and ``synthetic-<problem type>`` for a fresh table with no public copy. Test rows stay
the published rows; the context is capped at ``--max-train-rows``.

What the ``order`` probe detects is one mechanism: training on the table in a fixed column order. A model that
saw the table under column augmentation is contaminated and order-invariant, and a PASS says nothing about
it. The thresholds are set from the #648 figures (LimiX-2 diabetes d = 3.1, blood-transfusion 0.38; honest
models within 0.12 on 51 datasets) and will move as more models are audited. Verdicts: ``PASS``, ``WARN``
(read the details), ``FAIL`` (write it up for the model's authors), ``INFO`` and ``SKIP``. The report ends
with the ``leakage_datasets`` tuple of the failed datasets and the ``warn_datasets`` tuple, in the form
``ModelDescriptor`` takes; ``--json`` keeps every error, column order and configuration for a rerun. A fit
that raises is a ``probe`` row (the dataset keeps its order verdict when it has one). Exit code 1 when a
dataset failed or a fit raised, 2 on a usage error, 0 otherwise.

Run it with ``python -P`` from the repository root (see :mod:`tabarena.tools.audit_warmup`).
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import platform
import shutil
import sys
import tempfile
import time
import zlib
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from tabarena.tools.audit_system import (
    FAIL,
    INFO,
    PASS,
    SKIP,
    SYNTHETIC_PREFIX,
    WARN,
    CheckResult,
    ProbeData,
    _classes,
    _error,
    _proba_matrix,
    _subsample,
    _write_json,
    format_result,
    make_synthetic_split,
    probe_dataset,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

CHECKS = ("order", "ensemble", "labels", "shuffled", "jitter")
#: Probes delegated to ``audit_system.probe_dataset``; they raise a dataset to WARN at most.
SECONDARY_CHECKS = ("labels", "shuffled", "jitter")
#: Verdict row per dataset, built from the probes above.
VERDICT_CHECK = "verdict"

#: ``d_mean`` an honest model stays within: the largest move on 51 datasets x 9 splits for TabPFN-3.5 and
#: Kumo Tabular (polish bankruptcy, 0.0088 vs 0.0099, a few samples deciding) in the #648 write-up.
ORDER_LOG_RATIO_WARN = 0.12
#: ``d_min`` a FAIL needs: between the honest bound above and the weakest proven case (LimiX-2 on
#: blood-transfusion-service-center, 0.1774 vs 0.2596 over 30 splits, d = 0.38).
ORDER_LOG_RATIO_FAIL = 0.25
#: MADs above the model's own median ``d_mean`` (over the run's datasets) for a dataset to stand out.
ORDER_MAD_MULTIPLIER = 3.0
#: Floor on the MAD, so an order-invariant model does not turn float noise into outliers.
ORDER_MIN_MAD = 0.03
#: Random orders in the screen and in the confirmation rerun (plus published and reversed in both).
ORDER_N_RANDOM = 3
ORDER_N_RANDOM_CONFIRM = 9
#: ``rank_p`` of the published order a confirmed FAIL needs: 1 / 11 when it wins against every one of the
#: confirmation's 10 permutations on every split.
ORDER_RANK_P_CONFIRM = 0.1
#: Datasets a run needs before the across-dataset rule has a median and a MAD.
CALIBRATION_MIN_DATASETS = 5
#: Datasets of one problem type before its own calibration replaces the pooled one.
CALIBRATION_MIN_PER_TYPE = 5
#: ``log(err_released / err_off)`` above which stripping the ensemble helped too much.
ENSEMBLE_LOG_RATIO_WARN = 0.25
#: Error floor for the log ratios (1 - AUC and log loss); regression uses ``ERR_EPS_REGRESSION_STD`` standard
#: deviations of the test target.
ERR_EPS = 1e-3
ERR_EPS_REGRESSION_STD = 1e-6
DEFAULT_MAX_TRAIN_ROWS = 10_000
DEFAULT_MAX_TEST_ROWS = 5_000
THRESHOLDS = (
    "ORDER_LOG_RATIO_WARN",
    "ORDER_LOG_RATIO_FAIL",
    "ORDER_MAD_MULTIPLIER",
    "ORDER_MIN_MAD",
    "ORDER_N_RANDOM",
    "ORDER_N_RANDOM_CONFIRM",
    "ORDER_RANK_P_CONFIRM",
    "CALIBRATION_MIN_DATASETS",
    "CALIBRATION_MIN_PER_TYPE",
    "ENSEMBLE_LOG_RATIO_WARN",
    "ERR_EPS",
)
PUBLISHED, REVERSED = "published", "reversed"
OPENML_PREFIX, CSV_PREFIX = "openml:", "csv:"
#: Packages whose versions the JSON records.
VERSIONED_PACKAGES = (
    "tabarena",
    "autogluon.core",
    "torch",
    "tabpfn",
    "tabicl",
    "tabdpt",
    "scikit-learn",
    "numpy",
    "pandas",
)


# --------------------------------------------------------------------------------------------------
# The model under audit
# --------------------------------------------------------------------------------------------------
@dataclass
class FittedModel:
    """A fitted AutoGluon model with the label cleaner that maps its probabilities back to the classes."""

    model: Any
    label_cleaner: Any
    workdir: str

    def cleanup(self) -> None:
        """Drop the model and its directory, and give the GPU memory back (a second call is a no-op)."""
        if self.model is None:
            return
        self.model = None
        shutil.rmtree(self.workdir, ignore_errors=True)
        gc.collect()
        with contextlib.suppress(Exception):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


@dataclass
class ModelUnderAudit:
    """A registered model class plus the hyperparameters and resources every probe fits it with.

    Fits go through the model class directly (``model_cls(...).fit(X, y)``, the path of the #648 reproduction
    and of :mod:`tabarena.tools.audit_pickle`), so the columns reach the model in the order they are handed
    over. The signature of ``fit_predict`` and ``predict`` is the one ``audit_system.probe_dataset`` calls.
    """

    model_cls: type
    hyperparameters: dict = field(default_factory=dict)
    num_gpus: float = 1
    num_cpus: int | None = None
    name: str = "m"
    calls: list[dict] = field(default_factory=list)
    """``{"probe", "dataset", "rows", "wall_s"}`` per fit (plus ``"error"`` when it raised)."""
    fitted: list[FittedModel] = field(default_factory=list)
    """Every model this object fitted, so ``cleanup`` can drop the ones a delegated probe discarded."""

    def cleanup(self) -> None:
        """Drop every fitted model and its directory."""
        for fitted in self.fitted:
            fitted.cleanup()
        self.fitted.clear()

    @classmethod
    def from_registry(cls, method: str, *, hyperparameters: dict, num_gpus: float = 1, **kwargs) -> ModelUnderAudit:
        """The registered model ``method`` (``MethodMetadata.method``, e.g. ``"LimiX-2"``)."""
        return cls(resolve_model(method).model_cls, hyperparameters=dict(hyperparameters), num_gpus=num_gpus, **kwargs)

    def fit(self, data: ProbeData, X_train: pd.DataFrame, y_train: pd.Series, *, probe: str) -> FittedModel:
        """Fit one model on ``X_train`` / ``y_train`` (labels cleaned to what AutoGluon models take)."""
        from autogluon.core.data.label_cleaner import LabelCleaner
        from autogluon.core.metrics import get_metric

        cleaner = None
        if data.problem_type == "regression":
            y_fit = y_train.astype(float)
        else:
            cleaner = LabelCleaner.construct(
                problem_type=data.problem_type, y=y_train, y_uncleaned=pd.concat([y_train, data.y_test])
            )
            y_fit = cleaner.transform(y_train)
        # A multiclass split whose training part holds two classes gets a binary cleaner; the model must agree.
        problem_type = cleaner.problem_type_transform if cleaner is not None else data.problem_type
        workdir = tempfile.mkdtemp(prefix="tabarena_audit_contamination_")
        model = self.model_cls(
            path=workdir,
            name=self.name,
            problem_type=problem_type,
            eval_metric=get_metric(data.eval_metric, problem_type=problem_type),
            hyperparameters=dict(self.hyperparameters),
        )
        fit_kwargs: dict[str, Any] = {"num_gpus": self.num_gpus}
        if self.num_cpus is not None:
            fit_kwargs["num_cpus"] = self.num_cpus
        entry = {"probe": probe, "dataset": data.name, "rows": len(X_train)}
        start = time.monotonic()
        try:
            model.fit(X=X_train.copy(), y=y_fit, **fit_kwargs)
        except Exception as exc:
            shutil.rmtree(workdir, ignore_errors=True)
            self.calls.append(
                {**entry, "wall_s": time.monotonic() - start, "error": f"{type(exc).__name__}: {exc}"[:300]}
            )
            raise
        self.calls.append({**entry, "wall_s": time.monotonic() - start})
        fitted = FittedModel(model=model, label_cleaner=cleaner, workdir=workdir)
        self.fitted.append(fitted)
        return fitted

    def predict(self, fitted: FittedModel, data: ProbeData, X: pd.DataFrame, *, probe: str) -> Any:
        """Class probabilities as a frame with the original class labels as columns; values for regression."""
        if data.problem_type == "regression":
            return np.asarray(fitted.model.predict(X.copy()), dtype=float).reshape(-1)
        proba = np.asarray(fitted.model.predict_proba(X.copy()), dtype=float)
        cleaner = fitted.label_cleaner
        if data.problem_type == "binary":
            negative, positive = cleaner.ordered_class_labels
            return pd.DataFrame({negative: 1.0 - proba, positive: proba}, index=X.index)
        return cleaner.inverse_transform_proba(proba, as_pandas=True)

    def fit_predict(
        self, data: ProbeData, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, *, probe: str
    ) -> tuple[FittedModel, Any]:
        """Fit and predict, returning ``(fitted_model, test_prediction)`` like ``SystemUnderAudit``."""
        fitted = self.fit(data, X_train, y_train, probe=probe)
        return fitted, self.predict(fitted, data, X_test, probe=probe)


def resolve_model(method: str):
    """The ``ModelInfo`` registered under ``method`` (``MethodMetadata.method`` or the display name)."""
    from tabarena.models import get_model_registry

    registry = get_model_registry()
    if method in registry:
        return registry[method]
    for info in registry.values():
        if info.method_metadata.name == method:
            return info
    raise ValueError(f"No registered model {method!r}; `--list-models` prints the names.")


# --------------------------------------------------------------------------------------------------
# Turning the test-time augmentation off
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AugmentationOff:
    """Hyperparameters that run a model as one member without feature shifting, and where they came from."""

    hyperparameters: dict
    source: str
    """``"registry"`` (an entry below), ``"fallback"`` (the class's ``cheap_hyperparameters``) or ``"user"``."""
    note: str = ""


def strip_limix_config(config: dict, *, classification: bool) -> tuple[dict, list[int]]:
    """The #648 recipe on a LimiX-2 inference config: the members that keep the column order, without their
    polynomial interaction features and with the value rebalancing, the SVD features and the appended
    originals turned off.

    Returns the new config and the indices of the members it kept. The classification config has two such
    members (21 and 29 at LimiX ``774aa3e``); every regression member shuffles, so the first one is taken
    and its shuffler disabled.
    """
    members = list(config["pipelines"])
    kept = [i for i, p in enumerate(members) if not p.get("FeatureShuffler", {}).get("mode")]
    if not kept:
        if classification:
            raise ValueError("No LimiX inference member keeps the column order; the pinned config has changed.")
        kept = [0]
    pipelines = []
    for i in kept:
        member = json.loads(json.dumps(members[i]))
        member.setdefault("FeatureShuffler", {})["mode"] = None
        member.pop("PolynomialInteractionGenerator", None)
        member.setdefault("RebalanceFeatureDistribution", {}).update(
            worker_tags=[None], svd_tag=None, original_flag=False
        )
        pipelines.append(member)
    return {**config, "pipelines": pipelines}, kept


def _limix_2_off(problem_type: str) -> AugmentationOff:
    from importlib import resources

    classification = problem_type in ("binary", "multiclass")
    filename = "cls_default_noretrieval_v2.json" if classification else "reg_default_noretrieval_v2.json"
    try:
        text = resources.files("config").joinpath(filename).read_text()
    except ModuleNotFoundError as err:
        raise ImportError("LimiX-2 needs the LimiX inference package (the `limix_2` extra).") from err
    config, kept = strip_limix_config(json.loads(text), classification=classification)
    return AugmentationOff(
        {"inference_config": config},
        "registry",
        f"members {kept} of {filename}, without polynomial, rebalancing and SVD features (#648)",
    )


def _constant(hyperparameters: dict, note: str) -> Callable[[str], AugmentationOff]:
    return lambda _problem_type: AugmentationOff(dict(hyperparameters), "registry", note)


#: ``MethodMetadata.method`` to the hyperparameters that turn the test-time augmentation off. Each entry
#: names a knob the wrapper forwards to the library (checked against the wrapper and the pinned library).
AUGMENTATION_OFF: dict[str, Callable[[str], AugmentationOff]] = {
    "LimiX-2": _limix_2_off,
    # ``TabPFN3Model`` forwards every hyperparameter to ``TabPFNClassifier`` / ``TabPFNRegressor``;
    # ``ModelInterfaceConfig.FEATURE_SHIFT_METHOD`` takes ``"shuffle"``, ``"rotate"`` or ``None``.
    "TabPFN-3": _constant(
        {"n_estimators": 1, "inference_config": {"FEATURE_SHIFT_METHOD": None}}, "one member, FEATURE_SHIFT_METHOD None"
    ),
    "TabPFN-3.5": _constant(
        {"n_estimators": 1, "inference_config": {"FEATURE_SHIFT_METHOD": None}}, "one member, FEATURE_SHIFT_METHOD None"
    ),
    "TabPFN-3.5-Fast": _constant(
        {"n_estimators": 1, "inference_config": {"FEATURE_SHIFT_METHOD": None}}, "one member, FEATURE_SHIFT_METHOD None"
    ),
    # ``TabPFNModel`` (the v2.5 / v2.6 wrapper) folds ``inference_config/<KEY>`` into ``inference_config``.
    "TabPFN-v2.6": _constant(
        {"n_estimators": 1, "inference_config/FEATURE_SHIFT_METHOD": None}, "one member, FEATURE_SHIFT_METHOD None"
    ),
    "RealTabPFN-v2.5": _constant(
        {"n_estimators": 1, "inference_config/FEATURE_SHIFT_METHOD": None}, "one member, FEATURE_SHIFT_METHOD None"
    ),
    # ``TabICLModelBase`` forwards its hyperparameters to the estimator; ``feat_shuffle_method="none"`` keeps
    # the original feature order (``"latin"`` is the default).
    "TabICLv2": _constant({"n_estimators": 1, "feat_shuffle_method": "none"}, "one member, feat_shuffle_method none"),
    "TabICL_GPU": _constant({"n_estimators": 1, "feat_shuffle_method": "none"}, "one member, feat_shuffle_method none"),
    # TabDPT has no feature-order knob the wrapper forwards; ``n_ensembles`` is a predict-time knob of v1.2+.
    "TabDPT-1.3": _constant({"n_ensembles": 1}, "one ensemble; TabDPT exposes no feature-order switch"),
    "TabDPT-Turbo": _constant({"n_ensembles": 1}, "one ensemble; TabDPT exposes no feature-order switch"),
    "TabDPT_GPU": _constant({}, "TabDPT v1.1 exposes neither an ensemble nor a feature-order switch"),
}


def augmentation_off(method: str, model_cls: type, problem_type: str) -> AugmentationOff:
    """The augmentation-off hyperparameters of ``method``: its ``AUGMENTATION_OFF`` entry, else the class's
    ``cheap_hyperparameters`` (one member for most foundation models, but not necessarily without a feature
    shift), else nothing.
    """
    if method in AUGMENTATION_OFF:
        return AUGMENTATION_OFF[method](problem_type)
    cheap = dict(getattr(model_cls, "cheap_hyperparameters", None) or {})
    note = "cheap_hyperparameters of the wrapper; whether this turns the feature shift off is not verified"
    return AugmentationOff(
        cheap, "fallback", note if cheap else "no augmentation-off entry and no cheap_hyperparameters"
    )


# --------------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------------
@dataclass
class DatasetSplits:
    """One dataset's splits as the probes see them (``ProbeData`` per split, same columns throughout)."""

    name: str
    problem_type: str
    splits: list[ProbeData]
    split_indices: list[int]
    task_id: str | None = None
    spec: str | None = None
    """The dataset spec it was loaded from (``None`` for ``ProbeData`` handed in directly)."""


def _probe_data_from_task(
    task,
    *,
    name: str,
    problem_type: str,
    fold: int,
    repeat: int,
    identifiers: tuple[str, ...],
    max_train_rows: int | None,
    max_test_rows: int | None,
    seed: int,
) -> ProbeData:
    X_train, y_train, X_test, y_test = task.get_train_test_split(fold=fold, repeat=repeat)
    X_train, y_train = _subsample(X_train, y_train, max_train_rows, seed)
    X_test, y_test = _subsample(X_test, y_test, max_test_rows, seed)
    validation_metadata = task.get_validation_metadata()
    return ProbeData(
        name=name,
        problem_type=problem_type,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        target_name=validation_metadata.get_target_name(),
        validation_metadata=validation_metadata,
        identifiers=identifiers,
    )


def _resolve_split_indices(task, splits: Sequence[int] | str) -> list[tuple[int, int, int]]:
    """``(split index, fold, repeat)`` for ``"all"`` or the given split indices (``n_folds * repeat + fold``)."""
    n_repeats, n_folds, _ = task.get_split_dimensions()
    wanted = range(n_repeats * n_folds) if splits == "all" else [int(s) for s in splits]
    out = []
    for split in wanted:
        if not 0 <= split < n_repeats * n_folds:
            raise ValueError(f"Split {split} is outside the task's {n_repeats * n_folds} splits.")
        out.append((split, split % n_folds, split // n_folds))
    return out


def _tabarena_task(dataset: str):
    from tabarena.benchmark.task.spec import task_spec_from_task_id_str
    from tabarena.contexts import TabArenaContext

    meta = TabArenaContext().task_metadata
    rows = meta[(meta["dataset"] == dataset) | (meta["tabarena_task_name"] == dataset)]
    if rows.empty:
        raise ValueError(f"Dataset {dataset!r} is not in the TabArena-v0.1 task metadata.")
    row = rows.iloc[0]
    return task_spec_from_task_id_str(row["task_id_str"]).load(), row


def tabarena_dataset_names() -> list[str]:
    """Every TabArena-v0.1 dataset name."""
    from tabarena.contexts import TabArenaContext

    return list(TabArenaContext().task_metadata["dataset"])


def infer_problem_type(y: pd.Series) -> str:
    """``binary`` for two distinct values, ``multiclass`` for a categorical or small integer-coded target,
    ``regression`` otherwise.
    """
    n_unique = y.nunique(dropna=True)
    if n_unique == 2:
        return "binary"
    numeric = pd.api.types.is_numeric_dtype(y) and not isinstance(y.dtype, pd.CategoricalDtype)
    if not numeric or (n_unique <= 20 and np.all(np.asarray(y.dropna(), dtype=float) % 1 == 0)):
        return "multiclass"
    return "regression"


def _user_task(frame: pd.DataFrame, *, target: str, name: str, problem_type: str, seed: int):
    """An in-memory task over ``frame`` with a 3 x 3 repeated (stratified) k-fold split."""
    from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

    from tabarena.benchmark.task.user_task import UserTask, from_sklearn_splits_to_user_task_splits

    frame = frame.reset_index(drop=True)
    y = frame[target]
    if problem_type == "regression":
        splitter = RepeatedKFold(n_splits=3, n_repeats=3, random_state=seed)
        pairs = splitter.split(frame)
    else:
        splitter = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=seed)
        pairs = splitter.split(frame, y.astype(str))
    splits = from_sklearn_splits_to_user_task_splits(pairs, n_splits=3)
    return UserTask(task_name=name).create_task(
        target_feature=target,
        problem_type="regression" if problem_type == "regression" else "classification",
        dataset=frame,
        splits=splits,
    )


def _openml_frame(dataset_id: int) -> tuple[pd.DataFrame, str, str]:
    import openml

    ds = openml.datasets.get_dataset(
        dataset_id, download_data=True, download_qualities=False, download_features_meta_data=True
    )
    target = ds.default_target_attribute
    if not target or "," in target:
        raise ValueError(f"OpenML dataset {dataset_id} has no single default target; use csv:<path>:<target>.")
    frame, _, _, _ = ds.get_data(dataset_format="dataframe")
    return frame, target, f"openml_{dataset_id}_{ds.name}"


def load_dataset_splits(
    spec: str,
    *,
    splits: Sequence[int] | str = (0,),
    seed: int = 0,
    max_train_rows: int | None = DEFAULT_MAX_TRAIN_ROWS,
    max_test_rows: int | None = DEFAULT_MAX_TEST_ROWS,
) -> DatasetSplits:
    """Resolve one dataset spec (see the module docstring) to its splits."""
    if spec.startswith(SYNTHETIC_PREFIX):
        data = make_synthetic_split(spec.removeprefix(SYNTHETIC_PREFIX), seed=seed)
        return DatasetSplits(data.name, data.problem_type, [data], [0], spec=spec)
    if spec.startswith(OPENML_PREFIX):
        parts = spec.removeprefix(OPENML_PREFIX).split(":")
        frame, target, name = _openml_frame(int(parts[0]))
        problem_type = parts[1] if len(parts) > 1 else infer_problem_type(frame[target])
        task = _user_task(frame, target=target, name=name, problem_type=problem_type, seed=seed)
        identifiers: tuple[str, ...] = (name, parts[0])
        task_id = f"openml:{parts[0]}"
    elif spec.startswith(CSV_PREFIX):
        path, _, target = spec.removeprefix(CSV_PREFIX).rpartition(":")
        if not path or not target:
            raise ValueError(f"A csv spec is csv:<path>:<target>, got {spec!r}.")
        frame = pd.read_csv(path)
        name = Path(path).stem
        problem_type = infer_problem_type(frame[target])
        task = _user_task(frame, target=target, name=name, problem_type=problem_type, seed=seed)
        identifiers = (name,)
        task_id = spec
    else:
        task, row = _tabarena_task(spec)
        name, problem_type = str(row["dataset"]), str(row["problem_type"])
        identifiers = tuple(
            str(v)
            for v in {row["dataset"], row["tabarena_task_name"], row["dataset_name"], row["tid"], row["task_id_str"]}
        )
        task_id = str(row["tid"])
    probe_splits, indices = [], []
    for split, fold, repeat in _resolve_split_indices(task, splits):
        probe_splits.append(
            _probe_data_from_task(
                task,
                name=name,
                problem_type=problem_type,
                fold=fold,
                repeat=repeat,
                identifiers=identifiers,
                max_train_rows=max_train_rows,
                max_test_rows=max_test_rows,
                seed=seed,
            )
        )
        indices.append(split)
    return DatasetSplits(name, problem_type, probe_splits, indices, task_id=task_id, spec=spec)


def group_probe_data(entries: Iterable[ProbeData]) -> list[DatasetSplits]:
    """``ProbeData`` objects grouped by name into datasets (one split each, in order of appearance)."""
    groups: dict[str, DatasetSplits] = {}
    for data in entries:
        group = groups.setdefault(data.name, DatasetSplits(data.name, data.problem_type, [], []))
        group.splits.append(data)
        group.split_indices.append(len(group.splits) - 1)
    return list(groups.values())


# --------------------------------------------------------------------------------------------------
# Column orders and errors
# --------------------------------------------------------------------------------------------------
def column_orders(columns: Sequence, *, n_random: int, seed: int, dataset: str = "") -> dict[str, list]:
    """``published``, ``reversed`` and ``n_random`` seeded random orders of ``columns`` (``random0``, ...).

    The random orders depend on ``seed`` and ``dataset``, so every split of a dataset sees the same ones and
    the orders are reproducible from the JSON's ``seed``.
    """
    columns = list(columns)
    rng = np.random.default_rng([int(seed), zlib.crc32(dataset.encode())])
    orders = {PUBLISHED: columns, REVERSED: columns[::-1]}
    # A narrow table has few permutations; never store one that is already in use (it would score d = 0).
    n_random = min(n_random, max(math.factorial(len(columns)) - len(set(map(tuple, orders.values()))), 0))
    for i in range(n_random):
        while True:
            perm = [columns[j] for j in rng.permutation(len(columns))]
            if perm not in orders.values():
                break
        orders[f"random{i}"] = perm
    return orders


def permute_columns(X: pd.DataFrame, order: Sequence) -> pd.DataFrame:
    """``X`` with its columns in ``order`` (a permutation of them)."""
    if sorted(map(str, order)) != sorted(map(str, X.columns)):
        raise ValueError("`order` is not a permutation of the columns.")
    return X[list(order)]


def error_of(data: ProbeData, pred: Any) -> float:
    """The TabArena error of ``pred`` on ``data.y_test``: 1 - AUC, log loss or RMSE."""
    classes = _classes(data)
    if data.problem_type == "regression":
        return _error(data.problem_type, data.y_test, np.asarray(pred, dtype=float).reshape(-1), classes)
    return _error(data.problem_type, data.y_test, _proba_matrix(pred, classes), classes)


def error_floor(data: ProbeData) -> float:
    """The floor under the errors of a log ratio: ``ERR_EPS``, or a fraction of the target's spread."""
    if data.problem_type == "regression":
        return ERR_EPS_REGRESSION_STD * (float(data.y_test.astype(float).std()) or 1.0)
    return ERR_EPS


def log_ratio(numerator: float, denominator: float, *, eps: float = ERR_EPS) -> float:
    """``log(max(numerator, eps) / max(denominator, eps))``: symmetric, finite, zero for equal errors."""
    return math.log(max(numerator, eps) / max(denominator, eps))


# --------------------------------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------------------------------
def probe_order(mua: ModelUnderAudit, dataset: DatasetSplits, *, n_random: int, seed: int) -> CheckResult:
    """Fit under every column order on every split of ``dataset``; the verdict is set by ``calibrate_order``."""
    first = dataset.splits[0]
    orders = column_orders(first.X_train.columns, n_random=n_random, seed=seed, dataset=dataset.name)
    err: dict[str, dict[str, float]] = {name: {} for name in orders}
    fit_s: dict[str, dict[str, float]] = {name: {} for name in orders}
    for split, data in zip(dataset.split_indices, dataset.splits, strict=True):
        for name, order in orders.items():
            start = time.monotonic()
            fitted, pred = mua.fit_predict(
                data,
                permute_columns(data.X_train, order),
                data.y_train,
                permute_columns(data.X_test, order),
                probe=f"order:{name}",
            )
            fitted.cleanup()
            err[name][str(split)] = error_of(data, pred)
            fit_s[name][str(split)] = time.monotonic() - start
    stats = order_statistics(err, eps=error_floor(first))
    stats["metric"] = first.eval_metric
    return CheckResult(
        "order",
        dataset.name,
        INFO,
        _order_summary(stats, err),
        {"orders": {k: [str(c) for c in v] for k, v in orders.items()}, "err": err, "fit_s": fit_s, **stats},
    )


def order_statistics(err: dict[str, dict[str, float]], *, eps: float) -> dict:
    """``d_perm`` (per permutation, split mean), ``d_mean``, ``d_min``, ``rank_pub``, ``rank_p``, and the
    per-split values, from the error table ``err[order][split]``.
    """
    splits = list(err[PUBLISHED])
    others = [name for name in err if name != PUBLISHED]
    d_split = {
        split: {name: log_ratio(err[name][split], err[PUBLISHED][split], eps=eps) for name in others}
        for split in splits
    }
    d_perm = {name: float(np.mean([d_split[s][name] for s in splits])) for name in others}
    n_orders = len(err)
    rank_split = {
        split: 1
        + sum(err[name][split] < err[PUBLISHED][split] for name in others)
        + 0.5 * sum(err[name][split] == err[PUBLISHED][split] for name in others)
        for split in splits
    }
    rank_pub = float(np.mean(list(rank_split.values())))
    return {
        "d_perm": d_perm,
        "d_mean": float(np.mean(list(d_perm.values()))),
        "d_min": float(min(d_perm.values())),
        "d_min_per_split": {s: float(min(d_split[s].values())) for s in splits},
        "rank_pub": rank_pub,
        "rank_p": rank_pub / n_orders,
        "n_orders": n_orders,
        "n_splits": len(splits),
    }


def _order_summary(stats: dict, err: dict[str, dict[str, float]]) -> str:
    def mean_err(name: str) -> float:
        return float(np.mean(list(err[name].values())))

    randoms = [name for name in err if name.startswith("random")]
    random_text = ", random " + "/".join(f"{mean_err(n):.4f}" for n in randoms) if randoms else ""
    return (
        f"{stats['metric']} published {mean_err(PUBLISHED):.4f}, reversed {mean_err(REVERSED):.4f}{random_text}; "
        f"d_mean {stats['d_mean']:+.2f}, d_min {stats['d_min']:+.2f}, rank {stats['rank_pub']:.1f}/{stats['n_orders']} "
        f"over {stats['n_splits']} split(s)"
    )


def probe_ensemble(
    mua_released: ModelUnderAudit, dataset: DatasetSplits, order_result: CheckResult, *, seed: int
) -> CheckResult:
    """One fit with the released configuration on the published order of the first split, against the
    augmentation-off error of ``order_result`` on that split.
    """
    data = dataset.splits[0]
    split = str(dataset.split_indices[0])
    fitted, pred = mua_released.fit_predict(data, data.X_train, data.y_train, data.X_test, probe="ensemble")
    fitted.cleanup()
    err_released = error_of(data, pred)
    err_off = order_result.details["err"][PUBLISHED][split]
    d = log_ratio(err_released, err_off, eps=error_floor(data))
    verdict = WARN if d >= ENSEMBLE_LOG_RATIO_WARN else INFO
    summary = (
        f"{data.eval_metric} released {err_released:.4f} vs augmentation off {err_off:.4f} (published order, "
        f"split {split}): d {d:+.2f}"
    )
    if verdict == WARN:
        summary += "; stripping the ensemble helped"
    return CheckResult(
        "ensemble", dataset.name, verdict, summary, {"err_released": err_released, "err_off": err_off, "d": d}
    )


def probe_secondary(
    mua: ModelUnderAudit, dataset: DatasetSplits, checks: Sequence[str], *, seed: int
) -> list[CheckResult]:
    """The ``audit_system`` behaviour probes on the published order of the first split (plus its baseline)."""
    return probe_dataset(mua, dataset.splits[0], list(checks), seed=seed)


# --------------------------------------------------------------------------------------------------
# Calibration and verdicts
# --------------------------------------------------------------------------------------------------
def _robust_center(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median))) * 1.4826
    return median, max(mad, ORDER_MIN_MAD)


def calibrate_order(
    results: Sequence[CheckResult],
    problem_types: dict[str, str],
    *,
    confirmed: Iterable[str] = (),
    no_calibration: bool = False,
) -> dict:
    """Set every ``order`` result's verdict from the run's own distribution of ``d_mean`` (see the module
    docstring) and return the calibration used. ``confirmed`` names the datasets whose result comes from the
    confirmation rerun.
    """
    order = [r for r in results if r.check == "order"]
    confirmed = set(confirmed)
    by_type: dict[str, list[float]] = {}
    for r in order:
        by_type.setdefault(problem_types.get(r.dataset, ""), []).append(r.details["d_mean"])
    pooled = [d for ds in by_type.values() for d in ds]
    calibration: dict[str, Any] = {"n_datasets": len(pooled), "by_problem_type": {}, "pooled": None}
    if len(pooled) >= CALIBRATION_MIN_DATASETS:
        calibration["pooled"] = dict(zip(("median_d", "mad_d"), _robust_center(pooled), strict=True))
        for ptype, values in by_type.items():
            if len(values) >= CALIBRATION_MIN_PER_TYPE:
                calibration["by_problem_type"][ptype] = dict(
                    zip(("median_d", "mad_d", "n"), (*_robust_center(values), len(values)), strict=True)
                )
    # A model that prefers the published order on most of its datasets has no honest median to compare with.
    unreliable = calibration["pooled"] is not None and calibration["pooled"]["median_d"] >= ORDER_LOG_RATIO_WARN
    calibration["unreliable"] = unreliable
    for r in order:
        d = r.details
        center = calibration["by_problem_type"].get(problem_types.get(r.dataset, "")) or calibration["pooled"]
        if center is None and not no_calibration:
            r.verdict = SKIP
            r.details["z"] = None
            r.summary = _strip_verdict_note(r.summary) + (
                f"; no calibration ({len(pooled)} dataset(s), {CALIBRATION_MIN_DATASETS} needed; --no-calibration applies the floors alone)"
            )
            continue
        z = (d["d_mean"] - center["median_d"]) / center["mad_d"] if center else None
        r.details["z"] = z
        stands_out = z is None or unreliable or z >= ORDER_MAD_MULTIPLIER
        big_effect = d["d_min"] >= ORDER_LOG_RATIO_FAIL
        effect = big_effect and stands_out
        several_splits = d["n_splits"] > 1 and all(v > 0 for v in d["d_min_per_split"].values())
        is_confirmed = r.dataset in confirmed and d["rank_p"] <= ORDER_RANK_P_CONFIRM
        if effect and (is_confirmed or several_splits):
            r.verdict = FAIL
            note = "the published order beats every permutation" + (
                ", confirmed" if is_confirmed else f" on all {d['n_splits']} splits"
            )
        elif big_effect or (d["d_mean"] >= ORDER_LOG_RATIO_WARN and stands_out):
            r.verdict = WARN
            note = "suspect"
            if big_effect and not stands_out:
                note += "; within the run's own spread (most datasets move with the column order)"
            if effect and r.dataset not in confirmed:
                note += "; rerun with --confirm"
        else:
            r.verdict = PASS
            note = ""
        z_text = f"z {z:+.1f}" if z is not None else "no calibration"
        r.summary = _strip_verdict_note(r.summary) + f"; {z_text}" + (f"; {note}" if note else "")
    return calibration


def _strip_verdict_note(summary: str) -> str:
    """The order summary before the calibration added its note (so recalibration does not stack notes)."""
    marker = " split(s)"
    return summary.split(marker)[0] + marker if marker in summary else summary


def dataset_verdicts(results: Sequence[CheckResult]) -> list[CheckResult]:
    """One ``verdict`` row per dataset: the ``order`` verdict, raised to WARN by the ensemble or secondary probes."""
    by_dataset: dict[str, list[CheckResult]] = {}
    for r in results:
        if r.check != VERDICT_CHECK and r.dataset != "-":
            by_dataset.setdefault(r.dataset, []).append(r)
    out = []
    for dataset, rows in by_dataset.items():
        order = next((r for r in rows if r.check == "order"), None)
        verdict = order.verdict if order else SKIP
        reasons = [f"order {order.verdict}"] if order else ["no order probe"]
        raised = [r for r in rows if r.check == "ensemble" and r.verdict == WARN] + [
            r for r in rows if r.check in SECONDARY_CHECKS and r.verdict in (WARN, FAIL)
        ]
        failed_probe = [r for r in rows if r.check == "probe"]
        if raised and verdict in (PASS, INFO, SKIP):
            verdict = WARN
        reasons += [f"{r.check} {r.verdict}" for r in raised]
        if failed_probe:
            reasons.append("a fit raised")
        out.append(CheckResult(VERDICT_CHECK, dataset, verdict, ", ".join(reasons), {"from": [r.check for r in rows]}))
    return out


def leakage_tuple(results: Sequence[CheckResult], verdict: str = FAIL) -> tuple[str, ...]:
    """The datasets whose ``verdict`` row is ``verdict``, sorted, in the form ``ModelDescriptor`` takes."""
    return tuple(sorted(r.dataset for r in results if r.check == VERDICT_CHECK and r.verdict == verdict))


# --------------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------------
def audit_contamination(
    model: str | type,
    *,
    datasets: Sequence[str | ProbeData] = ("all",),
    checks: Sequence[str] = CHECKS,
    splits: Sequence[int] | str = (0,),
    n_random_orders: int = ORDER_N_RANDOM,
    confirm: bool = False,
    config: str = "off",
    hyperparameters_off: dict | None = None,
    hyperparameters_released: dict | None = None,
    num_gpus: float = 1,
    num_cpus: int | None = None,
    seed: int = 0,
    max_train_rows: int | None = DEFAULT_MAX_TRAIN_ROWS,
    max_test_rows: int | None = DEFAULT_MAX_TEST_ROWS,
    no_calibration: bool = False,
    log: Callable[[str], None] = print,
    meta: dict | None = None,
) -> list[CheckResult]:
    """Run the audit and return every check result plus one ``verdict`` row per dataset.

    ``model`` is a registry name or a model class (then ``hyperparameters_off`` is what the class runs with).
    ``datasets`` are specs (see the module docstring) or ``ProbeData`` objects, grouped into datasets by name.
    ``meta`` (optional) collects what the JSON records: the configurations, the calibration and the splits.
    """
    checks = tuple(checks)
    unknown = set(checks) - set(CHECKS)
    if unknown:
        raise ValueError(f"Unknown check(s) {sorted(unknown)}; valid: {CHECKS}")
    if config not in ("off", "released"):
        raise ValueError(f"config must be 'off' or 'released', got {config!r}")
    if "ensemble" in checks and "order" not in checks:
        raise ValueError("The ensemble probe runs on the datasets the order probe flags; add 'order' to the checks.")
    if "ensemble" in checks and config != "off":
        raise ValueError(
            "The ensemble probe compares the released configuration with augmentation off: use config 'off'."
        )
    meta = meta if meta is not None else {}
    method = model if isinstance(model, str) else None
    model_cls = resolve_model(model).model_cls if isinstance(model, str) else model

    # Datasets, grouped into splits.
    groups: list[DatasetSplits] = []
    inline: list[ProbeData] = []
    for entry in datasets:
        if isinstance(entry, ProbeData):
            inline.append(entry)
            continue
        if inline:
            groups += group_probe_data(inline)
            inline = []
        names = tabarena_dataset_names() if entry == "all" else [entry]
        for name in names:
            groups.append(
                load_dataset_splits(
                    name, splits=splits, seed=seed, max_train_rows=max_train_rows, max_test_rows=max_test_rows
                )
            )
    groups += group_probe_data(inline)
    meta["datasets"] = [
        {
            "name": g.name,
            "task_id": g.task_id,
            "problem_type": g.problem_type,
            "splits": [
                {"index": i, "n_train": len(d.X_train), "n_test": len(d.X_test), "n_features": d.X_train.shape[1]}
                for i, d in zip(g.split_indices, g.splits, strict=True)
            ],
        }
        for g in groups
    ]

    # The two configurations. Augmentation off depends on the problem type (LimiX has one config per task).
    released = dict(hyperparameters_released or {})
    offs: dict[str, AugmentationOff] = {}

    def off_for(problem_type: str) -> AugmentationOff:
        if problem_type not in offs:
            if hyperparameters_off is not None:
                offs[problem_type] = AugmentationOff(dict(hyperparameters_off), "user")
            elif method is None:
                offs[problem_type] = AugmentationOff({}, "user", "model class given directly")
            else:
                offs[problem_type] = augmentation_off(method, model_cls, problem_type)
        return offs[problem_type]

    def mua_for(dataset: DatasetSplits, which: str) -> ModelUnderAudit:
        hps = released if which == "released" else off_for(dataset.problem_type).hyperparameters
        return ModelUnderAudit(model_cls, hyperparameters=hps, num_gpus=num_gpus, num_cpus=num_cpus)

    results: list[CheckResult] = []
    problem_types = {g.name: g.problem_type for g in groups}
    calls: list[dict] = []
    for g in groups:
        off = off_for(g.problem_type)
        log(
            f"--- {g.name} ({g.problem_type}, {len(g.splits)} split(s), {len(g.splits[0].X_train)} train / "
            f"{len(g.splits[0].X_test)} test rows; config {config}: {off.note or off.hyperparameters})"
        )
        mua = mua_for(g, config)
        new: list[CheckResult] = []
        try:
            if "order" in checks:
                new.append(probe_order(mua, g, n_random=n_random_orders, seed=seed))
                log(format_result(new[-1]))
            secondary = [c for c in checks if c in SECONDARY_CHECKS]
            if secondary:
                for r in probe_secondary(mua, g, secondary, seed=seed):
                    new.append(r)
                    log(format_result(r))
        except Exception as exc:
            failed = next((c for c in mua.calls if "error" in c), None)
            where = f" on the {failed['probe']!r} fit" if failed else ""
            new.append(
                CheckResult("probe", g.name, FAIL, f"the model raised {type(exc).__name__}{where}: {str(exc)[:300]}")
            )
            log(format_result(new[-1]))
        finally:
            mua.cleanup()
            calls += mua.calls
        results += new

    calibration = calibrate_order(results, problem_types, no_calibration=no_calibration)
    confirmed: set[str] = set()
    if confirm and "order" in checks:
        # Recalibrating after a rerun can flag a dataset that passed the screen; confirm those too, a few times.
        for _ in range(3):
            suspects = [
                g
                for g in groups
                if g.name not in confirmed
                and any(r.check == "order" and r.verdict in (WARN, FAIL) and r.dataset == g.name for r in results)
            ]
            if not suspects:
                break
            for g in suspects:
                log(f"--- confirming {g.name}: {ORDER_N_RANDOM_CONFIRM} random orders on every split")
                full = (
                    load_dataset_splits(
                        g.spec, splits="all", seed=seed, max_train_rows=max_train_rows, max_test_rows=max_test_rows
                    )
                    if g.spec is not None
                    else g
                )
                mua = mua_for(full, config)
                try:
                    rerun = probe_order(mua, full, n_random=ORDER_N_RANDOM_CONFIRM, seed=seed)
                except Exception as exc:
                    results.append(
                        CheckResult(
                            "probe",
                            g.name,
                            FAIL,
                            f"the confirmation fit raised {type(exc).__name__}: {str(exc)[:300]}",
                        )
                    )
                    log(format_result(results[-1]))
                    confirmed.add(g.name)  # not retried
                    continue
                finally:
                    mua.cleanup()
                    calls += mua.calls
                results = [r for r in results if not (r.check == "order" and r.dataset == g.name)] + [rerun]
                confirmed.add(g.name)
            calibration = calibrate_order(results, problem_types, confirmed=confirmed, no_calibration=no_calibration)
    if calibration.get("unreliable"):
        results.append(
            CheckResult(
                "calibration",
                "-",
                WARN,
                f"the run's median d_mean is {calibration['pooled']['median_d']:+.2f}: the model prefers the "
                "published order on most of its datasets, so the floors apply alone",
                calibration,
            )
        )
        log(format_result(results[-1]))
    if "order" in checks:
        log("--- order verdicts after calibration")
        for r in results:
            if r.check == "order":
                log(format_result(r))
    if "ensemble" in checks and config == "off":
        for g in groups:
            order_result = next((r for r in results if r.check == "order" and r.dataset == g.name), None)
            if order_result is None or order_result.verdict not in (WARN, FAIL):
                continue
            log(f"--- ensemble probe on {g.name}: one fit with the released configuration")
            mua = mua_for(g, "released")
            try:
                results.append(probe_ensemble(mua, g, order_result, seed=seed))
            except Exception as exc:
                results.append(
                    CheckResult(
                        "probe", g.name, FAIL, f"the released fit raised {type(exc).__name__}: {str(exc)[:300]}"
                    )
                )
            finally:
                mua.cleanup()
                calls += mua.calls
            log(format_result(results[-1]))
    verdicts = dataset_verdicts(results)
    results += verdicts
    for r in verdicts:
        log(format_result(r))
    meta.update(
        {
            "config": config,
            "hyperparameters_released": released,
            "hyperparameters_released_sha256": _sha256(released),
            "hyperparameters_off": {ptype: off.hyperparameters for ptype, off in offs.items()},
            "hyperparameters_off_sha256": {ptype: _sha256(off.hyperparameters) for ptype, off in offs.items()},
            "augmentation_off_source": {ptype: off.source for ptype, off in offs.items()},
            "augmentation_off_note": {ptype: off.note for ptype, off in offs.items()},
            "calibration": calibration,
            "confirmed": sorted(confirmed),
            "calls": calls,
        }
    )
    return results


def _sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in VERSIONED_PACKAGES:
        try:
            out[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            out[name] = None
    return out


def _device_name(num_gpus: float) -> str:
    if not num_gpus:
        return "cpu"
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            return f"cuda:0 {torch.cuda.get_device_name(0)}"
    return "cpu (no CUDA device)"


def list_models() -> list[str]:
    """``method [augmentation off: ...]`` lines for every registered model."""
    from tabarena.models import get_model_registry

    lines = []
    for method, info in sorted(get_model_registry().items()):
        if method in AUGMENTATION_OFF:
            how = "registry entry"
        elif getattr(info.model_cls, "cheap_hyperparameters", None):
            how = f"fallback {info.model_cls.cheap_hyperparameters}"
        else:
            how = "none"
        lines.append(f"{method:<24} augmentation off: {how}")
    return lines


def _parse_splits(values: list[str]) -> Sequence[int] | str:
    if values == ["all"]:
        return "all"
    try:
        return [int(v) for v in values]
    except ValueError as err:
        raise argparse.ArgumentTypeError("--splits takes split indices or 'all'") from err


def main(argv: list[str] | None = None) -> int:
    from tabarena.models._registry import assert_autogluon_resolves

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="Registered model (MethodMetadata.method, e.g. LimiX-2).")
    parser.add_argument("--list-models", action="store_true", help="Print the registered models and exit.")
    parser.add_argument(
        "--datasets", nargs="+", default=["all"], help="TabArena-v0.1 names, 'all', openml:<id>, csv:<path>:<target>."
    )
    parser.add_argument("--checks", nargs="+", default=list(CHECKS), choices=CHECKS)
    parser.add_argument("--splits", nargs="+", default=["0"], help="Split indices (n_folds * repeat + fold) or 'all'.")
    parser.add_argument("--random-orders", type=int, default=ORDER_N_RANDOM, help="Random column orders in the screen.")
    parser.add_argument(
        "--confirm", action="store_true", help="Rerun suspect datasets with more orders on every split."
    )
    parser.add_argument(
        "--config", choices=("off", "released"), default="off", help="Augmentation off (default) or as released."
    )
    parser.add_argument(
        "--hyperparameters-off", default=None, help="JSON replacing the augmentation-off hyperparameters."
    )
    parser.add_argument("--hyperparameters-released", default=None, help="JSON replacing the registry default (empty).")
    parser.add_argument("--num-gpus", type=float, default=1)
    parser.add_argument("--num-cpus", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Default: a fresh seed, printed for reruns.")
    parser.add_argument("--max-train-rows", type=int, default=DEFAULT_MAX_TRAIN_ROWS)
    parser.add_argument("--max-test-rows", type=int, default=DEFAULT_MAX_TEST_ROWS)
    parser.add_argument(
        "--no-calibration", action="store_true", help="Apply the fixed floors without the across-dataset rule."
    )
    parser.add_argument("--json", help="Write every result (with details) to this path.")
    args = parser.parse_args(argv)
    assert_autogluon_resolves()
    if args.list_models:
        print("\n".join(list_models()))
        return 0
    if not args.model:
        parser.error("pass --model (or --list-models)")
        return 2
    try:
        splits = _parse_splits(args.splits)
    except argparse.ArgumentTypeError as err:
        parser.error(str(err))
        return 2

    seed = args.seed if args.seed is not None else int(time.time()) % 2**31
    info = resolve_model(args.model)
    model_cls = info.model_cls
    meta: dict[str, Any] = {
        "tool": "tabarena.tools.audit_contamination",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "model": args.model,
        "model_cls": f"{model_cls.__module__}.{model_cls.__qualname__}",
        "date_introduced": info.method_metadata.date_introduced,
        "checks": args.checks,
        "n_random_orders": args.random_orders,
        "confirm": args.confirm,
        "splits": splits,
        "max_train_rows": args.max_train_rows,
        "max_test_rows": args.max_test_rows,
        "num_gpus": args.num_gpus,
        "device": _device_name(args.num_gpus),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
        "thresholds": {name: globals()[name] for name in THRESHOLDS},
    }
    print(f"audit_contamination: {args.model} ({meta['model_cls']}), seed {seed}, device {meta['device']}")
    results = audit_contamination(
        args.model,
        datasets=args.datasets,
        checks=args.checks,
        splits=splits,
        n_random_orders=args.random_orders,
        confirm=args.confirm,
        config=args.config,
        hyperparameters_off=json.loads(args.hyperparameters_off) if args.hyperparameters_off else None,
        hyperparameters_released=json.loads(args.hyperparameters_released) if args.hyperparameters_released else None,
        num_gpus=args.num_gpus,
        num_cpus=args.num_cpus,
        seed=seed,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
        no_calibration=args.no_calibration,
        meta=meta,
    )
    verdict_rows = [r for r in results if r.check == VERDICT_CHECK]
    counts = {v: sum(r.verdict == v for r in verdict_rows) for v in (FAIL, WARN, PASS, SKIP)}
    raised = sum(r.check == "probe" for r in results)
    summary = ", ".join(f"{n} {v}" for v, n in counts.items() if n) + " dataset(s)"
    if raised:
        summary += f", {raised} fit(s) raised"
    print(f"audit_contamination summary: {summary}")
    print(f"leakage_datasets={leakage_tuple(results)!r}")
    print(f"warn_datasets={leakage_tuple(results, WARN)!r}")
    if args.json:
        _write_json(results, args.json, meta)
    return 1 if counts[FAIL] or raised else 0


if __name__ == "__main__":
    sys.exit(main())
