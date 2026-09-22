"""Every registered model without a declared class cap must fit a 15-class problem.

A tabular foundation model's checkpoint has a fixed-width classification head, ten classes for most
of them and 160 for TabPFN-3 and TabPFN-3.5. A wrapper either declares that width as ``ag.max_classes``
(AutoGluon then refuses a wider dataset before the fit) or handles wider label sets: through the
library's own scheme (Causilo, EXAONE Tabular, TabDPT, TabLDM and TabSwift decompose or group the
labels) or through the ``ManyClassClassifier`` output coding of tabpfn-extensions. This module pins
the second contract for every model that makes no declaration. A wrapper that exposes the head width
as ``many_class_threshold`` gets a second case that forces its output-coding branch below the native
width, so the branch runs without a dataset wider than the head.

The fits run on whatever device the machine has: the wrappers fall back to the CPU when no GPU was
allocated, and the toy problem is small enough for that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from autogluon.core.data import LabelCleaner
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from .smoke_configs import registry_or_fail, smoke_for

_REGISTRY = registry_or_fail()

#: Above the ten-class head of most checkpoints, below TabPFN-3's 160, with 20 rows per class.
N_CLASSES = 15


def _many_class_data(n_classes: int = N_CLASSES, seed: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """An ``n_classes``-way toy problem with one categorical column, preprocessed as AutoGluon hands data to a model."""
    X_np, y_np = make_classification(
        n_samples=20 * n_classes,
        n_features=8,
        n_informative=6,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=seed,
    )
    X = pd.DataFrame(X_np, columns=[f"num_{i}" for i in range(X_np.shape[1])])
    # A string column, so the frame a many-class wrapper hands its library carries a category dtype.
    bins = np.digitize(X_np[:, 0], np.quantile(X_np[:, 0], [0.25, 0.5, 0.75]))
    X["cat"] = pd.Series(bins).map(dict(enumerate("abcd"))).to_numpy()
    y = pd.Series(y_np, name="target")
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    generator = AutoMLPipelineFeatureGenerator(verbosity=0)
    cleaner = LabelCleaner.construct(problem_type="multiclass", y=y_train)
    return generator.fit_transform(X_train), cleaner.transform(y_train), generator.transform(X_test)


def _construct(method: str, tmp_path, ag_args_fit: dict | None = None):
    """A ``method`` model with its smoke hyperparameters, or a skip when it cannot run a multiclass fit here."""
    info = _REGISTRY[method]
    if info.superseded:
        pytest.skip(f"{method}: superseded; its pip_extra {info.pip_extra} conflicts with the installed version")
    supported = info.model_cls.supported_problem_types()
    if supported is not None and "multiclass" not in supported:
        pytest.skip(f"{method}: no multiclass support")
    hyperparameters = dict(smoke_for(method).hyperparameters)
    if ag_args_fit:
        hyperparameters["ag_args_fit"] = {**hyperparameters.get("ag_args_fit", {}), **ag_args_fit}
    return info.model_cls(path=str(tmp_path), problem_type="multiclass", hyperparameters=hyperparameters)


def _fit_and_check(model, n_classes: int = N_CLASSES) -> None:
    X_train, y_train, X_test = _many_class_data(n_classes)
    try:
        model.fit(X=X_train, y=y_train)
    except ImportError as err:
        pytest.skip(f"{model.name}: optional dependency not installed ({err})")
    proba = model.predict_proba(X=X_test)
    assert proba.shape == (len(X_test), n_classes)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    assert len(model.predict(X=X_test)) == len(X_test)
    # A save and load round trip, the way a bagged fit hands its children back to the parent.
    loaded = type(model).load(model.save())
    np.testing.assert_allclose(loaded.predict_proba(X=X_test), proba, atol=1e-4)


@pytest.mark.models
@pytest.mark.parametrize("method", sorted(_REGISTRY), ids=str)
def test_model_many_class(method: str, tmp_path) -> None:
    """A model that declares no ``ag.max_classes`` below :data:`N_CLASSES` fits and predicts that many classes."""
    model = _construct(method, tmp_path)
    max_classes = model._get_params_aux().get("max_classes")  # resolved at fit time, so read the defaults here
    if max_classes is not None and max_classes < N_CLASSES:
        pytest.skip(f"{method}: declares ag.max_classes={max_classes}")
    _fit_and_check(model)


@pytest.mark.models
@pytest.mark.parametrize("method", sorted(_REGISTRY), ids=str)
def test_model_many_class_output_coding(method: str, tmp_path) -> None:
    """A wrapper with a ``many_class_threshold`` runs its output-coding branch when the threshold is lowered.

    The threshold is set below :data:`N_CLASSES` through ``ag_args_fit``, so the branch runs on the same
    toy problem whatever the checkpoint's native width; the fitted estimator must be the
    ``ManyClassClassifier`` with a codebook, not its single-fit shortcut.
    """
    if "many_class_threshold" not in _construct(method, tmp_path)._get_params_aux():
        pytest.skip(f"{method}: no many_class_threshold")
    model = _construct(method, tmp_path, ag_args_fit={"many_class_threshold": 5})
    _fit_and_check(model)

    from tabpfn_extensions.many_class import ManyClassClassifier

    assert isinstance(model.model, ManyClassClassifier)
    assert not model.model.no_mapping_needed_
