"""Abstract base class for feature selection methods."""

from __future__ import annotations

import math
import tempfile
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from autogluon.core.data import LabelCleaner
from autogluon.features.generators.abstract import AbstractFeatureGenerator
from autogluon.tabular.models.lgb.lgb_model import LGBModel

if TYPE_CHECKING:
    import pandas as pd
    from autogluon.core.models.abstract.abstract_model import AbstractModel


@dataclass
class ProxyModelConfig:
    """Configuration for the proxy model used in feature selection methods."""

    model_class: AbstractModel = LGBModel
    """Proxy Model class to use within feature selection methods."""
    model_hyperparameters: dict = field(default_factory=dict)
    """Hyperparameters to use when fitting the proxy model. Empty dict is the default."""
    use_bagged_model: bool = False
    """Whether to use a bagged model as the proxy model."""
    bagging_hyperparameters: dict = field(default_factory=dict)
    """Hyperparameters to set for the bagging model (such as refitting or not)."""
    verbosity: int = 0
    """Verbosity to use when fitting the proxy model."""
    problem_type: str | None = None
    """The problem type of the current datasets."""
    eval_metric: str | None = None
    """The evaluation metric used to evaluate the proxy model."""


class AbstractFeatureSelector(AbstractFeatureGenerator):
    """Abstract Feature Selector."""

    name: str
    """The name of the feature selector. This is used for logging and debugging purposes, and should be
    overridden by each feature selector."""
    feature_scoring_method: bool = False
    """If True, the method computes feature scores (higher is better) for features, instead
    of directly selecting features. The subclass only needs to implement logic that fill _feature_scores.
    If False, the method cannot compute feature scores, and directly fill _selected_features.
    """

    max_features: int
    """The maximum number of features to select."""

    _original_features: list[str]
    """The list of original features before fitting the feature selector."""

    _selected_features: list[str] | None
    """The list of selected features after fitting the feature selector.
    None, if the feature selector is not fitted yet.
    """
    _feature_scores: dict[str, float] | None
    """Mapping from feature name to score (higher is more important).
    None if the feature selector is not fitted yet, or if the method does not compute feature scores.

    Note: this mapping may be incomplete, depending on the method and
    if the method finished evaluating all features within the time limit.
    """

    _rng: np.random.Generator
    """Random number generator for fallback feature selection."""

    _problem_type: str | None
    """The problem type of the current dataset"""

    def __init__(
        self,
        max_features: int | float | Callable[[int], int],
        *,
        proxy_mode_config: ProxyModelConfig | None = None,
        raise_on_useless_feature_selection: bool = True,
        **kwargs,
    ):
        """Init from super class with additional feature selection specific parameters.

        Parameters
        ----------
        max_features: int, float, or Callable[[int], int]
            The maximum number of features to select.
            If int, we assume a fixed number of features to select.
            If float, we assume a fraction of features to select (0 < max_features < 1).
                We always round up.
            If callable, it receives the total number of input features as its single argument
                and must return the desired feature count as an int. Resolved at fit time.
        proxy_mode_config:
            Configuration of the proxy model to use inside the feature selection method.
        raise_on_useless_feature_selection:
            If True, the method raises an error when the input data contains less than max_features.

        """
        super().__init__(**kwargs)

        self.max_features_input = max_features
        self.proxy_mode_config = proxy_mode_config
        self.raise_on_useless_feature_selection = raise_on_useless_feature_selection

        self._selected_features = None
        self._feature_scores = None

    def _fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        time_limit: int | None = None,
        eval_metric: str | None = None,
        problem_type: str | None = None,
        **kwargs,  # noqa: ARG002
    ) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """Fit and transform for feature selection methods."""
        self._original_features = list(X.columns)
        old_type_family_groups_special = {}
        if self.feature_metadata_in.type_group_map_special is not None:
            old_type_family_groups_special = deepcopy(self.feature_metadata_in.type_group_map_special)

        # Init random generator
        self._rng = np.random.default_rng(self.random_state)
        self.problem_type = problem_type

        self._resolve_proxy_config(eval_metric=eval_metric, problem_type=problem_type)
        self._resolve_max_features()

        # Sanity check for feature selection setup
        if len(self._original_features) <= self.max_features:
            if self.raise_on_useless_feature_selection:
                raise ValueError(
                    "The number of features in the input data is less than or equal to max_features. "
                    "Feature selection has no affect on the pipeline in this case. "
                    "Set raise_on_useless_feature_selection to False to disable this check."
                )
            return X, old_type_family_groups_special

        # Call feature selection method
        feature_fit_kwargs = dict(X=X, y=y, time_limit=time_limit)
        if self.feature_scoring_method:
            self._feature_scores = self._fit_feature_scoring(**feature_fit_kwargs)
            assert isinstance(
                self._feature_scores, dict
            ), "The feature scores must be a dictionary mapping from feature name to score."
            self._selected_features = self.selected_features_from_feature_scores()
        else:
            self._selected_features = self._fit_feature_selection(**feature_fit_kwargs)
        assert isinstance(self._selected_features, list), "The selected features must be a list of feature names."
        # Transform (aka select features)
        X_out = self._transform(X=X)
        assert list(X_out) == self._selected_features, "The output features must be the same as the selected features."

        # Update the type family groups in the output feature metadata.
        type_family_groups_special: dict[str, list[str]] = {
            type_group_name: [f for f in feature_names if f in self._selected_features]
            for type_group_name, feature_names in old_type_family_groups_special.items()
        }

        return X_out, type_family_groups_special

    def _resolve_proxy_config(self, *, eval_metric: str | None, problem_type: str | None) -> None:
        """Synchronize proxy model config with the given eval_metric/problem_type."""
        if self.proxy_mode_config is None:
            return
        assert isinstance(self.proxy_mode_config, ProxyModelConfig)
        if self.proxy_mode_config.problem_type is not None:
            problem_type = self.proxy_mode_config.problem_type
        if self.proxy_mode_config.eval_metric is not None:
            eval_metric = self.proxy_mode_config.eval_metric
        self.proxy_mode_config.problem_type = problem_type
        self.proxy_mode_config.eval_metric = eval_metric

    def _resolve_max_features(self) -> None:
        """Resolve max_features from int, fraction, or callable."""
        if callable(self.max_features_input):
            self.max_features = self.max_features_input(len(self._original_features))
            self._log(20, f"Resolved max_features to {self.max_features} via callable.")
        elif isinstance(self.max_features_input, float):
            if not (0 < self.max_features_input < 1):
                raise ValueError("If max_features is a float, it must be in the range (0, 1).")
            self.max_features = math.ceil(len(self._original_features) * self.max_features_input)
            self._log(20, f"Resolved max_features to {self.max_features} based on the fraction provided.")
        else:
            self.max_features = self.max_features_input

    def _fit_feature_selection(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> list[str]:
        """Code that fills self._selected_features. Used if feature_scoring_method is False."""
        raise NotImplementedError

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        """Code that fills self._feature_scores. Used if feature_scoring_method is True."""
        raise NotImplementedError

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input data according to the selected features."""
        if self._selected_features is None:
            raise ValueError("Feature selector has not been fitted yet. Please call fit_transform first.")

        # Check if all of selected features is in X
        missing_features = [feature for feature in self._selected_features if feature not in X.columns]
        if missing_features:
            raise ValueError(f"Some of the selected features are not in the input data: {missing_features}")

        return X[self._selected_features]

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        """By default, all our feature selector want to get all data as input.

        Note: as we assume that we call the feature selector as a post generator,
        all data should be in a proper format for the feature selector.
        TODO: if not, determine how to adapt the code that the feature selector performs
            its own preprocessing.
        """
        return dict()

    # TODO: investigate how to improve randomness of this fallback method
    #   Note: by default, the column order of input features is likely always the same.
    #   Thus, the randomly selected features are likely always the same across different runs,
    #   which is not ideal for a fallback method. So likely, we want to somehow adjust the random state
    #   based on some property of the input data, or pass a different random state per run.
    def fallback_feature_selection(self, *, selected_features: list[str] | None = None) -> list[str]:
        """Randomly select features as a fallback with support for partial fallback.

        Parameters
        ----------
        selected_features: list[str] | None
            The list of already selected features that should be included in the output. If None, no
            features are guaranteed to be included in the output.
        """
        if selected_features is None:
            selected_features = []
            to_select_from_features = self._original_features[:]
        else:
            to_select_from_features = [
                feature for feature in self._original_features if feature not in selected_features
            ]
        num_feature_to_select = self.max_features - len(selected_features)

        features = list(self._rng.choice(to_select_from_features, size=num_feature_to_select, replace=False))
        return [str(feature) for feature in features]

    def selected_features_from_feature_scores(self) -> list[str]:
        """Convert feature scores to selected features."""
        feature_scores = self._feature_scores
        if feature_scores is None:
            raise ValueError("Feature scores are not computed yet. Please call _fit_feature_scoring first.")

        # Sort features by score (higher is better) and select the top max_features
        sorted_features = sorted(feature_scores, key=feature_scores.get, reverse=True)
        selected_features = sorted_features[: self.max_features]

        # Fallback for missing feature scores
        if len(selected_features) < self.max_features:
            self._log(
                30,
                f"Warning: Not enough feature scores computed to select {self.max_features} features. "
                f"Selected {len(selected_features)} features based on available scores, "
                f"and randomly selected the rest.",
            )

            selected_features = selected_features + self.fallback_feature_selection(selected_features=selected_features)

        return selected_features

    # TODO: make this a standalone static method re-usable across TabArena/AutoGluon
    #   - Similar to code in autogluon.core.utils.feature_selection.FeatureSelector
    #     we can add logging, stability, more...
    #   - Add support for cross-validation (instead of holdout?)
    #   - Figure out how to handle TimeLimitExceeded raise in this code or the code that calls this.
    #       -> would need to skip step.
    def evaluate_proxy_model(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        time_limit: int | None = None,
        n_train_subsample: int | None = 10_000,
        n_val_fraction: float = 0.33,
        random_state: int = 0,
    ) -> float:
        """Evaluate the proxy model on the given data."""
        from autogluon.core.models import BaggedEnsembleModel
        from autogluon.core.utils.utils import generate_train_test_split
        from autogluon.features.generators import AutoMLPipelineFeatureGenerator

        if self.proxy_mode_config is None:
            raise ValueError(
                "Proxy mode is not configured but the feature selection method needs a proxy model. "
                "Pass a ProxyModelConfig to the feature selection method!"
            )
        assert self.proxy_mode_config.eval_metric is not None, "When ProxyModel is needed, eval_metric must be set."
        assert self.proxy_mode_config.problem_type is not None, "When ProxyModel is needed, problem_type must be set."
        self._log(20, f"\tFitting Proxy Model (remaining time_limit: {time_limit})")

        # Init Proxy model
        problem_type = self.proxy_mode_config.problem_type
        eval_metric = self.proxy_mode_config.eval_metric
        model_class = self.proxy_mode_config.model_class
        use_bagged_model = self.proxy_mode_config.use_bagged_model
        hps = self.proxy_mode_config.model_hyperparameters
        bagging_hps = self.proxy_mode_config.bagging_hyperparameters
        verbosity = self.proxy_mode_config.verbosity
        model_kwargs = dict(
            name=f"ProxyModel_{model_class.ag_name}",
            problem_type=problem_type,
            eval_metric=eval_metric,
            hyperparameters=hps,
        )

        # Validation splits + Subsampling
        X, y = X.copy(), y.copy()  # avoid modifying the input data
        X_train, X_val, y_train, y_val = generate_train_test_split(
            X=X, y=y, test_size=n_val_fraction, random_state=random_state, problem_type=problem_type
        )
        del X, y  # free up memory
        if (n_train_subsample is not None) and (len(X_train) > n_train_subsample):
            self._log(
                20,
                f"\tNumber of training samples {len(X_train)} is greater than {n_train_subsample}. "
                f"Using {n_train_subsample} samples as training data.",
            )
            drop_ratio = 1.0 - n_train_subsample / len(X_train)
            X_train, _, y_train, _ = generate_train_test_split(
                X=X_train, y=y_train, problem_type=problem_type, random_state=random_state, test_size=drop_ratio
            )

        # Preprocessing
        feature_generator, label_cleaner = (
            AutoMLPipelineFeatureGenerator(verbosity=verbosity),
            LabelCleaner.construct(problem_type=problem_type, y=y_train, verbose=verbosity > 1),
        )
        X_train, y_train = (
            feature_generator.fit_transform(X_train),
            label_cleaner.transform(y_train),
        )
        X_val, y_val = feature_generator.transform(X_val), label_cleaner.transform(y_val)

        # Run proxy model
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_kwargs["path"] = tmp_dir
            if use_bagged_model:
                model = BaggedEnsembleModel(
                    model_class(**model_kwargs),
                    hyperparameters=bagging_hps,
                )
            else:
                model = model_class(**model_kwargs)
            model.rename("FeatureSelector_" + model.name)
            model.fit(
                X=X_train,
                y=y_train,
                time_limit=time_limit,
                num_classes=label_cleaner.num_classes,
                label_cleaner=label_cleaner,
            )
            score = model.score(X=X_val, y=y_val)

        # Ensure Python dtype
        return float(score)
