from __future__ import annotations

import copy
import gc
import shutil
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from autogluon.core.data.label_cleaner import LabelCleanerMulticlassToBinary
from autogluon.core.models import AbstractModel
from loguru import logger

from tabarena.benchmark.exec_models.autogluon_utils import resolve_holdout_split, resolve_validation_splits
from tabarena.benchmark.exec_models.base import AbstractExecModel
from tabarena.benchmark.exec_models.utils import _apply_inv_perm
from tabarena.benchmark.preprocessing.pipeline import build_feature_generator, resolve_preprocessing_pipeline
from tabarena.benchmark.task.metadata import ValidationMetadata

if TYPE_CHECKING:
    from autogluon.tabular import TabularPredictor


# FIXME: determine if want to persist by default?
class AGWrapper(AbstractExecModel):
    """An AutoGluon ``TabularPredictor`` wrapped as an exec model.

    Fits a full ``TabularPredictor`` (whatever ``init_kwargs`` / ``fit_kwargs`` describe)
    and exposes it through the common exec-model interface. Feature/label preprocessing
    is disabled by default here, since AutoGluon does its own; the label column is appended
    to the frame internally under the name resolved in ``_build_predictor_args``
    (the validation metadata's ``target_name``, else ``"__label__"``).

    When ``use_task_specific_validation`` is set, the task-specific validation split is built
    during ``_fit`` against ``self.validation_metadata``. The bagged path (``num_bag_folds > 1``)
    uses ``resolve_validation_splits`` to produce ``k`` group/time-aware folds (re-injected as
    ``ag_args_ensemble['custom_splits']``); the non-bagged holdout path uses
    ``resolve_holdout_split`` to produce a single group/time-aware train/validation split, fed to
    ``TabularPredictor`` as ``tuning_data`` (a single model does not consume ``custom_splits``).

    Parameters
    ----------
    init_kwargs:
        Extra keyword arguments for the ``TabularPredictor(...)`` constructor.
    fit_kwargs:
        Extra keyword arguments for ``TabularPredictor.fit(...)``. ``num_bag_folds`` /
        ``num_bag_sets`` here drive the (optionally task-specific) validation protocol.
    persist:
        If True, persist the best model in memory around inference (faster repeated
        prediction at the cost of memory).
    validation_metadata:
        Task-derived ``ValidationMetadata`` (or a kwargs dict for one) describing the
        validation-split structure. The label column appended to the training frame is named
        after its ``target_name`` (falling back to ``"__label__"``).
    use_task_specific_validation:
        If True, adapt the validation split to ``validation_metadata`` during fitting;
        otherwise use standard splitting and ignore the metadata's split columns.
    """

    # Default AutoGluon can return a validation score
    can_get_error_val = True
    # Default AutoGluon can return OOF predictions for the best model.
    can_get_oof = True

    # AutoGluon does its own feature/label preprocessing, so disable ours by default.
    preprocess_data = False
    preprocess_label = False

    predictor: TabularPredictor
    """The fitted AutoGluon ``TabularPredictor`` (set by ``_fit``)."""

    def __init__(
        self,
        init_kwargs: dict | None = None,
        fit_kwargs: dict | None = None,
        persist: bool = False,
        validation_metadata: ValidationMetadata | dict | None = None,
        use_task_specific_validation: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if init_kwargs is None:
            init_kwargs = {}
        if fit_kwargs is None:
            fit_kwargs = {}
        self.init_kwargs = init_kwargs
        self.fit_kwargs = fit_kwargs
        self.validation_metadata = ValidationMetadata.from_config(validation_metadata)
        self.use_task_specific_validation = use_task_specific_validation
        self.persist = persist

    def _build_predictor_args(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
    ) -> tuple[pd.DataFrame, dict, dict]:
        """Build the ``(train_data, init_kwargs, fit_kwargs)`` for ``TabularPredictor.fit``.

        Works on deep copies of the configured ``init_kwargs`` / ``fit_kwargs`` so the
        wrapper can be re-fit. The steps:

        1. Pop ``num_bag_folds`` / ``num_bag_sets`` and run them through the task-specific
           validation protocol (``resolve_validation_splits``), which may adjust the fold /
           repeat counts and/or produce explicit ``custom_splits`` (re-injected into
           ``ag_args_ensemble``).
        2. On the non-bagged (holdout) path, carve a single task-aware validation split off the
           training data (``_apply_task_specific_holdout``) and hand it to ``TabularPredictor`` as
           explicit ``tuning_data`` — a single model does not consume the bagged ``custom_splits``.
        3. If ``feature_generator_cls`` is given, instantiate it (forwarding any group/time
           split columns it accepts) into ``fit_kwargs["feature_generator"]``.
        4. Assemble ``train_data`` by appending the label column; attach tuning/validation
           data when provided.

        Returns:
        -------
        (train_data, init_kwargs, fit_kwargs)
            Ready to pass to ``TabularPredictor(**init_kwargs).fit(train_data, **fit_kwargs)``.
        """
        init_kwargs = copy.deepcopy(self.init_kwargs)
        fit_kwargs = copy.deepcopy(self.fit_kwargs)

        # Name the internal label column from the validation metadata's target (else a safe
        # sentinel), and tell the predictor about it.
        label = self.validation_metadata.target_name or "__label__"
        init_kwargs["label"] = label

        num_folds = self._apply_validation_splits(fit_kwargs, X=X, y=y)
        if X_val is None:
            X, y, X_val, y_val = self._apply_task_specific_holdout(X=X, y=y, num_folds=num_folds)
        self._apply_feature_generator(fit_kwargs)

        # TODO: think about if we can reset the index here without breaking simulation artifacts
        train_data = self._attach_label(X, y, label=label)
        if X_val is not None:
            fit_kwargs["tuning_data"] = self._attach_label(X_val, y_val, label=label)

        return train_data, init_kwargs, fit_kwargs

    def _apply_validation_splits(self, fit_kwargs: dict, *, X: pd.DataFrame, y: pd.Series) -> int | None:
        """Resolve the fold/repeat counts (+ any custom splits) into ``fit_kwargs`` in place.

        Pops the requested ``num_bag_folds`` / ``num_bag_sets``; when task-specific validation
        is enabled they run through ``resolve_validation_splits`` (which may adjust them and/or
        produce explicit ``custom_splits``), then are written back.

        Returns the effective ``num_folds`` — ``None`` (or ``<= 1``) signals the non-bagged
        holdout path, which ``_build_predictor_args`` then handles via a single task-aware split.
        """
        num_folds = fit_kwargs.pop("num_bag_folds", None)
        num_repeats = fit_kwargs.pop("num_bag_sets", None)

        custom_splits = None
        if self.use_task_specific_validation:
            custom_splits, num_folds, num_repeats = resolve_validation_splits(
                self.validation_metadata,
                X=X.reset_index(drop=True),
                y=y.reset_index(drop=True),
                num_folds=num_folds,
                num_repeats=num_repeats,
            )

        if num_folds is not None:
            logger.info(f"Using num_folds: {num_folds}")
            fit_kwargs["num_bag_folds"] = num_folds
        if num_repeats is not None:
            logger.info(f"Using num_repeats: {num_repeats}")
            fit_kwargs["num_bag_sets"] = num_repeats
        if custom_splits is not None:
            logger.info("Using custom_splits for validation protocol.")
            fit_kwargs.setdefault("ag_args_ensemble", {})["custom_splits"] = custom_splits

        return num_folds

    def _apply_task_specific_holdout(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        num_folds: int | None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame | None, pd.Series | None]:
        """Carve a single task-aware (group/temporal) validation split off the training data.

        Non-bagged counterpart of the bagged ``custom_splits`` path: a single ``TabularPredictor``
        fit does not consume ``ag_args_ensemble['custom_splits']`` (that is read only by the bagged
        ensemble), so the resolved holdout rows are returned as explicit ``X_val`` / ``y_val`` and
        fed to ``TabularPredictor`` as ``tuning_data`` instead.

        Only acts on the holdout path — task-specific validation enabled and no bagging
        (``num_folds`` is ``None`` / ``<= 1``). Otherwise, or when the task carries no
        grouped/temporal structure (``resolve_holdout_split`` returns ``None``), returns the data
        unchanged with ``X_val=None`` so AutoGluon's built-in holdout is used.

        Returns ``(X_train, y_train, X_val, y_val)``; rows keep their original index.
        """
        if not self.use_task_specific_validation or (num_folds is not None and num_folds > 1):
            return X, y, None, None

        split = resolve_holdout_split(
            self.validation_metadata,
            X=X.reset_index(drop=True),
            y=y.reset_index(drop=True),
        )
        if split is None:
            return X, y, None, None

        train_idx, val_idx = split
        logger.info(
            f"Using task-specific holdout split as tuning_data: {len(train_idx)} train / "
            f"{len(val_idx)} validation rows.",
        )
        # Return standalone copies (not ``.iloc`` views): ``_attach_label`` may set the label
        # column on these in place (``_can_use_data_in_place``), which would otherwise raise a
        # pandas ``SettingWithCopyWarning`` on a slice. The original index is preserved.
        return (
            X.iloc[train_idx].copy(),
            y.iloc[train_idx].copy(),
            X.iloc[val_idx].copy(),
            y.iloc[val_idx].copy(),
        )

    def _apply_feature_generator(self, fit_kwargs: dict) -> None:
        """Instantiate ``feature_generator_cls`` into ``fit_kwargs["feature_generator"]`` (in place).

        No-op when ``feature_generator_cls`` is absent. Forwards any group/time split columns
        from the validation metadata that the generator's ``__init__`` accepts, via the shared
        :func:`~tabarena.benchmark.preprocessing.build_feature_generator`.
        """
        feature_generator_cls = fit_kwargs.pop("feature_generator_cls", None)
        feature_generator_kwargs = fit_kwargs.pop("feature_generator_kwargs", {})
        if feature_generator_cls is None:
            return

        fit_kwargs["feature_generator"] = build_feature_generator(
            feature_generator_cls,
            feature_generator_kwargs,
            group_cols=self.validation_metadata.group_on,
            group_labels=self.validation_metadata.group_labels,
            group_time_on=self.validation_metadata.group_time_on,
        )

    def _attach_label(self, X: pd.DataFrame, y: pd.Series, *, label: str) -> pd.DataFrame:
        """Return ``X`` with ``y`` appended as the ``label`` column.

        Copies ``X`` first unless the data is owned by this object (``_can_use_data_in_place``).
        """
        data = X if self._can_use_data_in_place else X.copy()
        data[label] = y
        return data

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs,
    ):
        """Resolve the validation protocol, then construct and fit the ``TabularPredictor``."""
        from autogluon.tabular import TabularPredictor

        train_data, init_kwargs, fit_kwargs = self._build_predictor_args(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
        )

        self.predictor = TabularPredictor(
            problem_type=self.problem_type,
            eval_metric=self.eval_metric,
            **init_kwargs,
        )
        self.predictor.fit(
            train_data=train_data,
            **fit_kwargs,
        )

        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict labels with the fitted predictor (already-preprocessed ``X``)."""
        return self.predictor.predict(X)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict class probabilities with the fitted predictor (already-preprocessed ``X``)."""
        return self.predictor.predict_proba(X)

    def pre_predict(self):
        """Persist the best model in memory before inference when ``persist`` is enabled."""
        if self.persist:
            self.predictor.persist(models="best", max_memory=None)

    def post_predict(self):
        """Release any persisted model after inference when ``persist`` is enabled."""
        if self.persist:
            self.predictor.unpersist()

    def get_oof(self) -> dict:
        """Return the predictor's simulation artifact, narrowed to the best model's val proba."""
        # TODO: Rename method
        simulation_artifact = self.predictor.simulation_artifact()
        simulation_artifact["pred_proba_dict_val"] = simulation_artifact["pred_proba_dict_val"][
            self.predictor.model_best
        ]
        return simulation_artifact

    def get_metric_error_val(self) -> float:
        """Return the best model's validation metric error from the predictor leaderboard."""
        # FIXME: this shouldn't be calculating its own val score, that should be external. This should simply give val pred and val pred proba
        leaderboard = self.predictor.leaderboard(score_format="error", set_refit_score_to_parent=True)
        metric_error_val = leaderboard.set_index("model").loc[self.predictor.model_best]["metric_error_val"]
        if metric_error_val is not None and not np.isnan(metric_error_val):
            metric_error_val = float(metric_error_val)
        return metric_error_val

    def cleanup(self):
        """Delete the predictor's on-disk artifacts and free CPU/GPU memory."""
        shutil.rmtree(self.predictor.path, ignore_errors=True)
        gc.collect()
        try:
            import torch
        except ImportError:
            pass
        else:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class AGSingleWrapper(AGWrapper):
    """Fit a single AutoGluon model (no weighted ensemble) inside a ``TabularPredictor``.

    This is the common path for benchmarking one model family: it forces
    ``fit_weighted_ensemble=False`` and passes ``{model_cls: model_hyperparameters}`` as the
    predictor's ``hyperparameters``. Predictor/ensemble-level options that would conflict
    with fitting a single model are rejected up front (see ``_validate_fit_kwargs``);
    model-level options belong in ``model_hyperparameters``.

    Parameters
    ----------
    model_cls: str | type[AbstractModel]
        The model class (or its AutoGluon registry key) to fit, as used in
        ``predictor.fit(..., hyperparameters={model_cls: model_hyperparameters})``.
    model_hyperparameters: dict
        Hyperparameters for ``model_cls`` (including any ``ag_args_fit`` / ``ag_args_ensemble``).
    calibrate: bool | str, default False
        Forwarded to ``TabularPredictor.fit(calibrate=...)``.
    init_kwargs, fit_kwargs:
        Extra predictor constructor / fit kwargs (the "extra" kwargs recorded in metadata).
    """

    def __init__(
        self,
        model_cls: str | type[AbstractModel],
        model_hyperparameters: dict,
        calibrate: bool | str = False,
        init_kwargs: dict | None = None,
        fit_kwargs: dict | None = None,
        **kwargs,
    ):
        assert isinstance(model_cls, str) or issubclass(model_cls, AbstractModel)
        assert isinstance(model_hyperparameters, dict)

        if fit_kwargs is None:
            fit_kwargs = {}
        if init_kwargs is None:
            init_kwargs = {}
        self._validate_fit_kwargs(fit_kwargs)

        # Record the user-provided "extra" kwargs (used for metadata), then derive the
        # effective fit kwargs by forcing the single-model contract on top of them.
        self.init_kwargs_extra = init_kwargs

        fit_kwargs = copy.deepcopy(fit_kwargs)
        fit_kwargs["calibrate"] = calibrate
        self.fit_kwargs_extra = fit_kwargs

        fit_kwargs = copy.deepcopy(fit_kwargs)
        fit_kwargs["fit_weighted_ensemble"] = False
        fit_kwargs["hyperparameters"] = {model_cls: model_hyperparameters}

        self._model_cls = model_cls
        self.model_hyperparameters = model_hyperparameters

        super().__init__(
            init_kwargs=init_kwargs,
            fit_kwargs=fit_kwargs,
            **kwargs,
        )

    @staticmethod
    def _validate_fit_kwargs(fit_kwargs: dict) -> None:
        """Reject ``fit_kwargs`` incompatible with fitting a single model.

        Options interpreted at the predictor/ensemble level (``presets``,
        ``num_stack_levels``, ``fit_weighted_ensemble``, ...) or with a dedicated wrapper
        argument (``calibrate``) must not be passed here; model-level options such as
        ``ag_args_fit`` / ``ag_args_ensemble`` belong in ``model_hyperparameters``.
        """
        disallowed = {
            "hyperparameters": "Must not specify `hyperparameters` in AGSingleWrapper.",
            "num_stack_levels": "num_stack_levels is not allowed for AGSingleWrapper.",
            "presets": "AGSingleWrapper does not support `presets`.",
            "fit_weighted_ensemble": (
                "Must not specify `fit_weighted_ensemble` in AGSingleWrapper... It is always set to False."
            ),
            "calibrate": "Specify calibrate directly rather than in `fit_kwargs`.",
            "ag_args_fit": "ag_args_fit must be specified in `model_hyperparameters`, not in `fit_kwargs`.",
            "ag_args_ensemble": "ag_args_ensemble must be specified in `model_hyperparameters`, not in `fit_kwargs`.",
        }
        for key, message in disallowed.items():
            assert key not in fit_kwargs, message

    def post_fit(self, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame):
        """Capture any model fit failures so the runner can record them on a crash."""
        self.failure_artifact = self.get_metadata_failure()

    def get_hyperparameters(self) -> dict:
        """Return the best model's hyperparameters in user-facing form."""
        return self.predictor.model_hyperparameters(model=self.predictor.model_best, output_format="user")

    @property
    def model_cls(self) -> type[AbstractModel]:
        """The model class, resolving an AutoGluon registry key string when needed."""
        if not isinstance(self._model_cls, str):
            model_cls = self._model_cls
        else:
            # TODO: Get it from predictor instead? What if we allow passing custom model registry?
            from autogluon.tabular.registry import (
                ag_model_registry,  # If this raises an exception, you need to update to latest mainline AutoGluon
            )

            model_cls = ag_model_registry.key_to_cls(key=self._model_cls)
        return model_cls

    def _load_model(self, assert_single_model: bool = True):
        """Load the fitted model object from the predictor's trainer.

        When ``assert_single_model`` is True, assert exactly one inferable model exists and
        load it; otherwise load the predictor's ``model_best``.
        """
        model_names = self.predictor.model_names(can_infer=True)
        if assert_single_model:
            assert len(model_names) == 1
            model_name = self.predictor.model_names()[0]
        else:
            model_name = self.predictor.model_best
        return self.predictor._trainer.load_model(model_name)

    def get_metadata_init(self) -> dict:
        """Metadata known at construction time (model class, hyperparameters, extra kwargs)."""
        metadata = {}
        metadata["hyperparameters"] = self.get_hyperparameters()
        metadata["model_cls"] = self.model_cls.__name__
        metadata["model_type"] = self.model_cls.ag_key
        metadata["name_prefix"] = self.model_cls.ag_name
        metadata["model_hyperparameters"] = self.model_hyperparameters
        metadata["init_kwargs_extra"] = self.init_kwargs_extra
        metadata["fit_kwargs_extra"] = self.fit_kwargs_extra
        return metadata

    def get_metadata_fit(self) -> dict:
        """Metadata available only after fitting (info, disk/compute usage, fit metadata)."""
        metadata = {}
        model = self._load_model(assert_single_model=False)
        metadata["info"] = model.get_info(include_feature_metadata=False)
        metadata["disk_usage"] = model.disk_usage()
        metadata["num_cpus"] = model.fit_num_cpus
        metadata["num_gpus"] = model.fit_num_gpus
        metadata["num_cpus_child"] = model.fit_num_cpus_child
        metadata["num_gpus_child"] = model.fit_num_gpus_child
        metadata["fit_metadata"] = model.get_fit_metadata()
        if hasattr(model, "_memory_usage_estimate"):
            metadata["memory_usage_estimate"] = model._memory_usage_estimate
        return metadata

    def get_metadata_failure(self) -> dict:
        """Record any per-model fit failures reported by the predictor."""
        return {
            "model_failures": self.predictor.model_failures(),
        }

    def get_metadata(self) -> dict:
        """Combined construction-time and post-fit metadata for this model."""
        metadata = self.get_metadata_init()
        metadata_fit = self.get_metadata_fit()

        metadata.update(metadata_fit)
        return metadata


class AGSingleBagWrapper(AGSingleWrapper):
    """A bagged ``AGSingleWrapper`` that also exposes its per-child (per-fold) artifacts.

    Identical fitting to ``AGSingleWrapper``, but advertises and provides the per-bagged-child
    out-of-fold validation indices and test predictions needed for ensemble simulation.
    """

    # Bagging exposes per-child OOF predictions and their validation indices.
    can_get_per_child_oof = True
    can_get_per_child_val_idx = True

    def bag_artifact(self, X_test: pd.DataFrame) -> dict:
        """Collect per-child test predictions and validation indices for the bagged model."""
        model = self._load_model()
        bag_info = {}
        bag_info["pred_proba_test_per_child"] = self.get_per_child_test(X_test=X_test, model=model)
        bag_info["val_idx_per_child"] = self.get_per_child_val_idx(model=model)
        return bag_info

    def get_per_child_val_idx(self, model=None) -> list[np.ndarray]:
        """Return each child's out-of-fold validation indices (into the internal train data)."""
        if model is None:
            model = self._load_model()
        X, y = self.predictor.load_data_internal()
        all_kfolds = []
        # TODO: Make this a bagged ensemble method
        if model._child_oof:
            all_kfolds = [(None, X.index.values)]
        else:
            for n_repeat, k in enumerate(model._k_per_n_repeat):
                kfolds = model._cv_splitters[n_repeat].split(X=X, y=y)
                cur_kfolds = kfolds[n_repeat * k : (n_repeat + 1) * k]
                all_kfolds += cur_kfolds

        val_idx_per_child = []
        for _fold_idx, (_train_idx, val_idx) in enumerate(all_kfolds):
            val_idx = pd.to_numeric(val_idx, downcast="integer")  # memory opt
            val_idx_per_child.append(val_idx)

        return val_idx_per_child

    # TODO: Can avoid predicting on test twice by doing it all in one go
    def get_per_child_test(self, X_test: pd.DataFrame, model=None) -> list[np.ndarray]:
        """Return each child's predictions on ``X_test`` (float32), in the original row order.

        Applies the same deterministic test-row shuffle as inference (see
        ``_shuffle_test_rows``) and inverts it on the per-child outputs.
        """
        X_test, inv_perm, original_index = self._shuffle_test_rows(X_test)

        X_test = self.transform_X(X=X_test)

        if model is None:
            model = self._load_model()
        X_test_inner = self.predictor.transform_features(data=X_test, model=model.name)

        if model.can_predict_proba():
            per_child_test_preds = model.predict_proba_children(X=X_test_inner)
        else:
            per_child_test_preds = model.predict_children(X=X_test_inner)

        if self.shuffle_test:
            # Inverse-permute outputs back to original X_test order
            per_child_test_preds = [
                _apply_inv_perm(y_pred, inv_perm, index=original_index) for y_pred in per_child_test_preds
            ]

        return [preds_child.astype(np.float32) for preds_child in per_child_test_preds]  # memory opt


class AGModelWrapper(AbstractExecModel):
    """Fit a single AutoGluon model directly, bypassing ``TabularPredictor``.

    Instantiates ``model_cls`` and calls its ``fit`` on all of ``X``/``y`` (no train/val
    split, no bagging, no ensemble). Used to benchmark methods that want to train on the
    full data (e.g. via ``AGModelOuterExperiment``). Unlike ``AGWrapper`` this does not
    carry the validation protocol and provides no OOF / metadata capabilities.

    Preprocessing is shared with the validation path via a named ``preprocessing_pipeline``
    (see :func:`~tabarena.benchmark.preprocessing.resolve_preprocessing_pipeline`): its
    model-agnostic feature generator is applied through ``AbstractExecModel``'s
    ``preprocess_data`` path (``_make_feature_generator``), and its model-specific step is
    injected into ``hyperparameters`` (which the AutoGluon model applies in its own ``fit``).
    ``None`` / ``"default"`` keeps AutoGluon's standard ``AutoMLPipelineFeatureGenerator``;
    ``"tabarena_default"`` uses the TabArena pipeline — functionally the same preprocessing as
    an ``AGWrapper`` configured with the same pipeline.

    Parameters
    ----------
    model_cls: type[AbstractModel]
        AutoGluon model class to fit.
    hyperparameters: dict, optional
        Model hyperparameters; the resolved pipeline's model-specific step is merged in.
    preprocessing_pipeline: str | None, default None
        Pipeline name passed to ``resolve_preprocessing_pipeline``.
    group_cols / group_labels / group_time_on:
        Optional grouped-task columns forwarded to the model-agnostic feature generator (only
        those it accepts); analogous to what ``AGWrapper`` sources from validation metadata.
    """

    model: AbstractModel
    """The fitted single AutoGluon model (set by ``_fit``)."""

    def __init__(
        self,
        model_cls: type[AbstractModel],
        hyperparameters: dict | None = None,
        *,
        fit_kwargs: dict | None = None,
        preprocessing_pipeline: str | None = None,
        group_cols: str | list[str] | None = None,
        group_labels=None,
        group_time_on: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert issubclass(model_cls, AbstractModel)
        self.model_cls = model_cls
        if hyperparameters is None:
            hyperparameters = {}
        # Passed straight to the model's `fit` (e.g. num_cpus / num_gpus / time_limit). The
        # AutoGluon model applies/ignores each as appropriate.
        self.fit_kwargs = fit_kwargs or {}

        pipeline = resolve_preprocessing_pipeline(preprocessing_pipeline)
        self.preprocessing_pipeline = preprocessing_pipeline
        self._feature_generator_cls = pipeline.feature_generator_cls
        self._feature_generator_kwargs = pipeline.feature_generator_kwargs
        self._group_cols = group_cols
        self._group_labels = group_labels
        self._group_time_on = group_time_on
        # Model-specific preprocessing rides on the model's hyperparameters (the AutoGluon model
        # applies it in its own fit), so this works the same here as in the AGWrapper path.
        self.hyperparameters = pipeline.apply_model_specific(hyperparameters)

    def _make_feature_generator(self):
        """Build the pipeline's model-agnostic feature generator (shared with ``AGWrapper``)."""
        return build_feature_generator(
            self._feature_generator_cls,
            self._feature_generator_kwargs,
            group_cols=self._group_cols,
            group_labels=self._group_labels,
            group_time_on=self._group_time_on,
        )

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Instantiate ``model_cls`` and fit it directly on the (preprocessed) data."""
        self.model = self.model_cls(
            path="",
            name=self.model_cls.__name__,
            problem_type=self.problem_type,
            eval_metric=self.eval_metric,
            hyperparameters=self.hyperparameters,
        )
        self.model.fit(
            X=X,
            y=y,
            **self.fit_kwargs,
        )
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict labels with the fitted model, preserving ``X``'s index."""
        y_pred = self.model.predict(X)
        return pd.Series(y_pred, index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict class probabilities, widening binary output to two columns."""
        y_pred_proba = self.model.predict_proba(X)
        if self.problem_type == "binary":
            y_pred_proba = LabelCleanerMulticlassToBinary.convert_binary_proba_to_multiclass_proba(y_pred_proba)
        return pd.DataFrame(y_pred_proba, index=X.index)
