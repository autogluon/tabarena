"""Plain-function helpers for the AutoGluon exec-model wrappers.

Two groups live here. The validation-split logic lets the wrappers adapt their internal
validation splitting to a task's structure (group-wise / time-based splits) and to the
experiment's :class:`~tabarena.benchmark.validation_protocol.ValidationProtocol` (fold and
repeat counts, the tiny-data regime), keyed off a
:class:`~tabarena.benchmark.task.metadata.ValidationMetadata` (the task-derived split structure)
and the protocol, rather than a mixin on the wrapper. Its two entry points are
:func:`resolve_validation_splits` for the bagged path (``k`` cross-validation folds) and
:func:`resolve_holdout_split` for the non-bagged path (a single train/validation split).

The second group resolves what a ``TabularPredictor.fit(**fit_kwargs)`` call will train before any
data is seen: :func:`resolve_effective_fit_kwargs` applies AutoGluon's presets the way the
predictor does, :func:`iter_configured_model_classes` and :func:`configured_model_classes` turn a
``hyperparameters`` argument (a dict, a named config or presets) into model classes, and
:func:`preset_uses_bagging` reads the bagging decision. The warm-up uses them to know which model
classes to warm; the result is never passed to ``fit``, so AutoGluon still resolves presets itself.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from tabarena.benchmark.task.metadata import GroupLabelTypes

if TYPE_CHECKING:
    from collections.abc import Iterator

    from autogluon.common.utils.validation_structure import ValidationStructure
    from autogluon.core.models import AbstractModel

    from tabarena.benchmark.task.metadata import ValidationMetadata
    from tabarena.benchmark.validation_protocol import ValidationProtocol


# --- Configured model classes (what a predictor fit will train) ---------------------------


def resolve_model_cls(model_cls: str | type) -> type[AbstractModel]:
    """The AutoGluon model class for ``model_cls``, resolving a registry key string when needed.

    Raises whatever the registry raises for an unknown key; callers that resolve best effort catch
    that themselves.
    """
    if not isinstance(model_cls, str):
        return model_cls
    from autogluon.tabular.registry import ag_model_registry

    return ag_model_registry.key_to_cls(key=model_cls)


def resolve_effective_fit_kwargs(fit_kwargs: dict | None) -> dict:
    """The ``TabularPredictor.fit`` keyword arguments after AutoGluon's presets are applied.

    Uses ``autogluon.common.utils.decorators._apply_presets`` with the tabular preset tables, the
    same code path ``TabularPredictor.fit`` runs, so explicit arguments win over preset values and
    presets are applied first to last. A ``hyperparameters`` value that is ``None`` resolves to the
    ``"default"`` hyperparameter config and a config name resolves to its dict, both deep-copied.
    Works on a shallow copy; the caller's ``fit_kwargs`` is never modified. On any failure inside
    AutoGluon the presets are left unapplied and a copy of the input is returned (best effort).
    """
    kwargs = dict(fit_kwargs or {})
    try:
        from autogluon.common.utils.decorators import _apply_presets
        from autogluon.tabular.configs.presets_configs import tabular_presets_alias, tabular_presets_dict

        _args, kwargs = _apply_presets(tabular_presets_dict, tabular_presets_alias, **kwargs)
        hyperparameters = kwargs.get("hyperparameters")
        if hyperparameters is None or isinstance(hyperparameters, str):
            from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

            kwargs["hyperparameters"] = get_hyperparameter_config(hyperparameters or "default")
    except Exception as exc:
        logger.debug(f"Could not resolve the effective fit kwargs ({exc!r}); using them as given.")
    return kwargs


def iter_configured_model_classes(
    hyperparameters: dict | str | None,
) -> Iterator[tuple[type[AbstractModel], dict | None]]:
    """Yield ``(model_cls, hyperparameters)`` for every model a ``hyperparameters`` argument configures.

    Accepts the dict form (``{key_or_class: config | [configs]}``), the stack-level form
    (``{1: {...}, "default": {...}}``), a hyperparameter config name (resolved through
    ``get_hyperparameter_config``) or ``None`` (nothing). String keys are resolved via the AutoGluon
    registry; unresolvable entries are skipped with a debug log because the consumers are best
    effort. A key configured with a list of configs yields the first as representative; the yielded
    config is ``None`` when it is not a dict. Row-count gates (``ag.min_rows`` / ``ag.max_rows``) are
    ignored because they depend on the data.
    """
    if hyperparameters is None:
        return
    if isinstance(hyperparameters, str):
        try:
            from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

            hyperparameters = get_hyperparameter_config(hyperparameters)
        except Exception as exc:
            logger.debug(f"Skipping unresolvable hyperparameter config {hyperparameters!r} ({exc})")
            return
    if not isinstance(hyperparameters, dict):
        return
    seen: set[type] = set()
    for key, configs in hyperparameters.items():
        if isinstance(configs, dict) and (isinstance(key, int) or key == "default") and not isinstance(key, type):
            # Stack-level form: the value is itself a {model: config} mapping.
            for model_cls, config in iter_configured_model_classes(configs):
                if model_cls not in seen:
                    seen.add(model_cls)
                    yield model_cls, config
            continue
        try:
            model_cls = resolve_model_cls(key)
        except Exception as exc:
            logger.debug(f"Skipping unresolvable model key {key!r} ({exc})")
            continue
        from autogluon.core.models import AbstractModel

        if not (isinstance(model_cls, type) and issubclass(model_cls, AbstractModel)) or model_cls in seen:
            continue
        seen.add(model_cls)
        config = configs[0] if isinstance(configs, list) and configs else configs
        yield model_cls, copy.deepcopy(config) if isinstance(config, dict) else None


def configured_model_classes(fit_kwargs: dict | None) -> list[tuple[type[AbstractModel], dict | None]]:
    """``(model_cls, hyperparameters)`` for every model a ``TabularPredictor.fit(**fit_kwargs)`` trains.

    Presets, config names and ``None`` are resolved through :func:`resolve_effective_fit_kwargs`
    first, so a preset-driven fit resolves to its model classes as well. Best effort, for warm-up
    and audits only.
    """
    effective = resolve_effective_fit_kwargs(fit_kwargs)
    return list(iter_configured_model_classes(effective.get("hyperparameters")))


def preset_uses_bagging(fit_kwargs: dict | None) -> bool:
    """Whether the effective fit kwargs configure a bagged fit (``num_bag_folds > 1`` or ``auto_stack``)."""
    effective = resolve_effective_fit_kwargs(fit_kwargs)
    num_bag_folds = effective.get("num_bag_folds")
    if isinstance(num_bag_folds, int | float) and num_bag_folds > 1:
        return True
    return bool(effective.get("auto_stack"))


# --- Validation splits ------------------------------------------------------------------


#: ``data_foundry``'s split seed (its ``SPLIT_RANDOM_STATE``, a local constant of its split builders), which TabArena's
#: resolved splits use. AutoGluon's
#: learner seeds its structure-aware splits from its own ``random_state`` (default 0), so a fit that lets
#: AutoGluon build the splits reproduces TabArena's folds only with this seed.
SPLIT_RANDOM_STATE = 4267


def validation_structure_from_metadata(
    metadata: ValidationMetadata,
    *,
    temporal_forward_only: bool = False,
    size_validation_on_groups: bool | None = None,
) -> ValidationStructure | None:
    """The AutoGluon :class:`ValidationStructure` a task's validation metadata declares, or None.

    The one projection for every path that lets AutoGluon resolve a task's validation splits itself
    (``TabularPredictor.fit(validation_structure=...)``) rather than receiving TabArena's resolved
    ``custom_splits`` / ``tuning_data``, so all of them declare the same structure for the same task.

    ``None`` for a task with no grouped or temporal structure, *even when it declares ``stratify_on``*.
    Such a task has no leakage to prevent, and AutoGluon's own bagging already stratifies classification
    folds by the label, which is what these tasks stratify on. Declaring a structure anyway would take
    over splitting only to produce equally valid but differently seeded folds; ``resolve_validation_splits``
    makes the same choice for the TabArena-resolved path. Only ``group_on`` and ``time_on`` therefore
    decide, and the metadata never sets both.

    ``group_time_on`` is not part of the structure. In TabArena it is the time *within* a group, read
    only by the group-aware feature generator to order rows inside a group for its "last" aggregates;
    it never takes part in splitting. AutoGluon's ``ValidationStructure`` has no such field.

    ``size_validation_on_groups`` decides what AutoGluon's size-dependent choices (fold counts, holdout
    size, every ``validation_size_curves`` entry) are measured against. ``None`` applies TabArena's rule
    for config runs: group instances exactly when the labels are per group, rows otherwise. A system whose
    configuration is written in training rows passes ``False``, or a 4,400-row task with 90 groups would
    be sized as a 90-row one.

    ``temporal_forward_only`` asks for forward-chaining temporal validation on a ``time_on`` task (fold
    *i* validates block *i+1* and trains only on earlier blocks); it has no effect otherwise.
    """
    from autogluon.common.utils.validation_structure import ValidationStructure

    if metadata.group_on is None and metadata.time_on is None:
        return None
    if size_validation_on_groups is None:
        size_validation_on_groups = metadata.group_labels == GroupLabelTypes.PER_GROUP
    return ValidationStructure(
        group_on=metadata.group_on,
        time_on=metadata.time_on,
        stratify_on=metadata.stratify_on,
        temporal_forward_only=temporal_forward_only,
        size_validation_on_groups=size_validation_on_groups,
    )


def resolve_validation_splits(
    metadata: ValidationMetadata,
    protocol: ValidationProtocol,
    *,
    X: pd.DataFrame,
    y: pd.Series,
    clamps: list[str] | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]] | None, int, int]:
    """Resolve the bagging folds / repeats and, if the task needs them, the custom splits.

    Used for task-specific validation (``protocol.task_specific_validation``): the protocol's counts
    are resolved for the training split's (group) instance count, then reduced where the task's
    structure demands it (a temporal task fits one repeat over at most as many folds as there are
    time intervals; a grouped task fits at most as many folds as groups; a stratified task at most as
    many folds as its minority class has samples). The reductions are appended to ``clamps`` when a
    list is given, so callers can record them.

    Returns:
    -------
    custom_splits: list[tuple[np.ndarray, np.ndarray]] or None
        None, if no custom splits are needed.
        Otherwise, a list of tuples of train/test indices (as np.ndarrays) to use
        for validation splitting.
        IMPORTANT: the split will return the index of the input data X!
    num_folds: int
        The number of folds to use for validation.
    num_repeats: int
        The number of repeats to use for validation.
    """
    custom_splits = None
    if clamps is None:
        clamps = []

    num_group_instances = get_num_group_instances(metadata, X=X)
    logger.info(
        "\n=== Using task-specific validation logic!"
        f"\n\tNumber of groups/instances in data: {num_group_instances}"
        f"\n\tStratify_on: {metadata.stratify_on}"
        f"\n\tGroup_on: {metadata.group_on}"
        f"\n\tTime_on: {metadata.time_on}"
        f"\n\tGroup_time_on: {metadata.group_time_on}"
        f"\n\tGroup_labels: {metadata.group_labels}"
        f"\n\tSplit_time_horizon: {metadata.split_time_horizon}"
        f"\n\tSplit_time_horizon_unit: {metadata.split_time_horizon_unit}",
    )

    num_folds, num_repeats = protocol.resolve_num_splits(num_group_instances)
    logger.info(f"\n\tProtocol {protocol.describe()}: num_bag_folds={num_folds}, num_bag_sets={num_repeats}.")

    stratify_on_data = None
    if metadata.stratify_on is not None:
        stratify_on_data = X[metadata.stratify_on] if metadata.stratify_on in X.columns else y
        # Enforce categorical dtype for stratification column, as some splitting logic relies on it.
        stratify_on_data = stratify_on_data.astype("category")

    groups_data = None
    group_labels = None
    if (metadata.time_on is not None) and (metadata.group_on is not None):
        raise NotImplementedError

    if metadata.time_on is not None:
        groups_data, num_folds_new = time_on_to_groups_data(X=X, time_on=metadata.time_on, num_folds=num_folds)
        if num_repeats != 1:
            clamps.append("time_on_single_repeat")
        num_repeats = 1
        logger.info(
            f"\n\tFolds time-based grouping: before={num_folds}; after={num_folds_new}\n\tnum_repeats set to 1!",
        )
        if num_folds_new != num_folds:
            clamps.append("folds_capped_by_time_intervals")
        num_folds = num_folds_new
        # Set group labels as needed for time split
        group_labels = GroupLabelTypes.PER_SAMPLE

    if metadata.group_on is not None:
        groups_data = group_on_to_groups_data(X=X, group_on=metadata.group_on)
        group_labels = metadata.group_labels

    if groups_data is not None:
        n_groups = groups_data.nunique()
        if n_groups < num_folds:
            logger.info(
                f"Number of unique groups in the data ({n_groups}) is less than the "
                f"number of folds ({num_folds})! Adjusting the number of folds to be equal to the number of "
                f"unique groups, and setting num_repeats to 1.",
            )
            num_folds = n_groups
            num_repeats = 1
            clamps.append("folds_capped_by_n_groups")

        if stratify_on_data is not None:
            n_samples_minority_class = int(stratify_on_data.value_counts().min())
            if n_samples_minority_class < num_folds:
                logger.warning(
                    f"Number of samples in minority class for stratification ({n_samples_minority_class}) is less "
                    f"than the number of folds ({num_folds})! We set num_folds to be equal to the number of samples"
                    " in the minority class, and num_repeats to 1.",
                )
                num_folds = n_samples_minority_class
                num_repeats = 1
                clamps.append("folds_capped_by_minority_class")

        custom_splits = _resolve_group_splits(
            metadata=metadata,
            X=X,
            num_folds=num_folds,
            num_repeats=num_repeats,
            stratify_on_data=stratify_on_data,
            groups_data=groups_data,
            group_labels=group_labels,
        )

    # Sanity checks for custom splits
    if custom_splits is not None:
        _validate_stratified_splits(custom_splits, stratify_on_data=stratify_on_data)

    return custom_splits, num_folds, num_repeats


def resolve_holdout_split(
    metadata: ValidationMetadata,
    protocol: ValidationProtocol,
    *,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build a single task-aware holdout ``(train_idx, val_idx)`` split.

    The single-split counterpart of :func:`resolve_validation_splits` for the non-bagged
    (holdout) path. Instead of ``k`` bagging folds it returns ONE train/validation split that
    still respects the task's structure:

    - ``group_on``: a group-disjoint split (no group spans train and validation), optionally
      stratified, via ``data_foundry``'s single grouped split.
    - ``time_on``: a *forward* holdout: the latest contiguous block of time (about one fold's worth)
      is held out for validation and everything earlier is used for training. This mirrors the
      benchmark's temporal gap between train and test (the test set is the latest time horizon); a
      random fold would instead leak future information into training.
    - neither (plain IID, with or without stratification): returns ``None``, and the caller leaves
      AutoGluon's built-in (label-stratified) holdout untouched, as there is no leakage to fix.

    The validation size targets ``1 / num_folds`` of the data, with ``num_folds`` resolved by the
    protocol for the training split's size (:meth:`ValidationProtocol.resolve_num_splits`), so a
    holdout validation set is sized like a single bagging fold of the same protocol. Indices are
    positional into the given (reset-index) ``X``, assumed to be a ``RangeIndex`` (the wrapper resets
    it before calling).

    Used for task-specific validation (``protocol.task_specific_validation``) on the non-bagged path.
    """
    if (metadata.time_on is not None) and (metadata.group_on is not None):
        raise NotImplementedError

    if (metadata.time_on is None) and (metadata.group_on is None):
        # No grouped/temporal structure to honor; AutoGluon's default holdout is appropriate.
        return None

    num_group_instances = get_num_group_instances(metadata, X=X)
    num_folds, _ = protocol.resolve_num_splits(num_group_instances)
    val_size = max(1, round(len(X) / num_folds))
    logger.info(
        "\n=== Building task-specific holdout split!"
        f"\n\tRows: {len(X)}; target validation rows (~1/{num_folds}): {val_size}"
        f"\n\tStratify_on: {metadata.stratify_on}"
        f"\n\tGroup_on: {metadata.group_on}"
        f"\n\tTime_on: {metadata.time_on}",
    )

    if metadata.time_on is not None:
        # Forward holdout: hold out the latest time block. Stratification is not meaningful across
        # a temporal cut, so it is intentionally not applied here.
        stratify_on_data = None
        train_idx, val_idx = _resolve_temporal_holdout_split(X=X, time_on=metadata.time_on, num_folds=num_folds)
    else:
        stratify_on_data = None
        if metadata.stratify_on is not None:
            stratify_on_data = X[metadata.stratify_on] if metadata.stratify_on in X.columns else y
            # Enforce categorical dtype for stratification column, as some splitting logic relies on it.
            stratify_on_data = stratify_on_data.astype("category")
        train_idx, val_idx = _resolve_group_holdout_split(
            metadata=metadata,
            X=X,
            val_size=val_size,
            stratify_on_data=stratify_on_data,
        )

    _validate_stratified_splits([(train_idx, val_idx)], stratify_on_data=stratify_on_data)
    return train_idx, val_idx


def _resolve_temporal_holdout_split(
    *,
    X: pd.DataFrame,
    time_on: str,
    num_folds: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward holdout: hold out the latest time interval, train on all earlier rows.

    Reuses :func:`time_on_to_groups_data` to bucket rows into contiguous, time-ordered intervals
    (larger label == later in time), then validates on the latest interval. ``num_folds`` sets the
    number of intervals, so the latest one is ~``1 / num_folds`` of the rows.
    """
    groups_data, _n_intervals = time_on_to_groups_data(X=X, time_on=time_on, num_folds=num_folds)
    latest_interval = int(groups_data.max())
    is_val = (groups_data == latest_interval).to_numpy()
    positions = np.arange(len(X))
    return positions[~is_val], positions[is_val]


def _resolve_group_holdout_split(
    metadata: ValidationMetadata,
    *,
    X: pd.DataFrame,
    val_size: int,
    stratify_on_data: pd.Series | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Single group-disjoint holdout split (optionally stratified) for the ``group_on`` column(s).

    The single-split analogue of :func:`_resolve_group_splits`: ``data_foundry``'s
    ``get_recommended_grouped_splits`` is asked for one train/test split of approximately
    ``val_size`` validation rows, honoring ``group_labels`` and any stratification column.
    """
    from data_foundry.curation_recommendations import get_recommended_grouped_splits

    groups_data = group_on_to_groups_data(X=X, group_on=metadata.group_on)
    assert not groups_data.isna().any(), (
        "Group column(s) contain NaN values, which is not allowed for group-wise splitting!"
    )
    if metadata.group_labels not in [
        GroupLabelTypes.PER_GROUP,
        GroupLabelTypes.PER_SAMPLE,
    ]:
        raise ValueError(f"Invalid group_labels value: {metadata.group_labels}")

    splits_data = pd.DataFrame({"group": groups_data.to_numpy()})
    stratify_on = None
    if stratify_on_data is not None:
        splits_data["stratify"] = stratify_on_data.to_numpy()
        stratify_on = "stratify"

    splits = get_recommended_grouped_splits(
        dataset=splits_data,
        group_on="group",
        stratify_on=stratify_on,
        n_splits=1,
        n_repeats=1,
        test_size=val_size,
        group_labels=metadata.group_labels,
    )
    train_idx, val_idx = _first_split(splits)
    return np.asarray(train_idx), np.asarray(val_idx)


def _first_split(splits: dict) -> tuple[list[int], list[int]]:
    """Extract the single ``(train_idx, test_idx)`` pair from a ``{repeat: {fold: split}}`` dict."""
    first_repeat = next(iter(splits.values()))
    return next(iter(first_repeat.values()))


def _validate_stratified_splits(
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    stratify_on_data: pd.Series | None,
) -> None:
    """Sanity-check each ``(train_idx, test_idx)`` split: non-empty, and (if stratifying) coverage.

    When ``stratify_on_data`` is given, asserts the train split carries every stratification value
    and the test split's values are a subset of the train's, raising for the binary case (metrics
    need both classes) and warning for the multiclass case when a value is missing from test.
    Shared by :func:`resolve_validation_splits` (bagging folds) and :func:`resolve_holdout_split`
    (single holdout). Indices are positional into ``stratify_on_data``.
    """
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0, "Train split is empty!"
        assert len(test_idx) > 0, "Test split is empty!"

        if stratify_on_data is None:
            continue

        stratify_values = set(stratify_on_data.unique())
        train_stratify_values = set(stratify_on_data.iloc[train_idx].unique())
        test_stratify_values = set(stratify_on_data.iloc[test_idx].unique())

        assert train_stratify_values == stratify_values, (
            "[Missing Train Stratification Values] "
            "Stratification values in train split do not match overall stratification values!"
            f"\n\tOverall stratification values: {stratify_values}"
            f"\n\tTrain stratification values: {train_stratify_values}"
        )
        assert test_stratify_values.issubset(train_stratify_values), (
            "[Unseen Test Stratification Values] "
            "Stratification values in test split are not a subset of train stratification values!"
            f"\n\tTrain stratification values: {train_stratify_values}"
            f"\n\tTest stratification values: {test_stratify_values}"
        )

        if train_stratify_values != stratify_values:
            # Check if test has all labels for binary, as metrics require it.
            if len(stratify_values) == 2:
                raise ValueError(
                    "[Binary Metric Missing Stratification Values in Test] "
                    "Stratification values in train and test splits do not match!"
                    f"\n\tOverall stratification values: {stratify_values}"
                    f"\n\tTrain stratification values: {train_stratify_values}"
                    f"\n\tTest stratification values: {test_stratify_values}",
                )

            # For multi-stratify values, we do not allow missing a stratify value in the test split
            logger.warning(
                "[Stratification Value Missing in Test Data] "
                "Stratification values in train and test splits are not identical."
                "This means the validation data is likely missing some classes."
                f"\n\tOverall stratification values: {stratify_values}"
                f"\n\tTrain stratification values: {train_stratify_values}"
                f"\n\tTest stratification values: {test_stratify_values}",
            )


def _resolve_group_splits(
    metadata: ValidationMetadata,
    *,
    X: pd.DataFrame,
    num_folds: int,
    num_repeats: int,
    stratify_on_data: pd.Series | None,
    groups_data: pd.Series,
    group_labels: GroupLabelTypes | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create a group-based split given the specified group_on column(s).
    Then transform this into split indices for AutoGluon's group splitter logic.

    Some comments about this logic:
        - If we are given a list of group_on columns, we create a combined group
            identifier by concatenating the values in those columns.
        - We use stratification if stratify_on is specified.
        - We dynamically adjust the number of splits to be the minimum of
            and the number of unique groups in the data.
    """
    from data_foundry.curation_recommendations import get_recommended_grouped_splits

    assert not groups_data.isna().any(), (
        "Group column(s) contain NaN values, which is not allowed for group-wise splitting!"
    )

    if group_labels not in [
        GroupLabelTypes.PER_GROUP,
        GroupLabelTypes.PER_SAMPLE,
    ]:
        raise ValueError(f"Invalid group_labels value: {group_labels}")

    if group_labels == GroupLabelTypes.PER_GROUP:
        n_groups_in_data = groups_data.nunique()
        assert n_groups_in_data > 1, (
            f"Need at least 2 unique groups for group-wise splitting, but got {n_groups_in_data} unique groups from column(s)!"
        )
        num_folds = min(num_folds, n_groups_in_data)

    # Build dummy split data object
    splits_data = pd.Series(np.zeros(len(X)), index=X.index).to_frame()
    splits_data["group"] = groups_data
    stratify_on = None
    if metadata.stratify_on is not None:
        splits_data["stratify"] = stratify_on_data
        stratify_on = "stratify"
    splits_data = splits_data.reset_index(drop=True)

    custom_splits = get_recommended_grouped_splits(
        dataset=splits_data,
        group_on="group",
        stratify_on=stratify_on,
        n_splits=num_folds,
        n_repeats=num_repeats,
        test_size=None,
        group_labels=group_labels,
    )
    del splits_data

    custom_splits_for_index = []
    for repeat_splits in custom_splits.values():
        for train_idx, test_idx in repeat_splits.values():
            custom_splits_for_index.append((train_idx, test_idx))
    return custom_splits_for_index


def group_on_to_groups_data(*, X: pd.DataFrame, group_on: str | list[str]) -> pd.Series:
    """Groups on to a unique group column."""
    # Get group label
    if isinstance(group_on, list):
        # If multiple columns are specified, create a combined group identifier
        groups_data = X[group_on].astype(str).agg("_".join, axis=1)
    else:
        groups_data = X[group_on]
    return groups_data.copy()


def time_on_to_groups_data(*, X: pd.DataFrame, time_on: str, num_folds: int) -> tuple[pd.Series, int]:
    """Go from time column to a group column for splits."""
    time_data = X[time_on]

    if pd.api.types.is_datetime64_any_dtype(time_data):
        time_data = time_data.astype("int64")
    assert pd.api.types.is_numeric_dtype(time_data), "Time_on column is not datetime or numeric!"

    return split_time_index_into_intervals(
        time_data=time_data,
        goal_n_intervals=num_folds,
    )


def get_num_group_instances(metadata: ValidationMetadata, X: pd.DataFrame) -> int:
    """Compute the number of rows that represent how much (multi-instance) samples
    the data has. This is used to determine which splits to use.
    """
    from tabarena.benchmark.task.openml import TabArenaTaskMetadataMixin

    return TabArenaTaskMetadataMixin.get_num_instance_groups(
        X=X,
        group_on=metadata.group_on,
        group_labels=metadata.group_labels,
    )


def split_time_index_into_intervals(
    *,
    time_data: pd.Series,
    goal_n_intervals: int,
    balance_on: str = "rows",
) -> tuple[pd.Series, int]:
    """Split a monotonically ordered time index into contiguous dynamic intervals.

    Rules:
    - Larger time values are always later in time
    - Equal time values are always assigned to the same interval
    - Intervals are created from the observed data, not equal-width spacing
    - Tries to create `goal_n_intervals`, but falls back to a smaller number if needed
    - Never returns fewer than 2 intervals

    Parameters
    ----------
    time_data : pd.Series
        Time input
    goal_n_intervals : int
        Desired number of intervals.
    balance_on : {"rows", "unique"}, default "rows"
        - "rows": balance intervals by number of rows
        - "unique": balance intervals by number of unique time values

    Returns:
    -------
    time_intervals: pd.Series
        Interval label for each row in the input time_data.
    actual_n_intervals : int
        Number of intervals actually used.
    """
    if goal_n_intervals < 2:
        raise ValueError("n_intervals must be at least 2.")
    if balance_on not in {"rows", "unique"}:
        raise ValueError("balance_on must be either 'rows' or 'unique'.")

    assert not time_data.isna().any(), "Time column contains nan values!"

    s = time_data.copy()

    # Aggregate identical time values so duplicates stay together
    counts = s.value_counts(dropna=False).sort_index().rename("row_count").to_frame()
    counts["unique_weight"] = 1

    n_unique = len(counts)
    if n_unique < 2:
        raise ValueError("Need at least 2 unique time values to create at least 2 intervals.")
    actual_n_intervals = min(goal_n_intervals, n_unique)
    if actual_n_intervals < 2:
        raise ValueError("Could not create at least 2 intervals.")

    weight_col = "row_count" if balance_on == "rows" else "unique_weight"
    weights = counts[weight_col].to_numpy()

    # Greedy partition of sorted unique values into contiguous groups
    # aiming for equal cumulative weight per interval.
    total_weight = weights.sum()
    cut_positions = []
    start = 0
    cumulative = np.cumsum(weights)

    for group_num in range(1, actual_n_intervals):
        target = group_num * total_weight / actual_n_intervals

        # Candidate cut indices are between unique values:
        # cut after index j means next group starts at j+1
        min_j = start
        max_j = n_unique - (actual_n_intervals - group_num) - 1

        # Choose cut whose cumulative weight is closest to target
        candidates = np.arange(min_j, max_j + 1)
        j = candidates[np.argmin(np.abs(cumulative[candidates] - target))]
        cut_positions.append(j)
        start = j + 1

    # Assign interval labels to each unique time value
    interval_labels_for_unique = np.empty(n_unique, dtype=int)
    prev = 0
    for interval_id, cut in enumerate(cut_positions):
        interval_labels_for_unique[prev : cut + 1] = interval_id
        prev = cut + 1
    interval_labels_for_unique[prev:] = len(cut_positions)

    mapping = pd.Series(interval_labels_for_unique, index=counts.index)

    time_intervals = time_data.map(mapping)

    return time_intervals, actual_n_intervals
