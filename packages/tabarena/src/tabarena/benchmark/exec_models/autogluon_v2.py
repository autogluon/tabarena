"""AutoGluon exec-model wrappers that delegate non-IID validation splitting to AutoGluon.

The V1 wrappers in :mod:`.autogluon` resolve a task's grouped / temporal validation splits in
TabArena (``resolve_validation_splits`` -> explicit ``ag_args_ensemble['custom_splits']``, or
``resolve_holdout_split`` -> explicit ``tuning_data``), then hand AutoGluon finished index lists.
AutoGluon now understands the structure itself: ``TabularPredictor.fit(validation_structure=...)``
takes the same declarative description (``group_on`` / ``time_on`` / ``stratify_on`` /
``group_time_on``) and builds the splits internally, for both the bagged and holdout paths.

The V2 wrappers here pass that description through instead of resolving anything, so a run
exercises AutoGluon's native implementation. Everything else is inherited unchanged: the same
model, hyperparameters, preprocessing pipeline, feature generator, and resources.

Two things the caller must get right for the two paths to agree:

- **Split seed.** AutoGluon's learner seeds structure splits from its own ``random_state``
  (default 0), while TabArena's splits come from ``data_foundry``, which uses 4267 internally.
  These wrappers therefore set the learner's ``random_state`` to :data:`SPLIT_RANDOM_STATE`, but
  only on a task that actually declares a structure -- the same seed also drives AutoGluon's
  default splitter, which unstructured tasks are left to use as-is.
- **Fold sizing.** TabArena's fold counts come from its own policy
  (``ValidationMetadata.resolve_number_of_splits``: a tiny-data regime below a group-instance
  threshold, fixed defaults above it), which these wrappers deliberately do NOT apply -- sizing
  is AutoGluon's to own, via ``validation_size_curves``. Pass explicit ``num_bag_folds`` /
  ``num_bag_sets`` to take sizing out of the comparison, or configure the curves to match the
  policy. ``size_validation_on_groups`` is set from the task so a curve that opts into group
  sizing reads the same count TabArena would.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from tabarena.benchmark.exec_models.autogluon import AGSingleBagWrapper, AGSingleWrapper, AGWrapper
from tabarena.benchmark.exec_models.autogluon_utils import get_num_group_instances

if TYPE_CHECKING:
    import pandas as pd
    from autogluon.common.utils.validation_structure import ValidationStructure

    from tabarena.benchmark.task.metadata import ValidationMetadata

#: ``data_foundry``'s split seed (``SPLIT_RANDOM_STATE``), which TabArena's resolved splits use.
#: AutoGluon's learner defaults to 0, so matching TabArena means passing this instead.
SPLIT_RANDOM_STATE = 4267


def validation_structure_from_metadata(
    metadata: ValidationMetadata,
    *,
    temporal_forward_only: bool = False,
    size_validation_on_groups: bool | None = None,
) -> ValidationStructure | None:
    """The :class:`ValidationStructure` a task's validation metadata declares, or None.

    The single projection every AutoGluon-resolved path uses -- :class:`AGWrapperV2` here, and
    any system wrapper that builds its own ``TabularPredictor`` -- so all of them declare the
    same structure for the same task. See :meth:`AGWrapperV2.validation_structure` for why
    ``stratify_on`` alone yields None and why ``group_time_on`` is not forwarded.

    ``size_validation_on_groups`` decides what AutoGluon's size-dependent choices -- fold counts,
    holdout, and every ``validation_size_curves`` entry -- are measured against. ``None`` applies
    TabArena's rule for config runs: group instances exactly when the group labels are per-group,
    rows otherwise. A system whose configuration is written in training rows passes ``False``,
    or a 4,400-row task with 90 groups would be sized as a 90-row one.
    """
    from autogluon.common.utils.validation_structure import ValidationStructure

    if metadata.group_on is None and metadata.time_on is None:
        return None
    if size_validation_on_groups is None:
        size_validation_on_groups = metadata.group_labels == "per_group"
    return ValidationStructure(
        group_on=metadata.group_on,
        time_on=metadata.time_on,
        stratify_on=metadata.stratify_on,
        temporal_forward_only=temporal_forward_only,
        size_validation_on_groups=size_validation_on_groups,
    )


class AGWrapperV2(AGWrapper):
    """An :class:`AGWrapper` that hands the task's structure to AutoGluon instead of resolving it.

    Parameters
    ----------
    temporal_forward_only: bool, default False
        Ask AutoGluon for forward-chaining temporal validation instead of the default
        leave-one-block-out: fold *i* validates time block *i+1* and trains only on earlier
        blocks, so no fold is trained on data from after the window it is scored on. Costs the
        earliest block (never validated, hence no out-of-fold prediction for those rows) and
        trains each fold on less data. No effect on a task without ``time_on`` /
        ``group_time_on``. Cannot be combined with stacking (AutoGluon raises).
    split_random_state: int, default :data:`SPLIT_RANDOM_STATE`
        Seed for AutoGluon's structure-aware splitting, injected as the learner's
        ``random_state`` when a structure is declared (see :meth:`_build_predictor_args`).
        Defaults to ``data_foundry``'s seed so the folds match TabArena's; a different value
        gives different (equally valid) folds.
    **kwargs:
        As :class:`AGWrapper`.
    """

    def __init__(
        self,
        split_random_state: int = SPLIT_RANDOM_STATE,
        temporal_forward_only: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.split_random_state = split_random_state
        self.temporal_forward_only = temporal_forward_only

    def _build_predictor_args(self, **kwargs) -> tuple[pd.DataFrame, dict, dict]:
        """Seed the learner for structure-aware splitting, but only when it does any.

        The learner's ``random_state`` seeds both ``ValidationStructure``'s splits and
        AutoGluon's *default* splitter. Setting it unconditionally would therefore reseed the
        default splitter on tasks that have no structure to honor -- which are exactly the tasks
        where V1 leaves that splitter alone, so the two paths would build different (equally
        valid) folds for a purely IID task. Injecting only alongside a declared
        ``validation_structure`` keeps this seed scoped to what it names.

        An explicit ``init_kwargs["learner_kwargs"]["random_state"]`` still wins.
        """
        train_data, init_kwargs, fit_kwargs = super()._build_predictor_args(**kwargs)
        if fit_kwargs.get("validation_structure") is not None:
            init_kwargs.setdefault("learner_kwargs", {}).setdefault("random_state", self.split_random_state)
        return train_data, init_kwargs, fit_kwargs

    def validation_structure(self) -> ValidationStructure | None:
        """The task's :class:`ValidationStructure`, projected from its validation metadata.

        Carries only the structure -- which columns group, order, and stratify the data. The
        fold *counts* are not part of it: they stay ``num_bag_folds`` / ``num_bag_sets`` (or come
        from AutoGluon's ``validation_size_curves``), unlike the V1 path where TabArena's policy
        decided them.

        ``None`` for a task with no grouped or temporal structure, *even when it declares
        ``stratify_on``*. Such a task has no leakage to prevent, and AutoGluon's built-in bagging
        already stratifies classification folds by the label -- which is what these tasks
        stratify on (``stratify_on`` equals the target). Declaring the structure anyway would
        take over splitting to produce equally valid but differently seeded folds, so V1 leaves
        AutoGluon's default splitter alone here (``resolve_validation_splits`` returns no
        ``custom_splits`` when neither ``group_on`` nor ``time_on`` is set, and
        ``resolve_holdout_split`` documents the same choice for the holdout path). Taking over
        only where there is leakage to fix is both what matches V1 and the better default.

        ``group_time_on`` is deliberately NOT forwarded, despite both sides having a field by
        that name. They mean different things:

        - TabArena's is time *within* a group, read only by the group-aware feature generator to
          order rows inside a group. ``resolve_validation_splits`` ignores it, so it has no
          bearing on how V1 builds folds.
        - AutoGluon's is a *split* directive: whole groups blocked in time order, mutually
          exclusive with ``group_on`` / ``time_on``.

        Forwarding it would change the split structure rather than reproduce it -- a task like
        ``parkinsons_biomedical_voice_measurements`` (``group_on=patient_id``,
        ``group_time_on=session_number``) would be blocked by session in time order instead of
        held group-disjoint by patient. TabArena has no split regime that is both grouped and
        temporal (``resolve_validation_splits`` raises ``NotImplementedError`` when ``group_on``
        and ``time_on`` are both set), so nothing here needs AutoGluon's ``group_time_on``.
        """
        return validation_structure_from_metadata(
            self.validation_metadata, temporal_forward_only=self.temporal_forward_only
        )

    def _apply_validation_splits(self, fit_kwargs: dict, *, X: pd.DataFrame, y: pd.Series) -> int | None:
        """Declare the structure in ``fit_kwargs`` and leave the fold counts alone.

        Overrides the V1 behavior of popping ``num_bag_folds`` / ``num_bag_sets``, running them
        through TabArena's resolver, and writing back adjusted counts plus ``custom_splits``.
        The counts pass through TabArena's size policy (``resolve_number_of_splits``) as on
        the V1 path; AutoGluon then resolves the splits itself, so any further clamping
        (fewer groups than folds, temporal blocks, repeats collapsed to 1) happens inside
        ``ValidationStructure.custom_splits``.
        """
        num_folds = fit_kwargs.get("num_bag_folds")
        if not self.use_task_specific_validation:
            return num_folds
        # TabArena's count policy, applied as the V1 path applies it: at or below the tiny-data
        # threshold of (group) instances the campaign protocol bags 5 folds x 5 sets rather than
        # the caller's 8 x 1. The structure declared below governs *which* rows share a fold;
        # without this the two paths build the same folds but different numbers of them, and a
        # campaign result cannot be reproduced on any small task.
        if num_folds is not None and num_folds > 1:
            num_folds, num_repeats = self.validation_metadata.resolve_number_of_splits(
                num_folds=num_folds,
                num_repeats=fit_kwargs.get("num_bag_sets"),
                num_group_instances=get_num_group_instances(self.validation_metadata, X=X),
            )
            fit_kwargs["num_bag_folds"] = num_folds
            if num_repeats is not None:
                fit_kwargs["num_bag_sets"] = num_repeats
        validation_structure = self.validation_structure()
        if validation_structure is None:
            logger.info(
                "Task declares no validation structure; leaving AutoGluon's default splitter in "
                f"place (num_bag_folds={num_folds}, num_bag_sets={fit_kwargs.get('num_bag_sets')})."
            )
            return num_folds
        fit_kwargs["validation_structure"] = validation_structure
        logger.info(
            f"Delegating validation splitting to AutoGluon: {validation_structure} "
            f"(num_bag_folds={num_folds}, num_bag_sets={fit_kwargs.get('num_bag_sets')}, "
            f"split_random_state={self.split_random_state})",
        )
        return num_folds

    def _apply_task_specific_holdout(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        num_folds: int | None,
    ) -> tuple[pd.DataFrame, pd.Series, None, None]:
        """Leave the data whole -- AutoGluon carves any structure-aware holdout itself.

        The V1 path resolves the holdout here and passes the rows as ``tuning_data``, because a
        non-bagged ``TabularPredictor`` fit ignores ``custom_splits``. With
        ``validation_structure`` declared, the predictor resolves that split internally (both for
        a plain holdout fit and for ``use_bag_holdout``), so carving it here would double up.
        """
        return X, y, None, None


class AGSingleWrapperV2(AGWrapperV2, AGSingleWrapper):
    """:class:`AGSingleWrapper` (one model, no weighted ensemble) on the native-structure path.

    Inherits ``fit_weighted_ensemble=False`` and ``calibrate=False`` from
    :class:`AGSingleWrapper`, so a fit here is the single configured model and nothing else.
    """


class AGSingleBagWrapperV2(AGWrapperV2, AGSingleBagWrapper):
    """:class:`AGSingleBagWrapper` (bagged, with per-child artifacts) on the native-structure path.

    The bagged wrapper used for benchmarking: AutoGluon builds the group/time-aware folds, and
    the per-child out-of-fold indices and test predictions are still exposed for ensemble
    simulation -- which also makes the realized folds directly comparable to the V1 path's.
    """
