"""HPO config generator for PrismBoost.

``manual_configs=[{}]`` is the leaderboard's default row: PrismBoost's ``"auto"`` capacity
rules, with the round count taken from the validation split by the wrapper's ladder.

The ranges follow the per-dataset Optuna optima of PrismBoost's 121-dataset PMLB study, widened
to that study's own search bounds where the optima crowd an edge (``max_depth`` reached 14 there,
``min_samples_leaf`` 49, ``learning_rate`` ~1.0). ``numeric_scaler`` and ``categorical_encoder``
are searched for the same reason the study searched them: a SEFR node weights feature ``j`` by
``(avg_pos_j - avg_neg_j) / (avg_pos_j + avg_neg_j)``, a ratio that moves when the column is
shifted, and no single scaling won across datasets (the study's optima split 30/25/24/22/20 over
the five, and TabArena-Lite agrees).

``n_estimators`` is deliberately absent: the wrapper picks the round count on ``X_val``, so
searching it would duplicate a value that is already chosen per fit. ``second_order`` is absent
because the C++ backend implements Newton splits only and a first-order config would fall back to
the much slower Python backend. ``class_weight`` is sampled and dropped on regression inside the
wrapper.
"""

from __future__ import annotations

from autogluon.common.space import Categorical, Int, Real

from tabarena.models.prismboost._internal.preprocessing import (
    CATEGORICAL_ENCODERS,
    NUMERIC_SCALERS,
)
from tabarena.models.prismboost.model import PrismBoostModel
from tabarena.utils.config_utils import ConfigGenerator

_SEARCH_SPACE = {
    "learning_rate": Real(0.01, 0.5, log=True),
    "max_depth": Int(2, 14),
    "min_samples_leaf": Int(2, 50),
    "min_samples_split": Int(8, 100),
    "subsample": Real(0.5, 1.0),
    "reg_lambda": Real(0.1, 50.0, log=True),
    "split_mode": Categorical("hybrid", "sefr_only", "axis_fallback", "hybrid_sampled"),
    "numeric_scaler": Categorical(*NUMERIC_SCALERS),
    "categorical_encoder": Categorical(*CATEGORICAL_ENCODERS),
    "class_weight": Categorical(None, "balanced"),
}

gen_prismboost = ConfigGenerator(
    model_cls=PrismBoostModel,
    manual_configs=[{}],
    search_space=_SEARCH_SPACE,
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_prismboost.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
