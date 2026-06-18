"""HPO config generator for ChimeraBoost.

manual_configs=[{}] is the single default config (the default leaderboard row).
The search space is the tuned entry's HPO budget (TabArena convention: 200 random
configs). Pre-registered from dev-side evidence only (Grinsztajn / OpenML / PMLB),
never from TabArena results.

Search-space notes:
* Core capacity/regularization knobs with known per-dataset variance:
  learning_rate, depth, l2_leaf_reg, min_child_weight, subsample, colsample,
  leaf_estimation_iterations, max_bins.
* Categorical-handling knobs: cat_smoothing (ordered-TS pseudocount),
  cat_n_permutations (anti-leakage averaging). The raw cat_combinations flag is
  EXCLUDED — explicit True bypasses the auto-rule's <=1000-pair resource guard
  and would explode on high-cardinality tasks.
* linear_leaves (regression wins; default only auto-on for binary) searched
  jointly with its regularizer linear_lambda.
* ordered_boosting (CatBoost-style ordered target stats), default off.
* n_estimators is NOT searched — every config inherits the model default
  (10000 cap + early stopping; LR is pinned at 0.1 under ES so a high cap is
  free headroom, and searching a budget cap only adds noise).
"""

from __future__ import annotations

from autogluon.common.space import Categorical, Int, Real

from tabarena.models.chimeraboost.model import ChimeraBoostModel
from tabarena.utils.config_utils import ConfigGenerator

_SEARCH_SPACE = {
    # n_estimators is not searched: the model default (10000 cap + early
    # stopping) applies to every config; searching a budget cap only adds noise.
    "learning_rate": Real(0.03, 0.3, log=True),
    "depth": Int(4, 8),
    "l2_leaf_reg": Real(0.1, 10.0, log=True),
    "min_child_weight": Real(0.0, 8.0),
    "subsample": Real(0.5, 1.0),
    "colsample": Real(0.5, 1.0),
    "leaf_estimation_iterations": Int(1, 5),
    "max_bins": Categorical(128, 254),
    "linear_leaves": Categorical(False, True),
    "linear_lambda": Real(0.1, 10.0, log=True),
    "cat_smoothing": Real(0.1, 10.0, log=True),
    "cat_n_permutations": Int(1, 8),
    "ordered_boosting": Categorical(False, True),
}

gen_chimeraboost = ConfigGenerator(
    model_cls=ChimeraBoostModel,
    manual_configs=[{}],
    search_space=_SEARCH_SPACE,
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_chimeraboost.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
