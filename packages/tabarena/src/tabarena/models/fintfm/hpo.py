from __future__ import annotations

from autogluon.common.space import Categorical

from tabarena.models.fintfm.model import FinTFMModel
from tabarena.utils.config_utils import ConfigGenerator

#: Unofficial search space over fintfm's inference-time options. fintfm has no train loop, so
#: every key here is a ``FinancialTFMClassifier`` argument; there is no softmax temperature or
#: other option the classifier lacks. None of these levers has been measured on TabArena yet,
#: and the per-member ones (``ensemble_transforms``, ``ensemble_svd_components``,
#: ``ensemble_class_shift``, ``max_features_strategy``) need fintfm 0.8.0 or later; the pinned
#: ``fintfm==0.10.1`` extra has all of them, and the partitioned contexts and probe per leaf
#: (fintfm FINDINGS §215) too. The default config (``manual_configs=[{}]``) uses none of them.
#:
#: Cost scales linearly with ``n_ensemble`` (doubled again by ``ensemble_label_swap``) and with
#: the square of ``max_context``, so a tuned run is GPU work (fintfm audit AU5.13 records about
#: 11 s per 1,000 rows on the local CPU/MPS machine).
search_space = {
    # Context rows per member. The checkpoint's trained range ends near 921 context rows
    # (fintfm audit AU5.14); 2,000 is the classifier's own default and extrapolates past it.
    "max_context": Categorical(500, 750, 1000, 2000),
    # "retrieval" is left out: fintfm FINDINGS §70 measured it as harmful at low prevalence,
    # it makes query chunking inexact, and it rejects ensemble_svd_components > 0.
    "context_strategy": Categorical("uniform", "hybrid", "balanced", "prototype"),
    # Every choice is >= 2, which ensemble_svd_components > 0 requires.
    "n_ensemble": Categorical(4, 8, 16),
    "ensemble_label_swap": Categorical(False, True),
    "ensemble_class_shift": Categorical(False, True),
    "feature_transform": Categorical("rank", "power", "squash", "winsor", "none"),
    # None gives every member feature_transform; a list overrides it member by member.
    "ensemble_transforms": Categorical(
        None,
        ["rank", "power"],
        ["rank", "power", "squash"],
    ),
    # Appended to odd-index members only, and capped at the checkpoint's feature budget.
    "ensemble_svd_components": Categorical(0, 8, 16),
    # Identical to "raise" on a table within the checkpoint's 136 features. On a wider one it
    # scores an ensemble over column subsets instead of raising, which on TabArena v0.1
    # admits APSFailure, kddcup09_appetency and Bioresponse to the tuned configs. The default
    # config keeps "raise", so those three tasks have no default-config result.
    "max_features_strategy": Categorical("subsample"),
}

gen_fintfm = ConfigGenerator(
    model_cls=FinTFMModel,
    search_space=search_space,
    manual_configs=[{}],
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_fintfm.generate_all_bag_experiments(num_random_configs=4),
        ),
    )
