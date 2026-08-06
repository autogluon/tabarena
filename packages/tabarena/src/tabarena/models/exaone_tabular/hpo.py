from __future__ import annotations

from autogluon.common.space import Categorical

from tabarena.models.exaone_tabular.model import REGRESSION_CHECKPOINTS, EXAONETabularModel
from tabarena.utils.config_utils import ConfigGenerator

search_space = {
    # Shared. `n_svd` leads on the classifier's released default of 8; the regressor's own default
    # is 16, which a sampled config therefore only lands on by drawing it.
    "n_svd": Categorical(8, 0, 16, 32),
    "use_quantile_map": Categorical(False, True),
    "rescale_for_column_count": Categorical(False, True),
    # Regression-only: the released checkpoints, and whether the augmentation runs as a split
    # ensemble. `svd_split` has no classification counterpart because that side averages member
    # probabilities rather than weighting them, so it has no way to price a second view.
    "regression_weight": Categorical(*REGRESSION_CHECKPOINTS),
    "regression_svd_split": Categorical(True, False),
}

# One curated config: the empty one, meaning the released defaults of whichever checkpoint is
# loaded. Spelled out, that is the classifier appending 8 support-SVD components per ensemble member
# (`ClassificationConfig.n_svd`) with no quantile map, and the default regression checkpoint reading
# out 999 quantiles as a trimmed mean over a split ensemble (`svd_split=True` pools an un-augmented
# pass with a 16-component one, priced by the NNLS member-weight fit). Setting nothing is what keeps
# it on each task's own defaults, which a sampled config cannot express through one shared `n_svd`.
manual_configs = [{}]

gen_exaone_tabular = ConfigGenerator(
    model_cls=EXAONETabularModel,
    search_space=search_space,
    manual_configs=manual_configs,
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_exaone_tabular.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
