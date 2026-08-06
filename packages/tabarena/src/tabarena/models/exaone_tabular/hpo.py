from __future__ import annotations

from tabarena.models.exaone_tabular.model import (
    REGRESSION_WEIGHT_3EED,
    REGRESSION_WEIGHT_8BD9,
    REGRESSION_WEIGHT_DEFAULT,
    EXAONETabularModel,
)
from tabarena.utils.config_utils import ConfigGenerator

# No sampled search space: the tuned + ensemble evaluation runs over the fixed portfolio below.
# Each config zips one classification arm with one regression arm, and the wrapper reads only the
# arm matching the fitted task, so per task every distinct arm appears exactly once.
#
# The classification arm varies two orthogonal inference knobs — the support-SVD augmentation width
# and the input transform — because only one classifier checkpoint is released, so its diversity has
# to come from inference behavior rather than from weights. `n_svd=8` is the released default (see
# `ClassificationConfig`), so c1/c4 switch the augmentation off rather than on.
#
# The regression arm varies the checkpoint and the SVD split, which is where the released weights do
# offer a choice: `svd_split=True` (the default) pools an un-augmented pass with an augmented one and
# lets the NNLS weight fit price them, while `False` runs the augmented pass alone.
#: Every arm names its checkpoint and both inference knobs outright, including where the value
#: matches the release default. The defaults have already moved once under this portfolio
#: (`ClassificationConfig.n_svd` became 8 upstream), so spelling them out keeps a config's meaning
#: fixed to what was benchmarked rather than to whatever the library currently defaults to.
manual_configs = [
    # c1: no classification augmentation; released regression checkpoint with the SVD split.
    {
        "classification": {"n_svd": 0, "use_quantile_map": False},
        "regression": {"weight": REGRESSION_WEIGHT_DEFAULT, "svd_split": True},
    },
    # c2: released classification default; released regressor without the SVD split.
    {
        "classification": {"n_svd": 8, "use_quantile_map": False},
        "regression": {"weight": REGRESSION_WEIGHT_DEFAULT, "svd_split": False},
    },
    # c3-c6: wider / quantile-mapped classification arms against the alternative regression
    # checkpoints, each of those taken without and with the SVD split.
    {
        "classification": {"n_svd": 16, "use_quantile_map": False},
        "regression": {"weight": REGRESSION_WEIGHT_3EED, "svd_split": False},
    },
    {
        "classification": {"n_svd": 0, "use_quantile_map": True},
        "regression": {"weight": REGRESSION_WEIGHT_3EED, "svd_split": True},
    },
    {
        "classification": {"n_svd": 8, "use_quantile_map": True},
        "regression": {"weight": REGRESSION_WEIGHT_8BD9, "svd_split": False},
    },
    {
        "classification": {"n_svd": 16, "use_quantile_map": True},
        "regression": {"weight": REGRESSION_WEIGHT_8BD9, "svd_split": True},
    },
]

gen_exaone_tabular = ConfigGenerator(
    model_cls=EXAONETabularModel,
    search_space={},
    manual_configs=manual_configs,
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_exaone_tabular.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
