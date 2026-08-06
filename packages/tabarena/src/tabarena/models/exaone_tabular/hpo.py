from __future__ import annotations

from tabarena.models.exaone_tabular.model import (
    REGRESSION_WEIGHT_3EED,
    REGRESSION_WEIGHT_8BD9,
    EXAONETabularModel,
)
from tabarena.utils.config_utils import ConfigGenerator

# No sampled search space: the tuned + ensemble evaluation runs over the fixed portfolio below.
# Each config zips one classification arm with one regression arm, and the wrapper reads only the
# arm matching the fitted task, so a single portfolio spans both a classification-only and a
# regression-only checkpoint.
#
# c1 is empty, which means the released defaults of whichever checkpoint gets loaded. Spelled out,
# that is: the classifier appending 8 support-SVD components per ensemble member
# (`ClassificationConfig.n_svd`) with no quantile map, and the default regression checkpoint reading
# out 999 quantiles as a trimmed mean over a split ensemble (`svd_split=True`, an un-augmented pass
# pooled with a 16-component one, priced by the NNLS member-weight fit).
#
# c2-c6 are single- or paired-axis divergences from that baseline, along the two axes the release
# actually exposes: the classification SVD width, and — for regression — the checkpoint together
# with whether the split ensemble runs. Only the alternative checkpoints are named; the default one
# is reached by saying nothing, so it never appears as a literal.
manual_configs = [
    # c1: released defaults for both tasks (see above).
    {},
    # c2: augmentation off for classification; regression on the default checkpoint, unsplit.
    {
        "classification": {"n_svd": 0},
        "regression": {"svd_split": False},
    },
    # c3-c6: the wider classification arm and the default one in turn, against the two alternative
    # regression checkpoints, each taken with and without the split ensemble.
    {
        "classification": {"n_svd": 16},
        "regression": {"weight": REGRESSION_WEIGHT_3EED},
    },
    {
        "regression": {"weight": REGRESSION_WEIGHT_3EED, "svd_split": False},
    },
    {
        "classification": {"n_svd": 0},
        "regression": {"weight": REGRESSION_WEIGHT_8BD9},
    },
    {
        "classification": {"n_svd": 16},
        "regression": {"weight": REGRESSION_WEIGHT_8BD9, "svd_split": False},
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
