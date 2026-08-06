from __future__ import annotations

from autogluon.common.space import Categorical

from tabarena.models.exaone_tabular.model import REGRESSION_CHECKPOINTS, EXAONETabularModel
from tabarena.utils.config_utils import ConfigGenerator

# A key is unprefixed where both estimators accept the knob and prefixed where only one does. The
# wrapper drops the other task's keys, so one flat draw configures both halves of the benchmark.
#
# Only inference behavior is sampled, never cost or architecture. Left out on purpose:
# `ensemble_count` and `support_row_limit` (they trade accuracy against compute, so sampling them
# makes configs incomparable rather than diverse), `point_estimate` and `member_weighting` (readout
# choices the release already settled), `compute_dtype` / `query_batch_limit` /
# `support_cache_offload` (speed and memory, not method), and `feature_limit` (the classifier takes
# `min(FEATURE_SELECTION.target_feature_count, feature_limit)` and both are 100, so raising it is a
# no-op). `seed` is already plumbed at the top level via `seed_name`.
#
# Classification contributes no axis of its own: one classifier checkpoint is released against eight
# regressor files, so `weight` is necessarily regression-only and classification varies through the
# shared augmentation and input-transform knobs.
search_space = {
    # Shared. `n_svd` leads on the classifier's released default of 8; the regressor's own default
    # is 16, which a sampled config therefore only lands on by drawing it.
    "n_svd": Categorical(8, 0, 16, 32),
    "use_quantile_map": Categorical(False, True),
    "rescale_for_column_count": Categorical(False, True),
    # Regression-only: the released checkpoints, and how the SVD augmentation is applied.
    "regression_weight": Categorical(*REGRESSION_CHECKPOINTS),
    "regression_svd_split": Categorical(True, False),
    # Withholds the augmentation from small, narrow, all-numeric tables. Off by default because the
    # split already prices the augmentation by weight; on, it decides all-or-nothing instead.
    "regression_svd_gate": Categorical(False, True),
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
