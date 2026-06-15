"""Quickstart: benchmark registry + custom models on TabArena-Lite and compare to the leaderboard.

The quickstart shows:
  * running a registry model by name (e.g. ``"LightGBM"``),
  * implementing and running your own custom model (``CustomRandomForestModel`` below),
  * hyperparameter search (``num_random_configs`` > 0 via the model's search space),
  * evaluating the run against the TabArena leaderboard.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from autogluon.core.models import AbstractModel
from autogluon.features import LabelEncoderFeatureGenerator
from tabarena.benchmark.experiment import (
    ExperimentBatchRunner,
    TabArenaV0pt1ExperimentBundle,
    build_jobs,
)
from tabarena.benchmark.task.metadata import TabArenaTaskMetadataCollection
from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.nips2025_utils.tabarena_context import TabArenaContext

if TYPE_CHECKING:
    import pandas as pd
    from tabarena.utils.config_utils import ConfigGenerator


class CustomRandomForestModel(AbstractModel):
    """Minimal custom model compatible with the scikit-learn API.

    For details on implementing an AutoGluon ``AbstractModel`` see
    https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-custom-model.html
    and the wrappers under ``tabarena/models/``. ``ag_key`` / ``ag_name`` are required by
    ``ConfigGenerator`` (it uses them to name the generated configs, e.g. ``CustomRF_c1_BAG_L1``).

    NOTE on the custom model class: it is defined here in ``__main__`` only because the runner
    runs with ``debug_mode=True`` (in-process "native" backend). For large-scale, Ray-backed
    runs (``debug_mode=False``) the model class MUST live in a separate importable module, since
    Ray workers cannot unpickle a class defined in ``__main__``.
    """

    ag_key = "CRF"
    ag_name = "CustomRF"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None

    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, **kwargs) -> np.ndarray:
        """Model-specific preprocessing: label-encode categoricals, fill NaNs, to float32."""
        X = super()._preprocess(X, **kwargs)
        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        return X.fillna(0).to_numpy(dtype=np.float32)

    def _fit(self, X: pd.DataFrame, y: pd.Series, num_cpus: int = 1, **kwargs) -> None:
        if self.problem_type == "regression":
            from sklearn.ensemble import RandomForestRegressor

            model_cls = RandomForestRegressor
        else:  # "binary" and "multiclass"
            from sklearn.ensemble import RandomForestClassifier

            model_cls = RandomForestClassifier

        X = self.preprocess(X, y=y, is_train=True)
        self.model = model_cls(**self._get_model_params())
        self.model.fit(X, y)

    # `_predict_proba` is inherited from AbstractModel and relies on the sklearn API.

    def _set_default_params(self) -> None:
        for param, val in {"n_estimators": 10, "n_jobs": -1, "random_state": 0}.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        # The model-agnostic preprocessor handles all other dtypes for us.
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update({"valid_raw_types": ["int", "float", "category"]})
        return default_auxiliary_params

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        return ["binary", "multiclass", "regression"]

    @classmethod
    def config_generator(cls) -> ConfigGenerator:
        """Bundle entry for this model: the default config plus a search space for HPO.

        ``manual_configs=[{}]`` always yields the default config; ``search_space`` is what
        the bundle samples when it is asked for ``num_random_configs > 0`` (see the ``models``
        list below). For a default-only run, pass the tuple ``(..., 0)`` in ``models``.
        """
        from autogluon.common.space import Int
        from tabarena.utils.config_utils import ConfigGenerator

        return ConfigGenerator(
            model_cls=cls,
            manual_configs=[{}],
            search_space={"n_estimators": Int(4, 50)},
        )


if __name__ == "__main__":
    # Output dirs, resolved next to this script so they don't depend on the working directory.
    here = Path(__file__).parent
    run_name = "quickstart_tabarena"
    results_dir = str(here / "experiments" / run_name)  # the runner's `expname` (results cache)
    eval_dir = here / "eval" / run_name  # leaderboard / figures `output_dir`

    # 1: TabArena-Lite = the first split (r0f0) of every dataset in the v0.1 suite.
    task_collection = TabArenaTaskMetadataCollection.subset_tasks(split_indices="lite")
    task_collection = task_collection.materialize()

    # 2: pick the models to run. A registry model is named by string; a custom model is passed
    #    as a `(config_generator, n_configs)` tuple. `n_configs` is the number of random HPO
    #    configs on top of the default (0 = default only).
    #    Registry names: see `tabarena.models.utils.get_configs_generator_from_name` (e.g.
    #    "LightGBM", "RandomForest", "CatBoost", "RealMLP", "TabM", "TabPFNv2", ...).
    bundle = TabArenaV0pt1ExperimentBundle(
        models=[
            ("Linear", 0),  # registry model, default config (reproduce TabArena's Linear)
            (CustomRandomForestModel.config_generator(), 1),  # custom model: default + 1 HPO config
        ],
    )
    experiments = bundle.build_experiments()

    # 3: experiments x the collection's splits -> a flat list of jobs.
    jobs = build_jobs(experiments, task_collection)

    # 4: run the jobs. `debug_mode=True` -> in-process native backend (see the module NOTE).
    runner = ExperimentBatchRunner(
        expname=results_dir,
        task_metadata=task_collection,
        debug_mode=True,
    )
    results_lst = runner.run_jobs(jobs)

    # 5: aggregate the raw results into a tidy per-(method, dataset, fold) frame.
    df_results = EndToEnd.from_raw_to_results_df(
        results_lst=results_lst,
        task_metadata=task_collection,
        new_result_prefix="[New] ",
    )
    print("\n=== raw per-fold results ===")
    print(df_results[["method", "dataset", "fold", "metric", "metric_error"]].to_string(index=False))

    # 6: compare against the cached TabArena leaderboard baselines.
    ta_context = TabArenaContext()
    leaderboard = ta_context.compare(
        output_dir=eval_dir,
        new_results=df_results,
        only_valid_tasks=df_results["method"].unique(),  # only compare on tasks we ran
    )
    leaderboard_website = ta_context.leaderboard_to_website_format(leaderboard=leaderboard)
    print("\n=== TabArena leaderboard ===")
    print(leaderboard_website.to_markdown(index=False))
    print(f"\nView saved figures in {eval_dir}")
