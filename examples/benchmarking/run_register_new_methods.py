"""Register locally-run models with an arena context (instead of passing ``new_results=``).

The counterpart to run_quickstart_tabarena.py: rather than threading a results DataFrame into
``compare(new_results=...)``, the arena context is the single hub — ``run_experiments`` runs the
models scoped to the chosen tasks and ``register`` turns the run into ``InMemoryMethodMetadata``
and pre-filters ``task_metadata`` to the tasks they ran. The new methods are then first-class —
picked up automatically by ``compare`` (through ``load_results``), restricted to their own tasks
(so ``compare`` scopes to them with nothing extra), and carried (with their metadata: hardware,
verified, ...) into ``leaderboard_to_website_format``.

Bounded for a fast, self-contained run: 3 small datasets, first split only.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from autogluon.core.models import AbstractModel
from autogluon.features import LabelEncoderFeatureGenerator

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.nips2025_utils.tabarena_context import TabArenaContext

if TYPE_CHECKING:
    import pandas as pd

    from tabarena.utils.config_utils import ConfigGenerator

DATASETS = ["blood-transfusion-service-center", "QSAR_fish_toxicity", "anneal"]


class CustomRandomForestModel(AbstractModel):
    ag_key = "CRF"
    ag_name = "CustomRF"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None

    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, **kwargs) -> np.ndarray:
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
        else:
            from sklearn.ensemble import RandomForestClassifier

            model_cls = RandomForestClassifier

        X = self.preprocess(X, y=y, is_train=True)
        self.model = model_cls(**self._get_model_params())
        self.model.fit(X, y)

    def _set_default_params(self) -> None:
        for param, val in {"n_estimators": 10, "n_jobs": -1, "random_state": 0}.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update({"valid_raw_types": ["int", "float", "category"]})
        return default_auxiliary_params

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        return ["binary", "multiclass", "regression"]

    @classmethod
    def config_generator(cls) -> ConfigGenerator:
        from autogluon.common.space import Int

        from tabarena.utils.config_utils import ConfigGenerator

        return ConfigGenerator(
            model_cls=cls,
            manual_configs=[{}],
            search_space={"n_estimators": Int(4, 50)},
        )


if __name__ == "__main__":
    here = Path(__file__).parent
    run_name = "register_new_methods"
    results_dir = str(here / "experiments" / run_name)
    eval_dir = here / "eval" / run_name

    # 1: the models to run: a registry model at its default config,
    #    and a custom model with one extra random HPO config.
    experiments = TabArenaV0pt1ExperimentBundle(
        models=[
            ("Linear", 0),
            (CustomRandomForestModel.config_generator(), 1),
        ],
    ).build_experiments()

    # 2: the context is the hub. run_experiments builds a runner scoped to 3 small datasets'
    #    first split (r0f0), runs it locally (debug_mode -> in-process native backend), and hands
    #    back the raw results. register=False here so we can register them explicitly below (and
    #    inspect the registered methods); pass register=True to do it in a single call.
    ta_context = TabArenaContext()
    results_lst = ta_context.run_experiments(
        experiments,
        expname=results_dir,
        subset="lite",
        datasets=DATASETS,
        register=False,
        # debug_mode=True,  # <-- For local debugger
    )

    # 3: register the run as InMemoryMethodMetadata, pre-filtering the context's task_metadata to
    #    the tasks just run — so `compare` scopes to them with nothing extra.
    new_methods = ta_context.register(results_lst, new_result_prefix="[New] ")
    print("\n=== registered in-memory methods ===")
    for m in new_methods:
        print(
            f"  method={m.method!r}  artifact_name={m.artifact_name!r}  "
            f"config_type={m.config_type!r}  display_name={m.display_name!r}  type={type(m).__name__}",
        )

    # 4: compare against the cached baselines.
    leaderboard = ta_context.compare(output_dir=eval_dir)
    leaderboard_website = ta_context.leaderboard_to_website_format(leaderboard=leaderboard)
    print("\n=== TabArena leaderboard (website format) ===")
    print(leaderboard_website.to_markdown(index=False))
    print(f"\nView saved figures in {eval_dir}")
