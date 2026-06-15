"""Register locally-run models with an arena context (instead of passing ``new_results=``).

The counterpart to run_quickstart_tabarena.py: rather than threading a results DataFrame into
``compare(new_results=...)``, this converts the run into ``InMemoryMethodMetadata`` objects
(``EndToEnd.from_raw_to_methods``) and registers them at context init via ``extra_methods=``.
The new methods are then first-class — picked up automatically by ``compare`` (through
``load_results``), restricted to their own tasks via ``only_valid_tasks=True``, and carried
(with their metadata: hardware, verified, ...) into ``leaderboard_to_website_format``.

Bounded for a fast, self-contained run: 3 small datasets, and the cached ``LightGBM`` as the
only baseline (so no ~30-method S3 download). fillna/calibration are disabled so no separate
RandomForest baseline is needed.
"""

from __future__ import annotations

import copy
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
from tabarena.nips2025_utils.artifacts import tabarena_method_metadata_collection
from tabarena.nips2025_utils.end_to_end import EndToEnd
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


def lightgbm_baseline_metadata():
    """The cached paper ``LightGBM`` MethodMetadata (the single baseline for this run)."""
    metas = [m for m in tabarena_method_metadata_collection.method_metadata_lst if m.method == "LightGBM"]
    assert len(metas) == 1, metas
    return copy.deepcopy(metas[0])


if __name__ == "__main__":
    here = Path(__file__).parent
    run_name = "register_new_methods"
    results_dir = str(here / "experiments" / run_name)
    eval_dir = here / "eval" / run_name

    # 1: 3 small datasets, first split only (r0f0).
    task_collection = TabArenaTaskMetadataCollection().subset_tasks(
        split_indices="lite",
        dataset_names=DATASETS,
    )
    task_collection = task_collection.materialize()

    # 2: same two models as the quickstart.
    bundle = TabArenaV0pt1ExperimentBundle(
        models=[
            ("Linear", 0),
            (CustomRandomForestModel.config_generator(), 1),
        ],
    )
    experiments = bundle.build_experiments()
    jobs = build_jobs(experiments, task_collection)

    # 3: run locally (in-process native backend).
    runner = ExperimentBatchRunner(expname=results_dir, task_metadata=task_collection, debug_mode=True)
    results_lst = runner.run_jobs(jobs)

    # 4: NEW PATH — convert the run into registerable InMemoryMethodMetadata objects.
    new_methods = EndToEnd.from_raw_to_methods(
        results_lst=results_lst,
        task_metadata=task_collection,
        new_result_prefix="[New] ",
    )
    print("\n=== registered in-memory methods ===")
    for m in new_methods:
        print(
            f"  method={m.method!r}  artifact_name={m.artifact_name!r}  "
            f"config_type={m.config_type!r}  display_name={m.display_name!r}  type={type(m).__name__}",
        )

    # 5: register the new methods at init (alongside the cached LightGBM baseline). No
    #    new_results= passed to compare; the methods are picked up via load_results(), and
    #    only_valid_tasks=True restricts the leaderboard to the tasks they ran.
    ta_context = TabArenaContext(
        methods=[lightgbm_baseline_metadata()],
        task_metadata=task_collection,
        extra_methods=new_methods,
        backend="native",
        fillna_method=None,  # avoid needing the RandomForest fillna baseline
        calibration_method=None,
    )

    leaderboard = ta_context.compare(
        output_dir=eval_dir,
        only_valid_tasks=True,
    )
    print("\n=== leaderboard (raw) ===")
    print(leaderboard.to_string())

    leaderboard_website = ta_context.leaderboard_to_website_format(leaderboard=leaderboard)
    print("\n=== TabArena leaderboard (website format) ===")
    print(leaderboard_website.to_markdown(index=False))
    print(f"\nView saved figures in {eval_dir}")
