"""Quickstart: benchmark registry + custom models on three datasets of TabArena-Lite and compare to the leaderboard.

Shows:
  * running a registry model by name (e.g. ``"Linear"``),
  * implementing and running your own custom model (``CustomRandomForestModel`` below),
  * hyperparameter search (``n_configs`` > 0 via the model's search space),
  * evaluating the run against the TabArena leaderboard.
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
        return X.fillna(0).to_numpy(dtype=np.float32)  # Only convert to numpy if needed!

    def _fit(self, X: pd.DataFrame, y: pd.Series, num_cpus: int = 1, **kwargs) -> None:
        """Fit logic. Only ``X`` / ``y`` are used here, but the runner passes much more via
        ``**kwargs`` — pull out what your model can exploit (see ``AbstractModel._fit``):

          * ``X_val`` / ``y_val`` — validation split (use it for early stopping / picking
            iterations; ``None`` disables validation-based early stopping),
          * ``time_limit`` — seconds to stay within; ideally early-stop before exceeding it,
          * ``num_cpus`` / ``num_gpus`` — compute budget for this fit,
        """
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
        the bundle samples when it is asked for ``n_configs > 0`` (see the ``models`` list
        below). For a default-only run, pass the tuple ``(..., 0)`` in ``models``.
        """
        from autogluon.common.space import Int

        from tabarena.utils.config_utils import ConfigGenerator

        return ConfigGenerator(
            model_cls=cls,
            manual_configs=[{}],
            search_space={"n_estimators": Int(4, 50)},
        )


if __name__ == "__main__":
    here = Path(__file__).parent
    run_name = "quickstart_tabarena"
    results_dir = str(here / "experiments" / run_name)  # the runner's `expname` (results cache)
    eval_dir = here / "eval" / run_name  # leaderboard / figures `output_dir`

    # 1: the models to run — a registry model by name at its default config, and a custom model
    #    passed as a `(config_generator, n_configs)` tuple with one extra random HPO config.
    #    Registry names: see `tabarena.models.utils.get_configs_generator_from_name` (e.g.
    #    "LightGBM", "RandomForest", "CatBoost", "RealMLP", "TabM", "TabPFNv2", ...).
    experiments = TabArenaV0pt1ExperimentBundle(
        models=[
            ("Linear", 0),
            (CustomRandomForestModel.config_generator(), 1),
        ],
    ).build_experiments()

    # 2: run_experiments scopes to the 3 small datasets' first split
    context = TabArenaContext()
    context.run_experiments(
        experiments,
        expname=results_dir,
        subset="lite",
        datasets=DATASETS,
        new_result_prefix="[New] ",
        debug_mode=True,  # <-- also lets you attach a local debugger
    )

    # 3: compare against the cached baselines; the registered new methods are picked up
    #    automatically and carried into the website-format leaderboard with their metadata.
    leaderboard = context.compare(output_dir=eval_dir)
    leaderboard_website = context.leaderboard_to_website_format(leaderboard=leaderboard)
    print("\n=== TabArena leaderboard (website format) ===")
    print(leaderboard_website.to_markdown(index=False))
    print(f"\nView saved figures in {eval_dir}")
