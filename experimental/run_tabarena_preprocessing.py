"""Demonstrate TabArena preprocessing integrated into AutoGluon on a synthetic dataset.

The dataset contains all four feature types that TabArena preprocessing handles:

    * **Numerical**  — float/int columns passed through unchanged.
    * **Categorical** — ``pd.Categorical`` columns kept as raw categories by the
      model-agnostic generator, then ordinal-encoded per model by the model-specific
      generator.
    * **Text**       — free-text columns expanded by ``TextSpecialFeatureGenerator``
      (char counts, word counts, …) and by ``StatisticalTextFeatureGenerator``
      (skrub ``StringEncoder`` embeddings), then PCA-reduced per model.
    * **Datetime**   — a ``datetime64`` column expanded into year/month/day/weekday
      features by ``DateTimeFeatureGenerator``.

The model-agnostic preprocessing (``TabArenaModelAgnosticPreprocessing``) is the
global ``feature_generator`` shared across all models.  The model-specific
preprocessing (``TabArenaModelSpecificPreprocessing``) is injected into each model's
hyperparameters and runs after the global step; it ordinal-encodes categoricals and
PCA-reduces text embeddings.

Only LightGBM (``"GBM"`` in AutoGluon) is used so the run is fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from tabarena.benchmark.preprocessing import (
    TabArenaModelAgnosticPreprocessing,
    TabArenaModelSpecificPreprocessing,
)

# ---------------------------------------------------------------------------
# 1. Synthetic dataset
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)
N = 800
LABEL = "target"

_TEXT_POOL = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is transforming the world rapidly",
    "Natural language processing handles text features well",
    "Deep learning models require large amounts of data",
    "Feature engineering significantly improves model performance",
    "Tabular data often contains a mix of feature types",
    "Preprocessing pipelines are essential for building good models",
    "AutoGluon automates the end-to-end machine learning pipeline",
    "Dimensionality reduction compresses high-dimensional embeddings",
    "Categorical features need special treatment in ML pipelines",
]

df = pd.DataFrame(
    {
        # --- numericals ---
        "num_gaussian": RNG.standard_normal(N),
        "num_uniform": RNG.uniform(0.0, 100.0, N),
        "num_int": RNG.integers(0, 50, size=N).astype(float),
        # --- categoricals (pd.Categorical keeps them as R_CATEGORY) ---
        "cat_priority": pd.Categorical(
            RNG.choice(["low", "medium", "high", "critical"], N)
        ),
        "cat_colour": pd.Categorical(RNG.choice(["red", "green", "blue"], N)),
        # --- text ---
        "text_description": [_TEXT_POOL[i % len(_TEXT_POOL)] for i in range(N)],
        "text_comment": [
            f"{_TEXT_POOL[(i + 3) % len(_TEXT_POOL)]} with additional context"
            for i in range(N)
        ],
        # --- datetime ---
        "event_date": pd.date_range("2019-01-01", periods=N, freq="D"),
        # --- binary label correlated with numericals ---
        LABEL: (RNG.standard_normal(N) > 0).astype(int),
    }
)
df["text_description"] = df["text_description"].astype("string")

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df[LABEL]
)
print(f"Train size: {len(train_df)}  Test size: {len(test_df)}")
print(f"Dtypes:\n{train_df.dtypes}\n")

# ---------------------------------------------------------------------------
# 2. Run AutoGluon with TabArena's preprocessing:
# ---------------------------------------------------------------------------
# Init model agnostic preprocessing
feature_generator = TabArenaModelAgnosticPreprocessing(
    # Set False to run the example faster without a GPU.
    # Also note that this downloads a model from HuggingFace the first time you run it, which can take some time.
    enable_sematic_text_features=False,
)

# Init model specific preprocessing
hyperparameters: dict = {"GBM": [{"num_boost_round": 10}]}
for model_name in list(hyperparameters.keys()):
    model_hp_list = hyperparameters[model_name][:]
    for i in range(len(model_hp_list)):
        model_hp_list[i] = TabArenaModelSpecificPreprocessing.add_to_hyperparameters(
            model_hp_list[i]
        )
    hyperparameters[model_name] = model_hp_list

# Run AutoGluon with TabArena preprocessing
predictor = TabularPredictor(
    label=LABEL,
    problem_type="binary",
    eval_metric="roc_auc",
    verbosity=2,
).fit(
    presets="best_quality",
    train_data=train_df,
    hyperparameters=hyperparameters,
    feature_generator=feature_generator,
    # Use sequential_local to show logs of preprocessing
    ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    # Other settings for to ignore for this example
    fit_weighted_ensemble=False,
    dynamic_stacking=False,
)

# Check output
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

print("\n=== Leaderboard ===")
leaderboard = predictor.leaderboard(test_df, display=True)

print("\n=== Data after model agnostic preprocessing ===")
data_internal, _ = predictor.load_data_internal()
print(data_internal.head())

# TODO: get preprocessing from model here...
print("\n=== Data after model specific preprocessing ===")
data_model = predictor.transform_features(data=train_df, model="LightGBM_BAG_L1")
print(data_model.head())
