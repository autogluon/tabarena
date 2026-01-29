import os
import pickle
import numpy as np
import pandas as pd

from autogluon.features import AutoMLPipelineFeatureGenerator

# from tabarena.benchmark.models.wrapper.AutoGluon_class import AGSingleBagWrapper
# from tabarena.benchmark.models.prep_ag.prep_lr.prep_lr_model import PrepLinearModel
# from tabarena.benchmark.models.prep_ag.prep_lgb.prep_lgb_model import PrepLGBModel
# from tabarena.benchmark.models.ag.tabpfnv2_5.tabpfnv2_5_model import RealTabPFNv25Model
# from tabarena.benchmark.models.prep_ag.prep_tabpfnv2_5.prep_tabpfnv2_5_model import PrepRealTabPFNv25Model
# from tabarena.benchmark.models.prep_ag.prep_tabm.prep_tabm_model import PrepTabMModel


# from autogluon.features import ArithmeticFeatureGenerator, OOFTargetEncodingFeatureGenerator, CategoricalInteractionFeatureGenerator, GroupByFeatureGenerator

# from autogluon.tabular.models import LGBModel, LinearModel, TabMModel


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from tabarena.icml2026.helpers import run_experiment

# import numpy as np
# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.neural_network import MLPRegressor


# def _neutral_numeric_signal(X, rng, n_rff=128, n_inter=20):
#     n, p = X.shape

#     W = rng.normal(0.0, 1.0, size=(p, n_rff))
#     b = rng.uniform(0.0, 2 * np.pi, size=n_rff)
#     rff = np.cos(X @ W + b)

#     H = np.tanh(X @ rng.normal(0.0, 1.0, size=(p, 32)))

#     inter = np.zeros((n, n_inter), dtype=float)
#     for k in range(n_inter):
#         i, j = rng.integers(0, p, size=2)
#         inter[:, k] = X[:, i] * X[:, j]

#     Z = np.concatenate([rff, H, inter], axis=1)
#     w = rng.normal(0.0, 1.0, size=Z.shape[1])
#     y = Z @ w
#     y = (y - y.mean()) / (y.std(ddof=0) + 1e-12)
#     return y


# def generate_train_test_data(
#     n=1000,
#     p=5,
#     cat_cardinalities=(4,),
#     test_size=0.25,
#     random_state=42,
#     target_model="neutral",  # "linear" | "rf" | "nn" | "neutral"
#     rf_params=None,
#     nn_params=None,
#     # UPDATED: includes slope component as 4th weight
#     component_weights=(0.4, 0.2, 0.2, 0.2),  # [numeric_base, total_single_cat, combo, slope]
#     # UPDATED: includes slope noise as 4th entry
#     noise_levels=(0.1, 0.1, 0.1, 0.1),       # [numeric_base_noise, per_cat_noise, combo_noise, slope_noise]
#     combo_offset_cfg=None,                   # required if w_combo > 0
#     offset_sigma=1.0,
#     combo_offset_sigma=None,
#     # slope configuration (still defines WHICH cat drives slopes + how heterogeneous)
#     slope_cfg=None,
#     x_cluster_cfg=None,
#     # slope_cfg example:
#     # {
#     #   "cat": "cat_1",          # name or 0-based index into categorical features
#     #   "strength": 1.0,         # spread of slope multipliers around 1.0
#     #   "mode": "multiplicative" # currently only multiplicative is supported here
#     # }
# ):
#     """
#     Target is a sum of up to 4 standardized components (each weighted + noisy):

#       y_final =
#           (w_num * num_base_std + noise) +
#           (w_single * sum_j single_cat_j_std + noise per cat) +
#           (w_combo * combo_std + noise) +
#           (w_slope * slope_component_std + noise)

#     Slope component (interaction-only) uses a categorical feature to modulate how numerics affect y:
#       g(cat) sampled per level around 1.0
#       slope_component_raw = y_base * g(cat) - y_base
#     """
#     rng = np.random.default_rng(random_state)

#     component_weights = np.asarray(component_weights, dtype=float)
#     noise_levels = np.asarray(noise_levels, dtype=float)

#     if component_weights.shape != (4,):
#         raise ValueError("component_weights must be length-4: [numeric_base, single_cat_total, combo, slope].")
#     if noise_levels.shape != (4,):
#         raise ValueError("noise_levels must be length-4: [numeric_base_noise, per_cat_noise, combo_noise, slope_noise].")
#     if np.any(component_weights < 0) or np.any(noise_levels < 0):
#         raise ValueError("component_weights and noise_levels must be non-negative.")

#     w_num, w_single_total, w_combo, w_slope = component_weights.tolist()
#     nl_num, nl_group, nl_combo, nl_slope = noise_levels.tolist()

#     # ----------------------------
#     # 1) Random numeric dataset
#     # ----------------------------
#     X = rng.normal(0.0, 1.0, size=(n, p))
#     y_random = rng.normal(0.0, 1.0, size=n)

#     num_cols = [f"x{i+1}" for i in range(p)]
#     df = pd.DataFrame(X, columns=num_cols)
#     df["y_random"] = y_random

#     # ----------------------------
#     # 2) Categorical features created early (slope depends on them)
#     # ----------------------------
#     if isinstance(cat_cardinalities, int):
#         cat_cardinalities = (cat_cardinalities,)
#     cat_cardinalities = tuple(int(c) for c in cat_cardinalities)
#     if len(cat_cardinalities) == 0:
#         raise ValueError("cat_cardinalities must contain at least one categorical feature.")
#     if any(c < 2 for c in cat_cardinalities):
#         raise ValueError("All categorical cardinalities must be >= 2.")

#     cat_cols = []
#     cat_levels = {}
#     cat_offsets = {}

#     for j, card in enumerate(cat_cardinalities, start=1):
#         col = f"cat_{j}"
#         levels = np.array([f"{col}_L{i}" for i in range(card)])
#         df[col] = rng.choice(levels, size=n, replace=True)
#         cat_cols.append(col)
#         cat_levels[col] = levels

#         if x_cluster_cfg is not None:
#             cat_col = x_cluster_cfg["cat"]
#             mean_sigma = x_cluster_cfg.get("mean_shift", 0.0)
#             scale_sigma = x_cluster_cfg.get("scale_shift", 0.0)

#             levels = cat_levels[cat_col]
#             mean_shifts = {
#                 lvl: rng.normal(0.0, mean_sigma, size=p) for lvl in levels
#             }
#             scale_shifts = {
#                 lvl: 1.0 + rng.normal(0.0, scale_sigma, size=p) for lvl in levels
#             }

#             for i in range(n):
#                 lvl = df.loc[i, cat_col]
#                 X[i] = X[i] * scale_shifts[lvl] + mean_shifts[lvl]

#             df[num_cols] = X

#     # ----------------------------
#     # 3) Base numeric signal y_base(X)
#     # ----------------------------
#     target_model_l = str(target_model).lower().strip()
#     if target_model_l in {"linear", "lin", "linearregression"}:
#         model = LinearRegression()
#         model.fit(df[num_cols].values, df["y_random"].values)
#         y_base = model.predict(df[num_cols].values)
#     elif target_model_l in {"rf", "random_forest", "randomforest"}:
#         params = {"n_estimators": 300, "random_state": random_state}
#         if rf_params:
#             params.update(rf_params)
#         model = RandomForestRegressor(**params)
#         model.fit(df[num_cols].values, df["y_random"].values)
#         y_base = model.predict(df[num_cols].values)
#     elif target_model_l in {"nn", "mlp", "neural", "neural_network"}:
#         params = {
#             "hidden_layer_sizes": (16, 16),
#             "activation": "relu",
#             "solver": "adam",
#             "max_iter": 500,
#             "random_state": random_state,
#         }
#         if nn_params:
#             params.update(nn_params)
#         model = MLPRegressor(**params)
#         model.fit(df[num_cols].values, df["y_random"].values)
#         y_base = model.predict(df[num_cols].values)
#     elif target_model_l == "neutral":
#         y_base = _neutral_numeric_signal(df[num_cols].values, rng)
#         model = None
#     else:
#         raise ValueError("target_model must be one of: 'linear', 'rf', 'nn', 'neutral'")

#     # standardize base numeric to unit std
#     y_base = (y_base - y_base.mean()) / (y_base.std(ddof=0) + 1e-12)

#     # ----------------------------
#     # 4) Build slope component (interaction-only) if requested / weighted
#     # ----------------------------
#     slope_info = None
#     if (w_slope > 0) and (slope_cfg is None):
#         raise ValueError("w_slope > 0 but slope_cfg is None. Provide slope_cfg={'cat':..., 'strength':...}.")

#     slope_component_raw = np.zeros(n, dtype=float)

#     if slope_cfg is not None and w_slope > 0:
#         if not isinstance(slope_cfg, dict):
#             raise ValueError("slope_cfg must be a dict or None.")

#         cat_ref = slope_cfg.get("cat", "cat_1")
#         strength = float(slope_cfg.get("strength", 1.0))
#         mode = str(slope_cfg.get("mode", "multiplicative")).lower().strip()
#         if mode != "multiplicative":
#             raise ValueError("This version supports slope_cfg['mode']='multiplicative' only.")

#         # resolve cat_ref to column name
#         if isinstance(cat_ref, int):
#             if cat_ref < 0 or cat_ref >= len(cat_cols):
#                 raise ValueError(f"slope_cfg['cat'] index out of range. cat_cols={cat_cols}")
#             slope_cat_col = cat_cols[cat_ref]
#         else:
#             slope_cat_col = str(cat_ref)
#             if slope_cat_col not in cat_cols:
#                 raise ValueError(f"slope_cfg['cat'] must be one of {cat_cols} (got {slope_cat_col!r}).")

#         levels = cat_levels[slope_cat_col]
#         slope_multipliers = {lvl: (1.0 + rng.normal(0.0, strength)) for lvl in levels}
#         g = df[slope_cat_col].map(slope_multipliers).to_numpy(dtype=float)

#         # interaction-only: deviation from global slope
#         # (so that if g=1 everywhere, slope_component_raw == 0)
#         slope_component_raw = y_base * g - y_base

#         slope_info = {
#             "cat": slope_cat_col,
#             "strength": strength,
#             "mode": mode,
#             "slope_multipliers": slope_multipliers,  # ground truth
#         }

#     # Standardize slope component to unit std (if non-degenerate)
#     slope_std = float(np.std(slope_component_raw, ddof=0))
#     slope_base = slope_component_raw / slope_std if slope_std > 0 else np.zeros(n, dtype=float)

#     # ----------------------------
#     # 5) Weight + noise numeric_base component
#     # ----------------------------
#     numeric_component = w_num * y_base
#     num_scale = float(np.std(numeric_component, ddof=0))
#     numeric_noise_sigma = nl_num * num_scale
#     numeric_component_noisy = numeric_component + rng.normal(0.0, numeric_noise_sigma, size=n)

#     # ----------------------------
#     # 6) Weight + noise slope component
#     # ----------------------------
#     slope_component = w_slope * slope_base
#     slope_scale = float(np.std(slope_component, ddof=0))
#     slope_noise_sigma = nl_slope * slope_scale
#     slope_component_noisy = slope_component + rng.normal(0.0, slope_noise_sigma, size=n)

#     # ----------------------------
#     # 7) Single categorical additive effects (split across cats)
#     # ----------------------------
#     m = len(cat_cols)
#     w_single_each = (w_single_total / m) if m > 0 else 0.0

#     single_components = []
#     for col in cat_cols:
#         levels = cat_levels[col]
#         offsets = {lvl: rng.normal(0.0, offset_sigma) for lvl in levels}
#         raw = df[col].map(offsets).to_numpy(dtype=float)

#         raw_std = float(np.std(raw, ddof=0))
#         raw_norm = raw / raw_std if raw_std > 0 else np.zeros(n, dtype=float)

#         comp_clean = w_single_each * raw_norm
#         comp_scale = float(np.std(comp_clean, ddof=0))
#         comp_noise_sigma = nl_group * comp_scale
#         comp_noisy = comp_clean + rng.normal(0.0, comp_noise_sigma, size=n)

#         single_components.append(comp_noisy)
#         cat_offsets[col] = offsets

#     single_total_component_noisy = np.sum(single_components, axis=0) if single_components else np.zeros(n)

#     # ----------------------------
#     # 8) Combo component (latent; not in X)
#     # ----------------------------
#     combo_offsets = None
#     combo_key_cols = None
#     combo_name = None
#     combo_component_noisy = np.zeros(n, dtype=float)
#     combo_offset_sigma_used = offset_sigma if combo_offset_sigma is None else float(combo_offset_sigma)

#     if w_combo > 0:
#         if combo_offset_cfg is None:
#             raise ValueError(
#                 "w_combo > 0 but combo_offset_cfg is None. "
#                 "Provide combo_offset_cfg={'cats':[...]} to define which categorical features to combine."
#             )
#         if not isinstance(combo_offset_cfg, dict):
#             raise ValueError("combo_offset_cfg must be a dict or None.")
#         cats = combo_offset_cfg.get("cats", None)
#         if cats is None:
#             raise ValueError("combo_offset_cfg must include 'cats' (list of indices or names).")

#         # resolve cats
#         if all(isinstance(c, int) for c in cats):
#             if any(c < 0 or c >= len(cat_cols) for c in cats):
#                 raise ValueError(f"'cats' indices out of range for cat_cols={cat_cols}.")
#             combo_key_cols = [cat_cols[i] for i in cats]
#         else:
#             combo_key_cols = [str(c) for c in cats]
#             missing = [c for c in combo_key_cols if c not in cat_cols]
#             if missing:
#                 raise ValueError(f"combo_offset_cfg cats not found: {missing}. Available: {cat_cols}")

#         if len(combo_key_cols) < 2:
#             raise ValueError("combo_offset_cfg 'cats' must reference at least 2 categorical features.")

#         combo_name = str(combo_offset_cfg.get("name", "latent_combo"))

#         combo_tuples = list(map(tuple, df[combo_key_cols].itertuples(index=False, name=None)))
#         unique_combos = sorted(set(combo_tuples))
#         combo_offsets = {cmb: rng.normal(0.0, combo_offset_sigma_used) for cmb in unique_combos}
#         combo_values = np.array([combo_offsets[cmb] for cmb in combo_tuples], dtype=float)

#         combo_std = float(np.std(combo_values, ddof=0))
#         combo_norm = combo_values / combo_std if combo_std > 0 else np.zeros(n, dtype=float)

#         combo_component = w_combo * combo_norm
#         combo_scale = float(np.std(combo_component, ddof=0))
#         combo_noise_sigma = nl_combo * combo_scale
#         combo_component_noisy = combo_component + rng.normal(0.0, combo_noise_sigma, size=n)
#     else:
#         combo_noise_sigma = 0.0

#     # ----------------------------
#     # Final target
#     # ----------------------------
#     df["y_final"] = (
#         numeric_component_noisy
#         + slope_component_noisy
#         + single_total_component_noisy
#         + combo_component_noisy
#     )

#     final_df = df[num_cols + cat_cols + ["y_final"]].copy()

#     # ----------------------------
#     # Train / test split
#     # ----------------------------
#     X_all = final_df[num_cols + cat_cols]
#     y_all = final_df["y_final"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X_all, y_all, test_size=test_size, random_state=random_state
#     )

#     target_type = "regression"

#     diagnostics = {
#         "component_weights": component_weights.tolist(),
#         "noise_levels": noise_levels.tolist(),
#         "numeric_noise_sigma": float(numeric_noise_sigma),
#         "slope_noise_sigma": float(slope_noise_sigma),
#         "single_each_weight": float(w_single_each),
#         "combo_noise_sigma": float(combo_noise_sigma),
#         "target_model": target_model_l,
#         "slope_used": None if slope_info is None else {
#             "cat": slope_info["cat"],
#             "strength": slope_info["strength"],
#             "mode": slope_info["mode"],
#         },
#     }

#     return {
#         "X_train": X_train,
#         "X_test": X_test,
#         "y_train": y_train,
#         "y_test": y_test,
#         "num_cols": num_cols,
#         "cat_cols": cat_cols,
#         "cat_levels": cat_levels,
#         "cat_offsets": cat_offsets,
#         "combo_offset_info": None if combo_offsets is None else {
#             "name": combo_name,
#             "cats": combo_key_cols,
#             "sigma": combo_offset_sigma_used,
#             "offsets": combo_offsets,
#         },
#         "slope_effect_info": slope_info,
#         "target_model_fitted": model,
#         "target_type": target_type,
#         "diagnostics": diagnostics,
#     }

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


def _neutral_numeric_signal(X, rng, n_rff=128, n_inter=20):
    n, p = X.shape
    W = rng.normal(0.0, 1.0, size=(p, n_rff))
    b = rng.uniform(0.0, 2 * np.pi, size=n_rff)
    rff = np.cos(X @ W + b)

    H = np.tanh(X @ rng.normal(0.0, 1.0, size=(p, 32)))

    inter = np.zeros((n, n_inter), dtype=float)
    for k in range(n_inter):
        i, j = rng.integers(0, p, size=2)
        inter[:, k] = X[:, i] * X[:, j]

    Z = np.concatenate([rff, H, inter], axis=1)
    w = rng.normal(0.0, 1.0, size=Z.shape[1])
    y = Z @ w
    y = (y - y.mean()) / (y.std(ddof=0) + 1e-12)
    return y


def generate_train_test_data(
    n=1000,
    p=5,
    cat_cardinalities=(4,),
    test_size=0.25,
    random_state=42,
    target_model="neutral",  # "linear" | "rf" | "nn" | "neutral"

    rf_params=None,
    nn_params=None,

    # component weights now include: [numeric_base, single_cat, combo, slope, centered_signal]
    component_weights=(0.3, 0.1, 0.1, 0.1, 0.4),
    # noise levels include: [numeric_base_noise, per_cat_noise, combo_noise, slope_noise, centered_signal_noise]
    noise_levels=(0.1, 0.1, 0.1, 0.1, 0.05),

    combo_offset_cfg=None,
    offset_sigma=1.0,
    combo_offset_sigma=None,

    slope_cfg=None,

    centering_cfg=None,
    # centering_cfg example:
    # {
    #   "cat": "cat_1",          # which categorical feature defines groups
    #   "num_col": "x1",         # which numeric column to shift by group means
    #   "mu_sigma": 5.0,         # between-group mean variation for that numeric column
    #   "eps_sigma": 1.0,        # within-group noise for that numeric column
    #   "target_fn": "linear",   # "linear" | "tanh" | "cubic"
    # }
):
    """
    Adds an OPTIONAL "centered signal" mechanism to enforce:

        y depends on (x - E[x|cat]) strongly,
        while E[x|cat] alone is not predictive.

    This makes the feature engineering operation:
        x - group_mean(x by cat)
    highly useful, while "just group mean" does not help.

    Final target is a sum of weighted standardized components:
        numeric_base + single_cat + combo + slope + centered_signal
    """
    rng = np.random.default_rng(random_state)

    component_weights = np.asarray(component_weights, dtype=float)
    noise_levels = np.asarray(noise_levels, dtype=float)
    if component_weights.shape != (5,):
        raise ValueError("component_weights must be length-5: [num_base, single_cat, combo, slope, centered].")
    if noise_levels.shape != (5,):
        raise ValueError("noise_levels must be length-5: [num_base, single_cat, combo, slope, centered].")
    if np.any(component_weights < 0) or np.any(noise_levels < 0):
        raise ValueError("component_weights and noise_levels must be non-negative.")

    w_num, w_single_total, w_combo, w_slope, w_center = component_weights.tolist()
    nl_num, nl_group, nl_combo, nl_slope, nl_center = noise_levels.tolist()

    # ----------------------------
    # 1) Base numeric dataset
    # ----------------------------
    X = rng.normal(0.0, 1.0, size=(n, p))
    y_random = rng.normal(0.0, 1.0, size=n)

    num_cols = [f"x{i+1}" for i in range(p)]
    df = pd.DataFrame(X, columns=num_cols)
    df["y_random"] = y_random

    # ----------------------------
    # 2) Categorical features
    # ----------------------------
    if isinstance(cat_cardinalities, int):
        cat_cardinalities = (cat_cardinalities,)
    cat_cardinalities = tuple(int(c) for c in cat_cardinalities)
    if len(cat_cardinalities) == 0:
        raise ValueError("cat_cardinalities must contain at least one categorical feature.")
    if any(c < 2 for c in cat_cardinalities):
        raise ValueError("All categorical cardinalities must be >= 2.")

    cat_cols, cat_levels, cat_offsets = [], {}, {}
    for j, card in enumerate(cat_cardinalities, start=1):
        col = f"cat_{j}"
        levels = np.array([f"{col}_L{i}" for i in range(card)])
        df[col] = rng.choice(levels, size=n, replace=True)
        cat_cols.append(col)
        cat_levels[col] = levels

    # ----------------------------
    # 3) OPTIONAL: enforce the "centering helps, group mean doesn't" pattern
    # ----------------------------
    centered_signal_raw = np.zeros(n, dtype=float)
    centering_info = None

    if centering_cfg is not None and w_center > 0:
        if not isinstance(centering_cfg, dict):
            raise ValueError("centering_cfg must be a dict or None.")

        cat_ref = centering_cfg.get("cat", "cat_1")
        num_col = centering_cfg.get("num_col", "x1")
        mu_sigma = float(centering_cfg.get("mu_sigma", 5.0))
        eps_sigma = float(centering_cfg.get("eps_sigma", 1.0))
        target_fn = str(centering_cfg.get("target_fn", "linear")).lower().strip()

        if isinstance(cat_ref, int):
            if cat_ref < 0 or cat_ref >= len(cat_cols):
                raise ValueError(f"centering_cfg['cat'] index out of range. cat_cols={cat_cols}")
            cat_col = cat_cols[cat_ref]
        else:
            cat_col = str(cat_ref)
            if cat_col not in cat_cols:
                raise ValueError(f"centering_cfg['cat'] must be one of {cat_cols} (got {cat_col!r}).")

        if num_col not in num_cols:
            raise ValueError(f"centering_cfg['num_col'] must be one of {num_cols} (got {num_col!r}).")

        # Create large between-group means for THIS numeric column
        levels = cat_levels[cat_col]
        mu_level = {lvl: rng.normal(0.0, mu_sigma) for lvl in levels}

        # Generate x = mu_g + eps (overwrites the chosen numeric column)
        eps = rng.normal(0.0, eps_sigma, size=n)
        mu = df[cat_col].map(mu_level).to_numpy(dtype=float)
        df[num_col] = mu + eps

        # Make target depend on within-group deviation ONLY (which equals eps)
        z = eps / (np.std(eps, ddof=0) + 1e-12)  # standardize within-group deviation

        if target_fn == "linear":
            centered_signal_raw = z
        elif target_fn == "tanh":
            centered_signal_raw = np.tanh(1.5 * z)
        elif target_fn == "cubic":
            centered_signal_raw = z**3
        else:
            raise ValueError("centering_cfg['target_fn'] must be one of: 'linear', 'tanh', 'cubic'.")

        centered_signal_raw = (centered_signal_raw - centered_signal_raw.mean()) / (centered_signal_raw.std(ddof=0) + 1e-12)

        centering_info = {
            "cat": cat_col,
            "num_col": num_col,
            "mu_sigma": mu_sigma,
            "eps_sigma": eps_sigma,
            "target_fn": target_fn,
            "mu_level": mu_level,  # ground-truth group means for the numeric column
        }

    # ----------------------------
    # 4) Base numeric signal y_base(X)
    # ----------------------------
    target_model_l = str(target_model).lower().strip()
    model = None

    if target_model_l in {"linear", "lin", "linearregression"}:
        model = LinearRegression()
        model.fit(df[num_cols].values, df["y_random"].values)
        y_base = model.predict(df[num_cols].values)
    elif target_model_l in {"rf", "random_forest", "randomforest"}:
        params = {"n_estimators": 300, "random_state": random_state}
        if rf_params:
            params.update(rf_params)
        model = RandomForestRegressor(**params)
        model.fit(df[num_cols].values, df["y_random"].values)
        y_base = model.predict(df[num_cols].values)
    elif target_model_l in {"nn", "mlp", "neural", "neural_network"}:
        params = {
            "hidden_layer_sizes": (16, 16),
            "activation": "relu",
            "solver": "adam",
            "max_iter": 500,
            "random_state": random_state,
        }
        if nn_params:
            params.update(nn_params)
        model = MLPRegressor(**params)
        model.fit(df[num_cols].values, df["y_random"].values)
        y_base = model.predict(df[num_cols].values)
    elif target_model_l == "neutral":
        y_base = _neutral_numeric_signal(df[num_cols].values, rng)
    else:
        raise ValueError("target_model must be one of: 'linear', 'rf', 'nn', 'neutral'")

    y_base = (y_base - y_base.mean()) / (y_base.std(ddof=0) + 1e-12)

    # ----------------------------
    # 5) Slope component (interaction-only) if enabled
    # ----------------------------
    slope_info = None
    slope_component_raw = np.zeros(n, dtype=float)
    if (w_slope > 0) and (slope_cfg is None):
        raise ValueError("w_slope > 0 but slope_cfg is None. Provide slope_cfg={'cat':..., 'strength':...}.")
    if slope_cfg is not None and w_slope > 0:
        cat_ref = slope_cfg.get("cat", "cat_1")
        strength = float(slope_cfg.get("strength", 1.0))
        mode = str(slope_cfg.get("mode", "multiplicative")).lower().strip()
        if mode != "multiplicative":
            raise ValueError("This version supports slope_cfg['mode']='multiplicative' only.")

        if isinstance(cat_ref, int):
            if cat_ref < 0 or cat_ref >= len(cat_cols):
                raise ValueError(f"slope_cfg['cat'] index out of range. cat_cols={cat_cols}")
            slope_cat_col = cat_cols[cat_ref]
        else:
            slope_cat_col = str(cat_ref)
            if slope_cat_col not in cat_cols:
                raise ValueError(f"slope_cfg['cat'] must be one of {cat_cols} (got {slope_cat_col!r}).")

        levels = cat_levels[slope_cat_col]
        slope_multipliers = {lvl: (1.0 + rng.normal(0.0, strength)) for lvl in levels}
        g = df[slope_cat_col].map(slope_multipliers).to_numpy(dtype=float)

        slope_component_raw = y_base * g - y_base
        slope_component_raw = (slope_component_raw - slope_component_raw.mean()) / (slope_component_raw.std(ddof=0) + 1e-12)

        slope_info = {
            "cat": slope_cat_col,
            "strength": strength,
            "mode": mode,
            "slope_multipliers": slope_multipliers,
        }

    # ----------------------------
    # 6) Additive single-cat components
    # ----------------------------
    m = len(cat_cols)
    w_single_each = (w_single_total / m) if m > 0 else 0.0

    single_total = np.zeros(n, dtype=float)
    for col in cat_cols:
        levels = cat_levels[col]
        offsets = {lvl: rng.normal(0.0, offset_sigma) for lvl in levels}
        raw = df[col].map(offsets).to_numpy(dtype=float)
        raw = raw / (np.std(raw, ddof=0) + 1e-12)

        comp = w_single_each * raw
        comp_noise_sigma = nl_group * (np.std(comp, ddof=0) + 1e-12)
        comp = comp + rng.normal(0.0, comp_noise_sigma, size=n)

        single_total += comp
        cat_offsets[col] = offsets

    # ----------------------------
    # 7) Combo component (latent)
    # ----------------------------
    combo_offsets = None
    combo_key_cols = None
    combo_name = None
    combo_component_noisy = np.zeros(n, dtype=float)
    combo_offset_sigma_used = offset_sigma if combo_offset_sigma is None else float(combo_offset_sigma)

    if w_combo > 0:
        if combo_offset_cfg is None:
            raise ValueError("w_combo > 0 but combo_offset_cfg is None. Provide combo_offset_cfg={'cats':[...]}.")

        cats = combo_offset_cfg.get("cats", None)
        if cats is None:
            raise ValueError("combo_offset_cfg must include 'cats' (list of indices or names).")

        if all(isinstance(c, int) for c in cats):
            if any(c < 0 or c >= len(cat_cols) for c in cats):
                raise ValueError(f"'cats' indices out of range for cat_cols={cat_cols}.")
            combo_key_cols = [cat_cols[i] for i in cats]
        else:
            combo_key_cols = [str(c) for c in cats]
            missing = [c for c in combo_key_cols if c not in cat_cols]
            if missing:
                raise ValueError(f"combo_offset_cfg cats not found: {missing}. Available: {cat_cols}")

        if len(combo_key_cols) < 2:
            raise ValueError("combo_offset_cfg 'cats' must reference at least 2 categorical features.")

        combo_name = str(combo_offset_cfg.get("name", "latent_combo"))

        combo_tuples = list(map(tuple, df[combo_key_cols].itertuples(index=False, name=None)))
        unique_combos = sorted(set(combo_tuples))
        combo_offsets = {cmb: rng.normal(0.0, combo_offset_sigma_used) for cmb in unique_combos}
        combo_values = np.array([combo_offsets[cmb] for cmb in combo_tuples], dtype=float)

        combo_values = combo_values / (np.std(combo_values, ddof=0) + 1e-12)
        combo_component = w_combo * combo_values
        combo_noise_sigma = nl_combo * (np.std(combo_component, ddof=0) + 1e-12)
        combo_component_noisy = combo_component + rng.normal(0.0, combo_noise_sigma, size=n)

    # ----------------------------
    # 8) Weight + noise for numeric_base, slope, centered components
    # ----------------------------
    numeric_component = w_num * y_base
    numeric_noise_sigma = nl_num * (np.std(numeric_component, ddof=0) + 1e-12)
    numeric_component_noisy = numeric_component + rng.normal(0.0, numeric_noise_sigma, size=n)

    slope_component = w_slope * slope_component_raw
    slope_noise_sigma = nl_slope * (np.std(slope_component, ddof=0) + 1e-12)
    slope_component_noisy = slope_component + rng.normal(0.0, slope_noise_sigma, size=n)

    centered_component = w_center * centered_signal_raw
    centered_noise_sigma = nl_center * (np.std(centered_component, ddof=0) + 1e-12)
    centered_component_noisy = centered_component + rng.normal(0.0, centered_noise_sigma, size=n)

    # ----------------------------
    # Final target
    # ----------------------------
    df["y_final"] = (
        numeric_component_noisy
        + slope_component_noisy
        + single_total
        + combo_component_noisy
        + centered_component_noisy
    )

    final_df = df[num_cols + cat_cols + ["y_final"]].copy()

    X_all = final_df[num_cols + cat_cols]
    y_all = final_df["y_final"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=random_state
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_levels": cat_levels,
        "cat_offsets": cat_offsets,
        "combo_offset_info": None if combo_offsets is None else {
            "name": combo_name,
            "cats": combo_key_cols,
            "sigma": combo_offset_sigma_used,
            "offsets": combo_offsets,
        },
        "slope_effect_info": slope_info,
        "centering_effect_info": centering_info,  # <- tells you which (cat,num) pair has the centering pattern
        "target_model_fitted": model,
        "target_type": "regression",
        "diagnostics": {
            "component_weights": component_weights.tolist(),
            "noise_levels": noise_levels.tolist(),
        },
    }


# Example:
# out = generate_train_test_data(
#     cat_cardinalities=(4, 3, 5),
#     target_model="nn",
#     combo_offset_cfg={"cats": [0, 2], "sigma": 1.5, "name": "cat1_x_cat3"}
# )

experiment_params = {
    # "groupby_test_linear4_bag": { # NOT LINEAR, but shows very clearly how much better GBM & TabM get with groupby, while TabPFN doesnt gain much from MEAN, and really nothing from REL
    #     "n": 1000,
    #     "p": 5,
    #     "cat_cardinalities": (100, ),
    #     "target_model": "linear",
    #     "component_weights": (0.05, 0.0, 0.0, 0.0, 0.95),  # w_center huge
    #     "noise_levels": (0.1, 0.0, 0.0, 0.0, 0.05),
    #     "centering_cfg": {
    #         "cat": "cat_1",
    #         "num_col": "x1",
    #         "mu_sigma": 10.0,    # big between-group shifts in x1
    #         "eps_sigma": 1.0,   # smaller within-group variation
    #         "target_fn": "linear",
    #     },
    #     "prep_types": ["None", "MEAN-GROUPBY", "REL-GROUPBY-keepMean", "REL-GROUPBY-meansubtract", "REL-GROUPBY-noPCT", "REL-GROUPBY", "OOF-TE"],
    # },
    # "groupby_test_new": { # Rel-groupby strong for trees, but PFNs can do it already
    #     "n": 1000,
    #     "p": 5,
    #     "cat_cardinalities": (10, ),
    #     "target_model": "rf",
    #     "component_weights": (0.05, 0.0, 0.0, 0.0, 0.95),  # w_center huge
    #     "noise_levels": (0.1, 0.0, 0.0, 0.0, 0.05),
    #     "centering_cfg": {
    #         "cat": "cat_1",
    #         "num_col": "x1",
    #         "mu_sigma": 10.0,    # big between-group shifts in x1
    #         "eps_sigma": 1.0,   # smaller within-group variation
    #         "target_fn": "linear",
    #     },
    #     "prep_types": ["None", "DROP-CAT", "DROP-NUM", "MEAN-GROUPBY", "OOF-TE", "REL-GROUPBY"],
    # },
    # "groupby_test": { # CURRENTLY SAVED AS groupby_test
    #     "n": 1000,
    #     "p": 5,
    #     "cat_cardinalities": (50, 50),
    #     "target_model": "linear",
    #     "component_weights": (0.2, 0.2, 0.0, 0.6),  # slope dominates
    #     "noise_levels": (0.2, 0.2, 0.0, 0.2),
    #     "slope_cfg": {"cat": "cat_1", "strength": 2.0},
    #     "x_cluster_cfg": {"cat": "cat_1", 
    #                       "mean_shift": 1.8,     # strength of cluster separation
    #                       "scale_shift": 1.3,    # optional
    #                       },
    #     "prep_types": ["None", "DROP-CAT", "DROP-NUM", "MEAN-GROUPBY", "OOF-TE", "REL-GROUPBY"],
    # },


    # "single_feature_high_cardinality_linear": {
    #     "target_model": "linear",   # "linear" | "rf" | "nn"
    #     "n":  10000,
    #     "cat_cardinalities": (2000,),
    #     "component_weights": (0.5, 0.5, 0.),   # combo dominates
    #     "noise_levels": (0.1, 0.5, 0.0),  # combo noise very low
    #     "prep_types": ["None", "DROP-CAT", "DROP-NUM", "OOF-TE", "OOF-TE-APPEND", "TE", "LOO"],
    #     },
    # "single_feature_high_cardinality_linear": {
    #     "target_model": "linear",   # "linear" | "rf" | "nn"
    #     "n":  1000,
    #     "cat_cardinalities": (500,),
    #     "component_weights": (0.5, 0.5, 0.),   # combo dominates
    #     "noise_levels": (0.1, 0.8, 0.0),  # combo noise very low
    #     "prep_types": ["None", "DROP-CAT", "DROP-NUM", "OOF-TE", "OOF-TE-APPEND", "TE", "LOO"],
    #     },
    # "single_feature_high_cardinality_rf": {
    #     "target_model": "rf",   # "linear" | "rf" | "nn"
    #     "n":  1000,
    #     "cat_cardinalities": (500,),
    #     "component_weights": (0.5, 0.5, 0.),   # combo dominates
    #     "noise_levels": (0.1, 0.8, 0.0),  # combo noise very low
    #     "prep_types": ["None", "DROP-CAT", "DROP-NUM", "OOF-TE", "OOF-TE-APPEND", "TE", "LOO"],
    #     },
    # "single_feature_high_cardinality_nn": {
    #     "target_model": "nn",   # "linear" | "rf" | "nn"
    #     "n":  1000,
    #     "cat_cardinalities": (500,),
    #     "component_weights": (0.5, 0.5, 0.),   # combo dominates
    #     "noise_levels": (0.1, 0.8, 0.0),  # combo noise very low
    #     "prep_types": ["None", "DROP-CAT", "DROP-NUM", "OOF-TE", "OOF-TE-APPEND", "TE", "LOO"],
    #     },

    # "combine_hard_to_learn_linear": {
    #     "target_model": "linear",   # "linear" | "rf" | "nn"
    #     "n":  1000,
    #     "cat_cardinalities": (50,50),
    #     "combo_offset_cfg": {"cats": ["cat_1", "cat_2"], "sigma": 1.5, "name": "cat1_x_cat2"},
    #     "component_weights": (0.4, 0.2, 0.4),   # combo dominates
    #     "noise_levels": (0.1, 0.1, 0.1),  # combo noise very low
    #     "prep_types": ["None", "DROP-CAT", "DROP-NUM", "OOF-TE", "OOF-TE-APPEND", "TE", "LOO", "CATINT", "CATINT_TE", "CATINT_LOO", "CATINT_OOFTE"],
    #     },
    # "combine_hard_to_learn_rf": {
    #     "target_model": "rf",   # "linear" | "rf" | "nn"
    #     "n":  1000,
    #     "cat_cardinalities": (50,50),
    #     "combo_offset_cfg": {"cats": ["cat_1", "cat_2"], "sigma": 1.5, "name": "cat1_x_cat2"},
    #     "component_weights": (0.4, 0.2, 0.4),   # combo dominates
    #     "noise_levels": (0.1, 0.1, 0.1),  # combo noise very low
    #     "prep_types": ["None", "DROP-CAT", "DROP-NUM", "OOF-TE", "OOF-TE-APPEND", "TE", "LOO", "CATINT", "CATINT_TE", "CATINT_LOO", "CATINT_OOFTE"],
    #     },
    # "combine_hard_to_learn_nn": {
    #     "target_model": "nn",   # "linear" | "rf" | "nn"
    #     "n":  1000,
    #     "cat_cardinalities": (50,50),
    #     "combo_offset_cfg": {"cats": ["cat_1", "cat_2"], "sigma": 1.5, "name": "cat1_x_cat2"},
    #     "component_weights": (0.4, 0.2, 0.4),   # combo dominates
    #     "noise_levels": (0.1, 0.1, 0.1),  # combo noise very low
    #     "prep_types": ["None", "DROP-CAT", "DROP-NUM", "OOF-TE", "OOF-TE-APPEND", "TE", "LOO", "CATINT", "CATINT_TE", "CATINT_LOO", "CATINT_OOFTE"],
    #     },


    # "single_high_cardinality": {
    #     "target_model": "nn",   # "linear" | "rf" | "nn"
    #     "n":  10000,
    #     "cat_cardinalities": (1000,),
    #     "component_weights": (0.5, 0.5, 0.),   # combo dominates
    #     "noise_levels": (0.1, 0.2, 0.2),  # combo noise very low
    #     "prep_types": ["None", "DROP-CAT", "DROP-NUM", "OOF-TE", "TE", "LOO"],
    #     },
    # "TE_overfits_small": {
    #     "target_model": "linear",   # "linear" | "rf" | "nn"
    #     "n":  1000,
    #     "cat_cardinalities": (500,),
    #     "component_weights": (0.5, 0.5, 0.),   # combo dominates
    #     "noise_levels": (0.1, 0.8, 0.0),  # combo noise very low
    #     "prep_types": ["None", "DROP-CAT", "DROP-NUM", "OOF-TE", "OOF-TE-APPEND", "TE", "LOO"],
    #     },

    ##################{"n_estimators": 300, "random_state": random_state}
       "combine_rf": {
        "target_model": "rf",   # "linear" | "rf" | "nn"
        "rf_params": {"n_estimators": 50, "max_depth": 2},
        "n":  1000,
        "cat_cardinalities": (50,50),
        "combo_offset_cfg": {"cats": ["cat_1", "cat_2"], "sigma": 1.5, "name": "cat1_x_cat2"},
        "component_weights": (.45, 0.05, 0.5, 0.0, 0.0),  # w_center huge
        "noise_levels": (0.1, 0.1, 0.2, 0.0, 0.0),
        "prep_types": ["None", "DROP-CAT", "DROP-NUM", "OOF-TE", "OOF-TE-APPEND", "CATINT", "CATINT_OOFTE"],
        },
    "three_feature_high_cardinality_uninformative_rf_50trees_depth5": { # 
        "n": 1000,
        "p": 5,
        "cat_cardinalities": (200, 200,200),
        "target_model": "rf",
        "rf_params": {"n_estimators": 50, "max_depth": 2},
        "component_weights": (.95, 0.05, 0.0, 0.0, 0.0),  # w_center huge
        "noise_levels": (0.1, 0.5, 0.0, 0.0, 0.0),
        "prep_types": ["None", "DROP-CAT", "DROP-NUM", "OOF-TE", "OOF-TE-Smooth20", "TE", "LOO"],
    },    
    "three_feature_high_cardinality_uninformative_rf_50trees": { # 
        "n": 1000,
        "p": 5,
        "cat_cardinalities": (200, 200,200),
        "target_model": "rf",
        "rf_params": {"n_estimators": 50, "max_depth": 2},
        "component_weights": (.95, 0.05, 0.0, 0.0, 0.0),  # w_center huge
        "noise_levels": (0.1, 0.5, 0.0, 0.0, 0.0),
        "prep_types": ["None", "DROP-CAT", "DROP-NUM", "OOF-TE", "OOF-TE-Smooth20", "TE", "LOO"],
    },    
    "three_feature_high_cardinality_rf_100trees": { # 
        "n": 1000,
        "p": 5,
        "rf_params": {"n_estimators": 100},
        "cat_cardinalities": (200, 200, 200),
        "target_model": "rf",
        "component_weights": (0.5, 0.5, 0.0, 0.0, 0.0),  # w_center huge
        "noise_levels": (0.1, 0.1, 0.0, 0.0, 0.0),
        "prep_types": ["None", "DROP-CAT", "DROP-NUM", "OOF-TE", "OOF-TE-APPEND", "TE", "LOO"],
    },
    "three_feature_high_cardinality_uninformative_nn": { # 
        "n": 1000,
        "p": 5,
        "cat_cardinalities": (200, 200,200),
        "target_model": "nn",
        "component_weights": (.95, 0.05, 0.0, 0.0, 0.0),  # w_center huge
        "noise_levels": (0.1, 0.5, 0.0, 0.0, 0.0),
        "prep_types": ["None", "DROP-CAT", "DROP-NUM", "OOF-TE", "OOF-TE-Smooth20", "TE", "LOO"],
    },    
    "three_feature_high_cardinality_rf": { # 
        "n": 1000,
        "p": 5,
        "cat_cardinalities": (200, 200, 200),
        "target_model": "rf",
        "component_weights": (0.5, 0.5, 0.0, 0.0, 0.0),  # w_center huge
        "noise_levels": (0.1, 0.1, 0.0, 0.0, 0.0),
        "prep_types": ["None", "DROP-CAT", "DROP-NUM", "OOF-TE", "OOF-TE-APPEND", "TE", "LOO"],
    },
    "single_feature_high_cardinality_rf": { # 
        "n": 1000,
        "p": 5,
        "cat_cardinalities": (500, ),
        "target_model": "rf",
        "component_weights": (0.5, 0.5, 0.0, 0.0, 0.0),  # w_center huge
        "noise_levels": (0.1, 0.8, 0.0, 0.0, 0.0),
        "prep_types": ["None", "DROP-CAT", "DROP-NUM", "OOF-TE", "OOF-TE-APPEND", "TE", "LOO"],
    },
    "groupby_test_with_slope": { # 
        "n": 1000,
        "p": 5,
        "cat_cardinalities": (100, ),
        "target_model": "rf",
        "component_weights": (0.3, 0.1, 0.0, 0.3, 0.3),  # w_center huge
        "noise_levels": (0.2, 0.2, 0.0, 0.0, 0.2),
        "centering_cfg": {
            "cat": "cat_1",
            "num_col": "x1",
            "mu_sigma": 10.0,    # big between-group shifts in x1
            "eps_sigma": 2.0,   # smaller within-group variation
            "target_fn": "tanh",
        },
        "slope_cfg": {"cat": "cat_1", "strength": 2.0},
        "prep_types": ["None", "DROP-CAT", "DROP-NUM", "MEAN-GROUPBY", "REL-GROUPBY-keepMean", "REL-GROUPBY-meansubtract", "REL-GROUPBY-noPCT", "REL-GROUPBY", "OOF-TE"],
    },
    "groupby_test_with_slope_10ksamples": { # 
        "n": 10000,
        "p": 5,
        "cat_cardinalities": (1000, ),
        "target_model": "rf",
        "component_weights": (0.3, 0.1, 0.0, 0.3, 0.3),  # w_center huge
        "noise_levels": (0.2, 0.2, 0.0, 0.0, 0.2),
        "centering_cfg": {
            "cat": "cat_1",
            "num_col": "x1",
            "mu_sigma": 10.0,    # big between-group shifts in x1
            "eps_sigma": 2.0,   # smaller within-group variation
            "target_fn": "tanh",
        },
        "slope_cfg": {"cat": "cat_1", "strength": 2.0},
        "prep_types": ["None", "DROP-CAT", "DROP-NUM", "MEAN-GROUPBY", "REL-GROUPBY-keepMean", "REL-GROUPBY-meansubtract", "REL-GROUPBY-noPCT", "REL-GROUPBY", "OOF-TE"],
    },

    # noise levels include: [numeric_base_noise, per_cat_noise, combo_noise, slope_noise, centered_signal_noise]
    }


if __name__ == "__main__":
    save_path = "tabarena/tabarena/tabarena/icml2026/results"
    exp_name = "simulated_cat_experiments_bag"

    num_bag_folds = 8
    verbosity = 0

    for param_set_name, params in experiment_params.items():
        print(f"=== Experiment: {param_set_name} ===")
        if os.path.exists(os.path.join(save_path, f'{exp_name}_{param_set_name}_results.pkl')):
            print(" Results already exist, skipping...")
            continue
        
        # n = params.get("n", 1000)
        # cat_cardinalities = params.get("cat_cardinalities", (4,4))
        # target_model = params.get("target_model", "linear")
        # combo_offset_cfg = params.get("combo_offset_cfg", None)
        # component_weights = params.get("component_weights", (0.5,0.2,0.3))
        # noise_levels = params.get("noise_levels", (0.1,0.1,0.1))
        prep_types = params.pop("prep_types", ["None"])
        data = generate_train_test_data(**params,#n=n, 
                                        #cat_cardinalities=cat_cardinalities, 
                                        # alpha=alpha,
                                        # component_weights=component_weights,
                                        # noise_levels=noise_levels,
                                        # target_model=target_model,
                                        # combo_offset_cfg=combo_offset_cfg, 
                                        # cat_vs_num_var_ratio=cat_vs_num_var_ratio,
                                        # cat_vs_num_reference=cat_vs_num_reference,
                                        # group_noise_frac=group_noise_variance,
                                        # combo_vs_single_var_ratio=combo_vs_single_var_ratio,
                                        )
        X = data["X_train"]
        y = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        target_type = data["target_type"]
        category_offsets = data["cat_offsets"]
        num_cols = data["num_cols"]
        
        from sklearn.metrics import log_loss, roc_auc_score, root_mean_squared_error
        if target_type == 'regression':
            metric = root_mean_squared_error
        elif target_type == 'binary':
            metric = lambda x,y: 1-roc_auc_score(x, y)
        else:
            metric = log_loss
        
        print(f" Base score: {metric(y_test, np.ones_like(y_test)*y.mean())}")
        if params.get("target_model", "neutral") != "neutral":
            print(f"Score from numeric model {metric(y_test,data["target_model_fitted"].predict(X_test.select_dtypes(np.number)))}")
        print(f" Base score by target mean: {metric(y_test, np.ones_like(y_test)*y.mean())}")

        ag_prep = AutoMLPipelineFeatureGenerator()
        X = ag_prep.fit_transform(X, y)
        X_test = ag_prep.transform(X_test)

        results = {"preds": {}, "performance": {}}

        # TODO: Properly add GBM-OHE
        for model_name in ["LR", "TABM", "GBM", "CAT", "PFN"]:
            print("--"*20)
            results["preds"][model_name] = {}
            results["performance"][model_name] = {}

            for prep_type in prep_types:
                preds, performance, X_used = run_experiment(X, y, X_test, y_test, model_name, prep_type, target_type, verbosity=verbosity, num_bag_folds=num_bag_folds)
                results["performance"][model_name][prep_type] = performance
                results["preds"][model_name][prep_type] = preds
                results["params"] = params
                print(f"Dataset: {param_set_name}, Model: {model_name}, Prep: {prep_type} (shape=[{X_used.shape}]), Performance: {performance:.4f}")

        with open(os.path.join(save_path, f'{exp_name}_{param_set_name}_results.pkl'), 'wb') as f:
            pickle.dump(results, f) 


