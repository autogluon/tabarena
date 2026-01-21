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

def _neutral_numeric_signal(X, rng, n_rff=128, n_inter=20):
    # X: (n, p)
    n, p = X.shape

    # 1) random Fourier-ish features (smooth, periodic)
    W = rng.normal(0.0, 1.0, size=(p, n_rff))
    b = rng.uniform(0.0, 2*np.pi, size=n_rff)
    rff = np.cos(X @ W + b)  # (n, n_rff)

    # 2) small tanh MLP-style random features (smooth, non-linear)
    H = np.tanh(X @ rng.normal(0.0, 1.0, size=(p, 32)))

    # 3) weak random pairwise interactions (gives some “tree-ish” and “nn-ish” structure)
    inter = np.zeros((n, n_inter))
    for k in range(n_inter):
        i, j = rng.integers(0, p, size=2)
        inter[:, k] = X[:, i] * X[:, j]

    Z = np.concatenate([rff, H, inter], axis=1)

    # random linear mix -> numeric signal
    w = rng.normal(0.0, 1.0, size=Z.shape[1])
    y = Z @ w

    # standardize
    y = (y - y.mean()) / (y.std(ddof=0) + 1e-12)
    return y


def generate_train_test_data(
    n=1000,
    p=5,
    cat_cardinalities=(4,),
    test_size=0.25,
    random_state=42,
    target_model="linear",             # "linear" | "rf" | "nn"
    rf_params=None,
    nn_params=None,
    # NEW requested controls
    component_weights=(0.5, 0.2, 0.3), # [numeric, total_single_cat, combo_cat]
    noise_levels=(0.1, 0.1, 0.1),      # [numeric_noise, per_cat_group_noise, combo_group_noise]
    # combo config (required to generate combo component)
    combo_offset_cfg=None,             # {"cats":[...]} indices or names; optional "name"
    # offset base scales (used only to *sample* raw offsets before rescaling by weights)
    offset_sigma=1.0,
    combo_offset_sigma=None,           # if None, uses offset_sigma
):
    """
    Generates synthetic regression data with:
      - numeric features x1..xp ~ N(0,1)
      - categorical features cat_1..cat_m with user-defined cardinalities
      - an optional latent combo effect from a combination of multiple categorical features (NOT in X)

    You control final target composition via:
      component_weights = [w_num, w_single_total, w_combo]
        - numeric component strength: w_num
        - total strength of individual categorical effects: w_single_total
          (split evenly across categorical features)
        - strength of the combo component: w_combo

      noise_levels = [nl_num, nl_group, nl_combo]
        - nl_num: noise added to numeric component, relative to its own scale
        - nl_group: noise added to each categorical feature's group effect, relative to that feature effect scale
        - nl_combo: noise added to combo component, relative to combo scale

    Interpretation of "noise relative to scale":
      For a base component z, we add eps ~ N(0, (nl * std(z))^2)

    Final target:
      y_final = numeric_component_noisy
                + sum_j single_cat_component_j_noisy
                + combo_component_noisy
    """
    rng = np.random.default_rng(random_state)

    # ----------------------------
    # Validate requested parameters
    # ----------------------------
    component_weights = np.asarray(component_weights, dtype=float)
    noise_levels = np.asarray(noise_levels, dtype=float)

    if component_weights.shape != (3,):
        raise ValueError("component_weights must be length-3: [numeric, single_cat_total, combo].")
    if noise_levels.shape != (3,):
        raise ValueError("noise_levels must be length-3: [numeric, per_cat_group, combo].")
    if np.any(component_weights < 0):
        raise ValueError("component_weights must be non-negative.")
    if np.any(noise_levels < 0):
        raise ValueError("noise_levels must be non-negative.")

    w_num, w_single_total, w_combo = component_weights.tolist()
    nl_num, nl_group, nl_combo = noise_levels.tolist()

    # ----------------------------
    # 1) Random numeric dataset
    # ----------------------------
    X = rng.normal(0.0, 1.0, size=(n, p))
    y_random = rng.normal(0.0, 1.0, size=n)

    num_cols = [f"x{i+1}" for i in range(p)]
    df = pd.DataFrame(X, columns=num_cols)
    df["y_random"] = y_random

    # ----------------------------
    # 2) Fit chosen model to create the numeric signal
    # ----------------------------
    target_model_l = str(target_model).lower().strip()
    if target_model_l in {"linear", "lin", "linearregression"}:
        model = LinearRegression()
    elif target_model_l in {"rf", "random_forest", "randomforest"}:
        params = {"n_estimators": 300, "random_state": random_state}
        if rf_params:
            params.update(rf_params)
        model = RandomForestRegressor(**params)
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
    elif target_model_l == "neutral":
        pass  # handled below
    else:
        raise ValueError("target_model must be one of: 'linear', 'rf', 'nn'")

    if target_model == "neutral":
        y_pred = _neutral_numeric_signal(df[num_cols].values, rng)
        model = None  # no fitted model
    else:
        model.fit(df[num_cols].values, df["y_random"].values)
        y_pred = model.predict(df[num_cols].values)

    # Normalize numeric component to unit std, then weight it
    y_pred_std = float(np.std(y_pred, ddof=0))
    numeric_base = y_pred / y_pred_std if y_pred_std > 0 else np.zeros(n, dtype=float)
    numeric_component = w_num * numeric_base

    # Add numeric noise relative to numeric_component scale
    num_scale = float(np.std(numeric_component, ddof=0))
    numeric_noise_sigma = nl_num * num_scale
    numeric_component_noisy = numeric_component + rng.normal(0.0, numeric_noise_sigma, size=n)

    # ----------------------------
    # 3) Categorical features + single (per-feature) effects
    # ----------------------------
    if isinstance(cat_cardinalities, int):
        cat_cardinalities = (cat_cardinalities,)
    cat_cardinalities = tuple(int(c) for c in cat_cardinalities)
    if len(cat_cardinalities) == 0:
        raise ValueError("cat_cardinalities must contain at least one categorical feature.")
    if any(c < 2 for c in cat_cardinalities):
        raise ValueError("All categorical cardinalities must be >= 2.")

    cat_cols = []
    cat_levels = {}
    cat_offsets = {}

    m = len(cat_cardinalities)
    # split the total single-cat weight equally across categorical features
    w_single_each = (w_single_total / m) if m > 0 else 0.0

    single_components = []          # list of per-feature single cat components (after weight+noise)
    single_components_clean = []    # before per-feature noise (for diagnostics)

    for j, card in enumerate(cat_cardinalities, start=1):
        col = f"cat_{j}"
        levels = np.array([f"{col}_L{i}" for i in range(card)])
        df[col] = rng.choice(levels, size=n, replace=True)

        offsets = {lvl: rng.normal(0.0, offset_sigma) for lvl in levels}
        raw = df[col].map(offsets).to_numpy(dtype=float)

        # normalize to unit std, then weight it
        raw_std = float(np.std(raw, ddof=0))
        raw_norm = raw / raw_std if raw_std > 0 else np.zeros(n, dtype=float)
        comp_clean = w_single_each * raw_norm

        # add per-feature group noise relative to that feature's scale
        comp_scale = float(np.std(comp_clean, ddof=0))
        comp_noise_sigma = nl_group * comp_scale
        comp_noisy = comp_clean + rng.normal(0.0, comp_noise_sigma, size=n)

        single_components_clean.append(comp_clean)
        single_components.append(comp_noisy)

        cat_cols.append(col)
        cat_levels[col] = levels
        cat_offsets[col] = offsets

    single_total_component_noisy = np.sum(single_components, axis=0) if single_components else np.zeros(n)

    # ----------------------------
    # 4) OPTIONAL latent combo component (NOT included in X)
    # ----------------------------
    combo_offsets = None
    combo_key_cols = None
    combo_name = None
    combo_values = np.zeros(n, dtype=float)
    combo_component_noisy = np.zeros(n, dtype=float)
    combo_offset_sigma_used = offset_sigma if combo_offset_sigma is None else float(combo_offset_sigma)

    if w_combo > 0:
        if combo_offset_cfg is None:
            raise ValueError(
                "component_weights[2] (combo weight) > 0 but combo_offset_cfg is None. "
                "Provide combo_offset_cfg={'cats':[...]} to define which categorical features to combine."
            )
        if not isinstance(combo_offset_cfg, dict):
            raise ValueError("combo_offset_cfg must be a dict or None.")
        cats = combo_offset_cfg.get("cats", None)
        if cats is None:
            raise ValueError("combo_offset_cfg must include 'cats' (list of indices or names).")

        # Accept indices [0,2] (0-based into cat_cols), or names ["cat_1","cat_3"]
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

        # normalize to unit std, then weight it
        combo_std = float(np.std(combo_values, ddof=0))
        combo_norm = combo_values / combo_std if combo_std > 0 else np.zeros(n, dtype=float)
        combo_component = w_combo * combo_norm

        # add combo noise relative to combo component scale
        combo_scale = float(np.std(combo_component, ddof=0))
        combo_noise_sigma = nl_combo * combo_scale
        combo_component_noisy = combo_component + rng.normal(0.0, combo_noise_sigma, size=n)
    else:
        combo_noise_sigma = 0.0

    # ----------------------------
    # Final target
    # ----------------------------
    df["y_final"] = numeric_component_noisy + single_total_component_noisy + combo_component_noisy

    final_df = df[num_cols + cat_cols + ["y_final"]].copy()

    # ----------------------------
    # Train / test split
    # ----------------------------
    X_all = final_df[num_cols + cat_cols]
    y_all = final_df["y_final"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=random_state
    )

    target_type = "regression"

    # ----------------------------
    # Useful diagnostics / metadata
    # ----------------------------
    diagnostics = {
        "component_weights": component_weights.tolist(),
        "noise_levels": noise_levels.tolist(),
        "numeric_component_std_clean": float(np.std(numeric_component, ddof=0)),
        "numeric_noise_sigma": float(numeric_noise_sigma),
        "single_each_weight": float(w_single_each),
        "single_total_std_noisy": float(np.std(single_total_component_noisy, ddof=0)),
        "combo_std_noisy": float(np.std(combo_component_noisy, ddof=0)),
        "combo_noise_sigma": float(combo_noise_sigma),
    }

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_levels": cat_levels,
        "cat_offsets": cat_offsets,  # per-feature level->raw_offset (pre-normalization)
        "combo_offset_info": None if combo_offsets is None else {
            "name": combo_name,
            "cats": combo_key_cols,
            "sigma": combo_offset_sigma_used,
            "offsets": combo_offsets,  # tuple(levels...)->raw_offset (pre-normalization)
        },
        "target_model_fitted": model,
        "target_type": target_type,
        "diagnostics": diagnostics,
    }



# Example:
# out = generate_train_test_data(
#     cat_cardinalities=(4, 3, 5),
#     target_model="nn",
#     combo_offset_cfg={"cats": [0, 2], "sigma": 1.5, "name": "cat1_x_cat3"}
# )

experiment_params = {
    "quick_test": {
        "target_model": "linear",   # "linear" | "rf" | "nn"
        "n":  1000,
        "cat_cardinalities": (100,),
        "component_weights": (0.5, 0.5, 0.),   # combo dominates
        "noise_levels": (0.1, 0.5, 0.0),  # combo noise very low
        "prep_types": ["None", "DROP-CAT", "DROP-NUM", "OOF-TE", "OOF-TE-APPEND", "TE", "LOO"],
        },
    # "single_feature_high_cardinality_linear": {
    #     "target_model": "linear",   # "linear" | "rf" | "nn"
    #     "n":  10000,
    #     "cat_cardinalities": (2000,),
    #     "component_weights": (0.5, 0.5, 0.),   # combo dominates
    #     "noise_levels": (0.1, 0.5, 0.0),  # combo noise very low
    #     "prep_types": ["None", "DROP-CAT", "DROP-NUM", "OOF-TE", "OOF-TE-APPEND", "TE", "LOO"],
    #     },
    "single_feature_high_cardinality_linear": {
        "target_model": "linear",   # "linear" | "rf" | "nn"
        "n":  1000,
        "cat_cardinalities": (500,),
        "component_weights": (0.5, 0.5, 0.),   # combo dominates
        "noise_levels": (0.1, 0.8, 0.0),  # combo noise very low
        "prep_types": ["None", "DROP-CAT", "DROP-NUM", "OOF-TE", "OOF-TE-APPEND", "TE", "LOO"],
        },
    "single_feature_high_cardinality_rf": {
        "target_model": "rf",   # "linear" | "rf" | "nn"
        "n":  1000,
        "cat_cardinalities": (500,),
        "component_weights": (0.5, 0.5, 0.),   # combo dominates
        "noise_levels": (0.1, 0.8, 0.0),  # combo noise very low
        "prep_types": ["None", "DROP-CAT", "DROP-NUM", "OOF-TE", "OOF-TE-APPEND", "TE", "LOO"],
        },
    "single_feature_high_cardinality_nn": {
        "target_model": "nn",   # "linear" | "rf" | "nn"
        "n":  1000,
        "cat_cardinalities": (500,),
        "component_weights": (0.5, 0.5, 0.),   # combo dominates
        "noise_levels": (0.1, 0.8, 0.0),  # combo noise very low
        "prep_types": ["None", "DROP-CAT", "DROP-NUM", "OOF-TE", "OOF-TE-APPEND", "TE", "LOO"],
        },

    "combine_hard_to_learn_linear": {
        "target_model": "linear",   # "linear" | "rf" | "nn"
        "n":  1000,
        "cat_cardinalities": (50,50),
        "combo_offset_cfg": {"cats": ["cat_1", "cat_2"], "sigma": 1.5, "name": "cat1_x_cat2"},
        "component_weights": (0.4, 0.2, 0.4),   # combo dominates
        "noise_levels": (0.1, 0.1, 0.1),  # combo noise very low
        "prep_types": ["None", "DROP-CAT", "DROP-NUM", "OOF-TE", "OOF-TE-APPEND", "TE", "LOO", "CATINT", "CATINT_TE", "CATINT_LOO", "CATINT_OOFTE"],
        },
    "combine_hard_to_learn_rf": {
        "target_model": "rf",   # "linear" | "rf" | "nn"
        "n":  1000,
        "cat_cardinalities": (50,50),
        "combo_offset_cfg": {"cats": ["cat_1", "cat_2"], "sigma": 1.5, "name": "cat1_x_cat2"},
        "component_weights": (0.4, 0.2, 0.4),   # combo dominates
        "noise_levels": (0.1, 0.1, 0.1),  # combo noise very low
        "prep_types": ["None", "DROP-CAT", "DROP-NUM", "OOF-TE", "OOF-TE-APPEND", "TE", "LOO", "CATINT", "CATINT_TE", "CATINT_LOO", "CATINT_OOFTE"],
        },
    "combine_hard_to_learn_nn": {
        "target_model": "nn",   # "linear" | "rf" | "nn"
        "n":  1000,
        "cat_cardinalities": (50,50),
        "combo_offset_cfg": {"cats": ["cat_1", "cat_2"], "sigma": 1.5, "name": "cat1_x_cat2"},
        "component_weights": (0.4, 0.2, 0.4),   # combo dominates
        "noise_levels": (0.1, 0.1, 0.1),  # combo noise very low
        "prep_types": ["None", "DROP-CAT", "DROP-NUM", "OOF-TE", "OOF-TE-APPEND", "TE", "LOO", "CATINT", "CATINT_TE", "CATINT_LOO", "CATINT_OOFTE"],
        },
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
    }


if __name__ == "__main__":
    save_path = "/ceph/atschalz/auto_prep/tabarena/tabarena/tabarena/icml2026/results"
    exp_name = "simulated_cat_experiments_bag"

    for param_set_name, params in experiment_params.items():
        print(f"=== Experiment: {param_set_name} ===")
        if os.path.exists(os.path.join(save_path, f'{exp_name}_{param_set_name}_results.pkl')):
            print(" Results already exist, skipping...")
            continue
        
        n = params.get("n", 1000)
        cat_cardinalities = params.get("cat_cardinalities", (4,4))
        target_model = params.get("target_model", "linear")
        combo_offset_cfg = params.get("combo_offset_cfg", None)
        component_weights = params.get("component_weights", (0.5,0.2,0.3))
        noise_levels = params.get("noise_levels", (0.1,0.1,0.1))
        prep_types = params.get("prep_types", ["None"])
        data = generate_train_test_data(n=n, 
                                        cat_cardinalities=cat_cardinalities, 
                                        # alpha=alpha,
                                        component_weights=component_weights,
                                        noise_levels=noise_levels,
                                        target_model=target_model,
                                        combo_offset_cfg=combo_offset_cfg, 
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
        if target_model != "neutral":
            print(f"Score from numeric model {metric(y_test,data["target_model_fitted"].predict(X_test.select_dtypes(np.number)))}")
        print(f" Base score by target mean: {metric(y_test, np.ones_like(y_test)*y.mean())}")

        ag_prep = AutoMLPipelineFeatureGenerator()
        X = ag_prep.fit_transform(X, y)
        X_test = ag_prep.transform(X_test)

        results = {"preds": {}, "performance": {}}

        # TODO: Properly add GBM-OHE
        for model_name in ["LR", "GBM", "TABM", "PFN"]:
            print("--"*20)
            results["preds"][model_name] = {}
            results["performance"][model_name] = {}

            for prep_type in prep_types:
                preds, performance, X_used = run_experiment(X, y, X_test, y_test, model_name, prep_type, target_type)
                results["performance"][model_name][prep_type] = performance
                results["preds"][model_name][prep_type] = preds
                print(f"Dataset: {param_set_name}, Model: {model_name}, Prep: {prep_type} (shape=[{X_used.shape}]), Performance: {performance:.4f}")

        with open(os.path.join(save_path, f'{exp_name}_{param_set_name}_results.pkl'), 'wb') as f:
            pickle.dump(results, f) 