"""TabRepo Rerun + manual changes.

We re-ran the work of the TabRepo paper, but use a portfolio of size 200 instead of the
100 size portfolio used by AutoGluon’s best_quality setting. We also included more model
families: Linear models and KNN.

# Removed because too slow predict for large datasets
LightGBM_r19
LightGBM_r96
LightGBM_r94_BAG_L1
LightGBM_r15_BAG_L1
LightGBM_r133_BAG_L1
LightGBM_r174
XGBoost_r31_BAG_L1
XGBoost_r33_BAG_L1

# Added manually:
LightGBM, XGBoost, and CatBoost with different max_bin values.
"""
from __future__ import annotations

zeroshot2024 = {
    "GBM": [
        {},
        {  # Added manually.
            "max_bin": 4095,
            "ag_args": {"priority": -1, "name_suffix": "Bin4095"},
        },
        {"extra_trees": True, "ag_args": {"name_suffix": "XT"}},
        {  # Old GBMLarge
            "learning_rate": 0.03,
            "num_leaves": 128,
            "feature_fraction": 0.9,
            "min_data_in_leaf": 3,
            "ag_args": {"name_suffix": "Large", "priority": 0, "hyperparameter_tune_kwargs": None},
        },
        {
            "extra_trees": False,
            "feature_fraction": 0.7023601671276614,
            "learning_rate": 0.012144796373999013,
            "min_data_in_leaf": 14,
            "num_leaves": 53,
            "ag_args": {"name_suffix": "_r131", "priority": -3},
        },
        {
            "extra_trees": True,
            "feature_fraction": 0.8999894845710796,
            "learning_rate": 0.051087336729504676,
            "min_data_in_leaf": 18,
            "num_leaves": 167,
            "ag_args": {"name_suffix": "_r191", "priority": -24},
        },
        {
            "extra_trees": False,
            "feature_fraction": 0.7325454610506641,
            "learning_rate": 0.009447054356012436,
            "min_data_in_leaf": 4,
            "num_leaves": 85,
            "ag_args": {"name_suffix": "_r54", "priority": -27},
        },
        {
            "extra_trees": True,
            "feature_fraction": 0.8682559906624081,
            "learning_rate": 0.09561511371136407,
            "min_data_in_leaf": 9,
            "num_leaves": 121,
            "ag_args": {"name_suffix": "_r81", "priority": -28},
        },
        {
            "extra_trees": False,
            "feature_fraction": 0.8254432681390782,
            "learning_rate": 0.031251656439648626,
            "min_data_in_leaf": 50,
            "num_leaves": 210,
            "ag_args": {"name_suffix": "_r135", "priority": -32},
        },
        {
            "extra_trees": False,
            "feature_fraction": 0.9668244885378855,
            "learning_rate": 0.07254551525590439,
            "min_data_in_leaf": 14,
            "num_leaves": 31,
            "ag_args": {"name_suffix": "_r145", "priority": -40},
        },
        {
            "extra_trees": False,
            "feature_fraction": 0.45835595623790437,
            "learning_rate": 0.09533195017847339,
            "min_data_in_leaf": 7,
            "num_leaves": 231,
            "ag_args": {"name_suffix": "_r41", "priority": -44},
        },
        {
            "extra_trees": False,
            "feature_fraction": 0.6245777099925497,
            "learning_rate": 0.04711573688184715,
            "min_data_in_leaf": 56,
            "num_leaves": 89,
            "ag_args": {"name_suffix": "_r130", "priority": -52},
        },
        {
            "extra_trees": True,
            "feature_fraction": 0.9666234339903601,
            "learning_rate": 0.04582977995120822,
            "min_data_in_leaf": 4,
            "num_leaves": 127,
            "ag_args": {"name_suffix": "_r55", "priority": -68},
        },
        {
            "extra_trees": False,
            "feature_fraction": 0.7016257244614168,
            "learning_rate": 0.007922167829715967,
            "min_data_in_leaf": 7,
            "num_leaves": 132,
            "ag_args": {"name_suffix": "_r149", "priority": -91},
        },
        {
            "extra_trees": True,
            "feature_fraction": 0.9046840778713597,
            "learning_rate": 0.07515257316211908,
            "min_data_in_leaf": 42,
            "num_leaves": 18,
            "ag_args": {"name_suffix": "_r43", "priority": -100},
        },
        {
            "extra_trees": True,
            "feature_fraction": 0.4601361323873807,
            "learning_rate": 0.07856777698860955,
            "min_data_in_leaf": 12,
            "num_leaves": 198,
            "ag_args": {"name_suffix": "_r42", "priority": -105},
        },
        {
            "extra_trees": False,
            "feature_fraction": 0.7532437659821729,
            "learning_rate": 0.08944644189688526,
            "min_data_in_leaf": 39,
            "num_leaves": 53,
            "ag_args": {"name_suffix": "_r153", "priority": -118},
        },
        {
            "extra_trees": True,
            "feature_fraction": 0.43613528297756193,
            "learning_rate": 0.03685135839677242,
            "min_data_in_leaf": 57,
            "num_leaves": 27,
            "ag_args": {"name_suffix": "_r13", "priority": -121},
        },
        {
            "extra_trees": False,
            "feature_fraction": 0.7579932437770318,
            "learning_rate": 0.052301563688720604,
            "min_data_in_leaf": 37,
            "num_leaves": 136,
            "ag_args": {"name_suffix": "_r51", "priority": -131},
        },
        {
            "extra_trees": False,
            "feature_fraction": 0.41239059967943725,
            "learning_rate": 0.04848901712678711,
            "min_data_in_leaf": 5,
            "num_leaves": 67,
            "ag_args": {"name_suffix": "_r61", "priority": -132},
        },
        {
            "extra_trees": False,
            "feature_fraction": 0.40585986135777,
            "learning_rate": 0.012590980616372347,
            "min_data_in_leaf": 32,
            "num_leaves": 22,
            "ag_args": {"name_suffix": "_r106", "priority": -139},
        },
        {
            "extra_trees": False,
            "feature_fraction": 0.9744705133953723,
            "learning_rate": 0.020546267996855768,
            "min_data_in_leaf": 60,
            "num_leaves": 99,
            "ag_args": {"name_suffix": "_r66", "priority": -163},
        },
        {
            "extra_trees": False,
            "feature_fraction": 0.6937293621346563,
            "learning_rate": 0.013803836586316339,
            "min_data_in_leaf": 38,
            "num_leaves": 16,
            "ag_args": {"name_suffix": "_r49", "priority": -164},
        },
        {
            "extra_trees": False,
            "feature_fraction": 0.6090855934200983,
            "learning_rate": 0.04590490414627263,
            "min_data_in_leaf": 56,
            "num_leaves": 144,
            "ag_args": {"name_suffix": "_r144", "priority": -171},
        },
        {
            "extra_trees": False,
            "feature_fraction": 0.5730390983988963,
            "learning_rate": 0.010305352949119608,
            "min_data_in_leaf": 10,
            "num_leaves": 215,
            "ag_args": {"name_suffix": "_r121", "priority": -172},
        },
        {
            "extra_trees": False,
            "feature_fraction": 0.45118655387122203,
            "learning_rate": 0.009705399613761859,
            "min_data_in_leaf": 9,
            "num_leaves": 45,
            "ag_args": {"name_suffix": "_r198", "priority": -173},
        },
    ],
}
