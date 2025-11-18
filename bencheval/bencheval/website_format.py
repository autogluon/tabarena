from __future__ import annotations

import pandas as pd


class Constants:
    col_name: str = "method_type"
    tree: str = "Tree-based"
    foundational: str = "Foundation Model"
    neural_network: str ="Neural Network"
    baseline: str = "Baseline"
    reference: str ="Reference Pipeline"
    # Not Used
    other: str = "Other"


model_type_emoji = {
    Constants.tree: "🌳",
    Constants.foundational: "🧠⚡",
    Constants.neural_network: "🧠🔁",
    Constants.baseline: "📏",
    # Not used
    Constants.other: "❓",
    Constants.reference: "📊",
}


def get_model_family(model_name: str) -> str:
    prefixes_mapping = {
        Constants.reference: ["AutoGluon"],
        Constants.neural_network: ["REALMLP", "TABM", "FASTAI", "MNCA", "NN_TORCH", "MITRA", "LIMIX"],
        Constants.tree: ["GBM", "CAT", "EBM", "XGB", "XT", "RF", "XRFM"],
        Constants.foundational: ["TABDPT", "TABICL", "TABPFN", "MITRA", "LIMIX", "BETA", "TABFLEX"],
        Constants.baseline: ["KNN", "LR"],
    }

    for method_type, prefixes in prefixes_mapping.items():
        for prefix in prefixes:
            if model_name.lower().startswith(prefix.lower()):
                return method_type
    return Constants.other


def rename_map(model_name: str) -> str:
    rename_map = {
        "TABM": "TabM",
        "REALMLP": "RealMLP",
        "GBM": "LightGBM",
        "CAT": "CatBoost",
        "XGB": "XGBoost",
        "XT": "ExtraTrees",
        "RF": "RandomForest",
        "MNCA": "ModernNCA",
        "NN_TORCH": "TorchMLP",
        "FASTAI": "FastaiMLP",
        "TABPFNV2": "TabPFNv2",
        "EBM": "EBM",
        "TABDPT": "TabDPT",
        "TABICL": "TabICL",
        "KNN": "KNN",
        "LR": "Linear",
        "MITRA": "Mitra",
        "LIMIX": "LimiX",
        "XRFM": "xRFM",
        "TABFLEX": "TabFlex",
        "BETA": "BetaTabPFN",
    }

    # Sort keys by descending length so longest prefixes are matched first
    for prefix in sorted(rename_map, key=len, reverse=True):
        if model_name.startswith(prefix):
            if model_name == prefix:
                return rename_map[prefix]
            else:
                return model_name.replace(prefix, rename_map[prefix], 1)

    return model_name


def compute_map(method: str) -> str:
    _compute_map = {
        "AutoGluon 1.3 (4h)": "CPU",
        "AutoGluon 1.4 (4h)": "GPU",
        "LimiX (default)": "GPU",
    }
    gpu_postfix = "_GPU"
    if method in _compute_map:
        return _compute_map[method]
    return "CPU" if gpu_postfix not in method else "GPU"


def format_leaderboard(df_leaderboard: pd.DataFrame, include_type: bool = False) -> pd.DataFrame:
    df_leaderboard = df_leaderboard.copy(deep=True)

    # Add Model Family Information
    df_leaderboard["Type"] = df_leaderboard.loc[:, "method"].apply(
        lambda s: model_type_emoji[get_model_family(s)]
    )
    df_leaderboard["TypeName"] = df_leaderboard.loc[:, "method"].apply(
        lambda s: get_model_family(s)
    )
    df_leaderboard["method"] = df_leaderboard["method"].apply(rename_map)

    # elo,elo+,elo-,mrr
    df_leaderboard["Elo 95% CI"] = (
        "+"
        + df_leaderboard["elo+"].round(0).astype(int).astype(str)
        + "/-"
        + df_leaderboard["elo-"].round(0).astype(int).astype(str)
    )
    # select only the columns we want to display
    df_leaderboard["normalized-score"] = 1 - df_leaderboard["normalized-error"]
    df_leaderboard["hmr"] = 1 / df_leaderboard["mrr"]
    df_leaderboard["improvability"] = 100 * df_leaderboard["improvability"]

    # Imputed logic
    if "imputed" in df_leaderboard.columns:
        df_leaderboard["imputed"] = (100 * df_leaderboard["imputed"]).round(2)
        df_leaderboard["imputed_bool"] = False
        # Filter methods that are fully imputed.
        df_leaderboard = df_leaderboard[~(df_leaderboard["imputed"] == 100)]
        # Add imputed column and add name postfix
        imputed_mask = df_leaderboard["imputed"] != 0
        df_leaderboard.loc[imputed_mask, "imputed_bool"] = True
        df_leaderboard.loc[imputed_mask, "method"] = df_leaderboard.loc[
            imputed_mask, ["method", "imputed"]
        ].apply(lambda row: row["method"] + f" [{row['imputed']:.2f}% IMPUTED]", axis=1)
    else:
        df_leaderboard["imputed_bool"] = None
        df_leaderboard["imputed"] = None

    # Resolve GPU postfix
    gpu_postfix = "_GPU"
    df_leaderboard["Hardware"] = df_leaderboard["method"].apply(compute_map)
    df_leaderboard["method"] = df_leaderboard["method"].str.replace(gpu_postfix, "")

    df_leaderboard = df_leaderboard.loc[
        :,
        [
            "Type",
            "TypeName",
            "method",
            "elo",
            "Elo 95% CI",
            "normalized-score",
            "rank",
            "hmr",
            "improvability",
            "median_time_train_s_per_1K",
            "median_time_infer_s_per_1K",
            "imputed",
            "imputed_bool",
            "Hardware",
        ],
    ]

    # round for better display
    df_leaderboard[["elo", "Elo 95% CI"]] = df_leaderboard[["elo", "Elo 95% CI"]].round(
        0
    )
    df_leaderboard[["median_time_train_s_per_1K", "rank", "hmr"]] = df_leaderboard[
        ["median_time_train_s_per_1K", "rank", "hmr"]
    ].round(2)
    df_leaderboard[
        ["normalized-score", "median_time_infer_s_per_1K", "improvability"]
    ] = df_leaderboard[
        ["normalized-score", "median_time_infer_s_per_1K", "improvability"]
    ].round(3)

    df_leaderboard = df_leaderboard.sort_values(by="elo", ascending=False)
    df_leaderboard = df_leaderboard.reset_index(drop=True)
    df_leaderboard = df_leaderboard.reset_index(names="#")

    if not include_type:
        df_leaderboard = df_leaderboard.drop(columns=["Type", "TypeName"])

    # rename some columns
    return df_leaderboard.rename(
        columns={
            "median_time_train_s_per_1K": "Median Train Time (s/1K) [⬇️]",
            "median_time_infer_s_per_1K": "Median Predict Time (s/1K) [⬇️]",
            "method": "Model",
            "elo": "Elo [⬆️]",
            "rank": "Rank [⬇️]",
            "normalized-score": "Score [⬆️]",
            "hmr": "Harmonic Rank [⬇️]",
            "improvability": "Improvability (%) [⬇️]",
            "imputed": "Imputed (%) [⬇️]",
            "imputed_bool": "Imputed",
        }
    )
