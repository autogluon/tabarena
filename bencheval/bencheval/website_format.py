from __future__ import annotations

from functools import partial

import pandas as pd


class Constants:
    col_name: str = "method_type"
    tree: str = "Tree-based"
    foundational: str = "Foundation Model"
    neural_network: str = "Neural Network"
    baseline: str = "Baseline"
    reference: str = "Reference Pipeline"
    other: str = "Other"


model_type_emoji = {
    Constants.tree: "🌳",
    Constants.foundational: "🧠⚡",
    Constants.neural_network: "🧠🔁",
    Constants.baseline: "📏",
    Constants.other: "❓",
    Constants.reference: "📊",
}


def get_model_family(model_name: str) -> str:
    prefixes_mapping = {
        Constants.reference: ["AutoGluon"],
        Constants.neural_network: [
            "REALMLP",
            "TABM",
            "FASTAI",
            "MNCA",
            "NN_TORCH",
            "MITRA",
            "LIMIX",
        ],
        Constants.tree: ["GBM", "CAT", "EBM", "XGB", "XT", "RF"],
        Constants.foundational: [
            "TABDPT",
            "TABICL",
            "TABPFN",
            "MITRA",
            "LIMIX",
            "BETA",
            "TABFLEX",
            "REALTABPFN-V2.5",
        ],
        Constants.baseline: ["KNN", "LR"],
        Constants.other: ["XRFM"],
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
        "REALTABPFN-V2.5": "RealTabPFN-v2.5",
    }

    # Sort keys by descending length so longest prefixes are matched first
    for prefix in sorted(rename_map, key=len, reverse=True):
        if model_name.startswith(prefix):
            if model_name == prefix:
                return rename_map[prefix]
            return model_name.replace(prefix, rename_map[prefix], 1)

    return model_name


def add_metadata(row, metadata_df: pd.DataFrame):
    model_name = row["method"]
    if ", 4h)" in model_name:
        metadata_key_from_model_name = model_name
        is_reference_model = True
    else:
        metadata_key_from_model_name = model_name.split(" (")[0]
        is_reference_model = False

    try:
        metadata = metadata_df[
            metadata_df["name" if is_reference_model else "model_key"]
            == metadata_key_from_model_name
        ]
        assert len(metadata) == 1
    except AssertionError:
        metadata_key_from_model_name = metadata_key_from_model_name.replace("_GPU", "")
        metadata = metadata_df[
            metadata_df["name" if is_reference_model else "model_key"]
            == metadata_key_from_model_name
        ]
        assert len(metadata) == 1

    metadata = metadata.iloc[0]

    return pd.Series(
        {
            "Hardware": metadata["compute"].upper(),
            "Verified": metadata["verified"],
            "ReferenceURL": metadata["reference_url"],
        }
    )


def format_leaderboard(
    df_leaderboard: pd.DataFrame,
    *,
    method_metadata_info: pd.DataFrame | None = None,
    include_type: bool = False,
) -> pd.DataFrame:
    df_leaderboard = df_leaderboard.copy(deep=True)

    # Add metadata
    if method_metadata_info is None:
        df_leaderboard["Hardware"] = "Unknown"
        df_leaderboard["Verified"] = "Unknown"
    else:
        df_leaderboard[["Hardware", "Verified", "ReferenceURL"]] = df_leaderboard.apply(
            partial(add_metadata, metadata_df=method_metadata_info),
            result_type="expand",
            axis=1,
        )

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

    # FIXME: move to lb generation!
    df_leaderboard["method"] = df_leaderboard["method"].str.replace(
        "(tuned + ensemble)", "(tuned + ensembled)"
    )

    if method_metadata_info is not None:
        gpu_postfix = "_GPU"
        df_leaderboard["method"] = df_leaderboard["method"].str.replace(gpu_postfix, "")
        df_leaderboard["method"] = (
            "[" + df_leaderboard["method"] + "](" + df_leaderboard["ReferenceURL"] + ")"
        )
        df_leaderboard["Verified"] = df_leaderboard["Verified"].apply(
            lambda v: "✔️" if v else "➖"
        )

    df_leaderboard = df_leaderboard.loc[
        :,
        [
            "Type",
            "TypeName",
            "method",
            "Verified",
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
