from __future__ import annotations

from functools import partial

import pandas as pd

from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata

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


def strict_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: str | list[str],
    how: str = "left",
    validate: str | None = None,
) -> pd.DataFrame:
    """
    Merge two DataFrames, but if they share non-key columns, require that the
    shared columns have identical values (for matching keys) and raise if not.
    Prevents creating _x/_y columns by dropping shared columns from `right`
    after the check.
    """
    keys = [on] if isinstance(on, str) else list(on)

    # shared non-key columns
    shared = set(left.columns) & set(right.columns) - set(keys)

    if shared:
        # compare only on key+shared; drop duplicates on keys to avoid explode
        lhs = left[keys + sorted(shared)].drop_duplicates(keys)
        rhs = right[keys + sorted(shared)].drop_duplicates(keys)

        chk = lhs.merge(rhs, on=keys, how="inner", suffixes=("_l", "_r"))

        bad = []
        for c in shared:
            lcol, rcol = f"{c}_l", f"{c}_r"
            # treat NaN as equal
            eq = chk[lcol].eq(chk[rcol]) | (chk[lcol].isna() & chk[rcol].isna())
            if not bool(eq.all()):
                bad.append(c)

        if bad:
            raise ValueError(f"Mismatched shared columns for on={keys}: {sorted(bad)}")

    return left.merge(
        right.drop(columns=sorted(shared)) if shared else right,
        on=keys,
        how=how,
        validate=validate,
    )


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
            "SAP-RPT-OSS",
            "TABICLV2",
        ],
        Constants.baseline: ["KNN", "LR"],
        Constants.other: ["XRFM"],
    }

    for method_type, prefixes in prefixes_mapping.items():
        for prefix in prefixes:
            if model_name.lower().startswith(prefix.lower()):
                return method_type
    return Constants.other


def get_rename_map() -> dict[str, str]:
    _rename_map = {
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
        "SAP-RPT-OSS": "SAP-RPT-OSS",
    }
    return _rename_map


def rename_method(model_name: str, rename_map: dict[str, str]) -> str:
    # Sort keys by descending length so longest prefixes are matched first
    for prefix in sorted(rename_map, key=len, reverse=True):
        if model_name.startswith(prefix):
            if model_name == prefix:
                return rename_map[prefix]
            return model_name.replace(prefix, rename_map[prefix], 1)

    return model_name


def add_metadata(row, metadata_df: pd.DataFrame):
    method = row["method"]
    if method not in metadata_df.index:
        return pd.Series(
            {
                "Hardware": "Missing",
                "Verified": "Missing",
                "ReferenceURL": None,
            }
        )
    metadata = metadata_df.loc[method]
    config_type = metadata["config_type"]

    model_family = get_model_family(config_type if not pd.isna(config_type) else method)

    # Add Model Family Information
    out_dict = {
        "Type": model_type_emoji[model_family],
        "TypeName": model_family,
    }

    display_name = MethodMetadata.compute_method_name(
        method=method,
        method_type=metadata["method_type"],
        method_subtype=metadata["method_subtype"],
        config_type=metadata["config_type"],
        display_name=metadata["display_name"],
    )

    if metadata.get("reference_url", None) is not None:
        display_name = add_url(display_name, metadata["reference_url"])

    if pd.isna(metadata["verified"]):
        verified = "Unknown"
    else:
        verified = "✔️" if metadata["verified"] else "➖"
    if pd.isna(metadata["compute"]):
        hardware = "Unknown"
    else:
        hardware = metadata["compute"].upper()

    return pd.Series(
        {
            "method": display_name,
            "Hardware": hardware,
            "Verified": verified,
            **out_dict,
        }
    )


def add_url(method: str, url: str | None) -> str:
    if pd.isna(url) or not url:
        return method
    return "[" + method + "](" + url + ")"


def legacy_formatting(df_leaderboard: pd.DataFrame) -> pd.DataFrame:
    df_leaderboard = df_leaderboard.copy(deep=True)
    df_leaderboard["Hardware"] = "Unknown"
    df_leaderboard["Verified"] = "Unknown"

    # Add Model Family Information
    df_leaderboard["Type"] = df_leaderboard.loc[:, "method"].apply(
        lambda s: model_type_emoji[get_model_family(s)]
    )
    df_leaderboard["TypeName"] = df_leaderboard.loc[:, "method"].apply(
        lambda s: get_model_family(s)
    )

    _rename_map = get_rename_map()
    df_leaderboard["method"] = df_leaderboard["method"].apply(
        lambda method: rename_method(model_name=method, rename_map=_rename_map)
    )
    return df_leaderboard


def format_leaderboard(
    df_leaderboard: pd.DataFrame,
    *,
    method_metadata_info: pd.DataFrame | None = None,
    include_type: bool = False,
) -> pd.DataFrame:
    df_leaderboard = df_leaderboard.copy(deep=True)

    # Add metadata
    if method_metadata_info is None:
        df_leaderboard = legacy_formatting(df_leaderboard=df_leaderboard)
    else:
        method_info_map = strict_merge(df_leaderboard, method_metadata_info.drop(columns=["method_type"]), on=["ta_name", "ta_suite"])
        method_info_map = method_info_map.set_index("method")
        df_leaderboard[["method", "Hardware", "Verified", "Type", "TypeName"]] = df_leaderboard.apply(
            partial(add_metadata, metadata_df=method_info_map),
            result_type="expand",
            axis=1,
        )

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
            "Verified",
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
