from __future__ import annotations

import re
from functools import partial
from typing import TYPE_CHECKING

import pandas as pd

from tabarena.models._method_metadata import MethodMetadata

if TYPE_CHECKING:
    from collections.abc import Container, Iterable


class Constants:
    col_name: str = "method_type"
    tree: str = "Tree-based"
    foundational: str = "Foundation Model"
    neural_network: str = "Neural Network"
    baseline: str = "Baseline"
    #: Whole pipelines rather than single models: AutoGluon, TabFM+, an agent, a hosted API,
    #: a portfolio. Set from `MethodMetadata.method_class`, never from the method's name.
    system: str = "System"
    other: str = "Other"


model_type_emoji = {
    Constants.tree: "🌳",
    Constants.foundational: "🧠⚡",
    Constants.neural_network: "🧠🔁",
    Constants.baseline: "📏",
    Constants.other: "❓",
    Constants.system: "📊",
}

#: How each `MethodTag` is presented: the chip shown next to a system's name, and the hover
#: text explaining why it matters. Keys are the tag strings `MethodMetadata.tags` carries.
#: The single source of truth for tag presentation, handed to the generated table and the
#: leaderboard app rather than restated in either.
TAG_SPECS: dict[str, dict[str, str]] = {
    "with-llm": {
        "emoji": "🤖",
        "label": "with LLMs",
        "hint": (
            "An LLM is involved somewhere in this system, possibly as an agent. Its results "
            "depend on a model whose training data cannot be inspected and which may already "
            "have seen the test data."
        ),
    },
    "closed-source-api": {
        "emoji": "🔒",
        "label": "closed-source API",
        "hint": (
            "This system runs on its provider's servers. We cannot see what it does with the data "
            "we send, test features included, and it can change between runs. Its train and predict "
            "times are not comparable with the models'. Do not trust these results by default."
        ),
    },
}


def strict_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: str | list[str],
    how: str = "left",
    validate: str | None = None,
) -> pd.DataFrame:
    """Merge two DataFrames, but if they share non-key columns, require that the
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


def get_model_family(model_name: str, system_names: Container[str] = frozenset()) -> str:
    """Classify a method into its family. Accepts both raw config-type keys
    (e.g. "GBM", "MNCA") and display names (e.g. "LightGBM", "ModernNCA"),
    since callers pass whichever identifier they have at hand.

    Systems are never inferred from the name: pass ``system_names`` (the display names of
    everything with ``method_class="system"``, see :func:`system_display_names`) and a match
    short-circuits to :attr:`Constants.system`. That is what keeps a newly added system out of
    "❓ Other" without anyone remembering to add its prefix to the table below. Callers holding
    the metadata row itself (:func:`add_metadata`) read ``method_class`` directly instead.
    """
    if model_name in system_names:
        return Constants.system
    prefixes_mapping = {
        Constants.neural_network: [
            "REALMLP",
            "TABM",
            "FASTAI",
            "MNCA",
            "MODERNNCA",
            "NN_TORCH",
            "TORCHMLP",
        ],
        Constants.tree: [
            "GBM",
            "LIGHTGBM",
            "CAT",
            "EBM",
            "XGB",
            "XT",
            "EXTRATREES",
            "RF",
            "RANDOMFOREST",
            "PB",
            "PERPETUALBOOSTER",
            "CHIMERA",
            "CTB",
            "PRISMBOOST",
        ],
        Constants.foundational: [
            "CAUSILO",
            "TABDPT",
            "TABDPT_TURBO",
            "TABDPT-1.3",
            "TABICL",
            "TABPFN",
            "REALTABPFN",
            "MITRA",
            "MITRA_V2",
            "LIMIX",
            "TA-LIMIX",
            "TA-LIMIX-2",
            "BETA",
            "TABFLEX",
            "TABFM",
            "REALTABPFN-V2.5",
            "SAP-RPT-OSS",
            "TABICLV2",
            "TABSTAR",
            "TA-TABPFN-3",
            "TA-TABPFN-3.5",
            "TA-TABPFN-3.5-FAST",
            "TA-ORION-MSP",
            "ORIONMSP",
            "TA-ILTM",
            "TA-NORI",
            "TABSWIFT",
            "EXAONE",
            "XIAOMI-TABLDM",
            "TA-KUMO-TABULAR",
            "FTFM",
            "LIGHTPFN",
        ],
        Constants.baseline: ["KNN", "LR", "LINEAR"],
        Constants.other: ["XRFM", "APLR"],
    }

    def _strip_ta(name: str) -> str:
        # BeyondArena config_types are prefixed with "TA-" (e.g. "TA-REALMLP", "TA-TABPFN-2.6");
        # strip it so the bare family prefixes below match regardless of that prefix.
        return name[3:] if name.lower().startswith("ta-") else name

    name = _strip_ta(model_name).lower()
    for method_type, prefixes in prefixes_mapping.items():
        for prefix in prefixes:
            if name.startswith(_strip_ta(prefix).lower()):
                return method_type
    return Constants.other


def system_display_names(method_metadata_info: pd.DataFrame | None) -> frozenset[str]:
    """Display names of every method in ``method_metadata_info`` that is a system.

    Feeds :func:`get_model_family`'s ``system_names`` on the figure paths, which only ever see
    a method's rendered name and so cannot read ``method_class`` off the row themselves.
    """
    if method_metadata_info is None or "method_class" not in method_metadata_info.columns:
        return frozenset()
    is_system = method_metadata_info["method_class"] == "system"
    return frozenset(method_metadata_info.loc[is_system, "display_name"].dropna())


#: Tags that shade a system's color, in precedence order: a system carrying several takes the
#: first. Each step darkens the System orange, the less of the system can be inspected, and an
#: untagged system keeps the plain family color. The colors are
#: :data:`tabarena.plot.plot_pareto_focus.SYSTEM_SHADE_COLORS`.
SYSTEM_SHADE_TAGS: tuple[str, ...] = ("closed-source-api", "with-llm")


def system_shade(tags: Iterable[str] | str | None) -> str | None:
    """The tag that shades a system's color, or None when it keeps the plain System color.

    ``tags`` is a tag sequence or one written to CSV: a website frame's semicolon-joined ``Tags``
    cell, or the tuple repr in a ``method_info.csv``. An empty cell reads back as NaN.
    """
    if tags is None or isinstance(tags, float):
        return None
    present = set(re.findall(r"[\w-]+", tags) if isinstance(tags, str) else tags)
    return next((tag for tag in SYSTEM_SHADE_TAGS if tag in present), None)


def system_shades(method_metadata_info: pd.DataFrame | None) -> dict[str, str]:
    """Display name -> :func:`system_shade` of every shaded system in ``method_metadata_info``.

    The counterpart of :func:`system_display_names` for the figure paths, which see only a
    method's rendered name.
    """
    if method_metadata_info is None or not {"method_class", "tags"} <= set(method_metadata_info.columns):
        return {}
    systems = method_metadata_info[method_metadata_info["method_class"] == "system"]
    shades = {name: system_shade(tags) for name, tags in zip(systems["display_name"], systems["tags"], strict=True)}
    return {name: shade for name, shade in shades.items() if shade is not None and not pd.isna(name)}


def get_rename_map() -> dict[str, str]:
    return {
        "TABM": "TabM",
        "REALMLP": "RealMLP",
        "GBM": "LightGBM",
        "CAT": "CatBoost",
        "XGB": "XGBoost",
        "XT": "ExtraTrees",
        "RF": "RandomForest",
        "PB": "PerpetualBooster",
        "CHIMERA": "ChimeraBoost",
        "CTB": "CTBoost",
        "MNCA": "ModernNCA",
        "NN_TORCH": "TorchMLP",
        "FASTAI": "FastaiMLP",
        "TABPFNV2": "TabPFNv2",
        "EBM": "EBM",
        "TABDPT": "TabDPT",
        "TABDPT_TURBO": "TabDPT-Turbo",
        "TABDPT-1.3": "TabDPT-1.3",
        "TABICL": "TabICL",
        "KNN": "KNN",
        "LR": "Linear",
        "MITRA": "Mitra",
        "MITRA_V2": "Mitra-v2",
        "LIMIX": "LimiX",
        "XRFM": "xRFM",
        "TABFLEX": "TabFlex",
        "BETA": "BetaTabPFN",
        "REALTABPFN-V2.5": "RealTabPFN-v2.5",
        "SAP-RPT-OSS": "SAP-RPT-OSS",
        "TABSWIFT": "TabSwift",
        "XIAOMI-TABLDM": "Xiaomi-TabLDM",
    }


def rename_method(model_name: str, rename_map: dict[str, str]) -> str:
    # Sort keys by descending length so longest prefixes are matched first
    for prefix in sorted(rename_map, key=len, reverse=True):
        if model_name.startswith(prefix):
            if model_name == prefix:
                return rename_map[prefix]
            return model_name.replace(prefix, rename_map[prefix], 1)

    return model_name


def add_metadata(
    row,
    metadata_df: pd.DataFrame,
    include_url: bool = True,
):
    method = row["method"]
    if method not in metadata_df.index:
        # Same keys as the happy path, so the caller's column assignment lines up either way.
        return pd.Series(
            {
                "method": method,
                "Hardware": "Missing",
                "Verified": "Missing",
                "Type": model_type_emoji[Constants.other],
                "TypeName": Constants.other,
                "MethodClass": "model",
                "Tags": "",
                "Commercial": True,
                "License": "",
                "FitsAtPredict": False,
                "Organization": "",
                "Leakage": "",
                "LeakageDatasets": "",
            },
        )
    metadata = metadata_df.loc[method]
    config_type = metadata["config_type"]

    # A system's family is declared, not guessed: `method_class` says so directly, which is
    # what keeps a newly added system out of "❓ Other". Only models are name-classified.
    method_class = metadata.get("method_class", "model")
    if pd.isna(method_class):
        method_class = "model"
    tags = metadata.get("tags", ())
    # A method absent from a metadata frame's `tags` column comes back as NaN (a truthy
    # float), so the falsy-default idiom is not enough here.
    if tags is None or isinstance(tags, float):
        tags = ()
    if method_class == "system":
        model_family = Constants.system
    else:
        model_family = get_model_family(config_type if not pd.isna(config_type) else method)
    # Metadata frames written before the license fields existed carry neither column; a
    # missing or NaN value means "no restriction declared", which is the field's default.
    commercial_use = metadata.get("commercial_use", True)
    if commercial_use is None or pd.isna(commercial_use):
        commercial_use = True
    license_name = metadata.get("license", None)
    if license_name is None or pd.isna(license_name):
        license_name = ""
    # Frames written before the field existed lack it; that reads as a separate fit and predict.
    fits_at_predict = metadata.get("fits_at_predict", False)
    if fits_at_predict is None or pd.isna(fits_at_predict):
        fits_at_predict = False
    # Likewise for frames written before the organization field; "" when unknown.
    organization = metadata.get("organization", None)
    if organization is None or pd.isna(organization):
        organization = ""
    # And for frames written before the leakage fields; "" when the method carries no warning.
    leakage_evidence = metadata.get("leakage_evidence", None)
    if leakage_evidence is None or (isinstance(leakage_evidence, float) and pd.isna(leakage_evidence)):
        leakage_evidence = ""
    leakage_datasets = metadata.get("leakage_datasets", ())
    if leakage_datasets is None or isinstance(leakage_datasets, float):
        leakage_datasets = ()

    # Add Model Family Information
    out_dict = {
        "Type": model_type_emoji[model_family],
        "TypeName": model_family,
        "MethodClass": method_class,
        # Semicolon-joined so the CSV stays one cell per row; the leaderboard splits it back
        # out into chips.
        "Tags": ";".join(tags),
        "Commercial": bool(commercial_use),
        "License": str(license_name),
        # Whether the train time includes the prediction call (`MethodMetadata.fits_at_predict`).
        "FitsAtPredict": bool(fits_at_predict),
        # Who develops the method (`MethodMetadata.organization`); the leaderboard shows its logo.
        "Organization": str(organization),
        # The evidence that test data leaked into its training (`MethodMetadata.leakage_evidence`), and
        # the datasets it is proven on, semicolon-joined like `Tags`; empty for a method without one.
        "Leakage": str(leakage_evidence),
        "LeakageDatasets": ";".join(leakage_datasets),
    }

    display_name = MethodMetadata.compute_method_name(
        method=method,
        method_type=metadata["method_type"],
        method_subtype=metadata["method_subtype"],
        config_type=metadata["config_type"],
        display_name=metadata["display_name"],
    )

    if include_url and metadata.get("reference_url", None) is not None:
        display_name = add_url(display_name, metadata["reference_url"])

    verified = "Unknown" if pd.isna(metadata["verified"]) else "✔️" if metadata["verified"] else "➖"
    hardware = "Unknown" if pd.isna(metadata["compute"]) else metadata["compute"].upper()

    return pd.Series(
        {
            "method": display_name,
            "Hardware": hardware,
            "Verified": verified,
            **out_dict,
        },
    )


def add_url(method: str, url: str | None) -> str:
    if pd.isna(url) or not url:
        return method
    return "[" + method + "](" + url + ")"


def legacy_formatting(df_leaderboard: pd.DataFrame) -> pd.DataFrame:
    df_leaderboard = df_leaderboard.copy(deep=True)
    df_leaderboard["Hardware"] = "Unknown"
    df_leaderboard["Verified"] = "Unknown"
    # Without a metadata frame there is nothing to declare a system, so every row is a model.
    df_leaderboard["MethodClass"] = "model"
    df_leaderboard["Tags"] = ""
    df_leaderboard["Commercial"] = True
    df_leaderboard["License"] = ""
    df_leaderboard["FitsAtPredict"] = False
    df_leaderboard["Organization"] = ""
    df_leaderboard["Leakage"] = ""
    df_leaderboard["LeakageDatasets"] = ""

    # Add Model Family Information
    df_leaderboard["Type"] = df_leaderboard.loc[:, "method"].apply(
        lambda s: model_type_emoji[get_model_family(s)],
    )
    df_leaderboard["TypeName"] = df_leaderboard.loc[:, "method"].apply(
        get_model_family,
    )

    _rename_map = get_rename_map()
    df_leaderboard["method"] = df_leaderboard["method"].apply(
        lambda method: rename_method(model_name=method, rename_map=_rename_map),
    )
    return df_leaderboard


def format_leaderboard(
    df_leaderboard: pd.DataFrame,
    *,
    method_metadata_info: pd.DataFrame | None = None,
    include_type: bool = False,
    include_url: bool = False,
    include_imputed_in_name: bool = True,
    compact: bool = False,
) -> pd.DataFrame:
    df_leaderboard = df_leaderboard.copy(deep=True)

    # Add metadata
    if method_metadata_info is None:
        df_leaderboard = legacy_formatting(df_leaderboard=df_leaderboard)
    else:
        method_info_map = strict_merge(
            df_leaderboard, method_metadata_info.drop(columns=["method_type"]), on=["ta_name", "ta_suite"]
        )
        method_info_map = method_info_map.set_index("method")
        df_leaderboard[
            [
                "method",
                "Hardware",
                "Verified",
                "Type",
                "TypeName",
                "MethodClass",
                "Tags",
                "Commercial",
                "License",
                "FitsAtPredict",
                "Organization",
                "Leakage",
                "LeakageDatasets",
            ]
        ] = df_leaderboard.apply(
            partial(add_metadata, metadata_df=method_info_map, include_url=include_url),
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
        if include_imputed_in_name:
            df_leaderboard.loc[imputed_mask, "method"] = df_leaderboard.loc[
                imputed_mask,
                ["method", "imputed"],
            ].apply(lambda row: row["method"] + f" [{row['imputed']:.2f}% IMPUTED]", axis=1)
    else:
        df_leaderboard["imputed_bool"] = None
        df_leaderboard["imputed"] = None

    # FIXME: move to lb generation!
    df_leaderboard["method"] = df_leaderboard["method"].str.replace(
        "(tuned + ensemble)",
        "(tuned + ensembled)",
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
            "MethodClass",
            "Tags",
            "Commercial",
            "License",
            "FitsAtPredict",
            "Organization",
            "Leakage",
            "LeakageDatasets",
        ],
    ]

    # round for better display
    df_leaderboard[["elo", "Elo 95% CI"]] = df_leaderboard[["elo", "Elo 95% CI"]].round(
        0,
    )
    df_leaderboard[["median_time_train_s_per_1K", "rank", "hmr"]] = df_leaderboard[
        ["median_time_train_s_per_1K", "rank", "hmr"]
    ].round(2)
    df_leaderboard[["normalized-score", "median_time_infer_s_per_1K", "improvability"]] = df_leaderboard[
        ["normalized-score", "median_time_infer_s_per_1K", "improvability"]
    ].round(3)

    df_leaderboard = df_leaderboard.sort_values(by="elo", ascending=False)
    df_leaderboard = df_leaderboard.reset_index(drop=True)
    df_leaderboard = df_leaderboard.reset_index(names="#")

    if not include_type:
        df_leaderboard = df_leaderboard.drop(
            columns=[
                "Type",
                "TypeName",
                "MethodClass",
                "Tags",
                "Commercial",
                "License",
                "FitsAtPredict",
                "Organization",
                "Leakage",
                "LeakageDatasets",
            ]
        )

    if compact:
        df_leaderboard = df_leaderboard[
            [
                "method",
                "elo",
                "improvability",
                "median_time_train_s_per_1K",
                "median_time_infer_s_per_1K",
            ]
        ]

        return df_leaderboard.rename(
            columns={
                "median_time_train_s_per_1K": "TrainTime (s/1K)",
                "median_time_infer_s_per_1K": "PredTime (s/1K)",
                "method": "Model",
                "elo": "Elo",
                "improvability": "Impro%",
            },
        )

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
        },
    )
