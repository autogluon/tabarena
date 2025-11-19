from __future__ import annotations

default_ensemble_size = 40


# FIXME: Make this non-hardcoded
def get_method_rename_map() -> dict:
    return {
        'KNN': 'KNN',
        'LR': 'Linear',
        'RF': 'RandomForest',
        'XT': 'ExtraTrees',
        'EBM': 'EBM',
        'XGB': 'XGBoost',
        'GBM': 'LightGBM',
        'CAT': 'CatBoost',
        'FASTAI': 'FastaiMLP',
        'NN_TORCH': 'TorchMLP',
        'MNCA_GPU': 'ModernNCA',
        'TABM_GPU': 'TabM',
        'REALMLP_GPU': 'RealMLP',
        'TABDPT_GPU': 'TabDPT',
        'TABICL_GPU': 'TabICL',
        'TABPFNV2_GPU': 'TabPFNv2',

        'MNCA': 'ModernNCA (CPU)',
        'TABM': 'TabM (CPU)',
        'REALMLP': 'RealMLP (CPU)',

        "MITRA_GPU": "Mitra",
        "LIMIX_GPU": "LimiX",
        "XRFM_GPU": "xRFM",
        "BETA_GPU": "BetaTabPFN",
        "TABFLEX_GPU": "TabFlex",
        "REALTABPFN-V2.5": "RealTabPFN-2.5",
    }


def get_framework_type_method_names(
    framework_types,
    max_runtimes: list[tuple[int, str]] = None,
    include_default: bool = True,
    include_best: bool = True,
    include_holdout: bool = True,
    f_map_type_name: dict | None = None,
):
    """

    Parameters
    ----------
    framework_types
    max_runtimes: list[tuple[int, str]], default None
        A list of tuples of:
            1. run time in seconds, or None if uncapped.
            2. custom suffix for the f_map[framework] key (such as `"tuned"`). By specifying `"_4h"`, you will get `"tuned_4h"`
    include_default
    include_best

    Returns
    -------
    f_map, f_map_type, f_map_inverse

    """
    if max_runtimes is None:
        max_runtimes = []
    f_map = dict()
    f_map_type = dict()
    f_map_inverse = dict()

    if f_map_type_name is None:
        f_map_type_name = get_method_rename_map()
    for framework_type in framework_types:
        f_map_cur = dict()
        for max_runtime, suffix in max_runtimes:
            if suffix is None:
                suffix = ""
            f_map_cur[f"tuned{suffix}"] = framework_name(framework_type, max_runtime=max_runtime, ensemble_size=1, tuned=True)
            f_map_cur[f"tuned_ensembled{suffix}"] = framework_name(framework_type, max_runtime=max_runtime, tuned=True)

        if include_default:
            f_map_cur["default"] = framework_name(framework_type, tuned=False)
        if include_best:
            f_map_cur["best"] = framework_name(framework_type, tuned=False, suffix=" (best)")
        if include_holdout:
            f_map_cur["holdout"] = framework_name(framework_type, tuned=False, suffix=" (holdout)")
            f_map_cur["holdout_tuned"] = framework_name(framework_type, tuned=False, suffix=" (tuned, holdout)")
            f_map_cur["holdout_tuned_ensembled"] = framework_name(framework_type, tuned=False, suffix=" (tuned + ensemble, holdout)")

        f_map_inverse_cur = {v: k for k, v in f_map_cur.items()}
        # f_map_type_cur = {v: f_map_type_name[framework_type] for k, v in f_map_cur.items()}
        f_map_type_cur = {v: framework_type for k, v in f_map_cur.items()}
        # f_map[f_map_type_name[framework_type]] = f_map_cur
        f_map[framework_type] = f_map_cur
        f_map_type.update(f_map_type_cur)
        f_map_inverse.update(f_map_inverse_cur)
    return f_map, f_map_type, f_map_inverse, f_map_type_name


def get_f_map_suffix_plots() -> dict:
    f_map_suffix_plots = dict(
        default="-D",
        tuned="-T",
        tuned_ensembled="-TE",
        best="-B",
        holdout="-D (H)",
        holdout_tuned="-T (H)",
        holdout_tuned_ensembled="-TE (H)",
    )
    return f_map_suffix_plots


def framework_name(framework_type, max_runtime=None, ensemble_size=default_ensemble_size, tuned: bool=True, all: bool = False, prefix: str = None, suffix: str = None) -> str:
    method = framework_type if framework_type else "All"
    if prefix is None:
        prefix = ""
    if all:
        method = "All"
    if suffix is None:
        if not tuned:
            suffix = " (default)"
        else:
            suffix = " (tuned + ensemble)" if ensemble_size > 1 else " (tuned)"
    if max_runtime:
        suffix += time_suffix(max_runtime=max_runtime)
    method = f"{method}{prefix}{suffix}"
    return method


def time_suffix(max_runtime: float) -> str:
    if max_runtime:
        if max_runtime >= 3600:
            str_num_hours = f"{int(max_runtime / 3600)}" if max_runtime % 3600 == 0 else f"{max_runtime / 3600:0.2f}"
            return f" ({str_num_hours}h)"
        else:
            str_num_mins = f"{int(max_runtime / 60)}" if max_runtime % 60 == 0 else f"{max_runtime / 60:0.2f}"
            return f" ({str_num_mins}m)"
    else:
        return ""
