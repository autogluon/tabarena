from __future__ import annotations

from huggingface_hub import hf_hub_download

# Toggle individual models on/off here.
DOWNLOAD_TABPFN = True
DOWNLOAD_TABICL = True
DOWNLOAD_TABDPT = True
DOWNLOAD_MITRA = True
DOWNLOAD_TABFLEX = True
DOWNLOAD_LIMIX = True
DOWNLOAD_SAP_RPT_OSS = True
DOWNLOAD_TABSTAR = True


def download_tabpfn() -> None:
    # Note: models from version 2.5 are gated! You need to accept the terms and
    # conditions on Hugging Face and login on your device with the Hugging Face CLI
    # to download the weights.
    try:
        from tabpfn.model_loading import download_all_models, resolve_model_path
    except ImportError:
        print("TabPFN not installed. Skipping downloading its models.")
        return
    _, model_dir, _, _ = resolve_model_path(model_path=None, which="classifier")
    download_all_models(to=model_dir[0])


def download_tabicl() -> None:
    try:
        from tabicl import TabICLClassifier
    except ImportError:
        print("TabICL not installed. Skipping downloading its models.")
        return
    TabICLClassifier(checkpoint_version="tabicl-classifier-v1.1-0506.ckpt")._load_model()
    TabICLClassifier(checkpoint_version="tabicl-classifier-v1-0208.ckpt")._load_model()


def download_tabdpt() -> None:
    try:
        from tabdpt.estimator import TabDPTEstimator
    except ImportError:
        print("TabDPT not installed. Skipping downloading its models.")
        return
    TabDPTEstimator.download_weights()


def download_mitra() -> None:
    for repo_id in ["autogluon/mitra-classifier", "autogluon/mitra-regressor"]:
        hf_hub_download(repo_id=repo_id, filename="config.json")
        hf_hub_download(repo_id=repo_id, filename="model.safetensors")


def download_tabflex() -> None:
    try:
        from tabarena.benchmark.models.ag.tabflex.tabflex_model import TabFlexModel
    except ImportError:
        print("TabFlexModel not found. Skipping downloading its models.")
        return
    TabFlexModel._download_all_models()


def download_limix() -> None:
    try:
        from tabarena.models.limix.model import LimiXModel
    except ImportError:
        print("LimiXModel not found. Skipping downloading its models.")
        return
    LimiXModel.download_model()


def download_sap_rpt_oss() -> None:
    # Gated, requires accepting terms on Hugging Face!
    try:
        from tabarena.models.sap_rpt_oss.model import pre_download_model
    except ImportError:
        print("SAP RPT-1 OSS model not found. Skipping downloading its model.")
        return
    pre_download_model()


def download_tabstar() -> None:
    try:
        from tabstar.tabstar_model import BaseTabSTAR
    except ImportError:
        print("TabStar import not found. Skipping downloading its models.")
        return
    BaseTabSTAR.download_base_model()


# TODO: move each function into the matching model class and call them through a
# standardized interface so this file becomes a thin dispatcher.
DOWNLOADERS: list[tuple[str, bool, callable]] = [
    ("TabPFN", DOWNLOAD_TABPFN, download_tabpfn),
    ("TabICL", DOWNLOAD_TABICL, download_tabicl),
    ("TabDPT", DOWNLOAD_TABDPT, download_tabdpt),
    ("Mitra", DOWNLOAD_MITRA, download_mitra),
    ("TabFlex", DOWNLOAD_TABFLEX, download_tabflex),
    ("LimiX", DOWNLOAD_LIMIX, download_limix),
    ("SAP-RPT-OSS", DOWNLOAD_SAP_RPT_OSS, download_sap_rpt_oss),
    ("TabSTAR", DOWNLOAD_TABSTAR, download_tabstar),
]


def main() -> None:
    for name, enabled, fn in DOWNLOADERS:
        if not enabled:
            print(f"[skip] {name}: disabled by flag")
            continue
        print(f"[download] {name}")
        fn()


if __name__ == "__main__":
    main()
