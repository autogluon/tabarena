from __future__ import annotations

from huggingface_hub import hf_hub_download

if __name__ == "__main__":
    # TabPFNv2
    try:
        from tabpfn.model.loading import download_all_models, resolve_model_path
    except ImportError:
        print("TabPFNv2 not installed. Skipping downloading its models.")
    else:
        # TODO: need to hardcode the model names as names have changed in the code below...
        _, model_dir, _, _ = resolve_model_path(model_path=None, which="classifier")
        download_all_models(to=model_dir)

    # TabICL
    try:
        from tabicl import TabICLClassifier
    except ImportError:
        print("TabICL not installed. Skipping downloading its models.")
    else:
        TabICLClassifier(checkpoint_version="tabicl-classifier-v1.1-0506.ckpt")._load_model()
        TabICLClassifier(checkpoint_version="tabicl-classifier-v1-0208.ckpt")._load_model()

    # TabDPT
    try:
        from tabdpt.estimator import TabDPTEstimator
    except ImportError:
        print("TabDPT not installed. Skipping downloading its models.")
    else:
        TabDPTEstimator.download_weights()

    # Mitra
    for repo_id in ["autogluon/mitra-classifier", "autogluon/mitra-regressor"]:
        hf_hub_download(repo_id=repo_id, filename="config.json")
        hf_hub_download(repo_id=repo_id, filename="model.safetensors")

    # TabFlex
    try:
        from tabarena.benchmark.models.ag.tabflex.tabflex_model import TabFlexModel
    except ImportError:
        print("TabFlexModel not found. Skipping downloading its models.")
    else:
        TabFlexModel._download_all_models()

    # LimiX
    try:
        from tabarena.benchmark.models.ag.limix.limix_model import LimiXModel
    except ImportError:
        print("LimiXModel not found. Skipping downloading its models.")
    else:
        LimiXModel.download_model()