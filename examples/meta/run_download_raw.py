from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.nips2025_utils.tabarena_context import TabArenaContext

"""
This is an example script showcasing how to access the different types of artifacts in TabArena.
Running this script without edits will download all artifacts in TabArena, requiring ~1 TB of disk space.
"""
if __name__ == "__main__":
    tabarena_context = TabArenaContext()
    method_metadata_lst: list[MethodMetadata] = tabarena_context.method_metadata_collection.method_metadata_lst

    for method_metadata in method_metadata_lst:
        method_downloader = method_metadata.method_downloader(verbose=True)

        # download raw data for all methods, very large (1 TB)
        # raw contains all available information. Uniquely, it contains:
        # 1. test prediction (probabilities) for all inner-fold models and the overall bagged ensemble
        # 2. val prediction (probabilities) for all inner-fold models.
        # 3. test and val scores
        # 4. Model hyperparameters
        # 5. Train time, inference time, total time
        # 6. Available memory, disk space usage, cpu count, gpu count
        # 6. Numerous task metadata information
        # 7. Numerous model metadata information
        # saved to: ~/.cache/tabarena/artifacts/
        method_downloader.download_raw()

        # download processed data, much smaller (100 GB)
        # processed data contains the information needed to simulate model portfolios and hyperparameter optimization.
        # processed data is stored in a `EvaluationRepository` object with many quality of life features.
        # We recommend most users to interact with the processed data instead of the raw data.
        # saved to: ~/.cache/tabarena/artifacts/
        method_downloader.download_processed()

        # download results data (<100 MB)
        # The results data are stored as pandas DataFrames with (method, dataset, fold) as the unique key.
        # contains the test error, val error, training time, inference time, and more.
        # saved to: ~/.cache/tabarena/artifacts/
        method_downloader.download_results()

    # Get the hyperparameters for all configs
    configs_hyperparameters: dict[str, dict] = tabarena_context.load_configs_hyperparameters(download="auto")
