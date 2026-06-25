"""TabArena Artifact Download (all tiers, all methods).

Example script showing how to download the different tiers of artifacts in TabArena.

WARNING: Running this script unedited downloads *every* tier for *every* method
(raw + processed + results), which requires ~1 TB of disk space. Restrict
``method_metadata_lst`` and/or comment out tiers you do not need before running.

All artifacts are saved under ``~/.cache/tabarena/artifacts/`` (override the cache
location with the ``TABARENA_CACHE`` environment variable).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabarena.contexts import TabArenaContext

if TYPE_CHECKING:
    from tabarena.models._method_metadata import MethodMetadata

if __name__ == "__main__":
    tabarena_context = TabArenaContext()
    method_metadata_lst: list[MethodMetadata] = tabarena_context.method_metadata_collection.method_metadata_lst

    for method_metadata in method_metadata_lst:
        method_downloader = method_metadata.method_downloader(verbose=True)

        # Raw data: very large (~1 TB for all methods). Raw contains all available
        # information and, uniquely:
        # 1. test predictions (probabilities) for all inner-fold models and the bagged ensemble
        # 2. val predictions (probabilities) for all inner-fold models
        # 3. test and val scores
        # 4. model hyperparameters
        # 5. train time, inference time, total time
        # 6. available memory, disk space usage, cpu count, gpu count
        # 7. numerous task metadata fields
        # 8. numerous model metadata fields
        method_downloader.download_raw()

        # Processed data: much smaller (~100 GB for all methods). Contains the
        # information needed to simulate model portfolios and hyperparameter
        # optimization, stored in an `EvaluationRepository` with many quality-of-life
        # features. We recommend most users interact with processed data, not raw.
        method_downloader.download_processed()

        # Results data (<100 MB): pandas DataFrames keyed by (method, dataset, fold)
        # with test error, val error, training time, inference time, and more.
        method_downloader.download_results()

    # Get the hyperparameters for all configs
    configs_hyperparameters: dict[str, dict] = tabarena_context.load_configs_hyperparameters(download="auto")
