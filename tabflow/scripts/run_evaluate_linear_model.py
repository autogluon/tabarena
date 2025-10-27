from pathlib import Path

import pandas as pd

from tabrepo.nips2025_utils.tabarena_context import TabArenaContext
from tabrepo.nips2025_utils.end_to_end_single import EndToEndSingle, EndToEndResultsSingle


"""
First refer to `run_jobs_linear_model.py
"""
if __name__ == '__main__':
    bucket = "prateek-ag"
    prefix = "tabarena-lr-2025-10-16"
    local_dir =  Path(f"/home/ubuntu/workspace/data/{prefix}/")
    from s3_downloader import copy_s3_prefix_to_local
    copy_s3_prefix_to_local(
        bucket=bucket,
        prefix=prefix,
        dest_dir=local_dir,
        max_workers=64,
        exclude=["*.log"],
    )

    method = "LinearModel"
    path_raw = local_dir / "data"
    fig_output_dir = Path("tabarena_figs") / method
    cache = True

    """
    Run logic end-to-end and cache all results:
    1. load raw artifacts
        path_raw should be a directory containing `results.pkl` files for each run.
        In the current code, we require `path_raw` to contain the results of only 1 type of method.
    2. infer method_metadata
    3. cache method_metadata
    4. cache raw artifacts
    5. infer task_metadata
    5. generate processsed
    6. cache processed
    7. generate results
    8. cache results

    Once this is executed once, it does not need to be ran again.
    """
    from tabrepo.nips2025_utils.artifacts._tabarena_method_metadata_2025_10_20 import lr_metadata

    if cache:
        task_metadata = TabArenaContext().task_metadata
        end_to_end = EndToEndSingle.from_path_raw(
            path_raw=path_raw,
            method_metadata=lr_metadata,
            task_metadata=task_metadata,
        )

    """
    Load cached results and compare on TabArena
    1. Generates figures and leaderboard using the TabArena methods and the user's method
    2. Compares on all datasets if `filter_dataset_fold=False`, else only tasks from the user's method if `filter_dataset_fold=True`.
    3. Missing values are imputed to default RandomForest.
    """
    end_to_end_results = EndToEndResultsSingle.from_cache(method=lr_metadata)

    leaderboard: pd.DataFrame = end_to_end_results.compare_on_tabarena(
        output_dir=fig_output_dir,
        only_valid_tasks=True,
    )

    print(leaderboard.to_markdown())
