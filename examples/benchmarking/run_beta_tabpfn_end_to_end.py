from __future__ import annotations

from pathlib import Path

from tabarena.nips2025_utils.artifacts.download_utils import download_and_extract_zip
from tabarena.nips2025_utils.end_to_end_single import EndToEndResultsSingle, EndToEndSingle

if __name__ == "__main__":
    method = "BetaTabPFN"
    path_raw = Path("local_data") / method
    fig_output_dir = Path("tabarena_figs") / method
    download = True
    cache = True
    if download:
        url = "https://data.lennart-purucker.com/tabarena/data_BetaTabPFN.zip"
        download_and_extract_zip(url=url, path_local=path_raw)

    """
    Run logic end-to-end and cache all results:
    1. load raw artifacts
        path_raw should be a directory containing `results.pkl` files for each run.
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
    if cache:
        end_to_end = EndToEndSingle.from_path_raw(path_raw=path_raw)

    """
    Load cached results and compare on TabArena
    1. Generates figures and leaderboard using the TabArena methods and the user's method
    2. Missing values are imputed to default RandomForest.
    """
    end_to_end_results = EndToEndResultsSingle.from_cache(method=method)
    leaderboard = end_to_end_results.compare_on_tabarena(output_dir=fig_output_dir)
    print(leaderboard)
