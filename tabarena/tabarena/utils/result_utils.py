from pathlib import Path
from autogluon.common.loaders import load_pd
from autogluon.common.savers import save_pd
from tabarena.loaders import Paths


def results_path() -> Path:
    res = Paths.results_root_cache
    if not res.exists():
        res.mkdir(parents=True, exist_ok=True)
    return res

def shrink_result_file_size(path_load, path_save):
    result_df = load_pd.load(path_load)

    result_df = result_df.drop(columns=[
        'app_version',
        'can_infer',
        'fit_order',
        'mode',
        'id',
        'seed',
        'stack_level',
        'fit_time',
        'fit_time_marginal',
        'pred_time_test_marginal',
        'pred_time_val_marginal',
        'pred_time_val',
        'utc',
        'version',
    ])

    save_pd.save(path=path_save, df=result_df)


def shrink_ranked_result_file_size(path_load, path_save):
    result_df = load_pd.load(path_load)
    save_pd.save(path=path_save, df=result_df)
