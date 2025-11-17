import time

from autogluon.common.loaders import load_pkl
from autogluon.common.savers import save_pkl


def download_zs_metadata(
        path_prefix_in,
        path_prefix_out,
        name_in_gt,
        name_in_pred_proba,
        name_out_gt=None,
        name_out_pred_proba=None):
    if name_out_gt is None:
        name_out_gt = name_in_gt
    if name_out_pred_proba is None:
        name_out_pred_proba = name_in_pred_proba
    path_gt = path_prefix_in + name_in_gt
    path_pred_proba = path_prefix_in + name_in_pred_proba
    save_path_gt = str(path_prefix_out / name_out_gt)
    save_path_pred_proba = str(path_prefix_out / name_out_pred_proba)

    ts = time.time()
    print(f'Downloading Ground Truth File: {path_gt}')
    zeroshot_gt = load_pkl.load(path_gt)
    print(f'Downloaded Ground Truth File... {round(time.time() - ts, 2)}s')

    ts = time.time()
    print(f'Saving Ground Truth File: {save_path_gt}')
    save_pkl.save(path=save_path_gt, object=zeroshot_gt)
    print(f'Saved Ground Truth File... {round(time.time() - ts, 2)}s')

    ts = time.time()
    print(f'Downloading Pred Proba File: {path_pred_proba}')
    zeroshot_pred_proba = load_pkl.load(path_pred_proba)
    print(f'Downloaded Pred Proba File... {round(time.time() - ts, 2)}s')

    ts = time.time()
    print(f'Saving Pred Proba File: {save_path_pred_proba}')
    save_pkl.save(path=save_path_pred_proba, object=zeroshot_pred_proba)
    print(f'Saved Pred Proba File... {round(time.time() - ts, 2)}s')
