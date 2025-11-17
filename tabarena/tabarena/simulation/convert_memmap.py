"""
Script to convert old pickle format to memmap representation. Can be removed at some point, just left for future reference.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from tabarena.predictions.tabular_predictions import path_memmap
from tabarena.utils.parallel_for import parallel_for


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def convert_memmap_pred_from_pickle(folder_pickle: Path, output_dir: Path, dtype: str = "float32"):
    """
    Converts old format to memmap, can be deleted once we converted previous pickle formats.
    :param folder_pickle:
    :param output_dir:
    :param dtype:
    :return:
    """
    assert folder_pickle.exists(), f"folder {folder_pickle} does not exist"
    assert dtype in ["float16", "float32"]
    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_files = list(Path(folder_pickle).rglob("*zeroshot_pred_proba.pkl"))

    def convert_file(filename):
        dataset = str(filename.parent.parent.stem)
        fold = int(filename.parent.stem)
        target_folder = path_memmap(output_dir, dataset, fold)
        target_folder.mkdir(exist_ok=True, parents=True)
        print(dataset, fold)
        if all((target_folder / f).exists() for f in ["pred-val.dat", "pred-test.dat", "metadata.json"]):
            print(f"skipping generation of {dataset} {fold} as files already exists")
            return
        preds = load_pickle(filename)
        models = list(preds[dataset][fold]["pred_proba_dict_val"].keys())
        assert set(preds[dataset][fold]["pred_proba_dict_test"].keys()) == set(models), \
            "different models available on validation and testing"
        pred_val = np.stack([preds[dataset][fold]["pred_proba_dict_val"][model] for model in models])
        pred_test = np.stack([preds[dataset][fold]["pred_proba_dict_test"][model] for model in models])

        with open(target_folder / "metadata.json", "w") as f:
            metadata_dict = {
                "models": models,
                "dataset": dataset,
                "fold": fold,
                "pred_val_shape": pred_val.shape,
                "pred_test_shape": pred_test.shape,
                "dtype": dtype,
            }

            f.write(json.dumps(metadata_dict))

        fp = np.memmap(target_folder / "pred-val.dat", dtype=dtype, mode='w+', shape=pred_val.shape)
        fp[:] = pred_val[:]

        fp = np.memmap(target_folder / "pred-test.dat", dtype=dtype, mode='w+', shape=pred_test.shape)
        fp[:] = pred_test[:]

    parallel_for(
        convert_file, context={}, inputs=[[x] for x in pkl_files], engine="sequential", #engine_kwargs={"num_cpus": 32}
    )


def convert_memmap_label_from_pickle(folder_pickle: Path, output_dir: Path, dtype: str = "float32"):
    assert folder_pickle.exists(), f"folder {folder_pickle} does not exist"
    assert dtype in ["float16", "float32"]
    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_files = list(Path(folder_pickle).rglob("*zeroshot_gt.pkl"))

    def convert_file(filename):
        dataset = str(filename.parent.parent.stem)
        fold = int(filename.parent.stem)
        target_folder = path_memmap(output_dir, dataset, fold)
        target_folder.mkdir(exist_ok=True, parents=True)
        # print(dataset, fold)
        if all((target_folder / f).exists() for f in ["label-val.csv.zip", "label-test.csv.zip"]):
            print(f"skipping generation of {dataset} {fold} as files already exists")
            return
        labels = load_pickle(filename)[dataset][fold]
        labels_val = labels['y_val']
        labels_test = labels['y_test']
        labels_val.to_csv(target_folder / "label-val.csv.zip", index=True)
        labels_test.to_csv(target_folder / "label-test.csv.zip", index=True)

    parallel_for(
        convert_file, context={}, inputs=[[x] for x in pkl_files], engine="sequential",
    )


if __name__ == '__main__':
    context = "2023_08_21_micro"
    prefix = "/home/ubuntu/workspace/code/autogluon-zeroshot/data/results/"
    path_pickle = f"{prefix}{context}/zeroshot_metadata/"
    memmap_dir = f"{prefix}{context}/model_predictions"
    convert_memmap_pred_from_pickle(Path(path_pickle), Path(memmap_dir), dtype="float32")
    convert_memmap_label_from_pickle(Path(path_pickle), Path(memmap_dir), dtype="float32")

    memmap_dir = f"{prefix}{context}/model_predictions_float16/"
    convert_memmap_pred_from_pickle(Path(path_pickle), Path(memmap_dir), dtype="float16")
    convert_memmap_label_from_pickle(Path(path_pickle), Path(memmap_dir), dtype="float16")
