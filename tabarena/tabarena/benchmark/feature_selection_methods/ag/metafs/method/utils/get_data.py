import datetime
import os

import numpy as np
import pandas as pd
import openml
from openfe import OpenFE, transform
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def get_all_amlb_dataset_ids():
    # Code from https://github.com/openml/automlbenchmark/blob/2c2a93dc3fc65fc3d6a77fe97ec9df1108551075/scripts/find_matching_datasets.py
    # small_config_url = "https://raw.githubusercontent.com/openml/automlbenchmark/2c2a93dc3fc65fc3d6a77fe97ec9df1108551075/resources/benchmarks/small.yaml"
    #medium_config_url = "https://raw.githubusercontent.com/openml/automlbenchmark/2c2a93dc3fc65fc3d6a77fe97ec9df1108551075/resources/benchmarks/medium.yaml"
    #large_config_url = "https://raw.githubusercontent.com/openml/automlbenchmark/2c2a93dc3fc65fc3d6a77fe97ec9df1108551075/resources/benchmarks/large.yaml"
    # small_configuration = yaml.load(requests.get(small_config_url).text, Loader=yaml.Loader)
    #medium_configuration = yaml.load(requests.get(medium_config_url).text, Loader=yaml.Loader)
    #large_configuration = yaml.load(requests.get(large_config_url).text, Loader=yaml.Loader)
    # benchmark_tids = set(
        # [problem.get("openml_task_id") for problem in small_configuration]
        # + [problem.get("openml_task_id") for problem in medium_configuration]
        # + [problem.get("openml_task_id") for problem in large_configuration]
    # )
    benchmark_tids = [146818, 146820, 168350, 168911, 190137, 190411, 359955, 359956, 359979]
    return benchmark_tids


def get_openml_dataset_split_and_metadata(openml_task_id: int) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict
]:
    task = openml.tasks.get_task(
        openml_task_id,
        download_splits=True,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    dataset_metadata = {"task_id": task.task_id, "task_type": task.task_type, "number_of_classes": 'N/A'}
    train_idx, test_idx = task.get_train_test_split_indices()
    X, y = task.get_X_and_y(dataset_format="dataframe")  # type: ignore
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    return X_train, y_train, X_test, y_test, dataset_metadata


def get_openml_dataset_split(openml_task_id: int) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame
]:
    task = openml.tasks.get_task(
        openml_task_id,
        download_splits=True,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    train_idx, test_idx = task.get_train_test_split_indices()
    X, y = task.get_X_and_y(dataset_format="dataframe")  # type: ignore
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    return X_train, y_train, X_test, y_test


def get_openml_dataset_and_metadata(openml_task_id: int) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict
]:
    task = openml.tasks.get_task(
        openml_task_id,
        download_splits=True,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    dataset_metadata = {"task_id": task.task_id, "task_type": task.task_type, "number_of_classes": len(task.class_labels) if task.class_labels else 'N/A'}
    X, y = task.get_X_and_y(dataset_format="dataframe")  # type: ignore
    return X, y, dataset_metadata


def get_openml_dataset(openml_task_id: int) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    task = openml.tasks.get_task(
        openml_task_id,
        download_splits=True,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    X, y = task.get_X_and_y(dataset_format="dataframe")  # type: ignore
    return X, y


def split_data(data, target_label, seed) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame
]:
    y = data[target_label]
    X = data.drop(target_label, axis=1)
    train_idx, test_idx, y_train, y_test = train_test_split(X.index, y, test_size=0.2, random_state=seed)
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    return X_train, y_train, X_test, y_test


def concat_data(X_train, y_train, X_test, y_test, target_label):
    y_train = y_train.to_frame(target_label)
    X_train.index = y_train.index
    train_data = pd.concat([X_train, y_train], axis=1)
    y_test = y_test.to_frame(target_label)
    X_test.index = y_test.index
    test_data = pd.concat([X_test, y_test], axis=1)
    data = pd.concat([train_data, test_data], axis=0)
    return data


def preprocess_data(train_x, test_x) -> (pd.DataFrame, pd.DataFrame):
    cols = train_x.columns
    cat_columns = train_x.select_dtypes(['category']).columns
    obj_columns = train_x.select_dtypes(['object']).columns
    train_x[cat_columns] = train_x[cat_columns].apply(lambda x: pd.factorize(x, use_na_sentinel=True)[0])
    test_x[cat_columns] = test_x[cat_columns].apply(lambda x: pd.factorize(x, use_na_sentinel=True)[0])
    train_x[obj_columns] = train_x[obj_columns].apply(lambda x: pd.factorize(x, use_na_sentinel=True)[0])
    test_x[obj_columns] = test_x[obj_columns].apply(lambda x: pd.factorize(x, use_na_sentinel=True)[0])
    imp_nan = SimpleImputer(missing_values=np.nan, strategy='mean')
    train_x = imp_nan.fit_transform(train_x)
    test_x = imp_nan.transform(test_x)
    imp_m1 = SimpleImputer(missing_values=-1, strategy='mean')
    train_x = imp_m1.fit_transform(train_x)
    test_x = imp_m1.transform(test_x)
    train_x = pd.DataFrame(train_x).fillna(0)
    test_x = pd.DataFrame(test_x).fillna(0)
    train_x.columns = cols
    test_x.columns = cols
    return train_x, test_x


def get_dataset_split(dataset_id: int, seed) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict
]:
    if dataset_id == 1:
        data = pd.read_parquet("data/original/1.parquet")
        y = data["Event"]
        X = data.drop(columns=["Event"])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        dataset_metadata = {"task_id": 1, "task_type": "Supervised Classification", "number_of_classes": 'N/A'}
    elif dataset_id == 2:
        data = pd.read_parquet("data/original/2.parquet")
        data = data.dropna(subset=["Event"])
        y = data["Event"]
        print(y.value_counts(dropna=False))
        X = data.drop(columns=["Event", "Sea breezes", "Fogs", "Storms", "Passage of Front"])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        dataset_metadata = {"task_id": 2, "task_type": "Multiclass", "number_of_classes": 'N/A'}
    else:
        task = openml.tasks.get_task(
            dataset_id,
            download_splits=True,
            download_data=True,
            download_qualities=True,
            download_features_meta_data=True,
        )
        dataset_metadata = {"task_id": task.task_id, "task_type": task.task_type, "number_of_classes": 'N/A'}
        train_idx, test_idx = task.get_train_test_split_indices()
        X, y = task.get_X_and_y(dataset_format="dataframe")  # type: ignore
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        data = concat_data(X_train, y_train, X_test, y_test, "target")
        X_train, y_train, X_test, y_test = split_data(data, "target", seed)
    return X_train, y_train, X_test, y_test, dataset_metadata


def get_physics_data():
    df = pd.read_excel('../data/original/matrice_cluster_usa_AS_PlusFronts.xlsx')
    # Write the Parquet file
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')  # converts string → datetime
    df['hour'] = df['DateTime'].dt.hour
    df['minute'] = df['DateTime'].dt.minute
    df['day'] = df['DateTime'].dt.day
    df['month'] = df['DateTime'].dt.month
    df['year'] = df['DateTime'].dt.year
    # then drop the original datetime columns
    X = df.drop(columns=["DateTime", "hour UTC", "Dates"])
    X.to_parquet('../data/original/2.parquet')


if __name__ == '__main__':
    get_physics_data()
