import re
import time

import pandas as pd

from .utils.create_feature_and_featurename import create_feature
from .utils.get_data import get_openml_dataset_split_and_metadata
from .utils.get_matrix import get_additional_pandas_columns
from .utils.get_metafeatures import get_pandas_metafeatures
from .utils.get_operators import get_operators


def get_operator_count(featurename, operators):
    featurename_correct = featurename.replace("+", "add")
    featurename_correct = featurename_correct.replace("-(", "subtract(")
    featurename_correct = featurename_correct.replace("*", "multiply")
    featurename_correct = featurename_correct.replace("/", "divide")
    operator_count = 0
    sorted_operators = sorted(operators, key=len, reverse=True)
    for i, n in enumerate(sorted_operators):
        if n == "+":
            sorted_operators[i] = "add"
        if n == "-":
            sorted_operators[i] = "subtract"
        if n == "*":
            sorted_operators[i] = "multiply"
        if n == "/":
            sorted_operators[i] = "divide"

    for op in sorted_operators:
        pattern = rf'\b{op}\s*\('
        matches = re.findall(pattern, featurename_correct)
        pattern_without = rf'\b{op}\s*\ - '
        matches_without = re.findall(pattern_without, featurename_correct)
        count = len(matches + matches_without)
        operator_count += count
    return operator_count


def split_top_level_args(arg_str):
    args = []
    bracket_level = 0
    current_arg = []
    for char in arg_str:
        if char == ',' and bracket_level == 0:
            args.append(''.join(current_arg).strip())
            current_arg = []
        else:
            if char == '(':
                bracket_level += 1
            elif char == ')':
                bracket_level -= 1
            current_arg.append(char)
    if current_arg:
        args.append(''.join(current_arg).strip())
    return args


def add_pandas_metadata_columns(dataset_metadata, X_train, result_matrix):
    columns = get_additional_pandas_columns()
    new_columns = pd.DataFrame(index=result_matrix.index, columns=columns)
    unary_operators, binary_operators = get_operators()
    operators = unary_operators + binary_operators + ["without"]
    for row in result_matrix.iterrows():
        dataset = row[1][0]
        featurename = row[1][1]
        operator_count = get_operator_count(featurename, operators)
        X_train_copy = X_train.copy()
        if featurename.startswith("without"):
            feature_to_delete = featurename.split(" - ")[1]
            X_train_copy = X_train_copy.drop(feature_to_delete, axis=1)
        elif operator_count > 1:
            features = featurename.split("(")[1].replace(")", "").replace(" ", "")
            inner = featurename.split("(", 1)[1].rsplit(")", 1)[0]
            args = split_top_level_args(inner)
            featurename1 = args[0]
            featurename2 = args[1] if len(args) > 1 else None
            if "(" in featurename1:
                feature1 = X_train_copy[featurename1]
            else:
                featurename1 = features.split(",")[0]
                feature1 = X_train_copy[featurename1]
            if featurename2 is not None:
                if "(" in featurename2:
                    feature2 = X_train_copy[featurename2]
                else:
                    feature2 = X_train_copy[featurename2]
            else:
                feature2 = None
            new_feature = create_feature(feature1, feature2, featurename)
            X_train_copy = X_train_copy.reset_index(drop=True)
            new_feature_df = pd.DataFrame(new_feature, columns=[featurename])
            new_feature_df = new_feature_df.reset_index(drop=True)
            X_train_copy = pd.concat([X_train_copy, new_feature_df], axis=1, ignore_index=False)
        else:
            features = featurename.split("(")[1].replace(")", "").replace(" ", "")
            if "," in features:
                featurename1 = features.split(",")[0]
                feature1 = X_train_copy[featurename1]
                featurename2 = features.split(",")[1]
                feature2 = X_train_copy[featurename2]
            else:
                featurename1 = features
                feature1 = X_train_copy[featurename1]
                feature2 = None
            new_feature = create_feature(feature1, feature2, featurename)
            X_train_copy = X_train_copy.reset_index(drop=True)
            new_feature_df = pd.DataFrame(new_feature, columns=[featurename])
            new_feature_df = new_feature_df.reset_index(drop=True)
            X_train_copy = pd.concat([X_train_copy, new_feature_df], axis=1, ignore_index=False)
        try:
            feature = pd.DataFrame(X_train_copy[featurename])
            feature_metadata = get_pandas_metafeatures(feature, featurename)
            new_row = pd.DataFrame(columns=columns)
            new_row.loc[len(result_matrix)] = [
                dataset_metadata["task_type"],
                feature_metadata["feature - count"],
                feature_metadata["feature - unique"],
                feature_metadata["feature - top"],
                feature_metadata["feature - freq"],
                feature_metadata["feature - mean"],
                feature_metadata["feature - std"],
                feature_metadata["feature - min"],
                feature_metadata["feature - 25"],
                feature_metadata["feature - 50"],
                feature_metadata["feature - 75"],
                feature_metadata["feature - max"],
            ]
            matching_indices = result_matrix[result_matrix["feature - name"] == str(featurename)].index
            for idx in matching_indices:
                new_columns.loc[idx] = new_row.iloc[0]
        except KeyError:
            feature = pd.DataFrame(X_train[feature_to_delete])
            feature_metadata = get_pandas_metafeatures(feature, feature_to_delete)
            new_row = pd.DataFrame(columns=columns)
            new_row.loc[len(result_matrix)] = [
                dataset_metadata["task_type"],
                feature_metadata["feature - count"],
                feature_metadata["feature - unique"],
                feature_metadata["feature - top"],
                feature_metadata["feature - freq"],
                feature_metadata["feature - mean"],
                feature_metadata["feature - std"],
                feature_metadata["feature - min"],
                feature_metadata["feature - 25"],
                feature_metadata["feature - 50"],
                feature_metadata["feature - 75"],
                feature_metadata["feature - max"],
            ]
            matching_indices = result_matrix[result_matrix["feature - name"] == str(featurename)].index
            for idx in matching_indices:
                new_columns.loc[idx] = new_row.iloc[0]
    insert_position = result_matrix.shape[1] - 2
    result_matrix = pd.concat([result_matrix.iloc[:, :insert_position], new_columns, result_matrix.iloc[:, insert_position:]], axis=1)
    return result_matrix


def add_pandas_metadata_selection_columns(dataset_metadata, X_train, result_matrix):
    columns = get_additional_pandas_columns()
    new_columns = pd.DataFrame(index=result_matrix.index, columns=columns)
    operators = ["without"]
    for row in result_matrix.iterrows():
        dataset = row[1][0]
        featurename = row[1][1]
        operator_count = get_operator_count(featurename, operators)
        X_train_copy = X_train.copy()
        feature_to_delete = featurename.split(" - ")[1]
        X_train_copy = X_train_copy.drop(feature_to_delete, axis=1)
        try:
            feature = pd.DataFrame(X_train_copy[featurename])
            feature_metadata = get_pandas_metafeatures(feature, featurename)
            new_row = pd.DataFrame(columns=columns)
            new_row.loc[len(result_matrix)] = [
                dataset_metadata["task_type"],
                feature_metadata["feature - count"],
                feature_metadata["feature - unique"],
                feature_metadata["feature - top"],
                feature_metadata["feature - freq"],
                feature_metadata["feature - mean"],
                feature_metadata["feature - std"],
                feature_metadata["feature - min"],
                feature_metadata["feature - 25"],
                feature_metadata["feature - 50"],
                feature_metadata["feature - 75"],
                feature_metadata["feature - max"],
            ]
            matching_indices = result_matrix[result_matrix["feature - name"] == str(featurename)].index
            for idx in matching_indices:
                new_columns.loc[idx] = new_row.iloc[0]
        except KeyError:
            feature = pd.DataFrame(X_train[feature_to_delete])
            feature_metadata = get_pandas_metafeatures(feature, feature_to_delete)
            new_row = pd.DataFrame(columns=columns)
            new_row.loc[len(result_matrix)] = [
                dataset_metadata["task_type"],
                feature_metadata["feature - count"],
                feature_metadata["feature - unique"],
                feature_metadata["feature - top"],
                feature_metadata["feature - freq"],
                feature_metadata["feature - mean"],
                feature_metadata["feature - std"],
                feature_metadata["feature - min"],
                feature_metadata["feature - 25"],
                feature_metadata["feature - 50"],
                feature_metadata["feature - 75"],
                feature_metadata["feature - max"],
            ]
            matching_indices = result_matrix[result_matrix["feature - name"] == str(featurename)].index
            for idx in matching_indices:
                new_columns.loc[idx] = new_row.iloc[0]
    insert_position = result_matrix.shape[1] - 2
    result_matrix = pd.concat([result_matrix.iloc[:, :insert_position], new_columns, result_matrix.iloc[:, insert_position:]], axis=1)
    return result_matrix


def main():
    result_matrix = pd.read_parquet("src/Metadata/core/Core_Matrix_Complete.parquet")
    columns = get_additional_pandas_columns()
    result_matrix_columns = result_matrix.columns.values.tolist()
    columns = columns + result_matrix_columns
    result_matrix_pandas = pd.DataFrame(columns=columns)
    start = time.time()
    counter = 0
    datasets = list(result_matrix.groupby('dataset - id').groups.keys())
    for dataset in datasets:
        print("Dataset: " + str(dataset))
        try:
            pd.read_parquet("src/Metadata/pandas/Pandas_Matrix_Complete" + str(dataset) + ".parquet")
        except FileNotFoundError:
            try:
                counter += 1
                start_dataset = time.time()
                X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(int(str(dataset)))
                result_matrix_dataset = result_matrix[result_matrix['dataset - id'] == dataset]
                result_matrix_dataset = add_pandas_metadata_columns(dataset_metadata, X_train, result_matrix_dataset)
                result_matrix_pandas.columns = result_matrix_dataset.columns
                result_matrix_pandas = pd.concat([result_matrix_pandas, result_matrix_dataset], axis=0)
                result_matrix_pandas.to_parquet("src/Metadata/pandas/Pandas_Matrix_Complete" + str(dataset) + ".parquet")
                end_dataset = time.time()
                print("Time for Pandas on Dataset " + str(dataset) + ": " + str(end_dataset - start_dataset))
            except TypeError:
                continue
            except KeyError:
                continue
    result_matrix_pandas.to_parquet("src/Metadata/pandas/Pandas_Matrix_Complete.parquet")
    end = time.time()
    print("Time for complete Pandas MF: " + str(end - start) + " on " + str(counter) + " datasets.")


if __name__ == '__main__':
    main()
