import numpy as np
import pandas as pd
import re
from .get_operators import get_operators


def create_feature_and_featurename(feature1, feature2, operator):
    if feature2 is None:
        feature, featurename = create_unary_feature_and_featurename(feature1, operator)
    else:
        feature, featurename = create_binary_feature_and_featurename(feature1, feature2, operator)
    return feature, featurename


def create_feature(feature1, feature2, featurename):
    if feature2 is None:
        operator = featurename.split("(")[0]
        feature, _ = create_unary_feature_and_featurename(feature1, operator)
    else:
        operator = featurename.split("(")[0]
        feature, _ = create_binary_feature_and_featurename(feature1, feature2, operator)
    return feature


def create_unary_feature_and_featurename(feature1, operator):
    try:
        feature1_float_list = [float(x) for x in feature1]
    except (ValueError, TypeError):
        feature1_factorized = pd.factorize(feature1)[0]
        feature1_float_list = [float(x) for x in feature1_factorized]
        feature1 = pd.Series(feature1_float_list)
    if operator == "min":
        feature = feature1.apply(lambda x: min(feature1_float_list)).to_list()
        featurename = "min(" + str(feature1.name) + ")"
    elif operator == "max":
        feature = feature1.apply(lambda x: max(feature1_float_list)).to_list()
        featurename = "max(" + str(feature1.name) + ")"
    elif operator == "freq":
        feature = feature1.apply(lambda x: feature1_float_list.count(float(x))).to_list()
        featurename = "freq(" + str(feature1.name) + ")"
    elif operator == "abs":
        feature = feature1.apply(lambda x: abs(float(x))).to_list()
        featurename = "abs(" + str(feature1.name) + ")"
    elif operator == "log":
        feature = feature1.apply(lambda x: np.log(np.abs(float(x)))).to_list()
        featurename = "log(" + str(feature1.name) + ")"
    elif operator == "sqrt":
        feature = feature1.apply(lambda x: np.sqrt(np.abs(float(x)))).to_list()
        featurename = "sqrt(" + str(feature1.name) + ")"
    elif operator == "square":
        feature = feature1.apply(lambda x: np.square(float(x))).to_list()
        featurename = "square(" + str(feature1.name) + ")"
    elif operator == "sigmoid":
        feature = feature1.apply(lambda x: 1 / (1 + np.exp(-float(x)))).to_list()
        featurename = "sigmoid(" + str(feature1.name) + ")"
    elif operator == "round":
        feature = feature1.apply(lambda x: np.floor(float(x))).to_list()
        featurename = "round(" + str(feature1.name) + ")"
    elif operator == "residual":
        feature = feature1.apply(lambda x: float(x) - np.floor(float(x))).to_list()
        featurename = "residual(" + str(feature1.name) + ")"
    elif operator == "delete":
        feature = None
        featurename = str(feature1.name)
    else:
        raise NotImplementedError(f"Unrecognized operator {operator}.")
    return feature, featurename


def create_binary_feature_and_featurename(feature1, feature2, operator):
    try:
        feature1_float_list = [float(x) for x in feature1]
        feature2_float_list = [float(x) for x in feature2]
    except (ValueError, TypeError):
        feature1_factorized = pd.factorize(feature1)[0]
        feature2_factorized = pd.factorize(feature2)[0]
        feature1_float_list = [float(x) for x in feature1_factorized]
        feature2_float_list = [float(x) for x in feature2_factorized]
    if operator == "+" or operator == "add":
        feature = [f1 + f2 for f1, f2 in zip(feature1_float_list, feature2_float_list)]
        featurename = "add(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "-" or operator == "subtract":
        feature = [f1 - f2 for f1, f2 in zip(feature1_float_list, feature2_float_list)]
        featurename = "subtract(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "*" or operator == "multiply":
        feature = [f1 * f2 for f1, f2 in zip(feature1_float_list, feature2_float_list)]
        featurename = "multiply(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "/" or operator == "divide":
        feature = [f1 / f2 if f2 != 0 else f1 for f1, f2 in zip(feature1_float_list, feature2_float_list)]
        featurename = "divide(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "GroupByThenMin":
        temp = feature1.groupby(feature2).min()
        temp.loc[np.nan] = np.nan
        feature = feature2.apply(lambda x: temp.loc[x]).to_list()
        featurename = "GroupByThenMin(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "GroupByThenMax":
        temp = feature1.groupby(feature2).max()
        temp.loc[np.nan] = np.nan
        feature = feature2.apply(lambda x: temp.loc[x]).to_list()
        featurename = "GroupByThenMax(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "GroupByThenMean":
        feature1 = pd.to_numeric(feature1, errors='coerce')
        temp = feature1.groupby(feature2)
        temp = temp.mean()
        temp.loc[np.nan] = np.nan
        feature = feature2.apply(lambda x: temp.loc[x]).to_list()
        featurename = "GroupByThenMean(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "GroupByThenMedian":
        feature1 = pd.to_numeric(feature1, errors='coerce')
        temp = feature1.groupby(feature2).median()
        temp.loc[np.nan] = np.nan
        feature = feature2.apply(lambda x: temp.loc[x]).to_list()
        featurename = "GroupByThenMedian(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "GroupByThenStd":
        feature1 = pd.to_numeric(feature1, errors='coerce')
        temp = feature1.groupby(feature2).std()
        temp.loc[np.nan] = np.nan
        feature = feature2.apply(lambda x: temp.loc[x]).to_list()
        featurename = "GroupByThenStd(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == 'GroupByThenRank':
        feature1 = pd.to_numeric(feature1, errors='coerce')
        feature = feature1.groupby(feature2).rank(ascending=True, pct=True).to_list()
        featurename = "GroupByThenRank(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "GroupByThenFreq":
        def _f(x):
            value_counts = x.value_counts()
            value_counts.loc[np.nan] = np.nan
            return x.apply(lambda x: value_counts.loc[x])

        feature = feature1.groupby(feature2).apply(_f).to_list()
        featurename = "GroupByThenFreq(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "GroupByThenNUnique":
        nunique = feature1.groupby(feature2).nunique()
        nunique.loc[np.nan] = np.nan
        feature = feature2.apply(lambda x: nunique.loc[x]).to_list()
        featurename = "GroupByThenNUnique(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "Combine":
        temp = feature1.astype(str) + '_' + feature2.astype(str)
        temp[feature1.isna() | feature2.isna()] = np.nan
        temp, _ = temp.factorize()
        feature = pd.Series(temp, index=feature1.index).astype("float64").to_list()
        featurename = "Combine(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    elif operator == "CombineThenFreq":
        temp = feature1.astype(str) + '_' + feature2.astype(str)
        temp[feature1.isna() | feature2.isna()] = np.nan
        value_counts = temp.value_counts()
        value_counts.loc[np.nan] = np.nan
        feature = temp.apply(lambda x: value_counts.loc[x]).to_list()
        featurename = "CombineThenFreq(" + str(feature1.name) + ", " + str(feature2.name) + ")"
    else:
        raise NotImplementedError(f"Unrecognized operator {operator}.")
    return feature, featurename


def create_featurenames(feature_list):
    unary_operators, binary_operators = get_operators()
    featurenames = []
    for feature1 in feature_list:
        for feature2 in feature_list:
            for operator in binary_operators:
                featurenames.append(operator + "(" + str(feature1) + ", " + str(feature2) + ")")
    for feature in feature_list:
        for operator in unary_operators:
            featurenames.append(operator + "(" + str(feature) + ")")
    return featurenames


def extract_operation_and_original_features_old(s):
    match = re.match(r"([^\s(]+)\s*\(([^)]+)\)", s)  # Capture operation and features
    if match:
        operation = match.group(1)  # The operation (before brackets)
        features = match.group(2).split(", ")  # The features inside brackets
        return operation, features
    match = re.match(r"(without - \s*)", s)
    if match:
        operation = "delete"  # The operation (before brackets)
        features = s.split(" - ")[1]  # The features inside brackets
        return operation, [features]
    return None, []


def extract_operation_and_original_features(s):
    def split_arguments(s):
        args = []
        depth = 0
        current = ''
        for char in s:
            if char == ',' and depth == 0:
                args.append(current.strip())
                current = ''
            else:
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                current += char
        if current:
            args.append(current.strip())
        return args

    # Handle 'without - xxx'
    if s.startswith("without - "):
        operation = "delete"
        features = [s.split(" - ")[1].strip()]
        return operation, features

    # Match top-level operation and inside of the parentheses
    match = re.match(r"([^\s(]+)\s*\((.*)\)", s)
    if match:
        operation = match.group(1)
        inside = match.group(2)
        features = split_arguments(inside)
        return operation, features

    return None, []


def create_featurenames_and_operator(feature_list):
    unary_operators, binary_operators = get_operators()
    operators = unary_operators + binary_operators
    featurenames = []
    for feature in feature_list:
        for operator in unary_operators:
            featurenames.append(operator + "(" + str(feature) + ")")
    for feature1 in feature_list:
        for operator in binary_operators:
            for feature2 in feature_list:
                featurenames.append(operator + "(" + str(feature1) + ", " + str(feature2) + ")")
    return featurenames, operators


def create_featurename(feature1, feature2, operator):
    if operator == "+":
        operator = "add"
    if operator == "-":
        operator = "subtract"
    if operator == "*":
        operator = "multiply"
    if operator == "/":
        operator = "divide"
    if feature2 is None:
        featurename = operator + "(" + str(feature1) + ")"
    else:
        featurename = operator + "(" + str(feature1) + ", " + str(feature2) + ")"
    return featurename
