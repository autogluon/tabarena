import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def factorize_dataset(X_train, y_train, X_test, y_test):
    lbl = preprocessing.LabelEncoder()
    for column in X_train.columns:  # select_dtypes(include=['object', 'category'])
        X_train[column] = lbl.fit_transform(X_train[column].astype(int))
    for column in X_test.columns:
        X_test[column] = lbl.fit_transform(X_test[column].astype(int))
    y_train_array, _ = pd.Series.factorize(y_train, use_na_sentinel=False)
    y_train = y_train.replace(y_train_array)
    y_test_array, _ = pd.Series.factorize(y_test, use_na_sentinel=False)
    y_test = y_test.replace(y_test_array)
    return X_train, y_train, X_test, y_test


def factorize_transformed_dataset(X_train, y_train, X_test, y_test):
    lbl = preprocessing.LabelEncoder()
    X_train = X_train.replace([np.inf, -np.inf], np.nan)  # Convert inf values to NaN
    X_train = X_train.dropna(axis=1, how="any")
    for column in X_train.columns:  # select_dtypes(include=['object', 'category'])
        X_train[column] = lbl.fit_transform(X_train[column].astype(int))
    X_test = X_test.replace([np.inf, -np.inf], np.nan)  # Convert inf values to NaN
    X_test = X_test.dropna(axis=1, how="any")
    for column in X_test.columns:
        X_test[column] = lbl.fit_transform(X_test[column].astype(int))
    y_train_array, _ = pd.Series.factorize(y_train, use_na_sentinel=False)
    y_train = y_train.replace(y_train_array)
    y_test_array, _ = pd.Series.factorize(y_test, use_na_sentinel=False)
    y_test = y_test.replace(y_test_array)
    return X_train, y_train, X_test, y_test


def factorize_data(X_train):
    lbl = preprocessing.LabelEncoder()
    for column in X_train.columns:
        X_train[column] = pd.factorize(X_train[column])
    return X_train


def factorize_data_split(X_train, y_train, X_test, y_test):
    lbl = preprocessing.LabelEncoder()
    for column in X_train.columns:  #select_dtypes(include=['object', 'category'])
        # X_train[column], _ = pd.factorize(X_train[column])
        X_train[column] = lbl.fit_transform(X_train[column].astype(int))
    for column in X_test.columns:
        X_test[column] = lbl.fit_transform(X_test[column].astype(int))
    y_train_array, _ = pd.Series.factorize(y_train, use_na_sentinel=False)
    y_train = y_train.replace(y_train_array)
    y_test_array, _ = pd.Series.factorize(y_test, use_na_sentinel=False)
    y_test = y_test.replace(y_test_array)
    return X_train, y_train, X_test, y_test


def factorize_data_old(X_train, y_train, X_test, y_test):
    # Identify categorical columns
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns

    # Apply LabelEncoder only to categorical columns
    for column in categorical_columns:
        lbl = preprocessing.LabelEncoder()
        X_train[column] = lbl.fit_transform(X_train[column].astype(str))  # Convert to string before encoding
        X_test[column] = lbl.transform(X_test[column].astype(str))  # Apply the same mapping to test data

    # Factorize target labels for consistency
    y_train, label_mapping = pd.factorize(y_train, use_na_sentinel=False)
    y_test = pd.Series(y_test).map(dict(enumerate(label_mapping))).fillna(0).astype(
        int)  # .interpolate(method="pad").astype(int)  # Ensure mapping consistency

    return X_train, y_train, X_test, y_test


def encode_categorical_features(X_train, X_test):
    cat_features = X_train.select_dtypes(include=['object', 'category']).columns
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    if len(cat_features) > 0:
        # Fit encoder on training data
        encoder.fit(X_train[cat_features])
        # Transform both train and test
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
        X_train_encoded[cat_features] = encoder.transform(X_train[cat_features])
        X_test_encoded[cat_features] = encoder.transform(X_test[cat_features])
        return X_train_encoded, X_test_encoded
    else:
        # No categorical features to encode
        return X_train, X_test


def factorize_features(X_train, X_test):
    """Encode categorical features and handle mixed data types"""

    # Identify categorical and numerical columns
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()

    if categorical_columns:
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numerical_columns),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns)
            ],
            remainder='passthrough'
        )

        # Fit on training data and transform both
        X_train_encoded = preprocessor.fit_transform(X_train)
        X_test_encoded = preprocessor.transform(X_test)

        return X_train_encoded, X_test_encoded
    else:
        # No categorical columns, return as-is
        return X_train, X_test


def factorize_target(y_train, y_test):
    """Encode categorical target variables"""

    # Check if target is categorical (object or category dtype)
    if y_train.dtype == 'object' or y_train.dtype.name == 'category':

        # Create and fit LabelEncoder on training data
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        return y_train_encoded, y_test_encoded
    else:
        # Already numerical, return as-is
        return y_train.values, y_test.values
