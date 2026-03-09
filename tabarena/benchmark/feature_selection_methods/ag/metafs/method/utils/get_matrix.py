import pandas as pd
import numpy as np
from pymfe.mfe import MFE

from .create_feature_and_featurename import create_featurenames


def get_matrix_columns():
    return ['dataset - id', 'dataset - task type', 'dataset - number of classes', 'feature - name', 'operator', 'feature - type', 'feature - count', 'feature - mean', 'feature - std', 'feature - min', 'feature - max', 'feature - lower percentile', 'feature - 50 percentile', 'feature - upper percentile', 'feature - unique', 'feature - top', 'feature - freq', 'model', 'improvement']

def get_matrix_core_columns():
    return ['dataset - id', 'feature - name', 'operator', 'model', 'improvement']


def get_additional_pandas_columns():
    return ['task_type', 'feature - count', 'feature - unique', 'feature - top', 'feature - freq', 'feature - mean', 'feature - std', 'feature - min', 'feature - 25%', 'feature - 50%', 'feature - 75%', 'feature - max']


def get_additional_mfe_columns():
    return [f"col_{i}" for i in range(141)]  # ['attr_to_inst', 'nr_inst', 'sparsity.mean']


def get_additional_mfe_columns_group(group):
    X_dummy = np.array([[0, 1], [1, 0]])
    y_dummy = np.array([0, 1])
    # mfe = MFE(groups=["general", "statistical", "info-theory", "model-based", "landmarking"])
    mfe = MFE(groups=group)
    mfe.fit(X_dummy, y_dummy)
    metafeatures = mfe.extract()
    columns = mfe.extract_metafeature_names()
    return columns


def get_additional_d2v_columns():
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]


def get_additional_tabpfn_columns():
    return ['embedding_norm_representation', 'embedding_pca_representation', 'embedding_mean_representation']


def add_new_featurenames(X_test):
    # Get new dataset with feature names and metafeatures and replicate each feature (=each row) x times, that we can repeat thus row with similar values, but instead of the feature name, we add a new name consisting of all available operators and respective features
    matrix_columns = get_matrix_columns()
    featurenames = create_featurenames(X_test["feature - name"].values)
    X_test_new = pd.DataFrame({
        matrix_columns[0]: np.repeat((data := X_test[matrix_columns[0]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[1]: np.repeat((data := X_test[matrix_columns[1]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[2]: np.repeat((data := X_test[matrix_columns[2]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[3]: np.repeat(np.array(featurenames), data.shape[0]).reshape(len(featurenames), data.shape[0]).T.flatten(),
        matrix_columns[4]: np.repeat((data := X_test[matrix_columns[4]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[5]: np.repeat((data := X_test[matrix_columns[5]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[6]: np.repeat((data := X_test[matrix_columns[6]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[7]: np.repeat((data := X_test[matrix_columns[7]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[8]: np.repeat((data := X_test[matrix_columns[8]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[9]: np.repeat((data := X_test[matrix_columns[9]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[10]: np.repeat((data := X_test[matrix_columns[10]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[11]: np.repeat((data := X_test[matrix_columns[11]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[12]: np.repeat((data := X_test[matrix_columns[12]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[13]: np.repeat((data := X_test[matrix_columns[13]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[14]: np.repeat((data := X_test[matrix_columns[14]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[15]: np.repeat((data := X_test[matrix_columns[15]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[16]: np.repeat((data := X_test[matrix_columns[16]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
        matrix_columns[17]: np.repeat((data := X_test[matrix_columns[17]].to_numpy()), len(featurenames)).reshape(-1, len(featurenames)).flatten(),
    })
    return X_test_new
