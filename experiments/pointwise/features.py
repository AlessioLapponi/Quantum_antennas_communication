import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
import joblib


BASE_FEATURES = [
    "gamma_mean",
    "gamma_delta",
    "omega_mean",
    "omega_delta",
    "t",
]


def build_feature_matrix(df, feature_columns=None):
    """
    Builds the input feature matrix.

    Default:
        gamma_mean, gamma_delta, omega_mean, omega_delta, t
    """

    if feature_columns is None:
        feature_columns = BASE_FEATURES

    X = df[feature_columns].to_numpy(dtype=np.float64)

    return X


def build_target_vector(df, target_column="C_E", use_log_target=False):
    """
    Builds the target vector.

    If use_log_target=True:
        y = log1p(C_E)
    """

    y = df[target_column].to_numpy(dtype=np.float64)

    if use_log_target:
        y = np.log1p(np.maximum(y, 0.0))

    return y


def inverse_target_transform(y_pred, use_log_target=False):
    """
    Inverts the optional log1p target transform.
    """

    y_pred = np.asarray(y_pred)

    if use_log_target:
        return np.expm1(y_pred)

    return y_pred


def fit_feature_scaler(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def save_scaler(scaler, path):
    joblib.dump(scaler, path)


def load_scaler(path):
    return joblib.load(path)