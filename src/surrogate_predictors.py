from pathlib import Path

import numpy as np
import joblib
import torch

from src.features import inverse_target_transform
from src.torch_models import CapacityMLP


# ============================================================
# FEATURE BUILDING
# ============================================================

def build_prediction_dataframe_like_features(
    gamma_A,
    gamma_B,
    omega_A,
    omega_B,
    t_values,
):
    """
    Builds feature matrix for surrogate prediction.

    Features:
        gamma_mean, gamma_delta, omega_mean, omega_delta, t
    """

    t_values = np.asarray(t_values, dtype=float)

    gamma_mean = 0.5 * (gamma_A + gamma_B)
    gamma_delta = gamma_B - gamma_A

    omega_mean = 0.5 * (omega_A + omega_B)
    omega_delta = omega_B - omega_A

    X = np.column_stack([
        np.full_like(t_values, gamma_mean, dtype=float),
        np.full_like(t_values, gamma_delta, dtype=float),
        np.full_like(t_values, omega_mean, dtype=float),
        np.full_like(t_values, omega_delta, dtype=float),
        t_values,
    ])

    return X


# ============================================================
# SCIKIT-LEARN MODEL
# ============================================================

def load_ml_surrogate(
    model_path=Path("models/hist_gradient_boosting_CE.joblib"),
    scaler_path=Path("models/hist_gradient_boosting_scaler.joblib"),
):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler


def predict_ml_capacity(
    gamma_A,
    gamma_B,
    omega_A,
    omega_B,
    t_values,
    model,
    scaler,
    use_log_target=False,
):
    X = build_prediction_dataframe_like_features(
        gamma_A=gamma_A,
        gamma_B=gamma_B,
        omega_A=omega_A,
        omega_B=omega_B,
        t_values=t_values,
    )

    X_scaled = scaler.transform(X)

    y_pred = model.predict(X_scaled)
    y_pred = inverse_target_transform(y_pred, use_log_target=use_log_target)

    return y_pred


# ============================================================
# PYTORCH MODEL
# ============================================================

def load_torch_surrogate(
    model_path=Path("models/torch_capacity_mlp.pt"),
    scaler_path=Path("models/torch_capacity_scaler.joblib"),
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    model_config = checkpoint["model_config"]

    model = CapacityMLP(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_hidden_layers=model_config["num_hidden_layers"],
        num_frequencies=model_config["num_frequencies"],
        dropout=model_config.get("dropout", 0.0),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    scaler = joblib.load(scaler_path)

    use_log_target = checkpoint.get("use_log_target", False)

    return model, scaler, use_log_target, device


def predict_torch_capacity(
    gamma_A,
    gamma_B,
    omega_A,
    omega_B,
    t_values,
    model,
    scaler,
    device,
    use_log_target=False,
):
    X = build_prediction_dataframe_like_features(
        gamma_A=gamma_A,
        gamma_B=gamma_B,
        omega_A=omega_A,
        omega_B=omega_B,
        t_values=t_values,
    )

    X_scaled = scaler.transform(X)

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()

    y_pred = inverse_target_transform(y_pred, use_log_target=use_log_target)

    return y_pred