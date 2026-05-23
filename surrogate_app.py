from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
import torch

from src.simulation import ChannelParams, run_simulation
from src.torch_curve_models import CurveCapacityMLP


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="Gaussian Channel Surrogate Comparison",
    layout="wide",
)

st.title("Gaussian Channel Surrogate Comparison")

st.markdown(
    r"""
This app compares the full numerical simulation with trained curve-level surrogate models.

The trainable input parameters are:

\[
\gamma_A,\quad \gamma_B,\quad \omega_A,\quad \omega_B.
\]

All other physical and numerical parameters are fixed to the values used during
training.
"""
)


# ============================================================
# FIXED PARAMETERS USED DURING TRAINING
# ============================================================

SIGMA = 0.01
D = 1.0
M_A = 1.0
M_B = 1.0
E = 100.0

T_MAX = 40.0
N_TIME_POINTS = 200

SIMULATION_KWARGS = {
    "t_max": T_MAX,
    "dt": 0.0005,
    "n_noise_times": N_TIME_POINTS,
    "n_integral_points": 400,
    "rtol": 1e-8,
    "atol": 1e-10,
    "h_tolerance": 1e-12,
    "h_margin_scale": 1e-6,
    "determinant_quality_scale": 1e-14,
    "outlier_window": 5,
    "outlier_threshold": 4.0,
    "outlier_alpha": 0.5,
    "bounce_window": 1,
    "tau_jump_filter": True,
    "tau_jump_local_window": 5,
    "tau_jump_local_factor": 100.0,
}


# ============================================================
# MODEL PATHS
# ============================================================

TORCH_MODEL_PATH = Path("models") / "torch_curve_capacity_mlp.pt"
TORCH_SCALER_PATH = Path("models") / "torch_curve_feature_scaler.joblib"

PCA_ML_MODEL_PATH = Path("models") / "pca_ml_curve_model.joblib"
PCA_ML_SCALER_PATH = Path("models") / "pca_ml_feature_scaler.joblib"
PCA_ML_BASIS_PATH = Path("models") / "pca_ml_basis.joblib"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Input parameters")

gamma_A = st.sidebar.number_input(
    r"$\gamma_A$",
    value=0.010,
    min_value=0.001,
    max_value=0.035,
    step=0.001,
    format="%.6f",
)

gamma_B = st.sidebar.number_input(
    r"$\gamma_B$",
    value=0.010,
    min_value=0.001,
    max_value=0.035,
    step=0.001,
    format="%.6f",
)

omega_A = st.sidebar.number_input(
    r"$\omega_A$",
    value=1.0,
    min_value=0.1,
    max_value=1.5,
    step=0.05,
    format="%.6f",
)

omega_B = st.sidebar.number_input(
    r"$\omega_B$",
    value=1.0,
    min_value=0.1,
    max_value=1.5,
    step=0.05,
    format="%.6f",
)

st.sidebar.header("What to compute")

run_numerical = st.sidebar.checkbox(
    "Full numerical simulation",
    value=False,
    help="If disabled, the expensive numerical simulation is not run.",
)

run_pca_ml = st.sidebar.checkbox(
    "PCA-ML surrogate",
    value=False,
    help="Optional scikit-learn curve surrogate. Works only if the PCA-ML model file is available.",
)

run_torch_curve = st.sidebar.checkbox(
    "PyTorch curve surrogate",
    value=True,
)

show_error_plots = st.sidebar.checkbox(
    "Show errors against numerical simulation",
    value=True,
)

show_prediction_table = st.sidebar.checkbox(
    "Show prediction table",
    value=False,
)

run_button = st.sidebar.button("Run comparison")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_physical_features_from_inputs(
    gamma_A,
    gamma_B,
    omega_A,
    omega_B,
    sigma=SIGMA,
    d=D,
):
    """
    Builds the same physics-informed features used during curve-model training.
    """

    gamma_mean = 0.5 * (gamma_A + gamma_B)
    gamma_delta = gamma_B - gamma_A

    omega_mean = 0.5 * (omega_A + omega_B)
    omega_delta = omega_B - omega_A

    Sigma2_A = np.sqrt(8.0 / np.pi) * gamma_A / sigma - omega_A**2
    Sigma2_B = np.sqrt(8.0 / np.pi) * gamma_B / sigma - omega_B**2

    Sigma2_mean = 0.5 * (Sigma2_A + Sigma2_B)
    Sigma2_delta = Sigma2_B - Sigma2_A

    coupling_delay = 2.0 * np.sqrt(gamma_A * gamma_B) / d

    abs_gamma_delta = abs(gamma_delta)
    abs_omega_delta = abs(omega_delta)

    gamma_delta_ratio = gamma_delta / (gamma_mean + 1e-12)
    omega_delta_ratio = omega_delta / (omega_mean + 1e-12)

    X = np.array(
        [[
            gamma_A,
            gamma_B,
            gamma_mean,
            gamma_delta,
            abs_gamma_delta,
            gamma_delta_ratio,

            omega_A,
            omega_B,
            omega_mean,
            omega_delta,
            abs_omega_delta,
            omega_delta_ratio,

            Sigma2_A,
            Sigma2_B,
            Sigma2_mean,
            Sigma2_delta,

            coupling_delay,
        ]],
        dtype=float,
    )

    derived = {
        "gamma_mean": gamma_mean,
        "gamma_delta": gamma_delta,
        "omega_mean": omega_mean,
        "omega_delta": omega_delta,
        "Sigma2_A": Sigma2_A,
        "Sigma2_B": Sigma2_B,
        "Sigma2_mean": Sigma2_mean,
        "Sigma2_delta": Sigma2_delta,
        "coupling_delay": coupling_delay,
    }

    return X, derived


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
@st.cache_resource
def load_torch_curve_surrogate():
    if not TORCH_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing PyTorch model: {TORCH_MODEL_PATH}")

    if not TORCH_SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing PyTorch scaler: {TORCH_SCALER_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        TORCH_MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    model_config = checkpoint["model_config"]

    causal_mask = checkpoint.get("causal_mask", None)

    model = CurveCapacityMLP(
        input_dim=model_config["input_dim"],
        output_dim=model_config["output_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_hidden_layers=model_config["num_hidden_layers"],
        dropout=model_config.get("dropout", 0.0),
        use_softplus=model_config.get("use_softplus", True),
        causal_mask=causal_mask,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    scaler = joblib.load(TORCH_SCALER_PATH)

    t_grid = np.asarray(checkpoint["t_grid"], dtype=float)

    return model, scaler, t_grid, device, checkpoint

@st.cache_resource
def load_pca_ml_surrogate():
    if not PCA_ML_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing PCA-ML model: {PCA_ML_MODEL_PATH}. "
            "This is expected if the oversized PCA-ML binary is not tracked."
        )

    if not PCA_ML_SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing PCA-ML scaler: {PCA_ML_SCALER_PATH}")

    if not PCA_ML_BASIS_PATH.exists():
        raise FileNotFoundError(f"Missing PCA basis: {PCA_ML_BASIS_PATH}")

    model = joblib.load(PCA_ML_MODEL_PATH)
    scaler = joblib.load(PCA_ML_SCALER_PATH)
    pca = joblib.load(PCA_ML_BASIS_PATH)

    # The PCA-ML model uses the same fixed grid as training.
    t_grid = np.linspace(0.0, T_MAX, N_TIME_POINTS)

    return model, scaler, pca, t_grid


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_torch_curve(
    gamma_A,
    gamma_B,
    omega_A,
    omega_B,
):
    model, scaler, t_grid, device, checkpoint = load_torch_curve_surrogate()

    X, _ = build_physical_features_from_inputs(
        gamma_A=gamma_A,
        gamma_B=gamma_B,
        omega_A=omega_A,
        omega_B=omega_B,
    )

    X_scaled = scaler.transform(X)

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()[0]

    y_pred = np.maximum(y_pred, 0.0)
    y_pred[t_grid < D] = 0.0

    return t_grid, y_pred


def predict_pca_ml_curve(
    gamma_A,
    gamma_B,
    omega_A,
    omega_B,
):
    model, scaler, pca, t_grid = load_pca_ml_surrogate()

    X, _ = build_physical_features_from_inputs(
        gamma_A=gamma_A,
        gamma_B=gamma_B,
        omega_A=omega_A,
        omega_B=omega_B,
    )

    X_scaled = scaler.transform(X)

    coeffs = model.predict(X_scaled)
    y_pred = pca.inverse_transform(coeffs)[0]

    y_pred = np.maximum(y_pred, 0.0)
    y_pred[t_grid < D] = 0.0

    return t_grid, y_pred


def run_numerical_simulation(
    gamma_A,
    gamma_B,
    omega_A,
    omega_B,
):
    params = ChannelParams(
        gamma_A=gamma_A,
        gamma_B=gamma_B,
        omega_A=omega_A,
        omega_B=omega_B,
        sigma=SIGMA,
        d=D,
        m_A=M_A,
        m_B=M_B,
        E=E,
    )

    results = run_simulation(
        params=params,
        **SIMULATION_KWARGS,
    )

    return results["noise_times"], results["C_values"], results


# ============================================================
# PLOTTING / SUMMARY
# ============================================================

def summarize_curve(name, t_values, C_values):
    C_values = np.asarray(C_values, dtype=float)
    valid = np.isfinite(C_values)

    if not np.any(valid):
        return {
            "model": name,
            "max_C": np.nan,
            "t_max_C": np.nan,
            "mean_C": np.nan,
            "valid_fraction": 0.0,
        }

    valid_indices = np.where(valid)[0]
    local_idx = np.nanargmax(C_values[valid])
    max_idx = valid_indices[local_idx]

    return {
        "model": name,
        "max_C": float(C_values[max_idx]),
        "t_max_C": float(t_values[max_idx]),
        "mean_C": float(np.nanmean(C_values)),
        "valid_fraction": float(np.mean(valid)),
    }


def plot_capacity_comparison(predictions):
    fig, ax = plt.subplots(figsize=(9, 5))

    for name, payload in predictions.items():
        t_values = payload["t"]
        C_values = payload["C"]

        style = payload.get("style", "-")
        linewidth = payload.get("linewidth", 1.8)

        ax.plot(
            t_values,
            C_values,
            style,
            linewidth=linewidth,
            label=name,
        )

    ax.axvline(
        D,
        linestyle="--",
        color="gray",
        alpha=0.6,
        label=r"$t=d$",
    )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$C_E(t)$")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)


def plot_error_against_numerical(predictions):
    if "Numerical simulation" not in predictions:
        return

    t_num = predictions["Numerical simulation"]["t"]
    C_num = predictions["Numerical simulation"]["C"]

    fig, ax = plt.subplots(figsize=(9, 5))

    for name, payload in predictions.items():
        if name == "Numerical simulation":
            continue

        t_values = payload["t"]
        C_values = payload["C"]

        if len(t_values) != len(t_num) or not np.allclose(t_values, t_num):
            C_interp = np.interp(t_num, t_values, C_values)
        else:
            C_interp = C_values

        ax.plot(
            t_num,
            C_interp - C_num,
            label=f"{name} - numerical",
        )

    ax.axhline(0.0, linestyle="--", color="gray")
    ax.axvline(D, linestyle="--", color="gray", alpha=0.6)

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\Delta C_E(t)$")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)


def validate_domain(gamma_A, gamma_B, omega_A, omega_B):
    messages = []

    if not (0.001 <= gamma_A <= 0.035):
        messages.append(r"$\gamma_A$ is outside the training range $[0.001,0.035]$.")

    if not (0.001 <= gamma_B <= 0.035):
        messages.append(r"$\gamma_B$ is outside the training range $[0.001,0.035]$.")

    if not (0.1 <= omega_A <= 1.5):
        messages.append(r"$\omega_A$ is outside the training range $[0.1,1.5]$.")

    if not (0.1 <= omega_B <= 1.5):
        messages.append(r"$\omega_B$ is outside the training range $[0.1,1.5]$.")

    if abs(gamma_A - gamma_B) > 0.005:
        messages.append(r"$|\gamma_A-\gamma_B|>0.005$, outside the constrained training domain.")

    if abs(omega_A - omega_B) > 0.3:
        messages.append(r"$|\omega_A-\omega_B|>0.3$, outside the constrained training domain.")

    return messages


# ============================================================
# MAIN
# ============================================================

if run_button:
    predictions = {}
    numerical_results = None

    domain_messages = validate_domain(
        gamma_A=gamma_A,
        gamma_B=gamma_B,
        omega_A=omega_A,
        omega_B=omega_B,
    )

    if domain_messages:
        st.warning(
            "Some inputs are outside the training domain. Surrogate predictions may be unreliable."
        )
        for message in domain_messages:
            st.markdown(f"- {message}")

    X_features, derived = build_physical_features_from_inputs(
        gamma_A=gamma_A,
        gamma_B=gamma_B,
        omega_A=omega_A,
        omega_B=omega_B,
    )

    st.header("Input parameters")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(r"$\gamma_A$", f"{gamma_A:.6g}")
    col2.metric(r"$\gamma_B$", f"{gamma_B:.6g}")
    col3.metric(r"$\omega_A$", f"{omega_A:.6g}")
    col4.metric(r"$\omega_B$", f"{omega_B:.6g}")

    st.subheader("Derived physical features")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(r"$\Sigma_A^2$", f"{derived['Sigma2_A']:.6g}")
    col2.metric(r"$\Sigma_B^2$", f"{derived['Sigma2_B']:.6g}")
    col3.metric(r"$\Delta\gamma=\gamma_B-\gamma_A$", f"{derived['gamma_delta']:.6g}")
    col4.metric(r"$\Delta\omega=\omega_B-\omega_A$", f"{derived['omega_delta']:.6g}")

    # ------------------------------------------------------------
    # PyTorch curve surrogate
    # ------------------------------------------------------------

    if run_torch_curve:
        try:
            with st.spinner("Running PyTorch curve surrogate..."):
                t_torch, C_torch = predict_torch_curve(
                    gamma_A=gamma_A,
                    gamma_B=gamma_B,
                    omega_A=omega_A,
                    omega_B=omega_B,
                )

            predictions["PyTorch curve surrogate"] = {
                "t": t_torch,
                "C": C_torch,
                "style": ":",
                "linewidth": 2.2,
            }

        except Exception as exc:
            st.error(f"Could not run PyTorch curve surrogate: {exc}")

    # ------------------------------------------------------------
    # PCA-ML surrogate
    # ------------------------------------------------------------

    if run_pca_ml:
        try:
            with st.spinner("Running PCA-ML surrogate..."):
                t_ml, C_ml = predict_pca_ml_curve(
                    gamma_A=gamma_A,
                    gamma_B=gamma_B,
                    omega_A=omega_A,
                    omega_B=omega_B,
                )

            predictions["PCA-ML surrogate"] = {
                "t": t_ml,
                "C": C_ml,
                "style": "--",
                "linewidth": 2.0,
            }

        except Exception as exc:
            st.warning(f"Could not run PCA-ML surrogate: {exc}")

    # ------------------------------------------------------------
    # Numerical simulation
    # ------------------------------------------------------------

    if run_numerical:
        try:
            with st.spinner("Running full numerical simulation..."):
                t_num, C_num, numerical_results = run_numerical_simulation(
                    gamma_A=gamma_A,
                    gamma_B=gamma_B,
                    omega_A=omega_A,
                    omega_B=omega_B,
                )

            predictions["Numerical simulation"] = {
                "t": t_num,
                "C": C_num,
                "style": "-",
                "linewidth": 2.4,
            }

        except Exception as exc:
            st.error(f"Could not run numerical simulation: {exc}")

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------

    if predictions:
        st.header("Capacity comparison")

        plot_capacity_comparison(predictions)

        summaries = [
            summarize_curve(
                name=name,
                t_values=payload["t"],
                C_values=payload["C"],
            )
            for name, payload in predictions.items()
        ]

        st.subheader("Curve summaries")
        st.dataframe(pd.DataFrame(summaries), use_container_width=True)

        if show_error_plots and "Numerical simulation" in predictions and len(predictions) > 1:
            st.header("Prediction error against numerical simulation")
            plot_error_against_numerical(predictions)

        if numerical_results is not None:
            st.header("Numerical reliability")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Valid capacity fraction",
                f"{numerical_results['valid_fraction']:.2%}",
            )
            col2.metric(
                "Mean reliability",
                f"{numerical_results['mean_reliability']:.3f}",
            )
            col3.metric(
                r"$\tau$ discontinuity detected",
                str(numerical_results["tau_jump_detected"]),
            )

            if numerical_results["tau_jump_detected"]:
                st.warning(
                    f"Numerical tau discontinuity detected at "
                    f"t = {numerical_results['tau_jump_time']:.6g}. "
                    "Later numerical capacity points are marked invalid."
                )

        if show_prediction_table:
            st.header("Prediction table")

            table = {"t": next(iter(predictions.values()))["t"]}

            base_t = table["t"]

            for name, payload in predictions.items():
                t_values = payload["t"]
                C_values = payload["C"]

                if len(t_values) != len(base_t) or not np.allclose(t_values, base_t):
                    table[name] = np.interp(base_t, t_values, C_values)
                else:
                    table[name] = C_values

            st.dataframe(pd.DataFrame(table), use_container_width=True)

    else:
        st.warning("No model or numerical simulation was selected.")

else:
    st.info("Select what to compute, choose the input parameters, and press **Run comparison**.")