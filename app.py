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
    page_title="Quantum Antennas Gaussian Channel",
    layout="wide",
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select page",
    [
        "Numerical simulation & diagnostics",
        "Surrogate model comparison",
    ],
)


# ============================================================
# FIXED SURROGATE TRAINING PARAMETERS
# ============================================================

SURROGATE_SIGMA = 0.01
SURROGATE_D = 1.0
SURROGATE_M_A = 1.0
SURROGATE_M_B = 1.0
SURROGATE_E = 100.0

SURROGATE_T_MAX = 40.0
SURROGATE_N_TIME_POINTS = 200

SURROGATE_SIMULATION_KWARGS = {
    "t_max": SURROGATE_T_MAX,
    "dt": 0.0005,
    "n_noise_times": SURROGATE_N_TIME_POINTS,
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
# GENERAL HELPERS
# ============================================================

def close_and_show(fig):
    st.pyplot(fig)
    plt.close(fig)


def build_physical_features_from_inputs(
    gamma_A,
    gamma_B,
    omega_A,
    omega_B,
    sigma=SURROGATE_SIGMA,
    d=SURROGATE_D,
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


def validate_surrogate_domain(gamma_A, gamma_B, omega_A, omega_B):
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
        messages.append(
            r"$|\gamma_A-\gamma_B|>0.005$, outside the constrained training domain."
        )

    if abs(omega_A - omega_B) > 0.3:
        messages.append(
            r"$|\omega_A-\omega_B|>0.3$, outside the constrained training domain."
        )

    return messages


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


# ============================================================
# MODEL LOADING
# ============================================================

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
        raise FileNotFoundError(f"Missing PCA-ML model: {PCA_ML_MODEL_PATH}")

    if not PCA_ML_SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing PCA-ML scaler: {PCA_ML_SCALER_PATH}")

    if not PCA_ML_BASIS_PATH.exists():
        raise FileNotFoundError(f"Missing PCA basis: {PCA_ML_BASIS_PATH}")

    model = joblib.load(PCA_ML_MODEL_PATH)
    scaler = joblib.load(PCA_ML_SCALER_PATH)
    pca = joblib.load(PCA_ML_BASIS_PATH)

    t_grid = np.linspace(
        0.0,
        SURROGATE_T_MAX,
        SURROGATE_N_TIME_POINTS,
    )

    return model, scaler, pca, t_grid


# ============================================================
# SURROGATE PREDICTIONS
# ============================================================

def predict_torch_curve(gamma_A, gamma_B, omega_A, omega_B):
    model, scaler, t_grid, device, _ = load_torch_curve_surrogate()

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
    y_pred[t_grid < SURROGATE_D] = 0.0

    return t_grid, y_pred


def predict_pca_ml_curve(gamma_A, gamma_B, omega_A, omega_B):
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
    y_pred[t_grid < SURROGATE_D] = 0.0

    return t_grid, y_pred


def run_fixed_numerical_for_surrogate(gamma_A, gamma_B, omega_A, omega_B):
    params = ChannelParams(
        gamma_A=gamma_A,
        gamma_B=gamma_B,
        omega_A=omega_A,
        omega_B=omega_B,
        sigma=SURROGATE_SIGMA,
        d=SURROGATE_D,
        m_A=SURROGATE_M_A,
        m_B=SURROGATE_M_B,
        E=SURROGATE_E,
    )

    results = run_simulation(
        params=params,
        **SURROGATE_SIMULATION_KWARGS,
    )

    return results["noise_times"], results["C_values"], results


# ============================================================
# NUMERICAL PAGE PLOTS
# ============================================================

def show_tau_plot(t, tau, d, tau_jump_time=None, tau_zero_times=None):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(t, tau, label=r"$\tau(t)$")
    ax.axvline(d, linestyle="--", label=r"$t=d$")

    if tau_zero_times is not None and len(tau_zero_times) > 0:
        for i, zero_time in enumerate(tau_zero_times):
            ax.axvline(
                zero_time,
                linestyle=":",
                alpha=0.4,
                label=r"$\tau=0$ crossing" if i == 0 else None,
            )

    if tau_jump_time is not None:
        ax.axvline(
            tau_jump_time,
            linestyle="-.",
            label=r"detected $\tau$ discontinuity",
        )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\tau(t)$")
    ax.grid(True)
    ax.legend()

    close_and_show(fig)


def show_W_plot(noise_times, W_values, d):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(noise_times, W_values, label=r"$W(t)=\det N(t)$")
    ax.axvline(d, linestyle="--", label=r"$t=d$")

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$W(t)$")
    ax.grid(True)
    ax.legend()

    close_and_show(fig)


def show_log_comparison(noise_times, tau_on_noise_times, W_values, d):
    mask = (
        (noise_times >= d)
        & (np.abs(tau_on_noise_times) > 0.0)
        & (W_values > 0.0)
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    if np.any(mask):
        ax.plot(
            noise_times[mask],
            np.log10(np.abs(tau_on_noise_times[mask])),
            label=r"$\log_{10}|\tau(t)|$",
        )
        ax.plot(
            noise_times[mask],
            0.5 * np.log10(W_values[mask]),
            label=r"$\log_{10}\sqrt{W(t)}$",
        )

    ax.axvline(d, linestyle="--", label=r"$t=d$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"log-scale")
    ax.grid(True)
    ax.legend()

    close_and_show(fig)


def show_capacity_plot(noise_times, C_values, d, t_max_C, max_C):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(noise_times, C_values, label=r"$C_E(t)$")
    ax.axvline(d, linestyle="--", label=r"$t=d$")

    if np.isfinite(t_max_C) and np.isfinite(max_C):
        ax.scatter([t_max_C], [max_C], marker="o", label=r"maximum")

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$C_E(t)$")
    ax.grid(True)
    ax.legend()

    close_and_show(fig)


def show_reliability_plot(noise_times, reliability, valid_C, d):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(noise_times, reliability, label=r"reliability score")

    if np.any(~valid_C):
        ax.scatter(
            noise_times[~valid_C],
            np.zeros(np.sum(~valid_C)),
            marker="x",
            label=r"invalid $C_E(t)$ points",
        )

    ax.axvline(d, linestyle="--", label=r"$t=d$")

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"reliability")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
    ax.legend()

    close_and_show(fig)


def show_tau_step_jump_plot(t, tau, d, tau_jump_time=None):
    tau_steps = np.abs(np.diff(tau))
    step_times = t[1:]

    mask = step_times >= d

    fig, ax = plt.subplots(figsize=(8, 4))

    if np.any(mask):
        ax.plot(
            step_times[mask],
            tau_steps[mask],
            label=r"$|\tau_k-\tau_{k-1}|$ on fine grid",
        )

    ax.set_yscale("log")
    ax.axvline(d, linestyle="--", label=r"$t=d$")

    if tau_jump_time is not None:
        ax.axvline(
            tau_jump_time,
            linestyle=":",
            label=r"detected $\tau$ discontinuity",
        )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$|\Delta\tau|$")
    ax.grid(True)
    ax.legend()

    close_and_show(fig)


def show_reliability_components_plot(noise_times, diagnostics, d):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(noise_times, diagnostics["r_domain"], label=r"$r_{\rm domain}$")
    ax.plot(noise_times, diagnostics["r_det"], label=r"$r_{\rm det}$")
    ax.plot(noise_times, diagnostics["r_outlier"], label=r"$r_{\rm outlier}$")

    ax.axvline(d, linestyle="--", label=r"$t=d$")

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"component score")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
    ax.legend()

    close_and_show(fig)


def show_h_margin_plot(noise_times, diagnostics, d):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        noise_times,
        diagnostics["h_margin"],
        label=r"$\min(x_1-1/2,x_2-1/2)$",
    )

    ax.axhline(0.0, linestyle="--", label=r"domain boundary")
    ax.axvline(d, linestyle="--", label=r"$t=d$")

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$h$-domain margin")
    ax.grid(True)
    ax.legend()

    close_and_show(fig)


def show_determinant_quality_plot(noise_times, diagnostics, d):
    det_quality = diagnostics["det_quality"]
    mask = det_quality > 0.0

    fig, ax = plt.subplots(figsize=(8, 4))

    if np.any(mask):
        ax.plot(
            noise_times[mask],
            np.log10(det_quality[mask]),
            label=r"$\log_{10}(q_W)$",
        )

    ax.axvline(d, linestyle="--", label=r"$t=d$")

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\log_{10}$ determinant quality")
    ax.grid(True)
    ax.legend()

    close_and_show(fig)


def show_outlier_score_plot(noise_times, diagnostics, d):
    outlier_score = diagnostics["outlier_score"]
    bounce_mask = diagnostics["bounce_mask"]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(noise_times, outlier_score, label=r"local outlier score")

    if np.any(bounce_mask):
        ax.scatter(
            noise_times[bounce_mask],
            outlier_score[bounce_mask],
            s=20,
            label=r"protected $|\tau|$-bounce region",
        )

    ax.axvline(d, linestyle="--", label=r"$t=d$")

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"outlier score")
    ax.grid(True)
    ax.legend()

    close_and_show(fig)


def show_capacity_with_reliability_mask(
    noise_times,
    C_values,
    valid_C,
    reliability,
    reliability_threshold,
    d,
    tau_jump_time=None,
):
    training_mask = valid_C & (reliability >= reliability_threshold)
    rejected_mask = valid_C & (reliability < reliability_threshold)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(noise_times, C_values, alpha=0.4, label=r"$C_E(t)$")

    if np.any(training_mask):
        ax.scatter(
            noise_times[training_mask],
            C_values[training_mask],
            s=18,
            label=r"accepted for training",
        )

    if np.any(rejected_mask):
        ax.scatter(
            noise_times[rejected_mask],
            C_values[rejected_mask],
            s=18,
            marker="x",
            label=r"rejected / low reliability",
        )

    if np.any(~valid_C):
        ax.scatter(
            noise_times[~valid_C],
            np.zeros(np.sum(~valid_C)),
            s=18,
            marker="x",
            label=r"invalid",
        )

    ax.axvline(d, linestyle="--", label=r"$t=d$")

    if tau_jump_time is not None:
        ax.axvline(
            tau_jump_time,
            linestyle=":",
            label=r"detected $\tau$ discontinuity",
        )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$C_E(t)$")
    ax.grid(True)
    ax.legend()

    close_and_show(fig)


# ============================================================
# SURROGATE PAGE PLOTS
# ============================================================

def plot_capacity_comparison(predictions, d=SURROGATE_D):
    fig, ax = plt.subplots(figsize=(9, 5))

    for name, payload in predictions.items():
        ax.plot(
            payload["t"],
            payload["C"],
            payload.get("style", "-"),
            linewidth=payload.get("linewidth", 1.8),
            label=name,
        )

    ax.axvline(
        d,
        linestyle="--",
        color="gray",
        alpha=0.6,
        label=r"$t=d$",
    )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$C_E(t)$")
    ax.grid(True)
    ax.legend()

    close_and_show(fig)


def plot_error_against_numerical(predictions, d=SURROGATE_D):
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
    ax.axvline(d, linestyle="--", color="gray", alpha=0.6)

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\Delta C_E(t)$")
    ax.grid(True)
    ax.legend()

    close_and_show(fig)


# ============================================================
# PAGE 1: NUMERICAL SIMULATION AND DIAGNOSTICS
# ============================================================

def render_numerical_page():
    st.title("Communicating Quantum Antennas Gaussian Channel Simulator")

    st.markdown(
        r"""
This page runs the full numerical simulation of the interaction
channel between two non-identical harmonic oscillator detectors, acting as bosonic quantum antennas.

It computes:

- transmissivity $\tau(t)$, i.e. the fraction of input signal arriving to the output;
- noise determinant $W(t)$, i.e. the signal achieved by the receiver independent from the input;
- energy-constrained classical capacity $C_E(t)$, i.e. the amount of bits the antennas can reliable communicate with input energy $E$;
- reliability and numerical diagnostics (more info in the README.md).
"""
    )

    st.sidebar.header("Physical parameters")

    gamma_A = st.sidebar.number_input(
        r"$\gamma_A$",
        value=0.010,
        format="%.6f",
        key="num_gamma_A",
    )
    gamma_B = st.sidebar.number_input(
        r"$\gamma_B$",
        value=0.010,
        format="%.6f",
        key="num_gamma_B",
    )

    omega_A = st.sidebar.number_input(
        r"$\omega_A$",
        value=1.0,
        format="%.6f",
        key="num_omega_A",
    )
    omega_B = st.sidebar.number_input(
        r"$\omega_B$",
        value=1.0,
        format="%.6f",
        key="num_omega_B",
    )

    m_A = st.sidebar.number_input(
        r"$m_A$",
        value=1.0,
        format="%.6f",
        key="num_m_A",
    )
    m_B = st.sidebar.number_input(
        r"$m_B$",
        value=1.0,
        format="%.6f",
        key="num_m_B",
    )

    sigma = st.sidebar.number_input(
        r"$\sigma$",
        value=0.01,
        format="%.6f",
        key="num_sigma",
    )
    d = st.sidebar.number_input(
        r"$d$",
        value=1.0,
        format="%.6f",
        key="num_d",
    )

    E = st.sidebar.number_input(
        r"Energy bound $E$",
        value=100.0,
        format="%.6f",
        key="num_E",
    )

    st.sidebar.header("Numerical parameters")

    t_max = st.sidebar.number_input(
        r"$t_{\max}$",
        value=40.0,
        format="%.3f",
        key="num_t_max",
    )
    dt = st.sidebar.number_input(
        r"$dt$",
        value=0.0005,
        format="%.5f",
        key="num_dt",
    )

    n_noise_times = st.sidebar.number_input(
        r"Number of $W(t)$ / $C_E(t)$ time points",
        value=200,
        min_value=10,
        step=10,
        key="num_n_noise_times",
    )

    n_integral_points = st.sidebar.number_input(
        r"Integral points for $W(t)$",
        value=400,
        min_value=50,
        step=50,
        key="num_n_integral_points",
    )

    rtol = st.sidebar.number_input(
        r"$r_{\rm tol}$",
        value=1e-8,
        format="%.1e",
        key="num_rtol",
    )
    atol = st.sidebar.number_input(
        r"$a_{\rm tol}$",
        value=1e-10,
        format="%.1e",
        key="num_atol",
    )

    st.sidebar.header("Reliability filter parameters")

    reliability_threshold = st.sidebar.slider(
        r"Training reliability threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        key="num_reliability_threshold",
    )

    h_tolerance = st.sidebar.number_input(
        r"$h$-domain tolerance",
        value=1e-12,
        format="%.1e",
        key="num_h_tolerance",
    )

    h_margin_scale = st.sidebar.number_input(
        r"$h$-margin scale",
        value=1e-6,
        format="%.1e",
        key="num_h_margin_scale",
        help=(
            "Smaller values make the h-domain margin filter less strict. "
            "Larger values penalize points farther from the boundary."
        ),
    )

    determinant_quality_scale = st.sidebar.number_input(
        r"determinant quality scale",
        value=1e-14,
        format="%.1e",
        key="num_determinant_quality_scale",
        help=(
            "Larger values make the determinant filter stricter. "
            "Smaller values make it more permissive."
        ),
    )

    outlier_window = st.sidebar.number_input(
        r"outlier window",
        value=5,
        min_value=1,
        max_value=50,
        step=1,
        key="num_outlier_window",
    )

    outlier_threshold = st.sidebar.number_input(
        r"outlier threshold",
        value=4.0,
        min_value=0.5,
        max_value=50.0,
        step=0.5,
        key="num_outlier_threshold",
        help=r"Lower values make the local $C_E(t)$ outlier filter stricter.",
    )

    outlier_alpha = st.sidebar.number_input(
        r"outlier penalty $\alpha$",
        value=0.5,
        min_value=0.01,
        max_value=10.0,
        step=0.05,
        key="num_outlier_alpha",
        help=r"Higher values penalize detected outliers more strongly.",
    )

    bounce_window = st.sidebar.number_input(
        r"$|\tau|$-bounce protection window",
        value=1,
        min_value=0,
        max_value=20,
        step=1,
        key="num_bounce_window",
        help=(
            "Number of neighboring C_E points protected around |tau| bounces. "
            "Lower values are stricter."
        ),
    )

    st.sidebar.header(r"$\tau$ discontinuity filter")

    tau_jump_filter = st.sidebar.checkbox(
        r"Enable $\tau$ local-step discontinuity filter",
        value=True,
        key="num_tau_jump_filter",
    )

    tau_jump_local_window = st.sidebar.number_input(
        r"$\tau$ jump local window",
        value=5,
        min_value=2,
        max_value=50,
        step=1,
        key="num_tau_jump_local_window",
        help=(
            "Number of neighboring fine-grid tau steps used on each side. "
            "5 means 5 left and 5 right."
        ),
    )

    tau_jump_local_factor = st.sidebar.number_input(
        r"$\tau$ local jump factor",
        value=100.0,
        min_value=2.0,
        max_value=10000.0,
        step=5.0,
        key="num_tau_jump_local_factor",
        help=(
            "A fine-grid tau step is flagged if it is this many times larger "
            "than its neighboring fine-grid tau steps. Higher = less strict."
        ),
    )

    run_button = st.sidebar.button(
        "Run simulation",
        key="num_run_button",
    )

    if not run_button:
        st.info("Insert parameters in the sidebar and press **Run simulation**.")
        return

    params = ChannelParams(
        gamma_A=gamma_A,
        gamma_B=gamma_B,
        omega_A=omega_A,
        omega_B=omega_B,
        sigma=sigma,
        d=d,
        m_A=m_A,
        m_B=m_B,
        E=E,
    )

    with st.spinner("Running simulation..."):
        results = run_simulation(
            params=params,
            t_max=t_max,
            dt=dt,
            n_noise_times=int(n_noise_times),
            n_integral_points=int(n_integral_points),
            rtol=rtol,
            atol=atol,
            h_tolerance=h_tolerance,
            h_margin_scale=h_margin_scale,
            determinant_quality_scale=determinant_quality_scale,
            outlier_window=int(outlier_window),
            outlier_threshold=outlier_threshold,
            outlier_alpha=outlier_alpha,
            bounce_window=int(bounce_window),
            tau_jump_filter=tau_jump_filter,
            tau_jump_local_window=int(tau_jump_local_window),
            tau_jump_local_factor=tau_jump_local_factor,
        )

    st.success("Simulation completed.")

    st.header("Derived parameters")

    col1, col2, col3 = st.columns(3)

    col1.metric(r"$\Sigma_A^2$", f"{results['Sigma2_A']:.6g}")
    col2.metric(r"$\Sigma_B^2$", f"{results['Sigma2_B']:.6g}")
    col3.metric(r"causal time $d$", f"{d:.6g}")

    st.header("Capacity maximum")

    col1, col2, col3 = st.columns(3)

    col1.metric(r"$\max C_E$", f"{results['max_C']:.6g}")
    col2.metric(r"$t_{\max C_E}$", f"{results['t_max_C']:.6g}")
    col3.metric(r"$t_{\max C_E}-d$", f"{results['delay_max_C']:.6g}")

    st.header("Simulation plots")

    st.subheader(r"Transmissivity $\tau(t)$")
    show_tau_plot(
        results["t"],
        results["tau"],
        d,
        tau_jump_time=results["tau_jump_time"],
        tau_zero_times=results["tau_zero_times"],
    )

    st.subheader(r"Noise determinant $W(t)$")
    show_W_plot(
        results["noise_times"],
        results["W_values"],
        d,
    )

    st.subheader(r"Log comparison: $|\tau(t)|$ versus $\sqrt{W(t)}$")
    show_log_comparison(
        results["noise_times"],
        results["tau_on_noise_times"],
        results["W_values"],
        d,
    )

    st.subheader(r"Energy-constrained classical capacity $C_E(t)$")
    show_capacity_plot(
        results["noise_times"],
        results["C_values"],
        d,
        results["t_max_C"],
        results["max_C"],
    )

    st.header("Reliability diagnostics")

    training_mask = results["valid_C"] & (
        results["reliability"] >= reliability_threshold
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(r"valid $C_E(t)$ fraction", f"{results['valid_fraction']:.2%}")
    col2.metric(r"mean reliability", f"{results['mean_reliability']:.3f}")
    col3.metric(r"invalid $C_E(t)$ points", int(np.sum(~results["valid_C"])))
    col4.metric(
        r"usable training points",
        f"{int(np.sum(training_mask))}/{len(training_mask)}",
    )

    st.subheader("Pointwise reliability score")
    show_reliability_plot(
        results["noise_times"],
        results["reliability"],
        results["valid_C"],
        d,
    )

    st.subheader("Reliability components")
    show_reliability_components_plot(
        results["noise_times"],
        results["reliability_diagnostics"],
        d,
    )

    st.subheader("Capacity with reliability mask")
    show_capacity_with_reliability_mask(
        results["noise_times"],
        results["C_values"],
        results["valid_C"],
        results["reliability"],
        reliability_threshold,
        d,
        tau_jump_time=results["tau_jump_time"],
    )

    st.subheader(r"$\tau$ step-jump diagnostic")
    show_tau_step_jump_plot(
        results["t"],
        results["tau"],
        d,
        tau_jump_time=results["tau_jump_time"],
    )

    if results["tau_jump_detected"]:
        st.warning(
            f"Tau discontinuity detected on the fine tau grid at "
            f"t = {results['tau_jump_time']:.6g}. "
            f"Jump score = {results['tau_jump_score']:.6g}. "
            "All later C_E points are marked invalid."
        )
    else:
        st.success(r"No $\tau$ discontinuity detected on the fine $\tau$ grid.")

    with st.expander("Active reliability filter settings"):
        st.write(
            {
                "h_tolerance": h_tolerance,
                "h_margin_scale": h_margin_scale,
                "determinant_quality_scale": determinant_quality_scale,
                "outlier_window": int(outlier_window),
                "outlier_threshold": outlier_threshold,
                "outlier_alpha": outlier_alpha,
                "bounce_window": int(bounce_window),
                "tau_jump_filter": tau_jump_filter,
                "tau_jump_local_window": int(tau_jump_local_window),
                "tau_jump_local_factor": tau_jump_local_factor,
                "training_reliability_threshold": reliability_threshold,
            }
        )

    with st.expander("Detailed reliability diagnostics"):
        st.subheader(r"$h$-domain margin")
        show_h_margin_plot(
            results["noise_times"],
            results["reliability_diagnostics"],
            d,
        )

        st.subheader("Determinant quality")
        show_determinant_quality_plot(
            results["noise_times"],
            results["reliability_diagnostics"],
            d,
        )

        st.subheader(r"Local outlier score and protected $|\tau|$-bounce regions")
        show_outlier_score_plot(
            results["noise_times"],
            results["reliability_diagnostics"],
            d,
        )

        low_reliability_mask = results["valid_C"] & (
            results["reliability"] < reliability_threshold
        )

        if np.any(low_reliability_mask):
            st.warning("Some valid points have reliability below the threshold.")

            rel_diag = results["reliability_diagnostics"]

            low_rel_table = {
                "t": results["noise_times"][low_reliability_mask],
                "C_E": results["C_values"][low_reliability_mask],
                "tau": results["tau_on_noise_times"][low_reliability_mask],
                "W": results["W_values"][low_reliability_mask],
                "reliability": results["reliability"][low_reliability_mask],
                "h_margin": rel_diag["h_margin"][low_reliability_mask],
                "det_quality": rel_diag["det_quality"][low_reliability_mask],
                "outlier_score": rel_diag["outlier_score"][low_reliability_mask],
                "bounce_protected": rel_diag["bounce_mask"][low_reliability_mask],
            }

            st.dataframe(pd.DataFrame(low_rel_table), use_container_width=True)
        else:
            st.success("No valid points below the reliability threshold.")


# ============================================================
# PAGE 2: SURROGATE MODEL COMPARISON
# ============================================================

def render_surrogate_page():
    st.title("Gaussian Channel Surrogate Model Comparison")

    st.markdown(
        r"""
This page compares trained curve-level surrogate models for the capacity
$C_E(t)$.

The surrogate models were trained only on:

$$
\gamma_A,\quad \gamma_B,\quad \omega_A,\quad \omega_B.
$$

All other physical and numerical parameters are fixed to the training values.
"""
    )

    st.sidebar.header("Surrogate input parameters")

    gamma_A = st.sidebar.number_input(
        r"$\gamma_A$",
        value=0.010,
        min_value=0.001,
        max_value=0.035,
        step=0.001,
        format="%.6f",
        key="sur_gamma_A",
    )

    gamma_B = st.sidebar.number_input(
        r"$\gamma_B$",
        value=0.010,
        min_value=0.001,
        max_value=0.035,
        step=0.001,
        format="%.6f",
        key="sur_gamma_B",
    )

    omega_A = st.sidebar.number_input(
        r"$\omega_A$",
        value=1.0,
        min_value=0.1,
        max_value=1.5,
        step=0.05,
        format="%.6f",
        key="sur_omega_A",
    )

    omega_B = st.sidebar.number_input(
        r"$\omega_B$",
        value=1.0,
        min_value=0.1,
        max_value=1.5,
        step=0.05,
        format="%.6f",
        key="sur_omega_B",
    )

    st.sidebar.header("What to compute")

    run_pca_ml = st.sidebar.checkbox(
        "PCA-ML surrogate",
        value=True,
        key="sur_run_pca_ml",
        help="Runs the compressed scikit-learn PCA-ML curve surrogate.",
    )

    run_torch_curve = st.sidebar.checkbox(
        "PyTorch curve surrogate",
        value=True,
        key="sur_run_torch_curve",
    )

    run_numerical = st.sidebar.checkbox(
        "Full numerical simulation",
        value=False,
        key="sur_run_numerical",
        help="If disabled, the expensive numerical simulation is not run.",
    )

    show_error_plots = st.sidebar.checkbox(
        "Show errors against numerical simulation",
        value=True,
        key="sur_show_error_plots",
    )

    show_prediction_table = st.sidebar.checkbox(
        "Show prediction table",
        value=False,
        key="sur_show_prediction_table",
    )

    run_button = st.sidebar.button(
        "Run comparison",
        key="sur_run_button",
    )

    if not run_button:
        st.info("Select what to compute, choose the input parameters, and press **Run comparison**.")
        return

    predictions = {}
    numerical_results = None

    domain_messages = validate_surrogate_domain(
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

    _, derived = build_physical_features_from_inputs(
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
    col3.metric(
        r"$\Delta\gamma=\gamma_B-\gamma_A$",
        f"{derived['gamma_delta']:.6g}",
    )
    col4.metric(
        r"$\Delta\omega=\omega_B-\omega_A$",
        f"{derived['omega_delta']:.6g}",
    )

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
                "linewidth": 2.3,
            }

        except Exception as exc:
            st.error(f"Could not run PyTorch curve surrogate: {exc}")

    if run_numerical:
        try:
            with st.spinner("Running full numerical simulation..."):
                t_num, C_num, numerical_results = run_fixed_numerical_for_surrogate(
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

    if not predictions:
        st.warning("No model or numerical simulation was selected.")
        return

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
            r"valid $C_E(t)$ fraction",
            f"{numerical_results['valid_fraction']:.2%}",
        )
        col2.metric(
            "mean reliability",
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


# ============================================================
# APP ROUTER
# ============================================================

if page == "Numerical simulation & diagnostics":
    render_numerical_page()
else:
    render_surrogate_page()