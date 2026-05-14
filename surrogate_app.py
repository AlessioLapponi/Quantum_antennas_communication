import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.simulation import ChannelParams, run_simulation
from src.surrogate_predictors import (
    load_ml_surrogate,
    predict_ml_capacity,
    load_torch_surrogate,
    predict_torch_capacity,
)


st.set_page_config(
    page_title="Surrogate Channel Comparison",
    layout="wide",
)

st.title("Gaussian Channel Surrogate Comparison")

st.markdown(
    """
This app compares the full numerical simulation with trained surrogate models:

- numerical simulation
- scikit-learn ML surrogate
- PyTorch neural-network surrogate
"""
)


# ============================================================
# SIDEBAR PARAMETERS
# ============================================================

st.sidebar.header("Physical parameters")

gamma_A = st.sidebar.number_input("gamma_A", value=0.010, format="%.6f")
gamma_B = st.sidebar.number_input("gamma_B", value=0.010, format="%.6f")

omega_A = st.sidebar.number_input("omega_A", value=1.0, format="%.6f")
omega_B = st.sidebar.number_input("omega_B", value=1.0, format="%.6f")

m_A = st.sidebar.number_input("m_A", value=1.0, format="%.6f")
m_B = st.sidebar.number_input("m_B", value=1.0, format="%.6f")

sigma = st.sidebar.number_input("sigma", value=0.01, format="%.6f")
d = st.sidebar.number_input("d", value=1.0, format="%.6f")

E = st.sidebar.number_input("Energy bound E", value=100.0, format="%.6f")


st.sidebar.header("Prediction grid")

t_max = st.sidebar.number_input("t_max", value=40.0, format="%.3f")

n_time_points = st.sidebar.number_input(
    "Number of plotted time points",
    value=200,
    min_value=20,
    step=20,
)

t_values = np.linspace(0.0, t_max, int(n_time_points))


st.sidebar.header("Numerical simulation settings")

run_numerical = st.sidebar.checkbox(
    "Run numerical simulation",
    value=True,
)

dt = st.sidebar.number_input("dt", value=0.0005, format="%.5f")

n_integral_points = st.sidebar.number_input(
    "Integral points for W",
    value=400,
    min_value=50,
    step=50,
)

rtol = st.sidebar.number_input("rtol", value=1e-8, format="%.1e")
atol = st.sidebar.number_input("atol", value=1e-10, format="%.1e")


st.sidebar.header("Models to show")

show_numerical = st.sidebar.checkbox("Show numerical simulation", value=True)
show_ml = st.sidebar.checkbox("Show scikit-learn ML surrogate", value=True)
show_torch = st.sidebar.checkbox("Show PyTorch NN surrogate", value=True)

show_error_plots = st.sidebar.checkbox(
    "Show error plots against numerical simulation",
    value=True,
)

show_table = st.sidebar.checkbox(
    "Show prediction table",
    value=False,
)

run_button = st.sidebar.button("Run comparison")


# ============================================================
# HELPERS
# ============================================================

@st.cache_resource
def cached_load_ml_surrogate():
    return load_ml_surrogate()


@st.cache_resource
def cached_load_torch_surrogate():
    return load_torch_surrogate()


def plot_capacity_comparison(
    t_values,
    numerical_C=None,
    ml_C=None,
    torch_C=None,
):
    fig, ax = plt.subplots(figsize=(9, 5))

    if numerical_C is not None and show_numerical:
        ax.plot(
            t_values,
            numerical_C,
            label="Numerical simulation",
            linewidth=2,
        )

    if ml_C is not None and show_ml:
        ax.plot(
            t_values,
            ml_C,
            linestyle="--",
            label="scikit-learn surrogate",
        )

    if torch_C is not None and show_torch:
        ax.plot(
            t_values,
            torch_C,
            linestyle=":",
            label="PyTorch NN surrogate",
        )

    ax.axvline(d, linestyle="--", color="gray", alpha=0.6, label=r"$t=d$")

    ax.set_xlabel("t")
    ax.set_ylabel(r"$C_E(t)$")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)


def plot_error_against_numerical(
    t_values,
    numerical_C,
    ml_C=None,
    torch_C=None,
):
    fig, ax = plt.subplots(figsize=(9, 5))

    if ml_C is not None and show_ml:
        ax.plot(
            t_values,
            ml_C - numerical_C,
            label="ML - numerical",
        )

    if torch_C is not None and show_torch:
        ax.plot(
            t_values,
            torch_C - numerical_C,
            label="NN - numerical",
        )

    ax.axhline(0.0, linestyle="--", color="gray")
    ax.axvline(d, linestyle="--", color="gray", alpha=0.6)

    ax.set_xlabel("t")
    ax.set_ylabel(r"Prediction error")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)


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
        "max_C": C_values[max_idx],
        "t_max_C": t_values[max_idx],
        "mean_C": np.nanmean(C_values),
        "valid_fraction": np.mean(valid),
    }


# ============================================================
# MAIN
# ============================================================

if run_button:
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

    st.header("Input parameters")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("gamma_A", f"{gamma_A:.6g}")
    col2.metric("gamma_B", f"{gamma_B:.6g}")
    col3.metric("omega_A", f"{omega_A:.6g}")
    col4.metric("omega_B", f"{omega_B:.6g}")

    col1, col2, col3 = st.columns(3)

    col1.metric(r"Sigma_A^2", f"{params.Sigma2_A:.6g}")
    col2.metric(r"Sigma_B^2", f"{params.Sigma2_B:.6g}")
    col3.metric("t_max", f"{t_max:.6g}")

    numerical_C = None
    ml_C = None
    torch_C = None
    numerical_results = None

    # ------------------------------------------------------------
    # ML surrogate
    # ------------------------------------------------------------

    if show_ml:
        try:
            with st.spinner("Loading and running scikit-learn surrogate..."):
                ml_model, ml_scaler = cached_load_ml_surrogate()

                ml_C = predict_ml_capacity(
                    gamma_A=gamma_A,
                    gamma_B=gamma_B,
                    omega_A=omega_A,
                    omega_B=omega_B,
                    t_values=t_values,
                    model=ml_model,
                    scaler=ml_scaler,
                    use_log_target=False,
                )

        except Exception as exc:
            st.error(f"Could not run scikit-learn surrogate: {exc}")

    # ------------------------------------------------------------
    # PyTorch surrogate
    # ------------------------------------------------------------

    if show_torch:
        try:
            with st.spinner("Loading and running PyTorch surrogate..."):
                torch_model, torch_scaler, torch_use_log_target, device = cached_load_torch_surrogate()

                torch_C = predict_torch_capacity(
                    gamma_A=gamma_A,
                    gamma_B=gamma_B,
                    omega_A=omega_A,
                    omega_B=omega_B,
                    t_values=t_values,
                    model=torch_model,
                    scaler=torch_scaler,
                    device=device,
                    use_log_target=torch_use_log_target,
                )

        except Exception as exc:
            st.error(f"Could not run PyTorch surrogate: {exc}")

    # ------------------------------------------------------------
    # Numerical simulation
    # ------------------------------------------------------------

    if run_numerical:
        try:
            with st.spinner("Running full numerical simulation..."):
                numerical_results = run_simulation(
                    params=params,
                    t_max=t_max,
                    dt=dt,
                    n_noise_times=int(n_time_points),
                    n_integral_points=int(n_integral_points),
                    rtol=rtol,
                    atol=atol,
                    determinant_quality_scale=1e-14,
                    tau_jump_local_factor=100.0,
                )

                numerical_C = numerical_results["C_values"]

        except Exception as exc:
            st.error(f"Could not run numerical simulation: {exc}")

    st.header("Capacity comparison")

    plot_capacity_comparison(
        t_values=t_values,
        numerical_C=numerical_C,
        ml_C=ml_C,
        torch_C=torch_C,
    )

    summaries = []

    if numerical_C is not None:
        summaries.append(
            summarize_curve(
                "Numerical simulation",
                t_values,
                numerical_C,
            )
        )

    if ml_C is not None:
        summaries.append(
            summarize_curve(
                "scikit-learn surrogate",
                t_values,
                ml_C,
            )
        )

    if torch_C is not None:
        summaries.append(
            summarize_curve(
                "PyTorch NN surrogate",
                t_values,
                torch_C,
            )
        )

    if summaries:
        st.subheader("Curve summaries")
        st.dataframe(summaries)

    if (
        show_error_plots
        and numerical_C is not None
        and (ml_C is not None or torch_C is not None)
    ):
        st.header("Prediction error against numerical simulation")

        plot_error_against_numerical(
            t_values=t_values,
            numerical_C=numerical_C,
            ml_C=ml_C,
            torch_C=torch_C,
        )

    if numerical_results is not None:
        st.header("Numerical reliability")

        col1, col2, col3 = st.columns(3)

        col1.metric("Valid C_E fraction", f"{numerical_results['valid_fraction']:.2%}")
        col2.metric("Mean reliability", f"{numerical_results['mean_reliability']:.3f}")
        col3.metric(
            "Tau jump detected",
            str(numerical_results["tau_jump_detected"]),
        )

        if numerical_results["tau_jump_detected"]:
            st.warning(
                f"Numerical tau discontinuity detected at "
                f"t={numerical_results['tau_jump_time']:.6g}. "
                "Later numerical C_E points are marked invalid."
            )

    if show_table:
        st.header("Prediction table")

        table = {
            "t": t_values,
        }

        if numerical_C is not None:
            table["C_numerical"] = numerical_C

        if ml_C is not None:
            table["C_ML"] = ml_C

        if torch_C is not None:
            table["C_NN"] = torch_C

        st.dataframe(table)

else:
    st.info("Insert parameters and press **Run comparison**.")