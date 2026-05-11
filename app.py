import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from src.simulation import ChannelParams, run_simulation


st.set_page_config(
    page_title="Gaussian Channel Simulation",
    layout="wide",
)

st.title("Long-Term Interaction Gaussian Channel Simulator")

st.markdown(
    """
This app simulates the long-term interaction channel between two non-identical harmonic oscillator detectors.

It computes:

- transmissivity $\\tau(t)$
- noise determinant $W(t)$
- energy-constrained classical capacity $C_E(t)$
"""
)


# ============================================================
# SIDEBAR PARAMETERS
# ============================================================

st.sidebar.header("Physical parameters")

gamma_A = st.sidebar.number_input("gamma_A", value=0.010, format="%.6f")
gamma_B = st.sidebar.number_input("gamma_B", value=0.012, format="%.6f")

omega_A = st.sidebar.number_input("omega_A", value=1.0, format="%.6f")
omega_B = st.sidebar.number_input("omega_B", value=1.2, format="%.6f")

m_A = st.sidebar.number_input("m_A", value=1.0, format="%.6f")
m_B = st.sidebar.number_input("m_B", value=1.0, format="%.6f")

sigma = st.sidebar.number_input("sigma", value=0.01, format="%.6f")
d = st.sidebar.number_input("d", value=1.0, format="%.6f")

E = st.sidebar.number_input("Energy bound E", value=100.0, format="%.6f")


st.sidebar.header("Numerical parameters")

t_max = st.sidebar.number_input("t_max", value=40.0, format="%.3f")
dt = st.sidebar.number_input("dt", value=0.01, format="%.5f")

n_noise_times = st.sidebar.number_input(
    "Number of W/C time points",
    value=120,
    min_value=10,
    step=10,
)

n_integral_points = st.sidebar.number_input(
    "Integral points for W",
    value=400,
    min_value=50,
    step=50,
)

rtol = st.sidebar.number_input("rtol", value=1e-8, format="%.1e")
atol = st.sidebar.number_input("atol", value=1e-10, format="%.1e")


run_button = st.sidebar.button("Run simulation")


# ============================================================
# HELPER PLOTTING FUNCTIONS
# ============================================================

def show_tau_plot(t, tau, d):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, tau, label=r"$\tau(t)$")
    ax.axvline(d, linestyle="--", label=r"$t=d$")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\tau$")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)


def show_W_plot(noise_times, W_values, d):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(noise_times, W_values, label=r"$W(t)=\det N(t)$")
    ax.axvline(d, linestyle="--", label=r"$t=d$")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$W$")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)


def show_log_comparison(noise_times, tau_on_noise_times, W_values, d):
    mask = (
        (noise_times >= d)
        & (np.abs(tau_on_noise_times) > 0.0)
        & (W_values > 0.0)
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        noise_times[mask],
        np.log10(np.abs(tau_on_noise_times[mask])),
        label=r"$\log_{10}|\tau|$",
    )
    ax.plot(
        noise_times[mask],
        0.5 * np.log10(W_values[mask]),
        label=r"$\log_{10}\sqrt{W}$",
    )
    ax.axvline(d, linestyle="--", label=r"$t=d$")
    ax.set_xlabel("t")
    ax.set_ylabel("log-scale")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)


def show_capacity_plot(noise_times, C_values, d, t_max_C, max_C):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(noise_times, C_values, label=r"$C_E(t)$")
    ax.axvline(d, linestyle="--", label=r"$t=d$")
    ax.scatter([t_max_C], [max_C], marker="o", label="maximum")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$C_E$")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)


# ============================================================
# MAIN APP LOGIC
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

    with st.spinner("Running simulation..."):
        results = run_simulation(
            params=params,
            t_max=t_max,
            dt=dt,
            n_noise_times=int(n_noise_times),
            n_integral_points=int(n_integral_points),
            rtol=rtol,
            atol=atol,
        )

    st.success("Simulation completed.")

    st.header("Derived parameters")

    col1, col2, col3 = st.columns(3)

    col1.metric(r"Sigma_A^2", f"{results['Sigma2_A']:.6g}")
    col2.metric(r"Sigma_B^2", f"{results['Sigma2_B']:.6g}")
    col3.metric("Causal time d", f"{d:.6g}")

    st.header("Capacity maximum")

    col1, col2, col3 = st.columns(3)

    col1.metric("max C_E", f"{results['max_C']:.6g}")
    col2.metric("time of max C_E", f"{results['t_max_C']:.6g}")
    col3.metric("delay after causal contact", f"{results['delay_max_C']:.6g}")

    st.header("Plots")

    st.subheader("Transmissivity")
    show_tau_plot(results["t"], results["tau"], d)

    st.subheader("Noise determinant")
    show_W_plot(results["noise_times"], results["W_values"], d)

    st.subheader("Log comparison: tau vs sqrt(W)")
    show_log_comparison(
        results["noise_times"],
        results["tau_on_noise_times"],
        results["W_values"],
        d,
    )

    st.subheader("Energy-constrained classical capacity")
    show_capacity_plot(
        results["noise_times"],
        results["C_values"],
        d,
        results["t_max_C"],
        results["max_C"],
    )

else:
    st.info("Insert parameters in the sidebar and press **Run simulation**.")