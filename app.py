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
gamma_B = st.sidebar.number_input("gamma_B", value=0.010, format="%.6f")

omega_A = st.sidebar.number_input("omega_A", value=1.0, format="%.6f")
omega_B = st.sidebar.number_input("omega_B", value=1.0, format="%.6f")

m_A = st.sidebar.number_input("m_A", value=1.0, format="%.6f")
m_B = st.sidebar.number_input("m_B", value=1.0, format="%.6f")

sigma = st.sidebar.number_input("sigma", value=0.01, format="%.6f")
d = st.sidebar.number_input("d", value=1.0, format="%.6f")

E = st.sidebar.number_input("Energy bound E", value=100.0, format="%.6f")


st.sidebar.header("Numerical parameters")

t_max = st.sidebar.number_input("t_max", value=40.0, format="%.3f")
dt = st.sidebar.number_input("dt", value=0.0005, format="%.5f")

n_noise_times = st.sidebar.number_input(
    "Number of W/C time points",
    value=200,
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


st.sidebar.header("Reliability filter parameters")

reliability_threshold = st.sidebar.slider(
    "Training reliability threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.05,
)

h_tolerance = st.sidebar.number_input(
    "h-domain tolerance",
    value=1e-12,
    format="%.1e",
)

h_margin_scale = st.sidebar.number_input(
    "h-margin scale",
    value=1e-6,
    format="%.1e",
    help=(
        "Smaller values make the h-domain margin filter less strict. "
        "Larger values penalize points farther from the boundary."
    ),
)

determinant_quality_scale = st.sidebar.number_input(
    "determinant quality scale",
    value=1e-14,
    format="%.1e",
    help=(
        "Larger values make the determinant filter stricter. "
        "Smaller values make it more permissive."
    ),
)

outlier_window = st.sidebar.number_input(
    "outlier window",
    value=5,
    min_value=1,
    max_value=50,
    step=1,
)

outlier_threshold = st.sidebar.number_input(
    "outlier threshold",
    value=4.0,
    min_value=0.5,
    max_value=50.0,
    step=0.5,
    help="Lower values make the local C_E outlier filter stricter.",
)

outlier_alpha = st.sidebar.number_input(
    "outlier penalty alpha",
    value=0.5,
    min_value=0.01,
    max_value=10.0,
    step=0.05,
    help="Higher values penalize detected outliers more strongly.",
)

bounce_window = st.sidebar.number_input(
    "bounce protection window",
    value=1,
    min_value=0,
    max_value=20,
    step=1,
    help=(
        "Number of neighboring C_E points protected around |tau| bounces. "
        "Lower values are stricter."
    ),
)


st.sidebar.header("Tau discontinuity filter")

tau_jump_filter = st.sidebar.checkbox(
    "Enable tau local-step discontinuity filter",
    value=True,
)

tau_jump_local_window = st.sidebar.number_input(
    "tau jump local window",
    value=5,
    min_value=2,
    max_value=50,
    step=1,
    help=(
        "Number of neighboring fine-grid tau steps used on each side. "
        "5 means 5 left and 5 right."
    ),
)

tau_jump_local_factor = st.sidebar.number_input(
    "tau local jump factor",
    value=100.0,
    min_value=2.0,
    max_value=10000.0,
    step=5.0,
    help=(
        "A fine-grid tau step is flagged if it is this many times larger "
        "than its neighboring fine-grid tau steps. Higher = less strict."
    ),
)

run_button = st.sidebar.button("Run simulation")


# ============================================================
# HELPER PLOTTING FUNCTIONS
# ============================================================

def show_tau_plot(t, tau, d, tau_jump_time=None, tau_zero_times=None):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(t, tau, label=r"$\tau(t)$")
    ax.axvline(d, linestyle="--", label=r"$t=d$")

    if tau_zero_times is not None:
        for zero_time in tau_zero_times:
            ax.axvline(
                zero_time,
                linestyle=":",
                alpha=0.4,
                label=r"$\tau=0$ crossing" if zero_time == tau_zero_times[0] else None,
            )

    if tau_jump_time is not None:
        ax.axvline(
            tau_jump_time,
            linestyle="-.",
            label=r"detected $\tau$ discontinuity",
        )

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

    if np.any(mask):
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

    if np.isfinite(t_max_C) and np.isfinite(max_C):
        ax.scatter([t_max_C], [max_C], marker="o", label="maximum")

    ax.set_xlabel("t")
    ax.set_ylabel(r"$C_E$")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)


def show_reliability_plot(noise_times, reliability, valid_C, d):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(noise_times, reliability, label="reliability score")

    if np.any(~valid_C):
        ax.scatter(
            noise_times[~valid_C],
            np.zeros(np.sum(~valid_C)),
            marker="x",
            label="invalid C_E points",
        )

    ax.axvline(d, linestyle="--", label=r"$t=d$")

    ax.set_xlabel("t")
    ax.set_ylabel("reliability")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)


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

    ax.set_xlabel("t")
    ax.set_ylabel("absolute tau step")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)


def show_reliability_components_plot(noise_times, diagnostics, d):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(noise_times, diagnostics["r_domain"], label=r"$r_{\rm domain}$")
    ax.plot(noise_times, diagnostics["r_det"], label=r"$r_{\rm det}$")
    ax.plot(noise_times, diagnostics["r_outlier"], label=r"$r_{\rm outlier}$")

    ax.axvline(d, linestyle="--", label=r"$t=d$")

    ax.set_xlabel("t")
    ax.set_ylabel("component score")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)


def show_h_margin_plot(noise_times, diagnostics, d):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        noise_times,
        diagnostics["h_margin"],
        label=r"$\min(x_1-1/2,x_2-1/2)$",
    )
    ax.axhline(0.0, linestyle="--", label="domain boundary")
    ax.axvline(d, linestyle="--", label=r"$t=d$")

    ax.set_xlabel("t")
    ax.set_ylabel("h-domain margin")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)


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

    ax.set_xlabel("t")
    ax.set_ylabel(r"$\log_{10}$ determinant quality")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)


def show_outlier_score_plot(noise_times, diagnostics, d):
    outlier_score = diagnostics["outlier_score"]
    bounce_mask = diagnostics["bounce_mask"]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(noise_times, outlier_score, label="local outlier score")

    if np.any(bounce_mask):
        ax.scatter(
            noise_times[bounce_mask],
            outlier_score[bounce_mask],
            s=20,
            label=r"protected $|\tau|$-bounce region",
        )

    ax.axvline(d, linestyle="--", label=r"$t=d$")

    ax.set_xlabel("t")
    ax.set_ylabel("outlier score")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)


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
            label="accepted for training",
        )

    if np.any(rejected_mask):
        ax.scatter(
            noise_times[rejected_mask],
            C_values[rejected_mask],
            s=18,
            marker="x",
            label="rejected / low reliability",
        )

    if np.any(~valid_C):
        ax.scatter(
            noise_times[~valid_C],
            np.zeros(np.sum(~valid_C)),
            s=18,
            marker="x",
            label="invalid",
        )

    ax.axvline(d, linestyle="--", label=r"$t=d$")

    if tau_jump_time is not None:
        ax.axvline(
            tau_jump_time,
            linestyle=":",
            label=r"detected $\tau$ discontinuity",
        )

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
    show_tau_plot(
        results["t"],
        results["tau"],
        d,
        tau_jump_time=results["tau_jump_time"],
        tau_zero_times=results["tau_zero_times"],
    )

    st.subheader("Noise determinant")
    show_W_plot(
        results["noise_times"],
        results["W_values"],
        d,
    )

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

    st.header("Reliability diagnostics")

    training_mask = results["valid_C"] & (
        results["reliability"] >= reliability_threshold
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Valid C_E fraction", f"{results['valid_fraction']:.2%}")
    col2.metric("Mean reliability", f"{results['mean_reliability']:.3f}")
    col3.metric("Invalid C_E points", int(np.sum(~results["valid_C"])))
    col4.metric(
        "Usable training points",
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

    st.subheader("Tau step-jump diagnostic")
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
        st.success("No tau discontinuity detected on the fine tau grid.")

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
        st.subheader("h-domain margin")
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

        st.subheader("Local outlier score and protected bounce regions")
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

            st.dataframe(low_rel_table)
        else:
            st.success("No valid points below the reliability threshold.")

else:
    st.info("Insert parameters in the sidebar and press **Run simulation**.")