import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.special import dawsn


# ============================================================
# PARAMETERS BLOCK
# ============================================================

GAMMA_A = 0.005
GAMMA_B = 0.005

OMEGA_A = 1.0
OMEGA_B = 1.0

M_A = 1.0
M_B = 1.0

SIGMA = 0.01
D = 1.0

E = 100.0

T_MAX = 50.0
DT = 0.01

N_NOISE_TIMES = 120
N_INTEGRAL_POINTS = 400

RTOL = 1e-8
ATOL = 1e-10

SAVE_FIGURES = False
OUTPUT_PREFIX = "channel_output"


# ============================================================
# MODEL PARAMETERS
# ============================================================

@dataclass
class ChannelParams:
    gamma_A: float
    gamma_B: float
    omega_A: float
    omega_B: float
    sigma: float
    d: float
    m_A: float = 1.0
    m_B: float = 1.0
    E: float = 100.0

    @property
    def Sigma2_A(self):
        return np.sqrt(8.0 / np.pi) * self.gamma_A / self.sigma - self.omega_A**2

    @property
    def Sigma2_B(self):
        return np.sqrt(8.0 / np.pi) * self.gamma_B / self.sigma - self.omega_B**2

    @property
    def coupling_delay(self):
        return 2.0 * np.sqrt(self.gamma_A * self.gamma_B) / self.d


# ============================================================
# GREEN FUNCTIONS
# ============================================================

def solve_green_functions(params, t_max, dt=0.01, rtol=1e-8, atol=1e-10):
    """
    Solves the delayed Green-function system.

    State vector:
        y = (G_AA, dG_AA, G_AB, dG_AB, G_BA, dG_BA, G_BB, dG_BB)

    Initial conditions:
        G_AA(0)=G_AB(0)=G_BA(0)=G_BB(0)=0
        dG_AA(0)=1
        dG_BB(0)=1
        dG_AB(0)=dG_BA(0)=0
    """

    d = params.d
    kappa = params.coupling_delay

    y0 = np.array([
        0.0, 1.0,
        0.0, 0.0,
        0.0, 0.0,
        0.0, 1.0,
    ])

    all_t = []
    all_y = []

    t_known = np.array([0.0])
    y_known = y0.reshape(8, 1)

    def make_history_interpolator(t_known, y_known):
        if len(t_known) == 1:

            def hist_single_point(t_delay):
                if t_delay < 0.0:
                    return {"G_AA": 0.0, "G_AB": 0.0, "G_BA": 0.0, "G_BB": 0.0}

                return {
                    "G_AA": float(y_known[0, 0]),
                    "G_AB": float(y_known[2, 0]),
                    "G_BA": float(y_known[4, 0]),
                    "G_BB": float(y_known[6, 0]),
                }

            return hist_single_point

        interp_kind = "cubic" if len(t_known) >= 4 else "linear"

        interpolators = [
            interp1d(
                t_known,
                y_known[i],
                kind=interp_kind,
                bounds_error=False,
                fill_value=(y_known[i, 0], y_known[i, -1]),
            )
            for i in range(y_known.shape[0])
        ]

        def hist(t_delay):
            if t_delay < 0.0:
                return {"G_AA": 0.0, "G_AB": 0.0, "G_BA": 0.0, "G_BB": 0.0}

            vals = [float(interp(t_delay)) for interp in interpolators]

            return {
                "G_AA": vals[0],
                "G_AB": vals[2],
                "G_BA": vals[4],
                "G_BB": vals[6],
            }

        return hist

    current_t0 = 0.0
    current_y0 = y0.copy()

    while current_t0 < t_max:
        current_t1 = min(current_t0 + d, t_max)

        hist = make_history_interpolator(t_known, y_known)

        def rhs(t, y):
            G_AA, v_AA, G_AB, v_AB, G_BA, v_BA, G_BB, v_BB = y

            delayed = hist(t - d)
            theta = 1.0 if t >= d else 0.0

            rhs_AA = kappa * delayed["G_BA"] * theta
            rhs_AB = kappa * delayed["G_BB"] * theta
            rhs_BA = kappa * delayed["G_AA"] * theta
            rhs_BB = kappa * delayed["G_AB"] * theta

            a_AA = params.Sigma2_A * G_AA - 2.0 * params.gamma_A * v_AA + rhs_AA
            a_AB = params.Sigma2_A * G_AB - 2.0 * params.gamma_A * v_AB + rhs_AB
            a_BA = params.Sigma2_B * G_BA - 2.0 * params.gamma_B * v_BA + rhs_BA
            a_BB = params.Sigma2_B * G_BB - 2.0 * params.gamma_B * v_BB + rhs_BB

            return np.array([
                v_AA, a_AA,
                v_AB, a_AB,
                v_BA, a_BA,
                v_BB, a_BB,
            ])

        t_eval = np.arange(current_t0, current_t1 + 0.5 * dt, dt)
        t_eval = t_eval[t_eval <= current_t1]

        sol = solve_ivp(
            rhs,
            (current_t0, current_t1),
            current_y0,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            method="DOP853",
        )

        if not sol.success:
            raise RuntimeError(sol.message)

        seg_t = sol.t[1:]
        seg_y = sol.y[:, 1:]

        if len(seg_t) > 0:
            all_t.append(seg_t)
            all_y.append(seg_y)

            t_known = np.concatenate([t_known, seg_t])
            y_known = np.concatenate([y_known, seg_y], axis=1)

        current_t0 = current_t1
        current_y0 = sol.y[:, -1]

    t = np.concatenate(all_t)
    y = np.concatenate(all_y, axis=1)

    return t, y


def build_full_green_arrays(t, y):
    y0 = np.array([
        0.0, 1.0,
        0.0, 0.0,
        0.0, 0.0,
        0.0, 1.0,
    ])

    t_full = np.concatenate([[0.0], t])
    y_full = np.concatenate([y0.reshape(8, 1), y], axis=1)

    return t_full, y_full


def make_green_interpolators(t_full, y_full):
    names = [
        "G_AA", "dG_AA",
        "G_AB", "dG_AB",
        "G_BA", "dG_BA",
        "G_BB", "dG_BB",
    ]

    interpolators = {}

    for i, name in enumerate(names):
        interpolators[name] = interp1d(
            t_full,
            y_full[i],
            kind="cubic",
            bounds_error=False,
            fill_value=(y_full[i, 0], y_full[i, -1]),
        )

    return interpolators


def compute_ddot_G_BA_values(t_values, green_interp, params):
    t_values = np.asarray(t_values)

    G_BA_vals = green_interp["G_BA"](t_values)
    dG_BA_vals = green_interp["dG_BA"](t_values)

    delayed_argument = t_values - params.d
    G_AA_delay = np.zeros_like(t_values, dtype=float)

    mask = delayed_argument >= 0.0
    G_AA_delay[mask] = green_interp["G_AA"](delayed_argument[mask])

    return (
        params.Sigma2_B * G_BA_vals
        - 2.0 * params.gamma_B * dG_BA_vals
        + params.coupling_delay * G_AA_delay
    )


def compute_ddot_G_BB_values(t_values, green_interp, params):
    t_values = np.asarray(t_values)

    G_BB_vals = green_interp["G_BB"](t_values)
    dG_BB_vals = green_interp["dG_BB"](t_values)

    delayed_argument = t_values - params.d
    G_AB_delay = np.zeros_like(t_values, dtype=float)

    mask = delayed_argument >= 0.0
    G_AB_delay[mask] = green_interp["G_AB"](delayed_argument[mask])

    return (
        params.Sigma2_B * G_BB_vals
        - 2.0 * params.gamma_B * dG_BB_vals
        + params.coupling_delay * G_AB_delay
    )


def compute_tau_values(t_values, green_interp, params):
    t_values = np.asarray(t_values)

    G_BA = green_interp["G_BA"](t_values)
    dG_BA = green_interp["dG_BA"](t_values)
    ddG_BA = compute_ddot_G_BA_values(t_values, green_interp, params)

    return (params.omega_A / params.omega_B) * (dG_BA**2 - G_BA * ddG_BA)


# ============================================================
# NOISE KERNELS
# ============================================================

def lambda_from_gamma(gamma, mass):
    return np.sqrt(8.0 * np.pi * mass * gamma)


def nu_AA(delta, params):
    lam_A = lambda_from_gamma(params.gamma_A, params.m_A)
    sigma = params.sigma

    x = delta / (np.sqrt(2.0) * sigma)

    return (
        lam_A**2
        / (8.0 * np.pi**2 * sigma**2)
        * (1.0 - (np.sqrt(2.0) * delta / sigma) * dawsn(x))
    )


def nu_BB(delta, params):
    lam_B = lambda_from_gamma(params.gamma_B, params.m_B)
    sigma = params.sigma

    x = delta / (np.sqrt(2.0) * sigma)

    return (
        lam_B**2
        / (8.0 * np.pi**2 * sigma**2)
        * (1.0 - (np.sqrt(2.0) * delta / sigma) * dawsn(x))
    )


def nu_AB(delta, params):
    lam_A = lambda_from_gamma(params.gamma_A, params.m_A)
    lam_B = lambda_from_gamma(params.gamma_B, params.m_B)

    sigma = params.sigma
    d = params.d

    x_plus = (delta + d) / (np.sqrt(2.0) * sigma)
    x_minus = (delta - d) / (np.sqrt(2.0) * sigma)

    return (
        np.sqrt(2.0)
        * lam_A
        * lam_B
        / (8.0 * np.pi**2 * sigma * d)
        * (dawsn(x_plus) - dawsn(x_minus))
    )


def nu_BA(delta, params):
    return nu_AB(delta, params)


# ============================================================
# NOISE MATRIX N AND W = det(N)
# ============================================================

def trapezoid_weights(x):
    x = np.asarray(x)

    if len(x) == 1:
        return np.array([0.0])

    w = np.zeros_like(x, dtype=float)

    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])

    return w


def compute_N_initial_Bob(T, green_interp, params):
    G_BB = float(green_interp["G_BB"](T))
    dG_BB = float(green_interp["dG_BB"](T))
    ddG_BB = float(compute_ddot_G_BB_values(np.array([T]), green_interp, params)[0])

    omega_B = params.omega_B

    A = np.array([
        [dG_BB, omega_B * G_BB],
        [ddG_BB / omega_B, dG_BB],
    ])

    sigma_BB_0 = 0.5 * np.eye(2)

    return A @ sigma_BB_0 @ A.T


def compute_Nprime_B(T, green_interp, params, n_integral_points=400):
    if T <= 0.0:
        return np.zeros((2, 2))

    r = np.linspace(0.0, T, n_integral_points)
    w = trapezoid_weights(r)
    W2 = np.outer(w, w)

    u = T - r

    G_BA_u = green_interp["G_BA"](u)
    G_BB_u = green_interp["G_BB"](u)

    dG_BA_u = green_interp["dG_BA"](u)
    dG_BB_u = green_interp["dG_BB"](u)

    G_row = [G_BA_u, G_BB_u]
    dG_row = [dG_BA_u, dG_BB_u]

    delta = r[:, None] - r[None, :]

    nu = [
        [nu_AA(delta, params), nu_AB(delta, params)],
        [nu_BA(delta, params), nu_BB(delta, params)],
    ]

    scale = [
        np.sqrt(params.omega_A / params.m_A),
        np.sqrt(params.omega_B / params.m_B),
    ]

    K = [
        [scale[i] * nu[i][j] * scale[j] for j in range(2)]
        for i in range(2)
    ]

    def double_integral(x_list, y_list):
        value = 0.0

        for i in range(2):
            for j in range(2):
                value += np.sum(
                    W2
                    * x_list[i][:, None]
                    * K[i][j]
                    * y_list[j][None, :]
                )

        return value

    N11 = double_integral(G_row, G_row)
    N12 = double_integral(G_row, dG_row) / params.omega_B
    N22 = double_integral(dG_row, dG_row) / params.omega_B**2

    return np.array([
        [N11, N12],
        [N12, N22],
    ])


def compute_N_and_W(T, green_interp, params, n_integral_points=400):
    N_initial = compute_N_initial_Bob(T, green_interp, params)
    N_prime = compute_Nprime_B(
        T,
        green_interp,
        params,
        n_integral_points=n_integral_points,
    )

    N_total = N_initial + N_prime
    W_det = np.linalg.det(N_total)

    return N_total, W_det


# ============================================================
# CLASSICAL CAPACITY
# ============================================================

def h_bosonic(x):
    """
    h(x) = (x+1/2) log2(x+1/2) - (x-1/2) log2(x-1/2)

    Stable implementation using n = x - 1/2.
    """

    x = np.asarray(x, dtype=float)
    n = x - 0.5

    out = np.full_like(x, np.nan, dtype=float)

    mask_zero = np.isclose(n, 0.0, atol=1e-15, rtol=0.0)
    mask_pos = n > 0.0

    out[mask_zero] = 0.0

    n_pos = n[mask_pos]
    out[mask_pos] = (
        np.log1p(n_pos) + n_pos * np.log1p(1.0 / n_pos)
    ) / np.log(2.0)

    if out.shape == ():
        return float(out)

    return out


def compute_capacity_E(tau_values, W_values, params):
    tau_abs = np.abs(np.asarray(tau_values, dtype=float))
    W_values = np.asarray(W_values, dtype=float)

    C = np.full_like(tau_abs, np.nan, dtype=float)

    mask_valid = W_values >= 0.0

    sqrt_W = np.full_like(W_values, np.nan, dtype=float)
    sqrt_W[mask_valid] = np.sqrt(W_values[mask_valid])

    x1 = params.E / params.omega_A * tau_abs + sqrt_W
    x2 = 0.5 * tau_abs + sqrt_W

    C[mask_valid] = h_bosonic(x1[mask_valid]) - h_bosonic(x2[mask_valid])

    tiny_negative = (C < 0.0) & (C > -1e-10)
    C[tiny_negative] = 0.0

    return C


def save_or_show(filename=None):
    if SAVE_FIGURES and filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

def run_simulation(
    params,
    t_max=40.0,
    dt=0.01,
    n_noise_times=120,
    n_integral_points=400,
    rtol=1e-8,
    atol=1e-10,
):
    t, y = solve_green_functions(
        params,
        t_max=t_max,
        dt=dt,
        rtol=rtol,
        atol=atol,
    )

    t_full, y_full = build_full_green_arrays(t, y)
    green_interp = make_green_interpolators(t_full, y_full)

    tau_full = compute_tau_values(t_full, green_interp, params)
    tau = tau_full[1:]

    noise_times = np.linspace(0.0, t_max, n_noise_times)

    W_values = np.zeros_like(noise_times)
    N_values = []

    for idx, T in enumerate(noise_times):
        N_T, W_T = compute_N_and_W(
            T,
            green_interp,
            params,
            n_integral_points=n_integral_points,
        )

        N_values.append(N_T)
        W_values[idx] = W_T

    N_values = np.array(N_values)

    tau_on_noise_times = compute_tau_values(noise_times, green_interp, params)
    C_values = compute_capacity_E(tau_on_noise_times, W_values, params)

    ratio_tau_sqrtW = np.full_like(W_values, np.nan, dtype=float)

    mask_ratio = W_values > 0.0
    ratio_tau_sqrtW[mask_ratio] = (
        np.abs(tau_on_noise_times[mask_ratio])
        / np.sqrt(W_values[mask_ratio])
    )

    valid_C = np.isfinite(C_values)

    if np.any(valid_C):
        max_idx = np.nanargmax(C_values)
        max_C = C_values[max_idx]
        t_max_C = noise_times[max_idx]
        delay_max_C = t_max_C - params.d
    else:
        max_idx = None
        max_C = np.nan
        t_max_C = np.nan
        delay_max_C = np.nan

    return {
        "params": params,
        "t": t,
        "y": y,
        "t_full": t_full,
        "y_full": y_full,
        "tau": tau,
        "noise_times": noise_times,
        "tau_on_noise_times": tau_on_noise_times,
        "N_values": N_values,
        "W_values": W_values,
        "C_values": C_values,
        "ratio_tau_sqrtW": ratio_tau_sqrtW,
        "Sigma2_A": params.Sigma2_A,
        "Sigma2_B": params.Sigma2_B,
        "max_idx": max_idx,
        "max_C": max_C,
        "t_max_C": t_max_C,
        "delay_max_C": delay_max_C,
    }
