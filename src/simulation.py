import numpy as np

from dataclasses import dataclass
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.special import dawsn


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
                    return {
                        "G_AA": 0.0,
                        "G_AB": 0.0,
                        "G_BA": 0.0,
                        "G_BB": 0.0,
                    }

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
                return {
                    "G_AA": 0.0,
                    "G_AB": 0.0,
                    "G_BA": 0.0,
                    "G_BB": 0.0,
                }

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

    # Explicit 2x2 determinant.
    W_det = N_total[0, 0] * N_total[1, 1] - N_total[0, 1] * N_total[1, 0]

    return N_total, W_det


# ============================================================
# CLASSICAL CAPACITY
# ============================================================

def h_bosonic(x, tolerance=1e-12):
    """
    h(x) = (x+1/2) log2(x+1/2) - (x-1/2) log2(x-1/2)

    Domain: x >= 1/2.
    Tiny numerical violations are clipped to 1/2.
    """

    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)

    x_safe = x.copy()

    slightly_below = (x_safe < 0.5) & (x_safe >= 0.5 - tolerance)
    x_safe[slightly_below] = 0.5

    valid = x_safe >= 0.5

    n = x_safe[valid] - 0.5
    h_valid = np.zeros_like(n)

    positive = n > 0.0
    n_pos = n[positive]

    h_valid[positive] = (
        np.log1p(n_pos) + n_pos * np.log1p(1.0 / n_pos)
    ) / np.log(2.0)

    out[valid] = h_valid

    if out.shape == ():
        return float(out)

    return out


def compute_capacity_E(
    tau_values,
    W_values,
    params,
    mode="strict",
    tolerance=1e-12,
):
    """
    Energy-constrained classical capacity.

    strict:
        Uses the original formula.
        Invalid h-domain points become NaN.

    clip_nbar:
        Rewrites through nbar and imposes nbar >= 0.

    clip_h:
        Clips all h-domain violations to 1/2.
        Use only for visualization/debugging.
    """

    tau = np.asarray(tau_values, dtype=float)
    tau_abs = np.abs(tau)
    W_values = np.asarray(W_values, dtype=float)

    C = np.full_like(tau_abs, np.nan, dtype=float)

    valid_W = W_values >= 0.0
    sqrt_W = np.full_like(W_values, np.nan, dtype=float)
    sqrt_W[valid_W] = np.sqrt(W_values[valid_W])

    if mode == "strict":
        x1 = params.E / params.omega_A * tau_abs + sqrt_W
        x2 = 0.5 * tau_abs + sqrt_W

        valid_domain = (
            valid_W
            & np.isfinite(x1)
            & np.isfinite(x2)
            & (x1 >= 0.5 - tolerance)
            & (x2 >= 0.5 - tolerance)
        )

        x1_safe = x1.copy()
        x2_safe = x2.copy()

        x1_safe[(x1_safe < 0.5) & (x1_safe >= 0.5 - tolerance)] = 0.5
        x2_safe[(x2_safe < 0.5) & (x2_safe >= 0.5 - tolerance)] = 0.5

        C[valid_domain] = (
            h_bosonic(x1_safe[valid_domain])
            - h_bosonic(x2_safe[valid_domain])
        )

    elif mode == "clip_nbar":
        denominator = np.abs(1.0 - tau)
        safe_denominator = denominator.copy()
        safe_denominator[safe_denominator < tolerance] = np.nan

        nbar = sqrt_W / safe_denominator - 0.5
        nbar = np.maximum(0.0, nbar)

        x1 = tau_abs * params.E / params.omega_A + denominator * nbar + 0.5
        x2 = 0.5 * tau_abs + denominator * nbar + 0.5

        valid_domain = (
            valid_W
            & np.isfinite(x1)
            & np.isfinite(x2)
            & (x1 >= 0.5 - tolerance)
            & (x2 >= 0.5 - tolerance)
        )

        C[valid_domain] = (
            h_bosonic(x1[valid_domain])
            - h_bosonic(x2[valid_domain])
        )

    elif mode == "clip_h":
        x1 = params.E / params.omega_A * tau_abs + sqrt_W
        x2 = 0.5 * tau_abs + sqrt_W

        x1_safe = x1.copy()
        x2_safe = x2.copy()

        x1_safe[x1_safe < 0.5] = 0.5
        x2_safe[x2_safe < 0.5] = 0.5

        valid_domain = valid_W & np.isfinite(x1_safe) & np.isfinite(x2_safe)

        C[valid_domain] = (
            h_bosonic(x1_safe[valid_domain])
            - h_bosonic(x2_safe[valid_domain])
        )

    else:
        raise ValueError(
            "Unknown capacity mode. Use 'strict', 'clip_nbar', or 'clip_h'."
        )

    tiny_negative = (C < 0.0) & (C > -1e-10)
    C[tiny_negative] = 0.0

    return C


def compute_capacity_arguments(tau_values, W_values, params):
    tau = np.asarray(tau_values, dtype=float)
    tau_abs = np.abs(tau)
    W_values = np.asarray(W_values, dtype=float)

    sqrt_W = np.full_like(W_values, np.nan, dtype=float)
    mask = W_values >= 0.0
    sqrt_W[mask] = np.sqrt(W_values[mask])

    x1 = params.E / params.omega_A * tau_abs + sqrt_W
    x2 = 0.5 * tau_abs + sqrt_W

    return x1, x2, sqrt_W


def diagnose_capacity_inputs(t_values, tau_values, W_values, C_values, params, h_tolerance=1e-10):
    """
    Returns diagnostic arrays and masks for capacity computation.
    """

    tau_abs = np.abs(np.asarray(tau_values, dtype=float))
    W_values = np.asarray(W_values, dtype=float)
    C_values = np.asarray(C_values, dtype=float)

    sqrt_W = np.full_like(W_values, np.nan, dtype=float)

    mask_W_nonnegative = W_values >= 0.0
    sqrt_W[mask_W_nonnegative] = np.sqrt(W_values[mask_W_nonnegative])

    x1 = params.E / params.omega_A * tau_abs + sqrt_W
    x2 = 0.5 * tau_abs + sqrt_W

    bad = {
        "W_negative": W_values < 0.0,
        "W_nan": np.isnan(W_values),
        "W_inf": np.isinf(W_values),
        "tau_nan": np.isnan(tau_values),
        "tau_inf": np.isinf(tau_values),
        "x1_below_half": x1 < 0.5 - h_tolerance,
        "x2_below_half": x2 < 0.5 - h_tolerance,
        "x1_nan": np.isnan(x1),
        "x2_nan": np.isnan(x2),
        "x1_inf": np.isinf(x1),
        "x2_inf": np.isinf(x2),
        "C_nan": np.isnan(C_values),
        "C_inf": np.isinf(C_values),
    }

    return {
        "x1": x1,
        "x2": x2,
        "sqrt_W": sqrt_W,
        "bad_masks": bad,
    }


# ============================================================
# RELIABILITY
# ============================================================

def compute_tau_bounce_mask(
    tau_values,
    window=2,
    local_window=5,
    zero_fraction=0.05,
):
    """
    Protects regions where C_E may show a physical bounce because tau crosses zero.
    """

    tau = np.asarray(tau_values, dtype=float)
    tau_abs = np.abs(tau)
    n = len(tau)

    bounce_mask = np.zeros(n, dtype=bool)
    finite = np.isfinite(tau)

    def protect_index(k):
        i0 = max(0, k - window)
        i1 = min(n, k + window + 1)
        bounce_mask[i0:i1] = True

    for k in range(1, n):
        if not (finite[k] and finite[k - 1]):
            continue

        if tau[k] == 0.0 or tau[k - 1] == 0.0 or tau[k] * tau[k - 1] < 0.0:
            protect_index(k)
            protect_index(k - 1)

    for k in range(n):
        if not finite[k]:
            continue

        i0 = max(0, k - local_window)
        i1 = min(n, k + local_window + 1)

        local_abs = tau_abs[i0:i1]
        local_abs = local_abs[np.isfinite(local_abs)]

        if len(local_abs) < 3:
            continue

        local_amplitude = np.nanpercentile(local_abs, 80)

        if local_amplitude <= 0.0 or not np.isfinite(local_amplitude):
            continue

        if tau_abs[k] <= zero_fraction * local_amplitude:
            protect_index(k)

    return bounce_mask


def compute_determinant_quality(N_values, epsilon=1e-300):
    N_values = np.asarray(N_values, dtype=float)

    a = N_values[:, 0, 0] * N_values[:, 1, 1]
    b = N_values[:, 0, 1] * N_values[:, 1, 0]

    W_direct = a - b

    quality = np.abs(W_direct) / (np.abs(a) + np.abs(b) + epsilon)
    quality[~np.isfinite(quality)] = 0.0

    return quality


def compute_local_outlier_score(values, window=5, epsilon=1e-12):
    values = np.asarray(values, dtype=float)
    score = np.zeros_like(values, dtype=float)

    n = len(values)

    for k in range(n):
        i0 = max(0, k - window)
        i1 = min(n, k + window + 1)

        local = values[i0:i1]
        local = local[np.isfinite(local)]

        if len(local) < 5 or not np.isfinite(values[k]):
            score[k] = np.nan
            continue

        med = np.median(local)
        mad = np.median(np.abs(local - med)) + epsilon

        score[k] = np.abs(values[k] - med) / mad

    return score


def compute_reliability_scores(
    noise_times,
    tau_values,
    W_values,
    C_values,
    N_values,
    params,
    h_tolerance=1e-12,
    h_margin_scale=1e-6,
    determinant_quality_scale=1e-10,
    outlier_window=5,
    outlier_threshold=4.0,
    outlier_alpha=0.5,
    bounce_window=1,
):
    """
    Computes pointwise validity and reliability scores on the coarse C_E/W grid.
    """

    tau_values = np.asarray(tau_values, dtype=float)
    W_values = np.asarray(W_values, dtype=float)
    C_values = np.asarray(C_values, dtype=float)
    N_values = np.asarray(N_values, dtype=float)

    n = len(C_values)

    x1, x2, sqrt_W = compute_capacity_arguments(
        tau_values,
        W_values,
        params,
    )

    valid = (
        np.isfinite(tau_values)
        & np.isfinite(W_values)
        & np.isfinite(C_values)
        & (W_values > 0.0)
        & np.isfinite(x1)
        & np.isfinite(x2)
        & (x1 >= 0.5 - h_tolerance)
        & (x2 >= 0.5 - h_tolerance)
    )

    reliability = np.zeros(n, dtype=float)
    reliability[valid] = 1.0

    h_margin = np.minimum(x1 - 0.5, x2 - 0.5)

    r_domain = np.clip(
        h_margin / h_margin_scale,
        0.0,
        1.0,
    )
    r_domain[~np.isfinite(r_domain)] = 0.0

    det_quality = compute_determinant_quality(N_values)

    r_det = np.clip(
        det_quality / determinant_quality_scale,
        0.0,
        1.0,
    )
    r_det[~np.isfinite(r_det)] = 0.0

    outlier_score = compute_local_outlier_score(
        C_values,
        window=outlier_window,
    )

    bounce_mask = compute_tau_bounce_mask(
        tau_values,
        window=bounce_window,
        local_window=max(3, outlier_window),
        zero_fraction=0.05,
    )

    r_outlier = np.ones(n, dtype=float)

    bad_outlier = (
        np.isfinite(outlier_score)
        & (outlier_score > outlier_threshold)
        & (~bounce_mask)
    )

    r_outlier[bad_outlier] = np.exp(
        -outlier_alpha
        * (outlier_score[bad_outlier] - outlier_threshold)
    )

    r_outlier[~np.isfinite(r_outlier)] = 0.0

    reliability = reliability * r_domain * r_det * r_outlier

    reliability[~valid] = 0.0
    reliability[~np.isfinite(reliability)] = 0.0

    reliability = np.clip(
        reliability,
        0.0,
        1.0,
    )

    diagnostics = {
        "x1": x1,
        "x2": x2,
        "sqrt_W": sqrt_W,
        "h_margin": h_margin,
        "det_quality": det_quality,
        "outlier_score": outlier_score,
        "bounce_mask": bounce_mask,
        "r_domain": r_domain,
        "r_det": r_det,
        "r_outlier": r_outlier,
    }

    return valid, reliability, diagnostics


# ============================================================
# FINE-GRID TAU DISCONTINUITY FILTER
# ============================================================

def detect_first_tau_discontinuity(
    t_values,
    tau_values,
    local_window=5,
    local_jump_factor=50.0,
    min_time=None,
    ignore_first_points=5,
    epsilon=1e-300,
):
    """
    Detects the first local one-step discontinuity in fine-grid tau(t).

    For each fine-grid step:

        step[k] = |tau[k] - tau[k-1]|

    the function compares step[k] with neighboring steps:

        local_reference = median of the steps at left and right,
                          excluding step[k] itself.

    A discontinuity is detected if:

        step[k] > local_jump_factor * local_reference

    This is intentionally applied to the fine-grid tau used in the tau plot,
    not to tau_on_noise_times.
    """

    t_values = np.asarray(t_values, dtype=float)
    tau_values = np.asarray(tau_values, dtype=float)

    if len(t_values) != len(tau_values):
        raise ValueError("t_values and tau_values must have the same length.")

    if min_time is None:
        min_time = 0.0

    if len(tau_values) < 2 * local_window + 3:
        return None, None, None

    steps = np.abs(np.diff(tau_values))
    step_times = t_values[1:]

    n_steps = len(steps)

    for step_idx in range(n_steps):
        tau_idx = step_idx + 1

        if step_idx < ignore_first_points:
            continue

        if step_times[step_idx] < min_time:
            continue

        current_step = steps[step_idx]

        if not np.isfinite(current_step):
            return tau_idx, t_values[tau_idx], np.inf

        left_start = max(0, step_idx - local_window)
        left_end = step_idx

        right_start = step_idx + 1
        right_end = min(n_steps, step_idx + local_window + 1)

        left_steps = steps[left_start:left_end]
        right_steps = steps[right_start:right_end]

        local_steps = np.concatenate([left_steps, right_steps])
        local_steps = local_steps[np.isfinite(local_steps)]

        if len(local_steps) < max(3, local_window):
            continue

        local_reference = np.nanmedian(local_steps)

        if not np.isfinite(local_reference) or local_reference <= 0.0:
            local_reference = np.nanpercentile(local_steps, 75)

        if not np.isfinite(local_reference) or local_reference <= 0.0:
            local_reference = epsilon

        jump_score = current_step / (local_reference + epsilon)

        if jump_score > local_jump_factor:
            return tau_idx, t_values[tau_idx], jump_score

    return None, None, None


# ============================================================
# MAIN SIMULATION PIPELINE
# ============================================================

def run_simulation(
    params,
    t_max=40.0,
    dt=0.0005,
    n_noise_times=200,
    n_integral_points=400,
    rtol=1e-8,
    atol=1e-10,
    h_tolerance=1e-12,
    h_margin_scale=1e-6,
    determinant_quality_scale=1e-10,
    outlier_window=5,
    outlier_threshold=4.0,
    outlier_alpha=0.5,
    bounce_window=1,
    tau_jump_filter=True,
    tau_jump_local_window=5,
    tau_jump_local_factor=100.0,
):
    """
    Full numerical pipeline:

        params -> Green functions -> fine-grid tau(t) -> W(t) -> C_E(t)

    The tau discontinuity filter is applied to the fine-grid tau used for
    the tau plot. If a discontinuity is detected, all later coarse C_E/W
    points are marked invalid.
    """

    # ------------------------------------------------------------
    # 1. Solve Green functions
    # ------------------------------------------------------------

    t, y = solve_green_functions(
        params,
        t_max=t_max,
        dt=dt,
        rtol=rtol,
        atol=atol,
    )

    t_full, y_full = build_full_green_arrays(t, y)
    green_interp = make_green_interpolators(t_full, y_full)

    # ------------------------------------------------------------
    # 2. Compute tau(t) on the fine Green-function grid
    # ------------------------------------------------------------

    tau_full = compute_tau_values(
        t_full,
        green_interp,
        params,
    )

    tau = tau_full[1:]

    # ------------------------------------------------------------
    # 3. Compute W(t) on the coarse/noise grid
    # ------------------------------------------------------------

    noise_times = np.linspace(
        0.0,
        t_max,
        n_noise_times,
    )

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

    # ------------------------------------------------------------
    # 4. Compute tau and capacity on the coarse/noise grid
    # ------------------------------------------------------------

    tau_on_noise_times = compute_tau_values(
        noise_times,
        green_interp,
        params,
    )

    C_values = compute_capacity_E(
        tau_on_noise_times,
        W_values,
        params,
    )

    ratio_tau_sqrtW = np.full_like(
        W_values,
        np.nan,
        dtype=float,
    )

    mask_ratio = W_values > 0.0

    ratio_tau_sqrtW[mask_ratio] = (
        np.abs(tau_on_noise_times[mask_ratio])
        / np.sqrt(W_values[mask_ratio])
    )

    # ------------------------------------------------------------
    # 5. Capacity-input diagnostics
    # ------------------------------------------------------------

    diagnostics = diagnose_capacity_inputs(
        noise_times,
        tau_on_noise_times,
        W_values,
        C_values,
        params,
    )

    # ------------------------------------------------------------
    # 6. Reliability score before tau-jump truncation
    # ------------------------------------------------------------

    valid_C, reliability, reliability_diagnostics = compute_reliability_scores(
        noise_times=noise_times,
        tau_values=tau_on_noise_times,
        W_values=W_values,
        C_values=C_values,
        N_values=N_values,
        params=params,
        h_tolerance=h_tolerance,
        h_margin_scale=h_margin_scale,
        determinant_quality_scale=determinant_quality_scale,
        outlier_window=outlier_window,
        outlier_threshold=outlier_threshold,
        outlier_alpha=outlier_alpha,
        bounce_window=bounce_window,
    )

    # ------------------------------------------------------------
    # 7. Fine-grid tau local-step discontinuity filter
    # ------------------------------------------------------------

    tau_jump_idx = None
    tau_jump_time = None
    tau_jump_score = None

    if tau_jump_filter:
        tau_jump_idx, tau_jump_time, tau_jump_score = detect_first_tau_discontinuity(
            t_values=t,
            tau_values=tau,
            local_window=tau_jump_local_window,
            local_jump_factor=tau_jump_local_factor,
            min_time=params.d,
            ignore_first_points=5,
        )

        if tau_jump_time is not None:
            corrupted_mask = noise_times >= tau_jump_time

            valid_C[corrupted_mask] = False
            reliability[corrupted_mask] = 0.0

    # ------------------------------------------------------------
    # 8. Recompute summary quantities AFTER tau-jump filter
    # ------------------------------------------------------------

    valid_fraction = float(np.mean(valid_C))
    mean_reliability = float(np.nanmean(reliability))

    valid_for_max = valid_C & np.isfinite(C_values)

    if np.any(valid_for_max):
        valid_indices = np.where(valid_for_max)[0]
        local_max_idx = np.nanargmax(C_values[valid_for_max])
        max_idx = valid_indices[local_max_idx]

        max_C = C_values[max_idx]
        t_max_C = noise_times[max_idx]
        delay_max_C = t_max_C - params.d

    else:
        max_idx = None
        max_C = np.nan
        t_max_C = np.nan
        delay_max_C = np.nan

    # ------------------------------------------------------------
    # 9. Return outputs
    # ------------------------------------------------------------

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
        "diagnostics": diagnostics,
        "valid_C": valid_C,
        "reliability": reliability,
        "reliability_diagnostics": reliability_diagnostics,
        "valid_fraction": valid_fraction,
        "mean_reliability": mean_reliability,
        "tau_jump_filter": tau_jump_filter,
        "tau_jump_detected": tau_jump_time is not None,
        "tau_jump_idx": tau_jump_idx,
        "tau_jump_time": tau_jump_time,
        "tau_jump_score": tau_jump_score,
        "tau_jump_local_window": tau_jump_local_window,
        "tau_jump_local_factor": tau_jump_local_factor,
    }