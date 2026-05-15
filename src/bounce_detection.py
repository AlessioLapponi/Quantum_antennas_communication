import numpy as np

from src.simulation import (
    solve_green_functions,
    build_full_green_arrays,
    make_green_interpolators,
    compute_tau_values,
)


def detect_tau_zero_crossing_times(
    t_values,
    tau_values,
    min_time,
):
    """
    Detects zero crossings of fine-grid tau(t) after causal contact.

    Important:
        - Points with t < min_time are ignored.
        - We do NOT flag points where |tau| is merely small.
        - A zero crossing is detected only when tau changes sign:

              tau[k] * tau[k+1] < 0

    The crossing time is estimated by linear interpolation.
    """

    t_values = np.asarray(t_values, dtype=float)
    tau_values = np.asarray(tau_values, dtype=float)

    if len(t_values) != len(tau_values):
        raise ValueError("t_values and tau_values must have the same length.")

    zero_times = []

    for k in range(len(tau_values) - 1):
        t0 = t_values[k]
        t1 = t_values[k + 1]

        tau0 = tau_values[k]
        tau1 = tau_values[k + 1]

        # Ignore the entire pre-causal region.
        if t1 < min_time:
            continue

        if not (
            np.isfinite(t0)
            and np.isfinite(t1)
            and np.isfinite(tau0)
            and np.isfinite(tau1)
        ):
            continue

        # Skip pairs where one point is still before the causal time.
        if t0 < min_time:
            continue

        # Detect only genuine sign changes.
        if tau0 * tau1 < 0.0:
            denominator = tau1 - tau0

            if denominator == 0.0:
                zero_time = 0.5 * (t0 + t1)
            else:
                zero_time = t0 - tau0 * (t1 - t0) / denominator

            if zero_time >= min_time:
                zero_times.append(zero_time)

    if len(zero_times) == 0:
        return np.array([], dtype=float)

    zero_times = np.array(sorted(zero_times), dtype=float)

    # Remove accidental near-duplicates.
    unique_zero_times = [zero_times[0]]

    for zt in zero_times[1:]:
        if abs(zt - unique_zero_times[-1]) > 1e-8:
            unique_zero_times.append(zt)

    return np.array(unique_zero_times, dtype=float)


def build_bounce_mask_from_zero_times(
    coarse_times,
    zero_times,
    protection_time=0.25,
):
    """
    Builds a coarse-grid bounce mask from fine-grid tau zero-crossing times.

    A coarse C_E point is protected if it lies near a detected tau sign crossing.
    """

    coarse_times = np.asarray(coarse_times, dtype=float)
    zero_times = np.asarray(zero_times, dtype=float)

    bounce_mask = np.zeros_like(coarse_times, dtype=bool)

    if len(zero_times) == 0:
        return bounce_mask

    for zero_time in zero_times:
        bounce_mask |= np.abs(coarse_times - zero_time) <= protection_time

    return bounce_mask


def compute_fine_tau_zero_crossings_for_params(
    params,
    t_max=40.0,
    dt=0.0005,
    rtol=1e-8,
    atol=1e-10,
    min_time=None,
):
    """
    Recomputes only fine-grid tau(t), without W(t), and detects real tau sign crossings.

    This is much cheaper than recomputing the full simulation because it avoids
    the W double integrals.
    """

    if min_time is None:
        min_time = params.d

    t, y = solve_green_functions(
        params=params,
        t_max=t_max,
        dt=dt,
        rtol=rtol,
        atol=atol,
    )

    t_full, y_full = build_full_green_arrays(t, y)
    green_interp = make_green_interpolators(t_full, y_full)

    tau_full = compute_tau_values(
        t_full,
        green_interp,
        params,
    )

    tau = tau_full[1:]

    zero_times = detect_tau_zero_crossing_times(
        t_values=t,
        tau_values=tau,
        min_time=min_time,
    )

    return {
        "t_fine": t,
        "tau_fine": tau,
        "tau_zero_times": zero_times,
    }


def detect_tau_zero_crossings_from_existing_tau(
    t_values,
    tau_values,
    d,
):
    """
    Convenience function when fine-grid tau has already been computed.

    Use this inside the numerical simulation app/results without recomputing tau.
    """

    return detect_tau_zero_crossing_times(
        t_values=t_values,
        tau_values=tau_values,
        min_time=d,
    )