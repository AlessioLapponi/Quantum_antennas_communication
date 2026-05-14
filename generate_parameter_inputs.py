import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import qmc
from scipy.spatial import cKDTree


# ============================================================
# PARAMETER DOMAIN
# ============================================================

GAMMA_MIN = 0.001
GAMMA_MAX = 0.035
DELTA_GAMMA_MAX = 0.005

OMEGA_MIN = 0.1
OMEGA_MAX = 1.5
DELTA_OMEGA_MAX = 0.3


# ============================================================
# DATASET SIZE
# ============================================================

N_TOTAL = 2000

FRACTION_GENERAL = 0.70
FRACTION_BOUNDARY = 0.20
FRACTION_DIAGONAL = 0.10

N_GENERAL = int(N_TOTAL * FRACTION_GENERAL)
N_BOUNDARY = int(N_TOTAL * FRACTION_BOUNDARY)
N_DIAGONAL = N_TOTAL - N_GENERAL - N_BOUNDARY


# ============================================================
# SAMPLING SETTINGS
# ============================================================

SEED = 42

# Minimum distance in normalized transformed space:
# (gamma_mean, gamma_delta, omega_mean, omega_delta)
MIN_DISTANCE = 0.02

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "parameter_inputs.csv"


# ============================================================
# PARAMETER TRANSFORMATIONS
# ============================================================

def from_mean_delta(gamma_mean, gamma_delta, omega_mean, omega_delta):
    """
    Converts transformed parameters to physical parameters.

    gamma_delta = gamma_B - gamma_A
    omega_delta = omega_B - omega_A
    """

    gamma_A = gamma_mean - 0.5 * gamma_delta
    gamma_B = gamma_mean + 0.5 * gamma_delta

    omega_A = omega_mean - 0.5 * omega_delta
    omega_B = omega_mean + 0.5 * omega_delta

    return gamma_A, gamma_B, omega_A, omega_B


def is_valid_physical_point(gamma_A, gamma_B, omega_A, omega_B):
    """
    Enforces the hard physical/domain boundaries.
    """

    return (
        (gamma_A >= GAMMA_MIN)
        and (gamma_A <= GAMMA_MAX)
        and (gamma_B >= GAMMA_MIN)
        and (gamma_B <= GAMMA_MAX)
        and (omega_A >= OMEGA_MIN)
        and (omega_A <= OMEGA_MAX)
        and (omega_B >= OMEGA_MIN)
        and (omega_B <= OMEGA_MAX)
        and (abs(gamma_A - gamma_B) <= DELTA_GAMMA_MAX)
        and (abs(omega_A - omega_B) <= DELTA_OMEGA_MAX)
    )


def normalize_transformed_point(gamma_mean, gamma_delta, omega_mean, omega_delta):
    """
    Normalizes transformed parameters to [0,1]-like coordinates
    for distance checks.

    The normalized vector is:
        gamma_mean_norm
        gamma_delta_norm
        omega_mean_norm
        omega_delta_norm
    """

    gamma_mean_norm = (gamma_mean - GAMMA_MIN) / (GAMMA_MAX - GAMMA_MIN)
    gamma_delta_norm = (gamma_delta + DELTA_GAMMA_MAX) / (2.0 * DELTA_GAMMA_MAX)

    omega_mean_norm = (omega_mean - OMEGA_MIN) / (OMEGA_MAX - OMEGA_MIN)
    omega_delta_norm = (omega_delta + DELTA_OMEGA_MAX) / (2.0 * DELTA_OMEGA_MAX)

    return np.array([
        gamma_mean_norm,
        gamma_delta_norm,
        omega_mean_norm,
        omega_delta_norm,
    ])


def build_row(sample_id, sample_type, gamma_mean, gamma_delta, omega_mean, omega_delta):
    gamma_A, gamma_B, omega_A, omega_B = from_mean_delta(
        gamma_mean,
        gamma_delta,
        omega_mean,
        omega_delta,
    )

    return {
        "sample_id": sample_id,
        "sample_type": sample_type,
        "gamma_A": gamma_A,
        "gamma_B": gamma_B,
        "omega_A": omega_A,
        "omega_B": omega_B,
        "gamma_mean": gamma_mean,
        "gamma_delta": gamma_delta,
        "omega_mean": omega_mean,
        "omega_delta": omega_delta,
        "abs_gamma_delta": abs(gamma_delta),
        "abs_omega_delta": abs(omega_delta),
    }


# ============================================================
# SOBOL SAMPLERS
# ============================================================

def sobol_unit_points(dimension, n_points, seed):
    """
    Generates Sobol points in [0,1]^dimension.

    Uses a power-of-two pool larger than n_points, then truncates.
    """

    m = int(np.ceil(np.log2(max(n_points, 2))))
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=seed)
    points = sampler.random_base2(m=m)

    return points[:n_points]


def map_general_unit_to_transformed(u):
    """
    Maps 4D unit Sobol samples to transformed parameters.

    u = [u_gamma_mean, u_gamma_delta, u_omega_mean, u_omega_delta]
    """

    gamma_mean = GAMMA_MIN + u[0] * (GAMMA_MAX - GAMMA_MIN)
    gamma_delta = -DELTA_GAMMA_MAX + u[1] * (2.0 * DELTA_GAMMA_MAX)

    omega_mean = OMEGA_MIN + u[2] * (OMEGA_MAX - OMEGA_MIN)
    omega_delta = -DELTA_OMEGA_MAX + u[3] * (2.0 * DELTA_OMEGA_MAX)

    return gamma_mean, gamma_delta, omega_mean, omega_delta


def map_diagonal_unit_to_transformed(u):
    """
    Diagonal samples:
        gamma_A = gamma_B
        omega_A = omega_B

    Therefore:
        gamma_delta = 0
        omega_delta = 0
    """

    gamma_mean = GAMMA_MIN + u[0] * (GAMMA_MAX - GAMMA_MIN)
    gamma_delta = 0.0

    omega_mean = OMEGA_MIN + u[1] * (OMEGA_MAX - OMEGA_MIN)
    omega_delta = 0.0

    return gamma_mean, gamma_delta, omega_mean, omega_delta


def map_boundary_unit_to_transformed(u):
    """
    Boundary-focused samples.

    We deliberately sample near the allowed asymmetry boundaries:
        |gamma_delta| close to DELTA_GAMMA_MAX
        and/or
        |omega_delta| close to DELTA_OMEGA_MAX

    u has 4 dimensions.
    """

    gamma_mean = GAMMA_MIN + u[0] * (GAMMA_MAX - GAMMA_MIN)
    omega_mean = OMEGA_MIN + u[1] * (OMEGA_MAX - OMEGA_MIN)

    # Choose signs from u[2], u[3]
    gamma_sign = -1.0 if u[2] < 0.5 else 1.0
    omega_sign = -1.0 if u[3] < 0.5 else 1.0

    # Boundary band: 70%-100% of max delta
    gamma_edge_fraction = 0.70 + 0.30 * ((2.0 * u[2]) % 1.0)
    omega_edge_fraction = 0.70 + 0.30 * ((2.0 * u[3]) % 1.0)

    gamma_delta = gamma_sign * gamma_edge_fraction * DELTA_GAMMA_MAX
    omega_delta = omega_sign * omega_edge_fraction * DELTA_OMEGA_MAX

    return gamma_mean, gamma_delta, omega_mean, omega_delta


# ============================================================
# MINIMUM-DISTANCE ACCEPTANCE
# ============================================================

def is_far_enough(candidate_norm, accepted_norms, min_distance):
    """
    Checks whether the normalized candidate is at least min_distance
    away from all previously accepted points.
    """

    if len(accepted_norms) == 0:
        return True

    tree = cKDTree(np.array(accepted_norms))
    distance, _ = tree.query(candidate_norm, k=1)

    return distance >= min_distance


def generate_samples(
    n_target,
    sample_type,
    seed,
    accepted_norms,
    starting_sample_id,
    min_distance=MIN_DISTANCE,
    oversampling_factor=20,
):
    """
    Generates accepted samples of a given type:
        general
        boundary
        diagonal

    The global accepted_norms list is shared across all sample types,
    so the minimum-distance check avoids near-duplicates across the whole dataset.
    """

    rows = []
    sample_id = starting_sample_id

    n_candidates = max(1024, n_target * oversampling_factor)

    # Different dimensions depending on type
    if sample_type == "diagonal":
        unit_points = sobol_unit_points(dimension=2, n_points=n_candidates, seed=seed)
    else:
        unit_points = sobol_unit_points(dimension=4, n_points=n_candidates, seed=seed)

    for u in unit_points:
        if len(rows) >= n_target:
            break

        if sample_type == "general":
            gamma_mean, gamma_delta, omega_mean, omega_delta = map_general_unit_to_transformed(u)

        elif sample_type == "boundary":
            gamma_mean, gamma_delta, omega_mean, omega_delta = map_boundary_unit_to_transformed(u)

        elif sample_type == "diagonal":
            gamma_mean, gamma_delta, omega_mean, omega_delta = map_diagonal_unit_to_transformed(u)

        else:
            raise ValueError(f"Unknown sample_type: {sample_type}")

        gamma_A, gamma_B, omega_A, omega_B = from_mean_delta(
            gamma_mean,
            gamma_delta,
            omega_mean,
            omega_delta,
        )

        if not is_valid_physical_point(gamma_A, gamma_B, omega_A, omega_B):
            continue

        candidate_norm = normalize_transformed_point(
            gamma_mean,
            gamma_delta,
            omega_mean,
            omega_delta,
        )

        if not is_far_enough(candidate_norm, accepted_norms, min_distance):
            continue

        rows.append(
            build_row(
                sample_id=sample_id,
                sample_type=sample_type,
                gamma_mean=gamma_mean,
                gamma_delta=gamma_delta,
                omega_mean=omega_mean,
                omega_delta=omega_delta,
            )
        )

        accepted_norms.append(candidate_norm)
        sample_id += 1

    if len(rows) < n_target:
        raise RuntimeError(
            f"Could only generate {len(rows)} / {n_target} samples for type "
            f"'{sample_type}'. Try lowering MIN_DISTANCE or increasing oversampling_factor."
        )

    return rows, sample_id


# ============================================================
# MAIN
# ============================================================

def main():
    accepted_norms = []
    all_rows = []

    sample_id = 0

    print("Generating parameter inputs")
    print("-" * 60)
    print(f"Target total samples: {N_TOTAL}")
    print(f"General samples: {N_GENERAL}")
    print(f"Boundary samples: {N_BOUNDARY}")
    print(f"Diagonal samples: {N_DIAGONAL}")
    print(f"Minimum normalized distance: {MIN_DISTANCE}")
    print("-" * 60)

    general_rows, sample_id = generate_samples(
        n_target=N_GENERAL,
        sample_type="general",
        seed=SEED,
        accepted_norms=accepted_norms,
        starting_sample_id=sample_id,
        min_distance=MIN_DISTANCE,
    )
    all_rows.extend(general_rows)

    boundary_rows, sample_id = generate_samples(
        n_target=N_BOUNDARY,
        sample_type="boundary",
        seed=SEED + 1,
        accepted_norms=accepted_norms,
        starting_sample_id=sample_id,
        min_distance=MIN_DISTANCE,
    )
    all_rows.extend(boundary_rows)

    diagonal_rows, sample_id = generate_samples(
        n_target=N_DIAGONAL,
        sample_type="diagonal",
        seed=SEED + 2,
        accepted_norms=accepted_norms,
        starting_sample_id=sample_id,
        min_distance=MIN_DISTANCE,
    )
    all_rows.extend(diagonal_rows)

    df = pd.DataFrame(all_rows)

    # Final sanity checks
    assert np.all(df["gamma_A"] >= GAMMA_MIN)
    assert np.all(df["gamma_B"] >= GAMMA_MIN)
    assert np.all(df["gamma_A"] <= GAMMA_MAX)
    assert np.all(df["gamma_B"] <= GAMMA_MAX)

    assert np.all(df["omega_A"] >= OMEGA_MIN)
    assert np.all(df["omega_B"] >= OMEGA_MIN)
    assert np.all(df["omega_A"] <= OMEGA_MAX)
    assert np.all(df["omega_B"] <= OMEGA_MAX)

    assert np.all(np.abs(df["gamma_A"] - df["gamma_B"]) <= DELTA_GAMMA_MAX + 1e-15)
    assert np.all(np.abs(df["omega_A"] - df["omega_B"]) <= DELTA_OMEGA_MAX + 1e-15)

    df.to_csv(OUTPUT_FILE, index=False)

    print()
    print("Generation completed.")
    print(f"Saved to: {OUTPUT_FILE}")
    print()
    print("Dataset summary")
    print("-" * 60)
    print(df["sample_type"].value_counts())
    print()
    print(df.describe())


if __name__ == "__main__":
    main()