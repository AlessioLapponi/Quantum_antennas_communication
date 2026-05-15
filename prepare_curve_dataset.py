from pathlib import Path

import numpy as np
import pandas as pd

from src.simulation import ChannelParams
from src.bounce_detection import (
    compute_fine_tau_zero_crossings_for_params,
    build_bounce_mask_from_zero_times,
)


# ============================================================
# PATHS
# ============================================================

BATCH_DIR = Path("outputs/training_data_full/batches")
OUTPUT_DIR = Path("outputs/curve_dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "curve_dataset.npz"
SUMMARY_FILE = OUTPUT_DIR / "curve_dataset_summary.csv"


# ============================================================
# PHYSICAL / NUMERICAL SETTINGS
# ============================================================

M_A = 1.0
M_B = 1.0
SIGMA = 0.01
D = 1.0
E = 100.0

T_MAX = 40.0
DT_FINE_TAU = 0.0005
RTOL = 1e-8
ATOL = 1e-10

BOUNCE_PROTECTION_TIME = 0.25


# ============================================================
# DATA LOADING
# ============================================================

def load_all_batches():
    files = sorted(BATCH_DIR.glob("training_data_batch_*.csv"))

    if not files:
        raise FileNotFoundError(f"No training batch files found in {BATCH_DIR}")

    dfs = []

    for file in files:
        print(f"Loading {file}")
        dfs.append(pd.read_csv(file))

    return pd.concat(dfs, ignore_index=True)


def fill_invalid_curve_values(y, valid_mask):
    """
    Fills invalid C_E values only for array storage / PCA compatibility.

    The validity mask is still saved separately and must be used in training.
    """

    y = np.asarray(y, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    y_filled = y.copy()

    finite_valid = valid_mask & np.isfinite(y)

    if np.sum(finite_valid) == 0:
        return np.zeros_like(y_filled)

    indices = np.arange(len(y))

    y_filled[~finite_valid] = np.interp(
        indices[~finite_valid],
        indices[finite_valid],
        y[finite_valid],
    )

    y_filled = np.maximum(y_filled, 0.0)

    return y_filled


# ============================================================
# MAIN
# ============================================================

def main():
    df = load_all_batches()

    sample_ids = sorted(df["sample_id"].unique())

    X_params = []
    Y_curves = []
    Y_valid = []
    R_curves = []
    B_curves = []
    old_B_curves = []

    sample_id_list = []
    split_list = []
    sample_type_list = []

    tau_zero_counts = []
    valid_fractions = []
    mean_reliabilities = []

    t_grid_ref = None

    print()
    print("Building curve dataset")
    print("-" * 60)
    print(f"Number of curves: {len(sample_ids)}")
    print("-" * 60)

    for idx, sample_id in enumerate(sample_ids, start=1):
        curve_df = df[df["sample_id"] == sample_id].sort_values("t").copy()

        first = curve_df.iloc[0]

        print(f"[{idx}/{len(sample_ids)}] sample_id={sample_id}")

        t_grid = curve_df["t"].to_numpy(dtype=float)

        if t_grid_ref is None:
            t_grid_ref = t_grid.copy()
        else:
            if len(t_grid) != len(t_grid_ref) or not np.allclose(t_grid, t_grid_ref):
                raise ValueError(f"Inconsistent t grid for sample_id={sample_id}")

        gamma_A = float(first["gamma_A"])
        gamma_B = float(first["gamma_B"])
        omega_A = float(first["omega_A"])
        omega_B = float(first["omega_B"])

        gamma_mean = float(first["gamma_mean"])
        gamma_delta = float(first["gamma_delta"])
        omega_mean = float(first["omega_mean"])
        omega_delta = float(first["omega_delta"])

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

        y_raw = curve_df["C_E"].to_numpy(dtype=float)
        valid_mask = curve_df["valid_C"].to_numpy(dtype=bool)
        reliability = curve_df["reliability"].to_numpy(dtype=float)

        old_bounce_mask = curve_df["bounce_mask"].to_numpy(dtype=bool)

        y_filled = fill_invalid_curve_values(
            y=y_raw,
            valid_mask=valid_mask,
        )

        # Hard physical constraints for stored target
        y_filled[t_grid < D] = 0.0
        y_filled = np.maximum(y_filled, 0.0)

        fine_tau_result = compute_fine_tau_zero_crossings_for_params(
            params=params,
            t_max=T_MAX,
            dt=DT_FINE_TAU,
            rtol=RTOL,
            atol=ATOL,
            min_time=D,
        )

        zero_times = fine_tau_result["tau_zero_times"]

        bounce_mask_fine = build_bounce_mask_from_zero_times(
            coarse_times=t_grid,
            zero_times=zero_times,
            protection_time=BOUNCE_PROTECTION_TIME,
        )

        X_params.append([
            gamma_mean,
            gamma_delta,
            omega_mean,
            omega_delta,
        ])

        Y_curves.append(y_filled)
        Y_valid.append(valid_mask)
        R_curves.append(reliability)
        B_curves.append(bounce_mask_fine)
        old_B_curves.append(old_bounce_mask)

        sample_id_list.append(sample_id)
        split_list.append(first["split"])
        sample_type_list.append(first["sample_type"])

        tau_zero_counts.append(len(zero_times))
        valid_fractions.append(np.mean(valid_mask))
        mean_reliabilities.append(np.nanmean(reliability))

    X_params = np.asarray(X_params, dtype=np.float64)
    Y_curves = np.asarray(Y_curves, dtype=np.float64)
    Y_valid = np.asarray(Y_valid, dtype=bool)
    R_curves = np.asarray(R_curves, dtype=np.float64)
    B_curves = np.asarray(B_curves, dtype=bool)
    old_B_curves = np.asarray(old_B_curves, dtype=bool)

    sample_id_array = np.asarray(sample_id_list)
    split_array = np.asarray(split_list)
    sample_type_array = np.asarray(sample_type_list)

    np.savez_compressed(
        OUTPUT_FILE,
        X_params=X_params,
        Y_curves=Y_curves,
        Y_valid=Y_valid,
        R_curves=R_curves,
        B_curves=B_curves,
        old_B_curves=old_B_curves,
        t_grid=t_grid_ref,
        sample_id=sample_id_array,
        split=split_array,
        sample_type=sample_type_array,
        feature_names=np.array([
            "gamma_mean",
            "gamma_delta",
            "omega_mean",
            "omega_delta",
        ]),
    )

    summary_df = pd.DataFrame({
        "sample_id": sample_id_array,
        "split": split_array,
        "sample_type": sample_type_array,
        "tau_zero_count": tau_zero_counts,
        "valid_fraction": valid_fractions,
        "mean_reliability": mean_reliabilities,
        "old_bounce_points": old_B_curves.sum(axis=1),
        "fine_bounce_points": B_curves.sum(axis=1),
    })

    summary_df.to_csv(SUMMARY_FILE, index=False)

    print()
    print("Curve dataset saved.")
    print("-" * 60)
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Summary file: {SUMMARY_FILE}")
    print()
    print("Shapes:")
    print("X_params:", X_params.shape)
    print("Y_curves:", Y_curves.shape)
    print("Y_valid:", Y_valid.shape)
    print("R_curves:", R_curves.shape)
    print("B_curves:", B_curves.shape)


if __name__ == "__main__":
    main()