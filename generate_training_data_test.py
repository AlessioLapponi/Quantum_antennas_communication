import numpy as np
import pandas as pd

from pathlib import Path
from src.simulation import ChannelParams, run_simulation


# ============================================================
# INPUT / OUTPUT PATHS
# ============================================================

INPUT_FILE = Path("outputs") / "parameter_inputs.csv"

OUTPUT_DIR = Path("outputs") / "training_data_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "training_data_test.csv"
SUMMARY_FILE = OUTPUT_DIR / "training_data_test_summary.csv"


# ============================================================
# TEST SETTINGS
# ============================================================

N_TEST_SAMPLES = 10

# Fixed physical parameters not included in the input CSV
M_A = 1.0
M_B = 1.0
SIGMA = 0.01
D = 1.0
E = 100.0


# ============================================================
# NUMERICAL / RELIABILITY PARAMETERS
# ============================================================

SIMULATION_KWARGS = {
    "t_max": 40.0,
    "dt": 0.0005,
    "n_noise_times": 200,
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
# DATA EXTRACTION
# ============================================================

def build_long_dataframe(input_row, results):
    """
    Converts one simulation result into long-format rows.

    One row = one time point on the W/C_E grid.
    """

    noise_times = results["noise_times"]
    n = len(noise_times)

    rel_diag = results["reliability_diagnostics"]
    diagnostics = results["diagnostics"]

    df = pd.DataFrame({
        "sample_id": input_row["sample_id"],
        "sample_type": input_row.get("sample_type", "unknown"),

        "gamma_A": input_row["gamma_A"],
        "gamma_B": input_row["gamma_B"],
        "omega_A": input_row["omega_A"],
        "omega_B": input_row["omega_B"],

        "gamma_mean": input_row.get("gamma_mean", np.nan),
        "gamma_delta": input_row.get("gamma_delta", input_row["gamma_B"] - input_row["gamma_A"]),
        "omega_mean": input_row.get("omega_mean", np.nan),
        "omega_delta": input_row.get("omega_delta", input_row["omega_B"] - input_row["omega_A"]),

        "Sigma2_A": results["Sigma2_A"],
        "Sigma2_B": results["Sigma2_B"],

        "t": noise_times,
        "tau": results["tau_on_noise_times"],
        "W": results["W_values"],
        "C_E": results["C_values"],
        "ratio_tau_sqrtW": results["ratio_tau_sqrtW"],

        "valid_C": results["valid_C"],
        "reliability": results["reliability"],

        "x1": rel_diag["x1"],
        "x2": rel_diag["x2"],
        "sqrt_W": rel_diag["sqrt_W"],
        "h_margin": rel_diag["h_margin"],
        "det_quality": rel_diag["det_quality"],
        "outlier_score": rel_diag["outlier_score"],
        "bounce_mask": rel_diag["bounce_mask"],
        "r_domain": rel_diag["r_domain"],
        "r_det": rel_diag["r_det"],
        "r_outlier": rel_diag["r_outlier"],

        "tau_jump_detected": results["tau_jump_detected"],
        "tau_jump_time": results["tau_jump_time"],
        "tau_jump_score": results["tau_jump_score"],

        "max_C": results["max_C"],
        "t_max_C": results["t_max_C"],
        "delay_max_C": results["delay_max_C"],

        "valid_fraction": results["valid_fraction"],
        "mean_reliability": results["mean_reliability"],
    })

    assert len(df) == n

    return df


def build_summary_row(input_row, results, status="success", error_message=""):
    """
    One summary row per simulation.
    """

    return {
        "sample_id": input_row["sample_id"],
        "sample_type": input_row.get("sample_type", "unknown"),

        "gamma_A": input_row["gamma_A"],
        "gamma_B": input_row["gamma_B"],
        "omega_A": input_row["omega_A"],
        "omega_B": input_row["omega_B"],

        "gamma_mean": input_row.get("gamma_mean", np.nan),
        "gamma_delta": input_row.get("gamma_delta", input_row["gamma_B"] - input_row["gamma_A"]),
        "omega_mean": input_row.get("omega_mean", np.nan),
        "omega_delta": input_row.get("omega_delta", input_row["omega_B"] - input_row["omega_A"]),

        "Sigma2_A": results.get("Sigma2_A", np.nan) if isinstance(results, dict) else np.nan,
        "Sigma2_B": results.get("Sigma2_B", np.nan) if isinstance(results, dict) else np.nan,

        "max_C": results.get("max_C", np.nan) if isinstance(results, dict) else np.nan,
        "t_max_C": results.get("t_max_C", np.nan) if isinstance(results, dict) else np.nan,
        "delay_max_C": results.get("delay_max_C", np.nan) if isinstance(results, dict) else np.nan,

        "valid_fraction": results.get("valid_fraction", np.nan) if isinstance(results, dict) else np.nan,
        "mean_reliability": results.get("mean_reliability", np.nan) if isinstance(results, dict) else np.nan,

        "tau_jump_detected": results.get("tau_jump_detected", np.nan) if isinstance(results, dict) else np.nan,
        "tau_jump_time": results.get("tau_jump_time", np.nan) if isinstance(results, dict) else np.nan,
        "tau_jump_score": results.get("tau_jump_score", np.nan) if isinstance(results, dict) else np.nan,

        "status": status,
        "error_message": error_message,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    input_df = pd.read_csv(INPUT_FILE)

    if N_TEST_SAMPLES is not None:
        input_df = input_df.head(N_TEST_SAMPLES).copy()

    all_long_dfs = []
    summary_rows = []

    print("Starting training-data test generation")
    print("-" * 60)
    print(f"Input file: {INPUT_FILE}")
    print(f"Number of test simulations: {len(input_df)}")
    print(f"Output file: {OUTPUT_FILE}")
    print("-" * 60)

    for idx, row in input_df.iterrows():
        print()
        print(f"[{idx + 1}/{len(input_df)}] sample_id = {row['sample_id']}")
        print(
            f"gamma_A={row['gamma_A']:.6g}, gamma_B={row['gamma_B']:.6g}, "
            f"omega_A={row['omega_A']:.6g}, omega_B={row['omega_B']:.6g}"
        )

        params = ChannelParams(
            gamma_A=float(row["gamma_A"]),
            gamma_B=float(row["gamma_B"]),
            omega_A=float(row["omega_A"]),
            omega_B=float(row["omega_B"]),
            sigma=SIGMA,
            d=D,
            m_A=M_A,
            m_B=M_B,
            E=E,
        )

        try:
            results = run_simulation(
                params=params,
                **SIMULATION_KWARGS,
            )

            long_df = build_long_dataframe(row, results)
            all_long_dfs.append(long_df)

            summary_rows.append(
                build_summary_row(
                    row,
                    results,
                    status="success",
                    error_message="",
                )
            )

            print(
                f"success | valid_fraction={results['valid_fraction']:.2%} | "
                f"mean_reliability={results['mean_reliability']:.3f} | "
                f"max_C={results['max_C']:.6g} at t={results['t_max_C']:.6g}"
            )

            if results["tau_jump_detected"]:
                print(
                    f"tau jump detected at t={results['tau_jump_time']:.6g}, "
                    f"score={results['tau_jump_score']:.6g}"
                )

        except Exception as exc:
            error_message = str(exc)

            print(f"FAILED: {error_message}")

            summary_rows.append(
                build_summary_row(
                    row,
                    results={},
                    status="failed",
                    error_message=error_message,
                )
            )

    if all_long_dfs:
        training_df = pd.concat(all_long_dfs, ignore_index=True)
    else:
        training_df = pd.DataFrame()

    summary_df = pd.DataFrame(summary_rows)

    training_df.to_csv(OUTPUT_FILE, index=False)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    print()
    print("Test generation completed.")
    print("-" * 60)
    print(f"Saved long-format data to: {OUTPUT_FILE}")
    print(f"Saved summary data to: {SUMMARY_FILE}")
    print()
    print("Long-format dataset shape:", training_df.shape)
    print("Summary dataset shape:", summary_df.shape)

    if not training_df.empty:
        print()
        print("Training data quick diagnostics")
        print("-" * 60)
        print("valid_C fraction:", training_df["valid_C"].mean())
        print("mean reliability:", training_df["reliability"].mean())
        print("finite C_E fraction:", np.isfinite(training_df["C_E"]).mean())


if __name__ == "__main__":
    main()