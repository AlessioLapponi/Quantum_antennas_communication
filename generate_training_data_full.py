import numpy as np
import pandas as pd

from pathlib import Path
from src.simulation import ChannelParams, run_simulation


# ============================================================
# INPUT / OUTPUT PATHS
# ============================================================

INPUT_FILE = Path("outputs") / "parameter_inputs.csv"

OUTPUT_DIR = Path("outputs") / "training_data_full"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_DIR = OUTPUT_DIR / "batches"
BATCH_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUTPUT_DIR / "training_data_summary.csv"
SPLIT_FILE = OUTPUT_DIR / "parameter_inputs_with_split.csv"


# ============================================================
# DATASET / BATCH SETTINGS
# ============================================================

BATCH_SIZE = 20
TRAIN_FRACTION = 0.90
SHUFFLE_SEED = 123

# Set to None for full generation.
# Use a small integer for debugging, e.g. 40.
MAX_SAMPLES = None


# ============================================================
# FIXED PHYSICAL PARAMETERS
# ============================================================

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

def build_long_dataframe(input_row, results, batch_id, split):
    """
    Converts one simulation result into long-format rows.

    One row = one time point on the W/C_E grid.
    """

    noise_times = results["noise_times"]
    rel_diag = results["reliability_diagnostics"]

    df = pd.DataFrame({
        "batch_id": batch_id,
        "split": split,

        "sample_id": input_row["sample_id"],
        "sample_type": input_row.get("sample_type", "unknown"),

        "gamma_A": input_row["gamma_A"],
        "gamma_B": input_row["gamma_B"],
        "omega_A": input_row["omega_A"],
        "omega_B": input_row["omega_B"],

        "gamma_mean": input_row.get("gamma_mean", np.nan),
        "gamma_delta": input_row.get(
            "gamma_delta",
            input_row["gamma_B"] - input_row["gamma_A"],
        ),
        "omega_mean": input_row.get("omega_mean", np.nan),
        "omega_delta": input_row.get(
            "omega_delta",
            input_row["omega_B"] - input_row["omega_A"],
        ),

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

    return df


def build_summary_row(input_row, results, batch_id, split, status="success", error_message=""):
    """
    One summary row per simulation.
    """

    is_result = isinstance(results, dict) and len(results) > 0

    return {
        "batch_id": batch_id,
        "split": split,

        "sample_id": input_row["sample_id"],
        "sample_type": input_row.get("sample_type", "unknown"),

        "gamma_A": input_row["gamma_A"],
        "gamma_B": input_row["gamma_B"],
        "omega_A": input_row["omega_A"],
        "omega_B": input_row["omega_B"],

        "gamma_mean": input_row.get("gamma_mean", np.nan),
        "gamma_delta": input_row.get(
            "gamma_delta",
            input_row["gamma_B"] - input_row["gamma_A"],
        ),
        "omega_mean": input_row.get("omega_mean", np.nan),
        "omega_delta": input_row.get(
            "omega_delta",
            input_row["omega_B"] - input_row["omega_A"],
        ),

        "Sigma2_A": results.get("Sigma2_A", np.nan) if is_result else np.nan,
        "Sigma2_B": results.get("Sigma2_B", np.nan) if is_result else np.nan,

        "max_C": results.get("max_C", np.nan) if is_result else np.nan,
        "t_max_C": results.get("t_max_C", np.nan) if is_result else np.nan,
        "delay_max_C": results.get("delay_max_C", np.nan) if is_result else np.nan,

        "valid_fraction": results.get("valid_fraction", np.nan) if is_result else np.nan,
        "mean_reliability": results.get("mean_reliability", np.nan) if is_result else np.nan,

        "tau_jump_detected": results.get("tau_jump_detected", np.nan) if is_result else np.nan,
        "tau_jump_time": results.get("tau_jump_time", np.nan) if is_result else np.nan,
        "tau_jump_score": results.get("tau_jump_score", np.nan) if is_result else np.nan,

        "status": status,
        "error_message": error_message,
    }


# ============================================================
# SPLIT / BATCH PREPARATION
# ============================================================

def prepare_input_dataframe():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    if MAX_SAMPLES is not None:
        df = df.head(MAX_SAMPLES).copy()

    # Shuffle before splitting, because the input file is structured by sample_type.
    df = df.sample(frac=1.0, random_state=SHUFFLE_SEED).reset_index(drop=True)

    n_total = len(df)
    n_train = int(np.floor(TRAIN_FRACTION * n_total))

    df["split"] = "test"
    df.loc[:n_train - 1, "split"] = "train"

    df["batch_id"] = np.arange(n_total) // BATCH_SIZE

    df.to_csv(SPLIT_FILE, index=False)

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    input_df = prepare_input_dataframe()

    summary_rows = []

    n_total = len(input_df)
    n_batches = int(np.ceil(n_total / BATCH_SIZE))

    print("Starting full training-data generation")
    print("-" * 60)
    print(f"Input file: {INPUT_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total simulations: {n_total}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of batches: {n_batches}")
    print(f"Train fraction: {TRAIN_FRACTION:.0%}")
    print("-" * 60)

    for batch_id in range(n_batches):
        batch_df = input_df[input_df["batch_id"] == batch_id].copy()

        batch_file = BATCH_DIR / f"training_data_batch_{batch_id:04d}.csv"

        # Skip already completed batches.
        if batch_file.exists():
            print(f"Skipping existing batch {batch_id:04d}: {batch_file}")
            continue

        print()
        print("=" * 60)
        print(f"Batch {batch_id + 1}/{n_batches} | batch_id={batch_id}")
        print(f"Simulations in batch: {len(batch_df)}")
        print("=" * 60)

        batch_long_dfs = []

        for local_idx, (_, row) in enumerate(batch_df.iterrows(), start=1):
            global_idx = batch_id * BATCH_SIZE + local_idx

            print()
            print(
                f"[global {global_idx}/{n_total} | batch {local_idx}/{len(batch_df)}] "
                f"sample_id = {row['sample_id']} | split = {row['split']}"
            )
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

                long_df = build_long_dataframe(
                    input_row=row,
                    results=results,
                    batch_id=batch_id,
                    split=row["split"],
                )

                batch_long_dfs.append(long_df)

                summary_rows.append(
                    build_summary_row(
                        input_row=row,
                        results=results,
                        batch_id=batch_id,
                        split=row["split"],
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
                        input_row=row,
                        results={},
                        batch_id=batch_id,
                        split=row["split"],
                        status="failed",
                        error_message=error_message,
                    )
                )

        if batch_long_dfs:
            batch_output_df = pd.concat(batch_long_dfs, ignore_index=True)
        else:
            batch_output_df = pd.DataFrame()

        batch_output_df.to_csv(batch_file, index=False)

        # Save cumulative summary after every batch.
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(SUMMARY_FILE, index=False)

        print()
        print(f"Saved batch to: {batch_file}")
        print(f"Batch rows: {len(batch_output_df)}")
        print(f"Updated summary: {SUMMARY_FILE}")

    print()
    print("Generation completed.")
    print("-" * 60)
    print(f"Batches saved in: {BATCH_DIR}")
    print(f"Summary saved to: {SUMMARY_FILE}")
    print(f"Split input file saved to: {SPLIT_FILE}")


if __name__ == "__main__":
    main()