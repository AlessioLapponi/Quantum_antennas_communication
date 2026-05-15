from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

CURVE_DATASET_FILE = Path("outputs/curve_dataset/curve_dataset.npz")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "pca_ml_curve_model.joblib"
SCALER_PATH = MODEL_DIR / "pca_ml_feature_scaler.joblib"
PCA_PATH = MODEL_DIR / "pca_ml_basis.joblib"

METRICS_PATH = MODEL_DIR / "pca_ml_metrics.json"
PREDICTIONS_PATH = MODEL_DIR / "pca_ml_test_predictions.npz"

PCA_VARIANCE_PLOT_PATH = MODEL_DIR / "pca_ml_explained_variance.png"
EXAMPLE_CURVES_PLOT_PATH = MODEL_DIR / "pca_ml_example_curves.png"

PER_CURVE_ERRORS_PATH = MODEL_DIR / "pca_ml_per_curve_errors.csv"
PER_CURVE_ERROR_HIST_PATH = MODEL_DIR / "pca_ml_per_curve_rmse_hist.png"
WORST_CURVES_PLOT_PATH = MODEL_DIR / "pca_ml_worst_curves.png"


# ============================================================
# SETTINGS
# ============================================================

N_COMPONENTS = 80

MIN_VALID_FRACTION = 0.50
MIN_MEAN_RELIABILITY = 0.05

RANDOM_SEED = 42

# Hard physical settings
D = 1.0

CURVE_RMSE_THRESHOLDS = [0.01, 0.02, 0.05, 0.1]
RELATIVE_RMSE_THRESHOLDS = [0.05, 0.10, 0.20, 0.50]


# ============================================================
# HELPERS
# ============================================================

def load_curve_dataset(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Curve dataset not found: {path}")

    data = np.load(path, allow_pickle=True)

    return {
        "X_params": data["X_params"],
        "Y_curves": data["Y_curves"],
        "Y_valid": data["Y_valid"],
        "R_curves": data["R_curves"],
        "B_curves": data["B_curves"],
        "t_grid": data["t_grid"],
        "sample_id": data["sample_id"],
        "split": data["split"],
        "sample_type": data["sample_type"],
        "feature_names": data["feature_names"],
    }


def compute_curve_weights(Y_valid, R_curves):
    """
    One weight per curve.

    Curves with many valid/reliable points get higher weight.
    """

    valid = Y_valid.astype(bool)
    reliability = np.asarray(R_curves, dtype=float)

    weighted_reliability = np.where(valid, reliability, 0.0)

    valid_counts = np.sum(valid, axis=1)
    reliability_sums = np.sum(weighted_reliability, axis=1)

    curve_weights = np.zeros(len(valid_counts), dtype=float)

    mask = valid_counts > 0
    curve_weights[mask] = reliability_sums[mask] / valid_counts[mask]

    return curve_weights


def apply_physical_postprocessing(Y_pred, t_grid, d=D):
    """
    Enforces basic physical constraints on reconstructed curves:

        C_E >= 0
        C_E(t < d) = 0
    """

    Y_pred = np.asarray(Y_pred, dtype=float).copy()

    Y_pred = np.maximum(Y_pred, 0.0)
    Y_pred[:, t_grid < d] = 0.0

    return Y_pred


def curve_level_metrics(Y_true, Y_pred):
    """
    Computes pointwise global metrics and curve-level averaged metrics.
    """

    y_true_flat = Y_true.reshape(-1)
    y_pred_flat = Y_pred.reshape(-1)

    mae = mean_absolute_error(y_true_flat, y_pred_flat)

    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = float(np.sqrt(mse))

    r2 = r2_score(y_true_flat, y_pred_flat)

    per_curve_mae = np.mean(np.abs(Y_true - Y_pred), axis=1)
    per_curve_rmse = np.sqrt(np.mean((Y_true - Y_pred) ** 2, axis=1))

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mean_curve_mae": float(np.mean(per_curve_mae)),
        "median_curve_mae": float(np.median(per_curve_mae)),
        "mean_curve_rmse": float(np.mean(per_curve_rmse)),
        "median_curve_rmse": float(np.median(per_curve_rmse)),
    }


def max_capacity_metrics(Y_true, Y_pred, t_grid):
    """
    Compares max C_E and time of max C_E per curve.
    """

    true_max_idx = np.nanargmax(Y_true, axis=1)
    pred_max_idx = np.nanargmax(Y_pred, axis=1)

    true_max = Y_true[np.arange(len(Y_true)), true_max_idx]
    pred_max = Y_pred[np.arange(len(Y_pred)), pred_max_idx]

    true_tmax = t_grid[true_max_idx]
    pred_tmax = t_grid[pred_max_idx]

    max_mae = np.mean(np.abs(true_max - pred_max))
    tmax_mae = np.mean(np.abs(true_tmax - pred_tmax))

    return {
        "max_C_mae": float(max_mae),
        "t_max_C_mae": float(tmax_mae),
    }


def plot_per_curve_rmse_hist(per_curve_df, output_path):
    plt.figure(figsize=(8, 5))
    plt.hist(per_curve_df["rmse"], bins=30)
    plt.xlabel("Per-curve RMSE")
    plt.ylabel("Number of test curves")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_worst_curves(t_grid, Y_true, Y_pred, per_curve_df, output_path, n_worst=8):
    worst = per_curve_df.sort_values("rmse", ascending=False).head(n_worst)
    indices = worst["curve_index"].to_numpy(dtype=int)

    n_cols = 2
    n_rows = int(np.ceil(len(indices) / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 3.5 * n_rows),
        squeeze=False,
    )

    for ax, idx in zip(axes.ravel(), indices):
        ax.plot(t_grid, Y_true[idx], label="true")
        ax.plot(t_grid, Y_pred[idx], linestyle="--", label="predicted")

        rmse_value = per_curve_df.loc[
            per_curve_df["curve_index"] == idx,
            "rmse",
        ].iloc[0]

        title = f"curve {idx} | RMSE={rmse_value:.4g}"

        if "sample_id" in per_curve_df.columns:
            sample_id = per_curve_df.loc[
                per_curve_df["curve_index"] == idx,
                "sample_id",
            ].iloc[0]
            title += f" | sample_id={sample_id}"

        ax.set_title(title)
        ax.set_xlabel("t")
        ax.set_ylabel(r"$C_E$")
        ax.grid(True)
        ax.legend()

    for ax in axes.ravel()[len(indices):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_explained_variance(pca, output_path):
    """
    Plots remaining unexplained PCA variance:

        1 - cumulative explained variance

    on a logarithmic y-scale.
    """

    cumulative = np.cumsum(pca.explained_variance_ratio_)
    remaining = 1.0 - cumulative

    epsilon = 1e-16
    remaining = np.maximum(remaining, epsilon)

    plt.figure(figsize=(8, 5))
    plt.plot(
        np.arange(1, len(cumulative) + 1),
        remaining,
        marker="o",
    )
    plt.xlabel("Number of PCA components")
    plt.ylabel(r"$1 -$ cumulative explained variance")
    plt.yscale("log")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_example_curves(t_grid, Y_true, Y_pred, output_path, n_examples=8):
    rng = np.random.default_rng(RANDOM_SEED)

    n_available = len(Y_true)
    n_examples = min(n_examples, n_available)

    indices = rng.choice(n_available, size=n_examples, replace=False)

    n_cols = 2
    n_rows = int(np.ceil(n_examples / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 3.5 * n_rows),
        squeeze=False,
    )

    for ax, idx in zip(axes.ravel(), indices):
        ax.plot(t_grid, Y_true[idx], label="true")
        ax.plot(t_grid, Y_pred[idx], linestyle="--", label="PCA-ML")
        ax.axvline(D, linestyle=":", alpha=0.7)
        ax.set_title(f"Test curve index {idx}")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$C_E$")
        ax.grid(True)
        ax.legend()

    for ax in axes.ravel()[n_examples:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def per_curve_error_metrics(Y_true, Y_pred, sample_ids=None, sample_types=None):
    """
    Computes MAE, MSE, RMSE, max absolute error, and relative errors per test curve.
    """

    errors = Y_pred - Y_true

    per_curve_mse = np.mean(errors**2, axis=1)
    per_curve_rmse = np.sqrt(per_curve_mse)
    per_curve_mae = np.mean(np.abs(errors), axis=1)
    per_curve_max_abs_error = np.max(np.abs(errors), axis=1)

    true_max = np.max(Y_true, axis=1)
    pred_max = np.max(Y_pred, axis=1)

    max_C_abs_error = np.abs(pred_max - true_max)

    # Avoid division by zero for curves whose capacity is essentially zero.
    relative_rmse = per_curve_rmse / (true_max + 1e-12)

    df = pd.DataFrame({
        "curve_index": np.arange(len(Y_true)),
        "mse": per_curve_mse,
        "rmse": per_curve_rmse,
        "mae": per_curve_mae,
        "max_abs_error": per_curve_max_abs_error,
        "true_max_C": true_max,
        "pred_max_C": pred_max,
        "max_C_abs_error": max_C_abs_error,
        "relative_rmse": relative_rmse,
    })

    if sample_ids is not None:
        df["sample_id"] = sample_ids

    if sample_types is not None:
        df["sample_type"] = sample_types

    return df


def build_physical_features(X_params, sigma=0.01, d=1.0):
    """
    Builds physics-informed features from:

        gamma_mean, gamma_delta, omega_mean, omega_delta

    The signed deltas are kept because A/B asymmetry is physically relevant.
    Absolute deltas are added only as extra asymmetry-magnitude features.
    """

    gamma_mean = X_params[:, 0]
    gamma_delta = X_params[:, 1]
    omega_mean = X_params[:, 2]
    omega_delta = X_params[:, 3]

    gamma_A = gamma_mean - 0.5 * gamma_delta
    gamma_B = gamma_mean + 0.5 * gamma_delta

    omega_A = omega_mean - 0.5 * omega_delta
    omega_B = omega_mean + 0.5 * omega_delta

    Sigma2_A = np.sqrt(8.0 / np.pi) * gamma_A / sigma - omega_A**2
    Sigma2_B = np.sqrt(8.0 / np.pi) * gamma_B / sigma - omega_B**2

    Sigma2_mean = 0.5 * (Sigma2_A + Sigma2_B)
    Sigma2_delta = Sigma2_B - Sigma2_A

    coupling_delay = 2.0 * np.sqrt(gamma_A * gamma_B) / d

    abs_gamma_delta = np.abs(gamma_delta)
    abs_omega_delta = np.abs(omega_delta)

    gamma_delta_ratio = gamma_delta / (gamma_mean + 1e-12)
    omega_delta_ratio = omega_delta / (omega_mean + 1e-12)

    X_features = np.column_stack([
        gamma_A,
        gamma_B,
        gamma_mean,
        gamma_delta,
        abs_gamma_delta,
        gamma_delta_ratio,

        omega_A,
        omega_B,
        omega_mean,
        omega_delta,
        abs_omega_delta,
        omega_delta_ratio,

        Sigma2_A,
        Sigma2_B,
        Sigma2_mean,
        Sigma2_delta,

        coupling_delay,
    ])

    feature_names = [
        "gamma_A",
        "gamma_B",
        "gamma_mean",
        "gamma_delta",
        "abs_gamma_delta",
        "gamma_delta_ratio",

        "omega_A",
        "omega_B",
        "omega_mean",
        "omega_delta",
        "abs_omega_delta",
        "omega_delta_ratio",

        "Sigma2_A",
        "Sigma2_B",
        "Sigma2_mean",
        "Sigma2_delta",

        "coupling_delay",
    ]

    return X_features, feature_names


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading curve dataset...")
    data = load_curve_dataset(CURVE_DATASET_FILE)

    # ------------------------------------------------------------
    # 1. Load and build physics-informed features
    # ------------------------------------------------------------

    X_raw = data["X_params"]

    X, feature_names = build_physical_features(
        X_raw,
        sigma=0.01,
        d=1.0,
    )

    Y = data["Y_curves"]
    Y_valid = data["Y_valid"]
    R_curves = data["R_curves"]
    t_grid = data["t_grid"]
    split = data["split"]
    sample_id = data["sample_id"]
    sample_type = data["sample_type"]

    curve_weights = compute_curve_weights(Y_valid, R_curves)
    valid_fraction = np.mean(Y_valid, axis=1)

    usable_curve_mask = (
        np.isfinite(Y).all(axis=1)
        & np.isfinite(X).all(axis=1)
        & (valid_fraction >= MIN_VALID_FRACTION)
        & (curve_weights >= MIN_MEAN_RELIABILITY)
    )

    train_mask = (split == "train") & usable_curve_mask
    test_mask = (split == "test") & usable_curve_mask

    X_train = X[train_mask]
    Y_train = Y[train_mask]
    w_train = curve_weights[train_mask]

    X_test = X[test_mask]
    Y_test = Y[test_mask]

    sample_id_test = sample_id[test_mask]
    sample_type_test = sample_type[test_mask]

    print()
    print("Dataset summary")
    print("-" * 60)
    print(f"Total curves: {len(X)}")
    print(f"Usable curves: {int(np.sum(usable_curve_mask))}")
    print(f"Train curves: {len(X_train)}")
    print(f"Test curves: {len(X_test)}")
    print(f"Time points per curve: {Y.shape[1]}")
    print(f"Feature names: {feature_names}")

    if len(X_train) == 0:
        raise ValueError("No training curves available after filtering.")

    if len(X_test) == 0:
        raise ValueError("No test curves available after filtering.")

    # ------------------------------------------------------------
    # 2. Scale physical features
    # ------------------------------------------------------------

    print()
    print("Scaling input parameters...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------
    # 3. Fit PCA on full curves
    # ------------------------------------------------------------

    n_components = min(N_COMPONENTS, len(X_train), Y_train.shape[1])

    print()
    print(f"Using PCA components: {n_components}")

    pca = PCA(
        n_components=n_components,
        random_state=RANDOM_SEED,
    )

    print("Fitting PCA on training curves...")

    train_coeffs = pca.fit_transform(Y_train)
    test_coeffs = pca.transform(Y_test)

    explained = float(np.sum(pca.explained_variance_ratio_))

    print(f"Cumulative explained variance: {explained:.8f}")
    print(f"Remaining unexplained variance: {1.0 - explained:.8e}")

    # ------------------------------------------------------------
    # 4. Train HistGradientBoosting regressors on PCA coefficients
    # ------------------------------------------------------------

    print()
    print("Training HistGradientBoostingRegressor on PCA coefficients...")

    base_regressor = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1e-4,
        random_state=RANDOM_SEED,
    )

    model = MultiOutputRegressor(base_regressor)

    model.fit(
        X_train_scaled,
        train_coeffs,
        sample_weight=w_train,
    )

    # ------------------------------------------------------------
    # 5. Predict and reconstruct curves
    # ------------------------------------------------------------

    print("Predicting test curves...")

    pred_coeffs = model.predict(X_test_scaled)
    Y_pred = pca.inverse_transform(pred_coeffs)

    Y_pred = apply_physical_postprocessing(
        Y_pred,
        t_grid=t_grid,
        d=D,
    )

    Y_test_processed = apply_physical_postprocessing(
        Y_test,
        t_grid=t_grid,
        d=D,
    )

    # ------------------------------------------------------------
    # 6. Global and curve-level metrics
    # ------------------------------------------------------------

    print("Evaluating...")

    metrics = curve_level_metrics(
        Y_true=Y_test_processed,
        Y_pred=Y_pred,
    )

    metrics.update(
        max_capacity_metrics(
            Y_true=Y_test_processed,
            Y_pred=Y_pred,
            t_grid=t_grid,
        )
    )

    # ------------------------------------------------------------
    # 7. Per-curve error diagnostics
    # ------------------------------------------------------------

    per_curve_df = per_curve_error_metrics(
        Y_true=Y_test_processed,
        Y_pred=Y_pred,
        sample_ids=sample_id_test,
        sample_types=sample_type_test,
    )

    per_curve_df.to_csv(PER_CURVE_ERRORS_PATH, index=False)

    plot_per_curve_rmse_hist(
        per_curve_df=per_curve_df,
        output_path=PER_CURVE_ERROR_HIST_PATH,
    )

    plot_worst_curves(
        t_grid=t_grid,
        Y_true=Y_test_processed,
        Y_pred=Y_pred,
        per_curve_df=per_curve_df,
        output_path=WORST_CURVES_PLOT_PATH,
        n_worst=8,
    )

    curve_failure_summary = {}
    n_test_curves = len(per_curve_df)

    for threshold in CURVE_RMSE_THRESHOLDS:
        count = int(np.sum(per_curve_df["rmse"] > threshold))
        curve_failure_summary[f"curves_rmse_gt_{threshold}"] = count
        curve_failure_summary[f"fraction_rmse_gt_{threshold}"] = float(
            count / n_test_curves
        )

    for threshold in RELATIVE_RMSE_THRESHOLDS:
        count = int(np.sum(per_curve_df["relative_rmse"] > threshold))
        curve_failure_summary[f"curves_relative_rmse_gt_{threshold}"] = count
        curve_failure_summary[f"fraction_relative_rmse_gt_{threshold}"] = float(
            count / n_test_curves
        )

    metrics.update(curve_failure_summary)

    # ------------------------------------------------------------
    # 8. Metadata
    # ------------------------------------------------------------

    metrics.update(
        {
            "model": "PCA + HistGradientBoostingRegressor",
            "n_components": int(n_components),
            "explained_variance": explained,
            "remaining_unexplained_variance": float(1.0 - explained),
            "min_valid_fraction": MIN_VALID_FRACTION,
            "min_mean_reliability": MIN_MEAN_RELIABILITY,
            "train_curves": int(len(X_train)),
            "test_curves": int(len(X_test)),
            "feature_names": feature_names,
            "regressor": {
                "type": "HistGradientBoostingRegressor",
                "max_iter": 500,
                "learning_rate": 0.05,
                "max_leaf_nodes": 31,
                "l2_regularization": 1e-4,
            },
        }
    )

    print()
    print("Metrics")
    print("-" * 60)
    print(json.dumps(metrics, indent=2))

    print()
    print("Worst predicted curves by RMSE")
    print("-" * 60)
    print(
        per_curve_df
        .sort_values("rmse", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    # ------------------------------------------------------------
    # 9. Save outputs
    # ------------------------------------------------------------

    print()
    print("Saving outputs...")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(pca, PCA_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    np.savez_compressed(
        PREDICTIONS_PATH,
        t_grid=t_grid,
        X_test=X_test,
        Y_test=Y_test_processed,
        Y_pred=Y_pred,
        pred_coeffs=pred_coeffs,
        true_coeffs=test_coeffs,
        sample_id_test=sample_id_test,
        sample_type_test=sample_type_test,
        feature_names=np.array(feature_names),
    )

    plot_explained_variance(
        pca=pca,
        output_path=PCA_VARIANCE_PLOT_PATH,
    )

    plot_example_curves(
        t_grid=t_grid,
        Y_true=Y_test_processed,
        Y_pred=Y_pred,
        output_path=EXAMPLE_CURVES_PLOT_PATH,
        n_examples=8,
    )

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved scaler: {SCALER_PATH}")
    print(f"Saved PCA basis: {PCA_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved predictions: {PREDICTIONS_PATH}")
    print(f"Saved per-curve errors: {PER_CURVE_ERRORS_PATH}")
    print(f"Saved RMSE histogram: {PER_CURVE_ERROR_HIST_PATH}")
    print(f"Saved worst-curves plot: {WORST_CURVES_PLOT_PATH}")
    print(f"Saved PCA variance plot: {PCA_VARIANCE_PLOT_PATH}")
    print(f"Saved example curves plot: {EXAMPLE_CURVES_PLOT_PATH}")


if __name__ == "__main__":
    main()