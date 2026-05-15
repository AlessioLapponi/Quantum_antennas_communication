from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.torch_curve_models import CurveCapacityMLP


# ============================================================
# PATHS
# ============================================================

CURVE_DATASET_FILE = Path("outputs/curve_dataset/curve_dataset.npz")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "torch_curve_capacity_mlp.pt"
SCALER_PATH = MODEL_DIR / "torch_curve_feature_scaler.joblib"

METRICS_PATH = MODEL_DIR / "torch_curve_metrics.json"
PREDICTIONS_PATH = MODEL_DIR / "torch_curve_test_predictions.npz"

LOSS_HISTORY_PATH = MODEL_DIR / "torch_curve_loss_history.csv"
LOSS_PLOT_PATH = MODEL_DIR / "torch_curve_loss_curve.png"

EXAMPLE_CURVES_PLOT_PATH = MODEL_DIR / "torch_curve_example_curves.png"
PER_CURVE_ERRORS_PATH = MODEL_DIR / "torch_curve_per_curve_errors.csv"
PER_CURVE_ERROR_HIST_PATH = MODEL_DIR / "torch_curve_per_curve_rmse_hist.png"
WORST_CURVES_PLOT_PATH = MODEL_DIR / "torch_curve_worst_curves.png"


# ============================================================
# SETTINGS
# ============================================================

MIN_VALID_FRACTION = 0.50
MIN_MEAN_RELIABILITY = 0.05

RANDOM_SEED = 42

D = 1.0
SIGMA = 0.01

BATCH_SIZE = 64
EPOCHS = 150
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

HIDDEN_DIM = 256
NUM_HIDDEN_LAYERS = 5
DROPOUT = 0.0

USE_SOFTPLUS = True

# Loss weights
DATA_LOSS_WEIGHT = 1.0
SMOOTHNESS_LOSS_WEIGHT = 1e-4

# Huber threshold for curvature penalty.
# Smaller = more sensitive to curvature.
SMOOTHNESS_HUBER_DELTA = 0.01

CURVE_RMSE_THRESHOLDS = [0.01, 0.02, 0.05, 0.1]
RELATIVE_RMSE_THRESHOLDS = [0.05, 0.10, 0.20, 0.50]


# ============================================================
# DATASET
# ============================================================

class CurveDataset(Dataset):
    def __init__(self, X, Y, valid_mask, reliability, bounce_mask):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.valid_mask = torch.tensor(valid_mask, dtype=torch.float32)
        self.reliability = torch.tensor(reliability, dtype=torch.float32)
        self.bounce_mask = torch.tensor(bounce_mask, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.Y[idx],
            self.valid_mask[idx],
            self.reliability[idx],
            self.bounce_mask[idx],
        )


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


def build_physical_features(X_params, sigma=0.01, d=1.0):
    """
    Builds physics-informed features from:

        gamma_mean, gamma_delta, omega_mean, omega_delta

    Signed deltas are kept because A/B asymmetry is physically relevant.
    Absolute deltas are added only as asymmetry-magnitude features.
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


def compute_curve_weights(Y_valid, R_curves):
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
    Y_pred = np.asarray(Y_pred, dtype=float).copy()

    Y_pred = np.maximum(Y_pred, 0.0)
    Y_pred[:, t_grid < d] = 0.0

    return Y_pred


def masked_weighted_mse_loss(y_pred, y_true, valid_mask, reliability):
    """
    Loss only on valid points, weighted by reliability.
    """

    weights = valid_mask * reliability
    squared_error = (y_pred - y_true) ** 2

    numerator = torch.sum(weights * squared_error)
    denominator = torch.sum(weights) + 1e-12

    return numerator / denominator


def huber_curvature_loss(
    y_pred,
    valid_mask,
    bounce_mask,
    delta=0.01,
):
    """
    Cusp-aware smoothness loss.

    Penalizes second finite differences away from tau-zero bounce regions.

    Around bounce regions, the penalty is suppressed because cusps are physical.
    """

    second_diff = y_pred[:, 2:] - 2.0 * y_pred[:, 1:-1] + y_pred[:, :-2]

    # A midpoint is smooth only if the three involved points are valid.
    valid_mid = (
        valid_mask[:, 2:]
        * valid_mask[:, 1:-1]
        * valid_mask[:, :-2]
    )

    # A midpoint is protected if any nearby point is a bounce point.
    bounce_mid = torch.clamp(
        bounce_mask[:, 2:] + bounce_mask[:, 1:-1] + bounce_mask[:, :-2],
        0.0,
        1.0,
    )

    smooth_mask = valid_mid * (1.0 - bounce_mid)

    abs_diff = torch.abs(second_diff)

    delta_tensor = torch.tensor(
        delta,
        dtype=y_pred.dtype,
        device=y_pred.device,
    )

    quadratic = torch.minimum(abs_diff, delta_tensor)
    linear = abs_diff - quadratic

    huber = 0.5 * quadratic**2 + delta_tensor * linear

    numerator = torch.sum(smooth_mask * huber)
    denominator = torch.sum(smooth_mask) + 1e-12

    return numerator / denominator


def curve_level_metrics(Y_true, Y_pred):
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
    true_max_idx = np.nanargmax(Y_true, axis=1)
    pred_max_idx = np.nanargmax(Y_pred, axis=1)

    true_max = Y_true[np.arange(len(Y_true)), true_max_idx]
    pred_max = Y_pred[np.arange(len(Y_pred)), pred_max_idx]

    true_tmax = t_grid[true_max_idx]
    pred_tmax = t_grid[pred_max_idx]

    return {
        "max_C_mae": float(np.mean(np.abs(true_max - pred_max))),
        "t_max_C_mae": float(np.mean(np.abs(true_tmax - pred_tmax))),
    }


def per_curve_error_metrics(Y_true, Y_pred, sample_ids=None, sample_types=None):
    errors = Y_pred - Y_true

    per_curve_mse = np.mean(errors**2, axis=1)
    per_curve_rmse = np.sqrt(per_curve_mse)
    per_curve_mae = np.mean(np.abs(errors), axis=1)
    per_curve_max_abs_error = np.max(np.abs(errors), axis=1)

    true_max = np.max(Y_true, axis=1)
    pred_max = np.max(Y_pred, axis=1)

    max_C_abs_error = np.abs(pred_max - true_max)
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


def plot_loss_history(loss_history_df, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history_df["epoch"], loss_history_df["train_loss"], label="train loss")
    plt.plot(loss_history_df["epoch"], loss_history_df["test_loss"], label="test loss")
    plt.plot(loss_history_df["epoch"], loss_history_df["train_data_loss"], linestyle="--", label="train data loss")
    plt.plot(loss_history_df["epoch"], loss_history_df["test_data_loss"], linestyle="--", label="test data loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


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
        ax.plot(t_grid, Y_pred[idx], linestyle="--", label="Torch curve NN")
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


def evaluate_loader(model, loader, device):
    model.eval()

    total_loss_values = []
    data_loss_values = []
    smoothness_loss_values = []

    with torch.no_grad():
        for X_batch, Y_batch, valid_batch, reliability_batch, bounce_batch in loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            valid_batch = valid_batch.to(device)
            reliability_batch = reliability_batch.to(device)
            bounce_batch = bounce_batch.to(device)

            pred = model(X_batch)

            data_loss = masked_weighted_mse_loss(
                pred,
                Y_batch,
                valid_batch,
                reliability_batch,
            )

            smoothness_loss = huber_curvature_loss(
                pred,
                valid_batch,
                bounce_batch,
                delta=SMOOTHNESS_HUBER_DELTA,
            )

            total_loss = (
                DATA_LOSS_WEIGHT * data_loss
                + SMOOTHNESS_LOSS_WEIGHT * smoothness_loss
            )

            total_loss_values.append(total_loss.item())
            data_loss_values.append(data_loss.item())
            smoothness_loss_values.append(smoothness_loss.item())

    return {
        "loss": float(np.mean(total_loss_values)),
        "data_loss": float(np.mean(data_loss_values)),
        "smoothness_loss": float(np.mean(smoothness_loss_values)),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading curve dataset...")
    data = load_curve_dataset(CURVE_DATASET_FILE)

    X_raw = data["X_params"]

    X, feature_names = build_physical_features(
        X_raw,
        sigma=SIGMA,
        d=D,
    )

    Y = data["Y_curves"]
    Y_valid = data["Y_valid"]
    R_curves = data["R_curves"]
    B_curves = data["B_curves"]
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
    V_train = Y_valid[train_mask]
    R_train = R_curves[train_mask]
    B_train = B_curves[train_mask]

    X_test = X[test_mask]
    Y_test = Y[test_mask]
    V_test = Y_valid[test_mask]
    R_test = R_curves[test_mask]
    B_test = B_curves[test_mask]

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

    print()
    print("Scaling physical features...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = CurveDataset(
        X_train_scaled,
        Y_train,
        V_train,
        R_train,
        B_train,
    )

    test_dataset = CurveDataset(
        X_test_scaled,
        Y_test,
        V_test,
        R_test,
        B_test,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    causal_mask = (t_grid >= D).astype(np.float32)

    model = CurveCapacityMLP(
        input_dim=X_train_scaled.shape[1],
        output_dim=Y.shape[1],
        hidden_dim=HIDDEN_DIM,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        dropout=DROPOUT,
        use_softplus=USE_SOFTPLUS,
        causal_mask=causal_mask,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_test_loss = np.inf
    best_state = None
    loss_history = []

    print()
    print("Training curve-output PyTorch model...")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_total_losses = []
        train_data_losses = []
        train_smoothness_losses = []

        for X_batch, Y_batch, valid_batch, reliability_batch, bounce_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            valid_batch = valid_batch.to(device)
            reliability_batch = reliability_batch.to(device)
            bounce_batch = bounce_batch.to(device)

            optimizer.zero_grad()

            pred = model(X_batch)

            data_loss = masked_weighted_mse_loss(
                pred,
                Y_batch,
                valid_batch,
                reliability_batch,
            )

            smoothness_loss = huber_curvature_loss(
                pred,
                valid_batch,
                bounce_batch,
                delta=SMOOTHNESS_HUBER_DELTA,
            )

            total_loss = (
                DATA_LOSS_WEIGHT * data_loss
                + SMOOTHNESS_LOSS_WEIGHT * smoothness_loss
            )

            total_loss.backward()
            optimizer.step()

            train_total_losses.append(total_loss.item())
            train_data_losses.append(data_loss.item())
            train_smoothness_losses.append(smoothness_loss.item())

        train_loss = float(np.mean(train_total_losses))
        train_data_loss = float(np.mean(train_data_losses))
        train_smoothness_loss = float(np.mean(train_smoothness_losses))

        test_eval = evaluate_loader(
            model=model,
            loader=test_loader,
            device=device,
        )

        test_loss = test_eval["loss"]
        test_data_loss = test_eval["data_loss"]
        test_smoothness_loss = test_eval["smoothness_loss"]

        loss_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_data_loss": train_data_loss,
            "train_smoothness_loss": train_smoothness_loss,
            "test_loss": test_loss,
            "test_data_loss": test_data_loss,
            "test_smoothness_loss": test_smoothness_loss,
        })

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "test_loss": test_loss,
            }

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6e} | "
            f"test_loss={test_loss:.6e} | "
            f"train_data={train_data_loss:.6e} | "
            f"test_data={test_data_loss:.6e}"
        )

    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    # Save loss history before final metrics.
    loss_history_df = pd.DataFrame(loss_history)
    loss_history_df.to_csv(LOSS_HISTORY_PATH, index=False)

    plot_loss_history(
        loss_history_df=loss_history_df,
        output_path=LOSS_PLOT_PATH,
    )

    # Save checkpoint before metrics, so a metric error does not kill the run.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_dim": X_train_scaled.shape[1],
                "output_dim": Y.shape[1],
                "hidden_dim": HIDDEN_DIM,
                "num_hidden_layers": NUM_HIDDEN_LAYERS,
                "dropout": DROPOUT,
                "use_softplus": USE_SOFTPLUS,
            },
            "feature_names": feature_names,
            "t_grid": t_grid,
            "causal_mask": causal_mask,
            "metrics": {
                "best_epoch": int(best_state["epoch"]) if best_state is not None else None,
                "best_test_loss": float(best_test_loss),
            },
        },
        MODEL_PATH,
    )

    joblib.dump(scaler, SCALER_PATH)

    print(f"Checkpoint saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")
    print(f"Loss history saved to: {LOSS_HISTORY_PATH}")
    print(f"Loss plot saved to: {LOSS_PLOT_PATH}")

    print()
    print("Final evaluation...")

    model.eval()

    predictions = []

    with torch.no_grad():
        for X_batch, _, _, _, _ in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            predictions.append(pred.cpu().numpy())

    Y_pred = np.concatenate(predictions, axis=0)

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

    plot_example_curves(
        t_grid=t_grid,
        Y_true=Y_test_processed,
        Y_pred=Y_pred,
        output_path=EXAMPLE_CURVES_PLOT_PATH,
        n_examples=8,
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

    metrics.update({
        "model": "Curve-output PyTorch MLP",
        "hidden_dim": HIDDEN_DIM,
        "num_hidden_layers": NUM_HIDDEN_LAYERS,
        "dropout": DROPOUT,
        "use_softplus": USE_SOFTPLUS,
        "epochs": EPOCHS,
        "best_epoch": int(best_state["epoch"]) if best_state is not None else None,
        "best_test_loss": float(best_test_loss),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "data_loss_weight": DATA_LOSS_WEIGHT,
        "smoothness_loss_weight": SMOOTHNESS_LOSS_WEIGHT,
        "smoothness_huber_delta": SMOOTHNESS_HUBER_DELTA,
        "min_valid_fraction": MIN_VALID_FRACTION,
        "min_mean_reliability": MIN_MEAN_RELIABILITY,
        "train_curves": int(len(X_train)),
        "test_curves": int(len(X_test)),
        "feature_names": feature_names,
    })

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

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    np.savez_compressed(
        PREDICTIONS_PATH,
        t_grid=t_grid,
        X_test=X_test,
        Y_test=Y_test_processed,
        Y_pred=Y_pred,
        sample_id_test=sample_id_test,
        sample_type_test=sample_type_test,
        feature_names=np.array(feature_names),
    )

    # Save final checkpoint again, now including full metrics.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_dim": X_train_scaled.shape[1],
                "output_dim": Y.shape[1],
                "hidden_dim": HIDDEN_DIM,
                "num_hidden_layers": NUM_HIDDEN_LAYERS,
                "dropout": DROPOUT,
                "use_softplus": USE_SOFTPLUS,
            },
            "feature_names": feature_names,
            "t_grid": t_grid,
            "causal_mask": causal_mask,
            "metrics": metrics,
        },
        MODEL_PATH,
    )

    print()
    print("Saved outputs")
    print("-" * 60)
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved scaler: {SCALER_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved predictions: {PREDICTIONS_PATH}")
    print(f"Saved loss history: {LOSS_HISTORY_PATH}")
    print(f"Saved loss plot: {LOSS_PLOT_PATH}")
    print(f"Saved per-curve errors: {PER_CURVE_ERRORS_PATH}")
    print(f"Saved RMSE histogram: {PER_CURVE_ERROR_HIST_PATH}")
    print(f"Saved worst-curves plot: {WORST_CURVES_PLOT_PATH}")
    print(f"Saved example curves plot: {EXAMPLE_CURVES_PLOT_PATH}")


if __name__ == "__main__":
    main()