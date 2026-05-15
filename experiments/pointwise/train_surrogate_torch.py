from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loading import (
    load_training_batches,
    filter_training_points,
    train_test_split_by_column,
)
from src.features import (
    build_feature_matrix,
    build_target_vector,
    inverse_target_transform,
    fit_feature_scaler,
    save_scaler,
)
from src.torch_models import CapacityMLP


# ============================================================
# SETTINGS
# ============================================================

BATCH_DIR = Path("outputs/training_data_full/batches")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "torch_capacity_mlp.pt"
SCALER_PATH = MODEL_DIR / "torch_capacity_scaler.joblib"
METRICS_PATH = MODEL_DIR / "torch_capacity_metrics.json"
LOSS_HISTORY_PATH = MODEL_DIR / "torch_capacity_loss_history.csv"
LOSS_PLOT_PATH = MODEL_DIR / "torch_capacity_loss_curve.png"

RELIABILITY_THRESHOLD = 0.2
USE_LOG_TARGET = False

FEATURE_COLUMNS = [
    "gamma_mean",
    "gamma_delta",
    "omega_mean",
    "omega_delta",
    "t",
]

RANDOM_SEED = 42

BATCH_SIZE = 4096
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

HIDDEN_DIM = 256
NUM_HIDDEN_LAYERS = 4
NUM_FREQUENCIES = 8
DROPOUT = 0.0


# ============================================================
# DATASET
# ============================================================

class CapacityDataset(Dataset):
    def __init__(self, X, y, weights):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.weights[idx]


def weighted_mse_loss(pred, target, weights):
    weights = torch.clamp(weights, min=0.0)

    numerator = torch.sum(weights * (pred - target) ** 2)
    denominator = torch.sum(weights) + 1e-12

    return numerator / denominator


# ============================================================
# MAIN
# ============================================================

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    df = load_training_batches(BATCH_DIR)

    print("Filtering valid/reliable points...")
    df = filter_training_points(
        df,
        reliability_threshold=RELIABILITY_THRESHOLD,
        target_column="C_E",
    )

    print(f"Filtered dataset shape: {df.shape}")

    train_df, test_df = train_test_split_by_column(df)

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")

    X_train = build_feature_matrix(train_df, FEATURE_COLUMNS)
    y_train = build_target_vector(
        train_df,
        target_column="C_E",
        use_log_target=USE_LOG_TARGET,
    )
    w_train = train_df["reliability"].to_numpy(dtype=np.float64)

    X_test = build_feature_matrix(test_df, FEATURE_COLUMNS)
    y_test = build_target_vector(
        test_df,
        target_column="C_E",
        use_log_target=USE_LOG_TARGET,
    )
    w_test = test_df["reliability"].to_numpy(dtype=np.float64)

    print("Scaling features...")
    scaler = fit_feature_scaler(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = CapacityDataset(X_train_scaled, y_train, w_train)
    test_dataset = CapacityDataset(X_test_scaled, y_test, w_test)

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

    model = CapacityMLP(
        input_dim=len(FEATURE_COLUMNS),
        hidden_dim=HIDDEN_DIM,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_frequencies=NUM_FREQUENCIES,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_test_loss = np.inf
    best_state = None

    loss_history = []

    print("Training PyTorch MLP...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for X_batch, y_batch, w_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            w_batch = w_batch.to(device)

            optimizer.zero_grad()

            pred = model(X_batch)
            loss = weighted_mse_loss(pred, y_batch, w_batch)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        test_losses = []

        with torch.no_grad():
            for X_batch, y_batch, w_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                w_batch = w_batch.to(device)

                pred = model(X_batch)
                loss = weighted_mse_loss(pred, y_batch, w_batch)

                test_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        test_loss = float(np.mean(test_losses))
#
    #
        loss_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
            }
        )
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
            f"test_loss={test_loss:.6e}"
        )

    print(f"Saved loss history to: {LOSS_HISTORY_PATH}")
    print(f"Saved loss curve to: {LOSS_PLOT_PATH}")

    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_dim": len(FEATURE_COLUMNS),
                "hidden_dim": HIDDEN_DIM,
                "num_hidden_layers": NUM_HIDDEN_LAYERS,
                "num_frequencies": NUM_FREQUENCIES,
                "dropout": DROPOUT,
            },
            "feature_columns": FEATURE_COLUMNS,
            "use_log_target": USE_LOG_TARGET,
            "best_epoch": int(best_state["epoch"]) if best_state is not None else None,
            "best_test_loss": float(best_test_loss),
        },
        MODEL_PATH,
    )

    save_scaler(scaler, SCALER_PATH)

    print(f"Checkpoint saved before final metrics to: {MODEL_PATH}")
#
    loss_history_df = pd.DataFrame(loss_history)
    loss_history_df.to_csv(LOSS_HISTORY_PATH, index=False)
#
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history_df["epoch"], loss_history_df["train_loss"], label="train loss")
    plt.plot(loss_history_df["epoch"], loss_history_df["test_loss"], label="test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted MSE loss")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=300)
    plt.close()

    print("Final evaluation...")

    model.eval()
    predictions = []

    with torch.no_grad():
        for X_batch, _, _ in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            predictions.append(pred.cpu().numpy())

    y_pred = np.concatenate(predictions)

    y_test_original = inverse_target_transform(
        y_test,
        use_log_target=USE_LOG_TARGET,
    )
    y_pred_original = inverse_target_transform(
        y_pred,
        use_log_target=USE_LOG_TARGET,
    )

    mae = mean_absolute_error(y_test_original, y_pred_original)
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, y_pred_original)

    metrics = {
        "model": "PyTorch CapacityMLP",
        "reliability_threshold": RELIABILITY_THRESHOLD,
        "use_log_target": USE_LOG_TARGET,
        "feature_columns": FEATURE_COLUMNS,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "epochs": EPOCHS,
        "best_epoch": int(best_state["epoch"]) if best_state is not None else None,
        "best_test_loss": float(best_test_loss),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "hidden_dim": HIDDEN_DIM,
        "num_hidden_layers": NUM_HIDDEN_LAYERS,
        "num_frequencies": NUM_FREQUENCIES,
    }

    print("Metrics:")
    print(json.dumps(metrics, indent=2))

    print("Saving model...")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_dim": len(FEATURE_COLUMNS),
                "hidden_dim": HIDDEN_DIM,
                "num_hidden_layers": NUM_HIDDEN_LAYERS,
                "num_frequencies": NUM_FREQUENCIES,
                "dropout": DROPOUT,
            },
            "feature_columns": FEATURE_COLUMNS,
            "use_log_target": USE_LOG_TARGET,
            "metrics": metrics,
        },
        MODEL_PATH,
    )

    save_scaler(scaler, SCALER_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved scaler to: {SCALER_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()