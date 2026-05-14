from pathlib import Path
import json

import numpy as np
import joblib

from sklearn.ensemble import HistGradientBoostingRegressor
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


# ============================================================
# SETTINGS
# ============================================================

BATCH_DIR = Path("outputs/training_data_full/batches")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "hist_gradient_boosting_CE.joblib"
SCALER_PATH = MODEL_DIR / "hist_gradient_boosting_scaler.joblib"
METRICS_PATH = MODEL_DIR / "hist_gradient_boosting_metrics.json"

RELIABILITY_THRESHOLD = 0.2
USE_LOG_TARGET = False

FEATURE_COLUMNS = [
    "gamma_mean",
    "gamma_delta",
    "omega_mean",
    "omega_delta",
    "t",
]


# ============================================================
# MAIN
# ============================================================

def main():
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

    print("Scaling features...")
    scaler = fit_feature_scaler(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training HistGradientBoostingRegressor...")

    model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1e-4,
        random_state=42,
    )

    model.fit(
        X_train_scaled,
        y_train,
        sample_weight=w_train,
    )

    print("Evaluating...")

    y_pred = model.predict(X_test_scaled)

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
        "model": "HistGradientBoostingRegressor",
        "reliability_threshold": RELIABILITY_THRESHOLD,
        "use_log_target": USE_LOG_TARGET,
        "feature_columns": FEATURE_COLUMNS,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }

    print("Metrics:")
    print(json.dumps(metrics, indent=2))

    print("Saving model...")
    joblib.dump(model, MODEL_PATH)
    save_scaler(scaler, SCALER_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved scaler to: {SCALER_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()