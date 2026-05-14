from pathlib import Path
import pandas as pd
import numpy as np


def load_training_batches(
    batch_dir=Path("outputs/training_data_full/batches"),
    pattern="training_data_batch_*.csv",
):
    """
    Loads all training-data batch CSV files into one DataFrame.
    """

    batch_dir = Path(batch_dir)
    files = sorted(batch_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No batch files found in {batch_dir}")

    dfs = []

    for file in files:
        print(f"Loading {file}")
        dfs.append(pd.read_csv(file))

    df = pd.concat(dfs, ignore_index=True)

    return df


def filter_training_points(
    df,
    reliability_threshold=0.2,
    target_column="C_E",
):
    """
    Keeps only valid and reliable points.
    """

    mask = (
        (df["valid_C"] == True)
        & np.isfinite(df[target_column])
        & np.isfinite(df["reliability"])
        & (df["reliability"] >= reliability_threshold)
    )

    return df.loc[mask].copy()


def train_test_split_by_column(df, split_column="split"):
    """
    Uses the precomputed split column.
    """

    train_df = df[df[split_column] == "train"].copy()
    test_df = df[df[split_column] == "test"].copy()

    if train_df.empty:
        raise ValueError("Training set is empty.")

    if test_df.empty:
        raise ValueError("Test set is empty.")

    return train_df, test_df