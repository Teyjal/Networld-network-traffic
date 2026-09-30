"""
NetWorld - Incremental Feb 14 Sequence Creation & Scaler Fitting
Reads:
  data/processed/incremental_feb14/networld_combined_features_feb14.csv

Outputs:
  data/processed/incremental_feb14/X_train_sequences.npy
  data/processed/incremental_feb14/y_train.npy
  data/processed/incremental_feb14/X_val_sequences.npy
  data/processed/incremental_feb14/y_val.npy
  data/processed/incremental_feb14/X_test_sequences.npy
  data/processed/incremental_feb14/y_test.npy
  models/incremental_feb14/networld_incremental_feb14_scaler.pkl
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

INPUT_FILE = "data/processed/incremental_feb14/networld_combined_features_feb14.csv"
OUTPUT_DIR = "data/processed/incremental_feb14"
SCALER_PATH = "models/incremental_feb14/networld_incremental_feb14_scaler.pkl"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)

SEQUENCE_LENGTH = 20
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

FEATURES = [
    "Dst Port",
    "Protocol",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Fwd Pkt Len Mean",
    "Bwd Pkt Len Mean",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "Fwd Pkts/s",
    "Bwd Pkts/s",
    "Pkt Len Mean",
    "Pkt Len Std",
    "FIN Flag Cnt",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
    "URG Flag Cnt",
    "Down/Up Ratio",
    "Pkt Size Avg",
    "Init Fwd Win Byts",
    "Init Bwd Win Byts",
    "Fwd Act Data Pkts",
    "Active Mean",
    "Active Std",
    "Idle Mean",
    "Idle Std",
]


def create_sequences(dataframe, name):
    X_data = dataframe[FEATURES].values.astype(np.float32)
    y_data = dataframe["Infiltration"].values.astype(np.int8)

    num_sequences = len(dataframe) - SEQUENCE_LENGTH
    if num_sequences <= 0:
        raise ValueError(f"Insufficient rows ({len(dataframe)}) for sequence length {SEQUENCE_LENGTH}")

    x_path = os.path.join(OUTPUT_DIR, f"X_{name.lower()}_sequences.npy")
    y_path = os.path.join(OUTPUT_DIR, f"y_{name.lower()}.npy")

    print(f"\nCreating {name} sequences ({num_sequences:,} sequences)...")

    # Allocate directly on disk as a standard .npy file
    X_sequences = np.lib.format.open_memmap(
        x_path,
        mode="w+",
        dtype=np.float32,
        shape=(num_sequences, SEQUENCE_LENGTH, len(FEATURES)),
    )

    chunk_size = 100000
    for start_idx in range(0, num_sequences, chunk_size):
        end_idx = min(start_idx + chunk_size, num_sequences)
        sub_data = X_data[start_idx : end_idx + SEQUENCE_LENGTH - 1]
        chunk_window = np.lib.stride_tricks.sliding_window_view(
            sub_data, (SEQUENCE_LENGTH, len(FEATURES))
        ).squeeze(axis=1)
        X_sequences[start_idx:end_idx] = chunk_window

    X_sequences.flush()
    print(f"{name} X shape: {X_sequences.shape}")

    y_next = y_data[SEQUENCE_LENGTH:].copy()
    print(f"{name} y shape: {y_next.shape}")

    unique, counts = np.unique(y_next, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    print(f"{name} target distribution: {dist}")

    np.save(y_path, y_next)
    print(f"Saved: {x_path}")
    print(f"Saved: {y_path}")

    return X_sequences, y_next, dist


def main():
    print("=" * 60)
    print("NetWorld Incremental Feb 14 Sequence Creation")
    print("=" * 60)

    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, usecols=FEATURES + ["Timestamp", "Infiltration"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    df = df.sort_values("Timestamp", kind="stable").reset_index(drop=True)
    print("Dataset shape:", df.shape)

    total = len(df)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print("\nRaw-flow split:")
    print(f"Train     : {len(train_df):,} rows")
    print(f"Validation: {len(val_df):,} rows")
    print(f"Test      : {len(test_df):,} rows")

    X_train, y_train, train_dist = create_sequences(train_df, "train")
    X_val, y_val, val_dist = create_sequences(val_df, "val")
    X_test, y_test, test_dist = create_sequences(test_df, "test")

    # Fit scaler strictly on training data
    print("\n" + "=" * 60)
    print("Fitting StandardScaler on training data...")
    print("=" * 60)

    sample_count = min(100000, len(X_train))
    sample = X_train[:sample_count].astype(np.float64).reshape(-1, len(FEATURES))
    print(f"Scaler fitting sample shape: {sample.shape}")

    scaler = StandardScaler()
    scaler.fit(sample)

    # Verify no NaN or Inf in scaler parameters
    mean_has_nan = np.isnan(scaler.mean_).any() or np.isinf(scaler.mean_).any()
    scale_has_nan = np.isnan(scaler.scale_).any() or np.isinf(scaler.scale_).any()
    if mean_has_nan or scale_has_nan:
        raise ValueError("Scaler contains NaN or Inf values!")

    print(f"Scaler mean shape: {scaler.mean_.shape}, scale shape: {scaler.scale_.shape}")
    print("Scaler parameter verification: PASSED (no NaN / Inf)")

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to:\n  {SCALER_PATH}")

    metadata = {
        "dataset": "Baseline (02-03, 15-02) + Wednesday-14-02-2018",
        "total_rows": total,
        "sequence_length": SEQUENCE_LENGTH,
        "features": len(FEATURES),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_sequences": len(X_train),
        "val_sequences": len(X_val),
        "test_sequences": len(X_test),
        "train_dist": train_dist,
        "val_dist": val_dist,
        "test_dist": test_dist,
    }
    meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {meta_path}")
    print("\nSequence creation & scaling completed successfully.")


if __name__ == "__main__":
    main()
