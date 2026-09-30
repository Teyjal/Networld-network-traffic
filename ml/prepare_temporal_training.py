import os
import json
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler

TRAIN_X = "data/processed/temporal_sequences/X_train_sequences.npy"
VAL_X = "data/processed/temporal_sequences/X_val_sequences.npy"
TEST_X = "data/processed/temporal_sequences/X_test_sequences.npy"

TRAIN_Y = "data/processed/temporal_sequences/y_train.npy"
VAL_Y = "data/processed/temporal_sequences/y_val.npy"
TEST_Y = "data/processed/temporal_sequences/y_test.npy"

OUTPUT_DIR = "data/processed/temporal_training"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCALER_FILE = "models/networld_temporal_scaler.pkl"

print("Loading temporal datasets...")

X_train = np.load(TRAIN_X, mmap_mode="r")
X_val = np.load(VAL_X, mmap_mode="r")
X_test = np.load(TEST_X, mmap_mode="r")

y_train = np.load(TRAIN_Y, mmap_mode="r")
y_val = np.load(VAL_Y, mmap_mode="r")
y_test = np.load(TEST_Y, mmap_mode="r")

print("Train X:", X_train.shape)
print("Val X  :", X_val.shape)
print("Test X :", X_test.shape)

print("\nTarget distributions:")

print(
    "Train:",
    dict(zip(*np.unique(y_train, return_counts=True)))
)

print(
    "Val:",
    dict(zip(*np.unique(y_val, return_counts=True)))
)

print(
    "Test:",
    dict(zip(*np.unique(y_test, return_counts=True)))
)


# ==========================================
# Fit scaler ONLY on training data
# ==========================================

print("\nFitting scaler on training data...")

scaler = StandardScaler()

# Sample training sequences to avoid unnecessary memory use
SAMPLE_SIZE = min(100000, len(X_train))

sample_indices = np.linspace(
    0,
    len(X_train) - 1,
    SAMPLE_SIZE,
    dtype=np.int64
)

sample = X_train[sample_indices]

sample_2d = sample.reshape(
    -1,
    sample.shape[-1]
)

print("Scaler sample:", sample_2d.shape)

scaler.fit(sample_2d)

joblib.dump(
    scaler,
    SCALER_FILE
)

print("\nScaler saved:")
print(SCALER_FILE)


# ==========================================
# Metadata
# ==========================================

metadata = {
    "sequence_length": int(X_train.shape[1]),
    "feature_count": int(X_train.shape[2]),
    "train_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "test_samples": int(len(X_test)),
    "train_benign": int(np.sum(y_train == 0)),
    "train_attack": int(np.sum(y_train == 1)),
    "validation_benign": int(np.sum(y_val == 0)),
    "validation_attack": int(np.sum(y_val == 1)),
    "test_benign": int(np.sum(y_test == 0)),
    "test_attack": int(np.sum(y_test == 1))
}

metadata_file = os.path.join(
    OUTPUT_DIR,
    "temporal_dataset_metadata.json"
)

with open(metadata_file, "w") as f:
    json.dump(
        metadata,
        f,
        indent=4
    )

print("\nMetadata saved:")
print(metadata_file)

print("\nTemporal training preparation complete.")