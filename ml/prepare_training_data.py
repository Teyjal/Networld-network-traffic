import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# Paths
# --------------------------------------------------

X_FILE = "data/processed/sequences/X_sequences.npy"
Y_FILE = "data/processed/sequences/y_next.npy"

OUTPUT_DIR = "data/processed/training"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCALER_FILE = "models/networld_scaler.pkl"

# --------------------------------------------------
# Configuration
# --------------------------------------------------

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# --------------------------------------------------
# Load using memory mapping
# --------------------------------------------------

print("Loading sequences using memory mapping...")

X = np.load(X_FILE, mmap_mode="r")
y = np.load(Y_FILE, mmap_mode="r")

print("X shape:", X.shape)
print("y shape:", y.shape)

total_samples = len(X)

# --------------------------------------------------
# Chronological split
# --------------------------------------------------

train_end = int(total_samples * TRAIN_RATIO)
val_end = train_end + int(total_samples * VAL_RATIO)

print("\nChronological split:")
print("Total samples:", total_samples)
print("Training samples:", train_end)
print("Validation samples:", val_end - train_end)
print("Test samples:", total_samples - val_end)

# --------------------------------------------------
# Save split indices
# --------------------------------------------------

np.save(
    os.path.join(OUTPUT_DIR, "train_indices.npy"),
    np.arange(0, train_end)
)

np.save(
    os.path.join(OUTPUT_DIR, "val_indices.npy"),
    np.arange(train_end, val_end)
)

np.save(
    os.path.join(OUTPUT_DIR, "test_indices.npy"),
    np.arange(val_end, total_samples)
)

# --------------------------------------------------
# Fit scaler ONLY on training data
# --------------------------------------------------

print("\nFitting StandardScaler on training data...")

scaler = StandardScaler()

# We don't need every training sequence.
# Sample the flows from the training sequences
# to estimate the scaling parameters.

SAMPLE_SIZE = min(100000, train_end)

sample_indices = np.linspace(
    0,
    train_end - 1,
    SAMPLE_SIZE,
    dtype=np.int64
)

# Extract the sampled sequences
sample = X[sample_indices]

# Reshape:
# (samples, 20, 36)
#       ↓
# (samples * 20, 36)

sample_2d = sample.reshape(-1, sample.shape[-1])

print("Scaler sample shape:", sample_2d.shape)

scaler.fit(sample_2d)

# --------------------------------------------------
# Save scaler
# --------------------------------------------------

joblib.dump(scaler, SCALER_FILE)

print("\nScaler saved:")
print(SCALER_FILE)

# --------------------------------------------------
# Save metadata
# --------------------------------------------------

metadata = {
    "total_samples": int(total_samples),
    "sequence_length": int(X.shape[1]),
    "feature_count": int(X.shape[2]),
    "train_samples": int(train_end),
    "validation_samples": int(val_end - train_end),
    "test_samples": int(total_samples - val_end),
}

import json

metadata_file = os.path.join(
    OUTPUT_DIR,
    "dataset_metadata.json"
)

with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=4)

print("\nMetadata saved:")
print(metadata_file)

print("\nTraining data preparation complete.")