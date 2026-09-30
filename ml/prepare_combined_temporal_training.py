import os
import json
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

SEQUENCE_DIR = "data/processed/combined_temporal_sequences"

SCALER_PATH = "models/networld_combined_temporal_scaler.pkl"

OUTPUT_DIR = "data/processed/combined_temporal_training"

INPUT_SIZE = 36

SAMPLE_SEQUENCES = 100000


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("Preparing Combined Temporal Training")
print("=" * 60)

print("\nLoading combined temporal datasets...")


X_train = np.load(
    os.path.join(
        SEQUENCE_DIR,
        "X_train_sequences.npy"
    ),
    mmap_mode="r"
)


X_val = np.load(
    os.path.join(
        SEQUENCE_DIR,
        "X_val_sequences.npy"
    ),
    mmap_mode="r"
)


X_test = np.load(
    os.path.join(
        SEQUENCE_DIR,
        "X_test_sequences.npy"
    ),
    mmap_mode="r"
)


# IMPORTANT:
# Your actual target files are y_train.npy,
# y_val.npy and y_test.npy.

y_train = np.load(
    os.path.join(
        SEQUENCE_DIR,
        "y_train.npy"
    ),
    mmap_mode="r"
)


y_val = np.load(
    os.path.join(
        SEQUENCE_DIR,
        "y_val.npy"
    ),
    mmap_mode="r"
)


y_test = np.load(
    os.path.join(
        SEQUENCE_DIR,
        "y_test.npy"
    ),
    mmap_mode="r"
)


print("\nTrain X:", X_train.shape)
print("Val X  :", X_val.shape)
print("Test X :", X_test.shape)


# ============================================================
# TARGET DISTRIBUTIONS
# ============================================================

print("\nTarget distributions:")


train_values, train_counts = np.unique(
    y_train,
    return_counts=True
)

print(
    "Train:",
    dict(
        zip(
            train_values.tolist(),
            train_counts.tolist()
        )
    )
)


val_values, val_counts = np.unique(
    y_val,
    return_counts=True
)

print(
    "Val:",
    dict(
        zip(
            val_values.tolist(),
            val_counts.tolist()
        )
    )
)


test_values, test_counts = np.unique(
    y_test,
    return_counts=True
)

print(
    "Test:",
    dict(
        zip(
            test_values.tolist(),
            test_counts.tolist()
        )
    )
)


# ============================================================
# FIT SCALER
# ============================================================

print("\nFitting scaler on training data...")


sample_count = min(
    SAMPLE_SEQUENCES,
    len(X_train)
)


sample = X_train[
    :sample_count
].astype(
    np.float64
)


sample = sample.reshape(
    -1,
    INPUT_SIZE
)


print(
    "Scaler sample:",
    sample.shape
)


scaler = StandardScaler()

scaler.fit(
    sample
)


# ============================================================
# SAVE SCALER
# ============================================================

print("\nSaving scaler...")


with open(
    SCALER_PATH,
    "wb"
) as f:

    pickle.dump(
        scaler,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )


print("\nScaler saved:")
print(SCALER_PATH)


# ============================================================
# VERIFY SCALER
# ============================================================

print("\nVerifying scaler file...")


with open(
    SCALER_PATH,
    "rb"
) as f:

    test_scaler = pickle.load(f)


print("Scaler verification: SUCCESS")

print(
    "Scaler type:",
    type(test_scaler).__name__
)

print(
    "Number of features:",
    len(test_scaler.mean_)
)


# ============================================================
# METADATA
# ============================================================

metadata = {

    "sequence_length": 20,

    "input_features": 36,

    "train_sequences": int(
        X_train.shape[0]
    ),

    "validation_sequences": int(
        X_val.shape[0]
    ),

    "test_sequences": int(
        X_test.shape[0]
    ),

    "train_benign": int(
        np.sum(y_train == 0)
    ),

    "train_attack": int(
        np.sum(y_train == 1)
    ),

    "validation_benign": int(
        np.sum(y_val == 0)
    ),

    "validation_attack": int(
        np.sum(y_val == 1)
    ),

    "test_benign": int(
        np.sum(y_test == 0)
    ),

    "test_attack": int(
        np.sum(y_test == 1)
    ),

    "scaler_sample_sequences": int(
        sample_count
    ),

    "scaler_sample_rows": int(
        sample.shape[0]
    )
}


metadata_file = os.path.join(
    OUTPUT_DIR,
    "combined_temporal_dataset_metadata.json"
)


with open(
    metadata_file,
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


print("\nMetadata saved:")
print(metadata_file)


print(
    "\nCombined temporal training preparation complete."
)