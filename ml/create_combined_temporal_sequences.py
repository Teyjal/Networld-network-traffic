import pandas as pd
import numpy as np
import os

INPUT_FILE = "data/processed/networld_combined_features.csv"

OUTPUT_DIR = "data/processed/combined_temporal_sequences"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

print("Loading combined dataset...")

df = pd.read_csv(
    INPUT_FILE,
    usecols=FEATURES + ["Timestamp", "Infiltration"]
)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="coerce"
)

df = df.dropna(
    subset=["Timestamp"]
)

df = df.sort_values(
    "Timestamp",
    kind="stable"
).reset_index(drop=True)

print("Dataset shape:", df.shape)

# ============================================================
# Chronological split
# ============================================================

total = len(df)

train_end = int(
    total * TRAIN_RATIO
)

val_end = train_end + int(
    total * VAL_RATIO
)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

print("\nRaw-flow split:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# ============================================================
# Sequence creation
# ============================================================

def create_sequences(dataframe, name):

    X_data = dataframe[
        FEATURES
    ].values.astype(np.float32)

    y_data = dataframe[
        "Infiltration"
    ].values.astype(np.int8)

    num_sequences = (
        len(dataframe) - SEQUENCE_LENGTH
    )

    X_sequences = np.zeros(
        (
            num_sequences,
            SEQUENCE_LENGTH,
            len(FEATURES)
        ),
        dtype=np.float32
    )

    y_next = np.zeros(
        num_sequences,
        dtype=np.int8
    )

    print(
        f"\nCreating {name} sequences..."
    )

    for i in range(num_sequences):

        X_sequences[i] = X_data[
            i:i + SEQUENCE_LENGTH
        ]

        y_next[i] = y_data[
            i + SEQUENCE_LENGTH
        ]

    print(
        f"{name} X:",
        X_sequences.shape
    )

    print(
        f"{name} y:",
        y_next.shape
    )

    print(
        f"{name} target distribution:"
    )

    unique, counts = np.unique(
        y_next,
        return_counts=True
    )

    for value, count in zip(
        unique,
        counts
    ):
        print(
            f"  {value}: {count}"
        )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"X_{name.lower()}_sequences.npy"
        ),
        X_sequences
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"y_{name.lower()}.npy"
        ),
        y_next
    )


# ============================================================
# Create sequences
# ============================================================

create_sequences(
    train_df,
    "train"
)

create_sequences(
    val_df,
    "val"
)

create_sequences(
    test_df,
    "test"
)

print(
    "\nCombined temporal sequence creation complete."
)

print(
    "Saved to:",
    OUTPUT_DIR
)