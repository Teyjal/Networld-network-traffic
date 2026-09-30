import pandas as pd
import numpy as np
import os

INPUT_FILE = "data/processed/networld_features.csv"

OUTPUT_DIR = "data/processed/sequences"
os.makedirs(OUTPUT_DIR, exist_ok=True)

X_OUTPUT = os.path.join(OUTPUT_DIR, "X_sequences.npy")
Y_OUTPUT = os.path.join(OUTPUT_DIR, "y_next.npy")

SEQUENCE_LENGTH = 20

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

print("Loading feature dataset...")

df = pd.read_csv(
    INPUT_FILE,
    usecols=FEATURES + ["Timestamp", "Infiltration"]
)

print("Dataset shape:", df.shape)

# Convert timestamp
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="coerce"
)

# Remove invalid timestamps
df = df.dropna(subset=["Timestamp"])

# Sort chronologically
df = df.sort_values("Timestamp")
df = df.reset_index(drop=True)

print("After timestamp processing:", df.shape)

# Extract feature matrix
X_data = df[FEATURES].values.astype(np.float32)

# Target: infiltration status of the NEXT flow
y_data = df["Infiltration"].values.astype(np.int8)

print("Feature matrix shape:", X_data.shape)
print("Target shape:", y_data.shape)

# Number of sequences
num_sequences = len(df) - SEQUENCE_LENGTH

print("Number of sequences:", num_sequences)

# Create sequence arrays
X_sequences = np.zeros(
    (num_sequences, SEQUENCE_LENGTH, len(FEATURES)),
    dtype=np.float32
)

y_next = np.zeros(
    num_sequences,
    dtype=np.int8
)

print("Creating sequences...")

for i in range(num_sequences):
    X_sequences[i] = X_data[i:i + SEQUENCE_LENGTH]
    y_next[i] = y_data[i + SEQUENCE_LENGTH]

print("Sequence creation complete.")

print("X shape:", X_sequences.shape)
print("y shape:", y_next.shape)

print("\nTarget distribution:")

unique, counts = np.unique(
    y_next,
    return_counts=True
)

for value, count in zip(unique, counts):
    print(f"{value}: {count}")

print("\nSaving sequences...")

np.save(X_OUTPUT, X_sequences)
np.save(Y_OUTPUT, y_next)

print("Saved:")
print(X_OUTPUT)
print(Y_OUTPUT)