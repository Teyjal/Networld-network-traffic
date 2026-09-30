import pandas as pd
import os

INPUT_FILE = "data/processed/cic_ids2018_friday_02_03_clean.csv"

OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "networld_features.csv"
)

# Features selected for the NetWorld network state
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

# Keep timestamp for temporal ordering
COLUMNS_TO_READ = ["Timestamp"] + FEATURES + ["Label", "Infiltration"]

print("Loading selected features...")

df = pd.read_csv(
    INPUT_FILE,
    usecols=COLUMNS_TO_READ
)

print("Original selected-data shape:", df.shape)

# Convert timestamp
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="coerce"
)

# Remove invalid timestamps
df = df.dropna(subset=["Timestamp"])

# Sort chronologically
df = df.sort_values("Timestamp")

# Reset index
df = df.reset_index(drop=True)

print("Final shape:", df.shape)

print("\nFeature count:", len(FEATURES))

print("\nInfiltration distribution:")
print(df["Infiltration"].value_counts())

print("\nSaving feature dataset...")

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Saved:", OUTPUT_FILE)