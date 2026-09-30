import pandas as pd
import numpy as np
import os

INPUT_FILE = (
    "data/raw/CIC-IDS2018/"
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"
)

OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "cic_ids2018_friday_02_03_clean.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)

# Clean column names
df.columns = df.columns.str.strip()

# Remove duplicate rows
before = len(df)
df = df.drop_duplicates()

print("Removed duplicates:", before - len(df))

# Replace infinite values with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Fill missing numeric values with median
numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    if df[column].isna().any():
        df[column] = df[column].fillna(df[column].median())

# Remove rows without labels
df = df.dropna(subset=["Label"])

# Create binary infiltration target
df["Infiltration"] = (
    df["Label"]
    .str.strip()
    .str.lower()
    .ne("benign")
    .astype(np.int8)
)

print("\nProcessed shape:", df.shape)

print("\nLabels:")
print(df["Label"].value_counts())

print("\nInfiltration:")
print(df["Infiltration"].value_counts())

print("\nRemaining missing values:",
      df.isna().sum().sum())

print(
    "Remaining infinite values:",
    np.isinf(
        df.select_dtypes(include=np.number)
    ).sum().sum()
)

df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:")
print(OUTPUT_FILE)