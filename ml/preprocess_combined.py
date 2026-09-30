import os
import numpy as np
import pandas as pd

# ============================================================
# NetWorld - Combined CIC-IDS2018 Preprocessing
# ============================================================

BASE_DIR = os.path.join("data", "raw", "CIC-IDS2018")
OUTPUT_DIR = os.path.join("data", "processed")

# The 2 original CIC-IDS2018 processed CSV files
INPUT_FILENAMES = [
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
]

INPUT_FILES = [os.path.join(BASE_DIR, fname) for fname in INPUT_FILENAMES]

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "networld_combined_features.csv"
)

# Valid CIC-IDS2018 timestamp range
VALID_TIMESTAMP_START = pd.Timestamp("2018-02-01 00:00:00")
VALID_TIMESTAMP_END = pd.Timestamp("2018-03-31 23:59:59")

# ============================================================
# 36 Model Features
# ============================================================

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

USECOLS = ["Timestamp"] + FEATURES + ["Label"]


# ============================================================
# File Verification
# ============================================================

def verify_input_files(file_paths):
    """
    Verifies that all required input CSV files exist before processing.
    Raises FileNotFoundError listing all missing files if any are not found.
    """
    missing_files = [f for f in file_paths if not os.path.exists(f)]
    if missing_files:
        missing_list_str = "\n".join(f"  - {f}" for f in missing_files)
        raise FileNotFoundError(
            f"The following required input file(s) were not found:\n{missing_list_str}"
        )
    print(f"All {len(file_paths)} input files successfully verified.")


# ============================================================
# Process Individual File
# ============================================================

def process_file(path, file_number, total_files):
    filename = os.path.basename(path)
    print("\n" + "=" * 60)
    print(f"Processing File [{file_number}/{total_files}]: {filename}")
    print("=" * 60)
    print(f"Path: {path}")

    # Load only required columns
    df = pd.read_csv(
        path,
        usecols=USECOLS,
        low_memory=False
    )
    orig_rows = len(df)
    print(f"Original shape: {df.shape}")

    # --------------------------------------------------------
    # 1. Remove Repeated CSV Header Rows
    #    (e.g., row index 999999 in Friday-16-02-2018 where Timestamp == "Timestamp")
    # --------------------------------------------------------
    header_mask = df["Timestamp"].astype(str).str.strip().str.lower() == "timestamp"
    repeated_headers_removed = int(header_mask.sum())
    if repeated_headers_removed > 0:
        print(f"Detected and removed {repeated_headers_removed} repeated CSV header row(s).")
        df = df.loc[~header_mask].copy()
    else:
        print("Repeated header rows removed: 0")

    # --------------------------------------------------------
    # 2. Parse Timestamps
    # --------------------------------------------------------
    print("Parsing timestamps (dayfirst=True)...")
    parsed_timestamps = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
        dayfirst=True
    )

    # --------------------------------------------------------
    # 3. Handle Known Malformed Historical Timestamps (< 2018)
    #    (e.g., Wednesday-14-02-2018 has 5, Thursday-22-02-2018 has 9)
    # --------------------------------------------------------
    malformed_ts_removed = 0
    malformed_mask = parsed_timestamps.dt.year < 2018
    malformed_count = int(malformed_mask.sum())
    if malformed_count > 0:
        offending_raw = df.loc[malformed_mask, "Timestamp"].tolist()
        print(f"Identified {malformed_count} malformed historical timestamp(s) (< 2018):")
        for ts_val in offending_raw:
            print(f"  - {ts_val}")
        df = df.loc[~malformed_mask].copy()
        parsed_timestamps = parsed_timestamps.loc[~malformed_mask]
        malformed_ts_removed = malformed_count
        print(f"Removed {malformed_ts_removed} malformed historical timestamp row(s).")

    # --------------------------------------------------------
    # 4. Strict Invalid Timestamp Check
    # --------------------------------------------------------
    invalid_timestamps = int(parsed_timestamps.isna().sum())
    print(f"Remaining invalid timestamps: {invalid_timestamps}")

    if invalid_timestamps > 0:
        bad_sample = df.loc[parsed_timestamps.isna(), "Timestamp"].head(10).tolist()
        raise ValueError(
            f"File {file_number} ({filename}) contains "
            f"{invalid_timestamps} invalid timestamp(s) that could not be parsed: {bad_sample}"
        )

    df["Timestamp"] = parsed_timestamps

    # --------------------------------------------------------
    # 5. Timestamp Range Validation (2018-02-01 through 2018-03-31)
    # --------------------------------------------------------
    out_of_range_mask = (
        (df["Timestamp"] < VALID_TIMESTAMP_START) |
        (df["Timestamp"] > VALID_TIMESTAMP_END)
    )
    out_of_range_count = int(out_of_range_mask.sum())
    if out_of_range_count > 0:
        offending_sample = df.loc[out_of_range_mask, "Timestamp"].head(10).tolist()
        raise ValueError(
            f"File {file_number} ({filename}) contains {out_of_range_count} "
            f"timestamp(s) outside valid CIC-IDS2018 range "
            f"({VALID_TIMESTAMP_START} to {VALID_TIMESTAMP_END}): {offending_sample}"
        )

    # --------------------------------------------------------
    # 6. Clean Label
    # --------------------------------------------------------
    df["Label"] = df["Label"].astype(str).str.strip()

    # --------------------------------------------------------
    # 7. Numeric Conversion & Inf Handling
    # --------------------------------------------------------
    print("Cleaning numeric features...")
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace positive and negative infinity with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows containing missing values in FEATURES + Timestamp + Label
    before_missing = len(df)
    df.dropna(
        subset=FEATURES + ["Timestamp", "Label"],
        inplace=True
    )
    removed_missing = before_missing - len(df)
    print(f"Rows removed because of missing values: {removed_missing}")

    # --------------------------------------------------------
    # 8. Remove Exact Duplicates within File
    # --------------------------------------------------------
    before_duplicates = len(df)
    df.drop_duplicates(inplace=True)
    removed_duplicates = before_duplicates - len(df)
    print(f"Duplicate rows removed: {removed_duplicates}")

    # --------------------------------------------------------
    # 9. Binary Target (0 = Benign, 1 = Any attack)
    # --------------------------------------------------------
    df["Infiltration"] = (
        df["Label"].str.lower() != "benign"
    ).astype(np.int8)

    # --------------------------------------------------------
    # 10. Sort Chronologically
    # --------------------------------------------------------
    df.sort_values("Timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --------------------------------------------------------
    # 11. Clear Data-Quality Summary for this File
    # --------------------------------------------------------
    print("\n" + "-" * 50)
    print(f"DATA-QUALITY SUMMARY: {filename}")
    print("-" * 50)
    print(f"  original rows:                              {orig_rows}")
    print(f"  repeated header rows removed:               {repeated_headers_removed}")
    print(f"  malformed/out-of-range timestamp rows removed: {malformed_ts_removed}")
    print(f"  missing-value rows removed:                 {removed_missing}")
    print(f"  duplicate rows removed:                     {removed_duplicates}")
    print(f"  final rows:                                 {len(df)}")
    print(f"  timestamp range:                            {df['Timestamp'].min()} -> {df['Timestamp'].max()}")
    print("\n  label distribution:")
    for lbl, count in df["Label"].value_counts().items():
        print(f"    {lbl}: {count}")
    print("\n  infiltration distribution:")
    for target_val, count in df["Infiltration"].value_counts().items():
        print(f"    {target_val}: {count}")
    print("-" * 50)

    return df


# ============================================================
# Main Execution
# ============================================================

def main():
    print("=" * 60)
    print("NetWorld Combined CIC-IDS2018 Preprocessing")
    print("=" * 60)

    # Verify that all input files exist before processing anything
    verify_input_files(INPUT_FILES)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each CSV separately
    processed_dfs = []
    total_files = len(INPUT_FILES)
    for idx, path in enumerate(INPUT_FILES, start=1):
        processed_df = process_file(path, idx, total_files)
        processed_dfs.append(processed_df)

    # Combine all processed datasets
    print("\n" + "=" * 60)
    print("Combining all processed datasets...")
    print("=" * 60)

    combined = pd.concat(processed_dfs, ignore_index=True)
    print(f"Combined shape before deduplication: {combined.shape}")

    del processed_dfs
    import gc
    gc.collect()

    # Remove exact duplicate rows after combining
    before_dedup = len(combined)
    combined.drop_duplicates(inplace=True)
    removed_dedup = before_dedup - len(combined)
    print(f"Duplicate rows removed after combining: {removed_dedup}")

    # Sort the combined dataset chronologically by Timestamp
    print("Sorting combined dataset chronologically by Timestamp...")
    combined.sort_values("Timestamp", inplace=True)

    # Reset the index
    combined.reset_index(drop=True, inplace=True)

    # ============================================================
    # Final Validation
    # ============================================================
    print("\n" + "=" * 60)
    print("FINAL VALIDATION")
    print("=" * 60)

    invalid_timestamps = int(combined["Timestamp"].isna().sum())
    total_missing = int(combined.isna().sum().sum())
    is_sorted = bool(combined["Timestamp"].is_monotonic_increasing)
    out_of_range = int((
        (combined["Timestamp"] < VALID_TIMESTAMP_START) |
        (combined["Timestamp"] > VALID_TIMESTAMP_END)
    ).sum())
    unique_targets = set(combined["Infiltration"].unique())

    print(f"Final shape: {combined.shape}")
    print(f"Invalid timestamp count: {invalid_timestamps}")
    print(f"Total missing-value count: {total_missing}")
    print(f"Out-of-range timestamp count: {out_of_range}")
    print(f"Chronological ordering check: {is_sorted}")
    print(f"Infiltration unique values: {sorted(list(unique_targets))}")
    print(f"Timestamp minimum: {combined['Timestamp'].min()}")
    print(f"Timestamp maximum: {combined['Timestamp'].max()}")

    print("\nFinal Label distribution:")
    for lbl, count in combined["Label"].value_counts().items():
        print(f"  {lbl}: {count}")

    print("\nFinal Infiltration distribution:")
    for target_val, count in combined["Infiltration"].value_counts().items():
        print(f"  {target_val}: {count}")

    # 1. No NaN timestamps
    if invalid_timestamps > 0:
        raise ValueError(
            f"Final validation failed: Dataset still contains {invalid_timestamps} invalid timestamps."
        )

    # 2. No missing values
    if total_missing > 0:
        raise ValueError(
            f"Final validation failed: Dataset still contains {total_missing} missing values."
        )

    # 3. All timestamps within the valid CIC-IDS2018 range
    if out_of_range > 0:
        raise ValueError(
            f"Final validation failed: Dataset contains {out_of_range} timestamps outside "
            f"valid range ({VALID_TIMESTAMP_START} to {VALID_TIMESTAMP_END})."
        )

    # 4. Chronologically sorted
    if not is_sorted:
        raise ValueError("Final validation failed: Dataset is NOT chronologically sorted.")

    # 5. Binary Infiltration contains only 0 and 1
    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"Final validation failed: Infiltration target contains values other than 0 and 1: {unique_targets}"
        )

    # ============================================================
    # Save Dataset
    # ============================================================
    print("\n" + "=" * 60)
    print(f"Saving combined dataset to:\n  {OUTPUT_FILE}")
    print("=" * 60)

    combined.to_csv(OUTPUT_FILE, index=False, chunksize=100000)

    print(f"\nSaved successfully:\n{os.path.abspath(OUTPUT_FILE)}")
    print("\n" + "=" * 60)
    print("Combined preprocessing completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
