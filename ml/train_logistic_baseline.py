"""
NetWorld - Logistic Regression Baseline Model Training & Evaluation
SIH 2026 Project

Trains a Logistic Regression baseline model on the 720 flattened scaled temporal features
(20 sequential flows x 36 numerical features) using the exact same temporal dataset splits as the PyTorch LSTM.
"""

import os
import json
import pickle
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Configuration & Paths
SEQUENCE_DIR = "data/processed/combined_temporal_sequences"
SCALER_PATH = "models/networld_combined_temporal_scaler.pkl"
MODEL_OUTPUT_PATH = "models/networld_logistic_baseline.pkl"
RESULTS_OUTPUT_PATH = "models/logistic_baseline_results.json"
EVALUATION_RESULTS_PATH = "models/evaluation_results.json"

X_TRAIN_PATH = os.path.join(SEQUENCE_DIR, "X_train_sequences.npy")
Y_TRAIN_PATH = os.path.join(SEQUENCE_DIR, "y_train.npy")
X_VAL_PATH = os.path.join(SEQUENCE_DIR, "X_val_sequences.npy")
Y_VAL_PATH = os.path.join(SEQUENCE_DIR, "y_val.npy")
X_TEST_PATH = os.path.join(SEQUENCE_DIR, "X_test_sequences.npy")
Y_TEST_PATH = os.path.join(SEQUENCE_DIR, "y_test.npy")

SEQUENCE_LENGTH = 20
NUM_FEATURES = 36
FLATTENED_FEATURES = SEQUENCE_LENGTH * NUM_FEATURES  # 720


def scale_and_flatten(X_mmap, mean, scale, batch_size=100000):
    """
    Normalizes 3D sequence array (N, 20, 36) using StandardScaler parameters
    and flattens each sequence into a 720-dimensional feature vector (N, 720).
    """
    n_samples = len(X_mmap)
    X_flat = np.zeros((n_samples, FLATTENED_FEATURES), dtype=np.float32)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        chunk = X_mmap[start:end].astype(np.float32)
        chunk_scaled = (chunk - mean) / scale
        chunk_scaled = np.nan_to_num(chunk_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_flat[start:end] = chunk_scaled.reshape(end - start, FLATTENED_FEATURES)

    return X_flat


def main():
    print("=" * 60)
    print("NetWorld Logistic Regression Baseline Training")
    print("=" * 60)

    # 1. Load Scaler
    print(f"\nLoading scaler from {SCALER_PATH}...")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    mean = np.asarray(scaler.mean_, dtype=np.float32)
    scale = np.asarray(scaler.scale_, dtype=np.float32)
    scale = np.where(scale == 0.0, 1.0, scale)
    print("Scaler loaded successfully.")

    # 2. Load Datasets
    print("\nLoading dataset sequences...")
    X_train_mmap = np.load(X_TRAIN_PATH, mmap_mode="r")
    y_train = np.load(Y_TRAIN_PATH)
    X_test_mmap = np.load(X_TEST_PATH, mmap_mode="r")
    y_test = np.load(Y_TEST_PATH)

    print(f"Train samples: {len(y_train)} (Shape: {X_train_mmap.shape})")
    print(f"Test samples : {len(y_test)} (Shape: {X_test_mmap.shape})")

    # 3. Scale & Flatten Sequences to 720 Features
    print("\nScaling and flattening training sequences to 720 features...")
    X_train_flat = scale_and_flatten(X_train_mmap, mean, scale)
    print(f"X_train_flat shape: {X_train_flat.shape}")

    print("Scaling and flattening test sequences to 720 features...")
    X_test_flat = scale_and_flatten(X_test_mmap, mean, scale)
    print(f"X_test_flat shape: {X_test_flat.shape}")

    # 4. Train Logistic Regression
    print("\nTraining Logistic Regression baseline model (max_iter=100)...")
    clf = LogisticRegression(
        max_iter=100,
        solver="lbfgs",
        random_state=42,
        class_weight=None,
    )
    clf.fit(X_train_flat, y_train)
    print("Model training complete.")

    # 5. Evaluate on Test Set
    print("\nEvaluating Logistic Regression on test set...")
    y_pred = clf.predict(X_test_flat)

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION BASELINE TEST RESULTS")
    print("=" * 60)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"FPR       : {fpr:.4f}")
    print("\nConfusion Matrix Details:")
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    # 6. Save Model to Disk
    print(f"\nSaving trained model to {MODEL_OUTPUT_PATH}...")
    joblib.dump(clf, MODEL_OUTPUT_PATH)
    print("Model saved successfully.")

    # 7. Save Evaluation Results JSON
    results = {
        "model_name": "Logistic Regression Baseline",
        "representation": "720 flattened scaled features (20 sequential flows x 36 features)",
        "hyperparameters": {
            "solver": "lbfgs",
            "max_iter": 100,
            "class_weight": "None",
            "random_state": 42,
        },
        "dataset_info": {
            "training_dataset": "CIC-IDS2018 (Combined Temporal)",
            "evaluation_split": "Temporal Holdout Test Split (20%)",
            "total_test_samples": int(len(y_test)),
            "sequence_length": SEQUENCE_LENGTH,
            "feature_count": NUM_FEATURES,
            "flattened_features": FLATTENED_FEATURES,
        },
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "matrix": cm.tolist(),
            },
        },
    }

    with open(RESULTS_OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {RESULTS_OUTPUT_PATH}")

    # 8. Update main evaluation_results.json for backend API consistency
    if os.path.exists(EVALUATION_RESULTS_PATH):
        try:
            with open(EVALUATION_RESULTS_PATH, "r") as f:
                eval_data = json.load(f)
            eval_data["models"]["logistic_regression"] = {
                "model_name": "Logistic Regression Baseline",
                "representation": "720 flattened scaled features (20 x 36)",
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "false_positive_rate": round(fpr, 4),
                "confusion_matrix": {
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                    "matrix": cm.tolist(),
                },
            }
            with open(EVALUATION_RESULTS_PATH, "w") as f:
                json.dump(eval_data, f, indent=2)
            print(f"Updated {EVALUATION_RESULTS_PATH} with 720-feature Logistic Regression results.")
        except Exception as e:
            print(f"Note: Could not update {EVALUATION_RESULTS_PATH}: {e}")

    print("=" * 60)


if __name__ == "__main__":
    main()