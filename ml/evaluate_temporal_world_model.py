import os
import numpy as np
import torch
import joblib

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from world_model import NetWorldLSTM


# =========================
# Paths
# =========================

X_TEST_FILE = "data/processed/temporal_sequences/X_test_sequences.npy"
Y_TEST_FILE = "data/processed/temporal_sequences/y_test.npy"

SCALER_FILE = "models/networld_temporal_scaler.pkl"
MODEL_FILE = "models/networld_temporal_lstm_best.pt"


# =========================
# Configuration
# =========================

BATCH_SIZE = 64
INPUT_SIZE = 36

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", DEVICE)


# =========================
# Dataset
# =========================

class SequenceDataset(Dataset):

    def __init__(self, X, y, scaler):
        self.X = X
        self.y = y
        self.scaler = scaler

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

        sequence = self.X[index].copy()

        sequence = self.scaler.transform(
            sequence
        ).astype(np.float32)

        target = self.y[index]

        return (
            torch.tensor(sequence, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )


# =========================
# Load test data
# =========================

print("\nLoading test data...")

X_test = np.load(
    X_TEST_FILE,
    mmap_mode="r"
)

y_test = np.load(
    Y_TEST_FILE,
    mmap_mode="r"
)

print("Test X shape:", X_test.shape)
print("Test y shape:", y_test.shape)


# =========================
# Load scaler
# =========================

print("\nLoading scaler...")

scaler = joblib.load(
    SCALER_FILE
)

print("Scaler loaded.")


# =========================
# Create DataLoader
# =========================

test_dataset = SequenceDataset(
    X_test,
    y_test,
    scaler
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =========================
# Load model
# =========================

print("\nLoading best temporal model...")

model = NetWorldLSTM(
    input_size=INPUT_SIZE,
    hidden_size=128,
    num_layers=2,
    dropout=0.2
)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

print("Model loaded:")
print(MODEL_FILE)


# =========================
# Prediction
# =========================

print("\nRunning test evaluation...")

all_targets = []
all_predictions = []
all_probabilities = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(DEVICE)

        logits = model(X_batch)

        probabilities = torch.sigmoid(
            logits
        ).squeeze(1)

        predictions = (
            probabilities >= 0.5
        ).int()

        all_targets.extend(
            y_batch.numpy().astype(int)
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


# =========================
# Metrics
# =========================

y_true = np.array(all_targets)
y_pred = np.array(all_predictions)
y_prob = np.array(all_probabilities)

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred
).ravel()

fpr = fp / (fp + tn)


# =========================
# Results
# =========================

print("\n" + "=" * 50)
print("TEMPORAL LSTM TEST RESULTS")
print("=" * 50)

print(f"Test samples : {len(y_true)}")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"FPR          : {fpr:.4f}")

print("\nConfusion Matrix:")
print(
    f"TN: {tn}"
)
print(
    f"FP: {fp}"
)
print(
    f"FN: {fn}"
)
print(
    f"TP: {tp}"
)

print("\nTest target distribution:")
print(
    dict(
        zip(
            *np.unique(
                y_true,
                return_counts=True
            )
        )
    )
)

print("\nEvaluation complete.")