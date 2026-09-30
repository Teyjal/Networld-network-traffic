import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# Configuration
# ============================================================

SEQUENCE_LENGTH = 20
INPUT_SIZE = 36
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

BATCH_SIZE = 128

X_TEST_PATH = "data/processed/combined_temporal_sequences/X_test_sequences.npy"
Y_TEST_PATH = "data/processed/combined_temporal_sequences/y_test.npy"

SCALER_PATH = "models/networld_combined_temporal_scaler.pkl"
MODEL_PATH = "models/networld_combined_temporal_lstm_best.pt"


# ============================================================
# Model
# ============================================================

class NetWorldLSTM(nn.Module):

    def __init__(
        self,
        input_size=36,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.infiltration_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        last_hidden = output[:, -1, :]

        logits = self.infiltration_head(last_hidden)

        return logits


# ============================================================
# Dataset
# ============================================================

class TemporalDataset(Dataset):

    def __init__(self, X_path, y_path, scaler):

        self.X = np.load(X_path, mmap_mode="r")
        self.y = np.load(y_path)

        self.mean = scaler.mean_.astype(np.float32)
        self.scale = scaler.scale_.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        sequence = self.X[idx].astype(np.float32).copy()

        sequence -= self.mean
        sequence /= self.scale

        target = np.float32(self.y[idx])

        return (
            torch.from_numpy(sequence),
            torch.tensor(target)
        )


# ============================================================
# Main
# ============================================================

print("=" * 60)
print("NetWorld Combined Temporal Model - Test Evaluation")
print("=" * 60)


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("\nDevice:")
print(device)


# ------------------------------------------------------------
# Load scaler
# ------------------------------------------------------------

print("\nLoading scaler...")

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

print("Scaler loaded successfully.")


# ------------------------------------------------------------
# Load test dataset
# ------------------------------------------------------------

print("\nLoading test dataset...")

test_dataset = TemporalDataset(
    X_TEST_PATH,
    Y_TEST_PATH,
    scaler
)

print(f"Loaded X: {test_dataset.X.shape}")
print(f"Loaded y: {test_dataset.y.shape}")

print("\nTest target distribution:")

unique, counts = np.unique(
    test_dataset.y,
    return_counts=True
)

for label, count in zip(unique, counts):
    print(f"  Class {label}: {count}")


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"\nTest batches: {len(test_loader)}")


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("\nLoading best model...")

model = NetWorldLSTM(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")

else:

    model.load_state_dict(checkpoint)


model.to(device)
model.eval()

print("Best model loaded successfully.")


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

print("\nRunning test evaluation...")

all_predictions = []
all_probabilities = []
all_targets = []

with torch.no_grad():

    for batch_idx, (X_batch, y_batch) in enumerate(test_loader):

        X_batch = X_batch.to(device)

        logits = model(X_batch)

        probabilities = torch.sigmoid(logits).squeeze(1)

        predictions = (
            probabilities >= 0.5
        ).long()

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_targets.extend(
            y_batch.numpy().astype(int)
        )

        if (
            (batch_idx + 1) % 500 == 0
            or batch_idx + 1 == len(test_loader)
        ):
            print(
                f"Batch {batch_idx + 1}/{len(test_loader)}"
            )


y_true = np.array(all_targets)
y_pred = np.array(all_predictions)
y_prob = np.array(all_probabilities)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

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

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

if tn + fp > 0:
    fpr = fp / (fp + tn)
else:
    fpr = 0.0


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"FPR       : {fpr:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nConfusion Matrix Details:")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")

print("\nTest Samples:")
print(f"Total: {len(y_true)}")
print(f"Benign: {(y_true == 0).sum()}")
print(f"Attack: {(y_true == 1).sum()}")

print("\n" + "=" * 60)
print("Test evaluation complete.")
print("=" * 60)
