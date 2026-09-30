import os
import numpy as np
import torch
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from world_model import NetWorldLSTM


# ==============================
# Configuration
# ==============================

X_FILE = "data/processed/sequences/X_sequences.npy"
Y_FILE = "data/processed/sequences/y_next.npy"

MODEL_FILE = "models/networld_lstm_best.pt"
SCALER_FILE = "models/networld_scaler.pkl"

INPUT_SIZE = 36
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

BATCH_SIZE = 64


# ==============================
# Device
# ==============================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Evaluation device:", DEVICE)


# ==============================
# Load data
# ==============================

print("\nLoading test data...")

X = np.load(X_FILE, mmap_mode="r")
y = np.load(Y_FILE, mmap_mode="r")

total_samples = len(X)

train_end = int(total_samples * TRAIN_RATIO)
val_end = train_end + int(total_samples * VAL_RATIO)

test_start = val_end
test_end = total_samples

print("Total samples:", total_samples)
print("Test samples:", test_end - test_start)


# ==============================
# Load scaler
# ==============================

print("\nLoading scaler...")

scaler = joblib.load(SCALER_FILE)

print("Scaler loaded.")


# ==============================
# Load model
# ==============================

print("\nLoading trained model...")

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

model = NetWorldLSTM(
    input_size=checkpoint["input_size"],
    hidden_size=checkpoint["hidden_size"],
    num_layers=checkpoint["num_layers"],
    dropout=checkpoint["dropout"]
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

print("Model loaded:")
print(model)


# ==============================
# Evaluation
# ==============================

all_predictions = []
all_targets = []

print("\nStarting test evaluation...")

with torch.no_grad():

    for start in range(test_start, test_end, BATCH_SIZE):

        end = min(start + BATCH_SIZE, test_end)

        batch_X = X[start:end].copy()
        batch_y = y[start:end].copy()

        original_shape = batch_X.shape

        # Scale features
        batch_X = batch_X.reshape(
            -1,
            original_shape[-1]
        )

        batch_X = scaler.transform(batch_X)

        batch_X = batch_X.reshape(original_shape)

        batch_X = torch.tensor(
            batch_X,
            dtype=torch.float32
        ).to(DEVICE)

        # Prediction
        logits = model(batch_X).squeeze(1)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= 0.5
        ).int().cpu().numpy()

        all_predictions.extend(predictions)
        all_targets.extend(batch_y)


# ==============================
# Metrics
# ==============================

all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)

accuracy = accuracy_score(
    all_targets,
    all_predictions
)

precision = precision_score(
    all_targets,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_targets,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_targets,
    all_predictions,
    zero_division=0
)

cm = confusion_matrix(
    all_targets,
    all_predictions
)


# ==============================
# False Positive Rate
# ==============================

tn, fp, fn, tp = cm.ravel()

if (fp + tn) > 0:
    fpr = fp / (fp + tn)
else:
    fpr = 0.0


# ==============================
# Results
# ==============================

print("\n" + "=" * 50)
print("NETWORLD LSTM TEST RESULTS")
print("=" * 50)

print(f"Test samples : {len(all_targets)}")

print(f"\nAccuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1-score     : {f1:.4f}")
print(f"False Positive Rate : {fpr:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nConfusion Matrix Details:")
print(f"True Negatives  : {tn}")
print(f"False Positives : {fp}")
print(f"False Negatives : {fn}")
print(f"True Positives  : {tp}")

print("\nEvaluation complete.")