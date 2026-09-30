import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from world_model import NetWorldLSTM


# ============================================================
# Configuration
# ============================================================

X_FILE = "data/processed/sequences/X_sequences.npy"
Y_FILE = "data/processed/sequences/y_next.npy"

MODEL_DIR = "models"
MODEL_FILE = os.path.join(
    MODEL_DIR,
    "networld_lstm_best.pt"
)

SEQUENCE_LENGTH = 20
INPUT_SIZE = 36

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# Device
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Training device:", DEVICE)


# ============================================================
# Memory-efficient Dataset
# ============================================================

class SequenceDataset(Dataset):

    def __init__(self, X, y, start, end, scaler):
        self.X = X
        self.y = y
        self.start = start
        self.end = end
        self.scaler = scaler

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, index):

        real_index = self.start + index

        sequence = self.X[real_index].copy()
        target = self.y[real_index]

        # Scale the 36 features
        original_shape = sequence.shape

        sequence = sequence.reshape(-1, original_shape[-1])

        sequence = self.scaler.transform(sequence)

        sequence = sequence.reshape(original_shape)

        sequence = torch.tensor(
            sequence,
            dtype=torch.float32
        )

        target = torch.tensor(
            target,
            dtype=torch.float32
        )

        return sequence, target


# ============================================================
# Load data using memory mapping
# ============================================================

print("\nLoading sequence files...")

X = np.load(
    X_FILE,
    mmap_mode="r"
)

y = np.load(
    Y_FILE,
    mmap_mode="r"
)

print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# Dataset split
# ============================================================

total_samples = len(X)

train_end = int(
    total_samples * TRAIN_RATIO
)

val_end = train_end + int(
    total_samples * VAL_RATIO
)

print("\nDataset split:")
print("Train:", train_end)
print("Validation:", val_end - train_end)
print("Test:", total_samples - val_end)


# ============================================================
# Load scaler
# ============================================================

import joblib

SCALER_FILE = "models/networld_scaler.pkl"

scaler = joblib.load(
    SCALER_FILE
)

print("\nScaler loaded.")


# ============================================================
# Create datasets
# ============================================================

train_dataset = SequenceDataset(
    X,
    y,
    0,
    train_end,
    scaler
)

val_dataset = SequenceDataset(
    X,
    y,
    train_end,
    val_end,
    scaler
)


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# Model
# ============================================================

model = NetWorldLSTM(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)

model = model.to(DEVICE)

print("\nModel:")
print(model)


# ============================================================
# Loss and optimizer
# ============================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training
# ============================================================

best_val_loss = float("inf")

print("\nStarting training...")

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(batch_X)

        logits = logits.squeeze(1)

        loss = criterion(
            logits,
            batch_y
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item() * batch_X.size(0)
        )

        predictions = (
            torch.sigmoid(logits) >= 0.5
        )

        train_correct += (
            (predictions == batch_y)
            .sum()
            .item()
        )

        train_total += batch_y.size(0)

    train_loss /= train_total

    train_accuracy = (
        train_correct / train_total
    )


    # ========================================================
    # Validation
    # ========================================================

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for batch_X, batch_y in val_loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            logits = model(batch_X)

            logits = logits.squeeze(1)

            loss = criterion(
                logits,
                batch_y
            )

            val_loss += (
                loss.item() * batch_X.size(0)
            )

            predictions = (
                torch.sigmoid(logits) >= 0.5
            )

            val_correct += (
                (predictions == batch_y)
                .sum()
                .item()
            )

            val_total += batch_y.size(0)

    val_loss /= val_total

    val_accuracy = (
        val_correct / val_total
    )


    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f}"
    )


    # ========================================================
    # Save best model
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_size": INPUT_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "sequence_length": SEQUENCE_LENGTH,
                "best_val_loss": best_val_loss
            },
            MODEL_FILE
        )

        print(
            "✓ Best model saved:",
            MODEL_FILE
        )


print("\nTraining complete.")
print("Best validation loss:", best_val_loss)
print("Model:", MODEL_FILE)