import os
import numpy as np
import torch
import torch.nn as nn
import joblib

from torch.utils.data import Dataset, DataLoader

from world_model import NetWorldLSTM


# ==========================================
# Configuration
# ==========================================

TRAIN_X = "data/processed/temporal_sequences/X_train_sequences.npy"
TRAIN_Y = "data/processed/temporal_sequences/y_train.npy"

VAL_X = "data/processed/temporal_sequences/X_val_sequences.npy"
VAL_Y = "data/processed/temporal_sequences/y_val.npy"

SCALER_FILE = "models/networld_temporal_scaler.pkl"

MODEL_DIR = "models"
MODEL_FILE = os.path.join(
    MODEL_DIR,
    "networld_temporal_lstm_best.pt"
)

INPUT_SIZE = 36
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================
# Device
# ==========================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Training device:", DEVICE)


# ==========================================
# Dataset
# ==========================================

class SequenceDataset(Dataset):

    def __init__(
        self,
        X,
        y,
        scaler
    ):
        self.X = X
        self.y = y
        self.scaler = scaler

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

        sequence = self.X[index].copy()
        target = self.y[index]

        original_shape = sequence.shape

        sequence = sequence.reshape(
            -1,
            original_shape[-1]
        )

        sequence = self.scaler.transform(
            sequence
        )

        sequence = sequence.reshape(
            original_shape
        )

        sequence = torch.tensor(
            sequence,
            dtype=torch.float32
        )

        target = torch.tensor(
            target,
            dtype=torch.float32
        )

        return sequence, target


# ==========================================
# Load data
# ==========================================

print("\nLoading temporal datasets...")

X_train = np.load(
    TRAIN_X,
    mmap_mode="r"
)

y_train = np.load(
    TRAIN_Y,
    mmap_mode="r"
)

X_val = np.load(
    VAL_X,
    mmap_mode="r"
)

y_val = np.load(
    VAL_Y,
    mmap_mode="r"
)

print("Train X:", X_train.shape)
print("Train y:", y_train.shape)

print("Val X:", X_val.shape)
print("Val y:", y_val.shape)


# ==========================================
# Load scaler
# ==========================================

print("\nLoading temporal scaler...")

scaler = joblib.load(
    SCALER_FILE
)

print("Scaler loaded.")


# ==========================================
# Create datasets
# ==========================================

train_dataset = SequenceDataset(
    X_train,
    y_train,
    scaler
)

val_dataset = SequenceDataset(
    X_val,
    y_val,
    scaler
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==========================================
# Model
# ==========================================

model = NetWorldLSTM(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(DEVICE)

print("\nModel:")
print(model)


# ==========================================
# Loss
# ==========================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# Training
# ==========================================

best_val_loss = float("inf")

print("\nStarting temporal training...")

for epoch in range(EPOCHS):

    # ------------------------------
    # Training
    # ------------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(
            batch_X
        ).squeeze(1)

        loss = criterion(
            logits,
            batch_y
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item() *
            batch_X.size(0)
        )

        predictions = (
            torch.sigmoid(logits) >= 0.5
        ).float()

        train_correct += (
            predictions == batch_y
        ).sum().item()

        train_total += batch_y.size(0)

    train_loss /= train_total

    train_accuracy = (
        train_correct /
        train_total
    )


    # ------------------------------
    # Validation
    # ------------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for batch_X, batch_y in val_loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            logits = model(
                batch_X
            ).squeeze(1)

            loss = criterion(
                logits,
                batch_y
            )

            val_loss += (
                loss.item() *
                batch_X.size(0)
            )

            predictions = (
                torch.sigmoid(logits) >= 0.5
            ).float()

            val_correct += (
                predictions == batch_y
            ).sum().item()

            val_total += batch_y.size(0)

    val_loss /= val_total

    val_accuracy = (
        val_correct /
        val_total
    )


    # ------------------------------
    # Results
    # ------------------------------

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


    # ------------------------------
    # Save best model
    # ------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "input_size":
                    INPUT_SIZE,

                "hidden_size":
                    HIDDEN_SIZE,

                "num_layers":
                    NUM_LAYERS,

                "dropout":
                    DROPOUT,

                "sequence_length":
                    20,

                "best_val_loss":
                    best_val_loss
            },
            MODEL_FILE
        )

        print(
            "✓ Best temporal model saved:",
            MODEL_FILE
        )


print("\nTemporal training complete.")

print(
    "Best validation loss:",
    best_val_loss
)

print(
    "Model:",
    MODEL_FILE
)