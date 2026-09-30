import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================

SEQUENCE_DIR = "data/processed/combined_temporal_sequences"
SCALER_PATH = "models/networld_combined_temporal_scaler.pkl"
MODEL_PATH = "models/networld_combined_temporal_lstm_best.pt"

INPUT_SIZE = 36
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001

PRINT_EVERY = 500


# ============================================================
# Device
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


print("=" * 60)
print("NetWorld Combined Temporal Training - Optimized")
print("=" * 60)
print()
print("Device:")
print(DEVICE)
print()


# ============================================================
# Paths
# ============================================================

X_train_path = os.path.join(
    SEQUENCE_DIR, "X_train_sequences.npy"
)

y_train_path = os.path.join(
    SEQUENCE_DIR, "y_train.npy"
)

X_val_path = os.path.join(
    SEQUENCE_DIR, "X_val_sequences.npy"
)

y_val_path = os.path.join(
    SEQUENCE_DIR, "y_val.npy"
)


# ============================================================
# Load scaler
# ============================================================

print("Loading scaler...")

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

print("Scaler loaded successfully.")
print()


# ============================================================
# Dataset
# ============================================================

class TemporalDataset(Dataset):

    def __init__(self, X, y, scaler):

        self.X = X
        self.y = y
        self.scaler = scaler

        self.mean = np.asarray(
            scaler.mean_,
            dtype=np.float32
        )

        self.scale = np.asarray(
            scaler.scale_,
            dtype=np.float32
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        sequence = self.X[index].astype(
            np.float32,
            copy=True
        )

        # Fast StandardScaler equivalent:
        # (x - mean) / scale

        sequence -= self.mean
        sequence /= self.scale

        sequence = torch.from_numpy(sequence)

        target = torch.tensor(
            self.y[index],
            dtype=torch.float32
        )

        return sequence, target


# ============================================================
# Load datasets
# ============================================================

print("Loading training dataset...")

X_train = np.load(
    X_train_path,
    mmap_mode="r"
)

y_train = np.load(
    y_train_path,
    mmap_mode="r"
)

print("Loaded X:", X_train.shape)
print("Loaded y:", y_train.shape)
print()


print("Loading validation dataset...")

X_val = np.load(
    X_val_path,
    mmap_mode="r"
)

y_val = np.load(
    y_val_path,
    mmap_mode="r"
)

print("Loaded X:", X_val.shape)
print("Loaded y:", y_val.shape)
print()


# ============================================================
# Create datasets
# ============================================================

train_dataset = TemporalDataset(
    X_train,
    y_train,
    scaler
)

val_dataset = TemporalDataset(
    X_val,
    y_val,
    scaler
)


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)


print("DataLoaders ready.")
print(
    "Training batches:",
    len(train_loader)
)

print(
    "Validation batches:",
    len(val_loader)
)

print()


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

            nn.Linear(
                hidden_size,
                64
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                64,
                1
            )
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_hidden = output[:, -1, :]

        logits = self.infiltration_head(
            last_hidden
        )

        return logits


print("Creating NetWorld LSTM...")

model = NetWorldLSTM(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(DEVICE)

print(model)
print()


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

print("=" * 60)
print("Starting training...")
print("=" * 60)
print()


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )
    print("-" * 60)

    for batch_idx, (X_batch, y_batch) in enumerate(
        train_loader,
        start=1
    ):

        X_batch = X_batch.to(
            DEVICE,
            non_blocking=True
        )

        y_batch = y_batch.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(X_batch)

        logits = logits.squeeze(1)

        loss = criterion(
            logits,
            y_batch
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() *
            X_batch.size(0)
        )

        predictions = (
            torch.sigmoid(logits) >= 0.5
        )

        correct += (
            (predictions == y_batch)
            .sum()
            .item()
        )

        total += X_batch.size(0)

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_idx == 1
            or batch_idx % PRINT_EVERY == 0
            or batch_idx == len(train_loader)
        ):

            current_loss = (
                running_loss / total
            )

            current_accuracy = (
                correct / total
            )

            print(
                f"Batch "
                f"{batch_idx:,}/"
                f"{len(train_loader):,} | "
                f"Loss: {current_loss:.4f} | "
                f"Accuracy: {current_accuracy:.4f}"
            )

    train_loss = running_loss / total

    train_accuracy = correct / total


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(
                DEVICE,
                non_blocking=True
            )

            y_batch = y_batch.to(
                DEVICE,
                non_blocking=True
            )

            logits = model(X_batch)

            logits = logits.squeeze(1)

            loss = criterion(
                logits,
                y_batch
            )

            val_loss_total += (
                loss.item() *
                X_batch.size(0)
            )

            predictions = (
                torch.sigmoid(logits) >= 0.5
            )

            val_correct += (
                (predictions == y_batch)
                .sum()
                .item()
            )

            val_total += X_batch.size(0)

    val_loss = (
        val_loss_total /
        val_total
    )

    val_accuracy = (
        val_correct /
        val_total
    )


    # --------------------------------------------------------
    # Epoch results
    # --------------------------------------------------------

    print()
    print(
        f"Epoch {epoch + 1}/{EPOCHS} Complete"
    )

    print(
        f"Train Loss: "
        f"{train_loss:.4f} | "
        f"Train Accuracy: "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Val Loss: "
        f"{val_loss:.4f} | "
        f"Val Accuracy: "
        f"{val_accuracy:.4f}"
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "epoch":
                    epoch + 1,

                "val_loss":
                    val_loss,

                "input_size":
                    INPUT_SIZE,

                "hidden_size":
                    HIDDEN_SIZE,

                "num_layers":
                    NUM_LAYERS,

                "dropout":
                    DROPOUT
            },
            MODEL_PATH
        )

        print()
        print(
            "✓ Best combined temporal model saved:"
        )

        print(
            MODEL_PATH
        )

    print()
    print("=" * 60)
    print()


print("Training complete.")
print(
    "Best validation loss:",
    best_val_loss
)