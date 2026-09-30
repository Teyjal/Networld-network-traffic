"""
NetWorld - Incremental Feb 16 Diagnostic Model Training & Evaluation
Preserves original Experiment 1 architecture & training configuration:
  - Architecture: NetWorldLSTM (input=36, hidden=128, layers=2, dropout=0.2, head=64->1)
  - Loss: BCEWithLogitsLoss() [NO positive class weighting]
  - Optimizer: Adam, lr=1e-3, weight_decay=0 [NO AdamW, NO clipping]
  - Model selection: Minimum Validation Loss
  - 2 Epochs diagnostic run
  - Full evaluation on validation and test sets (Acc, Prec, Rec, F1, FPR, Confusion Matrix, ROC-AUC, PR-AUC)
  - Direct comparison to Original Experiment 1 test baseline and Feb-14 reference
"""

import os
import time
import json
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
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
)

SEQUENCE_DIR = "data/processed/incremental_feb16"
SCALER_PATH = "models/incremental_feb16/networld_incremental_feb16_scaler.pkl"
MODEL_PATH = "models/incremental_feb16/networld_incremental_feb16_best.pt"
METRICS_PATH = "models/incremental_feb16/metrics.json"

INPUT_SIZE = 36
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

BATCH_SIZE = 128
EPOCHS = 2
LEARNING_RATE = 0.001
PRINT_EVERY = 500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemporalDataset(Dataset):
    def __init__(self, X, y, scaler):
        self.X = X
        self.y = y
        self.mean = np.asarray(scaler.mean_, dtype=np.float32)
        self.scale = np.asarray(scaler.scale_, dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        sequence = self.X[index].astype(np.float32, copy=True)
        sequence -= self.mean
        sequence /= self.scale
        sequence = torch.from_numpy(sequence)
        target = torch.tensor(self.y[index], dtype=torch.float32)
        return sequence, target


class NetWorldLSTM(nn.Module):
    def __init__(self, input_size=36, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.infiltration_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        logits = self.infiltration_head(last_hidden)
        return logits


def evaluate_split(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for seqs, targets in loader:
            seqs = seqs.to(device)
            targets = targets.to(device)

            logits = model(seqs).squeeze(-1)
            loss = criterion(logits, targets)
            total_loss += loss.item() * len(targets)

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.append(probs)
            all_preds.append(preds)
            all_targets.append(targets.cpu().numpy().astype(int))

    avg_loss = total_loss / len(loader.dataset)
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        tn = int(cm[0, 0]) if len(cm) > 0 else 0
        fp, fn, tp, fpr = 0, 0, 0, 0.0

    try:
        roc_auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        roc_auc = float("nan")

    try:
        p_curve, r_curve, _ = precision_recall_curve(all_targets, all_probs)
        pr_auc = auc(r_curve, p_curve)
    except Exception:
        pr_auc = float("nan")

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def main():
    print("=" * 70)
    print("NetWorld Incremental Feb 16 Diagnostic Training (2 Epochs)")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # Load scaler
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # 1. Verify Scaler
    print("\n[VERIFICATION 1] Scaler Parameter Check:")
    has_nan_scale = np.isnan(scaler.scale_).any() or np.isnan(scaler.mean_).any()
    has_inf_scale = np.isinf(scaler.scale_).any() or np.isinf(scaler.mean_).any()
    print(f"  Scaler mean shape: {scaler.mean_.shape}, scale shape: {scaler.scale_.shape}")
    print(f"  Has NaN/Inf: {has_nan_scale or has_inf_scale}")
    if has_nan_scale or has_inf_scale:
        raise ValueError("Scaler check failed!")

    # Load sequence files
    X_train_path = os.path.join(SEQUENCE_DIR, "X_train_sequences.npy")
    y_train_path = os.path.join(SEQUENCE_DIR, "y_train.npy")
    X_val_path = os.path.join(SEQUENCE_DIR, "X_val_sequences.npy")
    y_val_path = os.path.join(SEQUENCE_DIR, "y_val.npy")
    X_test_path = os.path.join(SEQUENCE_DIR, "X_test_sequences.npy")
    y_test_path = os.path.join(SEQUENCE_DIR, "y_test.npy")

    print("\n[VERIFICATION 2] Loading Sequence Files & Shapes:")
    X_train = np.load(X_train_path, mmap_mode="r")
    y_train = np.load(y_train_path, mmap_mode="r")
    X_val = np.load(X_val_path, mmap_mode="r")
    y_val = np.load(y_val_path, mmap_mode="r")
    X_test = np.load(X_test_path, mmap_mode="r")
    y_test = np.load(y_test_path, mmap_mode="r")

    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val  : X={X_val.shape}, y={y_val.shape}")
    print(f"  Test : X={X_test.shape}, y={y_test.shape}")

    # 3. Class distributions
    print("\n[VERIFICATION 3] Target Class Distributions:")
    for split_name, y_arr in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        u, c = np.unique(y_arr, return_counts=True)
        dist = dict(zip(u.tolist(), c.tolist()))
        attack_pct = (dist.get(1, 0) / len(y_arr)) * 100
        print(f"  {split_name:5s}: total={len(y_arr):,}, Benign={dist.get(0, 0):,}, Attack={dist.get(1, 0):,} ({attack_pct:.2f}%)")

    # Create datasets & loaders
    train_dataset = TemporalDataset(X_train, y_train, scaler)
    val_dataset = TemporalDataset(X_val, y_val, scaler)
    test_dataset = TemporalDataset(X_test, y_test, scaler)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 4. Model Architecture & Forward Pass Verification
    print("\n[VERIFICATION 4] Model Architecture:")
    model = NetWorldLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)
    print(model)

    print("\n[VERIFICATION 5] Forward Pass Test:")
    dummy_input = torch.randn(4, 20, 36, device=DEVICE)
    dummy_output = model(dummy_input)
    print(f"  Dummy input shape : {dummy_input.shape}")
    print(f"  Dummy output shape: {dummy_output.shape}")
    if dummy_output.shape != (4, 1):
        raise ValueError("Forward pass shape mismatch!")

    # 5. Backward Pass & Gradient Check
    print("\n[VERIFICATION 6] Single-Batch Gradient Check:")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0)

    sample_x, sample_y = next(iter(train_loader))
    sample_x, sample_y = sample_x.to(DEVICE), sample_y.to(DEVICE)
    optimizer.zero_grad()
    sample_logits = model(sample_x).squeeze(-1)
    sample_loss = criterion(sample_logits, sample_y)
    sample_loss.backward()

    has_grads = all(p.grad is not None and not torch.all(p.grad == 0) for p in model.parameters() if p.requires_grad)
    print(f"  Sample batch loss: {sample_loss.item():.4f}")
    print(f"  Gradients non-zero across all trainable layers: {has_grads}")
    if not has_grads:
        raise ValueError("Gradient check failed!")

    optimizer.zero_grad()
    print("\nAll 6 pre-training verifications PASSED. Starting 2-Epoch Diagnostic Training...")

    # Training Loop
    best_val_loss = float("inf")
    epoch_metrics = []
    total_start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        total_samples = 0

        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        for batch_idx, (seqs, targets) in enumerate(train_loader, start=1):
            seqs = seqs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            logits = model(seqs).squeeze(-1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(targets)
            total_samples += len(targets)

            if batch_idx % PRINT_EVERY == 0 or batch_idx == len(train_loader):
                cur_avg = running_loss / total_samples
                elapsed = time.time() - epoch_start
                print(f"  Batch [{batch_idx:6d}/{len(train_loader):6d}] - Train Loss: {cur_avg:.4f} - Elapsed: {elapsed:.1f}s")

        train_epoch_loss = running_loss / total_samples
        epoch_duration = time.time() - epoch_start

        # Validation
        val_eval_start = time.time()
        val_res = evaluate_split(model, val_loader, criterion, DEVICE)
        val_duration = time.time() - val_eval_start

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss     : {train_epoch_loss:.4f} (Time: {epoch_duration:.1f}s)")
        print(f"  Val Loss       : {val_res['loss']:.4f} (Eval Time: {val_duration:.1f}s)")
        print(f"  Val Accuracy   : {val_res['accuracy']:.4f}")
        print(f"  Val Precision  : {val_res['precision']:.4f}")
        print(f"  Val Recall     : {val_res['recall']:.4f}")
        print(f"  Val F1         : {val_res['f1']:.4f}")
        print(f"  Val FPR        : {val_res['fpr']:.4f}")
        print(f"  Val ROC-AUC    : {val_res['roc_auc']:.4f}")
        print(f"  Val PR-AUC     : {val_res['pr_auc']:.4f}")
        print(f"  Val CM         : TN={val_res['tn']}, FP={val_res['fp']}, FN={val_res['fn']}, TP={val_res['tp']}")

        is_best = val_res["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_res["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_res["loss"],
                    "val_f1": val_res["f1"],
                    "val_precision": val_res["precision"],
                    "val_recall": val_res["recall"],
                },
                MODEL_PATH,
            )
            print(f"  => Best model saved to {MODEL_PATH} (Criterion: min val_loss = {best_val_loss:.4f})")

        epoch_metrics.append({
            "epoch": epoch,
            "train_loss": train_epoch_loss,
            "val_loss": val_res["loss"],
            "val_accuracy": val_res["accuracy"],
            "val_precision": val_res["precision"],
            "val_recall": val_res["recall"],
            "val_f1": val_res["f1"],
            "val_fpr": val_res["fpr"],
            "val_cm": {"tn": val_res["tn"], "fp": val_res["fp"], "fn": val_res["fn"], "tp": val_res["tp"]},
            "epoch_time_seconds": epoch_duration,
        })

    total_training_time = time.time() - total_start_time
    print(f"\n2-Epoch training completed in {total_training_time:.1f}s ({total_training_time / 60:.2f} mins).")

    # Load best checkpoint for test evaluation
    print("\n" + "=" * 70)
    print("Evaluating Best Model Checkpoint on Test Set...")
    print("=" * 70)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from Epoch {checkpoint['epoch']} with Val Loss = {checkpoint['val_loss']:.4f}")

    test_res = evaluate_split(model, test_loader, criterion, DEVICE)

    print("\nTEST SET RESULTS:")
    print(f"  Test Loss      : {test_res['loss']:.4f}")
    print(f"  Accuracy       : {test_res['accuracy']:.4f}")
    print(f"  Precision      : {test_res['precision']:.4f}")
    print(f"  Recall         : {test_res['recall']:.4f}")
    print(f"  F1 Score       : {test_res['f1']:.4f}")
    print(f"  FPR            : {test_res['fpr']:.4f}")
    print(f"  ROC-AUC        : {test_res['roc_auc']:.4f}")
    print(f"  PR-AUC         : {test_res['pr_auc']:.4f}")
    print(f"  Confusion Matrix: TN={test_res['tn']}, FP={test_res['fp']}, FN={test_res['fn']}, TP={test_res['tp']}")

    # Save metrics
    results_summary = {
        "dataset_experiment": "Baseline + Friday-16-02-2018",
        "epochs_trained": EPOCHS,
        "total_training_time_seconds": total_training_time,
        "best_epoch": checkpoint["epoch"],
        "epoch_metrics": epoch_metrics,
        "test_results": test_res,
        "original_baseline": {
            "accuracy": 0.6935,
            "precision": 0.6874,
            "recall": 0.4319,
            "f1": 0.5305,
            "fpr": 0.1315,
            "tn": 162102,
            "fp": 24535,
            "fn": 70968,
            "tp": 53957,
        },
        "feb14_reference": {
            "accuracy": 0.7425,
            "precision": 0.6800,
            "recall": 0.4077,
            "f1": 0.5098,
            "fpr": 0.0938,
            "roc_auc": 0.7919,
            "pr_auc": 0.6323,
            "tn": 264425,
            "fp": 27361,
            "fn": 84491,
            "tp": 58153,
        }
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(results_summary, f, indent=2)

    # Print Comparison Table: Baseline vs Baseline + Feb-16
    orig = results_summary["original_baseline"]
    feb14 = results_summary["feb14_reference"]
    print("\n" + "=" * 80)
    print("CRITICAL COMPARISON: ORIGINAL BASELINE vs BASELINE + FEB-16 (AND FEB-14 REFERENCE)")
    print("=" * 80)
    print(f"{'Metric':<14} | {'Baseline':<12} | {'Feb-16 (New)':<14} | {'Change vs Base':<16} | {'Feb-14 Ref':<12}")
    print("-" * 80)
    for m in ["accuracy", "precision", "recall", "f1", "fpr"]:
        diff = test_res[m] - orig[m]
        sign = "+" if diff >= 0 else ""
        print(f"{m.capitalize():<14} | {orig[m]:<12.4f} | {test_res[m]:<14.4f} | {sign}{diff:<16.4f} | {feb14[m]:<12.4f}")

    print("-" * 80)
    print(f"{'TN':<14} | {orig['tn']:<12,d} | {test_res['tn']:<14,d} | {test_res['tn'] - orig['tn']:+16,d} | {feb14['tn']:<12,d}")
    print(f"{'FP':<14} | {orig['fp']:<12,d} | {test_res['fp']:<14,d} | {test_res['fp'] - orig['fp']:+16,d} | {feb14['fp']:<12,d}")
    print(f"{'FN':<14} | {orig['fn']:<12,d} | {test_res['fn']:<14,d} | {test_res['fn'] - orig['fn']:+16,d} | {feb14['fn']:<12,d}")
    print(f"{'TP':<14} | {orig['tp']:<12,d} | {test_res['tp']:<14,d} | {test_res['tp'] - orig['tp']:+16,d} | {feb14['tp']:<12,d}")
    print("=" * 80)


if __name__ == "__main__":
    main()
