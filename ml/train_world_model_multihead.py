"""
Training Script for NetWorld Multi-Head Temporal World Model
Learns both:
  1. Next network state transitions: S(t+1) via State Decoder (Smooth L1 Loss)
  2. Next infiltration risk: P(attack) via Risk Head (BCEWithLogitsLoss)

Uses chronological sequences from networld_combined_features.csv.
Initializes the LSTM backbone from the verified baseline checkpoint to preserve learned representations,
then optimizes the combined multi-task objective.
Saves the new model checkpoint separately to models/networld_future_world_model_best.pt.
"""

import os
import sys

# Ensure workspace root is in sys.path
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

import time
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from ml.feature_schema import FEATURE_NAMES, NUM_FEATURES, SEQUENCE_LENGTH
from ml.world_model_multihead import NetWorldMultiHeadWorldModel

INPUT_CSV = "data/processed/networld_combined_features.csv"
SCALER_PATH = "models/networld_combined_temporal_scaler.pkl"
BASELINE_CHECKPOINT = "models/networld_combined_temporal_lstm_best.pt"
OUTPUT_MODEL_PATH = "models/networld_future_world_model_best.pt"
OUTPUT_METRICS_PATH = "models/networld_future_world_model_metrics.json"

BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 2
LAMBDA_STATE = 1.0
LAMBDA_RISK = 1.0


def main():
    print("=" * 70)
    print("NetWorld Multi-Head Temporal World Model Training")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    # 1. Load Scaler
    print(f"\n[1/7] Loading existing fitted StandardScaler: {SCALER_PATH}")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    mean = np.asarray(scaler.mean_, dtype=np.float32)
    scale = np.asarray(scaler.scale_, dtype=np.float32)
    scale = np.where(scale == 0.0, 1.0, scale)

    # 2. Load dataset
    print(f"\n[2/7] Loading chronological traffic dataset: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, usecols=FEATURE_NAMES + ["Timestamp", "Infiltration"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp", kind="stable").reset_index(drop=True)
    total_flows = len(df)
    print(f"Total flows loaded: {total_flows:,}")

    # Extract raw arrays
    features_raw = df[FEATURE_NAMES].values.astype(np.float32)
    labels_raw = df["Infiltration"].values.astype(np.float32)

    # Normalize features using existing scaler parameters
    print("Normalizing features with existing scaler parameters...")
    features_scaled = (features_raw - mean) / scale
    features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. Create chronological splits (70% Train, 15% Val, 15% Test)
    print("\n[3/7] Partitioning chronological sequence data...")
    train_end = int(total_flows * 0.70)
    val_end = train_end + int(total_flows * 0.15)

    def extract_paired_sequences(start_idx, end_idx, sample_limit=None):
        """
        Creates:
          X: (N, 20, 36) - Sequence of 20 historical flows
          S_next: (N, 36) - Next physical flow state at t+1
          y_next: (N, 1)  - Infiltration label at t+1
        """
        num_avail = (end_idx - start_idx) - SEQUENCE_LENGTH
        if sample_limit and num_avail > sample_limit:
            # Subsample contiguous blocks to preserve temporal dynamics while controlling training time
            indices = np.linspace(start_idx, end_idx - SEQUENCE_LENGTH - 1, sample_limit, dtype=int)
        else:
            indices = np.arange(start_idx, end_idx - SEQUENCE_LENGTH)

        N = len(indices)
        X = np.zeros((N, SEQUENCE_LENGTH, NUM_FEATURES), dtype=np.float32)
        S_next = np.zeros((N, NUM_FEATURES), dtype=np.float32)
        y_next = np.zeros((N, 1), dtype=np.float32)

        for i, idx in enumerate(indices):
            X[i] = features_scaled[idx : idx + SEQUENCE_LENGTH]
            S_next[i] = features_scaled[idx + SEQUENCE_LENGTH]
            y_next[i, 0] = labels_raw[idx + SEQUENCE_LENGTH]

        return X, S_next, y_next

    # Using 100,000 paired training samples for efficient diagnostic training convergence
    X_train, S_train, y_train = extract_paired_sequences(0, train_end, sample_limit=100000)
    X_val, S_val, y_val = extract_paired_sequences(train_end, val_end, sample_limit=20000)
    X_test, S_test, y_test = extract_paired_sequences(val_end, total_flows, sample_limit=20000)

    print(f"Train split : X={X_train.shape}, S_next={S_train.shape}, y_next={y_train.shape} (Attack: {np.mean(y_train)*100:.2f}%)")
    print(f"Val split   : X={X_val.shape}, S_next={S_val.shape}, y_next={y_val.shape} (Attack: {np.mean(y_val)*100:.2f}%)")
    print(f"Test split  : X={X_test.shape}, S_next={S_test.shape}, y_next={y_test.shape} (Attack: {np.mean(y_test)*100:.2f}%)")

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(S_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(S_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(S_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Instantiate Multi-Head World Model and Initialize Backbone
    print("\n[4/7] Instantiating Multi-Head World Model & Loading Backbone Weights...")
    model = NetWorldMultiHeadWorldModel(
        input_size=NUM_FEATURES,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
    ).to(device)

    # Transfer learned weights from baseline checkpoint where compatible
    if os.path.exists(BASELINE_CHECKPOINT):
        print(f"Initializing backbone encoder & risk head from {BASELINE_CHECKPOINT}...")
        ckpt = torch.load(BASELINE_CHECKPOINT, map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

        model_dict = model.state_dict()
        transferred = 0
        for k, v in state_dict.items():
            if k.startswith("lstm.") and k in model_dict and model_dict[k].shape == v.shape:
                model_dict[k] = v
                transferred += 1
            elif k.startswith("infiltration_head."):
                # Map old infiltration_head to new risk_head
                new_k = k.replace("infiltration_head.", "risk_head.")
                if new_k in model_dict and model_dict[new_k].shape == v.shape:
                    model_dict[new_k] = v
                    transferred += 1

        model.load_state_dict(model_dict)
        print(f"Successfully transferred {transferred} parameter tensors from baseline checkpoint.")

    # 5. Objectives & Optimizer
    state_criterion = nn.SmoothL1Loss()  # Robust to feature outliers
    risk_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Multi-Task Training Loop
    print("\n[5/7] Executing Multi-Task Training Loop...")
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_total = 0.0
        train_state_loss = 0.0
        train_risk_loss = 0.0
        start_t = time.time()

        for batch_x, batch_s, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_s = batch_s.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            risk_logits, next_state_pred = model(batch_x)

            l_state = state_criterion(next_state_pred, batch_s)
            l_risk = risk_criterion(risk_logits, batch_y)
            loss = (LAMBDA_STATE * l_state) + (LAMBDA_RISK * l_risk)

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * len(batch_x)
            train_state_loss += l_state.item() * len(batch_x)
            train_risk_loss += l_risk.item() * len(batch_x)

        epoch_time = time.time() - start_t
        train_loss_avg = train_loss_total / len(train_dataset)
        train_s_avg = train_state_loss / len(train_dataset)
        train_r_avg = train_risk_loss / len(train_dataset)

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_state_loss = 0.0
        val_risk_loss = 0.0
        all_preds = []
        all_targets = []
        state_errors = []

        with torch.no_grad():
            for batch_x, batch_s, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_s = batch_s.to(device)
                batch_y = batch_y.to(device)

                risk_logits, next_state_pred = model(batch_x)
                l_state = state_criterion(next_state_pred, batch_s)
                l_risk = risk_criterion(risk_logits, batch_y)
                loss = (LAMBDA_STATE * l_state) + (LAMBDA_RISK * l_risk)

                val_loss_total += loss.item() * len(batch_x)
                val_state_loss += l_state.item() * len(batch_x)
                val_risk_loss += l_risk.item() * len(batch_x)

                probs = torch.sigmoid(risk_logits).cpu().numpy().flatten()
                all_preds.extend(probs)
                all_targets.extend(batch_y.cpu().numpy().flatten())

                mae_batch = torch.abs(next_state_pred - batch_s).mean(dim=1).cpu().numpy()
                state_errors.extend(mae_batch)

        val_loss_avg = val_loss_total / len(val_dataset)
        val_s_avg = val_state_loss / len(val_dataset)
        val_r_avg = val_risk_loss / len(val_dataset)
        val_mae = float(np.mean(state_errors))

        binary_preds = (np.array(all_preds) >= 0.5).astype(int)
        binary_targets = np.array(all_targets).astype(int)
        val_acc = float(accuracy_score(binary_targets, binary_preds))
        val_f1 = float(f1_score(binary_targets, binary_preds, zero_division=0))
        val_auc = float(roc_auc_score(binary_targets, all_preds)) if len(np.unique(binary_targets)) > 1 else 0.5

        print(
            f"Epoch {epoch}/{EPOCHS} [{epoch_time:.1f}s]: "
            f"TrainLoss={train_loss_avg:.4f} (State={train_s_avg:.4f}, Risk={train_r_avg:.4f}) | "
            f"ValLoss={val_loss_avg:.4f} (StateMAE={val_mae:.4f}, Acc={val_acc*100:.2f}%, F1={val_f1:.4f}, AUC={val_auc:.4f})"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss_avg,
            "train_state_loss": train_s_avg,
            "train_risk_loss": train_r_avg,
            "val_loss": val_loss_avg,
            "val_state_mae": val_mae,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_auc": val_auc,
        })

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            print(f"  -> Checkpoint improved. Saving to {OUTPUT_MODEL_PATH}")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss_avg,
                "input_size": NUM_FEATURES,
                "hidden_size": 128,
                "num_layers": 2,
                "feature_names": FEATURE_NAMES,
            }, OUTPUT_MODEL_PATH)

    # 7. Test Evaluation & Multi-Step Rollout Validation
    print("\n[6/7] Evaluating Best Model Checkpoint on Held-Out Test Split...")
    ckpt = torch.load(OUTPUT_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_preds = []
    test_targets = []
    test_state_errors = []

    with torch.no_grad():
        for batch_x, batch_s, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_s = batch_s.to(device)
            batch_y = batch_y.to(device)

            risk_logits, next_state_pred = model(batch_x)
            probs = torch.sigmoid(risk_logits).cpu().numpy().flatten()
            test_preds.extend(probs)
            test_targets.extend(batch_y.cpu().numpy().flatten())

            mae_batch = torch.abs(next_state_pred - batch_s).mean(dim=1).cpu().numpy()
            test_state_errors.extend(mae_batch)

    test_bin_preds = (np.array(test_preds) >= 0.5).astype(int)
    test_targets_arr = np.array(test_targets).astype(int)
    test_acc = float(accuracy_score(test_targets_arr, test_bin_preds))
    test_prec = float(precision_score(test_targets_arr, test_bin_preds, zero_division=0))
    test_rec = float(recall_score(test_targets_arr, test_bin_preds, zero_division=0))
    test_f1 = float(f1_score(test_targets_arr, test_bin_preds, zero_division=0))
    test_auc = float(roc_auc_score(test_targets_arr, test_preds)) if len(np.unique(test_targets_arr)) > 1 else 0.5
    test_mae = float(np.mean(test_state_errors))

    # K-Step Rollout Error Accumulation Analysis (Horizons K=1, 3, 5)
    print("\n[7/7] Measuring Multi-Step Rollout Error Compounding across Horizons (K=1, 3, 5)...")
    rollout_sample_size = 500
    rollout_test_x = X_test[:rollout_sample_size]
    rollout_metrics = {}

    for K in [1, 3, 5]:
        step_errors = []
        for i in range(rollout_sample_size):
            curr_window = np.copy(rollout_test_x[i])
            for step in range(1, K + 1):
                t_in = torch.from_numpy(curr_window).unsqueeze(0).to(device).float()
                with torch.no_grad():
                    _, s_pred = model(t_in)
                s_pred_np = s_pred.squeeze(0).cpu().numpy()
                curr_window = np.vstack([curr_window[1:], s_pred_np.reshape(1, NUM_FEATURES)])
            # Error at step K compared against ground truth if within bounds
            actual_future_idx = val_end + i + SEQUENCE_LENGTH + (K - 1)
            if actual_future_idx < total_flows:
                actual_state = features_scaled[actual_future_idx]
                step_errors.append(float(np.mean(np.abs(s_pred_np - actual_state))))

        rollout_metrics[f"K_{K}_mean_state_mae"] = float(np.mean(step_errors)) if step_errors else 0.0
        print(f"Horizon K={K}: Mean State MAE = {rollout_metrics[f'K_{K}_mean_state_mae']:.4f}")

    final_results = {
        "model_version": "NetWorld-MultiHead-WorldModel-v1",
        "description": "Multi-Head LSTM Network State Transition & Infiltration Risk Model",
        "checkpoint_path": OUTPUT_MODEL_PATH,
        "scaler_path": SCALER_PATH,
        "test_metrics": {
            "accuracy": test_acc,
            "precision": test_prec,
            "recall": test_rec,
            "f1": test_f1,
            "roc_auc": test_auc,
            "state_mae_step1": test_mae,
        },
        "rollout_horizon_metrics": rollout_metrics,
        "training_history": history,
    }

    with open(OUTPUT_METRICS_PATH, "w") as f:
        json.dump(final_results, f, indent=2)

    print("\n" + "=" * 70)
    print("Multi-Head World Model Training Complete!")
    print(f"Checkpoint saved: {OUTPUT_MODEL_PATH}")
    print(f"Metrics saved:    {OUTPUT_METRICS_PATH}")
    print(f"Test Accuracy:    {test_acc*100:.2f}% | Test F1: {test_f1:.4f} | State MAE: {test_mae:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
