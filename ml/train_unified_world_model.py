"""
NetWorld Unified Temporal World Model Training Script
Trains the 51-D Unified World Model (36 Flow + 15 Packet = 51 Features):
- Predicts next 51-D network state transition S(t+1) via State Decoder (Smooth L1 Loss)
- Predicts infiltration risk P(attack) via Risk Head (BCEWithLogitsLoss)
- Preserves learned backbone representations via weight transfer from baseline checkpoint.
- Saves checkpoint to models/networld_unified_world_model_best.pt
"""

import os
import sys

workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

import time
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from ml.feature_schema import FEATURE_NAMES, NUM_FEATURES, SEQUENCE_LENGTH
from ml.packet_schema import (
    PACKET_FEATURE_NAMES,
    NUM_PACKET_FEATURES,
    UNIFIED_FEATURE_NAMES,
    NUM_UNIFIED_FEATURES,
)
from ml.packet_extractor import PacketFeatureExtractor
from ml.unified_scaler import get_unified_scaler
from ml.world_model_multihead import NetWorldMultiHeadWorldModel, create_unified_world_model

INPUT_CSV = "data/processed/networld_combined_features.csv"
BASE_CHECKPOINT = "models/networld_future_world_model_best.pt"
if not os.path.exists(BASE_CHECKPOINT):
    BASE_CHECKPOINT = "models/networld_combined_temporal_lstm_best.pt"

OUTPUT_MODEL_PATH = "models/networld_unified_world_model_best.pt"
OUTPUT_METRICS_PATH = "models/networld_unified_world_model_metrics.json"

BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 2
LAMBDA_STATE = 1.0
LAMBDA_RISK = 1.0


def main():
    print("=" * 70)
    print("NetWorld Unified Multi-Head World Model Training (51 Features)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    # 1. Initialize Unified Scaler
    print(f"\n[1/6] Loading Unified Scaler (36 Flow + 15 Packet)...")
    scaler = get_unified_scaler()
    scaler.save_unified_scaler()

    # 2. Load dataset
    print(f"\n[2/6] Loading dataset: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, usecols=FEATURE_NAMES + ["Timestamp", "Infiltration"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp", kind="stable").reset_index(drop=True)
    total_flows = len(df)
    print(f"Total flows loaded: {total_flows:,}")

    # Extract flow features
    raw_flow = df[FEATURE_NAMES].values.astype(np.float32)
    labels = df["Infiltration"].values.astype(np.float32)

    # Extract/derive packet features
    print("Extracting 15 packet-level telemetry features...")
    raw_packet, is_pkt_avail = PacketFeatureExtractor.extract_from_df(df)
    print(f"Extracted packet array of shape {raw_packet.shape}, source: {'direct' if is_pkt_avail else 'synthesized_baseline'}")

    # Normalize into Unified 51-D representation
    print("Normalizing into Unified Network State (51-D)...")
    unified_scaled = scaler.normalize_unified(raw_flow, raw_packet)
    print(f"Unified scaled matrix shape: {unified_scaled.shape}")

    # 3. Create chronological splits
    print("\n[3/6] Partitioning chronological sequences (70% Train, 15% Val, 15% Test)...")
    train_end = int(total_flows * 0.70)
    val_end = train_end + int(total_flows * 0.15)

    def extract_paired_sequences(start_idx, end_idx, sample_limit=None):
        num_avail = (end_idx - start_idx) - SEQUENCE_LENGTH
        if sample_limit and num_avail > sample_limit:
            indices = np.linspace(start_idx, end_idx - SEQUENCE_LENGTH - 1, sample_limit, dtype=int)
        else:
            indices = np.arange(start_idx, end_idx - SEQUENCE_LENGTH)

        N = len(indices)
        X = np.zeros((N, SEQUENCE_LENGTH, NUM_UNIFIED_FEATURES), dtype=np.float32)
        S_next = np.zeros((N, NUM_UNIFIED_FEATURES), dtype=np.float32)
        y_next = np.zeros((N, 1), dtype=np.float32)

        for i, idx in enumerate(indices):
            X[i] = unified_scaled[idx : idx + SEQUENCE_LENGTH]
            S_next[i] = unified_scaled[idx + SEQUENCE_LENGTH]
            y_next[i] = labels[idx + SEQUENCE_LENGTH]

        return torch.from_numpy(X), torch.from_numpy(S_next), torch.from_numpy(y_next)

    # Subsample 15,000 train, 3,000 val, 3,000 test sequences for fast responsive convergence
    train_X, train_S_next, train_y = extract_paired_sequences(0, train_end, sample_limit=15000)
    val_X, val_S_next, val_y = extract_paired_sequences(train_end, val_end, sample_limit=3000)
    test_X, test_S_next, test_y = extract_paired_sequences(val_end, total_flows, sample_limit=3000)

    print(f"Train sequences: {train_X.shape[0]:,}")
    print(f"Val sequences:   {val_X.shape[0]:,}")
    print(f"Test sequences:  {test_X.shape[0]:,}")

    train_loader = DataLoader(TensorDataset(train_X, train_S_next, train_y), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_X, val_S_next, val_y), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_X, test_S_next, test_y), batch_size=BATCH_SIZE, shuffle=False)

    # 4. Instantiate model with transferred baseline weights
    print(f"\n[4/6] Instantiating Unified World Model (input_size={NUM_UNIFIED_FEATURES})...")
    model = create_unified_world_model(
        base_checkpoint_path=BASE_CHECKPOINT,
        input_size=NUM_UNIFIED_FEATURES,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        device=device,
    )
    print(f"Transferred learned weights from: {BASE_CHECKPOINT}")

    criterion_state = nn.SmoothL1Loss()
    criterion_risk = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # 5. Training loop
    print(f"\n[5/6] Training Unified World Model for {EPOCHS} epochs...")
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_loss_state, total_loss_risk = 0.0, 0.0, 0.0
        start_time = time.time()

        for batch_x, batch_s, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_s = batch_s.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            risk_logits, s_pred = model(batch_x)

            loss_state = criterion_state(s_pred, batch_s)
            loss_risk = criterion_risk(risk_logits, batch_y)
            loss = (LAMBDA_STATE * loss_state) + (LAMBDA_RISK * loss_risk)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(batch_x)
            total_loss_state += loss_state.item() * len(batch_x)
            total_loss_risk += loss_risk.item() * len(batch_x)

        train_loss = total_loss / len(train_X)
        train_s_loss = total_loss_state / len(train_X)
        train_r_loss = total_loss_risk / len(train_X)

        # Validation
        model.eval()
        val_loss, val_s_loss, val_r_loss = 0.0, 0.0, 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch_x, batch_s, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_s = batch_s.to(device)
                batch_y = batch_y.to(device)

                risk_logits, s_pred = model(batch_x)
                loss_state = criterion_state(s_pred, batch_s)
                loss_risk = criterion_risk(risk_logits, batch_y)
                loss = (LAMBDA_STATE * loss_state) + (LAMBDA_RISK * loss_risk)

                val_loss += loss.item() * len(batch_x)
                val_s_loss += loss_state.item() * len(batch_x)
                val_r_loss += loss_risk.item() * len(batch_x)

                val_preds.extend(torch.sigmoid(risk_logits).cpu().numpy().flatten())
                val_targets.extend(batch_y.cpu().numpy().flatten())

        val_loss /= len(val_X)
        val_s_loss /= len(val_X)
        val_r_loss /= len(val_X)
        val_acc = accuracy_score(val_targets, [1 if p >= 0.5 else 0 for p in val_preds])
        elapsed = time.time() - start_time

        print(
            f"Epoch [{epoch}/{EPOCHS}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} (S: {train_s_loss:.4f}, R: {train_r_loss:.4f}) | "
            f"Val Loss: {val_loss:.4f} (S: {val_s_loss:.4f}, R: {val_r_loss:.4f}) | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "input_size": NUM_UNIFIED_FEATURES,
                    "hidden_size": 128,
                    "num_layers": 2,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "flow_feature_count": NUM_FEATURES,
                    "packet_feature_count": NUM_PACKET_FEATURES,
                    "total_feature_count": NUM_UNIFIED_FEATURES,
                    "unified_features": UNIFIED_FEATURE_NAMES,
                },
                OUTPUT_MODEL_PATH
            )
            print(f"  --> Saved new best checkpoint to {OUTPUT_MODEL_PATH}")

    # 6. Evaluation on Test Set
    print("\n[6/6] Evaluating on Unseen Test Sequences...")
    best_ckpt = torch.load(OUTPUT_MODEL_PATH, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    test_preds, test_targets = [], []
    state_errors = []
    flow_state_errors = []
    pkt_state_errors = []

    with torch.no_grad():
        for batch_x, batch_s, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_s = batch_s.to(device)
            batch_y = batch_y.to(device)

            risk_logits, s_pred = model(batch_x)
            test_preds.extend(torch.sigmoid(risk_logits).cpu().numpy().flatten())
            test_targets.extend(batch_y.cpu().numpy().flatten())

            err = torch.abs(s_pred - batch_s).cpu().numpy()
            state_errors.append(err)
            flow_state_errors.append(err[:, :NUM_FEATURES])
            pkt_state_errors.append(err[:, NUM_FEATURES:])

    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    binary_preds = (test_preds >= 0.5).astype(int)

    all_err = np.concatenate(state_errors, axis=0)
    all_flow_err = np.concatenate(flow_state_errors, axis=0)
    all_pkt_err = np.concatenate(pkt_state_errors, axis=0)

    acc = float(accuracy_score(test_targets, binary_preds))
    prec = float(precision_score(test_targets, binary_preds, zero_division=0))
    rec = float(recall_score(test_targets, binary_preds, zero_division=0))
    f1 = float(f1_score(test_targets, binary_preds, zero_division=0))
    auc = float(roc_auc_score(test_targets, test_preds)) if len(np.unique(test_targets)) > 1 else 0.5

    mean_mae = float(np.mean(all_err))
    flow_mae = float(np.mean(all_flow_err))
    pkt_mae = float(np.mean(all_pkt_err))

    print("\n" + "=" * 70)
    print("FINAL TEST METRICS (Unified World Model - 51 Features):")
    print(f"  Risk Infiltration Accuracy: {acc * 100:.2f}%")
    print(f"  Risk Precision:            {prec:.4f}")
    print(f"  Risk Recall:               {rec:.4f}")
    print(f"  Risk F1-Score:             {f1:.4f}")
    print(f"  Risk ROC-AUC:              {auc:.4f}")
    print(f"  Unified State Mean MAE:    {mean_mae:.4f}")
    print(f"  Flow State MAE (36 feats): {flow_mae:.4f}")
    print(f"  Packet State MAE (15 feats): {pkt_mae:.4f}")
    print("=" * 70)

    # Multi-step rollout MAE check (horizons 1, 3, 5)
    rollout_maes = {}
    with torch.no_grad():
        for horizon in [1, 3, 5]:
            errors = []
            for i in range(min(50, len(test_X) - horizon)):
                curr_window = test_X[i:i+1].clone().to(device)
                for step in range(1, horizon + 1):
                    _, next_s = model(curr_window)
                    if step == horizon:
                        true_s = test_X[i + horizon, -1:, :].to(device)
                        errors.append(torch.mean(torch.abs(next_s.unsqueeze(1) - true_s)).item())
                    curr_window = torch.cat([curr_window[:, 1:, :], next_s.unsqueeze(1)], dim=1)
            rollout_maes[f"horizon_{horizon}"] = round(float(np.mean(errors)), 4) if errors else 0.0

    print("Multi-step Rollout MAEs across horizons:")
    for h, m in rollout_maes.items():
        print(f"  {h}: {m}")

    metrics_payload = {
        "model_architecture": "NetWorldMultiHeadWorldModel",
        "input_features": NUM_UNIFIED_FEATURES,
        "flow_features": NUM_FEATURES,
        "packet_features": NUM_PACKET_FEATURES,
        "test_accuracy": round(acc, 4),
        "test_f1": round(f1, 4),
        "test_precision": round(prec, 4),
        "test_recall": round(rec, 4),
        "test_roc_auc": round(auc, 4),
        "unified_state_mae": round(mean_mae, 4),
        "flow_state_mae": round(flow_mae, 4),
        "packet_state_mae": round(pkt_mae, 4),
        "rollout_mae_by_horizon": rollout_maes,
        "device": str(device),
        "checkpoint": OUTPUT_MODEL_PATH,
    }

    with open(OUTPUT_METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\nMetrics report saved to: {OUTPUT_METRICS_PATH}")
    print("Unified model training and verification completed successfully!")


if __name__ == "__main__":
    main()
