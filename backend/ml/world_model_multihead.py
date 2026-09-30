from __future__ import annotations

"""
NetWorld Multi-Head Temporal World Model
Extends the baseline NetWorldLSTM to a true Predictive World Model:
Input:  (batch_size, 20, 36) - Temporal sequence of 20 network flow states
Latent: (batch_size, 128)    - Hidden network state from 2-layer LSTM encoder
Heads:
  1. State Decoder: (batch_size, 36) - Predicts next network state S(t+1)
  2. Risk Head:     (batch_size, 1)  - Predicts raw logit for P(infiltration)
  3. Attack Stage:  Derived via evidence-based MITRE mapping layer (no fabricated labels)
"""

import os
from typing import Tuple, Dict, Any, Optional

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False


class NetWorldMultiHeadWorldModel(nn.Module if HAS_TORCH else object):
    """
    Multi-Head Temporal World Model for Autonomous Cyberattack Forecasting.
    Simultaneously forecasts next network state transition S(t+1) and infiltration risk.
    """

    def __init__(
        self,
        input_size: int = 36,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if not HAS_TORCH:
            return
        super().__init__()

        # LSTM Temporal Encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Head 1: State Decoder (Predicts S(t+1) in 36-dimensional feature space)
        self.state_decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, input_size),
        )

        # Head 2: Risk Prediction Head (Predicts P(infiltration) logit)
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for temporal sequences of shape (batch_size, 20, 36).
        Returns:
            risk_logits: (batch_size, 1) - Raw logit for binary classification
            next_state:  (batch_size, 36) - Predicted next network state S(t+1)
        """
        output, (hidden, cell) = self.lstm(x)
        last_hidden = output[:, -1, :]  # Latent network representation

        risk_logits = self.risk_head(last_hidden)
        next_state = self.state_decoder(last_hidden)

        return risk_logits, next_state

    def predict_step(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience method for inference and autoregressive rollout.
        Returns probabilities and predicted states as detached tensors.
        """
        self.eval()
        with torch.no_grad():
            risk_logits, next_state = self.forward(x)
            risk_prob = torch.sigmoid(risk_logits)
            return {
                "risk_logits": risk_logits,
                "risk_prob": risk_prob,
                "next_state": next_state,
            }


def create_unified_world_model(
    base_checkpoint_path: str = "models/networld_future_world_model_best.pt",
    input_size: int = 51,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    device: Any = None,
) -> NetWorldMultiHeadWorldModel:
    """
    Creates a 51-D Unified World Model and transfers weights from the trained baseline/future checkpoint.
    Preserves all learned 36-feature temporal flow representations.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NetWorldMultiHeadWorldModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    if os.path.exists(base_checkpoint_path):
        ckpt = torch.load(base_checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

        model_dict = model.state_dict()
        for k, v in state_dict.items():
            if k == "lstm.weight_ih_l0":
                # Transfer the 36 flow feature channels directly
                base_in_features = v.shape[1]
                model_dict[k][:, :base_in_features] = v
                # Initialize new packet channels with small weights
                nn.init.normal_(model_dict[k][:, base_in_features:], mean=0.0, std=0.01)
            elif k == "state_decoder.5.weight":
                base_out_features = v.shape[0]
                model_dict[k][:base_out_features, :] = v
                nn.init.normal_(model_dict[k][base_out_features:, :], mean=0.0, std=0.01)
            elif k == "state_decoder.5.bias":
                base_out_features = v.shape[0]
                model_dict[k][:base_out_features] = v
                nn.init.zeros_(model_dict[k][base_out_features:])
            elif k.startswith("infiltration_head."):
                risk_k = k.replace("infiltration_head.", "risk_head.")
                if risk_k in model_dict and model_dict[risk_k].shape == v.shape:
                    model_dict[risk_k] = v
            elif k in model_dict and model_dict[k].shape == v.shape:
                model_dict[k] = v

        model.load_state_dict(model_dict)

    return model

