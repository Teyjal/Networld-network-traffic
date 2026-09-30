"""
Model and Scaler Loader Module
Handles loading and singleton management of the trained PyTorch NetWorldLSTM model and StandardScaler.

Artifact Targets:
- Model Weights: models/networld_combined_temporal_lstm_best.pt
- Scaler Object: models/networld_combined_temporal_scaler.pkl

Input Specifications:
- Input shape: (batch_size, 20, 36) -> 20 sequential network flows x 36 numerical features
- Architecture: 2-layer LSTM (hidden=128, dropout=0.2) + Linear(128, 64) + ReLU + Dropout(0.2) + Linear(64, 1)
- Output: 1 raw logit per sequence (apply sigmoid for probability)
"""

import os
import pickle
from typing import Tuple, Optional, Any

import torch
import torch.nn as nn


class NetWorldLSTM(nn.Module):
    """
    NetWorld Temporal LSTM Model Architecture.
    Matches trained PyTorch architecture from ml/train_combined_temporal_world_model.py.
    """

    def __init__(
        self,
        input_size: int = 36,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for temporal sequences of shape (batch_size, 20, 36).
        Returns raw logit of shape (batch_size, 1).
        """
        output, _ = self.lstm(x)
        # Select last hidden state of sequence window
        last_hidden = output[:, -1, :]
        logits = self.infiltration_head(last_hidden)
        return logits


class ModelLoader:
    """
    Singleton Loader for PyTorch model and StandardScaler.
    Ensures model weights and scaler are loaded only once into memory.
    """

    _instance: Optional["ModelLoader"] = None
    _model: Optional[NetWorldLSTM] = None
    _scaler: Optional[Any] = None
    _device: Optional[torch.device] = None

    DEFAULT_MODEL_PATH = "models/networld_combined_temporal_lstm_best.pt"
    DEFAULT_SCALER_PATH = "models/networld_combined_temporal_scaler.pkl"
    SEQUENCE_LENGTH = 20
    NUM_FEATURES = 36

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        scaler_path: str = DEFAULT_SCALER_PATH,
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path

    @classmethod
    def get_device(cls) -> torch.device:
        """
        Detect best available hardware accelerator:
        - Apple Silicon GPU (MPS)
        - CUDA GPU
        - CPU fallback
        """
        if cls._device is None:
            if torch.backends.mps.is_available():
                cls._device = torch.device("mps")
            elif torch.cuda.is_available():
                cls._device = torch.device("cuda")
            else:
                cls._device = torch.device("cpu")
        return cls._device

    def check_artifacts_exist(self) -> dict:
        """Verify presence of model weights and scaler files on disk."""
        abs_model = self._resolve_path(self.model_path)
        abs_scaler = self._resolve_path(self.scaler_path)
        return {
            "model_exists": os.path.exists(abs_model),
            "model_path": abs_model,
            "scaler_exists": os.path.exists(abs_scaler),
            "scaler_path": abs_scaler,
            "expected_input_shape": (self.SEQUENCE_LENGTH, self.NUM_FEATURES),
            "device": str(self.get_device()),
            "loaded": self.is_loaded,
        }

    def _resolve_path(self, rel_path: str) -> str:
        """Resolves relative file paths against workspace root or current directory."""
        if os.path.isabs(rel_path):
            return rel_path

        # Try relative to workspace root (2 levels up from backend/app/model)
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        workspace_root = os.path.dirname(backend_dir)
        candidate1 = os.path.join(workspace_root, rel_path)
        if os.path.exists(candidate1):
            return candidate1

        candidate2 = os.path.join(backend_dir, rel_path)
        if os.path.exists(candidate2):
            return candidate2

        return os.path.abspath(rel_path)

    def load(self) -> Tuple[NetWorldLSTM, Any, torch.device]:
        """
        Loads scaler and PyTorch LSTM model once and caches instances.
        Sets model to evaluation mode (`model.eval()`).
        """
        if self._model is not None and self._scaler is not None:
            return self._model, self._scaler, self.get_device()

        device = self.get_device()
        abs_model_path = self._resolve_path(self.model_path)
        abs_scaler_path = self._resolve_path(self.scaler_path)

        if not os.path.exists(abs_scaler_path):
            raise FileNotFoundError(f"Scaler file missing at: {abs_scaler_path}")
        if not os.path.exists(abs_model_path):
            raise FileNotFoundError(f"Model file missing at: {abs_model_path}")

        # 1. Load StandardScaler
        with open(abs_scaler_path, "rb") as f:
            self._scaler = pickle.load(f)

        # 2. Instantiate Architecture
        model = NetWorldLSTM(
            input_size=self.NUM_FEATURES,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
        )

        # 3. Load Checkpoint Safely
        checkpoint = torch.load(abs_model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()  # Enable evaluation mode

        self._model = model
        return self._model, self._scaler, device

    @property
    def is_loaded(self) -> bool:
        """Check if model and scaler are currently loaded in memory."""
        return self._model is not None and self._scaler is not None


def get_model_and_scaler() -> Tuple[NetWorldLSTM, Any, torch.device]:
    """Reusable helper to access loaded singleton model, scaler, and target compute device."""
    loader = ModelLoader()
    return loader.load()
