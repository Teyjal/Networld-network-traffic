"""
NetWorld Unified Network Scaler
Combines the existing 36-flow-feature scaler with the 15-packet-feature scaler
to normalize and denormalize the 51-dimensional Unified Network State.

Ensures 100% backward compatibility:
- Flow features (0..35) use the exact parameters from models/networld_combined_temporal_scaler.pkl
- Packet features (36..50) use calibrated empirical network baseline scaling parameters.
"""

import os
import sys
import pickle
import numpy as np
from typing import Dict, Any, Tuple, Optional

workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

try:
    from ml.feature_schema import FEATURE_NAMES, NUM_FEATURES
    from ml.packet_schema import (
        PACKET_FEATURE_NAMES,
        NUM_PACKET_FEATURES,
        UNIFIED_FEATURE_NAMES,
        NUM_UNIFIED_FEATURES,
        PACKET_SCALER_BASELINE,
        validate_and_sanitize_packet_array,
    )
except ImportError:
    from feature_schema import FEATURE_NAMES, NUM_FEATURES
    from packet_schema import (
        PACKET_FEATURE_NAMES,
        NUM_PACKET_FEATURES,
        UNIFIED_FEATURE_NAMES,
        NUM_UNIFIED_FEATURES,
        PACKET_SCALER_BASELINE,
        validate_and_sanitize_packet_array,
    )


class LightweightStandardScaler:
    """Lightweight scaler drop-in replacement when scikit-learn is not installed."""
    def __init__(self, mean=None, scale=None, var=None):
        self.mean_ = mean
        self.scale_ = scale
        self.var_ = var
        self.n_features_in_ = len(mean) if mean is not None else 36


class SafeScalerUnpickler(pickle.Unpickler):
    """Safely unpickles StandardScaler instances without requiring scikit-learn / scipy."""
    def find_class(self, module, name):
        if "StandardScaler" in name:
            return LightweightStandardScaler
        try:
            return super().find_class(module, name)
        except Exception:
            return LightweightStandardScaler


class UnifiedNetworkScaler:
    """
    Unified Scaler for 51-dimensional network state (36 Flow + 15 Packet).
    """

    def __init__(
        self,
        flow_scaler_path: str = "models/networld_combined_temporal_scaler.pkl",
        unified_scaler_path: str = "models/networld_unified_scaler.pkl",
    ):
        self.flow_scaler_path = flow_scaler_path
        self.unified_scaler_path = unified_scaler_path

        # Load original 36-feature flow scaler
        if not os.path.exists(flow_scaler_path):
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cand1 = os.path.join(workspace_root, flow_scaler_path)
            cand2 = os.path.join(backend_dir, flow_scaler_path)
            if os.path.exists(cand1):
                self.flow_scaler_path = cand1
            elif os.path.exists(cand2):
                self.flow_scaler_path = cand2
            else:
                raise FileNotFoundError(f"Original flow scaler not found at: {flow_scaler_path}")

        with open(self.flow_scaler_path, "rb") as f:
            try:
                self.flow_scaler = SafeScalerUnpickler(f).load()
            except Exception:
                f.seek(0)
                self.flow_scaler = pickle.load(f)

        self.flow_mean = np.asarray(self.flow_scaler.mean_, dtype=np.float32)
        self.flow_scale = np.asarray(self.flow_scaler.scale_, dtype=np.float32)
        self.flow_scale = np.where(self.flow_scale == 0.0, 1.0, self.flow_scale)

        # Build packet scaler parameters from baseline network specifications
        self.packet_mean = np.zeros(NUM_PACKET_FEATURES, dtype=np.float32)
        self.packet_scale = np.ones(NUM_PACKET_FEATURES, dtype=np.float32)

        for idx, name in enumerate(PACKET_FEATURE_NAMES):
            mean_val, std_val = PACKET_SCALER_BASELINE.get(name, (0.0, 1.0))
            self.packet_mean[idx] = mean_val
            self.packet_scale[idx] = std_val if std_val > 0.0 else 1.0

        # Unified 51-D vectors
        self.unified_mean = np.concatenate([self.flow_mean, self.packet_mean])
        self.unified_scale = np.concatenate([self.flow_scale, self.packet_scale])

    def normalize_flow(self, raw_flow: np.ndarray) -> np.ndarray:
        """Normalizes 36 flow features using original scaler."""
        arr = np.nan_to_num(np.asarray(raw_flow, dtype=np.float32))
        return (arr - self.flow_mean) / self.flow_scale

    def denormalize_flow(self, scaled_flow: np.ndarray) -> np.ndarray:
        """Denormalizes 36 flow features to physical units."""
        arr = np.nan_to_num(np.asarray(scaled_flow, dtype=np.float32))
        return (arr * self.flow_scale) + self.flow_mean

    def normalize_packet(self, raw_packet: np.ndarray) -> np.ndarray:
        """Normalizes 15 packet features."""
        arr = validate_and_sanitize_packet_array(np.asarray(raw_packet, dtype=np.float32))
        return (arr - self.packet_mean) / self.packet_scale

    def denormalize_packet(self, scaled_packet: np.ndarray) -> np.ndarray:
        """Denormalizes 15 packet features to physical units."""
        arr = np.nan_to_num(np.asarray(scaled_packet, dtype=np.float32))
        return (arr * self.packet_scale) + self.packet_mean

    def normalize_unified(
        self,
        raw_flow: np.ndarray,
        raw_packet: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Normalizes flow and packet features and concatenates into Unified Network State.
        If raw_packet is None, imputes neutral zeros in packet dimensions.
        Returns array of shape (..., 51).
        """
        scaled_flow = self.normalize_flow(raw_flow)

        if raw_packet is not None:
            scaled_packet = self.normalize_packet(raw_packet)
        else:
            # Impute neutral zero-standardized vector (corresponds to mean baseline)
            shape = list(scaled_flow.shape)
            shape[-1] = NUM_PACKET_FEATURES
            scaled_packet = np.zeros(shape, dtype=np.float32)

        return np.concatenate([scaled_flow, scaled_packet], axis=-1)

    def denormalize_unified(self, scaled_unified: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Splits and denormalizes 51-D unified array back to (raw_flow 36-D, raw_packet 15-D).
        """
        arr = np.asarray(scaled_unified, dtype=np.float32)
        if arr.shape[-1] == NUM_FEATURES:
            return self.denormalize_flow(arr), np.zeros((*arr.shape[:-1], NUM_PACKET_FEATURES), dtype=np.float32)

        scaled_flow = arr[..., :NUM_FEATURES]
        scaled_packet = arr[..., NUM_FEATURES:NUM_UNIFIED_FEATURES]

        raw_flow = self.denormalize_flow(scaled_flow)
        raw_packet = self.denormalize_packet(scaled_packet)
        return raw_flow, raw_packet

    def save_unified_scaler(self, output_path: Optional[str] = None) -> str:
        """Saves unified scaler to disk."""
        target = output_path or self.unified_scaler_path
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        with open(target, "wb") as f:
            pickle.dump(self, f)
        return target


# Helper function to get or create unified scaler
def get_unified_scaler() -> UnifiedNetworkScaler:
    return UnifiedNetworkScaler()
