"""
NetWorld Packet Feature Schema & Unified Network State Specification
Defines the lightweight packet-derived feature set, validation rules, units,
and the Unified Network State (36 Flow + 15 Packet = 51 Features).
"""

from typing import List, Dict, Any, Tuple
import numpy as np

# 15 Defensible Packet-Derived Features
PACKET_FEATURE_NAMES: List[str] = [
    # 1. TTL Statistics
    "Pkt TTL Mean",         # Mean Time-To-Live (operating system hop distance / route alteration)
    "Pkt TTL Min",          # Minimum TTL
    "Pkt TTL Max",          # Maximum TTL
    
    # 2. TCP Window Statistics
    "TCP Win Mean",         # Mean advertised TCP window size in bytes
    "TCP Win Min",          # Min advertised TCP window size (detects zero-window starvation)
    "TCP Win Max",          # Max advertised TCP window size
    
    # 3. Payload-Size Statistics
    "Payload Len Mean",     # Mean application payload size in bytes
    "Payload Len Std",      # Standard deviation of payload size (fixed-size beacons vs normal)
    "Payload Len Max",      # Maximum payload size in bytes
    "Payload Zero Ratio",   # Proportion of packets with 0 payload bytes (SYN scans / probes)
    
    # 4. Retransmission Indicators
    "Retransmission Cnt",   # Count of duplicate/retransmitted TCP packets
    "Retransmission Ratio", # Ratio of retransmissions to total packet count
    
    # 5. Port-Scan & Probe Indicators
    "Unique Dst Ports",     # Distinct destination ports contacted within the temporal window
    "SYN Only Ratio",       # Ratio of TCP packets with SYN flag set and ACK unset
    "RST Ratio",            # Ratio of TCP packets with RST flag set (rejected probe indicator)
]

NUM_PACKET_FEATURES: int = len(PACKET_FEATURE_NAMES)  # 15

# Feature Indices Map for Packet Features
PACKET_FEATURE_INDEX_MAP: Dict[str, int] = {name: idx for idx, name in enumerate(PACKET_FEATURE_NAMES)}

# Unified Feature list (36 flow + 15 packet = 51)
FLOW_FEATURE_NAMES: List[str] = [
    "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Mean", "Bwd Pkt Len Mean",
    "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Flow IAT Min", "Fwd IAT Mean", "Bwd IAT Mean", "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Mean", "Pkt Len Std", "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt",
    "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt", "Down/Up Ratio", "Pkt Size Avg",
    "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts", "Active Mean",
    "Active Std", "Idle Mean", "Idle Std",
]
UNIFIED_FEATURE_NAMES: List[str] = FLOW_FEATURE_NAMES + PACKET_FEATURE_NAMES
NUM_UNIFIED_FEATURES: int = len(UNIFIED_FEATURE_NAMES)  # 51
UNIFIED_FEATURE_INDEX_MAP: Dict[str, int] = {name: idx for idx, name in enumerate(UNIFIED_FEATURE_NAMES)}


# Specification of physical bounds, units, and default neutral values
PACKET_FEATURE_SPEC: Dict[str, Dict[str, Any]] = {
    "Pkt TTL Mean": {
        "unit": "hops/ttl",
        "min": 0.0,
        "max": 255.0,
        "default": 64.0,
        "description": "Mean IP packet Time-To-Live"
    },
    "Pkt TTL Min": {
        "unit": "hops/ttl",
        "min": 0.0,
        "max": 255.0,
        "default": 64.0,
        "description": "Minimum IP packet Time-To-Live"
    },
    "Pkt TTL Max": {
        "unit": "hops/ttl",
        "min": 0.0,
        "max": 255.0,
        "default": 64.0,
        "description": "Maximum IP packet Time-To-Live"
    },
    "TCP Win Mean": {
        "unit": "bytes",
        "min": 0.0,
        "max": 65535.0,
        "default": 14600.0,
        "description": "Mean TCP receive window advertisement"
    },
    "TCP Win Min": {
        "unit": "bytes",
        "min": 0.0,
        "max": 65535.0,
        "default": 1024.0,
        "description": "Minimum TCP receive window advertisement"
    },
    "TCP Win Max": {
        "unit": "bytes",
        "min": 0.0,
        "max": 65535.0,
        "default": 29200.0,
        "description": "Maximum TCP receive window advertisement"
    },
    "Payload Len Mean": {
        "unit": "bytes",
        "min": 0.0,
        "max": 65535.0,
        "default": 0.0,
        "description": "Mean application layer payload length"
    },
    "Payload Len Std": {
        "unit": "bytes",
        "min": 0.0,
        "max": 65535.0,
        "default": 0.0,
        "description": "Standard deviation of payload length"
    },
    "Payload Len Max": {
        "unit": "bytes",
        "min": 0.0,
        "max": 65535.0,
        "default": 0.0,
        "description": "Maximum application layer payload length"
    },
    "Payload Zero Ratio": {
        "unit": "ratio [0-1]",
        "min": 0.0,
        "max": 1.0,
        "default": 0.0,
        "description": "Fraction of packets with zero application payload"
    },
    "Retransmission Cnt": {
        "unit": "packets",
        "min": 0.0,
        "max": 100000.0,
        "default": 0.0,
        "description": "Total duplicate/retransmitted packets observed"
    },
    "Retransmission Ratio": {
        "unit": "ratio [0-1]",
        "min": 0.0,
        "max": 1.0,
        "default": 0.0,
        "description": "Ratio of retransmitted packets to total packets"
    },
    "Unique Dst Ports": {
        "unit": "count",
        "min": 0.0,
        "max": 65535.0,
        "default": 1.0,
        "description": "Distinct destination ports targeted in time window"
    },
    "SYN Only Ratio": {
        "unit": "ratio [0-1]",
        "min": 0.0,
        "max": 1.0,
        "default": 0.0,
        "description": "Ratio of SYN-only packets (SYN=1, ACK=0) indicative of port scan"
    },
    "RST Ratio": {
        "unit": "ratio [0-1]",
        "min": 0.0,
        "max": 1.0,
        "default": 0.0,
        "description": "Ratio of RST packets indicative of closed port rejection"
    },
}

# Baseline empirical scaling parameters for the 15 packet features (for standardized z-score)
PACKET_SCALER_BASELINE: Dict[str, Tuple[float, float]] = {
    # feature: (mean, std)
    "Pkt TTL Mean": (64.0, 32.0),
    "Pkt TTL Min": (60.0, 30.0),
    "Pkt TTL Max": (68.0, 32.0),
    "TCP Win Mean": (14600.0, 10000.0),
    "TCP Win Min": (8192.0, 8000.0),
    "TCP Win Max": (29200.0, 16000.0),
    "Payload Len Mean": (256.0, 350.0),
    "Payload Len Std": (128.0, 200.0),
    "Payload Len Max": (1024.0, 800.0),
    "Payload Zero Ratio": (0.25, 0.35),
    "Retransmission Cnt": (0.5, 2.0),
    "Retransmission Ratio": (0.02, 0.08),
    "Unique Dst Ports": (1.2, 2.5),
    "SYN Only Ratio": (0.05, 0.15),
    "RST Ratio": (0.03, 0.10),
}


def validate_and_sanitize_packet_dict(raw_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    Validates units, clips out-of-range values, and fills missing keys with neutral defaults.
    """
    sanitized: Dict[str, float] = {}
    for name in PACKET_FEATURE_NAMES:
        spec = PACKET_FEATURE_SPEC[name]
        val = raw_dict.get(name, None)
        if val is None or (isinstance(val, (float, int)) and np.isnan(val)):
            val = spec["default"]
        try:
            val_float = float(val)
        except (ValueError, TypeError):
            val_float = spec["default"]
            
        # Clip to physical bounds
        val_float = max(spec["min"], min(spec["max"], val_float))
        sanitized[name] = float(val_float)
    return sanitized


def validate_and_sanitize_packet_array(arr: np.ndarray) -> np.ndarray:
    """
    Validates and sanitizes a 2D numpy array of shape (N, 15) of packet features.
    """
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != NUM_PACKET_FEATURES:
        raise ValueError(
            f"Expected {NUM_PACKET_FEATURES} packet features, got array with {arr.shape[1]} columns"
        )
    cleaned = np.copy(arr)
    for col_idx, name in enumerate(PACKET_FEATURE_NAMES):
        spec = PACKET_FEATURE_SPEC[name]
        col = cleaned[:, col_idx]
        col = np.nan_to_num(col, nan=spec["default"], posinf=spec["max"], neginf=spec["min"])
        cleaned[:, col_idx] = np.clip(col, spec["min"], spec["max"])
    return cleaned
