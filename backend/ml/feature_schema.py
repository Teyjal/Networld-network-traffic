"""
NetWorld Feature Schema Specification
Single source of truth for the 36 numerical network flow features,
operational risk triage thresholds, and defense perturbation rules.
Shared across training, inference, autoregressive rollout, SHAP explainability, and FastAPI.
"""

from typing import List, Dict, Any

# Exact 36 feature names in strict chronological sequence order
FEATURE_NAMES: List[str] = [
    "Dst Port",
    "Protocol",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Fwd Pkt Len Mean",
    "Bwd Pkt Len Mean",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "Fwd Pkts/s",
    "Bwd Pkts/s",
    "Pkt Len Mean",
    "Pkt Len Std",
    "FIN Flag Cnt",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
    "URG Flag Cnt",
    "Down/Up Ratio",
    "Pkt Size Avg",
    "Init Fwd Win Byts",
    "Init Bwd Win Byts",
    "Fwd Act Data Pkts",
    "Active Mean",
    "Active Std",
    "Idle Mean",
    "Idle Std",
]

NUM_FEATURES: int = len(FEATURE_NAMES)
SEQUENCE_LENGTH: int = 20

# Feature indices map for rapid O(1) lookups
FEATURE_INDEX_MAP: Dict[str, int] = {name: idx for idx, name in enumerate(FEATURE_NAMES)}

# Lightweight Packet-Derived Features & Unified Network State Specification
try:
    from ml.packet_schema import (
        PACKET_FEATURE_NAMES,
        NUM_PACKET_FEATURES,
        PACKET_FEATURE_INDEX_MAP,
        PACKET_FEATURE_SPEC,
        validate_and_sanitize_packet_dict,
        validate_and_sanitize_packet_array,
    )
except ImportError:
    from packet_schema import (
        PACKET_FEATURE_NAMES,
        NUM_PACKET_FEATURES,
        PACKET_FEATURE_INDEX_MAP,
        PACKET_FEATURE_SPEC,
        validate_and_sanitize_packet_dict,
        validate_and_sanitize_packet_array,
    )

UNIFIED_FEATURE_NAMES: List[str] = FEATURE_NAMES + PACKET_FEATURE_NAMES
NUM_UNIFIED_FEATURES: int = len(UNIFIED_FEATURE_NAMES)  # 36 + 15 = 51
UNIFIED_FEATURE_INDEX_MAP: Dict[str, int] = {name: idx for idx, name in enumerate(UNIFIED_FEATURE_NAMES)}


# Operational UI triage thresholds (documented as operational heuristics)
RISK_THRESHOLDS: Dict[str, Any] = {
    "LOW_MAX": 0.30,
    "MEDIUM_MAX": 0.70,
    "LABELS": {
        "LOW": "Low Infiltration Risk (< 30%)",
        "MEDIUM": "Guarded Infiltration Risk (30% - 70%)",
        "HIGH": "Critical Infiltration Risk (>= 70%)",
    },
    "DISCLAIMER": (
        "Risk categories (LOW, MEDIUM, HIGH) are application-defined operational triage thresholds "
        "designed for security operations priority queueing, not validated physical probabilities."
    )
}

def classify_risk_level(prob: float) -> str:
    """Classifies an infiltration probability into operational triage categories."""
    if prob < RISK_THRESHOLDS["LOW_MAX"]:
        return "LOW"
    elif prob < RISK_THRESHOLDS["MEDIUM_MAX"]:
        return "MEDIUM"
    else:
        return "HIGH"


# Defense action perturbation mapping: defines which features are modified and how
# for counterfactual what-if simulation.
DEFENSE_PERTURBATION_SPEC: Dict[str, Dict[str, Any]] = {
    "block_ip": {
        "action_name": "Block IP",
        "description": "Drop all inbound and outbound traffic matching the malicious IP",
        "affected_features": [
            "Flow Pkts/s", "Fwd Pkts/s", "Bwd Pkts/s", "Flow Byts/s",
            "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
            "Fwd Act Data Pkts", "SYN Flag Cnt", "ACK Flag Cnt", "RST Flag Cnt", "FIN Flag Cnt", "PSH Flag Cnt",
            "Retransmission Cnt", "Retransmission Ratio", "Payload Len Mean", "Payload Len Max",
            "TCP Win Mean", "SYN Only Ratio"
        ],
        "operation": "zero",
        "supported": True
    },
    "quarantine_host": {
        "action_name": "Quarantine Host",
        "description": "Sever all non-essential ingress and egress connections to isolate the compromised host endpoint",
        "affected_features": [
            "Flow Pkts/s", "Fwd Pkts/s", "Bwd Pkts/s", "Flow Byts/s",
            "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
            "SYN Flag Cnt", "ACK Flag Cnt", "RST Flag Cnt", "FIN Flag Cnt", "PSH Flag Cnt", "URG Flag Cnt",
            "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts",
            "Active Mean", "Idle Mean", "TCP Win Mean", "Payload Len Mean",
            "Payload Len Max", "Retransmission Cnt", "Retransmission Ratio", "SYN Only Ratio", "RST Ratio"
        ],
        "operation": "zero",
        "supported": True
    },
    "isolate_host": {
        "action_name": "Quarantine Host",
        "description": "Sever all non-essential ingress and egress connections to isolate the compromised host endpoint",
        "affected_features": [
            "Flow Pkts/s", "Fwd Pkts/s", "Bwd Pkts/s", "Flow Byts/s",
            "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
            "SYN Flag Cnt", "ACK Flag Cnt", "RST Flag Cnt", "FIN Flag Cnt", "PSH Flag Cnt", "URG Flag Cnt",
            "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts",
            "Active Mean", "Idle Mean", "TCP Win Mean", "Payload Len Mean",
            "Payload Len Max", "Retransmission Cnt", "Retransmission Ratio", "SYN Only Ratio", "RST Ratio"
        ],
        "operation": "zero",
        "supported": True
    },
    "close_port": {
        "action_name": "Close Port",
        "description": "Block inbound and outbound requests on targeted destination service port",
        "affected_features": [
            "Flow Pkts/s", "Fwd Pkts/s", "Bwd Pkts/s", "Flow Byts/s",
            "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
            "SYN Flag Cnt", "ACK Flag Cnt", "RST Flag Cnt", "Init Fwd Win Byts",
            "TCP Win Mean", "Payload Len Mean", "Payload Len Max", "SYN Only Ratio", "RST Ratio"
        ],
        "operation": "zero",
        "supported": True
    },
    "rate_limit": {
        "action_name": "Rate Limit",
        "description": "Throttle packet transmission rate and bandwidth to quench flood attacks",
        "affected_features": [
            "Flow Pkts/s", "Fwd Pkts/s", "Bwd Pkts/s", "Flow Byts/s",
            "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Payload Len Mean", "Payload Len Max",
            "Retransmission Cnt"
        ],
        "operation": "scale",
        "scale_factor": 0.1,  # 90% rate restriction
        "supported": True
    },
    "restrict_traffic": {
        "action_name": "Rate Limit",
        "description": "Throttle packet transmission rate and bandwidth to quench flood attacks",
        "affected_features": [
            "Flow Pkts/s", "Fwd Pkts/s", "Bwd Pkts/s", "Flow Byts/s",
            "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Payload Len Mean", "Payload Len Max",
            "Retransmission Cnt"
        ],
        "operation": "scale",
        "scale_factor": 0.1,
        "supported": True
    },
    "block_port": {
        "action_name": "Close Port",
        "description": "Block inbound and outbound requests on targeted destination service port",
        "affected_features": [
            "Flow Pkts/s", "Fwd Pkts/s", "Bwd Pkts/s", "Flow Byts/s",
            "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
            "SYN Flag Cnt", "ACK Flag Cnt", "RST Flag Cnt", "Init Fwd Win Byts",
            "TCP Win Mean", "Payload Len Mean", "Payload Len Max", "SYN Only Ratio", "RST Ratio"
        ],
        "operation": "zero",
        "supported": True
    },
    "block_protocol": {
        "action_name": "Block Protocol",
        "description": "Drop flows matching the targeted transport protocol",
        "affected_features": [
            "Flow Pkts/s", "Fwd Pkts/s", "Bwd Pkts/s", "Flow Byts/s",
            "Tot Fwd Pkts", "Tot Bwd Pkts"
        ],
        "operation": "zero",
        "supported": True
    }
}
