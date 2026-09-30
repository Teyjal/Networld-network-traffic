"""
NetWorld Machine Learning Runtime Package for Deployment.
Provides lightweight feature schema, packet extractor, scaler, and model inference.
"""

from ml.feature_schema import (
    FEATURE_NAMES,
    NUM_FEATURES,
    SEQUENCE_LENGTH,
    classify_risk_level,
    DEFENSE_PERTURBATION_SPEC,
    UNIFIED_FEATURE_NAMES,
    NUM_UNIFIED_FEATURES,
    PACKET_FEATURE_NAMES,
    NUM_PACKET_FEATURES,
)

__all__ = [
    "FEATURE_NAMES",
    "NUM_FEATURES",
    "SEQUENCE_LENGTH",
    "classify_risk_level",
    "DEFENSE_PERTURBATION_SPEC",
    "UNIFIED_FEATURE_NAMES",
    "NUM_UNIFIED_FEATURES",
    "PACKET_FEATURE_NAMES",
    "NUM_PACKET_FEATURES",
]
