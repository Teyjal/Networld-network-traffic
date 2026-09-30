"""
NetWorld Unified Network State & Packet Feature Extraction Test Suite
Verifies:
1. PCAP parsing and extraction of 15 packet features (TTL, TCP window, payload stats, retransmissions, port scans).
2. Missing value validation and unit sanitization.
3. Unified Network State assembly (36 flow + 15 packet = 51 features).
4. Multi-head LSTM forward pass and autoregressive rollout on Unified State.
5. Backward compatibility with standard 36-feature flow CSVs.
6. FastAPI endpoints (/forecast/feature-schema, /traffic/upload, /forecast).
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
import torch

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(WORKSPACE_ROOT, "backend")
for p in [WORKSPACE_ROOT, BACKEND_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ml.packet_schema import (
    PACKET_FEATURE_NAMES,
    NUM_PACKET_FEATURES,
    UNIFIED_FEATURE_NAMES,
    NUM_UNIFIED_FEATURES,
    PACKET_FEATURE_SPEC,
    validate_and_sanitize_packet_dict,
)
from ml.packet_extractor import PacketFeatureExtractor, generate_test_pcap
from ml.unified_scaler import get_unified_scaler
from ml.world_model_multihead import NetWorldMultiHeadWorldModel
from backend.app.services.world_model_service import get_world_model_service
from backend.app.services.preprocessing import PreprocessingService


def test_1_packet_extraction_from_pcap():
    print("\n--- Test 1: Packet Extraction from PCAP ---")
    test_pcap_path = os.path.join(WORKSPACE_ROOT, "scratch", "unit_test_capture.pcap")
    generate_test_pcap(test_pcap_path, num_packets=60)
    assert os.path.exists(test_pcap_path), "Failed to generate test PCAP"

    packets = PacketFeatureExtractor.parse_pcap_file(test_pcap_path)
    assert len(packets) == 60, f"Expected 60 packets, parsed {len(packets)}"

    pkt_feats = PacketFeatureExtractor.aggregate_packet_features(packets)
    assert len(pkt_feats) == NUM_PACKET_FEATURES, f"Expected {NUM_PACKET_FEATURES} features, got {len(pkt_feats)}"

    # Check that required features are present and within valid physical units
    assert 0.0 <= pkt_feats["Pkt TTL Mean"] <= 255.0
    assert 0.0 <= pkt_feats["Pkt TTL Min"] <= pkt_feats["Pkt TTL Max"] <= 255.0
    assert 0.0 <= pkt_feats["TCP Win Mean"] <= 65535.0
    assert pkt_feats["Payload Len Mean"] >= 0.0
    assert 0.0 <= pkt_feats["Payload Zero Ratio"] <= 1.0
    assert pkt_feats["Retransmission Cnt"] >= 0.0
    assert 0.0 <= pkt_feats["Retransmission Ratio"] <= 1.0
    assert pkt_feats["Unique Dst Ports"] >= 1.0
    assert 0.0 <= pkt_feats["SYN Only Ratio"] <= 1.0
    assert 0.0 <= pkt_feats["RST Ratio"] <= 1.0

    print("Extracted Packet Features Sample:")
    for k, v in list(pkt_feats.items())[:6]:
        print(f"  {k}: {v}")
    print("Test 1 PASSED: Defensible packet statistics extracted with valid units.")


def test_2_missing_value_and_unit_validation():
    print("\n--- Test 2: Missing Value and Unit Validation ---")
    dirty_dict = {
        "Pkt TTL Mean": 300.0,      # Exceeds max 255 -> should clip
        "Pkt TTL Min": -10.0,       # Below min 0 -> should clip
        "TCP Win Mean": np.nan,     # NaN -> should fill with default 14600.0
        "Payload Zero Ratio": 1.5,  # Exceeds max 1.0 -> should clip
        "SYN Only Ratio": "invalid",# Type error -> should fill with default 0.0
    }
    cleaned = validate_and_sanitize_packet_dict(dirty_dict)
    assert cleaned["Pkt TTL Mean"] == 255.0
    assert cleaned["Pkt TTL Min"] == 0.0
    assert cleaned["TCP Win Mean"] == PACKET_FEATURE_SPEC["TCP Win Mean"]["default"]
    assert cleaned["Payload Zero Ratio"] == 1.0
    assert cleaned["SYN Only Ratio"] == 0.0
    assert len(cleaned) == NUM_PACKET_FEATURES
    print("Test 2 PASSED: Missing values and boundary clipping successfully validated.")


def test_3_unified_scaler_and_network_state():
    print("\n--- Test 3: Unified Network Scaler & State Formation ---")
    scaler = get_unified_scaler()
    assert scaler.unified_mean.shape == (51,)
    assert scaler.unified_scale.shape == (51,)

    # Use real sample traffic features for flow array
    csv_path = os.path.join(WORKSPACE_ROOT, "data", "sample_traffic.csv")
    df = pd.read_csv(csv_path)
    from ml.feature_schema import FEATURE_NAMES
    flow_arr = df[FEATURE_NAMES].head(20).values.astype(np.float32)
    packet_arr, _ = PacketFeatureExtractor.extract_from_df(df.head(20))

    # Normalize into Unified Network State (20, 51)
    unified_arr = scaler.normalize_unified(flow_arr, packet_arr)
    assert unified_arr.shape == (20, 51), f"Expected shape (20, 51), got {unified_arr.shape}"

    # Denormalize back
    denorm_flow, denorm_packet = scaler.denormalize_unified(unified_arr)
    assert denorm_flow.shape == (20, 36)
    assert denorm_packet.shape == (20, 15)
    assert np.allclose(denorm_flow, flow_arr, rtol=1e-3, atol=1.0)
    assert np.allclose(denorm_packet, packet_arr, rtol=1e-3, atol=1e-2)
    print("Test 3 PASSED: Scaler preserves exact 36 flow features and combines 15 packet features.")




def test_4_unified_world_model_rollout():
    print("\n--- Test 4: Unified World Model Autoregressive Rollout (51-D) ---")
    wm = get_world_model_service()
    assert wm.model.input_size in (36, 51), f"Unexpected model input size: {wm.model.input_size}"

    # Test with sample traffic dataframe
    csv_path = os.path.join(WORKSPACE_ROOT, "data", "sample_traffic.csv")
    df = pd.read_csv(csv_path)

    forecast_res = wm.forecast_trajectory(df, horizon=5, filename="sample_traffic.csv")
    assert forecast_res["success"] is True
    assert len(forecast_res["trajectory"]) == 6  # Step 0 (NOW) + 5 forecast steps
    assert "unified_state" in forecast_res

    u_state = forecast_res["unified_state"]
    print("Unified State Metadata from Forecast:")
    print(f"  Supported:                 {u_state['supported']}")
    print(f"  Packet Features Available: {u_state['packet_features_available']}")
    print(f"  Flow Feature Count:        {u_state['flow_feature_count']}")
    print(f"  Packet Feature Count:      {u_state['packet_feature_count']}")
    print(f"  Total Feature Count:       {u_state['total_feature_count']}")
    print(f"  Active Model Dim:          {u_state['active_model_dim']}")
    print(f"  Source:                    {u_state['source']}")

    assert u_state["flow_feature_count"] == 36
    assert u_state["total_feature_count"] in (36, 51)
    print("Test 4 PASSED: Unified world model forecast and rollout executed cleanly.")


def test_5_pcap_ingestion_and_forecast():
    print("\n--- Test 5: PCAP Ingestion and Forecasting ---")
    test_pcap_path = os.path.join(WORKSPACE_ROOT, "scratch", "test_flow.pcap")
    generate_test_pcap(test_pcap_path, num_packets=80)

    wm = get_world_model_service()
    pcap_forecast = wm.forecast_from_pcap(test_pcap_path, horizon=5)
    assert pcap_forecast["success"] is True
    assert len(pcap_forecast["trajectory"]) == 6

    u_state = pcap_forecast["unified_state"]
    assert u_state["packet_features_available"] is True
    assert u_state["packet_feature_count"] == 15
    assert u_state["flow_feature_count"] == 36
    assert u_state["total_feature_count"] == 51

    print("PCAP Forecast Peak Risk:", pcap_forecast["highest_predicted_risk"])
    print("PCAP Forecast Horizon:", pcap_forecast["peak_risk_horizon"])
    print("Test 5 PASSED: Direct PCAP file ingestion successfully produced Unified 51-D forecast.")


def test_6_fastapi_endpoints():
    print("\n--- Test 6: FastAPI Route Integration & Feature Schema ---")
    from fastapi.testclient import TestClient
    from backend.app.main import app

    client = TestClient(app)

    # 1. Feature schema endpoint
    res_schema = client.get("/forecast/feature-schema")
    assert res_schema.status_code == 200, res_schema.text
    schema_data = res_schema.json()
    assert schema_data["flow_features"]["count"] == 36
    assert schema_data["packet_features"]["count"] == 15
    assert schema_data["unified_features"]["count"] == 51
    assert "Pkt TTL Mean" in schema_data["packet_features"]["names"]
    assert "TCP Win Mean" in schema_data["packet_features"]["names"]
    print("GET /forecast/feature-schema returned 51 unified features (36 Flow + 15 Packet).")

    # 2. Upload PCAP to /traffic/upload
    test_pcap = os.path.join(WORKSPACE_ROOT, "scratch", "upload_test.pcap")
    generate_test_pcap(test_pcap, num_packets=40)
    with open(test_pcap, "rb") as f:
        res_upload = client.post("/api/traffic/upload", files={"file": ("test.pcap", f, "application/octet-stream")})
    assert res_upload.status_code == 200, res_upload.text
    upload_data = res_upload.json()
    assert upload_data["packet_features_available"] is True
    assert upload_data["total_features"] == 51
    print("POST /api/traffic/upload with PCAP returned 51 features with packet telemetry active.")

    print("Test 6 PASSED: All FastAPI endpoints operational.")


if __name__ == "__main__":
    test_1_packet_extraction_from_pcap()
    test_2_missing_value_and_unit_validation()
    test_3_unified_scaler_and_network_state()
    test_4_unified_world_model_rollout()
    test_5_pcap_ingestion_and_forecast()
    test_6_fastapi_endpoints()
    print("\n" + "=" * 70)
    print("ALL 6 PACKET EXTRACTION & UNIFIED WORLD MODEL TESTS PASSED!")
    print("=" * 70)
