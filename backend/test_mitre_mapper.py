import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np
import pandas as pd
from app.services.mitre_mapper import AttackStageMapperService

def test_attack_stage_mapper():
    mapper = AttackStageMapperService()
    
    print("=== TEST 1: Reconnaissance Traffic (SYN scan, multi-port) ===")
    df_recon = pd.DataFrame({
        "Dst Port": [80, 443, 22, 8080, 21],
        "Flow Duration": [120, 150, 90, 110, 80],
        "Tot Fwd Pkts": [2, 2, 2, 2, 2],
        "Tot Bwd Pkts": [0, 0, 0, 0, 0],
        "TotLen Fwd Pkts": [40, 40, 40, 40, 40],
        "TotLen Bwd Pkts": [0, 0, 0, 0, 0],
        "SYN Flag Cnt": [1, 1, 1, 1, 1],
        "ACK Flag Cnt": [0, 0, 0, 0, 0],
        "RST Flag Cnt": [1, 1, 0, 1, 0],
        "Pkt Len Mean": [40, 40, 40, 40, 40],
    })
    pkt_recon = {
        "Pkt TTL Mean": 64.0,
        "TCP Win Mean": 1024.0,
        "Payload Len Mean": 0.0,
        "Payload Len Std": 0.0,
        "Payload Len Max": 0.0,
        "Retransmission Cnt": 0,
        "Retransmission Ratio": 0.0,
        "Unique Dst Ports": 5,
        "SYN Only Ratio": 1.0,
        "RST Ratio": 0.6,
    }
    
    res_recon = mapper.map_mitre_stage(
        risk_probability=0.28,
        df=df_recon,
        packet_features=pkt_recon
    )
    
    print(f"Predicted Stage: {res_recon['stage']}")
    print(f"Formatted Output: {res_recon['formatted_output']}")
    print(f"Supported Stages Count: {res_recon['supported_stages_count']}")
    for item in res_recon["progression"]:
        print(f"  [{'ACTIVE' if item['supported'] else 'INACTIVE'}] {item['stage']}: {item['confidence_percent']}% | Rules: {item['rules_satisfied']} | Tech: {item['technique_id']}")
        if item['supported']:
            print(f"     Evidence: {item['supporting_evidence']}")
    
    # Assertions
    assert res_recon["stage"] == "Reconnaissance"
    recon_item = next(p for p in res_recon["progression"] if p["stage"] == "Reconnaissance")
    assert recon_item["supported"] == True
    assert recon_item["confidence"] > 0.0
    # Stages without evidence must be unsupported with 0.0 confidence
    exfil_item = next(p for p in res_recon["progression"] if p["stage"] == "Exfiltration")
    assert exfil_item["supported"] == False
    assert exfil_item["confidence"] == 0.0
    
    print("\n=== TEST 2: Exfiltration Traffic (Massive outbound byte volume & MTU payloads) ===")
    df_exfil = pd.DataFrame({
        "Dst Port": [443, 443, 443],
        "Flow Duration": [15000, 18000, 22000],
        "Tot Fwd Pkts": [500, 600, 800],
        "Tot Bwd Pkts": [10, 12, 15],
        "TotLen Fwd Pkts": [750000, 900000, 1200000],
        "TotLen Bwd Pkts": [600, 720, 900],
        "Down/Up Ratio": [0.001, 0.001, 0.001],
        "Pkt Len Mean": [1460, 1460, 1460],
    })
    pkt_exfil = {
        "Payload Len Mean": 1420.0,
        "Payload Len Max": 1460.0,
        "Unique Dst Ports": 1,
        "SYN Only Ratio": 0.0,
        "RST Ratio": 0.0,
    }
    
    res_exfil = mapper.map_mitre_stage(
        risk_probability=0.85,
        df=df_exfil,
        packet_features=pkt_exfil
    )
    
    print(f"Predicted Stage: {res_exfil['stage']}")
    print(f"Formatted Output: {res_exfil['formatted_output']}")
    for item in res_exfil["progression"]:
        print(f"  [{'ACTIVE' if item['supported'] else 'INACTIVE'}] {item['stage']}: {item['confidence_percent']}% | Rules: {item['rules_satisfied']}")
    
    assert res_exfil["stage"] == "Exfiltration"
    exfil_prog = next(p for p in res_exfil["progression"] if p["stage"] == "Exfiltration")
    assert exfil_prog["supported"] == True
    assert exfil_prog["confidence"] > 0.60
    
    print("\n=== ALL MITRE MAPPER TESTS PASSED ===")

if __name__ == "__main__":
    test_attack_stage_mapper()
