"""
Response and Verification Route Module
Handles defensive action recommendations, operator approvals, controlled lab containment,
new post-response traffic collection, real PyTorch LSTM re-inference, and mitigation verification.
"""

import os
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Query, Body, status
from pydantic import BaseModel, Field

from app.services.enforcement_adapter import enforcement_adapter
from app.services.verification_service import verification_service
from app.services.inference import InferenceService

router = APIRouter(tags=["Response & Verification"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")


class OperatorApprovalPayload(BaseModel):
    action: str = Field(..., description="Action name: block_source, close_port, isolate_host, rate_limit")
    target: str = Field(..., description="Allow-listed lab target (IP or port number)")
    reason: Optional[str] = Field(None, description="Operator reason for containment")
    operator_id: str = Field("operator_admin", description="Operator identity granting approval")
    approved: bool = Field(False, description="Explicit operator approval flag - must be True to execute")
    baseline_filename: Optional[str] = Field(None, description="Active baseline traffic dataset filename")


class RollbackPayload(BaseModel):
    response_id: str = Field(..., description="Response ID of the active lab rule to rollback")
    operator_id: str = Field("operator_admin", description="Operator identity authorizing rollback")


class ModePayload(BaseModel):
    mode: str = Field(..., description="'CONTROLLED_LAB' or 'REPLAY'")
    is_configured: Optional[bool] = Field(None, description="Flag whether lab testbed is configured")


@router.get("/")
@router.get("/status")
async def get_response_status():
    """
    Returns current status of controlled enforcement adapter, mode, and active lab rules.
    """
    return enforcement_adapter.get_status()


@router.post("/mode")
async def set_response_mode(payload: ModePayload):
    """
    Configures enforcement mode between CONTROLLED_LAB and REPLAY.
    Supports Case 1 (unconfigured/replay) and Case 3 (controlled lab response) testing.
    """
    enforcement_adapter.set_mode(payload.mode, payload.is_configured)
    return enforcement_adapter.get_status()


@router.get("/history")
async def get_response_history():
    """
    Returns complete authoritative audit log and history of lab response actions.
    """
    return {
        "status": "success",
        "audit_log": enforcement_adapter.audit_log,
        "audit_logs": enforcement_adapter.audit_log,
        "active_rules": list(enforcement_adapter.active_rules.values()),
    }


@router.post("/reset")
async def reset_response_state():
    """
    Resets active rules, audit log, and cached response records for a new session or upload.
    """
    enforcement_adapter.reset_state()
    return {"status": "success", "message": "Defender response state reset."}


@router.post("/recommend")
async def recommend_responses(baseline_filename: Optional[str] = Query(None)):
    """
    Inspects the active traffic sequence and generates context-aware recommended responses.
    Strictly data-driven: returns NO_TRAFFIC if no active uploaded dataset is provided.
    Never falls back to sample_traffic.csv or mock data.
    """
    if not baseline_filename:
        return {
            "status": "NO_TRAFFIC",
            "threat_context": None,
            "recommendations": [],
            "mode": enforcement_adapter.mode,
            "is_configured": enforcement_adapter.is_configured,
            "message": "No active traffic dataset. Upload traffic to initialize threat detection and response."
        }

    target_path = os.path.join(UPLOAD_DIR, baseline_filename)
    if not os.path.exists(target_path):
        found = False
        for f in os.listdir(UPLOAD_DIR):
            if f == baseline_filename or f.endswith(f"_{baseline_filename}"):
                target_path = os.path.join(UPLOAD_DIR, f)
                found = True
                break
        if not found:
            return {
                "status": "NO_TRAFFIC",
                "threat_context": None,
                "recommendations": [],
                "mode": enforcement_adapter.mode,
                "is_configured": enforcement_adapter.is_configured,
                "message": f"Traffic dataset '{baseline_filename}' not found. Please upload a dataset."
            }

    # 1. Run Real PyTorch LSTM inference on the active uploaded dataset
    from app.services.inference import InferenceService
    inference_service = InferenceService()
    try:
        forecast_res = inference_service.forecast_from_csv(target_path, horizon=10)
        current_risk = float(forecast_res.get("current_risk", 0.0))
        highest_risk = float(forecast_res.get("highest_predicted_risk", current_risk))
        risk_cat = forecast_res.get("overall_risk_category", "Low")
        attack_detected = forecast_res.get("attack_type_detected", "Benign")
    except Exception as e:
        current_risk = 0.0
        highest_risk = 0.0
        risk_cat = "Unknown"
        attack_detected = "Benign"

    # 2. Extract actual observed IPs and ports from the uploaded CSV
    import pandas as pd
    try:
        df = pd.read_csv(target_path)
        df.columns = df.columns.str.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read traffic dataset: {str(e)}")

    # Check for attack flows if Label exists
    attack_mask = pd.Series([False] * len(df))
    if "Label" in df.columns:
        attack_mask = df["Label"].astype(str).str.lower().str.strip() != "benign"

    def get_col_val(frame, candidates):
        for c in candidates:
            for col in frame.columns:
                if col.lower().strip() == c.lower().strip():
                    val = frame[col].dropna()
                    if not val.empty:
                        return str(val.iloc[0]).strip()
        return None

    observed_src = None
    observed_dst = None
    observed_port = None

    if attack_mask.any():
        df_attack = df[attack_mask]
        observed_src = get_col_val(df_attack, ["Src IP", "Source IP", "src_ip", "Source_IP"])
        observed_dst = get_col_val(df_attack, ["Dst IP", "Destination IP", "dst_ip", "Destination_IP"])
        raw_port = get_col_val(df_attack, ["Dst Port", "Destination Port", "dst_port", "Destination_Port"])
        if raw_port:
            try:
                observed_port = str(int(float(raw_port)))
            except Exception:
                observed_port = raw_port

    if not observed_src:
        observed_src = get_col_val(df, ["Src IP", "Source IP", "src_ip", "Source_IP"]) or "Observed Adversary"
    if not observed_dst:
        observed_dst = get_col_val(df, ["Dst IP", "Destination IP", "dst_ip", "Destination_IP"]) or "Target Host"
    if not observed_port:
        raw_port = get_col_val(df, ["Dst Port", "Destination Port", "dst_port", "Destination_Port"])
        if raw_port:
            try:
                observed_port = str(int(float(raw_port)))
            except Exception:
                observed_port = raw_port
        else:
            observed_port = "Dynamic"

    # 3. Requirement 8: Do NOT assume that every upload is an attack
    is_threat = (highest_risk >= 0.35) or (attack_detected and attack_detected.lower() != "benign") or attack_mask.any()

    if not is_threat:
        threat_info = {
            "status": "BENIGN",
            "attack_type": "Benign / Normal Traffic",
            "risk_score": current_risk,
            "highest_risk": highest_risk,
            "risk_category": risk_cat,
            "observed_source": observed_src,
            "observed_destination": observed_dst,
            "observed_port": observed_port,
            "baseline_filename": baseline_filename,
        }
        return {
            "status": "BENIGN",
            "threat_context": threat_info,
            "recommendations": [],
            "mode": enforcement_adapter.mode,
            "is_configured": enforcement_adapter.is_configured,
            "message": "No active threat detected. Current traffic is within safe baseline limits."
        }

    # 4. Active Threat Identified: build dynamic data-driven recommendations
    display_attack = attack_detected if (attack_detected and attack_detected.lower() != "benign") else "Infiltration Threat"
    threat_info = {
        "status": "ACTIVE_THREAT",
        "attack_type": display_attack,
        "risk_score": current_risk,
        "highest_risk": highest_risk,
        "risk_category": risk_cat,
        "observed_source": observed_src,
        "observed_destination": observed_dst,
        "observed_port": observed_port,
        "baseline_filename": baseline_filename,
    }

    recommendations = []
    if observed_src:
        recommendations.append({
            "action": "block_source",
            "title": f"Block Source IP {observed_src} at Lab Perimeter",
            "target": observed_src,
            "priority": "HIGH",
            "expected_impact": "Drops all inbound lateral movement flows from the adversary IP.",
            "description": f"Enforce immediate ACL block on source {observed_src} in lab testbed.",
        })
    if observed_port:
        recommendations.append({
            "action": "close_port",
            "title": f"Close Destination Port {observed_port}",
            "target": str(observed_port),
            "priority": "HIGH",
            "expected_impact": "Shuts down vulnerable target service port to disrupt lateral exploitation.",
            "description": f"Close port {observed_port} across internal lab zone.",
        })
    if observed_dst:
        recommendations.append({
            "action": "isolate_host",
            "title": f"Quarantine Compromised Target Host {observed_dst}",
            "target": observed_dst,
            "priority": "MEDIUM",
            "expected_impact": "Isolates victim host into lab quarantine VLAN to arrest attack progression.",
            "description": f"Isolate destination host {observed_dst}.",
        })
    if observed_src:
        recommendations.append({
            "action": "rate_limit",
            "title": f"Rate Limit Aggressive Flow Burst on {observed_src}",
            "target": observed_src,
            "priority": "LOW",
            "expected_impact": "Throttles flow rate by 50% to prevent connection flooding.",
            "description": f"Apply 50% rate throttling queue on {observed_src}.",
        })

    return {
        "status": "ready",
        "threat_context": threat_info,
        "recommendations": recommendations,
        "mode": enforcement_adapter.mode,
        "is_configured": enforcement_adapter.is_configured,
    }


class PreviewPayload(BaseModel):
    action: Optional[str] = None
    target: Optional[str] = None


@router.post("/preview")
async def preview_response(
    payload: Optional[PreviewPayload] = None,
    action: Optional[str] = Query(None),
    target: Optional[str] = Query(None)
):
    """
    Dry-run preview of a proposed response action without executing it.
    """
    act = (payload.action if payload and payload.action else action) or ""
    tgt = (payload.target if payload and payload.target else target) or ""
    if not act or not tgt:
        raise HTTPException(status_code=400, detail="Action and target are required for preview.")
    return enforcement_adapter.preview_action(act, tgt)


@router.post("/apply")
async def apply_response(payload: OperatorApprovalPayload):
    """
    Primary Controlled Response & Verification Endpoint.
    
    Workflow:
    1. Validates operator approval.
    2. Validates action and target against lab allow-list.
    3. Executes containment through SafeLabEnforcementAdapter.
    4. Collects real post-response traffic from monitored lab.
    5. Re-runs PyTorch LSTM World Model on the new post-response traffic.
    6. Returns authoritative BEFORE vs AFTER verification metrics and audit record.
    """
    # Requirement: Explicit Human Operator Approval
    if not payload.approved:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Operator approval required. Field 'approved' must be true to execute defensive response."
        )

    # Check lab configuration
    if not enforcement_adapter.is_configured or enforcement_adapter.mode != "CONTROLLED_LAB":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Real response is unavailable because no controlled enforcement environment is configured. Use Simulation Mode instead."
        )

    # Resolve baseline filename strictly from active upload
    baseline_filename = payload.baseline_filename
    if not baseline_filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Active baseline traffic dataset is required. Please upload traffic before executing response."
        )

    target_path = os.path.join(UPLOAD_DIR, baseline_filename)
    if not os.path.exists(target_path):
        found = False
        for f in os.listdir(UPLOAD_DIR):
            if f == baseline_filename or f.endswith(f"_{baseline_filename}"):
                target_path = os.path.join(UPLOAD_DIR, f)
                baseline_filename = f
                found = True
                break
        if not found:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Baseline traffic dataset '{baseline_filename}' was not found. Please upload traffic first."
            )

    # Step 1: Apply through Controlled Enforcement Adapter & collect post traffic
    apply_result = enforcement_adapter.apply_action(
        action=payload.action,
        target=payload.target,
        reason=payload.reason or "Adversary containment",
        operator_id=payload.operator_id,
        baseline_filename=baseline_filename
    )

    if not apply_result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=apply_result.get("error", "Failed to apply lab enforcement action.")
        )

    post_traffic_filename = apply_result["post_traffic_filename"]
    response_id = apply_result["response_id"]

    # Step 2: Verification Service re-runs PyTorch LSTM on newly observed traffic
    verification = verification_service.verify_response(
        baseline_filename=baseline_filename,
        post_response_filename=post_traffic_filename,
        response_id=response_id,
        action=payload.action,
        target=payload.target,
        horizon=10,
    )

    # Step 3: Record verification status in audit log
    v_status = verification.get("verification_status", "UNKNOWN")
    enforcement_adapter._log_event(
        "VERIFICATION_COMPLETE",
        f"PyTorch LSTM re-inference completed on post-response traffic. Verification Status: {v_status}",
        response_id=response_id
    )

    return {
        "success": True,
        "response_record": {
            "response_id": response_id,
            "timestamp": apply_result["applied_at"],
            "action": payload.action,
            "target": payload.target,
            "reason": payload.reason or "Containment",
            "requested_by": payload.operator_id,
            "approval_status": "APPROVED",
            "execution_status": "APPLIED",
            "before_risk": verification.get("metrics", {}).get("before_risk", 0.0),
            "post_response_risk": verification.get("metrics", {}).get("after_risk", 0.0),
            "verification_status": v_status,
            "verification_timestamp": verification.get("verification_timestamp"),
            "rollback_available": True,
        },
        "verification": verification,
        "mode": "CONTROLLED_LAB",
    }


@router.post("/rollback")
async def rollback_response(payload: RollbackPayload):
    """
    Rolls back an applied lab response action and removes rule from the lab testbed.
    """
    result = enforcement_adapter.rollback_action(
        response_id=payload.response_id,
        operator_id=payload.operator_id
    )
    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "Rollback failed.")
        )
    return result
