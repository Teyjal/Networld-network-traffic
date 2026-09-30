"""
Defender API Route Module
Dedicated endpoints for controlled-lab closed-loop mitigation, verification, and audit tracking.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

from app.services.session_manager import session_manager

router = APIRouter(tags=["Defender Closed-Loop Response"])


class DefenderApplyRequest(BaseModel):
    session_id: str = Field(..., description="Active session ID")
    action: str = Field(..., description="Action: block_source, close_port, isolate_host, rate_limit")
    target: str = Field(..., description="Target IP or port")
    operator_id: str = Field(..., description="Explicit human operator ID")
    authorization_reason: str = Field(..., description="Reason for containment authorization")
    authorized: bool = Field(False, description="Must be true for operator approval")


@router.get("/status")
async def get_defender_status():
    """
    Returns the current active analysis session status, threat context,
    response status, and verification state.
    """
    session = session_manager.get_session()
    if not session:
        return {
            "status": "NO_TRAFFIC",
            "session_id": None,
            "threat_context": None,
            "recommendations": [],
            "response_status": "NONE",
            "verification_status": "NOT_STARTED",
            "message": "No active traffic session. Upload traffic to initialize detection and response."
        }

    return {
        "status": "ACTIVE_SESSION",
        "session_id": session["session_id"],
        "uploaded_filename": session["uploaded_filename"],
        "original_filename": session["original_filename"],
        "threat_context": session["threat_context"],
        "recommendations": session["recommendations"],
        "pre_response_risk": session["pre_response_risk"],
        "pre_response_highest_risk": session["pre_response_highest_risk"],
        "pre_response_category": session["pre_response_category"],
        "response_status": session["response_status"],
        "active_response": session["active_response"],
        "flows_blocked_count": session["flows_blocked_count"],
        "post_response_flows_count": session["post_response_flows_count"],
        "post_response_risk": session["post_response_risk"],
        "post_response_highest_risk": session["post_response_highest_risk"],
        "post_response_category": session["post_response_category"],
        "risk_change_pts": session["risk_change_pts"],
        "verification_status": session["verification_status"],
        "verification_message": session["verification_message"],
        "audit_events": session.get("audit_events", []),
        "pre_response_forecast": session.get("pre_response_forecast"),
        "post_response_forecast": session.get("post_response_forecast"),
        "pre_response_explain": session.get("pre_response_explain"),
        "post_response_explain": session.get("post_response_explain"),
    }


@router.post("/apply")
async def apply_defender_mitigation(payload: DefenderApplyRequest):
    """
    Executes controlled-lab mitigation:
    1. Validates operator authorization.
    2. Filters/blocks flows matching target in replay dataset.
    3. Re-runs PyTorch LSTM model on new post-response traffic.
    4. Computes empirical risk change and verification.
    5. Appends to authoritative audit trail.
    """
    try:
        updated_session = session_manager.apply_response(
            session_id=payload.session_id,
            action=payload.action,
            target=payload.target,
            operator_id=payload.operator_id,
            authorization_reason=payload.authorization_reason,
            authorized=payload.authorized,
        )
        return {
            "success": True,
            "session_id": updated_session["session_id"],
            "response_status": updated_session["response_status"],
            "verification_status": updated_session["verification_status"],
            "verification_message": updated_session["verification_message"],
            "pre_response_risk": updated_session["pre_response_risk"],
            "post_response_risk": updated_session["post_response_risk"],
            "risk_change_pts": updated_session["risk_change_pts"],
            "flows_blocked_count": updated_session["flows_blocked_count"],
            "post_response_flows_count": updated_session["post_response_flows_count"],
            "session": updated_session,
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Defender response execution failed: {str(e)}"
        )


@router.get("/audit")
async def get_defender_audit():
    """
    Returns the authoritative audit trail from the active session.
    """
    session = session_manager.get_session()
    if not session:
        return {
            "session_id": None,
            "audit_events": [],
            "count": 0,
        }

    return {
        "session_id": session["session_id"],
        "audit_events": session["audit_events"],
        "count": len(session["audit_events"]),
    }


@router.post("/reset")
async def reset_defender_session():
    """
    Clears the active session and audit history.
    """
    session_manager.reset()
    return {"success": True, "message": "Defender session reset."}
