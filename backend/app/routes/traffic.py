"""
Traffic Analysis Route Module
Handles network flow ingestion, CSV file upload, feature validation, and sequence buffering.
"""

import os
import shutil
import uuid
import pandas as pd
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, status

from app.services.preprocessing import PreprocessingService
from app.services.enforcement_adapter import enforcement_adapter

router = APIRouter(tags=["Traffic"])

# Absolute path for temporary uploads directory within backend
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.get("/")
async def get_traffic_status():
    """
    Get current status of traffic monitoring and ingestion.
    """
    return {
        "status": "ready",
        "message": "Traffic route ready for network flow CSV uploads."
    }


from app.services.session_manager import session_manager


@router.get("/flows")
async def get_traffic_flows(
    limit: int = Query(50, ge=1, le=500),
    stage: str = Query("latest", description="pre, post, or latest")
):
    """
    Returns parsed real network flow records from the active session.
    Supports switching between PRE-RESPONSE and POST-RESPONSE flows.
    """
    active_session = session_manager.get_session()
    if active_session:
        if stage == "post":
            flows = active_session.get("post_response_flows") or []
            return flows[:limit]
        elif stage == "pre":
            flows = active_session.get("pre_response_flows") or []
            return flows[:limit]
        else:
            flows = active_session.get("post_response_flows") or active_session.get("pre_response_flows") or []
            return flows[:limit]

    # If no active session, do not return flows from arbitrary historical uploads
    return []


@router.post("/upload")
@router.post("/upload-pcap")
async def upload_traffic_file(file: UploadFile = File(...)):
    """
    Ingest and validate network traffic dataset (CSV or PCAP/PCAPNG) for NetWorld world model forecasting.
    
    Capabilities:
    1. Accepts CSV or PCAP files via multipart/form-data.
    2. For PCAP/PCAPNG: extracts packet-level statistics (TTL, TCP window, payload length, retransmissions, port scans)
       and aggregates into 36 flow features + 15 packet features (51 Unified Features).
    3. For CSV: validates presence of 36 flow features and inspects for packet-level extensions.
    4. Creates session and computes initial autoregressive forecast.
    """
    original_filename = file.filename or "uploaded_traffic.csv"
    ext = os.path.splitext(original_filename.lower())[1]

    if ext not in [".csv", ".pcap", ".pcapng"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a CSV (.csv) or PCAP capture (.pcap, .pcapng)."
        )

    # 2. Save uploaded file temporarily
    file_id = str(uuid.uuid4())
    temp_filename = f"{file_id}_{original_filename}"
    file_path = os.path.join(UPLOAD_DIR, temp_filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(e)}"
        )

    # 3. Validate features and calculate traffic statistics
    try:
        if ext in [".pcap", ".pcapng"]:
            stats, working_path = PreprocessingService.validate_and_analyze_pcap(file_path, original_filename)
        else:
            stats = PreprocessingService.validate_and_analyze_csv(file_path, original_filename)
            working_path = file_path

        # Create authoritative unified analysis session
        new_session = session_manager.create_session(working_path, original_filename, stats)
        stats["session_id"] = new_session["session_id"]
        stats["filename"] = temp_filename
        stats["threat_context"] = new_session["threat_context"]
        stats["pre_response_risk"] = new_session["pre_response_risk"]
        stats["forecast"] = new_session["pre_response_forecast"]
        stats["explain"] = new_session["pre_response_explain"]
        # Reset previous response state on enforcement adapter
        enforcement_adapter.reset_state()
        return stats
    except ValueError as val_err:
        # Clean up temporary file on validation failure
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except Exception as err:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing traffic dataset: {str(err)}"
        )

