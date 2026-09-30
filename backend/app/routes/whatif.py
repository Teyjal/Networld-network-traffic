"""
What-If Simulation Route Module
Handles counterfactual defender simulator requests.
"""

import os
import shutil
import uuid
import pandas as pd
from typing import Optional, Any
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, status, Body
from pydantic import BaseModel

from app.services.counterfactual import CounterfactualService
from app.services.session_manager import session_manager

router = APIRouter(tags=["What-If Simulation"])

from app.services.storage import UPLOAD_DIR, ensure_upload_dir


class WhatIfRequestJSON(BaseModel):
    action: str = "close_port"
    target_value: Optional[Any] = 80
    rate_factor: Optional[float] = 0.5
    horizon: Optional[int] = 5
    filename: Optional[str] = None


@router.get("/")
@router.get("/status")
async def get_whatif_status():
    """
    Get what-if simulator status and supported defensive actions.
    """
    return {
        "status": "ready",
        "supported_actions": CounterfactualService.SUPPORTED_ACTIONS,
        "message": "What-If Defender Simulator active for counterfactual traffic analysis."
    }


@router.post("")
@router.post("/")
@router.post("/simulate")
async def simulate_defensive_action(
    file: Optional[UploadFile] = File(None),
    filename: Optional[str] = Query(None),
    action: Optional[str] = Query(None, description="Defensive action (block_ip, close_port, isolate_host, block_protocol, rate_limit)"),
    target_value: Optional[str] = Query(None, description="Target port, IP, or protocol value"),
    rate_factor: Optional[float] = Query(0.5, description="Rate reduction factor for rate_limit (0.01 to 0.99)"),
    horizon: Optional[int] = Query(5, description="K-step lookahead horizon"),
    json_data: Optional[WhatIfRequestJSON] = Body(None),
):
    """
    Simulates defensive action on traffic dataset and computes risk reduction metrics against baseline.
    
    Supported Actions:
    - block_ip
    - close_port
    - isolate_host
    - block_protocol
    - rate_limit
    """
    effective_action = "close_port"
    effective_target = 80
    effective_rate_factor = 0.5
    effective_horizon = 5

    if json_data:
        effective_action = json_data.action or effective_action
        effective_target = json_data.target_value if json_data.target_value is not None else effective_target
        effective_rate_factor = json_data.rate_factor if json_data.rate_factor is not None else effective_rate_factor
        effective_horizon = json_data.horizon if json_data.horizon is not None else effective_horizon

    if action:
        effective_action = action
    if target_value is not None:
        effective_target = target_value
    if rate_factor is not None:
        effective_rate_factor = rate_factor
    if horizon is not None:
        effective_horizon = horizon

    target_path = None
    target_filename = "traffic_dataset.csv"
    cleanup_temp = False

    # 1. Handle Direct CSV Upload
    if file is not None:
        target_filename = file.filename or "uploaded_traffic.csv"
        if not target_filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file format. Input must be a CSV file (.csv)."
            )
        file_id = str(uuid.uuid4())
        ensure_upload_dir()
        target_path = os.path.join(UPLOAD_DIR, f"whatif_{file_id}_{target_filename}")
        try:
            with open(target_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            cleanup_temp = True
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save temporary simulation file: {str(e)}"
            )

    # 2. Handle Previously Uploaded Filename
    elif filename or (json_data and json_data.filename):
        requested_name = filename or (json_data.filename if json_data else "")
        target_filename = os.path.basename(requested_name)
        
        candidate_path = os.path.join(UPLOAD_DIR, target_filename)
        if not os.path.exists(candidate_path):
            found = False
            for f in (os.listdir(UPLOAD_DIR) if os.path.exists(UPLOAD_DIR) else []):
                if f.endswith(target_filename):
                    candidate_path = os.path.join(UPLOAD_DIR, f)
                    found = True
                    break
            if not found:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Uploaded dataset file '{target_filename}' not found. Please upload a CSV first."
                )
        target_path = candidate_path

    else:
        active_session = session_manager.get_session()
        if active_session:
            if active_session.get("post_response_filename"):
                target_filename = active_session["post_response_filename"]
                target_path = os.path.join(UPLOAD_DIR, target_filename)
            else:
                target_filename = active_session["uploaded_filename"]
                target_path = os.path.join(UPLOAD_DIR, target_filename)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No traffic dataset uploaded yet. Please upload a CSV file to simulate What-If defenses."
            )

    # 3. Read DataFrame & Run Counterfactual Simulation
    try:
        file_size = os.path.getsize(target_path) if target_path and os.path.exists(target_path) else 0
        if file_size > 10 * 1024 * 1024:
            df = pd.read_csv(target_path, nrows=1000)
        else:
            df = pd.read_csv(target_path)
        df.columns = df.columns.str.strip()
    except Exception as read_err:
        if cleanup_temp and target_path and os.path.exists(target_path):
            os.remove(target_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse CSV file for simulation: {str(read_err)}"
        )

    try:
        service = CounterfactualService()
        result = service.simulate_action(
            df=df,
            action=effective_action,
            target_value=effective_target,
            rate_factor=effective_rate_factor,
            horizon=effective_horizon,
            filename=target_filename
        )
        return result
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except Exception as sim_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Counterfactual simulation processing failure: {str(sim_err)}"
        )
    finally:
        if cleanup_temp and target_path and os.path.exists(target_path):
            try:
                os.remove(target_path)
            except Exception:
                pass
