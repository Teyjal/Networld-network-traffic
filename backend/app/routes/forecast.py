"""
Forecast Route Module
Handles temporal cyberattack forecasting requests using PyTorch LSTM model with configurable K-step lookahead horizon.
"""

import os
import shutil
import uuid
import pandas as pd
from typing import Optional, Any
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, status, Request, Body
from pydantic import BaseModel

from app.services.inference import InferenceService
from app.services.world_model_service import get_world_model_service
from app.model.loader import ModelLoader

router = APIRouter(tags=["Forecast"])

# Upload directory reference from central storage service
from app.services.storage import UPLOAD_DIR, ensure_upload_dir

# In-memory forecast store for GET /forecast/{id}
FORECAST_CACHE = {}


class ForecastRequestJSON(BaseModel):
    filename: Optional[str] = None
    horizon: Optional[int] = 5
    k: Optional[int] = None


@router.get("/")
@router.get("/status")
async def get_forecast_status():
    """
    Get forecast service status and loaded model hardware info.
    """
    loader = ModelLoader()
    artifact_info = loader.check_artifacts_exist()
    wm = get_world_model_service()
    return {
        "status": "ready",
        "model_info": artifact_info,
        "world_model_version": "NetWorld-Unified-WorldModel" if (wm.model and getattr(wm.model, "input_size", 0) == 51) else "NetWorld-MultiHead-WorldModel",
        "active_input_dim": getattr(wm.model, "input_size", 36),
        "unified_features_count": 51,
        "flow_features_count": 36,
        "packet_features_count": 15,
        "message": "Unified Multi-Head Temporal World Model Autoregressive Forecast active."
    }


@router.get("/feature-schema")
async def get_feature_schema():
    """
    Returns full specification of the 51 features in the Unified Network State:
    - 36 Flow Features (CIC-IDS2018 standard)
    - 15 Packet Features (TTL, TCP window, payload statistics, retransmissions, port scans)
    """
    from ml.feature_schema import FEATURE_NAMES, NUM_FEATURES
    from ml.packet_schema import (
        PACKET_FEATURE_NAMES,
        NUM_PACKET_FEATURES,
        UNIFIED_FEATURE_NAMES,
        NUM_UNIFIED_FEATURES,
        PACKET_FEATURE_SPEC,
    )
    wm = get_world_model_service()
    return {
        "flow_features": {
            "count": NUM_FEATURES,
            "names": FEATURE_NAMES,
        },
        "packet_features": {
            "count": NUM_PACKET_FEATURES,
            "names": PACKET_FEATURE_NAMES,
            "spec": PACKET_FEATURE_SPEC,
        },
        "unified_features": {
            "count": NUM_UNIFIED_FEATURES,
            "names": UNIFIED_FEATURE_NAMES,
        },
        "active_model_input_size": getattr(wm.model, "input_size", 36),
        "active_model_checkpoint": getattr(wm, "loaded_checkpoint", "models/networld_combined_temporal_lstm_best.npz"),
    }


from app.services.session_manager import session_manager



@router.post("")
@router.post("/")
@router.post("/predict")
async def create_forecast(
    file: Optional[UploadFile] = File(None),
    filename: Optional[str] = Query(None),
    stage: Optional[str] = Query(None, description="pre, post, or latest"),
    horizon: Optional[int] = Query(5, ge=1, le=50, description="K-step forecast horizon (default K=5)"),
    k: Optional[int] = Query(None, description="Alias for horizon"),
    json_data: Optional[ForecastRequestJSON] = Body(None),
):
    """
    Executes PyTorch LSTM temporal K-step horizon forecasting on traffic dataset.
    
    Accepts:
    - File upload (multipart/form-data) OR filename of previously uploaded traffic CSV.
    - Configurable lookahead horizon `K` (default 5).
    
    Response:
    - horizon: Configured K steps
    - timeline: Array of forecast steps distinguishing step 0 ('CURRENT') from steps 1..K ('FORECAST')
    - current_risk: Risk score for current sequence state
    - highest_predicted_risk: Peak forecast risk across timeline
    - overall_risk_category: LOW / MEDIUM / HIGH UI alert category
    """
    # Resolve K / horizon value
    effective_horizon = 5
    if k is not None:
        effective_horizon = k
    elif horizon is not None:
        effective_horizon = horizon
    elif json_data and json_data.k is not None:
        effective_horizon = json_data.k
    elif json_data and json_data.horizon is not None:
        effective_horizon = json_data.horizon

    target_path = None
    target_filename = "traffic_dataset.csv"
    cleanup_temp = False
    is_post_response = False

    active_session = session_manager.get_session()

    # Priority 1: Check active session stage ("post" or "pre")
    if stage == "post" and active_session and active_session.get("post_response_filename"):
        target_filename = active_session["post_response_filename"]
        target_path = os.path.join(UPLOAD_DIR, target_filename)
        is_post_response = True
    elif stage == "pre" and active_session:
        target_filename = active_session["uploaded_filename"]
        target_path = os.path.join(UPLOAD_DIR, target_filename)
    elif file is not None:
        # Direct CSV Upload
        target_filename = file.filename or "uploaded_traffic.csv"
        if not target_filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file format. Forecast input must be a CSV file (.csv)."
            )
        file_id = str(uuid.uuid4())
        ensure_upload_dir()
        target_path = os.path.join(UPLOAD_DIR, f"forecast_{file_id}_{target_filename}")
        try:
            with open(target_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            cleanup_temp = True
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save temporary forecast file: {str(e)}"
            )
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
    elif active_session:
        # Default to post-response if available, otherwise baseline
        if active_session.get("post_response_filename"):
            target_filename = active_session["post_response_filename"]
            target_path = os.path.join(UPLOAD_DIR, target_filename)
            is_post_response = True
        else:
            target_filename = active_session["uploaded_filename"]
            target_path = os.path.join(UPLOAD_DIR, target_filename)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No traffic dataset uploaded yet. Please upload a CSV file to generate model forecast."
        )

    # 3. Read DataFrame & Run K-step Horizon Forecast
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
            detail=f"Failed to parse CSV file for forecasting: {str(read_err)}"
        )

    try:
        world_service = get_world_model_service()
        result = world_service.forecast_trajectory(df, horizon=effective_horizon, filename=target_filename)
        forecast_id = str(uuid.uuid4())
        result["id"] = forecast_id
        result["is_post_response"] = is_post_response
        FORECAST_CACHE[forecast_id] = result
        return result
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except Exception as infer_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failure: {str(infer_err)}"
        )
    finally:
        if cleanup_temp and target_path and os.path.exists(target_path):
            try:
                os.remove(target_path)
            except Exception:
                pass


@router.get("/{forecast_id}")
async def get_forecast_by_id(forecast_id: str):
    """
    Retrieve stored forecast results by ID.
    """
    if forecast_id in FORECAST_CACHE:
        return FORECAST_CACHE[forecast_id]
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Forecast with ID '{forecast_id}' not found in active session cache."
    )


class WhatIfForecastRequestJSON(BaseModel):
    action: str = "close_port"
    target_value: Optional[Any] = 80
    rate_factor: Optional[float] = 0.5
    horizon: Optional[int] = 5
    filename: Optional[str] = None


@router.post("/what-if")
@router.post("/whatif")
async def forecast_what_if(
    file: Optional[UploadFile] = File(None),
    filename: Optional[str] = Query(None),
    action: Optional[str] = Query(None, description="Defensive action (block_ip, quarantine_host, close_port, rate_limit)"),
    target_value: Optional[str] = Query(None, description="Target port or IP"),
    rate_factor: Optional[float] = Query(0.5, description="Rate reduction factor for rate_limit"),
    horizon: Optional[int] = Query(5, description="Lookahead horizon K"),
    json_data: Optional[WhatIfForecastRequestJSON] = Body(None),
):
    """
    Counterfactual What-If defence simulator endpoint for /forecast/what-if.
    Computes baseline vs counterfactual rollout trajectories using the multi-head world model.
    Supported Actions: Block IP, Quarantine Host, Close Port, Rate Limit
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

    active_session = session_manager.get_session()

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
    elif filename or (json_data and json_data.filename):
        requested_name = filename or (json_data.filename if json_data else "")
        target_filename = os.path.basename(requested_name)
        candidate_path = os.path.join(UPLOAD_DIR, target_filename)
        if not os.path.exists(candidate_path):
            for f in (os.listdir(UPLOAD_DIR) if os.path.exists(UPLOAD_DIR) else []):
                if f.endswith(target_filename):
                    candidate_path = os.path.join(UPLOAD_DIR, f)
                    break
        if not os.path.exists(candidate_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Traffic file '{target_filename}' not found."
            )
        target_path = candidate_path
    elif active_session:
        target_filename = active_session["uploaded_filename"]
        target_path = os.path.join(UPLOAD_DIR, target_filename)
    else:
        existing_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".csv")] if os.path.exists(UPLOAD_DIR) else []
        if existing_files:
            latest_file = max(existing_files, key=lambda f: os.path.getmtime(os.path.join(UPLOAD_DIR, f)))
            target_path = os.path.join(UPLOAD_DIR, latest_file)
            target_filename = latest_file
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No CSV file provided for What-If simulation."
            )

    try:
        if os.path.getsize(target_path) > 10 * 1024 * 1024:
            df = pd.read_csv(target_path, nrows=1000)
        else:
            df = pd.read_csv(target_path)
        df.columns = df.columns.str.strip()
    except Exception as e:
        if cleanup_temp and target_path and os.path.exists(target_path):
            os.remove(target_path)
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {str(e)}")

    try:
        world_service = get_world_model_service()
        result = world_service.simulate_defense(
            df=df,
            action=effective_action,
            target_value=effective_target,
            rate_factor=effective_rate_factor,
            horizon=effective_horizon,
            filename=target_filename
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")
    finally:
        if cleanup_temp and target_path and os.path.exists(target_path):
            try:
                os.remove(target_path)
            except Exception:
                pass


