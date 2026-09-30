"""
Explainability Route Module
Handles feature attribution requests using PyTorch Integrated Gradients.
"""

import os
import shutil
import uuid
import pandas as pd
from typing import Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, status, Body
from pydantic import BaseModel

from app.services.shap_service import ShapExplainabilityService

router = APIRouter(tags=["Explainability"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


class ExplainRequestJSON(BaseModel):
    filename: Optional[str] = None
    sample_index: Optional[int] = -1
    top_n: Optional[int] = 10


@router.get("/")
@router.get("/status")
async def get_explain_status():
    """
    Get explainability service status.
    """
    return {
        "status": "ready",
        "method": "SHAP (GradientExplainer)",
        "message": "SHAP explainability service active for PyTorch LSTM sequence predictions."
    }


from app.services.session_manager import session_manager


@router.get("/shap")
async def get_shap_explanation(
    filename: Optional[str] = Query(None),
    stage: Optional[str] = Query(None, description="pre, post, or latest"),
    sample_index: Optional[int] = Query(-1, description="Index of sequence window to explain (-1 for latest)"),
    top_n: Optional[int] = Query(10, ge=1, le=36, description="Number of top features to return"),
):
    """
    GET endpoint to retrieve SHAP feature attributions for active session or specified file.
    """
    return await explain_prediction(
        file=None,
        filename=filename,
        stage=stage,
        sample_index=sample_index,
        top_n=top_n,
        json_data=None,
    )


@router.post("")
@router.post("/")
@router.post("/shap")
async def explain_prediction(
    file: Optional[UploadFile] = File(None),
    filename: Optional[str] = Query(None),
    stage: Optional[str] = Query(None, description="pre, post, or latest"),
    sample_index: Optional[int] = Query(-1, description="Index of sequence window to explain (-1 for latest)"),
    top_n: Optional[int] = Query(10, ge=1, le=36, description="Number of top features to return"),
    json_data: Optional[ExplainRequestJSON] = Body(None),
):
    """
    Computes feature attributions explaining model prediction for a sequence sample.
    
    Response:
    - prediction: Model output risk probability
    - overall_risk_category: LOW / MEDIUM / HIGH UI alert category
    - top_features: Sorted array of top N features with importance, direction, and raw values
    - method_used: Attribution method (IntegratedGradients)
    """
    effective_sample_idx = sample_index if sample_index is not None else -1
    effective_top_n = top_n if top_n is not None else 10

    if json_data:
        if json_data.sample_index is not None:
            effective_sample_idx = json_data.sample_index
        if json_data.top_n is not None:
            effective_top_n = json_data.top_n

    target_path = None
    target_filename = "traffic_dataset.csv"
    cleanup_temp = False

    target_path = None
    target_filename = "traffic_dataset.csv"
    cleanup_temp = False
    is_post_response = False

    active_session = session_manager.get_session()

    if stage == "post" and active_session and active_session.get("post_response_filename"):
        target_filename = active_session["post_response_filename"]
        target_path = os.path.join(UPLOAD_DIR, target_filename)
        is_post_response = True
    elif stage == "pre" and active_session:
        target_filename = active_session["uploaded_filename"]
        target_path = os.path.join(UPLOAD_DIR, target_filename)
    elif file is not None:
        target_filename = file.filename or "uploaded_traffic.csv"
        if not target_filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file format. Input must be a CSV file (.csv)."
            )
        file_id = str(uuid.uuid4())
        target_path = os.path.join(UPLOAD_DIR, f"explain_{file_id}_{target_filename}")
        try:
            with open(target_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            cleanup_temp = True
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save temporary explainability file: {str(e)}"
            )
    elif filename or (json_data and json_data.filename):
        requested_name = filename or (json_data.filename if json_data else "")
        target_filename = os.path.basename(requested_name)
        
        candidate_path = os.path.join(UPLOAD_DIR, target_filename)
        if not os.path.exists(candidate_path):
            found = False
            for f in os.listdir(UPLOAD_DIR):
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
            detail="No traffic dataset uploaded yet. Please upload a CSV file to generate feature attributions."
        )

    # 3. Read DataFrame & Compute Attributions
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
            detail=f"Failed to parse CSV file for explainability: {str(read_err)}"
        )

    try:
        service = ShapExplainabilityService()
        result = service.explain_dataframe_sequence(
            df=df,
            sample_index=effective_sample_idx,
            top_n=effective_top_n,
            filename=target_filename
        )
        result["is_post_response"] = is_post_response
        return result
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except Exception as explain_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explainability processing failure: {str(explain_err)}"
        )
    finally:
        if cleanup_temp and target_path and os.path.exists(target_path):
            try:
                os.remove(target_path)
            except Exception:
                pass
