"""
Validation Route Module
Exposes empirical model validation metrics and benchmark comparisons against baseline models.
"""

import os
import json
from fastapi import APIRouter, HTTPException, status

router = APIRouter(tags=["Validation"])

# Path resolution for evaluation results JSON artifact
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKSPACE_ROOT = os.path.dirname(BASE_DIR)

EVALUATION_JSON_PATHS = [
    os.path.join(WORKSPACE_ROOT, "models", "evaluation_results.json"),
    os.path.join(BASE_DIR, "models", "evaluation_results.json"),
    os.path.join(BASE_DIR, "evaluation_results.json"),
]


@router.get("")
@router.get("/")
@router.get("/metrics")
async def get_validation_metrics():
    """
    Exposes empirical validation metrics from saved evaluation results.
    Compares NetWorld Temporal LSTM against Logistic Regression baseline on the 20% temporal holdout split.
    
    If evaluation files do not exist yet, returns:
    {"status": "not_available", "message": "Run model evaluation first."}
    """
    metrics_file = None
    for path in EVALUATION_JSON_PATHS:
        if os.path.exists(path):
            metrics_file = path
            break

    if not metrics_file:
        return {
            "status": "not_available",
            "message": "Run model evaluation first."
        }

    try:
        with open(metrics_file, "r") as f:
            data = json.load(f)
        return data
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load evaluation metrics: {str(err)}"
        )
