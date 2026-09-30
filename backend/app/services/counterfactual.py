"""
Counterfactual What-If Simulator Service
Simulates defensive interventions on network traffic datasets (e.g. close port, block IP, rate limit)
by modifying a clean COPY of the traffic data and comparing PyTorch LSTM inference against baseline.

Supported Defensive Actions:
- block_ip
- close_port (or block_port)
- isolate_host
- block_protocol
- rate_limit

Requirements:
- Never modifies original dataset.
- Runs identical preprocessing & PyTorch model inference on baseline and simulated traffic.
- Computes baseline risk, counterfactual risk, timeline comparisons, and risk change delta.
- Clearly labels output as 'counterfactual simulation' with explicit disclaimers.
"""

from typing import Dict, Any, Optional, List
import pandas as pd

from app.services.world_model_service import get_world_model_service


class CounterfactualService:
    """
    Service for running counterfactual defensive simulations using the Multi-Head LSTM World Model.
    """

    SUPPORTED_ACTIONS = [
        "block_ip",
        "quarantine_host",
        "isolate_host",
        "close_port",
        "block_port",
        "rate_limit",
        "block_protocol",
    ]

    def __init__(self):
        self.world_service = get_world_model_service()

    def simulate_action(
        self,
        df: pd.DataFrame,
        action: str,
        target_value: Optional[Any] = None,
        rate_factor: float = 0.5,
        horizon: int = 5,
        filename: str = "uploaded_traffic.csv",
    ) -> Dict[str, Any]:
        """
        Executes What-If defence simulation using the LSTM World Model.
        Creates a modified network state, runs the same model on baseline and modified states,
        performs multi-step future rollout across t+1...t+k, and reports model-predicted risk change.
        """
        action_clean = action.strip().lower().replace("-", "_")
        if action_clean in ("isolate", "isolate_host", "quarantine"):
            action_clean = "quarantine_host"
        elif action_clean in ("block_port",):
            action_clean = "close_port"
        elif action_clean in ("restrict_traffic", "rate_limit_traffic"):
            action_clean = "rate_limit"

        return self.world_service.simulate_defense(
            df=df,
            action=action_clean,
            target_value=target_value,
            rate_factor=rate_factor,
            horizon=horizon,
            filename=filename,
        )
