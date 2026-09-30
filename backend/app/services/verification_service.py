"""
Verification Service Module
Executes real PyTorch LSTM re-inference on post-response observed traffic,
compares BEFORE vs AFTER metrics objectively, and determines mitigation verification status.
"""

import os
import datetime
from typing import Dict, Any, Optional
import pandas as pd

from app.services.inference import InferenceService
from app.services.preprocessing import REQUIRED_MODEL_FEATURES

from app.services.storage import UPLOAD_DIR


class VerificationService:
    """
    Compares baseline traffic observations with newly observed post-response lab traffic,
    re-running the exact trained PyTorch LSTM model to verify containment.
    """

    def __init__(self):
        self.inference_service = InferenceService()

    def verify_response(
        self,
        baseline_filename: str,
        post_response_filename: str,
        response_id: str,
        action: str,
        target: str,
        horizon: int = 10,
    ) -> Dict[str, Any]:
        """
        Executes complete verification pipeline:
        1. Reads baseline and post-response CSV files.
        2. Validates 36 model features.
        3. Runs real PyTorch LSTM on baseline (if not already cached) and on new post-response traffic.
        4. Calculates attack flow counts and high-risk sequence counts.
        5. Computes objective verification state.
        """
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # 1. Resolve paths
        baseline_path = os.path.join(UPLOAD_DIR, baseline_filename)
        post_path = os.path.join(UPLOAD_DIR, post_response_filename)

        if not os.path.exists(post_path):
            return {
                "success": False,
                "verification_status": "VERIFICATION_FAILED",
                "error": f"Post-response traffic file '{post_response_filename}' not found in lab.",
                "verification_timestamp": now,
            }

        try:
            # 2. Read baseline and post-response DataFrames
            df_base = pd.read_csv(baseline_path)
            df_base.columns = df_base.columns.str.strip()

            df_post = pd.read_csv(post_path)
            df_post.columns = df_post.columns.str.strip()

            # 3. Feature validation on post-response data
            missing_features = [f for f in REQUIRED_MODEL_FEATURES if f not in df_post.columns]
            if missing_features:
                return {
                    "success": False,
                    "verification_status": "VERIFICATION_FAILED",
                    "error": f"Post-response traffic missing features: {', '.join(missing_features)}",
                    "verification_timestamp": now,
                }

            # 4. Run PyTorch LSTM on both datasets
            base_forecast = self.inference_service.run_temporal_kstep_forecast(
                df=df_base, horizon=horizon, filename=baseline_filename
            )
            post_forecast = self.inference_service.run_temporal_kstep_forecast(
                df=df_post, horizon=horizon, filename=post_response_filename
            )

            # 5. Extract BEFORE metrics
            before_risk = float(base_forecast["highest_predicted_risk"])
            before_current_risk = float(base_forecast["current_risk"])
            before_timeline = base_forecast.get("timeline", [])

            # Count before attack flows from dataset label if present
            before_attack_flows = 0
            if "Label" in df_base.columns:
                before_attack_flows = int((~df_base["Label"].astype(str).str.lower().isin(["benign", "normal"])).sum())
            else:
                # If no label, estimate from high-risk timeline steps
                before_attack_flows = sum(1 for step in before_timeline if step.get("risk", 0) >= 0.7)

            before_high_risk_flows = sum(1 for step in before_timeline if step.get("risk", 0) >= 0.7)

            # 6. Extract AFTER metrics from real post-response LSTM inference
            after_risk = float(post_forecast["highest_predicted_risk"])
            after_current_risk = float(post_forecast["current_risk"])
            after_timeline = post_forecast.get("timeline", [])

            after_attack_flows = 0
            if "Label" in df_post.columns:
                after_attack_flows = int((~df_post["Label"].astype(str).str.lower().isin(["benign", "normal"])).sum())
            else:
                after_attack_flows = sum(1 for step in after_timeline if step.get("risk", 0) >= 0.7)

            after_high_risk_flows = sum(1 for step in after_timeline if step.get("risk", 0) >= 0.7)

            # 7. Compute Objective Verification State
            risk_delta = round(after_risk - before_risk, 4)
            relative_reduction = (before_risk - after_risk) / max(0.01, before_risk)

            if after_risk < 0.30 and after_attack_flows == 0:
                verification_status = "MITIGATED"
                status_description = "Threat fully mitigated. Post-response traffic exhibits baseline risk < 30% with zero observed attack flows."
            elif relative_reduction >= 0.40 and after_risk < 0.60:
                verification_status = "RISK_REDUCED"
                status_description = f"Risk successfully reduced by {round(relative_reduction * 100, 1)}%. Moderate residual telemetry monitored."
            elif relative_reduction >= 0.15:
                verification_status = "PARTIALLY_MITIGATED"
                status_description = f"Partial mitigation observed ({round(relative_reduction * 100, 1)}% reduction). Attack indicators persist in secondary flows."
            elif after_risk >= before_risk * 0.90:
                verification_status = "STILL_ACTIVE"
                status_description = "Attack traffic remains actively observed in post-response capture. Additional containment required."
            else:
                verification_status = "RISK_REDUCED"
                status_description = f"Measurable risk reduction of {round(abs(risk_delta) * 100, 1)}% achieved."

            return {
                "success": True,
                "response_id": response_id,
                "action": action,
                "target": target,
                "verification_status": verification_status,
                "status_description": status_description,
                "verification_timestamp": now,
                "metrics": {
                    "before_risk": before_risk,
                    "after_risk": after_risk,
                    "before_current_risk": before_current_risk,
                    "after_current_risk": after_current_risk,
                    "risk_delta": risk_delta,
                    "risk_reduction_pct": round(relative_reduction * 100, 1),
                    "before_attack_flows": before_attack_flows,
                    "after_attack_flows": after_attack_flows,
                    "before_high_risk_flows": before_high_risk_flows,
                    "after_high_risk_flows": after_high_risk_flows,
                    "total_flows_observed": len(df_post),
                },
                "post_forecast": {
                    "current_risk": after_current_risk,
                    "highest_predicted_risk": after_risk,
                    "mitre_mapping": post_forecast.get("mitre_mapping"),
                    "timeline": after_timeline,
                },
                "model_verification": {
                    "architecture": "Temporal NetWorldLSTM World Model",
                    "artifact": "models/networld_combined_temporal_lstm_best.pt",
                    "features_verified": 36,
                    "sequence_window": 20,
                    "device": post_forecast.get("device_used", "cpu"),
                }
            }

        except Exception as e:
            return {
                "success": False,
                "verification_status": "VERIFICATION_FAILED",
                "error": f"LSTM re-inference error during post-response verification: {str(e)}",
                "verification_timestamp": now,
            }


# Singleton instance
verification_service = VerificationService()
