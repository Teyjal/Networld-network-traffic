"""
Inference Service
Handles feature extraction, normalization, temporal sliding window creation (20 flows),
PyTorch LSTM model execution, K-step temporal horizon forecasting, and MITRE ATT&CK stage mapping.

Model context:
- Architecture: Temporal NetWorldLSTM World Model
- Model artifact: models/networld_combined_temporal_lstm_best.pt
- Scaler artifact: models/networld_combined_temporal_scaler.pkl
- Input sequence shape: (batch_size, 20, 36)
- Risk Threshold Categories (Application-Defined UI Triage):
  - LOW: probability < 0.3
  - MEDIUM: 0.3 <= probability < 0.7
  - HIGH: probability >= 0.7
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

try:
    import torch
except ImportError:
    torch = None

from app.services.preprocessing import REQUIRED_MODEL_FEATURES
from app.services.mitre_mapper import MitreMapperService
from app.model.loader import get_model_and_scaler, ModelLoader


class InferenceService:
    """
    Service for executing temporal LSTM model inference, K-step horizon forecasting,
    and rule-based MITRE ATT&CK stage mapping.
    """

    SEQUENCE_LENGTH = 20
    NUM_FEATURES = 36

    @staticmethod
    def classify_risk_category(prob: float) -> str:
        """
        Classifies infiltration probability score into application-defined UI triage categories.
        Note: These categories are UI alert triage heuristics, not guaranteed physical probabilities.
        """
        if prob < 0.3:
            return "LOW"
        elif prob < 0.7:
            return "MEDIUM"
        else:
            return "HIGH"

    def run_forecast(self, df: pd.DataFrame, filename: str = "uploaded_traffic.csv") -> Dict[str, Any]:
        """
        Executes PyTorch LSTM inference on the input DataFrame and maps MITRE ATT&CK stage.
        """
        # 1. Select exact 36 model features in correct order
        missing_feats = [col for col in REQUIRED_MODEL_FEATURES if col not in df.columns]
        if missing_feats:
            raise ValueError(f"Input data is missing required feature(s): {', '.join(missing_feats)}")

        # For fast responsive inference, evaluate the most recent flows (up to 500)
        if len(df) > 500:
            features_df = df[REQUIRED_MODEL_FEATURES].tail(500).copy()
        else:
            features_df = df[REQUIRED_MODEL_FEATURES].copy()

        # 2. Convert to numeric and sanitize invalid/infinite values
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce")

        raw_array = features_df.values.astype(np.float32)
        raw_array = np.nan_to_num(raw_array, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. Retrieve Singleton Model, Scaler, and target Compute Device
        model, scaler, device = get_model_and_scaler()

        # Extract scaler parameters for fast vector normalization: (x - mean) / scale
        mean = np.asarray(scaler.mean_, dtype=np.float32)
        scale = np.asarray(scaler.scale_, dtype=np.float32)
        scale = np.where(scale == 0.0, 1.0, scale)

        scaled_array = (raw_array - mean) / scale
        scaled_array = np.nan_to_num(scaled_array, nan=0.0, posinf=0.0, neginf=0.0)

        # 4. Construct rolling 20-step temporal sequence windows
        num_rows = len(scaled_array)
        sequences: List[np.ndarray] = []

        if num_rows >= self.SEQUENCE_LENGTH:
            # Evaluate the most recent active windows (up to 100) for fast real-time inference
            max_windows = 100
            total_possible = num_rows - self.SEQUENCE_LENGTH + 1
            start_i = max(0, total_possible - max_windows)
            for i in range(start_i, total_possible):
                seq = scaled_array[i : i + self.SEQUENCE_LENGTH]
                sequences.append(seq)
        else:
            padding_needed = self.SEQUENCE_LENGTH - num_rows
            padded_prefix = np.tile(scaled_array[0], (padding_needed, 1))
            seq = np.vstack([padded_prefix, scaled_array])
            sequences.append(seq)

        sequences_np = np.array(sequences, dtype=np.float32)  # Shape: (num_sequences, 20, 36)

        # 5. Run model inference (PyTorch or NumPy engine)
        if torch is not None and hasattr(model, "parameters"):
            tensor_input = torch.from_numpy(sequences_np).to(device)
            model.eval()
            with torch.no_grad():
                logits = model(tensor_input)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
        else:
            logits = model(sequences_np)
            if hasattr(logits, "cpu"):
                logits = logits.cpu().numpy()
            probs = (1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))).flatten()

        # 6. Format metrics and timeline
        probabilities = [round(float(p), 4) for p in probs]
        num_sequences = len(probabilities)

        current_risk = probabilities[-1] if probabilities else 0.0
        highest_risk = round(float(np.max(probabilities)), 4) if probabilities else 0.0
        overall_category = self.classify_risk_category(highest_risk)

        risk_timeline = []
        for idx, prob_val in enumerate(probabilities):
            risk_timeline.append({
                "sequence_index": idx,
                "probability": prob_val,
                "risk_category": self.classify_risk_category(prob_val)
            })

        # 7. Map MITRE ATT&CK Stage
        mitre_mapper = MitreMapperService()
        mitre_info = mitre_mapper.map_mitre_stage(
            risk_probability=highest_risk,
            df=df
        )

        return {
            "success": True,
            "filename": filename,
            "number_of_sequences": num_sequences,
            "current_risk": current_risk,
            "highest_predicted_risk": highest_risk,
            "overall_risk_category": overall_category,
            "mitre_mapping": mitre_info,
            "forecast_probabilities": probabilities,
            "risk_timeline": risk_timeline,
            "device_used": str(device),
            "status": "forecast_completed",
            "disclaimer": (
                "Forecast probabilities are neural network output scores indicating relative temporal anomaly risk. "
                "Risk categories (LOW, MEDIUM, HIGH) are application-defined UI triage thresholds and do not "
                "represent guaranteed physical probability bounds."
            )
        }

    def run_temporal_kstep_forecast(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
        filename: str = "uploaded_traffic.csv"
    ) -> Dict[str, Any]:
        """
        Executes K-step temporal horizon risk forecasting over sequential flow windows and includes MITRE ATT&CK stage mapping.
        """
        base_results = self.run_forecast(df, filename=filename)
        all_probs = base_results["forecast_probabilities"]
        num_windows = len(all_probs)

        # Enforce valid horizon boundaries
        horizon = max(1, min(int(horizon), 50))

        timeline = []
        if num_windows <= horizon:
            curr_prob = all_probs[0]
            timeline.append({
                "step": 0,
                "risk": curr_prob,
                "risk_category": self.classify_risk_category(curr_prob),
                "status": "CURRENT",
                "description": "Observed current traffic state (20-flow sequence window)"
            })
            for i in range(1, num_windows):
                prob = all_probs[i]
                timeline.append({
                    "step": i,
                    "risk": prob,
                    "risk_category": self.classify_risk_category(prob),
                    "status": "FORECAST",
                    "description": f"Temporal lookahead step {i}"
                })
        else:
            start_idx = num_windows - 1 - horizon
            curr_prob = all_probs[start_idx]
            timeline.append({
                "step": 0,
                "risk": curr_prob,
                "risk_category": self.classify_risk_category(curr_prob),
                "status": "CURRENT",
                "description": "Observed current traffic state (20-flow sequence window)"
            })
            for step in range(1, horizon + 1):
                prob = all_probs[start_idx + step]
                timeline.append({
                    "step": step,
                    "risk": prob,
                    "risk_category": self.classify_risk_category(prob),
                    "status": "FORECAST",
                    "description": f"Temporal lookahead step {step}"
                })

        current_risk = timeline[0]["risk"]
        forecast_risks = [item["risk"] for item in timeline]
        highest_risk = round(float(max(forecast_risks)), 4)

        # Map MITRE ATT&CK Stage using peak horizon risk
        mitre_mapper = MitreMapperService()
        mitre_info = mitre_mapper.map_mitre_stage(
            risk_probability=highest_risk,
            df=df
        )

        return {
            "success": True,
            "filename": filename,
            "horizon": horizon,
            "total_available_windows": num_windows,
            "current_risk": current_risk,
            "current_risk_category": self.classify_risk_category(current_risk),
            "highest_predicted_risk": highest_risk,
            "overall_risk_category": self.classify_risk_category(highest_risk),
            "mitre_mapping": mitre_info,
            "timeline": timeline,
            "device_used": base_results.get("device_used", "cpu"),
            "status": "forecast_completed",
            "disclaimer": (
                "Model evaluates temporal risk across sequential flow windows. "
                "Step 0 represents the observed current state; Steps 1..K represent temporal lookahead forecasts. "
                "Because the underlying PyTorch model is an LSTM classifier, future feature vectors are not synthesized."
            )
        }
