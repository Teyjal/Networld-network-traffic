"""
NetWorld Autoregressive K-Step Forward Rollout Engine
Executes real forward state-transition rollouts using NetWorldMultiHeadWorldModel:
  S(t) -> Model -> S(t+1), Risk(t+1)
  [S(t)[1:], S(t+1)] -> Model -> S(t+2), Risk(t+2)
  ...
  S(t+K), Risk(t+K)

Never fabricates intermediate numbers; every value is a direct mathematical result
of the neural network forward pass and autoregressive window update.
"""

import os
import sys

workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn

try:
    from ml.feature_schema import (
        FEATURE_NAMES,
        NUM_FEATURES,
        SEQUENCE_LENGTH,
        classify_risk_level,
        DEFENSE_PERTURBATION_SPEC,
        FEATURE_INDEX_MAP,
        UNIFIED_FEATURE_NAMES,
        NUM_UNIFIED_FEATURES,
        PACKET_FEATURE_NAMES,
        NUM_PACKET_FEATURES,
        PACKET_FEATURE_INDEX_MAP,
    )
except ImportError:
    from feature_schema import (
        FEATURE_NAMES,
        NUM_FEATURES,
        SEQUENCE_LENGTH,
        classify_risk_level,
        DEFENSE_PERTURBATION_SPEC,
        FEATURE_INDEX_MAP,
        UNIFIED_FEATURE_NAMES,
        NUM_UNIFIED_FEATURES,
        PACKET_FEATURE_NAMES,
        NUM_PACKET_FEATURES,
        PACKET_FEATURE_INDEX_MAP,
    )


class AutoregressiveRolloutEngine:
    """
    Executes K-step autoregressive rollout on temporal network flow sequences.
    Supports both 36-D flow state and 51-D Unified Network State (36 Flow + 15 Packet).
    """

    def __init__(
        self,
        model: nn.Module,
        scaler: Any,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.scaler = scaler
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

        # Check if unified scaler or standard scaler
        if hasattr(scaler, "unified_mean") and hasattr(scaler, "unified_scale"):
            self.mean = np.asarray(scaler.unified_mean, dtype=np.float32)
            self.scale = np.asarray(scaler.unified_scale, dtype=np.float32)
        elif hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
            self.mean = np.asarray(scaler.mean_, dtype=np.float32)
            self.scale = np.asarray(scaler.scale_, dtype=np.float32)
        else:
            self.mean = np.zeros(NUM_FEATURES, dtype=np.float32)
            self.scale = np.ones(NUM_FEATURES, dtype=np.float32)

        self.scale = np.where(self.scale == 0.0, 1.0, self.scale)

    def denormalize_state(self, scaled_vector: np.ndarray) -> np.ndarray:
        """Denormalizes a state vector back to original engineering units."""
        dim = scaled_vector.shape[-1]
        mean = self.mean[:dim]
        scale = self.scale[:dim]
        return (scaled_vector * scale) + mean

    def normalize_state(self, raw_vector: np.ndarray) -> np.ndarray:
        """Normalizes a raw state vector using the fitted scaler."""
        dim = raw_vector.shape[-1]
        mean = self.mean[:dim]
        scale = self.scale[:dim]
        return (raw_vector - mean) / scale


    def rollout(
        self,
        initial_sequence: np.ndarray,
        horizon: int = 5,
        intervention_spec: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs an autoregressive rollout of length `horizon` (K steps).
        Args:
            initial_sequence: Scaled 2D array of shape (20, 36) or (20, 51).
            horizon: Number of forward lookahead steps K (e.g. 1, 3, 5, 10).
            intervention_spec: Optional defense perturbation applied at each rollout step.
        Returns:
            trajectory: List of dictionaries for steps 0 (NOW) through K.
        """
        horizon = max(1, min(int(horizon), 20))
        seq_len, num_feats = initial_sequence.shape
        if seq_len != SEQUENCE_LENGTH or num_feats not in (NUM_FEATURES, NUM_UNIFIED_FEATURES):
            raise ValueError(
                f"Expected initial sequence of shape ({SEQUENCE_LENGTH}, {NUM_FEATURES}) or "
                f"({SEQUENCE_LENGTH}, {NUM_UNIFIED_FEATURES}), got {initial_sequence.shape}"
            )

        model_dim = getattr(self.model, "input_size", num_feats)

        trajectory: List[Dict[str, Any]] = []

        # Current working window (starts with the observed sequence aligned to model_dim)
        if num_feats < model_dim:
            pad_cols = model_dim - num_feats
            pad_block = np.zeros((seq_len, pad_cols), dtype=np.float32)
            current_window = np.hstack([initial_sequence, pad_block])
            num_feats = model_dim
        elif num_feats > model_dim:
            current_window = np.copy(initial_sequence[:, :model_dim])
            num_feats = model_dim
        else:
            current_window = np.copy(initial_sequence)

        # Step 0: Observed current state (NOW)
        with torch.no_grad():
            tensor_seq = torch.from_numpy(current_window).unsqueeze(0).to(self.device).float()
            risk_logits, next_state_pred = self.model(tensor_seq)
            curr_prob = float(torch.sigmoid(risk_logits).cpu().item())
            next_state_vec = next_state_pred.squeeze(0).cpu().numpy()

        last_observed_state_raw = self.denormalize_state(current_window[-1])
        top_active_features = self._extract_key_features(last_observed_state_raw)

        trajectory.append({
            "step": 0,
            "horizon_label": "NOW",
            "status": "CURRENT",
            "risk_probability": round(curr_prob, 4),
            "risk_percent": round(curr_prob * 100, 2),
            "risk_level": classify_risk_level(curr_prob),
            "key_features": top_active_features,
            "is_synthesized_state": False,
            "description": f"Observed network state ({num_feats} features)"
        })

        # Autoregressive forward steps 1..K
        for k in range(1, horizon + 1):
            # The next state vector predicted by the previous step
            predicted_state = np.nan_to_num(next_state_vec, nan=0.0, posinf=0.0, neginf=0.0)

            # If intervention active, apply defense constraint to synthesized state before appending
            if intervention_spec is not None:
                pred_denorm = self.denormalize_state(predicted_state)
                affected_feats = intervention_spec.get("affected_features", [])
                op = intervention_spec.get("operation", "zero")
                scale_f = intervention_spec.get("scale_factor", 0.1)

                for f_name in affected_feats:
                    if f_name in FEATURE_INDEX_MAP:
                        idx = FEATURE_INDEX_MAP[f_name]
                        if op == "zero":
                            pred_denorm[idx] = 0.0
                        elif op == "scale":
                            pred_denorm[idx] *= scale_f
                    elif f_name in PACKET_FEATURE_INDEX_MAP and len(pred_denorm) >= NUM_UNIFIED_FEATURES:
                        p_idx = NUM_FEATURES + PACKET_FEATURE_INDEX_MAP[f_name]
                        if op == "zero":
                            pred_denorm[p_idx] = 0.0
                        elif op == "scale":
                            pred_denorm[p_idx] *= scale_f

                predicted_state = self.normalize_state(pred_denorm)
                predicted_state = np.nan_to_num(predicted_state, nan=0.0, posinf=0.0, neginf=0.0)

            # Roll the sequence window: drop oldest flow at index 0, append predicted flow
            current_window = np.vstack([current_window[1:], predicted_state.reshape(1, num_feats)])

            # Forward pass on the updated sequence
            with torch.no_grad():
                tensor_seq = torch.from_numpy(current_window).unsqueeze(0).to(self.device).float()
                risk_logits, next_state_pred = self.model(tensor_seq)
                step_prob = float(torch.sigmoid(risk_logits).cpu().item())
                next_state_vec = next_state_pred.squeeze(0).cpu().numpy()

            denorm_state = self.denormalize_state(predicted_state)
            step_features = self._extract_key_features(denorm_state)

            trajectory.append({
                "step": k,
                "horizon_label": f"+{k}",
                "status": "FORECAST",
                "risk_probability": round(step_prob, 4),
                "risk_percent": round(step_prob * 100, 2),
                "risk_level": classify_risk_level(step_prob),
                "key_features": step_features,
                "is_synthesized_state": True,
                "description": f"Autoregressive forward state S(t+{k}) rollout"
            })

        return trajectory

    def simulate_defense(
        self,
        initial_sequence: np.ndarray,
        action: str,
        target_value: Optional[Any] = None,
        rate_factor: Optional[float] = None,
        horizon: int = 5,
    ) -> Dict[str, Any]:
        """
        Executes counterfactual simulation by perturbing initial network state according to the
        selected defense action and running the exact same autoregressive rollout model.
        Supported Actions: Block IP, Quarantine Host, Close Port, Rate Limit
        """
        action_raw = str(action).strip().lower().replace("-", "_")
        action_aliases = {
            "block_ip": "block_ip",
            "block ip": "block_ip",
            "ip_block": "block_ip",
            "quarantine_host": "quarantine_host",
            "quarantine host": "quarantine_host",
            "quarantine": "quarantine_host",
            "isolate_host": "quarantine_host",
            "isolate host": "quarantine_host",
            "isolate": "quarantine_host",
            "close_port": "close_port",
            "close port": "close_port",
            "block_port": "close_port",
            "block port": "close_port",
            "rate_limit": "rate_limit",
            "rate limit": "rate_limit",
            "rate_limit_traffic": "rate_limit",
            "rate limit traffic": "rate_limit",
            "restrict_traffic": "rate_limit",
            "restrict traffic": "rate_limit",
        }
        action_clean = action_aliases.get(action_raw, action_raw)

        if action_clean not in DEFENSE_PERTURBATION_SPEC:
            raise ValueError(
                f"Unsupported defense action: '{action}'. "
                f"Supported actions: Block IP, Quarantine Host, Close Port, Rate Limit"
            )

        spec = dict(DEFENSE_PERTURBATION_SPEC[action_clean])
        if rate_factor is not None and spec.get("operation") == "scale":
            spec["scale_factor"] = max(0.01, min(float(rate_factor), 0.99))

        affected_feature_names = spec["affected_features"]

        # 1. Baseline rollout on original sequence using World Model
        baseline_trajectory = self.rollout(initial_sequence, horizon=horizon)

        # 2. Denormalize, create modified network state via physical intervention, then re-normalize
        cf_sequence_raw = np.zeros_like(initial_sequence)
        for i in range(len(initial_sequence)):
            cf_sequence_raw[i] = self.denormalize_state(initial_sequence[i])

        # Apply defense perturbation to the active intervention window (last 5 sequence flows)
        for feat_name in affected_feature_names:
            if feat_name in FEATURE_INDEX_MAP:
                feat_idx = FEATURE_INDEX_MAP[feat_name]
                if spec["operation"] == "zero":
                    cf_sequence_raw[-5:, feat_idx] = 0.0
                elif spec["operation"] == "scale":
                    factor = spec.get("scale_factor", 0.1)
                    cf_sequence_raw[-5:, feat_idx] *= factor
            elif feat_name in PACKET_FEATURE_INDEX_MAP and cf_sequence_raw.shape[1] >= NUM_UNIFIED_FEATURES:
                pkt_idx = NUM_FEATURES + PACKET_FEATURE_INDEX_MAP[feat_name]
                if spec["operation"] == "zero":
                    cf_sequence_raw[-5:, pkt_idx] = 0.0
                elif spec["operation"] == "scale":
                    factor = spec.get("scale_factor", 0.1)
                    cf_sequence_raw[-5:, pkt_idx] *= factor

        # Re-normalize modified counterfactual sequence
        cf_sequence_scaled = np.zeros_like(initial_sequence)
        for i in range(len(cf_sequence_raw)):
            cf_sequence_scaled[i] = self.normalize_state(cf_sequence_raw[i])
        cf_sequence_scaled = np.nan_to_num(cf_sequence_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. Counterfactual rollout on modified network state using the exact same LSTM World Model
        cf_trajectory = self.rollout(cf_sequence_scaled, horizon=horizon, intervention_spec=spec)

        # 4. Compare predicted risk at t+1...t+k
        base_risks = [item["risk_probability"] for item in baseline_trajectory]
        cf_risks = [item["risk_probability"] for item in cf_trajectory]

        peak_base = max(base_risks) if base_risks else 0.0
        peak_cf = max(cf_risks) if cf_risks else 0.0
        risk_delta = round(peak_cf - peak_base, 4)

        step_comparisons = []
        for i in range(len(baseline_trajectory)):
            b_item = baseline_trajectory[i]
            c_item = cf_trajectory[i] if i < len(cf_trajectory) else b_item
            s_delta = round(c_item["risk_probability"] - b_item["risk_probability"], 4)
            step_comparisons.append({
                "step": b_item["step"],
                "horizon_label": b_item["horizon_label"],
                "baseline_risk": b_item["risk_probability"],
                "baseline_risk_percent": round(b_item["risk_probability"] * 100, 1),
                "baseline_risk_level": b_item["risk_level"],
                "whatif_risk": c_item["risk_probability"],
                "whatif_risk_percent": round(c_item["risk_probability"] * 100, 1),
                "whatif_risk_level": c_item["risk_level"],
                "risk_delta": s_delta,
                "risk_delta_percent": round(s_delta * 100, 1),
                "risk_reduced": s_delta < 0,
            })

        friendly_action_name = spec.get("action_name", action.replace("_", " ").title())

        return {
            "action": action_clean,
            "selected_action": friendly_action_name,
            "description": spec["description"],
            "affected_features": affected_feature_names,
            "baseline_trajectory": baseline_trajectory,
            "counterfactual_trajectory": cf_trajectory,
            "whatif_trajectory": cf_trajectory,
            "step_comparisons": step_comparisons,
            "baseline_peak_risk": round(peak_base, 4),
            "counterfactual_peak_risk": round(peak_cf, 4),
            "whatif_peak_risk": round(peak_cf, 4),
            "risk_delta": risk_delta,
            "risk_change": risk_delta,
            "risk_change_percent": round(risk_delta * 100, 1),
            "risk_reduced": risk_delta < 0,
            "mitigation_percentage": round(abs(risk_delta) * 100, 2) if risk_delta < 0 else 0.0,
            "target_value": target_value,
            "disclaimer": (
                "Model-predicted change in risk based on counterfactual perturbation of the physical network state "
                "and forward LSTM rollout. This simulation reports statistical risk shift and does not claim "
                "guaranteed physical prevention of real-world network attacks."
            ),
        }

    def _extract_key_features(self, raw_vector: np.ndarray) -> Dict[str, float]:
        """Extracts key human-interpretable flow & packet characteristics for UI display."""
        key_names = [
            "Flow Pkts/s", "Flow Byts/s", "Pkt Size Avg",
            "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
            "SYN Flag Cnt", "ACK Flag Cnt", "RST Flag Cnt"
        ]
        summary = {}
        for name in key_names:
            if name in FEATURE_INDEX_MAP and len(raw_vector) > FEATURE_INDEX_MAP[name]:
                val = float(raw_vector[FEATURE_INDEX_MAP[name]])
                summary[name] = round(val, 2)

        # Extract packet-level key metrics if unified 51-D vector
        if len(raw_vector) >= NUM_UNIFIED_FEATURES:
            pkt_keys = [
                "Pkt TTL Mean", "TCP Win Mean", "Payload Len Mean",
                "Retransmission Cnt", "Unique Dst Ports", "SYN Only Ratio"
            ]
            for name in pkt_keys:
                if name in PACKET_FEATURE_INDEX_MAP:
                    pkt_idx = NUM_FEATURES + PACKET_FEATURE_INDEX_MAP[name]
                    if len(raw_vector) > pkt_idx:
                        summary[name] = round(float(raw_vector[pkt_idx]), 2)

        return summary

