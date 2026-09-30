"""
NetWorld Temporal World Model Service
Integrates the multi-head autoregressive predictive engine:
  S(t) -> Multi-Head LSTM -> S(t+1), Risk(t+1)
  Autoregressive Rollout -> S(t+K), Risk(t+K)
  Integrated Explainability (SHAP / Gradient attributions)
  Evidence-Based MITRE ATT&CK Mapping
  Counterfactual What-If Defense Simulation
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

try:
    import torch
except ImportError:
    torch = None

# Ensure backend dir and workspace root are in sys.path (prioritizing BACKEND_DIR)
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKSPACE_ROOT = os.path.dirname(BACKEND_DIR)
for p in [WORKSPACE_ROOT, BACKEND_DIR]:
    if p and p not in sys.path:
        sys.path.insert(0, p)
BASE_DIR = BACKEND_DIR if os.path.exists(os.path.join(BACKEND_DIR, "models")) else WORKSPACE_ROOT

try:
    from ml.feature_schema import (
        FEATURE_NAMES,
        NUM_FEATURES,
        SEQUENCE_LENGTH,
        classify_risk_level,
        DEFENSE_PERTURBATION_SPEC,
        UNIFIED_FEATURE_NAMES,
        NUM_UNIFIED_FEATURES,
        PACKET_FEATURE_NAMES,
        NUM_PACKET_FEATURES,
    )
    from ml.packet_extractor import PacketFeatureExtractor
    from ml.unified_scaler import get_unified_scaler, UnifiedNetworkScaler
except ImportError:
    backend_ml = os.path.join(BACKEND_DIR, "ml")
    if backend_ml not in sys.path:
        sys.path.insert(0, backend_ml)
    from feature_schema import (
        FEATURE_NAMES,
        NUM_FEATURES,
        SEQUENCE_LENGTH,
        classify_risk_level,
        DEFENSE_PERTURBATION_SPEC,
        UNIFIED_FEATURE_NAMES,
        NUM_UNIFIED_FEATURES,
        PACKET_FEATURE_NAMES,
        NUM_PACKET_FEATURES,
    )
    from packet_extractor import PacketFeatureExtractor
    from unified_scaler import get_unified_scaler, UnifiedNetworkScaler

if torch is not None:
    try:
        from ml.world_model_multihead import NetWorldMultiHeadWorldModel
        from ml.rollout import AutoregressiveRolloutEngine
    except ImportError:
        NetWorldMultiHeadWorldModel = None
        AutoregressiveRolloutEngine = None
else:
    NetWorldMultiHeadWorldModel = None
    AutoregressiveRolloutEngine = None

from app.services.mitre_mapper import MitreMapperService
from app.services.shap_service import ShapExplainabilityService


class WorldModelService:
    """
    Singleton service managing the multi-head temporal network world model,
    autoregressive K-step rollouts, Unified Network State (36 Flow + 15 Packet),
    and counterfactual defense interventions.
    """

    _instance: Optional["WorldModelService"] = None

    UNIFIED_MODEL_PATH = "models/networld_unified_world_model_best.pt"
    FUTURE_MODEL_PATH = "models/networld_future_world_model_best.pt"
    FALLBACK_MODEL_PATH = "models/networld_combined_temporal_lstm_best.pt"
    SCALER_PATH = "models/networld_combined_temporal_scaler.pkl"

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(WorldModelService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.scaler = get_unified_scaler()
        if torch is not None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._load_artifacts()
            if self.model is not None and AutoregressiveRolloutEngine is not None:
                self.rollout_engine = AutoregressiveRolloutEngine(
                    model=self.model,
                    scaler=self.scaler,
                    device=self.device,
                )
            else:
                self.rollout_engine = None
        else:
            self.device = "cpu"
            self.model = None
            self.rollout_engine = None
        self.mitre_mapper = MitreMapperService()
        self.shap_service = ShapExplainabilityService()
        self._initialized = True

    def _resolve_path(self, rel_path: str) -> str:
        if os.path.isabs(rel_path):
            return rel_path
        cand1 = os.path.join(WORKSPACE_ROOT, rel_path)
        if os.path.exists(cand1):
            return cand1
        cand2 = os.path.join(BACKEND_DIR, rel_path)
        if os.path.exists(cand2):
            return cand2
        return os.path.abspath(rel_path)

    def _load_artifacts(self):
        # 1. Load Unified Scaler (combines 36 flow + 15 packet scalers)
        self.scaler: UnifiedNetworkScaler = get_unified_scaler()

        # 2. Determine model checkpoint path and input dimension
        unified_path = self._resolve_path(self.UNIFIED_MODEL_PATH)
        future_path = self._resolve_path(self.FUTURE_MODEL_PATH)
        fallback_path = self._resolve_path(self.FALLBACK_MODEL_PATH)

        if os.path.exists(unified_path):
            model_file = unified_path
            input_dim = NUM_UNIFIED_FEATURES  # 51
        elif os.path.exists(future_path):
            model_file = future_path
            input_dim = NUM_FEATURES  # 36
        elif os.path.exists(fallback_path):
            model_file = fallback_path
            input_dim = NUM_FEATURES  # 36
        else:
            raise FileNotFoundError(f"Model checkpoint missing at: {unified_path}")

        # Instantiate Multi-Head Architecture
        self.model = NetWorldMultiHeadWorldModel(
            input_size=input_dim,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
        ).to(self.device)

        ckpt = torch.load(model_file, map_location=self.device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

        model_dict = self.model.state_dict()
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                model_dict[k] = v
            elif k.startswith("infiltration_head."):
                new_k = k.replace("infiltration_head.", "risk_head.")
                if new_k in model_dict and model_dict[new_k].shape == v.shape:
                    model_dict[new_k] = v

        self.model.load_state_dict(model_dict)
        self.model.eval()
        self.loaded_checkpoint = model_file
        self.input_dim = input_dim

    def reload(self):
        """Reloads checkpoint from disk (e.g. after training finishes)."""
        self._load_artifacts()
        self.rollout_engine = AutoregressiveRolloutEngine(
            model=self.model,
            scaler=self.scaler,
            device=self.device,
        )

    def prepare_sequence_window(
        self, df: pd.DataFrame, packet_data: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Validates, sanitizes, normalizes, and extracts the active 20-step sequence window.
        Returns:
            seq_window: scaled array of shape (20, 51) or (20, 36)
            packet_meta: availability metadata and sample packet features
        """
        missing_feats = [col for col in FEATURE_NAMES if col not in df.columns]
        if missing_feats:
            raise ValueError(f"Input dataset missing required feature(s): {', '.join(missing_feats)}")


        # For fast responsive rollout, evaluate the most recent flows (up to 100)
        if len(df) > 100:
            features_df = df[FEATURE_NAMES].tail(100).copy()
        else:
            features_df = df[FEATURE_NAMES].copy()

        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce")

        # Extract raw flow array
        raw_flow = features_df.values.astype(np.float32)
        raw_flow = np.nan_to_num(raw_flow, nan=0.0, posinf=0.0, neginf=0.0)
        num_rows = len(raw_flow)
        if num_rows == 0:
            raise ValueError("Input dataset is empty.")

        # Extract packet-level features if provided or from df
        if packet_data is not None:
            raw_packet = packet_data
            is_pkt_avail = True
        else:
            raw_packet, is_pkt_avail = PacketFeatureExtractor.extract_from_df(df.tail(100))


        model_dim = getattr(self.model, "input_size", NUM_UNIFIED_FEATURES if is_pkt_avail else NUM_FEATURES)
        if model_dim == NUM_UNIFIED_FEATURES:
            scaled_array = self.scaler.normalize_unified(raw_flow, raw_packet)
        else:
            scaled_array = self.scaler.normalize_flow(raw_flow)

        scaled_array = np.nan_to_num(scaled_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Extract latest 20 flows or pad
        if num_rows >= SEQUENCE_LENGTH:
            seq_window = scaled_array[-SEQUENCE_LENGTH:]
        else:
            padding_needed = SEQUENCE_LENGTH - num_rows
            padded_prefix = np.tile(scaled_array[0], (padding_needed, 1))
            seq_window = np.vstack([padded_prefix, scaled_array])

        sample_packet_stats = {}
        if len(raw_packet) > 0:
            for i, name in enumerate(PACKET_FEATURE_NAMES):
                sample_packet_stats[name] = round(float(raw_packet[-1, i]), 2)

        packet_meta = {
            "packet_features_available": bool(is_pkt_avail),
            "flow_feature_count": NUM_FEATURES,
            "packet_feature_count": NUM_PACKET_FEATURES if is_pkt_avail else 0,
            "total_feature_count": NUM_UNIFIED_FEATURES if is_pkt_avail else NUM_FEATURES,
            "active_model_dim": getattr(self.model, "input_size", NUM_UNIFIED_FEATURES if is_pkt_avail else NUM_FEATURES),
            "source": "pcap_packet_telemetry" if is_pkt_avail else "flow_telemetry_neutral_baseline",
            "packet_features": sample_packet_stats,
        }

        return seq_window, packet_meta

    def forecast_trajectory(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
        filename: str = "traffic.csv",
        packet_data: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Executes true autoregressive forward state & risk rollout on the active sequence window.
        """
        horizon = max(1, min(int(horizon), 20))
        seq_window, packet_meta = self.prepare_sequence_window(df, packet_data=packet_data)

        # Execute K-step Autoregressive Rollout (or pure NumPy fallback if torch unavailable)
        if self.rollout_engine is not None:
            trajectory = self.rollout_engine.rollout(seq_window, horizon=horizon)
        else:
            from app.services.inference import InferenceService
            infer_service = InferenceService()
            kstep_result = infer_service.run_temporal_kstep_forecast(df, horizon=horizon, filename=filename)
            trajectory = []
            for item in kstep_result["timeline"]:
                step_idx = item["step"]
                trajectory.append({
                    "step": step_idx,
                    "risk_probability": item["risk"],
                    "risk_level": item["risk_category"],
                    "status": item["status"],
                    "description": item["description"],
                    "horizon_label": f"+{step_idx}" if step_idx > 0 else "NOW",
                    "key_features": {}
                })

        current_step = trajectory[0]
        peak_risk = max(item["risk_probability"] for item in trajectory)
        peak_step_idx = max(range(len(trajectory)), key=lambda i: trajectory[i]["risk_probability"])

        # Timeline format for backward compatibility with frontend charts
        timeline = []
        for item in trajectory:
            timeline.append({
                "step": item["step"],
                "risk": item["risk_probability"],
                "risk_category": item["risk_level"],
                "status": item["status"],
                "description": item["description"],
                "horizon_label": item["horizon_label"],
                "key_features": item.get("key_features", {})
            })

        # Map MITRE ATT&CK Stage using peak horizon risk and observed behavior
        mitre_info = self.mitre_mapper.map_mitre_stage(
            risk_probability=peak_risk,
            df=df
        )

        # Compute SHAP / Gradient Explainability for the current state
        try:
            shap_result = self.shap_service.explain_dataframe_sequence(df=df, top_n=5)
            top_drivers = shap_result.get("top_drivers", [])
        except Exception:
            top_drivers = []

        return {
            "success": True,
            "filename": filename,
            "horizon": horizon,
            "model_version": "NetWorld-Unified-WorldModel" if (self.model and getattr(self.model, "input_size", 0) == NUM_UNIFIED_FEATURES) else "NetWorld-MultiHead-WorldModel",
            "device": str(self.device),
            "current_risk": current_step["risk_probability"],
            "current_risk_category": current_step["risk_level"],
            "highest_predicted_risk": round(peak_risk, 4),
            "overall_risk_category": classify_risk_level(peak_risk),
            "peak_risk_horizon": trajectory[peak_step_idx]["horizon_label"],
            "trajectory": trajectory,
            "timeline": timeline,
            "mitre_mapping": mitre_info,
            "top_risk_drivers": top_drivers,
            "state_synthesis_supported": True,
            "unified_state": {
                "supported": True,
                "packet_features_available": packet_meta["packet_features_available"],
                "flow_feature_count": packet_meta["flow_feature_count"],
                "packet_feature_count": packet_meta["packet_feature_count"],
                "total_feature_count": packet_meta["total_feature_count"],
                "active_model_dim": packet_meta["active_model_dim"],
                "source": packet_meta["source"],
                "packet_features": packet_meta["packet_features"],
                "description": "Unified Network State combines 36 flow-level features with 15 defensible packet statistics.",
            },
            "status": "forecast_completed",
            "disclaimer": (
                "Forecast reflects autoregressive rollout: S(t) -> S(t+1) -> ... -> S(t+K). "
                "Step 0 represents the observed current state; Steps 1..K represent synthesized future network states. "
                "Risk categories (LOW, MEDIUM, HIGH) are operational triage heuristics."
            )
        }

    def forecast_from_pcap(
        self,
        filepath: str,
        horizon: int = 5,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Parses PCAP binary, extracts packet & flow telemetry, and forecasts trajectory.
        """
        parsed_pkts = PacketFeatureExtractor.parse_pcap_file(filepath)
        if not parsed_pkts:
            raise ValueError(f"No valid IPv4 packets found in PCAP: {filepath}")

        # Aggregate packet features
        pkt_features = PacketFeatureExtractor.aggregate_packet_features(parsed_pkts)

        # Create flow-level records from parsed packets
        flow_map = {}
        for p in parsed_pkts:
            key = (p.src_ip, p.dst_ip, p.src_port, p.dst_port, p.protocol)
            if key not in flow_map:
                flow_map[key] = []
            flow_map[key].append(p)

        flow_rows = []
        for (sip, dip, sport, dport, proto), p_list in flow_map.items():
            tot_fwd = len(p_list)
            tot_bytes = sum(p.payload_len for p in p_list)
            dur = max(0.001, (p_list[-1].timestamp - p_list[0].timestamp) * 1000)
            pkt_lens = [p.payload_len for p in p_list]
            row_dict = {f: 0.0 for f in FEATURE_NAMES}
            row_dict["Dst Port"] = float(dport)
            row_dict["Protocol"] = float(proto)
            row_dict["Flow Duration"] = float(dur)
            row_dict["Tot Fwd Pkts"] = float(tot_fwd)
            row_dict["TotLen Fwd Pkts"] = float(tot_bytes)
            row_dict["Flow Byts/s"] = float(tot_bytes / (dur / 1000)) if dur > 0 else 0.0
            row_dict["Flow Pkts/s"] = float(tot_fwd / (dur / 1000)) if dur > 0 else 0.0
            row_dict["Pkt Len Mean"] = float(np.mean(pkt_lens)) if pkt_lens else 0.0
            row_dict["Pkt Len Std"] = float(np.std(pkt_lens)) if pkt_lens else 0.0
            row_dict["Pkt Size Avg"] = float(np.mean(pkt_lens)) if pkt_lens else 0.0
            row_dict["SYN Flag Cnt"] = float(sum(1 for p in p_list if (p.tcp_flags & 0x02)))
            row_dict["ACK Flag Cnt"] = float(sum(1 for p in p_list if (p.tcp_flags & 0x10)))
            row_dict["RST Flag Cnt"] = float(sum(1 for p in p_list if (p.tcp_flags & 0x04)))
            row_dict["Init Fwd Win Byts"] = float(p_list[0].tcp_window) if p_list else 14600.0
            # Also attach packet features
            for pf_name, pf_val in pkt_features.items():
                row_dict[pf_name] = pf_val
            flow_rows.append(row_dict)

        df_pcap = pd.DataFrame(flow_rows)
        # Pad to at least 20 rows if smaller
        if len(df_pcap) < SEQUENCE_LENGTH:
            repeat_factor = int(np.ceil(SEQUENCE_LENGTH / len(df_pcap)))
            df_pcap = pd.concat([df_pcap] * repeat_factor, ignore_index=True).iloc[:SEQUENCE_LENGTH]

        return self.forecast_trajectory(df=df_pcap, horizon=horizon, filename=filename or os.path.basename(filepath))

    def simulate_defense(
        self,
        df: pd.DataFrame,
        action: str,
        target_value: Optional[Any] = None,
        rate_factor: Optional[float] = None,
        horizon: int = 5,
        filename: str = "traffic.csv",
    ) -> Dict[str, Any]:
        """
        Executes counterfactual What-If defense simulation on the temporal sequence using the World Model.
        Creates a modified network state, runs the identical LSTM World Model on both baseline and modified
        states, performs multi-step future rollout, and compares predicted risk across t+1...t+k.
        Supported Actions: Block IP, Quarantine Host, Close Port, Rate Limit
        """
        horizon = max(1, min(int(horizon), 20))
        seq_window, packet_meta = self.prepare_sequence_window(df)

        # Normalize action name
        action_clean = action.strip().lower()
        if action_clean in ("quarantine", "quarantine host", "quarantine_host"):
            action_clean = "quarantine_host"
        elif action_clean in ("isolate", "isolate host", "isolate_host"):
            action_clean = "quarantine_host"
        elif action_clean in ("close port", "close_port", "block_port", "block port"):
            action_clean = "close_port"
        elif action_clean in ("rate limit", "rate_limit", "rate limit traffic", "restrict_traffic", "restrict traffic"):
            action_clean = "rate_limit"
        elif action_clean in ("block ip", "block_ip"):
            action_clean = "block_ip"

        # Run counterfactual simulation via rollout engine (or pure NumPy fallback)
        if self.rollout_engine is not None:
            sim_result = self.rollout_engine.simulate_defense(
                initial_sequence=seq_window,
                action=action_clean,
                target_value=target_value,
                rate_factor=rate_factor,
                horizon=horizon,
            )
        else:
            from app.services.inference import InferenceService
            infer_service = InferenceService()
            base_res = infer_service.run_temporal_kstep_forecast(df, horizon=horizon, filename=filename)
            baseline_trajectory = []
            for item in base_res["timeline"]:
                s = item["step"]
                baseline_trajectory.append({
                    "step": s,
                    "risk_probability": item["risk"],
                    "risk_level": item["risk_category"],
                    "status": item["status"],
                    "description": item["description"],
                    "horizon_label": f"+{s}" if s > 0 else "NOW",
                })

            df_cf = df.copy()
            spec = dict(DEFENSE_PERTURBATION_SPEC.get(action_clean, {}))
            affected_feature_names = spec.get("affected_features", [])
            for feat_name in affected_feature_names:
                if feat_name in df_cf.columns:
                    if spec.get("operation") == "zero":
                        df_cf[feat_name] = 0.0
                    elif spec.get("operation") == "scale":
                        factor = spec.get("scale_factor", 0.1)
                        if rate_factor is not None:
                            factor = max(0.01, min(float(rate_factor), 0.99))
                        df_cf[feat_name] = df_cf[feat_name] * factor

            cf_res = infer_service.run_temporal_kstep_forecast(df_cf, horizon=horizon, filename=filename)
            cf_trajectory = []
            for item in cf_res["timeline"]:
                s = item["step"]
                cf_trajectory.append({
                    "step": s,
                    "risk_probability": item["risk"],
                    "risk_level": item["risk_category"],
                    "status": item["status"],
                    "description": item["description"],
                    "horizon_label": f"+{s}" if s > 0 else "NOW",
                })

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
            mitigation_pct = round(abs(min(0.0, risk_delta)) / (peak_base if peak_base > 0 else 1.0) * 100, 1)

            sim_result = {
                "action": action_clean,
                "selected_action": friendly_action_name,
                "description": spec.get("description", f"Counterfactual evaluation of {friendly_action_name}"),
                "affected_features": affected_feature_names,
                "baseline_trajectory": baseline_trajectory,
                "counterfactual_trajectory": cf_trajectory,
                "whatif_trajectory": cf_trajectory,
                "step_comparisons": step_comparisons,
                "baseline_peak_risk": round(peak_base, 4),
                "counterfactual_peak_risk": round(peak_cf, 4),
                "risk_delta": risk_delta,
                "mitigation_percentage": mitigation_pct,
                "risk_reduced": risk_delta < 0,
            }

        # Construct backward-compatible timelines
        base_timeline = [
            {
                "step": item["step"],
                "risk": item["risk_probability"],
                "risk_category": item["risk_level"],
                "status": item["status"],
                "description": item["description"],
                "horizon_label": item.get("horizon_label", f"+{item['step']}"),
            }
            for item in sim_result["baseline_trajectory"]
        ]

        cf_timeline = [
            {
                "step": item["step"],
                "risk": item["risk_probability"],
                "risk_category": item["risk_level"],
                "status": item["status"],
                "description": item["description"],
                "horizon_label": item.get("horizon_label", f"+{item['step']}"),
            }
            for item in sim_result["counterfactual_trajectory"]
        ]

        friendly_action = sim_result.get("selected_action", action_clean.replace("_", " ").title())
        base_peak = sim_result["baseline_peak_risk"]
        cf_peak = sim_result["counterfactual_peak_risk"]
        risk_delta = sim_result["risk_delta"]

        return {
            "success": True,
            "filename": filename,
            "action": action_clean,
            "selected_action": friendly_action,
            "target_value": target_value,
            "simulation": True,
            "simulation_type": "world_model_what_if_rollout",
            "model_version": "NetWorld-Unified-WorldModel" if (self.model and getattr(self.model, "input_size", 0) == NUM_UNIFIED_FEATURES) else "NetWorld-MultiHead-WorldModel",
            "description": sim_result["description"],
            "affected_features": sim_result["affected_features"],
            "affected_records": len(df),
            "affected_records_percentage": 100.0,
            "risk_change": risk_delta,
            "risk_change_percent": round(risk_delta * 100, 1),
            "mitigation_percentage": sim_result["mitigation_percentage"],
            "risk_reduced": sim_result["risk_reduced"],
            "step_comparisons": sim_result["step_comparisons"],
            "baseline_trajectory": sim_result["baseline_trajectory"],
            "counterfactual_trajectory": sim_result["counterfactual_trajectory"],
            "whatif_trajectory": sim_result["counterfactual_trajectory"],
            "baseline": {
                "risk": base_peak,
                "peak_risk": base_peak,
                "current_risk": sim_result["baseline_trajectory"][0]["risk_probability"] if sim_result["baseline_trajectory"] else base_peak,
                "risk_category": sim_result["baseline_trajectory"][0]["risk_level"] if sim_result["baseline_trajectory"] else "HIGH",
                "trajectory": sim_result["baseline_trajectory"],
                "timeline": base_timeline,
            },
            "counterfactual": {
                "risk": cf_peak,
                "peak_risk": cf_peak,
                "current_risk": sim_result["counterfactual_trajectory"][0]["risk_probability"] if sim_result["counterfactual_trajectory"] else cf_peak,
                "risk_category": sim_result["counterfactual_trajectory"][0]["risk_level"] if sim_result["counterfactual_trajectory"] else "LOW",
                "trajectory": sim_result["counterfactual_trajectory"],
                "timeline": cf_timeline,
            },
            "whatif": {
                "risk": cf_peak,
                "peak_risk": cf_peak,
                "current_risk": sim_result["counterfactual_trajectory"][0]["risk_probability"] if sim_result["counterfactual_trajectory"] else cf_peak,
                "risk_category": sim_result["counterfactual_trajectory"][0]["risk_level"] if sim_result["counterfactual_trajectory"] else "LOW",
                "trajectory": sim_result["counterfactual_trajectory"],
                "timeline": cf_timeline,
            },
            "summary": (
                f"What-If Defence Simulation for '{friendly_action}': "
                f"Peak infiltration risk shifted from {round(base_peak * 100, 1)}% to {round(cf_peak * 100, 1)}% "
                f"across lookahead horizons t+1...t+{horizon} (Model-predicted risk change: {round(risk_delta * 100, 1)}%)."
            ),
            "disclaimer": (
                "Model-predicted change in risk based on counterfactual perturbation of the physical network state "
                "and forward LSTM rollout. This simulation reports statistical risk shift and does not claim "
                "guaranteed physical prevention of real-world network attacks."
            ),
        }


# Singleton accessor
def get_world_model_service() -> WorldModelService:
    return WorldModelService()
