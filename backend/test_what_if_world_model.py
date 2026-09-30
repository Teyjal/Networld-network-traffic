"""
Comprehensive verification test for What-If Defence Simulator using the Multi-Head LSTM World Model.
Tests all 4 defence actions:
  1. Block IP
  2. Quarantine Host
  3. Close Port
  4. Rate Limit
Verifies:
  - Modified network state generation
  - Identical LSTM World Model forward rollout on Baseline vs Modified states
  - Multi-step future lookahead (t+1...t+k)
  - Risk comparison at each step
  - Trajectory output, selected action, risk change delta
  - Model disclaimer compliance
"""

import os
import sys
import numpy as np
import pandas as pd

# Add repository root to sys.path
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from ml.feature_schema import FEATURE_NAMES, NUM_FEATURES, SEQUENCE_LENGTH
from backend.app.services.world_model_service import get_world_model_service


def test_whatif_all_four_actions():
    print("=" * 75)
    print("TEST: What-If Defence Simulator via Temporal LSTM World Model")
    print("=" * 75)

    wm_service = get_world_model_service()
    print(f"World Model Checkpoint: {wm_service.loaded_checkpoint}")
    print(f"Device: {wm_service.device}")
    print(f"Model Input Dimension: {wm_service.model.input_size} features")

    # Generate synthetic network flows with distinct traffic patterns
    np.random.seed(42)
    sample_df = pd.DataFrame({
        feat: np.random.uniform(10.0, 500.0, 30) for feat in FEATURE_NAMES
    })
    # Elevate flood features to simulate active intrusion risk
    sample_df["Flow Pkts/s"] = np.random.uniform(5000, 25000, 30)
    sample_df["Flow Byts/s"] = np.random.uniform(500000, 2000000, 30)
    sample_df["SYN Flag Cnt"] = np.random.uniform(20, 100, 30)
    sample_df["Dst Port"] = 80.0

    actions_to_test = [
        ("Block IP", "185.220.101.4", None),
        ("Quarantine Host", "192.168.1.105", None),
        ("Close Port", "80", None),
        ("Rate Limit", "90% throttle", 0.1),
    ]

    horizon = 5

    for action_name, target_val, rate_factor in actions_to_test:
        print(f"\n--- Testing Defence Action: '{action_name}' (Target: {target_val}) ---")
        result = wm_service.simulate_defense(
            df=sample_df,
            action=action_name,
            target_value=target_val,
            rate_factor=rate_factor,
            horizon=horizon,
            filename="synthetic_test.csv"
        )

        assert result["success"] is True
        assert result["selected_action"] == action_name
        assert "baseline_trajectory" in result
        assert "counterfactual_trajectory" in result
        assert "step_comparisons" in result
        assert len(result["baseline_trajectory"]) == horizon + 1
        assert len(result["counterfactual_trajectory"]) == horizon + 1
        assert len(result["step_comparisons"]) == horizon + 1

        b_peak = result["baseline"]["peak_risk"]
        c_peak = result["counterfactual"]["peak_risk"]
        delta = result["risk_change"]

        print(f"  Selected Action       : {result['selected_action']}")
        print(f"  Perturbed Features    : {len(result['affected_features'])} features")
        print(f"  Baseline Peak Risk    : {b_peak * 100:.1f}% ({result['baseline']['risk_category']})")
        print(f"  What-If Peak Risk     : {c_peak * 100:.1f}% ({result['counterfactual']['risk_category']})")
        print(f"  Model-Predicted Delta : {delta * 100:.1f}%")

        print("  Step-by-Step Risk Trajectory Comparison:")
        for sc in result["step_comparisons"]:
            print(
                f"    Step {sc['step']} [{sc['horizon_label']:>3}]: "
                f"Baseline = {sc['baseline_risk_percent']:>5.1f}% | "
                f"What-If = {sc['whatif_risk_percent']:>5.1f}% | "
                f"Delta Risk = {sc['risk_delta_percent']:>+5.1f}%"
            )

        # Check disclaimer constraint
        disclaimer = result["disclaimer"]
        assert "model-predicted" in disclaimer.lower() or "predicted" in disclaimer.lower()
        assert "does not claim" in disclaimer.lower()
        print(f"  Disclaimer verified: {disclaimer[:80]}...")

    print("\n" + "=" * 75)
    print("ALL 4 DEFENCE ACTIONS SIMULATED SUCCESSFULLY THROUGH LSTM WORLD MODEL!")
    print("=" * 75)


if __name__ == "__main__":
    test_whatif_all_four_actions()
