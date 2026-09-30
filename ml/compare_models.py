"""
NetWorld - Model Comparison Report
SIH 2026 Project

Loads evaluation metrics for NetWorld Temporal LSTM and Logistic Regression Baseline
and prints an unranked, objective comparison report.
"""

import os
import json


BENCHMARK_RESULTS_PATH = "models/benchmark_comparison_results.json"
LOGISTIC_RESULTS_PATH = "models/logistic_baseline_results.json"
EVALUATION_RESULTS_PATH = "models/evaluation_results.json"


def main():
    print("=" * 85)
    print("NetWorld Cyberattack Forecasting AI - Benchmark Comparison Report")
    print("=" * 85)

    lstm_metrics = None
    lr_metrics = None
    wm_metrics = None

    # Load unified benchmark results if available
    if os.path.exists(BENCHMARK_RESULTS_PATH):
        with open(BENCHMARK_RESULTS_PATH, "r") as f:
            bm_data = json.load(f)
            models = bm_data.get("models", {})
            lr_metrics = models.get("logistic_regression", {}).get("metrics", {})
            lstm_metrics = models.get("current_lstm", {}).get("metrics", {})
            wm_metrics = models.get("world_model", {}).get("metrics", {})

    # Fallback to individual result files
    if not lr_metrics and os.path.exists(LOGISTIC_RESULTS_PATH):
        with open(LOGISTIC_RESULTS_PATH, "r") as f:
            lr_data = json.load(f)
            lr_metrics = lr_data.get("metrics", {})
    
    if not lstm_metrics and os.path.exists(EVALUATION_RESULTS_PATH):
        with open(EVALUATION_RESULTS_PATH, "r") as f:
            eval_data = json.load(f)
            lstm_metrics = eval_data.get("models", {}).get("networld_lstm", {})

    # Print Measured Comparison Table (Unranked)
    print("\n" + "-" * 85)
    print(f"{'Model Name':<32} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1':<9} | {'FPR':<9}")
    print("-" * 85)

    if lr_metrics:
        lr_acc = lr_metrics.get("accuracy", 0.0)
        lr_prec = lr_metrics.get("precision", 0.0)
        lr_rec = lr_metrics.get("recall", 0.0)
        lr_f1 = lr_metrics.get("f1", 0.0)
        lr_fpr = lr_metrics.get("false_positive_rate", 0.0)
        print(f"{'Logistic Regression Baseline':<32} | {lr_acc:<9.4f} | {lr_prec:<9.4f} | {lr_rec:<9.4f} | {lr_f1:<9.4f} | {lr_fpr:<9.4f}")

    if lstm_metrics:
        lstm_acc = lstm_metrics.get("accuracy", 0.0)
        lstm_prec = lstm_metrics.get("precision", 0.0)
        lstm_rec = lstm_metrics.get("recall", 0.0)
        lstm_f1 = lstm_metrics.get("f1", 0.0)
        lstm_fpr = lstm_metrics.get("false_positive_rate", 0.0)
        print(f"{'Current NetWorld LSTM':<32} | {lstm_acc:<9.4f} | {lstm_prec:<9.4f} | {lstm_rec:<9.4f} | {lstm_f1:<9.4f} | {lstm_fpr:<9.4f}")

    if wm_metrics:
        wm_acc = wm_metrics.get("accuracy", 0.0)
        wm_prec = wm_metrics.get("precision", 0.0)
        wm_rec = wm_metrics.get("recall", 0.0)
        wm_f1 = wm_metrics.get("f1", 0.0)
        wm_fpr = wm_metrics.get("false_positive_rate", 0.0)
        print(f"{'NetWorld World Model':<32} | {wm_acc:<9.4f} | {wm_prec:<9.4f} | {wm_rec:<9.4f} | {wm_f1:<9.4f} | {wm_fpr:<9.4f}")

    print("-" * 85)

    print("\nExperimental Setup & Hyperparameters:")
    print("  - Evaluation Dataset: CIC-IDS2018 (20% Temporal Holdout Split)")
    print("  - Sequence Window: 20 sequential flows x 36 numerical features")
    print("  - NetWorld LSTM: 2-layer LSTM (hidden=128, dropout=0.2) with binary head")
    print("  - Logistic Regression: Trained on 720 flattened scaled features (20 x 36), class_weight=None, solver=lbfgs")

    print("\n" + "=" * 75)
    print("ARCHITECTURAL DIFFERENCES & METHODOLOGY ANALYSIS")
    print("=" * 75)
    print("""
1. Representation & State Memory:
   - Logistic Regression treats the 720 inputs as a static, unordered vector of independent variables.
     It computes a linear combination of all 720 features simultaneously without explicit temporal state tracking.
   - NetWorld Temporal LSTM processes the 20 network flows sequentially step-by-step.
     It maintains internal recurrent memory states (hidden state vector h_t and cell state c_t) to capture
     temporal order, state transitions, and step-by-step sequential dynamics across network flows.

2. Non-Linear Interaction Modeling:
   - Logistic Regression models a single linear decision boundary in 720-dimensional space.
   - NetWorld Temporal LSTM uses non-linear gate activations (input, forget, output gates, and ReLU heads)
     capable of learning complex non-linear flow correlations and multi-step attack patterns.

3. Objective Metric Note:
   - Measured metrics reflect performance on the 20% temporal test holdout.
   - Neither model is declared a universal winner; each presents distinct trade-offs between linear baseline simplicity
     and recurrent temporal pattern modeling.
""")
    print("=" * 75)


if __name__ == "__main__":
    main()
