# NetWorld Model Benchmark Evaluation Report

**Evaluation Split**: CIC-IDS2018 (Combined Temporal) - Temporal Holdout Test Split (20%)  
**Total Held-Out Test Samples**: 311,562 (186,637 Benign, 124,925 Attack)  
**Input Window**: 20 sequential flows $\times$ 36 features (720 flattened for Logistic Regression)  
**Standard Scaler**: Exact same fitted `StandardScaler` (`models/networld_combined_temporal_scaler.pkl`) applied identically across all models.

---

## 1. Measured Performance Comparison Table

> **Evaluation Policy Note**: Every metric below is an **empirically measured value** computed directly on the exact same 311,562 held-out test sequences. No values are simulated, extrapolated, or hard-coded.

| Model | Accuracy | Precision | Recall | F1 Score | False Positive Rate (FPR) | Next-State Dynamics $S(t+1)$ | What-If Simulation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Linear Baseline)** | **67.84%** (0.6784) | **64.35%** (0.6435) | **44.38%** (0.4438) | **52.53%** (0.5253) | **16.46%** (0.1646) | None (Static) | ❌ Unsupported |
| **Current NetWorld LSTM** | **69.35%** (0.6935) | **68.74%** (0.6874) | **43.19%** (0.4319) | **53.05%** (0.5305) | **13.15%** (0.1315) | None (Classification only) | ❌ Unsupported |
| **NetWorld World Model (Multi-Head)** | **69.31%** (0.6931) | **68.42%** (0.6842) | **43.58%** (0.4358) | **53.24%** (0.5324) | **13.47%** (0.1347) | **Explicit Head (MAE: 0.3685)** |  **Supported ($t+1 \dots t+k$)** |

---

## 2. Confusion Matrix Breakdown (Exact Counts on 311,562 Test Samples)

| Metric / Count | Logistic Regression Baseline | Current NetWorld LSTM | NetWorld World Model |
| :--- | :---: | :---: | :---: |
| **True Negatives (TN)** | 155,923 | 162,102 | 161,505 |
| **False Positives (FP)** | 30,714 | 24,535 | 25,132 |
| **False Negatives (FN)** | 69,480 | 70,968 | 70,485 |
| **True Positives (TP)** | 55,445 | 53,957 | 54,440 |
| **Total Test Evaluated** | **311,562** | **311,562** | **311,562** |

---

## 3. In-Depth Architectural & Operational Analysis

### A. Logistic Regression Baseline (Linear Projection)
- **Input Representation**: Flattens the $20 \times 36$ temporal matrix into an unstructured $720$-dimensional static feature vector.
- **Mechanism**: Learns a single linear hyperplane $z = W x + b$. It ignores sequence order entirely, treating flow 1 and flow 20 as unordered independent dimensions.
- **Operational Reality**:
  - Achieves **44.38% Recall**, detecting 55,445 attack instances.
  - However, it suffers from a **16.46% False Positive Rate (FPR)**, generating **30,714 false alarms** on benign traffic.
  - In a real Security Operations Center (SOC), a 16.46% FPR on high-throughput enterprise networks would generate tens of thousands of bogus alerts daily, causing severe alert fatigue.

### B. Current NetWorld LSTM (Recurrent Temporal Classifier)
- **Input Representation**: Processes 20 network flows sequentially step-by-step as $(B, 20, 36)$.
- **Mechanism**: Maintains recurrent hidden states ($h_t$) and cell memory states ($c_t$) across the 20-step temporal window, allowing it to capture sequential order, evolving flow volume, and burst patterns.
- **Operational Reality**:
  - Significantly suppresses false alarms: FPR drops from **16.46% down to 13.15%** (reducing false positives by **6,179 events**).
  - Elevates **Precision to 68.74%** (vs. 64.35% for Logistic Regression).
  - Limitation: It is strictly a **discriminative classifier**. It can only output a static risk score for the current moment $t$ and has no capability to simulate future states or forecast counterfactual defence actions.

### C. NetWorld World Model (Predictive Multi-Head Dynamics Model)
- **Input Representation**: Sequential temporal window $(B, 20, 36)$ mapped into a latent temporal representation $z_t \in \mathbb{R}^{128}$.
- **Dual-Head Architecture**:
  1. **Risk Prediction Head**: Outputs calibrated infiltration probability logit $\hat{y}_{t+1}$ with **68.42% Precision**, **43.58% Recall**, and low **13.47% FPR**.
  2. **State Transition Decoder Head**: Autoregressively predicts the physical network flow vector $\hat{S}_{t+1} \in \mathbb{R}^{36}$ with a measured Mean Absolute Error (MAE) of **0.3685**.
- **Operational Advantage**:
  - **Autoregressive Multi-Step Rollout**: Enables recursive projection over $t+1 \dots t+k$ steps without waiting for external packet arrivals.
  - **What-If Defence Simulation**: Enables operators to inject interventions (`Block IP`, `Quarantine Host`, `Close Port`, `Rate Limit`) directly into the predicted state $\hat{S}_{t+\tau}$ and measure the resulting trajectory divergence $\Delta Risk_{t+1 \dots t+k}$ before taking action in the production environment.

---

## 4. Summary Table for Slide Presentation

```text
========================================================================================================
MODEL BENCHMARK COMPARISON ON HELD-OUT TEST SPLIT (N = 311,562)
========================================================================================================
Model                         Accuracy   Precision   Recall     F1 Score   FPR       What-If Capable
--------------------------------------------------------------------------------------------------------
Logistic Regression Baseline   67.84%     64.35%     44.38%    52.53%    16.46%   No (Static Linear)
Current NetWorld LSTM          69.35%     68.74%     43.19%    53.05%    13.15%   No (Classifier Only)
NetWorld World Model           69.31%     68.42%     43.58%    53.24%    13.47%   Yes (Autoregressive Rollout)
========================================================================================================
```
