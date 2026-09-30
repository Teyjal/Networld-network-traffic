Absolutely — here’s a **GitHub-ready README** for NetWorld. I’ve kept the claims aligned with what you’ve actually built/planned, rather than overstating the current prototype as a finished production world model.

````markdown
# 🌐 NetWorld
## Predictive Digital Twin for Proactive Network Attack Forecasting

> **Observe → Model → Forecast → Explain → Simulate → Decide**

NetWorld is a predictive cybersecurity research prototype designed to analyze the temporal evolution of network traffic, estimate infiltration risk, explain model predictions, map observed behaviour to attack stages, and simulate potential defensive interventions.

Unlike a conventional flow-level intrusion detector that evaluates network flows independently, NetWorld focuses on **temporal network behaviour** and aims to model how the network state may evolve over future steps.

---

## 🚀 Overview

Modern network attacks often develop through a sequence of changing behaviours rather than appearing as a single isolated malicious flow.

NetWorld addresses this challenge by transforming network traffic into time-ordered network states and applying temporal machine learning to identify patterns associated with infiltration.

The system combines:

- 📊 Network traffic preprocessing
- 🧠 Temporal LSTM-based risk prediction
- 🔮 Future-state / trajectory forecasting
- 🔍 SHAP-based explainability
- 🎯 MITRE ATT&CK behaviour mapping
- 🧪 What-If counterfactual defence simulation
- 📚 RAG/LLM-based contextual intelligence
- 🖥️ Interactive cybersecurity command center
- 🌙 Light and dark production-oriented UI

---

## 🎯 Problem

Traditional intrusion detection systems often focus on classifying individual network flows.

This can lose important temporal information about how an attack develops across multiple flows and time steps.

NetWorld explores a different approach:

```text
Network Traffic
      ↓
Temporal Network State
      ↓
LSTM Model
      ↓
Risk / Future State
      ↓
Attack Progression
      ↓
Explainability
      ↓
What-If Defence Simulation
````

---

## 💡 Core Idea

NetWorld represents network activity as a sequence of observations rather than treating every flow independently.

For the current temporal implementation:

```text
20-flow sequence × 36 features
              ↓
         LSTM Encoder
              ↓
       Network Representation
              ↓
       Infiltration Risk
```

The planned temporal world-model architecture extends this toward:

```text
Current Network State
          ↓
     LSTM Encoder
          ↓
   ┌──────┴─────────┐
   ↓                ↓
Next-State       Risk Prediction
Prediction
   ↓                ↓
S(t+1)          P(Infiltration)
   ↓
Feed predicted state back
   ↓
S(t+2)
   ↓
S(t+3)
   ↓
...
```

This enables a future trajectory rather than only a current classification.

---

# 🧠 System Architecture

```text
                 ┌─────────────────────┐
                 │   Network Traffic   │
                 │      CSV / PCAP     │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │    Preprocessing    │
                 │ Validation / Scaling│
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │  Temporal Network   │
                 │       State         │
                 │   20 × 36 features  │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │    LSTM Temporal    │
                 │        Model        │
                 └──────────┬──────────┘
                            ↓
              ┌─────────────┴─────────────┐
              ↓                           ↓
       Risk Prediction              State Forecast
              ↓                           ↓
              └─────────────┬─────────────┘
                            ↓
                  Future Risk Trajectory
                            ↓
            ┌───────────────┼───────────────┐
            ↓               ↓               ↓
          SHAP            MITRE          What-If
       Explainability     Mapping        Simulation
            ↓               ↓               ↓
            └───────────────┼───────────────┘
                            ↓
                   Defender Decision
```

---

# 🔬 Key Features

## 1. 🧠 Temporal Risk Prediction

NetWorld uses an LSTM-based temporal model to analyze sequences of network traffic.

Instead of evaluating a single flow independently, the model receives a sequence of observations:

```text
Flow t-19
Flow t-18
Flow t-17
...
Flow t
```

and estimates infiltration risk from the temporal pattern.

---

## 2. 🔮 Future Network Trajectory

The long-term objective of NetWorld is to evolve from risk classification into a temporal world model.

The system can represent:

```text
NOW → T+1 → T+2 → T+3 → T+4 → T+5
```

and visualize how predicted network behaviour and risk evolve over the forecast horizon.

This forms the basis of the **NetWorld Predictive Digital Twin**.

---

## 3. 🔍 Explainable AI with SHAP

NetWorld integrates SHAP-based explainability to answer:

> **Why did the model produce this risk prediction?**

The explanation identifies network features that contribute positively or negatively to the prediction.

Example:

```text
WHY THIS RISK?

Predicted Risk
     ↓
Top Contributing Features

Flow Duration       ↑
Packet Rate         ↑
Destination Port   ↑
Flow IAT            ↓
TCP Flags           ↑
```

The system is designed to use actual model-generated SHAP values rather than hard-coded explanations.

---

## 4. 🎯 MITRE ATT&CK Mapping

Predicted network behaviour can be mapped to relevant MITRE ATT&CK techniques and attack stages through a knowledge/rule-based mapping layer.

Conceptually:

```text
Predicted Behaviour
        ↓
Behaviour Evidence
        ↓
MITRE Technique
        ↓
Attack Stage
        ↓
Confidence
```

Potential stages include:

```text
Reconnaissance
      ↓
Initial Access
      ↓
Lateral Movement
      ↓
Command & Control
      ↓
Exfiltration
```

Only stages supported by the implemented evidence and mapping logic should be displayed.

---

## 5. 🧪 What-If Defence Simulator

NetWorld allows defenders to evaluate hypothetical interventions before applying them.

Example actions:

* Block Source IP
* Quarantine Host
* Close Destination Port
* Rate-limit aggressive flows

The simulation follows:

```text
                 BASELINE
                    │
             Current State
                    │
               World Model
                    │
             Future Trajectory
                    │
                   Risk


              WHAT-IF ACTION
                    │
             Modified State
                    │
               World Model
                    │
             Future Trajectory
                    │
                   Risk
```

The system compares the model-predicted trajectories.

> **Important:** The simulator represents a counterfactual model experiment. A simulated reduction in predicted risk is not a guarantee that a real attack would be prevented.

---

## 6. 📦 Flow + Packet-Level Features

The current model primarily uses flow-derived network features.

The planned unified representation can incorporate selected packet-level characteristics where packet/PCAP data supports them:

```text
FLOW FEATURES
├── Duration
├── Packets
├── Bytes
├── Flow IAT
└── TCP Flags

PACKET FEATURES
├── TTL statistics
├── TCP window statistics
├── Payload-size statistics
├── Retransmission count
└── Port-scan behaviour
```

These can be combined into a unified temporal network state.

---

## 7. 📚 RAG + LLM Intelligence

NetWorld can use Retrieval-Augmented Generation to connect model outputs with trusted cybersecurity knowledge.

Conceptually:

```text
Model Prediction
      +
SHAP Evidence
      +
MITRE Mapping
      ↓
Knowledge Retrieval
      ↓
Contextual LLM Response
      ↓
Evidence-linked Guidance
```

The LLM layer is intended to provide contextual explanations and guidance rather than replace the underlying ML prediction.

---

# 🖥️ Dashboard

NetWorld provides a cybersecurity command-center interface designed around:

```text
CURRENT STATE
      ↓
PREDICTED STATE
      ↓
ATTACK STAGE
      ↓
WHY THIS ALERT?
      ↓
WHAT-IF DEFENCE
      ↓
DEFENDER DECISION
```

The interface includes:

* Network activity visualization
* Current risk
* Predicted risk
* Future trajectory
* Attack-stage progression
* SHAP explanations
* Digital Twin visualization
* What-If Defence Simulator
* Validation metrics
* Light / Dark themes
* Traffic upload workflow

---

# 📊 Current Model

The current temporal model is an LSTM-based architecture:

```text
Input
36 features
     ↓
2-layer LSTM
128 hidden units
     ↓
Dense layer
64 units
     ↓
Risk output
```

Configuration:

```text
Input features: 36
Sequence length: 20
LSTM hidden size: 128
LSTM layers: 2
Dropout: 0.2
```

The trained model and scaler are stored separately from source code.

---

# 📈 Current Experiment

The current preprocessing pipeline uses CIC-IDS2018-derived network traffic.

Temporal sequences were constructed using:

```text
Sequence length = 20 flows
Features = 36
```

Current dataset preparation produced:

```text
Training sequences:   1,454,022
Validation sequences:   311,560
Test sequences:         311,562
```

The current trained LSTM evaluation produced:

| Metric    | Current LSTM |
| --------- | -----------: |
| Accuracy  |       0.6935 |
| Precision |       0.6874 |
| Recall    |       0.4319 |
| F1        |       0.5305 |
| FPR       |       0.1315 |

These values represent the current experimental model and should not be interpreted as production-level performance.

---

# 🧪 Benchmarking

NetWorld is intended to be evaluated against a conventional baseline using the **same features, preprocessing, and test split**.

Planned comparison:

| Model               | Precision | Recall |     F1 |    FPR |
| ------------------- | --------: | -----: | -----: | -----: |
| Logistic Regression |       TBD |    TBD |    TBD |    TBD |
| Temporal LSTM       |    0.6874 | 0.4319 | 0.5305 | 0.1315 |
| World Model         |       TBD |    TBD |    TBD |    TBD |

> Metrics marked `TBD` should be replaced only after the corresponding experiment has actually been executed.

---

# 📁 Project Structure

A typical project structure is:

```text
network_traffic/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── test_external/
│
├── models/
│   ├── networld_combined_temporal_lstm_best.pt
│   └── networld_combined_temporal_scaler.pkl
│
├── ml/
│   ├── preprocessing/
│   ├── sequence/
│   ├── training/
│   ├── evaluation/
│   └── explainability/
│
├── backend/
│   └── FastAPI services
│
├── frontend/
│   └── React application
│
├── notebooks/
│
├── scripts/
│
├── requirements.txt
│
└── README.md
```

> Adjust the structure above to match the actual repository folders before publishing.

---

# ⚙️ Technology Stack

### Machine Learning

* Python
* PyTorch
* LSTM
* Scikit-learn
* SHAP
* NumPy
* Pandas

### Backend

* FastAPI
* Python
* REST APIs

### Frontend

* React
* JavaScript / TypeScript
* Modern CSS
* Interactive data visualizations

### Cybersecurity Intelligence

* MITRE ATT&CK
* Retrieval-Augmented Generation
* Large Language Models

### Data

* CIC-IDS2018
* CSV network-flow telemetry
* PCAP/packet data where available

---

# 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/networld.git
cd networld
```

Create a Python environment:

```bash
python -m venv .venv
```

### Windows

```powershell
.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install frontend dependencies:

```bash
cd frontend
npm install
```

---

# ▶️ Running the Project

## Start the backend

From the backend/project root:

```bash
uvicorn <your_backend_module>:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

## Start the frontend

```bash
cd frontend
npm run dev
```

The development interface will normally be available at:

```text
http://localhost:5173
```

> Replace the backend module/path with the actual entry point used in the repository.

---

# 📥 Traffic Input

NetWorld is designed to process network traffic represented as time-ordered flow data.

For a compatible dataset:

```text
CSV
 ↓
Validation
 ↓
Feature Selection
 ↓
Scaling
 ↓
20-flow Temporal Window
 ↓
LSTM
 ↓
Risk Prediction
```

External datasets should not be passed directly into the model unless their features have been mapped to the model's expected representation.

---

# 🔐 Security & Responsible Use

NetWorld is a **research and defensive cybersecurity prototype**.

It is intended for:

* Security research
* Network monitoring
* Threat analysis
* Model evaluation
* Defensive simulation
* Educational demonstrations

The system should be tested only on networks, datasets, and environments for which the user has appropriate authorization.

The What-If simulator should be treated as a model-based counterfactual experiment unless explicitly integrated into an authorized isolated test environment.

---

# ⚠️ Limitations

NetWorld is a research prototype and has several limitations:

* Current model performance is dataset-dependent.
* CIC-IDS2018-derived evaluation does not automatically establish real-world generalization.
* The current LSTM implementation primarily performs temporal infiltration-risk prediction.
* A full future-state world model requires explicit state-transition training and validation.
* Packet-level features require suitable packet/PCAP telemetry.
* MITRE stage mapping depends on the quality of the implemented evidence/mapping layer.
* LLM-generated guidance can contain errors and should be treated as supporting intelligence.
* What-If results represent model-predicted counterfactual changes rather than guaranteed real-world outcomes.

---

# 🔭 Future Work

Planned development includes:

### 1. True Temporal World Model

Add an explicit state-transition head:

```text
S(t) → S(t+1)
```

and use the predicted state for multi-step rollout.

### 2. K-Step Forecasting

Implement:

```text
NOW
 ↓
T+1
 ↓
T+2
 ↓
T+3
 ↓
...
T+k
```

with future risk trajectories.

### 3. Packet-Level Integration

Add validated packet-derived features such as:

* TTL
* TCP window
* Payload statistics
* Retransmissions
* Port-scan behaviour

### 4. Cross-Dataset Evaluation

Evaluate generalization on an unseen dataset without retraining where compatible feature mapping is possible.

### 5. Model Benchmarking

Compare:

```text
Logistic Regression
        vs
Temporal LSTM
        vs
Temporal World Model
```

using the same evaluation protocol.

### 6. Improved Counterfactual Modelling

Connect What-If interventions directly to the world-model state representation and evaluate their effect over multiple future steps.

---

# 🏆 Project Vision

NetWorld is built around a simple idea:

> **Don't just ask whether an attack is happening. Ask where the network is heading.**

The intended workflow is:

```text
OBSERVE
   ↓
MODEL
   ↓
FORECAST
   ↓
EXPLAIN
   ↓
SIMULATE
   ↓
DECIDE
```

NetWorld aims to move network defence from purely reactive detection toward **predictive, explainable, and simulation-assisted decision support**.

---


