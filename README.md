# 🌐 NetWorld: A Predictive Digital Twin for Proactive Network Attack Forecasting

<div align="center">

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![React](https://img.shields.io/badge/React-19.0-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![Vite](https://img.shields.io/badge/Vite-6.0+-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev)
[![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-v4-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com)
[![SIH](https://img.shields.io/badge/Smart%20India%20Hackathon-2026-FF9933?style=for-the-badge)](https://sih.gov.in)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

<br />

> **“Forecast the trajectory. Explain the risk. Test the response.”**  
> NetWorld transforms network security operations from reactive alert triaging into proactive, counterfactual attack forecasting through a predictive deep temporal world model and digital twin.

[Features](#-key-features) • [Architecture](#-end-to-end-architecture) • [Empirical Benchmarks](#-empirical-research-validation) • [Quick Start](#-quick-start-installation) • [API Reference](#-api-endpoints) • [Workflow](#-closed-loop-operational-workflow)

</div>

---

## 📌 Executive Summary

Conventional Network Intrusion Detection Systems (NIDS) and SIEM solutions operate retrospectively: they inspect isolated flows, apply signature rules or static ML classifiers, and answer a singular question: **"Is this current packet or flow malicious?"**

By the time an alert fires, perimeter ingress has succeeded, credentials have been compromised, and adversaries have begun lateral movement.

**NetWorld fundamentally shifts the paradigm:**
- **Network as a Dynamic Temporal System**: Formulates network activity as a continuous sequence of high-dimensional state vectors $S_t = [\text{traffic behaviour} + \text{packet behaviour} + \text{temporal statistics}]$.
- **Deep Temporal World Model (LSTM)**: Learns environmental transition dynamics $P(S_{t+1} \mid S_t)$ to project future infiltration states $T+1 \dots T+K$ steps ahead without waiting for packet arrivals.
- **Explainable Threat Attribution**: Pairs SHAP feature importance with MITRE ATT&CK tactics (Reconnaissance $\to$ Initial Access $\to$ Lateral Movement $\to$ C2 $\to$ Exfiltration) to validate the underlying attack progression.
- **Counterfactual "What-If" Defence Engine**: Allows SOC defenders to simulate defensive interventions (`Block IP`, `Quarantine Host`, `Close Port`, `Rate Limit`, `Block Protocol`) directly in the latent world model, observing trajectory divergence ($\Delta Risk$) before committing actions to the production network.

---

## 🚀 Key Features

### 1. 🎛️ Aerospace-Grade Predictive Command Center
- Real-time command dashboard featuring high-contrast White + Navy Blue aesthetics.
- Horizon threat radar, dynamic telemetry gauges, live alert tickers, and network health monitors.
- Multi-language localization support and theme management.

### 2. 🧬 Deep Temporal World Model ($P(S_{t+1} \mid S_t)$)
- Multi-head recurrent architecture with dual outputs:
  1. **Risk Prediction Head**: Computes calibrated attack probability logit $\hat{y}_{t+1}$.
  2. **Next-State Decoder Head**: Predicts physical flow feature transitions $\hat{S}_{t+1} \in \mathbb{R}^{36}$ with an empirical MAE of **0.3685**.
- Latent state representation visualization ($z_t \in \mathbb{R}^{128}$) capturing burst dynamics and flow volume evolution.

### 3. 🔮 K-Step Autoregressive Attack Forecasting
- Unrolls predictions $T+1, T+2, \dots, T+K$ into the future.
- Generates predictive confidence bands, stage transition likelihoods, and lead-time alerts before critical impact thresholds are breached.

### 4. 🧪 Counterfactual "What-If" Defense Simulator
- Test hypothetical defensive mitigations against active attack trajectories:
  - `Block Malicious IP / CIDR Subnet`
  - `Quarantine / Isolate Compromised Host`
  - `Sever / Close Target Port (e.g. 445 SMB, 3389 RDP)`
  - `Apply Token Bucket Rate Limiting`
  - `Enforce Protocol Boundary Filters`
- Real-time side-by-side trajectory comparison: **Baseline Risk vs. Counterfactual Risk ($\Delta$)**.

### 5. 🔍 Explainable AI (XAI) & MITRE ATT&CK Mapping
- **SHAP (SHapley Additive exPlanations)**: Identifies exact flow and packet features driving elevated risk (e.g., forward inter-arrival time, TCP SYN anomalies, packet length variance).
- **MITRE ATT&CK Correlation**: Automatically maps telemetry indicators to adversary tactics and techniques across the Cyber Kill Chain.

### 6. 🌐 Predictive Network Digital Twin
- Interactive topology graph mapping ingress routers, DMZ web services, corporate subnets, and core database assets.
- Node risk propagation and predictive blast-radius heatmaps.

### 7. 🛡️ Closed-Loop Defender Playbooks
- Priority mitigation queues with calculated Blast Radius scores, Implementation Complexity, and Estimated MTTD/MTTR reduction.
- Automated generation of firewall commands (`iptables`, AWS Security Group rules, Cisco ACL snippets).

### 8. 📄 Built-In Architecture Dossier (`DOC NW-ARCH`)
- In-app interactive reproduction of the two-page classified research dossier and system schematic. Accessible directly from the bottom of the dashboard sidebar.

---

## 🔄 Closed-Loop Operational Workflow

```
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                       NETWORLD CLOSED-LOOP ENGINE                       │
   └─────────────────────────────────────────────────────────────────────────┘
                                        │
                         [1] OBSERVE TELEMETRY
                             Flows, Packets, Windows (S_t)
                                        │
                                        ▼
                         [2] MODEL DYNAMICS (LSTM)
                             Learns P(S_{t+1} | S_t) in Latent Space
                                        │
                                        ▼
                         [3] FORECAST TRAJECTORY
                             Autoregressive Rollout T+1 ... T+K
                                        │
                                        ▼
                         [4] EXPLAIN PREDICTION
                             SHAP Feature Importance & MITRE Kill Chain
                                        │
                                        ▼
                         [5] SIMULATE DEFENCE (WHAT-IF)
                             Inject Intervention -> Measure Δ Risk
                                        │
                                        ▼
                         [6] DECIDE & EXECUTE
                             Commit Proven Defence Playbook to Production
```

---

## 🏗️ End-to-End Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 01. INGESTION & DATA LAYER                                                  │
│ CSV / PCAP / Live Telemetry (CIC-IDS2017 & CSE-CIC-IDS2018)                 │
│ Features: Flow Duration, IAT Statistics, TCP Window, TTL, Flags, Bytes/s   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 02. STATE BUILDER LAYER                                                     │
│ Data Cleaning ➔ Robust Normalization ➔ Chronological Sort ➔ Sliding Windows │
│ Sequence: [S_{t-19}, S_{t-18}, ..., S_t] where S_t ∈ R^36                   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 03. TEMPORAL WORLD MODEL LAYER (PyTorch Multi-Head LSTM)                    │
│ Latent Temporal Encoder: h_t, c_t ∈ R^128                                   │
│ ├─ Head A (Risk): Calibrated Infiltration Probability logit y_hat           │
│ └─ Head B (State Dynamics): Next Feature State Prediction S_hat_{t+1}       │
└───────────────────┬─────────────────────────────────────┬───────────────────┘
                    │                                     │
                    ▼                                     ▼
┌──────────────────────────────────────┐┌─────────────────────────────────────┐
│ 04. K-STEP FORECAST LAYER            ││ 05. COUNTERFACTUAL WHAT-IF ENGINE   │
│ Recursive Autoregressive Rollout:    ││ Injects action operator: A(S_t)     │
│ S_t ➔ S_{t+1} ➔ ... ➔ S_{t+k}        ││ Generates counterfactual trajectory │
│ Computes continuous trajectory risk  ││ Computes risk reduction delta (Δ)   │
└───────────────────┬──────────────────┘└─────────────────┬───────────────────┘
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 06. EXPLAINABILITY & TRIAGE LAYER                                           │
│ ├─ SHAP KernelExplainer: Attributing top driving network features           │
│ └─ MITRE ATT&CK Mapper: Tactics (Recon ➔ Initial Access ➔ Lateral ➔ Exfil)  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 07. COMMAND CENTER FRONTEND (React 19 + TypeScript + Vite + Tailwind CSS v4) │
│ CommandCenter • TrafficAnalysis • WorldModel • AttackForecast • DigitalTwin │
│ WhatIfSimulator • DefenderResponse • Explainability • Validation • Dossier  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Empirical Research Validation

Every metric below is an **empirically measured value** computed on the exact same **311,562 held-out test sequences** (186,637 Benign, 124,925 Attack) from the **CIC-IDS2018** temporal split.

### Model Benchmark Comparison Table

| Model Architecture | Accuracy | Precision | Recall | F1 Score | False Positive Rate (FPR) | Next-State Dynamics Head | Counterfactual What-If |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression Baseline** | 67.84% | 64.35% | 44.38% | 52.53% | 16.46% (30,714 false alarms) | None (Static Flattened) | ❌ Unsupported |
| **Standard NetWorld LSTM** | **69.35%** | **68.74%** | 43.19% | 53.05% | **13.15%** (24,535 false alarms) | None (Classification only) | ❌ Unsupported |
| **NetWorld World Model (Multi-Head)** | **69.31%** | **68.42%** | **43.58%** | **53.24%** | **13.47%** (25,132 false alarms) | **Explicit Head (MAE: 0.3685)** | **Supported ($t+1 \dots t+k$)** |

### Confusion Matrix Breakdown ($N = 311,562$)

| Metric | Logistic Regression | Standard NetWorld LSTM | NetWorld World Model |
| :--- | :---: | :---: | :---: |
| **True Negatives (TN)** | 155,923 | 162,102 | 161,505 |
| **False Positives (FP)** | 30,714 | 24,535 | 25,132 |
| **False Negatives (FN)** | 69,480 | 70,968 | 70,485 |
| **True Positives (TP)** | 55,445 | 53,957 | 54,440 |

> **Operational Insight:** NetWorld's temporal sequence modeling eliminates over **5,500 false alerts** compared to standard linear baselines, while unlocking state transition forecasting ($P(S_{t+1} \mid S_t)$) and counterfactual what-if simulation that no static classifier can perform.

---

## 💻 Tech Stack

### Frontend Command Suite
- **Framework**: [React 19](https://react.dev/) + [TypeScript](https://www.typescriptlang.org/)
- **Routing & SSR**: [TanStack Router](https://tanstack.com/router) & [TanStack Start](https://tanstack.com/start)
- **Styling**: [Tailwind CSS v4](https://tailwindcss.com/) with high-contrast White + Navy Blue design tokens
- **Data Visualization**: [Recharts](https://recharts.org/), Custom SVG Digital Twin Canvas
- **Motion & UI**: [Framer Motion](https://www.framer.com/motion/), [Lucide React Icons](https://lucide.dev/), [Radix UI](https://www.radix-ui.com/)
- **Build Tool**: [Vite 6](https://vitejs.dev/)

### Backend & Machine Learning
- **API Framework**: [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
- **ML / Deep Learning**: [PyTorch](https://pytorch.org/), [Scikit-learn](https://scikit-learn.org/)
- **Explainability**: [SHAP](https://shap.readthedocs.io/) (KernelExplainer)
- **Data Processing**: [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), Joblib
- **Cybersecurity Standards**: [MITRE ATT&CK v14](https://attack.mitre.org/) Framework

---

## 📁 Repository Structure

```text
network_traffic-main/
├── backend/                        # FastAPI Application & ML Services
│   ├── app/
│   │   ├── main.py                 # FastAPI entrypoint & middleware configuration
│   │   ├── routes/
│   │   │   ├── traffic.py          # CSV/PCAP ingestion & flow inspection
│   │   │   ├── forecast.py         # Multi-step autoregressive forecasting
│   │   │   ├── explain.py          # SHAP attribution & MITRE mapping
│   │   │   ├── whatif.py           # Counterfactual intervention engine
│   │   │   └── validation.py       # Live benchmark & evaluation metrics
│   │   ├── services/
│   │   │   ├── preprocessing.py    # Feature normalization & window slicing
│   │   │   ├── inference.py        # Model forward pass handler
│   │   │   ├── counterfactual.py   # What-if perturbation logic
│   │   │   ├── mitre_mapper.py     # ATT&CK tactic/technique association
│   │   │   └── shap_service.py     # Feature attribution computation
│   │   └── model/
│   │       └── loader.py           # Torch checkpoint & scaler loader
│   ├── requirements.txt            # Python dependencies
│   └── uploads/                    # Telemetry staging area
│
├── ml/                             # Offline Training & Evaluation Pipeline
│   ├── feature_schema.py           # Standardized 36/51 network feature definitions
│   ├── world_model_multihead.py    # PyTorch Multi-Head LSTM Architecture
│   ├── rollout.py                  # Autoregressive forward simulation engine
│   ├── train_combined_temporal.py  # Model training pipeline
│   └── evaluate_benchmark_comparison.py # Full empirical evaluation test suite
│
├── models/                         # Serialized Model Artifacts
│   ├── networld_combined_temporal_world_model.pth # Pretrained PyTorch weights
│   ├── networld_combined_temporal_scaler.pkl      # Pre-fitted StandardScaler
│   └── evaluation_results.json    # Canonical empirical test split metrics
│
├── src/                            # React 19 + TanStack Command Suite
│   ├── components/
│   │   ├── layout/AppShell.tsx     # Command center layout & sidebar navigation
│   │   ├── dossier/ArchitectureDossierModal.tsx # Classified research report viewer
│   │   ├── common/                 # Reusable buttons, cards, panels, badges
│   │   └── visualizers/            # Dynamic radar, graph, and trajectory charts
│   ├── pages/
│   │   ├── CommandCenter.tsx       # Central executive operations dashboard
│   │   ├── TrafficAnalysis.tsx     # Flow telemetry inspector & parser
│   │   ├── WorldModel.tsx          # Latent state dynamics inspector
│   │   ├── AttackForecast.tsx      # Multi-step predictive timeline
│   │   ├── DigitalTwin.tsx         # Interactive topology & blast radius
│   │   ├── WhatIfSimulator.tsx     # Counterfactual intervention laboratory
│   │   ├── DefenderResponse.tsx    # Automated response playbooks
│   │   ├── Explainability.tsx      # SHAP feature importance & MITRE mapping
│   │   └── Validation.tsx          # Empirical benchmark evaluation dashboards
│   ├── context/                    # Theme (Navy/White) & Language contexts
│   └── services/api.ts             # Typed REST API client
│
├── BENCHMARK_REPORT.md             # Detailed benchmark documentation
├── package.json                    # Node dependencies & build scripts
└── vite.config.ts                  # Vite build & proxy configuration
```

---

## ⚡ Quick Start & Installation

### Prerequisites
- **Node.js**: v18.0.0 or higher
- **Python**: v3.10 or higher
- **Package Manager**: `npm`, `yarn`, `pnpm`, or `bun`

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/<your-username>/network_traffic-main.git
cd network_traffic-main
```

---

### Step 2: Backend Setup (FastAPI & ML Engine)

1. Open a terminal and navigate to the `backend/` directory:
   ```bash
   cd backend
   ```

2. Create and activate a Python virtual environment:
   ```bash
   # On macOS/Linux:
   python3 -m venv venv
   source venv/bin/activate

   # On Windows (PowerShell):
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install required dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Start the FastAPI backend server:
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

   The backend will be live at [`http://localhost:8000`](http://localhost:8000).  
   Interactive Swagger docs are available at [`http://localhost:8000/docs`](http://localhost:8000/docs).

---

### Step 3: Frontend Setup (React Command Suite)

1. Open a second terminal at the project root:
   ```bash
   # Install frontend dependencies
   npm install
   ```

2. Launch the Vite development server:
   ```bash
   npm run dev
   ```

3. Open your browser and navigate to:
   ```text
   http://localhost:8080
   ```

---

## 📡 API Endpoints

| HTTP Method | Route | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Check backend server and model availability status |
| `POST` | `/api/traffic/upload` | Ingest and parse network telemetry CSV/PCAP |
| `GET` | `/api/traffic/flows` | Retrieve parsed flows and statistical window metrics |
| `POST` | `/api/forecast/predict` | Generate $K$-step autoregressive future risk trajectory |
| `POST` | `/api/whatif/simulate` | Simulate counterfactual defence actions (`Block IP`, etc.) |
| `POST` | `/api/explain/shap` | Compute feature attributions and driving factors |
| `GET` | `/api/explain/mitre` | Retrieve MITRE ATT&CK tactic/technique alignment |
| `GET` | `/api/validation/benchmarks`| Fetch empirical evaluation results on CIC-IDS2018 |

---

## 🎯 Usage Walkthrough

1. **Ingest Telemetry**:
   - On the top header, click **Upload Traffic** and select a network CSV (e.g. from CIC-IDS2017/2018) or use the pre-loaded benchmark baseline.
2. **Observe Current State**:
   - Navigate to **Traffic Analysis** to inspect packet arrival rates, flow sizes, and TCP flags.
3. **Execute Attack Forecast**:
   - Click **Run Forecast** or open **Attack Forecast** to view the projected multi-step trajectory ($T+1 \dots T+K$) and estimated infiltration probability.
4. **Inspect XAI & MITRE Mapping**:
   - Visit **Explainability** to understand *why* the model predicts an attack: view top SHAP feature drivers and the active Cyber Kill Chain tactic.
5. **Run What-If Defence Simulation**:
   - Open **What-If Simulator**, select a defensive intervention (e.g., *Isolate Host* or *Block IP*), and observe the immediate risk drop ($\Delta Risk$) on the future trajectory.
6. **Deploy Defender Response**:
   - Switch to **Defender Response** to generate remediation scripts and firewall configurations.
7. **Read Architecture Report**:
   - Click **ARCHITECTURE REPORT** at the bottom of the left sidebar at any time to open the built-in classified research dossier.

---

## 🛡️ License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👥 Acknowledgments & References

- **Dataset**: Canadian Institute for Cybersecurity — [CIC-IDS2017 & CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html)
- **Taxonomy**: MITRE ATT&CK® Enterprise Framework — [MITRE Corporation](https://attack.mitre.org/)
- **Research Framework**: Developed for the **Smart India Hackathon (SIH 2026)** — Document Reference `DOC NW-ARCH`.

<div align="center">
<sub>Engineered with precision for Next-Generation Predictive Cyber Defense.</sub>
</div>
