# AEGIS AI: Explainable Agentic Predictive Maintenance

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)

**AEGIS AI** ("Intelligent Shield") is an advanced explainable predictive maintenance framework designed for **Remaining Useful Life (RUL) estimation and intelligent equipment health monitoring**. Built using NASA's **C-MAPSS turbofan engine dataset**, the framework bridges the gap between deep learning predictions and actionable maintenance decisions by integrating **Explainable AI (XAI)**, **Uncertainty Quantification using Monte Carlo Dropout**, and a **Rule-based Multi-Agent Reasoning System**.

---

##  Key Features

1. **Deep Time-Series Forecasting**
   Implements sliding-window preprocessing over multivariate operational sensor data to capture temporal degradation patterns and predict future equipment health.

2. **Uncertainty-Aware Predictions**
   Uses **Monte Carlo Dropout** to estimate predictive uncertainty and provide confidence-aware RUL predictions, helping identify potentially high-risk predictions.

3. **Transparent Occlusion-Based XAI**
   Identifies the sensor inputs and temporal regions that have the greatest influence on the predicted RUL, improving the interpretability of the deep learning model.

4. **Rule-based Multi-Agent Diagnostic Chain**
   Implements a sequential reasoning architecture:

   **Diagnostic → Root-Cause → Verification → Maintenance**

   The agents analyze predicted degradation and anomaly signals to identify potential causes, verify supporting evidence, and generate prioritized maintenance recommendations.

5. **Interactive Digital Twin Control Center**
   Provides a Streamlit-based dashboard for monitoring sensor telemetry, visualizing degradation trends, exploring model predictions, exploring an *illustrative heuristic* stress scenario, and interpreting model outputs.

---

## 📊 Model Performance

### NASA C-MAPSS FD001

Evaluated on the official unseen test split of the **C-MAPSS FD001 benchmark** (100 test engines, final 30-cycle window per engine) using the standard LSTM checkpoint `data/processed/lstm_rul_model.pth`. Reproduce with `python evaluate.py`; the dashboard reads the same numbers from `data/processed/test_predictions.csv`.

| Metric (training-consistent scaler, see Scaler note) |           Result |
| ------------- | ---------------: |
| **Test MAE**  | **16.66 cycles** |
| **Test RMSE** | **23.60 cycles** |

### Model Architecture

* **Model:** 2-Layer LSTM
* **Hidden Units:** 64
* **Dropout:** 0.2
* **Projection:** 64 → 32 → 1
* **Input:** Multivariate time-series sensor data
* **Output:** RUL prediction

---

## 📂 Repository Structure

```text
AEGIS-AI-Predictive-Maintenance/
│
├── preprocessing.py
│   └── Data cleaning, normalization, sensor selection,
│       and time-series window generation
│
├── model.py
│   └── LSTM architecture and training (produces the checkpoint the dashboard uses)
│
├── train_piecewise_lstm.py
│   └── Optional piecewise-RUL variant (not used by the dashboard)
│
├── evaluate.py
│   └── Model evaluation and MAE/RMSE calculation
│
├── agents.py
│   └── Multi-agent reasoning system:
│       Diagnostic → Root-Cause → Verification → Maintenance
│
├── app.py
│   └── Interactive Streamlit dashboard and monitoring interface
│
├── requirements.txt
│   └── Python dependencies
│
└── README.md
    └── Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/navami-cmd/AEGIS-AI-Predictive-Maintenance.git
cd AEGIS-AI-Predictive-Maintenance
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Dashboard

```bash
streamlit run app.py
```

---

## 🧩 Core Technical Workflow

### Data Pipeline — `preprocessing.py`

Processes the NASA C-MAPSS dataset by cleaning the sensor data, selecting relevant operational and sensor features, generating RUL targets, scaling the inputs, and creating fixed-length time-series sequences for LSTM training.

### Deep Learning Engine — `model.py`

Trains the core LSTM-based temporal model to learn degradation patterns from historical sensor sequences and estimate the Remaining Useful Life of an engine.

### Evaluation Framework — `evaluate.py`

Evaluates the trained model on unseen test engines and calculates standard regression metrics including **MAE** and **RMSE** to measure prediction performance.

### Rule-based Reasoning — `agents.py`

Transforms model predictions and detected degradation signals into structured maintenance insights through a sequential multi-agent reasoning pipeline:

**Diagnostic → Root-Cause → Verification → Maintenance**

---

## 🔬 Technical Stack

**Machine Learning**

* PyTorch
* LSTM
* Time-Series Forecasting
* RUL Estimation
* Monte Carlo Dropout

**Explainable AI**

* Occlusion-Based XAI
* Sensor Contribution Analysis

**Agentic AI**

* Multi-Agent Reasoning
* Fault Diagnosis
* Root-Cause Analysis
* Evidence Verification
* Maintenance Planning

**Data & Visualization**

* NumPy
* Pandas
* Scikit-learn
* Plotly
* Streamlit



---

## Uncertainty (MC Dropout)

* One shared routine, `mc_utils.mc_dropout_predict()`, is used by the dashboard, the agents, the
  work-order export and the scripts: **100 passes**, dropout active in both the LSTM and the output
  head, seeded by (engine, cycle) so results are reproducible.
* The interval is `point prediction ± z · σ` (σ = std of the MC samples). **It is uncalibrated** unless
  `calibration.py` adopts a calibrated version from validation data. The dashboard and work order always
  say which one is shown and keep the uncalibrated interval for comparison.
* Intervals are uncertainty indicators, **not** guaranteed probabilities. Measure them with
  `coverage.py` (test coverage) and `calibration.py` (validation-only calibration study).

## Explainability (signed occlusion)

`effect = RUL(sensor's 30-cycle history replaced by its mean) − RUL(original)`, in cycles. A positive
effect means the sensor's recent pattern pulls the estimate toward shorter life. Shares are each
sensor's fraction of the total absolute effect. This is **model sensitivity**, measured one sensor at a
time; it is not evidence of a physical fault. Sensor meanings come from the NASA C-MAPSS documentation
(`sensor_info.py`).

## Agents

Diagnostic → Root-Cause → Verification → Maintenance are **deterministic rule-based agents** (not
trained models, not LLMs). The only fixed numbers are the 30/60-cycle policy bands. Decisions use the
prediction interval, contribution shares and effect size relative to the MC σ.

## Scaler note

The LSTM was trained on inputs scaled by `preprocessing.py` (scaler fitted on the 80 % training
engines). `python scaler_check.py` tells you whether `data/processed/scaler.pkl` still matches that
recipe. With the matching scaler the test metrics are MAE 16.66 / RMSE 23.60 (the values reported above);
with a scaler fitted on all engines they were 16.42 / 23.26. `AEGIS_FINAL.ps1` restores the matching scaler
only after verifying that it reproduces the saved training inputs exactly. The dashboard always reports
whatever `evaluate.py` last wrote.

## Checks

```powershell
python -m unittest -v test_agents test_core      # no GPU/data needed
python -m unittest -v test_integration           # real checkpoint + data
python mc_utils.py                               # sampler self-check
python scaler_check.py                           # scaler consistency (read-only)
python evaluate.py ; python coverage.py ; python calibration.py
streamlit run app.py
```

## Limitations

* FD001 only (one operating condition, one fault mode); results do not transfer to other settings.
* 17 input channels include two constant sensors (`sensor_5`, `sensor_16`) that carry no signal.
* Calibration is empirical on 20 validation engines; test coverage is measured on 100 engines and has
  a few percentage points of sampling error.
* The what-if simulator is an illustrative heuristic, not a validated simulation.
