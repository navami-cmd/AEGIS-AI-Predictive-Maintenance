# AEGIS AI: Explainable Agentic Predictive Maintenance

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)

**AEGIS AI** ("Intelligent Shield") is an advanced explainable predictive maintenance framework designed for **Remaining Useful Life (RUL) estimation and intelligent equipment health monitoring**. Built using NASA's **C-MAPSS turbofan engine dataset**, the framework bridges the gap between deep learning predictions and actionable maintenance decisions by integrating **Explainable AI (XAI)**, **Uncertainty Quantification using Monte Carlo Dropout**, and an **Autonomous Multi-Agent Reasoning System**.

---

##  Key Features

1. **Deep Time-Series Forecasting**
   Implements sliding-window preprocessing over multivariate operational sensor data to capture temporal degradation patterns and predict future equipment health.

2. **Uncertainty-Aware Predictions**
   Uses **Monte Carlo Dropout** to estimate predictive uncertainty and provide confidence-aware RUL predictions, helping identify potentially high-risk predictions.

3. **Transparent Occlusion-Based XAI**
   Identifies the sensor inputs and temporal regions that have the greatest influence on the predicted RUL, improving the interpretability of the deep learning model.

4. **Autonomous Multi-Agent Diagnostic Chain**
   Implements a sequential reasoning architecture:

   **Diagnostic → Root-Cause → Verification → Maintenance**

   The agents analyze predicted degradation and anomaly signals to identify potential causes, verify supporting evidence, and generate prioritized maintenance recommendations.

5. **Interactive Digital Twin Control Center**
   Provides a Streamlit-based dashboard for monitoring sensor telemetry, visualizing degradation trends, exploring model predictions, simulating equipment stress, and interpreting model outputs.

---

## 📊 Model Performance

### NASA C-MAPSS FD001

Evaluated on the official unseen test split of the **C-MAPSS FD001 benchmark**.

| Metric        |           Result |
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
├── train_piecewise_lstm.py
│   └── LSTM architecture and piecewise RUL training pipeline
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

### Deep Learning Engine — `train_piecewise_lstm.py`

Trains the core LSTM-based temporal model to learn degradation patterns from historical sensor sequences and estimate the Remaining Useful Life of an engine.

### Evaluation Framework — `evaluate.py`

Evaluates the trained model on unseen test engines and calculates standard regression metrics including **MAE** and **RMSE** to measure prediction performance.

### Autonomous Reasoning — `agents.py`

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

