# AEGIS AI: Explainable Agentic Predictive Maintenance

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%2B-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)

**AEGIS AI** ("Intelligent Shield") is an advanced, production-grade predictive maintenance framework designed for turbofan engine Remaining Useful Life (RUL) estimation. Built using NASA’s C-MAPSS dataset, the project bridges the gap between raw deep learning forecasts and actionable industrial maintenance by integrating **Explainable AI (XAI)**, **Uncertainty Quantification (MC Dropout)**, and an **Autonomous Multi-Agent Reasoning Network**.

---

##  Key Features

1. **Deep Time-Series Forecasting:** Implements optimized sliding-window preprocessing over 17 operational sensor streams to capture complex degradation patterns.
2. **Uncertainty-Aware Predictions:** Utilizes Monte Carlo Dropout to calculate epistemic uncertainty bounds ($\pm$ cycles), safeguarding operators against high-risk predictions.
3. **Transparent Occlusion-Based XAI:** Maps sensor contributions directly to final RUL outputs, solving the classic "black box" problem of deep neural networks in industrial settings.
4. **Autonomous Multi-Agent Diagnostic Chain:** Features a sequential multi-agent architecture (`Diagnostic` $\rightarrow$ `Root-Cause` $\rightarrow$ `Verification` $\rightarrow$ `Maintenance`) that automatically analyzes anomaly signals and generates prioritized work orders.
5. **Interactive Digital Twin Control Center:** A fully responsive Streamlit dashboard providing real-time telemetry, stress simulation, and model evaluation metrics.

---

## 📊 Model Performance (NASA C-MAPSS FD001)

Evaluated on the official unseen test split of the C-MAPSS FD001 benchmark:
* **Test Mean Absolute Error (MAE):** `16.66 cycles`
* **Test Root Mean Squared Error (RMSE):** `23.60 cycles`
* **Architecture:** 2-Layer LSTM (64 hidden units, 0.2 dropout rate) with fully connected projection layers ($64 \rightarrow 32 \rightarrow 1$).

---

## 📂 Repository Structure

```text
AEGIS-AI-Predictive-Maintenance/
│
├── preprocessing.py          # Data cleaning, normalization, and 30-cycle time-series slicing
├── train_piecewise_lstm.py   # Core LSTM model architecture and piecewise RUL training loop
├── evaluate.py               # Evaluation script for computing MAE and RMSE metrics against test data
├── agents.py                 # Multi-agent autonomous reasoning network (Diagnostic to Maintenance)
├── app.py                    # Interactive Streamlit control center and digital twin UI
├── requirements.txt          # Project python package dependencies
└── README.md                 # Project documentation