"""
uncertainty.py - standalone MC Dropout demo.

Uses the SAME shared routine as the dashboard, the agents and the work-order export
(mc_utils.mc_dropout_predict: 100 seeded passes, dropout active in the LSTM and in the
output head), so a number printed here equals the number shown in the dashboard for
the same window. Run:  python uncertainty.py
"""

import os

import numpy as np

from mc_utils import (
    INTERVAL_LABEL,
    MC_PASSES,
    load_lstm,
    mc_dropout_predict,
    prediction_interval,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_PATH = os.path.join(DATA_DIR, "lstm_rul_model.pth")

_model = None


def _get_model():
    global _model
    if _model is None:
        with open(os.path.join(DATA_DIR, "useful_sensors.txt")) as f:
            n_sensors = len([line for line in f if line.strip()])
        _model = load_lstm(MODEL_PATH, n_sensors)
    return _model


def predict_with_uncertainty(sequence, engine_id=0, cycle=0, passes=MC_PASSES):
    """
    sequence: scaled window, shape (30, n_sensors).
    Returns (point_rul, lower, upper, sigma, samples). The interval is the
    UNCALIBRATED 95% interval (point +/- 1.96 * sigma); see calibration.py.
    engine_id / cycle only seed the sampling, so the same pair is reproducible.
    """
    result = mc_dropout_predict(_get_model(), sequence, engine_id, cycle, passes=passes)
    lower, upper = prediction_interval(result["point"], result["std"], 0.95)
    return result["point"], lower, upper, result["std"], result["samples"]


if __name__ == "__main__":
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    point, lower, upper, sigma, samples = predict_with_uncertainty(X_val[0])

    print("\n========================================")
    print("UNCERTAINTY ESTIMATION (shared MC Dropout)")
    print("========================================")
    print(f"Predicted RUL : {point:.2f} cycles (dropout off)")
    print(f"Uncertainty   : ±{sigma:.2f} cycles (std of {len(samples)} MC samples)")
    print(f"{INTERVAL_LABEL}")
    print(f"Lower / Upper : {lower:.2f} / {upper:.2f} cycles")
    print("========================================")
