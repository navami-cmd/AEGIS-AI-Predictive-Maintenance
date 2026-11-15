"""
coverage.py - empirical coverage and width of the MC Dropout prediction intervals.

Run from the project folder:   python coverage.py

What is measured (FD001 test set, 100 engines, last 30-cycle window each):
  * For nominal levels 50 / 80 / 90 / 95 %, how often the true RUL falls
    inside  point +/- z * std_mc , and the mean interval width.
  * The same split by true-RUL range (<=30, 30-60, >60 cycles).
  * Correlation between the MC std and the absolute error.

Calibration rule (important):
  The TEST set is never used to fit anything. As an optional diagnostic the
  script fits one width multiplier per level on VALIDATION windows only and
  then reports how that multiplier performs on the untouched test set. These
  factors are saved for you to review; the dashboard keeps calling the
  intervals "uncalibrated" until you decide otherwise.

Outputs (data/processed/):
  coverage_results.csv   - one row per split / subset / nominal level
  mc_test_summary.csv    - one row per test engine
  coverage_summary.json  - compact numbers used by the dashboard caption

The numerical helpers below need only numpy/pandas; PyTorch is used only in main().
"""

import argparse
import json
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd

from mc_utils import (
    BASE_SEED,
    COVERAGE_LEVELS,
    DEFAULT_LEVEL,
    INTERVAL_LABEL,
    MC_PASSES,
    WINDOW_SIZE,
    prediction_interval,
    z_value,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
CMAPSS_DIR = os.path.join(BASE_DIR, "data", "CMAPSSData")

VAL_ID_OFFSET = 100000          # seed ids for validation windows (engine ids unknown)
SUBSETS = [
    ("all", None, None),
    ("RUL<=30", None, 30),
    ("30<RUL<=60", 30, 60),
    ("RUL>60", 60, None),
]


# ------------------------------------------------------------
# Pure numerical helpers (testable without PyTorch)
# ------------------------------------------------------------
def collect(predict_fn, windows, ids, cycles):
    """Run predict_fn(window, engine_id, cycle) for every window."""
    rows = []
    for w, i, c in zip(windows, ids, cycles):
        r = predict_fn(w, int(i), int(c))
        rows.append((r["point"], r["mean"], r["std"]))
    a = np.asarray(rows, dtype=np.float64)
    return a[:, 0], a[:, 1], a[:, 2]          # point, mc_mean, mc_std


def interval_metrics(y, point, std, levels=COVERAGE_LEVELS, scale_by_level=None):
    """Coverage / width at each nominal level. scale_by_level multiplies the width."""
    y, point, std = map(np.asarray, (y, point, std))
    rows = []
    for level in levels:
        scale = 1.0 if scale_by_level is None else float(scale_by_level[level])
        z = z_value(level) * scale
        lo = np.maximum(0.0, point - z * std)
        hi = point + z * std
        rows.append({
            "nominal_level": level,
            "n": int(len(y)),
            "coverage": float(np.mean((y >= lo) & (y <= hi))) if len(y) else float("nan"),
            "mean_width": float(np.mean(hi - lo)) if len(y) else float("nan"),
            "below_lower": float(np.mean(y < lo)) if len(y) else float("nan"),
            "above_upper": float(np.mean(y > hi)) if len(y) else float("nan"),
            "width_scale_applied": scale,
        })
    return rows


def metrics_by_subset(y, point, std, levels=COVERAGE_LEVELS, scale_by_level=None):
    y = np.asarray(y)
    out = []
    for name, lo_edge, hi_edge in SUBSETS:
        mask = np.ones(len(y), dtype=bool)
        if lo_edge is not None:
            mask &= y > lo_edge
        if hi_edge is not None:
            mask &= y <= hi_edge
        for row in interval_metrics(y[mask], np.asarray(point)[mask],
                                    np.asarray(std)[mask], levels, scale_by_level):
            row["subset"] = name
            out.append(row)
    return out


def fit_scale_factors(y, point, std, levels=COVERAGE_LEVELS):
    """
    Width multipliers fitted on the data given (call with VALIDATION data only).
    Split-conformal style: the finite-sample-adjusted quantile of |error| / std.
    Returns {level: scale}, where scale multiplies the Gaussian z.
    """
    y, point, std = map(np.asarray, (y, point, std))
    ratios = np.sort(np.abs(y - point) / np.maximum(std, 1e-6))
    n = len(ratios)
    scales = {}
    for level in levels:
        k = min(n, math.ceil((n + 1) * level))      # 1-based rank
        scales[level] = float(ratios[k - 1] / z_value(level))
    return scales


def std_error_correlations(y, point, std):
    err = np.abs(np.asarray(point) - np.asarray(y))
    s = pd.Series(np.asarray(std))
    return {
        "pearson": float(s.corr(pd.Series(err))),
        "spearman": float(s.corr(pd.Series(err), method="spearman")),
    }


def run_coverage(predict_fn, test_set, val_set=None, levels=COVERAGE_LEVELS):
    """
    test_set / val_set = (windows, ids, cycles, y_true).
    Returns (results_df, test_per_engine_df, summary_dict).
    """
    t_w, t_ids, t_cyc, t_y = test_set
    t_y = np.asarray(t_y, dtype=np.float64)
    t_point, t_mean, t_std = collect(predict_fn, t_w, t_ids, t_cyc)

    results = []
    for row in metrics_by_subset(t_y, t_point, t_std, levels):
        results.append({"split": "test", **row})

    corr = std_error_correlations(t_y, t_point, t_std)
    lo95, hi95 = zip(*[prediction_interval(p, s, DEFAULT_LEVEL)
                       for p, s in zip(t_point, t_std)])
    per_engine = pd.DataFrame({
        "Engine": np.asarray(t_ids, dtype=int),
        "Actual_RUL": t_y,
        "Predicted_RUL": t_point,
        "MC_Mean": t_mean,
        "MC_Std": t_std,
        "Lower_95": lo95,
        "Upper_95": hi95,
    })
    per_engine["Covered_95"] = (per_engine["Actual_RUL"] >= per_engine["Lower_95"]) & \
                               (per_engine["Actual_RUL"] <= per_engine["Upper_95"])

    test_all = {f"{r['nominal_level']:.2f}": r for r in interval_metrics(t_y, t_point, t_std, levels)}
    summary = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label": INTERVAL_LABEL,
        "calibration_status": "uncalibrated",
        "interval_definition": "point prediction +/- z * MC-Dropout std (Gaussian z)",
        "mc_passes": MC_PASSES,
        "base_seed": BASE_SEED,
        "test": {
            "n_engines": int(len(t_y)),
            "levels": {k: {"coverage": v["coverage"], "mean_width": v["mean_width"]}
                       for k, v in test_all.items()},
            "std_vs_abs_error_corr": corr,
        },
    }

    if val_set is not None:
        v_w, v_ids, v_cyc, v_y = val_set
        v_y = np.asarray(v_y, dtype=np.float64)
        v_point, _, v_std = collect(predict_fn, v_w, v_ids, v_cyc)
        for row in metrics_by_subset(v_y, v_point, v_std, levels):
            results.append({"split": "validation", **row})
        scales = fit_scale_factors(v_y, v_point, v_std, levels)       # VALIDATION ONLY
        held_out = interval_metrics(t_y, t_point, t_std, levels, scales)  # applied to test
        for row in held_out:
            results.append({"split": "test_with_validation_scale", "subset": "all", **row})
        summary["validation_calibration"] = {
            "n_validation_windows": int(len(v_y)),
            "fitted_on": "validation windows only (never test data)",
            "width_scale_by_level": {f"{k:.2f}": v for k, v in scales.items()},
            "test_coverage_with_scale": {
                f"{r['nominal_level']:.2f}": {"coverage": r["coverage"],
                                              "mean_width": r["mean_width"]}
                for r in held_out},
            "note": ("Diagnostic only. Validation windows come from all life stages of "
                     "20 engines, while the test set uses one final window per engine, "
                     "so the factors may not transfer. Not applied in the dashboard."),
        }

    cols = ["split", "subset", "nominal_level", "n", "coverage", "mean_width",
            "below_lower", "above_upper", "width_scale_applied"]
    return pd.DataFrame(results)[cols], per_engine, summary


# ------------------------------------------------------------
# Command-line entry point (needs PyTorch + the project data)
# ------------------------------------------------------------
def _load_test_set(sensors, scaler):
    cols = ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
    cols += [f"sensor_{i}" for i in range(1, 22)]
    test = pd.read_csv(os.path.join(CMAPSS_DIR, "test_FD001.txt"),
                       sep=r"\s+", header=None, names=cols)
    rul = np.loadtxt(os.path.join(CMAPSS_DIR, "RUL_FD001.txt"))
    windows, ids, cycles = [], [], []
    for unit in sorted(test["unit"].unique()):
        g = test[test["unit"] == unit].sort_values("cycle")
        windows.append(scaler.transform(g.tail(WINDOW_SIZE)[sensors]))
        ids.append(int(unit))
        cycles.append(int(g["cycle"].max()))
    return windows, ids, cycles, rul


def _load_val_set(stride, max_rul):
    X = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_val.npy")).astype(np.float64)
    idx = np.arange(0, len(X), stride)
    idx = idx[y[idx] <= max_rul]
    return list(X[idx]), list(VAL_ID_OFFSET + idx), [0] * len(idx), y[idx]


def main():
    import joblib
    from mc_utils import load_lstm, mc_dropout_predict

    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=MC_PASSES)
    ap.add_argument("--val-stride", type=int, default=10,
                    help="use every k-th validation window; 0 = skip validation diagnostic")
    ap.add_argument("--val-max-rul", type=float, default=125.0,
                    help="only validation windows with RUL <= this (125 = standard cap)")
    args = ap.parse_args()

    with open(os.path.join(DATA_DIR, "useful_sensors.txt")) as f:
        sensors = [line.strip() for line in f if line.strip()]
    scaler = joblib.load(os.path.join(DATA_DIR, "scaler.pkl"))
    model = load_lstm(os.path.join(DATA_DIR, "lstm_rul_model.pth"), len(sensors))

    def predict_fn(window, engine_id, cycle):
        return mc_dropout_predict(model, window, engine_id, cycle, passes=args.passes)

    test_set = _load_test_set(sensors, scaler)
    val_set = _load_val_set(args.val_stride, args.val_max_rul) if args.val_stride > 0 else None
    print(f"Test engines: {len(test_set[1])} | validation windows: "
          f"{0 if val_set is None else len(val_set[1])} | passes: {args.passes}")

    results, per_engine, summary = run_coverage(predict_fn, test_set, val_set)
    summary["mc_passes"] = args.passes

    results.to_csv(os.path.join(DATA_DIR, "coverage_results.csv"), index=False)
    per_engine.to_csv(os.path.join(DATA_DIR, "mc_test_summary.csv"), index=False)
    with open(os.path.join(DATA_DIR, "coverage_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    show = results[(results["split"] == "test") & (results["subset"] == "all")]
    print("\nTEST SET - nominal vs measured coverage (intervals are UNCALIBRATED)")
    print(show[["nominal_level", "coverage", "mean_width"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nstd vs |error| correlation:", summary["test"]["std_vs_abs_error_corr"])
    if "validation_calibration" in summary:
        print("\nValidation-fitted width scale (diagnostic, NOT applied):")
        for k, v in summary["validation_calibration"]["width_scale_by_level"].items():
            held = summary["validation_calibration"]["test_coverage_with_scale"][k]["coverage"]
            print(f"  nominal {k}: scale x{v:.2f} -> held-out test coverage {held:.3f}")
    print("\nSaved: coverage_results.csv, mc_test_summary.csv, coverage_summary.json "
          f"in {DATA_DIR}")


if __name__ == "__main__":
    main()
