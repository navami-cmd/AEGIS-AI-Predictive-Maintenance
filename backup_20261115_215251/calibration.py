"""
calibration.py - can the MC Dropout intervals be made more reliable?

Run from the project folder:   python calibration.py

METHOD (validation data only)
  1. Run the shared MC Dropout routine (100 seeded passes) on the validation
     windows (RUL <= 125, the standard cap; engine ids and cycles are rebuilt from
     the same split preprocessing.py used, and checked against y_val).
  2. For each nominal level (50/80/90/95 %) fit ONE width multiplier k so that
         point +/- k * z(level) * sigma_MC
     covers the validation truth at that level (split-conformal style: the
     finite-sample-adjusted quantile of |error| / sigma).
  3. Stability. (a) 200 random engine-level splits of the 20 validation engines:
     fit k on half of the engines, measure coverage on the OTHER half, and compare
     with the uncalibrated interval. Splits are by ENGINE because windows from one
     engine are strongly correlated. (b) refit with different MC random seeds.
  4. ADOPTION RULE (declared here, before any result is seen). A level is adopted
     only if ALL hold on validation data:
        C1  mean held-out calibrated coverage within +/- COVERAGE_TOL of nominal
        C2  mean |coverage - nominal| is smaller than for the uncalibrated interval
        C3  coefficient of variation of k across the engine splits <= SCALE_CV_MAX
        C4  relative range of k across MC seeds <= SEED_RANGE_MAX
  5. Only AFTER the decision is stored does the script evaluate on the FD001 test
     set, for REPORTING ONLY. Test data never influences k or the adoption flags.

The uncalibrated interval is always kept. Nothing here is a guarantee: an
adopted interval is empirically better calibrated on validation data, and test
coverage is reported next to it so you can judge transfer.

Outputs (data/processed/): calibration_summary.json, calibration_splits.csv
"""

import argparse
import json
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd

from coverage import fit_scale_factors, interval_metrics, collect
from mc_utils import BASE_SEED, COVERAGE_LEVELS, MC_PASSES, WINDOW_SIZE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
CMAPSS_DIR = os.path.join(BASE_DIR, "data", "CMAPSSData")

# ---- declared before any result is seen ----
VAL_MAX_RUL = 125.0
N_SPLITS = 200
SPLIT_SEED = 12345
COVERAGE_TOL = 0.03
SCALE_CV_MAX = 0.15
SEED_RANGE_MAX = 0.10
EXTRA_SEEDS = (11, 29)
SEED_SUBSAMPLE_STRIDE = 3


def lk(level):
    return f"{level:.2f}"


# ------------------------------------------------------------
# Validation data: rebuild engine ids / cycles for X_val
# ------------------------------------------------------------
def reconstruct_validation_index(train_df, y_val, window=WINDOW_SIZE, seed=42, train_fraction=0.8):
    """Same split as preprocessing.py; verified against y_val (raises if misaligned)."""
    units = train_df["unit"].unique()
    np.random.seed(seed)
    np.random.shuffle(units)
    val_units = sorted(units[int(len(units) * train_fraction):])

    ids, cycles, rul = [], [], []
    for u in val_units:
        last = int(train_df.loc[train_df["unit"] == u, "cycle"].max())
        for c in range(window, last + 1):
            ids.append(int(u))
            cycles.append(c)
            rul.append(last - c)
    if len(ids) != len(y_val) or not np.allclose(rul, y_val):
        raise ValueError(
            "Could not align X_val with the validation engines. Re-run preprocessing.py "
            "or check that train_FD001.txt is unchanged."
        )
    return np.asarray(ids), np.asarray(cycles)


# ------------------------------------------------------------
# Pure numerical study (testable without PyTorch)
# ------------------------------------------------------------
def engine_split_study(engine_ids, y, point, sigma, levels=COVERAGE_LEVELS,
                       n_splits=N_SPLITS, seed=SPLIT_SEED):
    engines = np.unique(engine_ids)
    if len(engines) < 4:
        raise ValueError("Need at least 4 validation engines for the split study.")
    rows = []
    for s in range(n_splits):
        rng = np.random.default_rng(seed + s)
        perm = rng.permutation(engines)
        half = len(perm) // 2
        fit_mask = np.isin(engine_ids, perm[:half])
        eval_mask = ~fit_mask
        scales = fit_scale_factors(y[fit_mask], point[fit_mask], sigma[fit_mask], levels)
        unc = interval_metrics(y[eval_mask], point[eval_mask], sigma[eval_mask], levels)
        cal = interval_metrics(y[eval_mask], point[eval_mask], sigma[eval_mask], levels, scales)
        for lvl, u, c in zip(levels, unc, cal):
            rows.append({
                "split": s, "level": lvl, "scale": scales[lvl],
                "cov_uncal": u["coverage"], "cov_cal": c["coverage"],
                "width_uncal": u["mean_width"], "width_cal": c["mean_width"],
                "n_fit": int(fit_mask.sum()), "n_eval": int(eval_mask.sum()),
            })
    return pd.DataFrame(rows)


def split_summary(split_df, levels=COVERAGE_LEVELS):
    out = {}
    for lvl in levels:
        d = split_df[split_df["level"] == lvl]
        sc = d["scale"].to_numpy()
        out[lk(lvl)] = {
            "scale_mean": float(sc.mean()), "scale_sd": float(sc.std(ddof=1)),
            "scale_cv": float(sc.std(ddof=1) / sc.mean()),
            "scale_p05": float(np.percentile(sc, 5)), "scale_p95": float(np.percentile(sc, 95)),
            "heldout_cov_uncal_mean": float(d["cov_uncal"].mean()),
            "heldout_cov_cal_mean": float(d["cov_cal"].mean()),
            "heldout_cov_cal_sd": float(d["cov_cal"].std(ddof=1)),
            "heldout_abs_err_uncal": float((d["cov_uncal"] - lvl).abs().mean()),
            "heldout_abs_err_cal": float((d["cov_cal"] - lvl).abs().mean()),
            "width_uncal_mean": float(d["width_uncal"].mean()),
            "width_cal_mean": float(d["width_cal"].mean()),
        }
    return out


def seed_scale_table(y, point, sigma_by_seed, levels=COVERAGE_LEVELS):
    """sigma_by_seed: {seed: sigma array}; fits k per seed on the same windows."""
    table = {}
    for lvl in levels:
        ks = {int(sd): fit_scale_factors(y, point, sig, [lvl])[lvl] for sd, sig in sigma_by_seed.items()}
        vals = np.array(list(ks.values()))
        table[lk(lvl)] = {"scales": ks,
                          "relative_range": float((vals.max() - vals.min()) / vals.mean())}
    return table


def decide_adoption(split_stats, seed_stats, levels=COVERAGE_LEVELS):
    detail, adopted = {}, {}
    for lvl in levels:
        k = lk(lvl)
        s, d = split_stats[k], seed_stats[k]
        c1 = abs(s["heldout_cov_cal_mean"] - lvl) <= COVERAGE_TOL
        c2 = s["heldout_abs_err_cal"] < s["heldout_abs_err_uncal"]
        c3 = s["scale_cv"] <= SCALE_CV_MAX
        c4 = d["relative_range"] <= SEED_RANGE_MAX
        detail[k] = {"C1_coverage_within_tol": bool(c1), "C2_better_than_uncalibrated": bool(c2),
                     "C3_scale_stable_across_splits": bool(c3), "C4_scale_stable_across_seeds": bool(c4)}
        adopted[k] = bool(c1 and c2 and c3 and c4)
    return adopted, detail


def run_calibration(predict_factory, val_set, test_set=None, seeds=None,
                    levels=COVERAGE_LEVELS, n_splits=N_SPLITS,
                    stride=SEED_SUBSAMPLE_STRIDE):
    """
    predict_factory(base_seed) -> predict_fn(window, engine_id, cycle) -> dict(point, mean, std)
    val_set  = (windows, engine_ids, cycles, y_true)   VALIDATION data
    test_set = same shape; used ONLY for the final report-only section
    """
    seeds = list(seeds or [BASE_SEED, *EXTRA_SEEDS])
    v_w, v_ids, v_cyc, v_y = val_set
    v_y = np.asarray(v_y, dtype=np.float64)
    v_ids = np.asarray(v_ids)

    # 1) validation MC statistics (primary seed)
    point, _, sigma = collect(predict_factory(seeds[0]), v_w, v_ids, v_cyc)

    # 2) engine-split stability
    split_df = engine_split_study(v_ids, v_y, point, sigma, levels, n_splits)
    split_stats = split_summary(split_df, levels)

    # 3) MC-seed stability on a subsample of windows (same windows for every seed)
    sub = np.arange(0, len(v_y), stride)
    sig_by_seed = {seeds[0]: sigma[sub]}
    for sd in seeds[1:]:
        _, _, sg = collect(predict_factory(sd), [v_w[i] for i in sub],
                           v_ids[sub], np.asarray(v_cyc)[sub])
        sig_by_seed[sd] = sg
    seed_stats = seed_scale_table(v_y[sub], point[sub], sig_by_seed, levels)

    # 4) decision - made from validation evidence only, BEFORE the test set is touched
    adopted, detail = decide_adoption(split_stats, seed_stats, levels)
    final_scales = fit_scale_factors(v_y, point, sigma, levels)        # all validation windows
    val_in_sample = interval_metrics(v_y, point, sigma, levels, final_scales)
    val_uncal = interval_metrics(v_y, point, sigma, levels)

    summary = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mc_passes": MC_PASSES, "base_seed": seeds[0], "mc_seeds_checked": seeds,
        "method": ("width multiplier k per level, fitted on validation windows only "
                   "(split-conformal style quantile of |error|/sigma); interval = point ± k·z·sigma"),
        "criteria": {"coverage_tol": COVERAGE_TOL, "scale_cv_max": SCALE_CV_MAX,
                     "seed_range_max": SEED_RANGE_MAX, "n_engine_splits": n_splits,
                     "val_max_rul": VAL_MAX_RUL},
        "validation": {"n_windows": int(len(v_y)), "n_engines": int(len(np.unique(v_ids))),
                       "uncalibrated_in_sample": {lk(r["nominal_level"]): r["coverage"] for r in val_uncal},
                       "calibrated_in_sample": {lk(r["nominal_level"]): r["coverage"] for r in val_in_sample}},
        "final_scale_factors": {lk(l): float(k) for l, k in final_scales.items()},
        "adopted": adopted, "adoption_detail": detail,
        "split_stability": split_stats, "seed_stability": seed_stats,
    }

    # 5) test set: REPORT ONLY (no fitting, no decisions)
    if test_set is not None:
        t_w, t_ids, t_cyc, t_y = test_set
        t_y = np.asarray(t_y, dtype=np.float64)
        t_point, _, t_sigma = collect(predict_factory(seeds[0]), t_w, t_ids, t_cyc)
        unc = interval_metrics(t_y, t_point, t_sigma, levels)
        cal = interval_metrics(t_y, t_point, t_sigma, levels, final_scales)
        summary["test_report_only"] = {
            "note": "Evaluation only. Test data was not used to fit k or to decide adoption.",
            "n_engines": int(len(t_y)),
            "levels": {lk(u["nominal_level"]): {
                "uncalibrated": {"coverage": u["coverage"], "mean_width": u["mean_width"]},
                "with_validation_scale": {"coverage": c["coverage"], "mean_width": c["mean_width"]},
            } for u, c in zip(unc, cal)},
        }
    return summary, split_df


# ------------------------------------------------------------
# Command line
# ------------------------------------------------------------
def main():
    import joblib
    from coverage import _load_test_set
    from mc_utils import load_lstm, mc_dropout_predict

    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=MC_PASSES)
    ap.add_argument("--splits", type=int, default=N_SPLITS)
    args = ap.parse_args()

    with open(os.path.join(DATA_DIR, "useful_sensors.txt")) as f:
        sensors = [line.strip() for line in f if line.strip()]
    scaler = joblib.load(os.path.join(DATA_DIR, "scaler.pkl"))
    model = load_lstm(os.path.join(DATA_DIR, "lstm_rul_model.pth"), len(sensors))

    X = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_val.npy")).astype(np.float64)
    cols = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [f"sensor_{i}" for i in range(1, 22)]
    train_df = pd.read_csv(os.path.join(CMAPSS_DIR, "train_FD001.txt"),
                           sep=r"\s+", header=None, names=cols)
    ids, cycles = reconstruct_validation_index(train_df, y)
    keep = y <= VAL_MAX_RUL
    val_set = (list(X[keep]), ids[keep], cycles[keep], y[keep])

    def factory(base_seed):
        return lambda w, e, c: mc_dropout_predict(model, w, e, c, passes=args.passes, base_seed=base_seed)

    print(f"Validation windows (RUL<={VAL_MAX_RUL:.0f}): {int(keep.sum())} from "
          f"{len(np.unique(ids[keep]))} engines | passes {args.passes}")
    summary, split_df = run_calibration(factory, val_set, _load_test_set(sensors, scaler),
                                        n_splits=args.splits)
    summary["mc_passes"] = args.passes

    with open(os.path.join(DATA_DIR, "calibration_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    split_df.to_csv(os.path.join(DATA_DIR, "calibration_splits.csv"), index=False)

    print("\nVALIDATION - engine-split study (fit on half the engines, test on the other half)")
    print(f"{'nominal':>8} {'k (mean±sd)':>14} {'held-out cov uncal':>19} {'held-out cov cal':>17} "
          f"{'width cal':>10} {'adopted':>8}")
    for lvl in COVERAGE_LEVELS:
        k = lk(lvl); s = summary["split_stability"][k]
        print(f"{lvl:>8.2f} {s['scale_mean']:>8.2f}±{s['scale_sd']:.2f} "
              f"{s['heldout_cov_uncal_mean']:>19.3f} {s['heldout_cov_cal_mean']:>17.3f} "
              f"{s['width_cal_mean']:>10.1f} {str(summary['adopted'][k]):>8}")
    print("\nSeed stability (relative range of k):",
          {k: round(v["relative_range"], 3) for k, v in summary["seed_stability"].items()})
    print("\nTEST SET (report only - not used for fitting or the adoption decision)")
    for k, v in summary["test_report_only"]["levels"].items():
        print(f"  {k}: uncalibrated {v['uncalibrated']['coverage']:.3f} -> "
              f"with validation scale {v['with_validation_scale']['coverage']:.3f} "
              f"(width {v['uncalibrated']['mean_width']:.1f} -> {v['with_validation_scale']['mean_width']:.1f})")
    print("\nAdopted levels:", [k for k, v in summary["adopted"].items() if v] or "none")
    print("Saved calibration_summary.json and calibration_splits.csv in", DATA_DIR)


if __name__ == "__main__":
    main()
