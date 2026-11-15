"""
scaler_check.py - is data/processed/scaler.pkl the scaler the LSTM was trained with?

    python scaler_check.py                       read-only check
    python scaler_check.py --restore             back up scaler.pkl, then replace it
    python scaler_check.py --restore-if-verified restore ONLY if the evidence is conclusive

preprocessing.py fits its MinMaxScaler on the 80% training engines (seed 42) and the LSTM is
trained on those scaled values. The proof used here: the recipe scaler must reproduce the saved
X_val windows (taken from several validation engines) to within 1e-9 while scaler.pkl does not.
Exit code 0 = scaler OK or restored, 3 = mismatch left unchanged, 1 = error.
After a restore, re-run evaluate.py (test metrics change slightly: that is expected and reported).
"""
import argparse
import os
import shutil
import sys

import joblib
import numpy as np
import pandas as pd

from data_checks import compare_scalers, training_recipe_scaler

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "processed")
SCALER = os.path.join(DATA, "scaler.pkl")
BACKUP = os.path.join(DATA, "scaler_before_restore.pkl")
W = 30


def window_evidence(train, sensors, X, disk, recipe):
    """Max abs difference vs saved X_val over windows from several validation engines."""
    units = train["unit"].unique()
    np.random.seed(42)
    np.random.shuffle(units)
    val_units = sorted(units[int(len(units) * 0.8):])
    d_disk = d_recipe = 0.0
    offset, checked = 0, 0
    for u in val_units:
        g = train[train["unit"] == u].sort_values("cycle")
        n_win = len(g) - W + 1
        if checked < 6:
            for k in sorted({0, n_win // 2, n_win - 1}):
                w = g.iloc[k:k + W][sensors]
                d_disk = max(d_disk, float(np.abs(disk.transform(w) - X[offset + k]).max()))
                d_recipe = max(d_recipe, float(np.abs(recipe.transform(w) - X[offset + k]).max()))
            checked += 1
        offset += n_win
    if offset != len(X):
        raise ValueError(f"X_val has {len(X)} windows but the split implies {offset}")
    return d_disk, d_recipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--restore-if-verified", action="store_true")
    args = ap.parse_args()

    with open(os.path.join(DATA, "useful_sensors.txt")) as f:
        sensors = [line.strip() for line in f if line.strip()]
    cols = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [f"sensor_{i}" for i in range(1, 22)]
    train = pd.read_csv(os.path.join(BASE, "data", "CMAPSSData", "train_FD001.txt"),
                        sep=r"\s+", header=None, names=cols)
    X = np.load(os.path.join(DATA, "X_val.npy"))

    disk = joblib.load(SCALER)
    recipe = training_recipe_scaler(train, sensors)
    same, bad = compare_scalers(disk, recipe)
    d_disk, d_recipe = window_evidence(train, sensors, X, disk, recipe)

    print(f"scaler.pkl identical to the preprocessing recipe: {same}")
    print(f"saved X_val vs scaler.pkl : max diff {d_disk:.4f}")
    print(f"saved X_val vs recipe     : max diff {d_recipe:.2e}")
    if same:
        print("SCALER_STATUS: OK")
        return 0

    print("sensors whose min/max differ:", [sensors[i] for i in bad])
    verified = d_recipe <= 1e-9 and d_disk > 1e-9
    print("Recipe scaler reproduces the training inputs exactly: ", verified)

    if args.restore or (args.restore_if_verified and verified):
        shutil.copy2(SCALER, BACKUP)
        joblib.dump(recipe, SCALER)
        print(f"Old scaler backed up to {BACKUP}; training-consistent scaler written.")
        print("SCALER_STATUS: RESTORED")
        return 0

    print("Nothing was changed.")
    print("SCALER_STATUS: MISMATCH_UNVERIFIED" if not verified else "SCALER_STATUS: MISMATCH_LEFT")
    return 3


if __name__ == "__main__":
    sys.exit(main())
