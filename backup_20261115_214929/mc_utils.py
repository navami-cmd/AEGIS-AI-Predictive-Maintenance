"""
mc_utils.py - the ONE shared Monte Carlo (MC) Dropout routine for AEGIS.

Used by: coverage.py, uncertainty.py and app.py (dashboard, charts, agents and
work-order export), so every part of the project reports the same uncertainty
for the same engine and observation window.

What it does
------------
* Runs the trained LSTM `passes` times (default 100) with dropout switched ON
  in BOTH places the model was trained with:
    - the fully connected head  (nn.Dropout, p = 0.2)
    - the LSTM inter-layer dropout (nn.LSTM(dropout = 0.2))
  Note: nn.LSTM's internal dropout is not an nn.Dropout module, so it has to
  be enabled by putting the LSTM module itself in train mode.
* Is reproducible: the random seed is derived from (engine_id, cycle), so the
  same engine and window always give identical samples (same PyTorch version
  and device). The global random state is saved and restored, so nothing else
  is affected.
* Returns the point prediction (dropout OFF - identical to evaluate.py), the MC
  samples, the MC mean and the sample standard deviation (n - 1).

Intervals built from these samples are UNCALIBRATED unless coverage.py has
shown otherwise. They are centred on the point prediction:
    point +/- z * std_mc            (z from the Gaussian, e.g. 1.96 for 95%)
"""

import os
from statistics import NormalDist

import numpy as np
import torch
import torch.nn as nn

# ------------------------------------------------------------
# Settings (change here, and every file follows)
# ------------------------------------------------------------
MC_PASSES = 100
BASE_SEED = 2024
WINDOW_SIZE = 30
COVERAGE_LEVELS = (0.50, 0.80, 0.90, 0.95)
DEFAULT_LEVEL = 0.95
INTERVAL_LABEL = "MC Dropout interval (uncalibrated)"


# ------------------------------------------------------------
# Model definition - IDENTICAL to model.py / evaluate.py
# (so the existing checkpoint loads without any change)
# ------------------------------------------------------------
class LSTMModel(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :]).squeeze(1)


def load_lstm(model_path, n_inputs, device="cpu"):
    """Load the existing trained checkpoint (read-only) in eval mode."""
    model = LSTMModel(n_inputs)
    state = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def make_seed(engine_id, cycle, base_seed=BASE_SEED):
    """Deterministic seed for one (engine, cycle) observation window."""
    seq = np.random.SeedSequence([int(base_seed), int(engine_id), int(cycle)])
    return int(seq.generate_state(1)[0] % (2 ** 31 - 1))


def enable_mc_dropout(model):
    """eval() everywhere, then switch ON both dropout sources."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.LSTM)):
            module.train()


def z_value(level):
    """Two-sided Gaussian multiplier, e.g. 0.95 -> 1.96."""
    return NormalDist().inv_cdf(0.5 + float(level) / 2.0)


def prediction_interval(center, std, level=DEFAULT_LEVEL, scale=1.0, floor=0.0):
    """center +/- scale * z(level) * std, lower bound clipped at `floor`."""
    half = float(scale) * z_value(level) * float(std)
    return max(float(floor), float(center) - half), float(center) + half


# ------------------------------------------------------------
# The shared MC Dropout routine
# ------------------------------------------------------------
def mc_dropout_predict(
    model,
    window,
    engine_id,
    cycle,
    passes=MC_PASSES,
    base_seed=BASE_SEED,
):
    """
    window    : scaled array/tensor of shape (30, n_sensors)
    engine_id : engine number (used only for the seed)
    cycle     : last cycle of the window (used only for the seed)

    Returns a dict: point, mean, std, samples (np.ndarray), passes, seed.
    """
    x = torch.as_tensor(np.asarray(window, dtype=np.float32))
    if x.ndim != 2 or x.shape[0] != WINDOW_SIZE:
        raise ValueError(
            f"window must have shape ({WINDOW_SIZE}, n_sensors), got {tuple(x.shape)}"
        )
    device = next(model.parameters()).device
    x = x.unsqueeze(0).to(device)
    seed = make_seed(engine_id, cycle, base_seed)

    try:
        # 1) Point prediction: dropout OFF (same as evaluate.py)
        model.eval()
        with torch.no_grad():
            point = float(model(x).item())

        # 2) MC samples: dropout ON, all passes in one batch. Every batch
        #    row gets its own independent dropout masks.
        enable_mc_dropout(model)
        cuda_devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            with torch.no_grad():
                batch = x.expand(int(passes), -1, -1).contiguous()
                samples = model(batch).detach().cpu().numpy().astype(np.float64)
    finally:
        model.eval()  # always leave the shared model in normal eval mode

    return {
        "point": point,
        "mean": float(samples.mean()),
        "std": float(samples.std(ddof=1)),
        "samples": samples,
        "passes": int(passes),
        "seed": seed,
    }


# ------------------------------------------------------------
# Self-check:  python mc_utils.py
# ------------------------------------------------------------
def _self_check():
    import joblib
    import pandas as pd

    base = os.path.dirname(os.path.abspath(__file__))
    proc = os.path.join(base, "data", "processed")
    cmapss = os.path.join(base, "data", "CMAPSSData")

    with open(os.path.join(proc, "useful_sensors.txt")) as f:
        sensors = [line.strip() for line in f if line.strip()]
    scaler = joblib.load(os.path.join(proc, "scaler.pkl"))
    model = load_lstm(os.path.join(proc, "lstm_rul_model.pth"), len(sensors))

    cols = ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
    cols += [f"sensor_{i}" for i in range(1, 22)]
    test = pd.read_csv(os.path.join(cmapss, "test_FD001.txt"),
                       sep=r"\s+", header=None, names=cols)

    eng = 4
    g = test[test["unit"] == eng].sort_values("cycle")
    cyc = int(g["cycle"].max())
    win = scaler.transform(g.tail(WINDOW_SIZE)[sensors])

    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    enable_mc_dropout(model)
    check("dropout ON in FC head", model.fc[2].training)
    check("dropout ON in LSTM (inter-layer)", model.lstm.training)
    model.eval()
    check("eval() turns both OFF again",
          (not model.fc[2].training) and (not model.lstm.training))

    a = mc_dropout_predict(model, win, eng, cyc)
    b = mc_dropout_predict(model, win, eng, cyc)
    c = mc_dropout_predict(model, win, eng, cyc - 1)
    check("number of passes", len(a["samples"]) == MC_PASSES, f"({len(a['samples'])})")
    check("same engine+cycle -> identical samples",
          np.array_equal(a["samples"], b["samples"]))
    check("different cycle -> different samples",
          not np.array_equal(a["samples"], c["samples"]))
    check("samples actually vary (dropout is active)", a["std"] > 0,
          f"(std = {a['std']:.3f})")
    check("model back in eval mode after sampling",
          (not model.training) and (not model.lstm.training))

    csv_path = os.path.join(proc, "test_predictions.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        ref = float(df.loc[df["Engine"].astype(int) == eng, "Predicted_RUL"].iloc[0])
        check("point prediction matches test_predictions.csv",
              abs(a["point"] - ref) < 1e-3,
              f"({a['point']:.5f} vs {ref:.5f})")

    lo, hi = prediction_interval(a["point"], a["std"], 0.95)
    print(f"\nEngine {eng}, cycle {cyc}: point {a['point']:.2f} | MC mean {a['mean']:.2f} "
          f"| std {a['std']:.2f} | 95% interval [{lo:.1f}, {hi:.1f}] ({INTERVAL_LABEL})")
    print("ALL CHECKS PASSED" if all(results) else "SOME CHECKS FAILED")


if __name__ == "__main__":
    _self_check()
