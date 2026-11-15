"""
xai.py - standalone signed occlusion demo.

Uses the SAME implementation as the dashboard (xai_utils.occlusion_effects):
    effect = RUL(sensor history replaced by its 30-cycle mean) - RUL(original)
Positive effect: the sensor's recent pattern pulls the estimate toward shorter life.
This is model sensitivity, not proof of a physical fault. Run:  python xai.py
"""

import os

import numpy as np
import torch

from mc_utils import load_lstm
from xai_utils import occlusion_effects

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

with open(os.path.join(DATA_DIR, "useful_sensors.txt")) as f:
    sensors = [line.strip() for line in f if line.strip()]

model = load_lstm(os.path.join(DATA_DIR, "lstm_rul_model.pth"), len(sensors))


def point_predict(window):
    with torch.no_grad():
        x = torch.tensor(np.asarray(window, dtype=np.float32)).unsqueeze(0)
        return float(model(x).item())


sample = np.load(os.path.join(DATA_DIR, "X_val.npy"))[0]
result = occlusion_effects(point_predict, sample, sensors)

print("\n==============================")
print("XAI SENSOR CONTRIBUTION (signed occlusion)")
print("==============================")
print(f"\nOriginal predicted RUL: {result['baseline']:.2f} cycles")
print("\nTop contributing sensors:")
for e in result["effects"][:10]:
    print(f"#{e['rank']:<2} {e['sensor']:10s} effect {e['delta']:+8.3f} cycles | "
          f"share {e['share']:6.1%} | {e['tier']}")

with open(os.path.join(DATA_DIR, "xai_results.txt"), "w", encoding="utf-8") as f:
    f.write("XAI SENSOR CONTRIBUTION (signed occlusion)\n")
    f.write("effect = RUL(sensor flattened to its window mean) - RUL(original)\n")
    f.write("positive = the sensor's pattern pulls the estimate toward shorter life\n")
    f.write("Model sensitivity, not proof of a physical fault.\n\n")
    f.write(f"Original Predicted RUL: {result['baseline']:.2f} cycles\n\n")
    for e in result["effects"]:
        f.write(f"{e['sensor']} | effect {e['delta']:+.4f} | share {e['share']:.4f} | {e['tier']}\n")

print("\nXAI results saved to:", os.path.join(DATA_DIR, "xai_results.txt"))
