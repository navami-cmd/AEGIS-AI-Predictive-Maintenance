"""
Integration tests with the REAL PyTorch checkpoint and project data.
Run from the project folder:   python -m unittest -v test_integration
They are skipped automatically if PyTorch or the data files are missing.
"""
import json
import os
import unittest

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "processed")
CMAPSS = os.path.join(BASE, "data", "CMAPSSData")
NEEDED = [os.path.join(DATA, n) for n in
          ("lstm_rul_model.pth", "scaler.pkl", "useful_sensors.txt", "test_predictions.csv")] + \
         [os.path.join(CMAPSS, n) for n in ("test_FD001.txt", "RUL_FD001.txt")]

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
READY = HAVE_TORCH and all(os.path.exists(p) for p in NEEDED)


@unittest.skipUnless(READY, "needs PyTorch, the checkpoint and the project data files")
class RealPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import joblib
        import torch
        from mc_utils import MC_PASSES, load_lstm, mc_dropout_predict
        from data_checks import prepare_window
        from pipeline import analyze_window

        cls.torch, cls.MC_PASSES = torch, MC_PASSES
        cls.mc_dropout_predict, cls.prepare_window, cls.analyze_window = mc_dropout_predict, prepare_window, analyze_window
        with open(os.path.join(DATA, "useful_sensors.txt")) as f:
            cls.sensors = [line.strip() for line in f if line.strip()]
        cls.scaler = joblib.load(os.path.join(DATA, "scaler.pkl"))
        cls.model = load_lstm(os.path.join(DATA, "lstm_rul_model.pth"), len(cls.sensors))
        cols = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [f"sensor_{i}" for i in range(1, 22)]
        cls.test = pd.read_csv(os.path.join(CMAPSS, "test_FD001.txt"), sep=r"\s+", header=None, names=cols)
        cls.csv = pd.read_csv(os.path.join(DATA, "test_predictions.csv"))

    def point_fn(self, w):
        with self.torch.no_grad():
            return float(self.model(self.torch.tensor(np.asarray(w, dtype=np.float32)).unsqueeze(0)).item())

    def mc_fn(self, w, e, c):
        return self.mc_dropout_predict(self.model, w, e, c, passes=self.MC_PASSES)

    def analyse(self, eid):
        p = self.prepare_window(self.test, eid, self.sensors, self.scaler)
        return p, self.analyze_window(self.point_fn, self.mc_fn, p["scaled"], self.sensors, eid, p["current_cycle"])

    def test_every_engine_end_to_end(self):
        from work_order import build_work_order, to_json, validate_work_order
        for eid in sorted(self.test["unit"].unique()):
            p, a = self.analyse(int(eid))
            ref = float(self.csv.loc[self.csv["Engine"] == eid, "Predicted_RUL"].iloc[0])
            self.assertAlmostEqual(a["point"], ref, places=3, msg=f"engine {eid} vs test_predictions.csv")
            self.assertGreater(a["sigma"], 0)
            self.assertEqual(len(a["mc"]["samples"]), self.MC_PASSES)
            wo = build_work_order(int(eid), p["current_cycle"], a, self.MC_PASSES, timestamp="T")
            self.assertEqual(validate_work_order(wo), [])
            self.assertEqual(json.loads(to_json(wo)), wo)
            self.assertEqual(wo["predicted_rul"], a["point"])
            self.assertEqual(wo["uncertainty"], a["sigma"])
            self.assertEqual(wo["health_status"], a["agents"]["diagnostic"]["status"].upper())

    def test_reproducible_and_model_left_in_eval_mode(self):
        _, a = self.analyse(4)
        _, b = self.analyse(4)
        self.assertTrue(np.array_equal(a["mc"]["samples"], b["mc"]["samples"]))
        self.assertEqual(a["sigma"], b["sigma"])
        self.assertFalse(self.model.training or self.model.lstm.training or self.model.fc[2].training)

    def test_both_dropouts_are_active_during_sampling(self):
        from mc_utils import enable_mc_dropout
        enable_mc_dropout(self.model)
        self.assertTrue(self.model.lstm.training and self.model.fc[2].training)
        self.model.eval()

    def test_reported_metrics_match_csv(self):
        err = self.csv["Predicted_RUL"] - self.csv["Actual_RUL"]
        true = np.loadtxt(os.path.join(CMAPSS, "RUL_FD001.txt"))
        np.testing.assert_allclose(self.csv["Actual_RUL"].to_numpy(), true)
        print(f"\n  test MAE {err.abs().mean():.2f}  RMSE {np.sqrt((err ** 2).mean()):.2f}")


if __name__ == "__main__":
    unittest.main()
