"""
Torch-free unit tests: sensor info, signed occlusion, data checks, intervals,
work order, calibration statistics.     Run:  python -m unittest -v test_core
"""
import json
import math
import unittest

import numpy as np
import pandas as pd

from test_support import ensure_torch_importable
ensure_torch_importable()

import calibration as CAL                      # noqa: E402
import data_checks as DC                       # noqa: E402
import pipeline as PL                          # noqa: E402
import sensor_info as SI                       # noqa: E402
import work_order as WO                        # noqa: E402
from xai_utils import occlusion_effects        # noqa: E402

SENSORS = [f"sensor_{i}" for i in (2, 3, 4, 7, 11)]


def linear_model(coefs):
    """f(window) = sum_j coefs[j] * last-cycle value of sensor j."""
    c = np.asarray(coefs, dtype=float)
    return lambda w: float(np.asarray(w)[-1] @ c)


def rising_window():
    w = np.tile(np.linspace(0.0, 1.0, 30)[:, None], (1, len(SENSORS)))
    return w


class SensorInfo(unittest.TestCase):
    def test_documented_and_unknown(self):
        self.assertEqual(SI.sensor_symbol("sensor_11"), "Ps30")
        self.assertEqual(SI.sensor_description("sensor_7"), "Total pressure at HPC outlet")
        self.assertIsNone(SI.sensor_description("sensor_99"))
        self.assertIsNone(SI.sensor_symbol("humidity"))
        self.assertEqual(SI.sensor_label("sensor_99"), "sensor_99")
        self.assertEqual(SI.sensor_label("sensor_11"), "sensor_11 (Ps30)")


class SignedOcclusion(unittest.TestCase):
    def test_exact_formula_and_direction(self):
        w = rising_window()
        coefs = [10.0, -4.0, 0.0, 2.0, 0.5]
        r = occlusion_effects(linear_model(coefs), w, SENSORS)
        mean, last = w[:, 0].mean(), w[-1, 0]
        by = {e["sensor"]: e for e in r["effects"]}
        # delta = f(flattened) - f(original) = coef * (mean - last)
        self.assertAlmostEqual(by["sensor_2"]["delta"], 10.0 * (mean - last), places=10)
        self.assertEqual(by["sensor_2"]["direction"], "lowers")     # delta < 0
        self.assertEqual(by["sensor_3"]["direction"], "raises")     # delta > 0 (coef < 0)
        self.assertEqual(by["sensor_4"]["direction"], "none")
        self.assertAlmostEqual(by["sensor_2"]["importance"], abs(by["sensor_2"]["delta"]))

    def test_shares_ranks_tiers_and_order(self):
        names = [f"sensor_{i}" for i in range(2, 14)]
        w = np.tile(np.linspace(0, 1, 30)[:, None], (1, len(names)))
        r = occlusion_effects(linear_model(np.arange(len(names), 0, -1)), w, names)
        eff = r["effects"]
        self.assertAlmostEqual(sum(e["share"] for e in eff), 1.0, places=10)
        self.assertEqual([e["rank"] for e in eff], list(range(1, len(names) + 1)))
        self.assertTrue(all(eff[i]["importance"] >= eff[i + 1]["importance"] for i in range(len(eff) - 1)))
        self.assertEqual([e["tier"] for e in eff][:3], ["Top 3"] * 3)
        self.assertEqual(eff[3]["tier"], "Top 10")
        self.assertEqual(eff[10]["tier"], "Lower")

    def test_all_zero_effects_are_safe(self):
        r = occlusion_effects(lambda w: 42.0, rising_window(), SENSORS)
        self.assertTrue(r["all_zero"])
        self.assertEqual(r["total_abs_effect"], 0.0)
        for e in r["effects"]:
            self.assertEqual(e["share"], 0.0)
            self.assertEqual(e["direction"], "none")
            self.assertFalse(math.isnan(e["delta"]))
        json.dumps(r, allow_nan=False)

    def test_deterministic_tie_order(self):
        w = np.zeros((30, len(SENSORS)))
        w[:, :] = np.linspace(0, 1, 30)[:, None]
        a = occlusion_effects(linear_model([1, 1, 1, 1, 1]), w, SENSORS)
        b = occlusion_effects(linear_model([1, 1, 1, 1, 1]), w, SENSORS)
        self.assertEqual([e["sensor"] for e in a["effects"]], [e["sensor"] for e in b["effects"]])

    def test_invalid_inputs(self):
        f = linear_model([1] * 5)
        bad = rising_window(); bad[3, 2] = np.nan
        with self.assertRaises(ValueError):
            occlusion_effects(f, bad, SENSORS)
        with self.assertRaises(ValueError):
            occlusion_effects(f, rising_window()[:, :3], SENSORS)
        with self.assertRaises(ValueError):
            occlusion_effects(lambda w: float("nan"), rising_window(), SENSORS)
        with self.assertRaises(ValueError):
            occlusion_effects(f, rising_window(), SENSORS, baseline=999.0)


class DataChecks(unittest.TestCase):
    def frame(self, n=40):
        rows = []
        for unit in (1, 2):
            for c in range(1, n + 1):
                rows.append({"unit": unit, "cycle": c, **{s: float(c + k) for k, s in enumerate(SENSORS)}})
        return pd.DataFrame(rows)

    def scaler(self, df):
        from sklearn.preprocessing import MinMaxScaler
        return MinMaxScaler().fit(df[SENSORS])

    def test_engine_ids(self):
        ids = [1, 2, 3]
        self.assertEqual(DC.validate_engine_id("2", ids), 2)
        self.assertEqual(DC.validate_engine_id(3.0, ids), 3)
        for bad in (99, "abc", None, 2.5, float("nan")):
            with self.assertRaises(DC.DataError):
                DC.validate_engine_id(bad, ids)

    def test_normal_window(self):
        df = self.frame(); r = DC.prepare_window(df, 1, SENSORS, self.scaler(df))
        self.assertEqual(r["scaled"].shape, (30, 5))
        self.assertEqual(r["current_cycle"], 40)
        self.assertEqual(r["warnings"], [])

    def test_missing_values_filled_and_reported(self):
        df = self.frame(); df.loc[(df.unit == 1) & (df.cycle == 40), "sensor_4"] = np.nan
        df.loc[(df.unit == 1) & (df.cycle == 39), "sensor_7"] = np.inf
        r = DC.prepare_window(df, 1, SENSORS, self.scaler(self.frame()))
        self.assertTrue(np.all(np.isfinite(r["scaled"])))
        self.assertTrue(any("filled" in w for w in r["warnings"]))

    def test_whole_sensor_missing_is_error(self):
        df = self.frame(); df.loc[df.unit == 1, "sensor_11"] = np.nan
        with self.assertRaises(DC.DataError):
            DC.prepare_window(df, 1, SENSORS, self.scaler(self.frame()))

    def test_missing_column_is_error(self):
        df = self.frame().drop(columns=["sensor_7"])
        with self.assertRaises(DC.DataError):
            DC.prepare_window(df, 1, SENSORS, self.scaler(self.frame()))

    def test_short_engine_padded(self):
        df = self.frame(); df = df[~((df.unit == 2) & (df.cycle > 12))]
        r = DC.prepare_window(df, 2, SENSORS, self.scaler(self.frame()))
        self.assertEqual(r["scaled"].shape, (30, 5))
        self.assertTrue(any("padded" in w for w in r["warnings"]))

    def test_unsorted_and_duplicates(self):
        df = self.frame().sample(frac=1.0, random_state=0)
        df = pd.concat([df, df[(df.unit == 1) & (df.cycle == 40)]])
        r = DC.prepare_window(df, 1, SENSORS, self.scaler(self.frame()))
        self.assertEqual(r["current_cycle"], 40)
        self.assertTrue(any("Duplicate" in w for w in r["warnings"]))
        ref = DC.prepare_window(self.frame(), 1, SENSORS, self.scaler(self.frame()))
        np.testing.assert_allclose(r["scaled"], ref["scaled"])

    def test_scaler_recipe_matches_preprocessing(self):
        rng = np.random.default_rng(0)
        rows = []
        for u in range(1, 11):
            for c in range(1, 15):
                rows.append({"unit": u, "cycle": c, **{s: rng.normal(u, 1) for s in SENSORS}})
        df = pd.DataFrame(rows)
        units = df["unit"].unique(); np.random.seed(42); np.random.shuffle(units)
        held_out_engine = units[int(len(units) * 0.8)]            # not in the 80% training set
        df.loc[df["unit"] == held_out_engine, "sensor_2"] += 1000.0   # extreme value only there
        a = DC.training_recipe_scaler(df, SENSORS)
        b = DC.training_recipe_scaler(df, SENSORS)
        same, bad = DC.compare_scalers(a, b)
        self.assertTrue(same)
        from sklearn.preprocessing import MinMaxScaler
        same_all, _ = DC.compare_scalers(MinMaxScaler().fit(df[SENSORS]), a)
        self.assertFalse(same_all)          # all-engine fit differs from the 80% recipe


class IntervalsAndWorkOrder(unittest.TestCase):
    CAL = {"adopted": {"0.95": True}, "final_scale_factors": {"0.95": 1.5}}

    def test_build_intervals(self):
        u = PL.build_intervals(80.0, 10.0, None)
        self.assertEqual(u["status"], "uncalibrated"); self.assertIsNone(u["calibrated"])
        c = PL.build_intervals(80.0, 10.0, self.CAL)
        self.assertEqual(c["status"], "calibrated")
        self.assertAlmostEqual(c["calibrated"][1] - 80.0, 1.5 * 1.959964 * 10.0, places=4)
        self.assertEqual(c["uncalibrated"], u["uncalibrated"])         # always kept
        for bad in ({"adopted": {"0.95": False}, "final_scale_factors": {"0.95": 1.5}},
                    {"adopted": {"0.95": True}, "final_scale_factors": {"0.95": float("nan")}},
                    {"adopted": {"0.95": True}, "final_scale_factors": {"0.95": -1}},
                    {"adopted": {}, "final_scale_factors": {}}):
            self.assertEqual(PL.build_intervals(80.0, 10.0, bad)["status"], "uncalibrated")

    def analysis(self, cal=None):
        w = rising_window()
        mc = lambda win, e, c: {"point": linear_model([10, -4, 0, 2, .5])(win), "mean": 1.0,
                                "std": 4.0, "samples": np.zeros(100)}
        return PL.analyze_window(linear_model([10, -4, 0, 2, .5]), mc, w, SENSORS, 7, 99, cal)

    def test_pipeline_single_source_of_truth(self):
        a = self.analysis()
        d = a["agents"]["diagnostic"]
        self.assertAlmostEqual(d["point_rul"], a["point"])
        self.assertEqual(a["health"], d["status"].upper())
        self.assertEqual(a["xai"]["baseline"], a["point"])
        self.assertEqual(d["interval"], list(a["intervals"]["primary"]))

    def test_work_order_valid_json_and_consistent(self):
        a = self.analysis(self.CAL)
        wo = WO.build_work_order(7, 99, a, 100, timestamp="T")
        self.assertEqual(WO.validate_work_order(wo), [])
        wo2 = json.loads(WO.to_json(wo))
        self.assertEqual(wo, wo2)
        self.assertEqual(wo["interval_status"], "calibrated")
        self.assertEqual(wo["predicted_rul"], a["point"])
        self.assertEqual(wo["uncertainty"], a["sigma"])
        self.assertEqual(wo["work_order_id"], "WO-AEGIS-E007-C0099")
        self.assertEqual(wo["top_contributing_sensor"], wo["primary_fault_signal"])   # compat alias
        self.assertIn("deprecated_keys", wo)
        self.assertIn("not a trained AI or LLM", wo["generated_by"])
        self.assertEqual(wo, WO.build_work_order(7, 99, a, 100, timestamp="T"))       # deterministic

    def test_validator_catches_bad_orders(self):
        wo = WO.build_work_order(7, 99, self.analysis(), 100, timestamp="T")
        bad = dict(wo); bad["predicted_rul"] = None
        self.assertTrue(WO.validate_work_order(bad))
        bad = dict(wo); bad["interval_lower"], bad["interval_upper"] = 9.0, 1.0
        self.assertTrue(WO.validate_work_order(bad))
        bad = dict(wo); del bad["priority"]
        self.assertTrue(WO.validate_work_order(bad))
        self.assertEqual(WO._clean({"x": float("nan"), "y": np.float64(2.0), "z": [np.int64(3)]}),
                         {"x": None, "y": 2.0, "z": [3]})


class CalibrationStatistics(unittest.TestCase):
    def synthetic(self, sigma_factor, n_eng=20, per=40, seed=0):
        """errors ~ N(0, true_sd); model claims sigma = true_sd / sigma_factor."""
        rng = np.random.default_rng(seed)
        ids = np.repeat(np.arange(n_eng), per)
        point = np.full(len(ids), 100.0)
        true_sd = 8.0
        y = point + rng.normal(0, true_sd, len(ids))
        sigma = np.full(len(ids), true_sd / sigma_factor)
        return ids, y, point, sigma

    def test_fixes_an_overconfident_model(self):
        ids, y, p, s = self.synthetic(sigma_factor=1.5)       # claims 1.5x too little uncertainty
        df = CAL.engine_split_study(ids, y, p, s, n_splits=100)
        st = CAL.split_summary(df)
        self.assertLess(st["0.95"]["heldout_cov_uncal_mean"], 0.90)
        self.assertAlmostEqual(st["0.95"]["heldout_cov_cal_mean"], 0.95, delta=0.03)
        self.assertAlmostEqual(st["0.95"]["scale_mean"], 1.5, delta=0.2)
        seeds = CAL.seed_scale_table(y, p, {1: s, 2: s * 1.01})
        adopted, _ = CAL.decide_adoption(st, seeds)
        self.assertTrue(adopted["0.95"])

    def test_adoption_refuses_unstable_scale(self):
        ids, y, p, s = self.synthetic(sigma_factor=1.5)
        st = CAL.split_summary(CAL.engine_split_study(ids, y, p, s, n_splits=50))
        seeds = CAL.seed_scale_table(y, p, {1: s, 2: s * 1.6})          # k differs by ~60 %
        adopted, detail = CAL.decide_adoption(st, seeds)
        self.assertFalse(adopted["0.95"])
        self.assertFalse(detail["0.95"]["C4_scale_stable_across_seeds"])

    def test_already_calibrated_model_gets_scale_near_one(self):
        ids, y, p, s = self.synthetic(sigma_factor=1.0)
        st = CAL.split_summary(CAL.engine_split_study(ids, y, p, s, n_splits=60))
        self.assertAlmostEqual(st["0.90"]["scale_mean"], 1.0, delta=0.15)

    def test_needs_enough_engines(self):
        ids, y, p, s = self.synthetic(1.0, n_eng=3)
        with self.assertRaises(ValueError):
            CAL.engine_split_study(ids, y, p, s)

    def test_validation_index_alignment_check(self):
        rows = []
        for u in range(1, 11):
            for c in range(1, 41):
                rows.append({"unit": u, "cycle": c})
        df = pd.DataFrame(rows)
        units = df.unit.unique(); np.random.seed(42); np.random.shuffle(units)
        val = sorted(units[8:])
        y = np.concatenate([np.arange(40 - 30, -1, -1) for _ in val]).astype(float)
        ids, cyc = CAL.reconstruct_validation_index(df, y)
        self.assertEqual(len(ids), len(y)); self.assertEqual(cyc[0], 30)
        with self.assertRaises(ValueError):
            CAL.reconstruct_validation_index(df, y[:-1])


if __name__ == "__main__":
    unittest.main()
