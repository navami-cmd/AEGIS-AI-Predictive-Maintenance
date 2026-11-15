"""Unit tests for the four rule-based agents. Run:  python -m unittest -v test_agents"""
import copy
import json
import math
import unittest

import agents as A


def effects(deltas, names=None):
    """Build occlusion-style items from signed deltas."""
    names = names or [f"sensor_{i + 2}" for i in range(len(deltas))]
    imp = [abs(d) for d in deltas]
    tot = sum(imp)
    items = [{"sensor": n, "importance": i, "delta": d,
              "share": (i / tot if tot else 0.0), "tier": "Top 3" if k < 3 else "Lower"}
             for k, (n, d, i) in enumerate(zip(names, deltas, imp))]
    items.sort(key=lambda x: -x["importance"])
    return items


EV = effects([8.0, -6.0, 4.0, 1.0, 0.5, 0.2, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def run(rul, sigma, items=EV, interval=None):
    return A.run_agent_pipeline(rul, sigma, items, interval=interval)


class StatusBoundaries(unittest.TestCase):
    def test_policy_edges(self):
        self.assertEqual(A.status_from_rul(30.0), "Critical")
        self.assertEqual(A.status_from_rul(30.01), "Warning")
        self.assertEqual(A.status_from_rul(60.0), "Warning")
        self.assertEqual(A.status_from_rul(60.01), "Healthy")
        self.assertEqual(A.status_from_rul(-5), "Critical")

    def test_diagnostic_matches_policy(self):
        for rul in (0, 29.9, 30, 45, 60, 61, 150):
            self.assertEqual(run(rul, 5)["diagnostic"]["status"], A.status_from_rul(rul))


class IntervalAwareness(unittest.TestCase):
    def test_straddle_flags(self):
        d = run(70, 10, interval=(50.0, 90.0))["diagnostic"]
        self.assertTrue(d["status_ambiguous"])
        self.assertEqual(d["worst_case_status"], "Warning")
        self.assertTrue(any("Warning boundary" in c for c in d["boundary_crossings"]))

    def test_inside_one_band(self):
        d = run(100, 5, interval=(90.0, 110.0))["diagnostic"]
        self.assertFalse(d["status_ambiguous"])
        self.assertEqual(d["boundary_crossings"], [])

    def test_default_interval_when_missing(self):
        d = run(100, 10)["diagnostic"]
        self.assertAlmostEqual(d["interval"][1] - d["interval"][0], 2 * 1.959964 * 10, places=3)


class MaintenanceLadder(unittest.TestCase):
    def pr(self, rul, lo, hi):
        return run(rul, 5, interval=(lo, hi))["maintenance"]["priority"]

    def test_base_priorities(self):
        self.assertEqual(self.pr(100, 95, 105), "Routine")
        self.assertEqual(self.pr(45, 40, 50), "High")
        self.assertEqual(self.pr(20, 15, 25), "Immediate")

    def test_interval_escalation_for_healthy(self):
        self.assertEqual(self.pr(70, 55, 85), "Elevated")   # lower bound reaches Warning
        self.assertEqual(self.pr(70, 20, 120), "High")      # lower bound reaches Critical

    def test_warning_stays_high_but_flags_critical_risk(self):
        m = run(45, 5, interval=(25.0, 65.0))["maintenance"]
        self.assertEqual(m["priority"], "High")
        self.assertIn("Critical band", m["action"])

    def test_uncertainty_never_lowers_priority(self):
        order = {p: i for i, p in enumerate(A.PRIORITY_LADDER)}
        for rul in (10, 25, 30, 35, 55, 60, 65, 90, 140):
            prev = -1
            for sigma in range(0, 61, 2):
                lo = max(0.0, rul - 1.959964 * sigma)
                hi = rul + 1.959964 * sigma
                p = order[run(rul, sigma, interval=(lo, hi))["maintenance"]["priority"]]
                self.assertGreaterEqual(p, prev, f"rul={rul} sigma={sigma}")
                prev = p

    def test_invalid_rul_needs_manual_review(self):
        for bad in (float("nan"), float("inf"), None, "abc"):
            r = run(bad, 5)
            self.assertEqual(r["diagnostic"]["status"], "Unknown")
            self.assertEqual(r["maintenance"]["priority"], "Manual review")


class RootCauseAndVerification(unittest.TestCase):
    def test_shares_and_direction_text(self):
        rc = run(50, 5)["root_cause"]
        self.assertEqual(rc["primary_cause"], "sensor_2")
        self.assertAlmostEqual(sum(x["share"] for x in EV), 1.0, places=9)
        h = rc["hypotheses"][0]
        self.assertIn("pulling the estimate toward shorter life", h["hypothesis"])   # delta > 0
        self.assertIn("pushing the estimate toward longer life", rc["hypotheses"][1]["hypothesis"])
        self.assertIn("not a confirmed fault", h["hypothesis"])

    def test_all_zero_effects_are_safe(self):
        zero = effects([0.0] * 17)
        r = run(40, 5, items=zero)
        self.assertEqual(r["root_cause"]["hypotheses"], [])
        self.assertEqual(r["root_cause"]["primary_cause"], "No dominant signal")
        self.assertEqual(r["verification"]["verified_causes"], [])
        self.assertEqual(r["maintenance"]["priority"], "High")
        self.assertIn("whole", r["maintenance"]["action"])

    def test_empty_and_missing_input(self):
        for items in ([], None):
            r = run(40, 5, items=items)
            self.assertEqual(r["root_cause"]["hypotheses"], [])
            self.assertEqual(r["maintenance"]["priority"], "High")

    def test_verification_checks_use_sigma(self):
        small = effects([1.0, 0.9, 0.8] + [0.7] * 14)              # shares ~ even split
        v = run(50, 5.0, items=small)["verification"]["verified_causes"][0]
        self.assertFalse(v["checks"]["effect_vs_mc_sigma"])          # 1.0 < sigma 5
        big = run(50, 0.5, items=small)["verification"]["verified_causes"][0]
        self.assertTrue(big["checks"]["effect_vs_mc_sigma"])         # 1.0 >= sigma 0.5

    def test_sigma_missing_is_not_assessable_not_crash(self):
        v = run(50, 0.0)["verification"]["verified_causes"][0]
        self.assertIsNone(v["checks"]["effect_vs_mc_sigma"])
        self.assertIn("not assessable", v["result"])

    def test_verdict_levels(self):
        v = run(50, 1.0)["verification"]["verified_causes"]
        self.assertEqual(v[0]["verdict"], "Supported")                  # 8.0 >> sigma, large share
        weak = effects([0.2] * 17)
        w = run(50, 5.0, items=weak)["verification"]["verified_causes"][0]
        self.assertIn(w["verdict"], ("Partly supported", "Weak"))


class OutputContract(unittest.TestCase):
    KEYS = {
        "diagnostic": ["status", "diagnosis", "evidence"],
        "root_cause": ["hypotheses", "primary_cause"],
        "verification": ["verified_causes", "uncertainty_note"],
        "maintenance": ["priority", "action", "focus", "reason"],
    }

    def test_keys_app_depends_on(self):
        r = run(55, 8)
        for agent, keys in self.KEYS.items():
            for k in keys:
                self.assertIn(k, r[agent], f"{agent}.{k}")
        self.assertIn("result", r["verification"]["verified_causes"][0])

    def test_deterministic_and_json_safe(self):
        a, b = run(55, 8), run(55, 8)
        self.assertEqual(a, b)
        json.dumps(a, allow_nan=False)

    def test_inputs_not_mutated(self):
        before = copy.deepcopy(EV)
        run(55, 8)
        self.assertEqual(before, EV)

    def test_no_causal_overclaim_wording(self):
        text = json.dumps(run(25, 8)).lower()
        for banned in ("is causing", "caused by", "root cause is", "faulty"):
            self.assertNotIn(banned, text)

    def test_backward_compatible_signature(self):
        A.run_agent_pipeline(50, 5, EV)               # old 3-argument call


if __name__ == "__main__":
    unittest.main()
