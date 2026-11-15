"""
work_order.py - builds and validates the exported maintenance work order.

Everything comes from the same analyze_window() result the dashboard shows, so the
export cannot disagree with the screen. The work-order id is deterministic
(engine + cycle) so repeated exports of the same observation are identical apart
from the timestamp.
"""

import json
import math
from datetime import datetime

import numpy as np

GENERATED_BY = "AEGIS rule-based agents (deterministic rules, not a trained AI or LLM agent)"
DISCLAIMER = ("Sensor contributions describe model sensitivity, not diagnosed physical faults. "
              "Intervals are uncertainty indicators, not guaranteed probabilities.")

REQUIRED_KEYS = [
    "work_order_id", "engine_id", "current_cycle", "predicted_rul", "uncertainty",
    "mc_passes", "interval_level", "interval_lower", "interval_upper", "interval_status",
    "health_status", "priority", "recommended_action", "top_contributing_sensor",
]


def _clean(obj):
    """Make an object JSON-safe (numpy -> python, NaN/inf -> None)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def build_work_order(engine_id, current_cycle, analysis, mc_passes, timestamp=None):
    """analysis = pipeline.analyze_window() result."""
    ag = analysis["agents"]
    iv = analysis["intervals"]
    top = analysis["xai"]["effects"]
    top_sensor = ag["root_cause"]["primary_cause"]
    lo, hi = iv["primary"]

    wo = {
        "work_order_id": f"WO-AEGIS-E{int(engine_id):03d}-C{int(current_cycle):04d}",
        "engine_id": int(engine_id),
        "current_cycle": int(current_cycle),
        "predicted_rul": float(analysis["point"]),
        "uncertainty": float(analysis["sigma"]),
        "uncertainty_note": "std of MC Dropout samples (cycles)",
        "mc_passes": int(mc_passes),
        "interval_level": float(iv["level"]),
        "interval_lower": float(lo),
        "interval_upper": float(hi),
        "interval_status": iv["status"],
        "interval_label": iv["primary_label"],
        "uncalibrated_interval": {"lower": float(iv["uncalibrated"][0]), "upper": float(iv["uncalibrated"][1])},
        "calibrated_interval": (
            None if iv["calibrated"] is None else
            {"lower": float(iv["calibrated"][0]), "upper": float(iv["calibrated"][1]),
             "width_scale": float(iv["scale"])}),
        "health_status": analysis["health"],
        "priority": ag["maintenance"]["priority"],
        "recommended_action": ag["maintenance"]["action"],
        "inspection_focus": ag["maintenance"]["focus"],
        "priority_rationale": ag["maintenance"]["rationale"],
        "boundary_crossings": ag["diagnostic"]["boundary_crossings"],
        "verification_summary": ag["verification"]["evidence_summary"],
        "top_contributing_sensor": top_sensor,
        "top_contributing_sensors": [
            {"sensor": e["sensor"], "symbol": e["symbol"], "description": e["description"],
             "effect_cycles": e["delta"], "share": e["share"], "tier": e["tier"]}
            for e in top[:3]
        ],
        "sensor_effect_definition": ("effect = RUL(sensor history replaced by its 30-cycle mean) - RUL(original); "
                                     "positive = the sensor's pattern pulls the estimate toward shorter life"),
        # Deprecated alias kept so older consumers keep working.
        "primary_fault_signal": top_sensor,
        "deprecated_keys": {"primary_fault_signal": "use top_contributing_sensor (model sensitivity, not a fault)"},
        "generated_by": GENERATED_BY,
        "disclaimer": DISCLAIMER,
        "timestamp": timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return _clean(wo)


def validate_work_order(wo):
    """Returns a list of problems (empty list = valid)."""
    problems = [f"missing key: {k}" for k in REQUIRED_KEYS if k not in wo]
    for k in ("predicted_rul", "uncertainty", "interval_lower", "interval_upper"):
        v = wo.get(k)
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            problems.append(f"{k} is not a finite number")
    if not problems:
        if wo["interval_lower"] > wo["interval_upper"]:
            problems.append("interval_lower > interval_upper")
        if wo["uncertainty"] < 0:
            problems.append("uncertainty is negative")
        if wo["interval_status"] not in ("calibrated", "uncalibrated"):
            problems.append("interval_status must be 'calibrated' or 'uncalibrated'")
    return problems


def to_json(wo):
    """Strict JSON text (raises if anything non-finite slipped through)."""
    return json.dumps(wo, indent=4, allow_nan=False)
