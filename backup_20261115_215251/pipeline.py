"""
pipeline.py - the AEGIS analysis chain for ONE engine window, in one place.

    window -> MC Dropout (shared) -> point RUL + sigma -> prediction interval
           -> signed occlusion XAI -> four rule-based agents

The dashboard (app.py), the tests and any export all call analyze_window(), so the
RUL, uncertainty, interval, contributions and agent outputs cannot disagree.

point_fn(window) -> float : model prediction, dropout OFF
mc_fn(window, engine_id, cycle) -> dict : mc_utils.mc_dropout_predict result
"""

import math

from agents import run_agent_pipeline
from mc_utils import DEFAULT_LEVEL, INTERVAL_LABEL, prediction_interval
from xai_utils import occlusion_effects


def build_intervals(point, sigma, calibration=None, level=DEFAULT_LEVEL):
    """
    Always returns the uncalibrated interval. A calibrated one is added ONLY if the
    validation study (calibration.py) adopted it for this level. `primary` is the
    calibrated interval when available, otherwise the uncalibrated one.
    """
    key = f"{level:.2f}"
    unc = prediction_interval(point, sigma, level)
    cal, scale = None, None
    if calibration:
        k = (calibration.get("final_scale_factors") or {}).get(key)
        if calibration.get("adopted", {}).get(key) and k is not None and math.isfinite(k) and k > 0:
            scale = float(k)
            cal = prediction_interval(point, sigma, level, scale=scale)

    if cal is not None:
        primary, status = cal, "calibrated"
        label = (f"Calibrated {level * 100:.0f}% interval (validation-fitted width ×{scale:.2f}; "
                 f"empirical, not guaranteed)")
    else:
        primary, status, label = unc, "uncalibrated", INTERVAL_LABEL

    return {"level": level, "uncalibrated": unc, "calibrated": cal, "scale": scale,
            "primary": primary, "status": status, "primary_label": label}


def analyze_window(point_fn, mc_fn, window, sensors, engine_id, cycle, calibration=None):
    mc = mc_fn(window, engine_id, cycle)
    point, sigma = float(mc["point"]), float(mc["std"])
    if not (math.isfinite(point) and math.isfinite(sigma)):
        raise ValueError("Model produced a non-finite prediction or uncertainty.")

    xai = occlusion_effects(point_fn, window, sensors, baseline=point)
    intervals = build_intervals(point, sigma, calibration)
    agents = run_agent_pipeline(
        point, sigma, xai["effects"],
        interval=intervals["primary"], interval_label=intervals["primary_label"],
    )
    return {
        "mc": mc, "point": point, "sigma": sigma, "intervals": intervals,
        "xai": xai, "agents": agents,
        "health": agents["diagnostic"]["status"].upper(),
    }
