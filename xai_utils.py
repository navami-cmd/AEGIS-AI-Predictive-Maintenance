"""
xai_utils.py - signed occlusion sensor contributions (model sensitivity).

EXACT FORMULA (for window X of shape (30, n_sensors) and model f):

    X_j  = X with sensor j's whole 30-cycle column replaced by that column's mean
    delta_j      = f(X_j) - f(X)          signed change in predicted RUL (cycles)
    importance_j = |delta_j|
    share_j      = importance_j / sum_k importance_k      (0 for all j if the sum is 0)

DIRECTION
    delta_j > 0 : removing sensor j's recent variation RAISES the RUL estimate,
                  i.e. its recent pattern was pulling the estimate toward SHORTER life.
    delta_j < 0 : removing it LOWERS the estimate, i.e. its pattern was pushing the
                  estimate toward LONGER life.
    delta_j = 0 : the model ignores this sensor in this window.

LIMITS: one sensor at a time (effects are not additive, shares are relative to the
sum of single-sensor effects), and the baseline is the window mean, not a healthy
reference. This measures what the MODEL is sensitive to. It is not evidence of
physical causation or of a faulty component.

predict_fn(window) -> float must be the same point prediction (dropout OFF) that
the dashboard reports as the predicted RUL.
"""

import math

import numpy as np

from sensor_info import sensor_description, sensor_symbol

TOP_TIERS = ((3, "Top 3"), (10, "Top 10"))


def tier_for_rank(rank):
    for limit, name in TOP_TIERS:
        if rank <= limit:
            return name
    return "Lower"


def occlusion_effects(predict_fn, window, sensors, baseline=None, atol=1e-3):
    """
    Returns {"baseline", "total_abs_effect", "all_zero", "effects": [...]} where
    effects is sorted by importance (ties broken by sensor order, so the output is
    deterministic). Raises ValueError for invalid input.
    """
    window = np.asarray(window, dtype=np.float64)
    if window.ndim != 2 or window.shape[1] != len(sensors):
        raise ValueError(
            f"window shape {window.shape} does not match {len(sensors)} sensors"
        )
    if not np.all(np.isfinite(window)):
        raise ValueError("window contains NaN or infinite values")

    original = float(predict_fn(window))
    if not math.isfinite(original):
        raise ValueError("model returned a non-finite prediction")
    if baseline is not None and abs(original - float(baseline)) > atol:
        raise ValueError(
            f"XAI baseline {original:.4f} differs from reported point prediction "
            f"{float(baseline):.4f}"
        )

    deltas = []
    for j in range(len(sensors)):
        modified = window.copy()
        modified[:, j] = modified[:, j].mean()
        new = float(predict_fn(modified))
        if not math.isfinite(new):
            raise ValueError(f"non-finite prediction when occluding {sensors[j]}")
        deltas.append(new - original)

    deltas = np.asarray(deltas)
    importance = np.abs(deltas)
    total = float(importance.sum())
    all_zero = total <= 1e-12
    order = sorted(range(len(sensors)), key=lambda j: (-importance[j], j))

    effects = []
    for rank, j in enumerate(order, start=1):
        d = float(deltas[j])
        if abs(d) <= 1e-12:
            direction = "none"
        else:
            direction = "raises" if d > 0 else "lowers"
        effects.append({
            "sensor": sensors[j],
            "symbol": sensor_symbol(sensors[j]),
            "description": sensor_description(sensors[j]),
            "importance": float(importance[j]),
            "delta": d,
            "share": 0.0 if all_zero else float(importance[j] / total),
            "direction": direction,
            "rank": rank,
            "tier": tier_for_rank(rank),
        })

    return {
        "baseline": original,
        "total_abs_effect": total,
        "all_zero": bool(all_zero),
        "effects": effects,
    }
