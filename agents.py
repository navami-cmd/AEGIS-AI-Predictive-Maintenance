"""
agents.py - the AEGIS four-agent reasoning chain.

IMPORTANT: these are deterministic, RULE-BASED agents. They are not trained
models and not LLM agents. Every output is computed from numbers produced
elsewhere in the pipeline:

    predicted RUL (point)  - LSTM, dropout off
    MC Dropout sigma       - shared mc_utils.mc_dropout_predict (100 seeded passes)
    prediction interval    - point +/- z * sigma (calibrated only if calibration.py
                             adopted it; otherwise labelled uncalibrated)
    sensor contributions   - signed occlusion effects (xai_utils.py)

Chain:  Diagnostic -> Root-Cause -> Verification -> Maintenance

The only fixed numbers are the two POLICY thresholds below (the project's existing
30/60-cycle RUL bands). They are maintenance policy choices, not learned or tuned
values. There are no sigma cut-offs and no absolute importance cut-offs: sensor
evidence is judged by contribution SHARE (against a uniform share) and by effect
size relative to the MC Dropout sigma.

Sensor contributions describe MODEL SENSITIVITY. They are not proof that a
physical component is faulty, and no agent claims otherwise.
"""

import math

from sensor_info import sensor_long_label

# ------------------------------------------------------------
# Policy thresholds (existing project convention)
# ------------------------------------------------------------
CRITICAL_RUL = 30.0      # predicted RUL <= 30 cycles  -> Critical
WARNING_RUL = 60.0       # predicted RUL <= 60 cycles  -> Warning
POLICY_NOTE = (
    f"Status bands are policy thresholds (Critical <= {CRITICAL_RUL:.0f}, "
    f"Warning <= {WARNING_RUL:.0f} cycles), not learned values."
)

STATUS_ORDER = ["Healthy", "Warning", "Critical"]          # increasing severity
PRIORITY_LADDER = ["Routine", "Elevated", "High", "Immediate"]
DEFAULT_INTERVAL_LABEL = "95% Gaussian interval from MC Dropout (uncalibrated)"
DISCLAIMER = (
    "Rule-based reasoning on model outputs. Sensor contribution reflects model "
    "sensitivity, not a diagnosed physical fault."
)
GAUSS_95 = 1.959964


def status_from_rul(rul):
    """Policy band for a RUL value (cycles)."""
    if rul <= CRITICAL_RUL:
        return "Critical"
    if rul <= WARNING_RUL:
        return "Warning"
    return "Healthy"


def _finite(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _normalize(sensor_importance):
    """Copy the items and make sure every item has importance, share, delta."""
    items = [dict(x) for x in (sensor_importance or [])]
    for it in items:
        it["importance"] = float(it["importance"]) if _finite(it.get("importance")) else 0.0
    total = sum(it["importance"] for it in items)
    for it in items:
        if not _finite(it.get("share")):
            it["share"] = (it["importance"] / total) if total > 0 else 0.0
        it["delta"] = float(it["delta"]) if _finite(it.get("delta")) else None
    items.sort(key=lambda x: -x["importance"])
    return items


# ============================================================
# 1. DIAGNOSTIC AGENT
# ============================================================
class DiagnosticAgent:
    def run(self, predicted_rul, uncertainty, sensor_importance,
            interval=None, interval_label=None):

        label = interval_label or DEFAULT_INTERVAL_LABEL

        if not _finite(predicted_rul):
            return {
                "valid": False, "status": "Unknown",
                "diagnosis": "The RUL prediction is not a valid number, so no status can be assigned.",
                "evidence": ["Predicted RUL: not available"],
                "interval": None, "interval_label": label,
                "worst_case_status": "Unknown", "best_case_status": "Unknown",
                "status_ambiguous": False, "boundary_crossings": [], "policy": POLICY_NOTE,
            }

        rul = float(predicted_rul)
        sigma = float(uncertainty) if _finite(uncertainty) else 0.0
        if interval is None or not (_finite(interval[0]) and _finite(interval[1])):
            interval = (max(0.0, rul - GAUSS_95 * sigma), rul + GAUSS_95 * sigma)
        lo, hi = float(interval[0]), float(interval[1])

        status = status_from_rul(rul)
        worst = status_from_rul(lo)
        best = status_from_rul(hi)
        ambiguous = not (worst == status == best)

        crossings = []
        if lo <= CRITICAL_RUL < rul:
            crossings.append(f"lower bound {lo:.1f} is at or below the {CRITICAL_RUL:.0f}-cycle Critical boundary")
        if lo <= WARNING_RUL < rul:
            crossings.append(f"lower bound {lo:.1f} is at or below the {WARNING_RUL:.0f}-cycle Warning boundary")
        if hi > WARNING_RUL >= rul:
            crossings.append(f"upper bound {hi:.1f} is above the {WARNING_RUL:.0f}-cycle Warning boundary")
        if hi > CRITICAL_RUL >= rul:
            crossings.append(f"upper bound {hi:.1f} is above the {CRITICAL_RUL:.0f}-cycle Critical boundary")

        text = {
            "Critical": "The engine shows a high risk of near-term failure.",
            "Warning": "The engine shows signs of degradation and requires attention.",
            "Healthy": "The engine is operating with relatively high remaining useful life.",
        }[status]
        if ambiguous:
            text += (
                f" The {label} ({lo:.1f} to {hi:.1f} cycles) spans more than one status band; "
                f"the conservative (lower-bound) status would be {worst}."
            )

        top3 = _normalize(sensor_importance)[:3]
        evidence = [
            f"Predicted RUL: {rul:.2f} cycles",
            f"Uncertainty: ±{sigma:.2f} cycles (MC Dropout σ)",
            f"{label}: {lo:.1f} to {hi:.1f} cycles",
            "Top contributing sensors (model sensitivity): "
            + (", ".join(x["sensor"] for x in top3) if top3 else "none"),
        ]

        return {
            "valid": True, "status": status, "diagnosis": text, "evidence": evidence,
            "interval": [lo, hi], "interval_label": label,
            "worst_case_status": worst, "best_case_status": best,
            "status_ambiguous": ambiguous, "boundary_crossings": crossings,
            "policy": POLICY_NOTE, "point_rul": rul,
        }


# ============================================================
# 2. ROOT-CAUSE AGENT
# ============================================================
class RootCauseAgent:
    """Ranks sensors by contribution SHARE and writes explainable hypotheses."""

    def run(self, sensor_importance, uncertainty=None, top_k=5):
        items = _normalize(sensor_importance)
        n = len(items)
        uniform = (1.0 / n) if n else 0.0
        active = [x for x in items if x["importance"] > 0]

        hypotheses = []
        for rank, it in enumerate(active[:top_k], start=1):
            label = sensor_long_label(it["sensor"])
            d = it["delta"]
            if d is None or d == 0:
                direction = ""
            elif d > 0:
                direction = " Its recent pattern is pulling the estimate toward shorter life."
            else:
                direction = " Its recent pattern is pushing the estimate toward longer life."
            effect = (f"{d:+.2f} cycles" if d is not None
                      else f"{it['importance']:.2f} cycles (unsigned)")
            hypotheses.append({
                "rank": rank,
                "sensor": it["sensor"],
                "label": label,
                "importance": it["importance"],
                "delta": d,
                "share": it["share"],
                "uniform_share": uniform,
                "tier": it.get("tier"),
                "hypothesis": (
                    f"The model's RUL estimate is sensitive to {label}: flattening its recent "
                    f"history changes the estimate by {effect} ({it['share']:.0%} of the total "
                    f"single-sensor effect; an even split would be {uniform:.0%})."
                    f"{direction} This is model sensitivity, not a confirmed fault."
                ),
            })

        top1 = active[0]["share"] if active else 0.0
        top3 = sum(x["share"] for x in active[:3])
        note = (
            "No sensor has a non-zero contribution in this window, so no hypothesis is generated."
            if not active else
            f"Top-3 sensors carry {top3:.0%} of the total effect (an even split over "
            f"{n} sensors would give {min(3, n) * uniform:.0%})."
        )
        return {
            "hypotheses": hypotheses,
            "primary_cause": active[0]["sensor"] if active else "No dominant signal",
            "primary_label": sensor_long_label(active[0]["sensor"]) if active else "No dominant signal",
            "concentration": {"top1_share": top1, "top3_share": top3,
                              "uniform_top3_share": min(3, n) * uniform},
            "note": note,
        }


# ============================================================
# 3. VERIFICATION AGENT
# ============================================================
class VerificationAgent:
    """
    Two evidence checks per hypothesis, both without fixed cut-offs:
      (a) share check : contribution share >= uniform share (1 / n sensors)
      (b) effect vs MC: |effect| >= MC Dropout sigma, i.e. the sensor's effect is at
                        least as large as the model's own prediction noise
                        (a signal-to-noise heuristic, not a statistical test)
    """

    def run(self, root_cause_result, sensor_importance, uncertainty, diagnostic_result=None):
        sigma = float(uncertainty) if _finite(uncertainty) and float(uncertainty) > 0 else None
        verified = []

        for h in root_cause_result.get("hypotheses", []):
            share_ok = h["share"] >= h["uniform_share"]
            effect = abs(h["delta"]) if h["delta"] is not None else h["importance"]
            snr_ok = (effect >= sigma) if sigma is not None else None

            checks = [share_ok] + ([snr_ok] if snr_ok is not None else [])
            passed = sum(bool(c) for c in checks)
            if passed == len(checks):
                verdict, conf = "Supported", "High"
            elif passed == 0:
                verdict, conf = "Weak", "Low"
            else:
                verdict, conf = "Partly supported", "Medium"

            snr_text = (
                f"effect {effect:.2f} cycles vs MC σ {sigma:.2f} ({'pass' if snr_ok else 'fail'})"
                if sigma is not None else "effect vs MC σ not assessable"
            )
            verified.append({
                "sensor": h["sensor"],
                "verdict": verdict,
                "result": (
                    f"{verdict}: share {h['share']:.1%} vs even split {h['uniform_share']:.1%} "
                    f"({'pass' if share_ok else 'fail'}); {snr_text}."
                ),
                "confidence": conf,
                "checks": {"share_vs_uniform": bool(share_ok),
                           "effect_vs_mc_sigma": None if snr_ok is None else bool(snr_ok)},
                "share": h["share"], "effect_cycles": effect,
            })

        n_sup = sum(v["verdict"] == "Supported" for v in verified)
        n_par = sum(v["verdict"] == "Partly supported" for v in verified)
        n_weak = sum(v["verdict"] == "Weak" for v in verified)

        sigma_text = f"{sigma:.1f}" if sigma is not None else "n/a"
        if diagnostic_result and diagnostic_result.get("interval"):
            lo, hi = diagnostic_result["interval"]
            note = (f"MC Dropout σ = {sigma_text} cycles; "
                    f"{diagnostic_result['interval_label']}: {lo:.1f} to {hi:.1f} cycles.")
            if diagnostic_result["boundary_crossings"]:
                note += (" Interval crosses a status boundary: "
                         + "; ".join(diagnostic_result["boundary_crossings"]) + ".")
            else:
                note += " The interval stays inside one status band."
        else:
            note = (f"MC Dropout σ = {sigma_text} cycles." if sigma is not None
                    else "Uncertainty not available.")

        return {
            "verified_causes": verified,
            "uncertainty_note": note,
            "n_supported": n_sup, "n_partial": n_par, "n_weak": n_weak,
            "evidence_summary": (
                f"{n_sup} of {len(verified)} hypotheses supported by both checks, "
                f"{n_par} partly, {n_weak} weak."
            ) if verified else "No hypotheses to verify.",
        }


# ============================================================
# 4. MAINTENANCE AGENT
# ============================================================
class MaintenanceAgent:
    """
    Priority ladder: Routine < Elevated < High < Immediate.

      base (point status):  Healthy -> Routine, Warning -> High, Critical -> Immediate
      interval escalation (uses the interval's LOWER bound, so wider uncertainty can
      only raise priority, never lower it):
        Healthy : +1 level per status band the lower bound reaches
                  (Warning -> Elevated, Critical -> High)
        Warning : stays High; if the lower bound reaches Critical the reason says so
                  and the action asks to prepare for an immediate inspection
        Critical: Immediate
      Verification results do NOT change the urgency (that would let a larger sigma
      weaken the evidence and lower the priority). They decide the inspection focus,
      the wording of the action and the evidence-confidence field.
    """

    def run(self, diagnostic_result, root_cause_result, verification_result):
        status = diagnostic_result["status"]
        if status == "Unknown":
            return {
                "priority": "Manual review",
                "action": "The prediction is invalid. Check the sensor data and model output before any decision.",
                "focus": "Data / model validation", "focus_sensors": [],
                "reason": "No valid RUL prediction was available.", "escalated": False,
                "rationale": [], "disclaimer": DISCLAIMER,
            }

        lo, hi = diagnostic_result["interval"]
        band = STATUS_ORDER.index(status)
        worst = STATUS_ORDER.index(diagnostic_result["worst_case_status"])
        e = max(0, worst - band)
        verdicts = [v["verdict"] for v in verification_result.get("verified_causes", [])]
        if verdicts and verdicts[0] == "Supported":
            evidence_confidence = "Strong"
        elif verdicts and verdicts[0] == "Partly supported":
            evidence_confidence = "Moderate"
        else:
            evidence_confidence = "Limited"

        base_level = {0: 0, 1: 2, 2: 3}[band]
        level = base_level
        rationale = [f"{status} by point prediction (policy bands)."]

        if band == 0 and e >= 1:
            level = min(2, e)
            rationale.append(
                f"The interval lower bound ({lo:.1f}) reaches the {STATUS_ORDER[worst]} band, "
                f"so priority is raised from Routine."
            )
        elif band == 1 and e >= 1:
            rationale.append(
                f"The interval lower bound ({lo:.1f}) reaches Critical, so the engine could already "
                f"be in the Critical band; priority stays High and an immediate inspection should be prepared."
            )
        elif band == 2:
            rationale.append("Critical already carries the highest priority.")

        priority = PRIORITY_LADDER[level]
        escalated = level > base_level

        actions = {
            "Immediate": "Immediate maintenance inspection recommended.",
            "High": "Schedule preventive maintenance soon and inspect the focus channels listed.",
            "Elevated": ("Plan an inspection at the next opportunity and re-run the prediction as new "
                         "cycles arrive; the interval reaches a lower status band."),
            "Routine": "Continue monitoring and perform routine maintenance.",
        }
        action = actions[priority]
        if band == 1 and e >= 1:
            action += " The interval reaches the Critical band; prepare for an immediate inspection."

        focus_sensors = [
            {"sensor": v["sensor"], "label": sensor_long_label(v["sensor"]), "verdict": v["verdict"]}
            for v in verification_result.get("verified_causes", [])
            if v["verdict"] in ("Supported", "Partly supported")
        ][:3]

        if level >= 1:
            if focus_sensors:
                focus = focus_sensors[0]["sensor"]
                action += " Channels with supporting evidence: " + "; ".join(
                    f"{f['label']} ({f['verdict'].lower()})" for f in focus_sensors) + "."
            else:
                focus = "Full-engine inspection (no single sensor supported)"
                action += (" Sensor-level evidence is weak, so inspect the engine as a whole rather "
                           "than a single channel.")
        else:
            focus = focus_sensors[0]["sensor"] if focus_sensors else "None required"

        reason = " ".join(filter(None, [
            f"{status} at point RUL {diagnostic_result['point_rul']:.1f} cycles;",
            f"{diagnostic_result['interval_label']}: {lo:.1f} to {hi:.1f} cycles.",
            verification_result.get("evidence_summary", ""),
            " ".join(rationale[1:]),
            DISCLAIMER,
        ]))

        return {
            "priority": priority, "action": action, "focus": focus,
            "focus_sensors": focus_sensors, "reason": reason,
            "escalated": escalated, "rationale": rationale,
            "evidence_confidence": evidence_confidence, "disclaimer": DISCLAIMER,
        }


# ============================================================
# ORCHESTRATION (same entry points as before)
# ============================================================
class AgentOrchestrator:

    def run(self, predicted_rul, uncertainty, sensor_importance,
            interval=None, interval_label=None):

        diagnostic = DiagnosticAgent().run(
            predicted_rul, uncertainty, sensor_importance, interval, interval_label)
        root_cause = RootCauseAgent().run(sensor_importance, uncertainty)
        verification = VerificationAgent().run(
            root_cause, sensor_importance, uncertainty, diagnostic)
        maintenance = MaintenanceAgent().run(diagnostic, root_cause, verification)

        return {
            "diagnostic": diagnostic,
            "root_cause": root_cause,
            "verification": verification,
            "maintenance": maintenance,
        }


def run_agent_pipeline(predicted_rul, uncertainty, sensor_importance,
                       interval=None, interval_label=None):
    """Backward compatible: the last two arguments are optional."""
    return AgentOrchestrator().run(
        predicted_rul, uncertainty, sensor_importance, interval, interval_label)
