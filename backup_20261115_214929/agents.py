import numpy as np


class DiagnosticAgent:
    def run(self, predicted_rul, uncertainty, sensor_importance):
        if predicted_rul <= 30:
            status = "Critical"
            diagnosis = "The engine shows a high risk of near-term failure."
        elif predicted_rul <= 60:
            status = "Warning"
            diagnosis = "The engine shows signs of degradation and requires attention."
        else:
            status = "Healthy"
            diagnosis = "The engine is operating with relatively high remaining useful life."

        top_sensors = sensor_importance[:3]

        evidence = [
            f"Predicted RUL: {predicted_rul:.2f} cycles",
            f"Uncertainty: ±{uncertainty:.2f} cycles",
            "Important sensors: " +
            ", ".join([x["sensor"] for x in top_sensors])
        ]

        return {
            "status": status,
            "diagnosis": diagnosis,
            "evidence": evidence
        }


class RootCauseAgent:
    def run(self, sensor_importance):
        top = sensor_importance[:5]

        hypotheses = []

        for item in top:
            sensor = item["sensor"]
            importance = item["importance"]

            if importance > 10:
                severity = "High"
            elif importance > 3:
                severity = "Medium"
            else:
                severity = "Low"

            hypotheses.append({
                "sensor": sensor,
                "importance": importance,
                "severity": severity,
                "hypothesis":
                    f"{sensor} may be contributing to the observed degradation."
            })

        return {
            "hypotheses": hypotheses,
            "primary_cause": top[0]["sensor"] if top else "Unknown"
        }


class VerificationAgent:
    def run(self, root_cause_result, sensor_importance, uncertainty):
        verified = []

        importance_values = [
            x["importance"] for x in sensor_importance
        ]

        average_importance = (
            np.mean(importance_values)
            if importance_values else 0
        )

        for hypothesis in root_cause_result["hypotheses"]:

            sensor = hypothesis["sensor"]
            importance = hypothesis["importance"]

            if importance >= average_importance:
                result = "Supported by model evidence"
                confidence = "High"
            else:
                result = "Weak supporting evidence"
                confidence = "Low"

            verified.append({
                "sensor": sensor,
                "result": result,
                "confidence": confidence
            })

        uncertainty_note = (
            "Prediction uncertainty is relatively low."
            if uncertainty < 15
            else
            "Prediction uncertainty is relatively high; inspection is recommended."
        )

        return {
            "verified_causes": verified,
            "uncertainty_note": uncertainty_note
        }


class MaintenanceAgent:
    def run(
        self,
        diagnostic_result,
        root_cause_result,
        verification_result
    ):

        status = diagnostic_result["status"]
        primary_cause = root_cause_result["primary_cause"]

        if status == "Critical":
            action = "Immediate maintenance inspection recommended."
            priority = "Immediate"
        elif status == "Warning":
            action = "Schedule preventive maintenance and inspect the identified sensor/component."
            priority = "High"
        else:
            action = "Continue monitoring and perform routine maintenance."
            priority = "Routine"

        return {
            "priority": priority,
            "action": action,
            "focus": primary_cause,
            "reason": (
                "Recommendation generated using the predicted RUL, "
                "sensor evidence, root-cause analysis, and verification results."
            )
        }


class AgentOrchestrator:

    def run(
        self,
        predicted_rul,
        uncertainty,
        sensor_importance
    ):

        diagnostic_agent = DiagnosticAgent()
        root_cause_agent = RootCauseAgent()
        verification_agent = VerificationAgent()
        maintenance_agent = MaintenanceAgent()

        diagnostic = diagnostic_agent.run(
            predicted_rul,
            uncertainty,
            sensor_importance
        )

        root_cause = root_cause_agent.run(
            sensor_importance
        )

        verification = verification_agent.run(
            root_cause,
            sensor_importance,
            uncertainty
        )

        maintenance = maintenance_agent.run(
            diagnostic,
            root_cause,
            verification
        )

        return {
            "diagnostic": diagnostic,
            "root_cause": root_cause,
            "verification": verification,
            "maintenance": maintenance
        }


def run_agent_pipeline(
    predicted_rul,
    uncertainty,
    sensor_importance
):
    orchestrator = AgentOrchestrator()

    return orchestrator.run(
        predicted_rul,
        uncertainty,
        sensor_importance
    )