"""Agent 06 — PredictiveMaintenance Agent.

Triggered by critical/high anomaly events. Runs SHAP feature importance,
ensemble forecasting on top features, and generates autonomous maintenance
schedules with LLM-synthesized insights.

SUB: f1:telemetry:anomaly
PUB: f1:predictive:maintenance
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from omniagents._types import EventSeverity
from omniagents.base import F1Agent
from omnianalytics.telemetry_loader import load_session_telemetry

logger = logging.getLogger(__name__)

TELEMETRY_FEATURES = ["speed", "n_gear", "rpm", "throttle", "brake", "drs"]

COMPONENT_MAP = {
    "Power Unit": ["rpm", "throttle"],
    "Brakes": ["brake"],
    "Drivetrain": ["speed", "n_gear"],
    "Aerodynamics": ["drs"],
}


class PredictiveMaintenanceAgent(F1Agent):
    """Orchestrates omnihealth predictive maintenance when anomalies are detected."""

    name = "predictive_maintenance"
    description = (
        "Triggered by critical/high anomaly events from the TelemetryAnomaly agent. "
        "Runs omnihealth assessment on car telemetry to produce SHAP-driven feature "
        "importance, ensemble forecasting, and autonomous maintenance schedules "
        "with LLM-synthesized race engineer insights."
    )
    subscriptions = ["f1:telemetry:anomaly"]
    publications = ["f1:predictive:maintenance"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._anomaly_buffer: Dict[int, List[Dict[str, Any]]] = {}

    async def on_event(self, topic: str, event: Dict[str, Any]):
        """Buffer critical/high severity anomaly events by session_key."""
        payload = event.get("payload", event)
        session_key = payload.get("session_key") or event.get("session_key")
        if not session_key:
            return

        severity = event.get("severity", "info")
        if severity in ("critical", "high"):
            if session_key not in self._anomaly_buffer:
                self._anomaly_buffer[session_key] = []
            self._anomaly_buffer[session_key].append(payload)
            logger.info("[predictive_maintenance] Buffered anomaly for session %s (severity=%s)", session_key, severity)

    async def run_analysis(
        self,
        session_key: int,
        driver_number: Optional[int] = None,
        year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run predictive maintenance analysis on car telemetry."""

        # 1. Load telemetry (shared loader with feature_store caching)
        telemetry_df = await asyncio.to_thread(
            load_session_telemetry, self._db, session_key, driver_number
        )
        if telemetry_df.empty:
            return {"status": "no_data", "session_key": session_key}

        # 2. Filter to available numeric telemetry features
        available = [f for f in TELEMETRY_FEATURES if f in telemetry_df.columns]
        if not available:
            logger.warning("No telemetry features found in columns: %s", telemetry_df.columns.tolist())
            return {"status": "no_features", "session_key": session_key}

        # 3. Downsample if >5000 rows
        if len(telemetry_df) > 5000:
            step = len(telemetry_df) // 2000
            telemetry_df = telemetry_df.iloc[::step].reset_index(drop=True)

        # 4. Build filtered component map
        filtered_map = {k: [c for c in v if c in available] for k, v in COMPONENT_MAP.items()}
        filtered_map = {k: v for k, v in filtered_map.items() if v}

        # 5. Run omnihealth assessment
        try:
            from omnihealth import assess as run_health_pipeline

            health_report = await asyncio.to_thread(
                run_health_pipeline,
                telemetry_df[available].fillna(0),
                filtered_map,
                horizon=30,
                include_schedule=True,
                include_timeseries=True,
                session_key=session_key,
                driver_number=driver_number,
                db=self._db,
            )
        except Exception:
            logger.exception("[predictive_maintenance] omnihealth.assess() failed for session %s", session_key)
            return {"status": "health_error", "session_key": session_key}

        # 6. Extract top features from HIGH/CRITICAL components
        top_features = []
        for hs in health_report.components:
            if hs.severity.value in ("high", "critical"):
                for f in hs.top_features:
                    feat_name = f.get("feature")
                    if feat_name and feat_name not in top_features:
                        top_features.append(feat_name)

        # 7. Generate LLM insight per risk assessment (medium+ risk)
        feature_insights = {}
        for ra in health_report.risk_assessments:
            if ra.risk_level.value in ("medium", "high", "critical"):
                insight = await self.reason(
                    f"Predictive maintenance for F1 telemetry feature '{ra.feature}' "
                    f"in session {session_key}{f' driver #{driver_number}' if driver_number else ''}. "
                    f"Current: {ra.current_value:.2f}, Forecast: {ra.forecast_value:.2f}, "
                    f"Trend: {ra.trend.value} ({ra.trend_pct:+.1f}%), Risk: {ra.risk_level.value}. "
                    "What car component is likely affected? What maintenance action should the team take?",
                    data_context=ra.to_dict(),
                )
                feature_insights[ra.feature] = insight

        # 8. Generate summary briefing
        summary = None
        if health_report.risk_assessments:
            summary = await self.reason(
                f"Synthesize predictive maintenance findings for session {session_key}. "
                f"Overall health: {health_report.overall_health}%, "
                f"Overall risk: {health_report.overall_risk.value}. "
                f"{health_report.schedule.total_tasks} maintenance tasks generated. "
                "Provide a concise race engineer briefing with prioritized actions.",
                data_context={
                    "schedule_summary": health_report.schedule.summary,
                    "priority_breakdown": health_report.schedule.priority_breakdown,
                    "top_features": top_features[:5],
                    "risk_count": len(health_report.risk_assessments),
                },
            )

        # 9. Determine event severity from overall risk
        sev_map = {
            "critical": EventSeverity.CRITICAL,
            "high": EventSeverity.HIGH,
            "medium": EventSeverity.MEDIUM,
            "low": EventSeverity.LOW,
        }
        event_severity = sev_map.get(health_report.overall_risk.value, EventSeverity.INFO)

        # 10. Publish to f1:predictive:maintenance
        output = {
            "session_key": session_key,
            "driver_number": driver_number,
            "overall_health": health_report.overall_health,
            "overall_risk": health_report.overall_risk.value,
            "top_features": top_features,
            "feature_insights": feature_insights,
            "schedule": health_report.schedule.to_dict(),
            "component_health": [c.to_dict() for c in health_report.components],
            "risk_assessments": [r.to_dict() for r in health_report.risk_assessments],
            "anomalies_buffered": len(self._anomaly_buffer.get(session_key, [])),
            "summary": summary or health_report.schedule.summary or "Predictive maintenance analysis completed",
        }

        await self.publish(
            topic="f1:predictive:maintenance",
            payload=output,
            severity=event_severity,
            session_key=session_key,
            driver_number=driver_number,
        )

        # 11. Save output and clear buffer
        await self.save_output("agent_predictive_maintenance", output)

        self._anomaly_buffer = {
            k: v for k, v in self._anomaly_buffer.items()
            if k != session_key
        }

        return {"summary": output["summary"], **output}

    # ── Internal methods ────────────────────────────────────────────────────

    def _load_telemetry(self, session_key: int, driver_number: Optional[int]) -> pd.DataFrame:
        """Load car telemetry from MongoDB."""
        if self._db is None:
            return pd.DataFrame()

        query: Dict[str, Any] = {"session_key": session_key}
        if driver_number:
            query["driver_number"] = driver_number

        # Try car_data first (backfilled OpenF1 data)
        collection = self._db["car_data"]
        cursor = collection.find(query, {"_id": 0}).sort("date", 1)
        docs = list(cursor)

        if not docs:
            # Fallback to openf1_car_data
            collection = self._db["openf1_car_data"]
            cursor = collection.find(query, {"_id": 0}).sort("date", 1)
            docs = list(cursor)

        if not docs:
            logger.warning("No telemetry found for session_key=%s driver=%s", session_key, driver_number)
            return pd.DataFrame()

        df = pd.DataFrame(docs)
        logger.info("Loaded %d telemetry samples for session %s", len(df), session_key)
        return df
