"""Agent 01 — TelemetryAnomaly Agent.

Ingests car telemetry from MongoDB, runs the 4-model anomaly ensemble,
identifies critical anomalies, and produces LLM-synthesized insights.

SUB: (trigger via API or run-all)
PUB: f1:telemetry:anomaly
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from omniagents._types import EventSeverity
from omniagents.base import F1Agent

logger = logging.getLogger(__name__)

# Telemetry features for anomaly detection
TELEMETRY_FEATURES = ["speed", "n_gear", "rpm", "throttle", "brake", "drs"]


class TelemetryAnomalyAgent(F1Agent):
    name = "telemetry_anomaly"
    description = (
        "Detects anomalous car behavior from telemetry data using a 4-model ensemble "
        "(IsolationForest, OneClassSVM, KNN, PCA). Flags RPM drops, brake spikes, and "
        "unusual speed/throttle patterns mid-race."
    )
    subscriptions: List[str] = []
    publications = ["f1:telemetry:anomaly"]

    async def run_analysis(
        self,
        session_key: int,
        driver_number: Optional[int] = None,
        year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run anomaly detection on car telemetry for a session."""

        # 1. Load telemetry from MongoDB
        telemetry_df = await asyncio.to_thread(
            self._load_telemetry, session_key, driver_number
        )
        if telemetry_df.empty:
            return {"status": "no_data", "session_key": session_key}

        # 2. Run ensemble anomaly detection
        results_df = await asyncio.to_thread(self._run_ensemble, telemetry_df)

        # 3. Extract anomalies by severity
        anomalies = self._extract_anomalies(results_df, session_key, driver_number)

        # 4. LLM reasoning on critical/high anomalies
        critical = [a for a in anomalies if a["severity"] in ("critical", "high")]
        insight = None
        if critical:
            insight = await self.reason(
                f"Analyze these {len(critical)} anomalies detected in session {session_key}"
                f"{f' for driver #{driver_number}' if driver_number else ''}. "
                "What do they indicate about car health? What actions should the team take?",
                data_context={"anomalies": critical[:10]},
            )

        # 5. Publish findings
        severity = EventSeverity.CRITICAL if any(a["severity"] == "critical" for a in anomalies) else (
            EventSeverity.HIGH if any(a["severity"] == "high" for a in anomalies) else EventSeverity.INFO
        )

        await self.publish(
            topic="f1:telemetry:anomaly",
            payload={
                "session_key": session_key,
                "driver_number": driver_number,
                "total_samples": len(results_df),
                "anomaly_count": len(anomalies),
                "severity_distribution": self._severity_dist(anomalies),
                "critical_anomalies": critical[:5],
                "insight": insight,
                "summary": insight or f"{len(anomalies)} anomalies detected ({len(critical)} critical/high)",
            },
            severity=severity,
            session_key=session_key,
            driver_number=driver_number,
        )

        # 6. Persist to MongoDB
        output = {
            "session_key": session_key,
            "driver_number": driver_number,
            "total_samples": len(results_df),
            "anomaly_count": len(anomalies),
            "severity_distribution": self._severity_dist(anomalies),
            "anomalies": anomalies[:50],
            "insight": insight,
        }
        await self.save_output("agent_telemetry_anomalies", output)

        return {"summary": output["insight"] or f"{len(anomalies)} anomalies detected", **output}

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

    def _run_ensemble(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the 4-model anomaly ensemble on telemetry data."""
        from pipeline.anomaly.ensemble import AnomalyDetectionEnsemble, AnomalyStatistics

        # Select features that exist in the data
        available = [f for f in TELEMETRY_FEATURES if f in df.columns]
        if not available:
            logger.error("No telemetry features found in columns: %s", df.columns.tolist())
            return df

        feature_data = df[available].copy()
        feature_data = feature_data.fillna(feature_data.median()).fillna(0)

        # Scale
        scaler = StandardScaler()
        scaled_arr = scaler.fit_transform(feature_data)
        scaled_df = pd.DataFrame(scaled_arr, columns=available, index=feature_data.index)

        # Run ensemble
        ensemble = AnomalyDetectionEnsemble()
        _, results = ensemble.run_anomaly_detection_models(df.copy(), scaled_df)

        # Run severity classification
        stats = AnomalyStatistics()
        results = stats.anomaly_insights(results)

        return results

    def _extract_anomalies(
        self, df: pd.DataFrame, session_key: int, driver_number: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Extract anomaly records from results DataFrame."""
        if "Anomaly_Level" not in df.columns:
            return []

        anomaly_rows = df[df["Anomaly_Level"].isin(["low", "medium", "high", "critical"])]
        anomalies = []

        for _, row in anomaly_rows.iterrows():
            record: Dict[str, Any] = {
                "severity": row.get("Anomaly_Level", "unknown"),
                "score": float(row.get("Enhanced_Anomaly_Score", 0)),
                "vote_count": int(row.get("Voting_Score", 0) * 4),
                "session_key": session_key,
                "driver_number": driver_number or row.get("driver_number"),
            }
            # Include telemetry snapshot
            for feat in TELEMETRY_FEATURES:
                if feat in row.index:
                    val = row[feat]
                    record[feat] = float(val) if not (isinstance(val, float) and np.isnan(val)) else None

            if "date" in row.index:
                record["timestamp"] = str(row["date"])

            anomalies.append(record)

        # Sort by severity score descending
        anomalies.sort(key=lambda a: a.get("score", 0), reverse=True)
        return anomalies

    @staticmethod
    def _severity_dist(anomalies: List[Dict]) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for a in anomalies:
            sev = a.get("severity", "unknown")
            dist[sev] = dist.get(sev, 0) + 1
        return dist
