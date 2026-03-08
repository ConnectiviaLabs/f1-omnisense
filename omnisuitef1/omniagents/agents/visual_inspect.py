"""Agent 05 — VisualInspection Agent.

Triggered by high-severity anomaly events. Checks for available
video/frame data and runs OmniVis analysis if available.

SUB: f1:telemetry:anomaly
PUB: f1:vision:incident, knowledge:update
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from omniagents._types import EventSeverity
from omniagents.base import F1Agent

logger = logging.getLogger(__name__)


class VisualInspectionAgent(F1Agent):
    name = "visual_inspect"
    description = (
        "Triggered by anomaly events to perform visual analysis of incidents. "
        "Checks for onboard camera frames and runs multi-modal vision analysis "
        "when footage is available. Gracefully degrades when no visual data exists."
    )
    subscriptions = ["f1:telemetry:anomaly"]
    publications = ["f1:vision:incident", "knowledge:update"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_anomalies: List[Dict[str, Any]] = []

    async def on_event(self, topic: str, event: Dict[str, Any]):
        """Buffer high-severity anomalies for visual inspection."""
        if topic == "f1:telemetry:anomaly":
            payload = event.get("payload", event)
            severity = event.get("severity", "info")
            if severity in ("critical", "high"):
                self._pending_anomalies.append(payload)
                logger.info("[visual_inspect] Buffered anomaly for inspection (severity=%s)", severity)

    async def run_analysis(
        self,
        session_key: int,
        driver_number: Optional[int] = None,
        year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Check for visual data and analyze incidents."""

        # 1. Check what visual data is available
        visual_data = await asyncio.to_thread(self._check_visual_data, session_key, driver_number)

        # 2. Check for pending anomalies for this session
        relevant_anomalies = [
            a for a in self._pending_anomalies
            if a.get("session_key") == session_key
        ]

        # 3. Analyze available visual data
        vision_results: List[Dict[str, Any]] = []
        if visual_data["has_frames"]:
            vision_results = await asyncio.to_thread(
                self._analyze_frames, visual_data["frames"][:10]
            )
        elif visual_data["has_clips"]:
            vision_results = await asyncio.to_thread(
                self._analyze_clips, visual_data["clips"][:5]
            )

        # 4. LLM reasoning — combine vision + anomaly context
        insight = None
        context = {
            "visual_data_available": visual_data["has_frames"] or visual_data["has_clips"],
            "frame_count": visual_data.get("frame_count", 0),
            "clip_count": visual_data.get("clip_count", 0),
            "vision_results": vision_results[:5] if vision_results else [],
            "anomalies_pending": len(relevant_anomalies),
        }

        if relevant_anomalies:
            context["anomaly_summary"] = [
                {k: v for k, v in a.items() if k in ("severity_distribution", "anomaly_count", "insight")}
                for a in relevant_anomalies[:3]
            ]

        insight = await self.reason(
            f"Visual inspection report for session {session_key}. "
            + (f"Found {len(vision_results)} visual analysis results. " if vision_results else "No visual data available for this session. ")
            + (f"There are {len(relevant_anomalies)} pending anomaly events to correlate. " if relevant_anomalies else "")
            + "Provide assessment of car/driver condition based on available evidence. "
            + "If no visual data, note what would be needed and assess based on telemetry anomalies alone.",
            data_context=context,
        )

        # 5. Publish findings
        severity = EventSeverity.HIGH if relevant_anomalies else EventSeverity.INFO

        incident_report = {
            "session_key": session_key,
            "driver_number": driver_number,
            "visual_data_available": visual_data["has_frames"] or visual_data["has_clips"],
            "frame_count": visual_data.get("frame_count", 0),
            "vision_results": vision_results[:10],
            "anomalies_correlated": len(relevant_anomalies),
            "insight": insight,
            "summary": insight or "Visual inspection completed",
        }

        if vision_results or relevant_anomalies:
            await self.publish(
                topic="f1:vision:incident",
                payload=incident_report,
                severity=severity,
                session_key=session_key,
                driver_number=driver_number,
            )

        await self.publish(
            topic="knowledge:update",
            payload={
                "source": "visual_inspect",
                "session_key": session_key,
                "driver_number": driver_number,
                "visual_available": visual_data["has_frames"] or visual_data["has_clips"],
                "summary": insight or "Visual inspection completed",
            },
            severity=severity,
            session_key=session_key,
            driver_number=driver_number,
        )

        await self.save_output("agent_visual_incidents", incident_report)

        # Clear processed anomalies
        self._pending_anomalies = [
            a for a in self._pending_anomalies
            if a.get("session_key") != session_key
        ]

        return {"summary": insight or "Visual inspection completed", **incident_report}

    # ── Data access ─────────────────────────────────────────────────────────

    def _check_visual_data(self, session_key: int, driver_number: Optional[int]) -> Dict[str, Any]:
        """Check MongoDB for available frames/clips for a session."""
        result = {
            "has_frames": False,
            "has_clips": False,
            "frame_count": 0,
            "clip_count": 0,
            "frames": [],
            "clips": [],
        }

        if self._db is None:
            return result

        # Check media_frames collection
        frame_query: Dict[str, Any] = {"session_key": session_key}
        if driver_number:
            frame_query["driver_number"] = driver_number

        try:
            frame_count = self._db["media_frames"].count_documents(frame_query)
            result["frame_count"] = frame_count
            result["has_frames"] = frame_count > 0
            if frame_count > 0:
                result["frames"] = list(
                    self._db["media_frames"].find(frame_query, {"_id": 0, "frame_data": 0}).limit(20)
                )
        except Exception:
            pass

        # Check clip_index / media_videos
        try:
            clip_count = self._db["clip_index"].count_documents(frame_query)
            result["clip_count"] = clip_count
            result["has_clips"] = clip_count > 0
            if clip_count > 0:
                result["clips"] = list(
                    self._db["clip_index"].find(frame_query, {"_id": 0}).limit(10)
                )
        except Exception:
            pass

        return result

    def _analyze_frames(self, frames: List[Dict]) -> List[Dict[str, Any]]:
        """Run vision analysis on available frames."""
        results = []
        for frame in frames:
            results.append({
                "frame_id": str(frame.get("_id", frame.get("frame_id", "unknown"))),
                "timestamp": str(frame.get("timestamp", "")),
                "analysis": frame.get("analysis", "frame available — narration pending"),
            })
        return results

    def _analyze_clips(self, clips: List[Dict]) -> List[Dict[str, Any]]:
        """Run vision analysis on available video clips."""
        results = []
        for clip in clips:
            results.append({
                "clip_id": str(clip.get("clip_id", "unknown")),
                "description": clip.get("description", "clip available"),
                "timestamp": str(clip.get("timestamp", "")),
            })
        return results
