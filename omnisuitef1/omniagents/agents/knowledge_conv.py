"""Agent 04 — KnowledgeConvergence Agent.

Fuses outputs from all other agents into unified race knowledge base.
Subscribes to all agent topics and produces consolidated race snapshots.

SUB: f1:telemetry:anomaly, f1:strategy:recommend, f1:pit:window:open, f1:vision:incident
PUB: knowledge:fused
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from omniagents._types import EventSeverity
from omniagents.base import F1Agent

logger = logging.getLogger(__name__)


class KnowledgeConvergenceAgent(F1Agent):
    name = "knowledge_convergence"
    description = (
        "Fuses outputs from all agents (anomaly detection, weather, pit strategy, "
        "visual inspection) into unified race knowledge snapshots. Produces "
        "consolidated race engineer briefings with multi-source evidence."
    )
    subscriptions = [
        "f1:telemetry:anomaly",
        "f1:strategy:recommend",
        "f1:pit:window:open",
        "f1:vision:incident",
        "f1:predictive:maintenance",
        "knowledge:update",
    ]
    publications = ["knowledge:fused"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Accumulator: session_key → list of agent findings
        self._knowledge_buffer: Dict[int, List[Dict[str, Any]]] = {}

    async def on_event(self, topic: str, event: Dict[str, Any]):
        """Accumulate events from other agents."""
        payload = event.get("payload", event)
        session_key = payload.get("session_key") or event.get("session_key")
        if not session_key:
            return

        if session_key not in self._knowledge_buffer:
            self._knowledge_buffer[session_key] = []

        self._knowledge_buffer[session_key].append({
            "topic": topic,
            "agent": event.get("agent", "unknown"),
            "timestamp": event.get("timestamp", time.time()),
            "severity": event.get("severity", "info"),
            "data": payload,
        })
        logger.info("[knowledge_convergence] Buffered event from %s for session %s", topic, session_key)

    async def run_analysis(
        self,
        session_key: int,
        driver_number: Optional[int] = None,
        year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Fuse all accumulated knowledge for a session into a single briefing."""

        buffer = self._knowledge_buffer.get(session_key, [])

        if not buffer:
            return {"status": "no_data", "session_key": session_key, "summary": "No agent data to fuse"}

        # 1. Organize by source agent
        by_source: Dict[str, List[Dict]] = {}
        for entry in buffer:
            src = entry.get("agent", "unknown")
            if src not in by_source:
                by_source[src] = []
            by_source[src].append(entry)

        # 2. Extract key findings per source
        findings = self._extract_findings(by_source)

        # 3. Determine overall race status
        severities = [e.get("severity", "info") for e in buffer]
        overall_severity = self._worst_severity(severities)

        # 4. LLM fusion — race engineer briefing
        briefing = await self.reason(
            f"You have received intelligence from {len(by_source)} agents for session {session_key}. "
            "Fuse these findings into a single race engineer briefing. "
            "Prioritize actionable insights. Highlight conflicts between agent assessments. "
            "Provide a clear recommendation.",
            data_context=findings,
        )

        # 5. Publish fused knowledge
        severity_map = {"critical": EventSeverity.CRITICAL, "high": EventSeverity.HIGH,
                        "medium": EventSeverity.MEDIUM, "low": EventSeverity.LOW}
        ev_severity = severity_map.get(overall_severity, EventSeverity.INFO)

        fused = {
            "session_key": session_key,
            "driver_number": driver_number,
            "sources": list(by_source.keys()),
            "source_count": len(by_source),
            "event_count": len(buffer),
            "findings": findings,
            "overall_severity": overall_severity,
            "briefing": briefing,
            "summary": briefing or f"Fused {len(buffer)} events from {len(by_source)} agents",
        }

        await self.publish(
            topic="knowledge:fused",
            payload=fused,
            severity=ev_severity,
            session_key=session_key,
            driver_number=driver_number,
        )

        await self.save_output("agent_knowledge_fused", fused)
        return fused

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _extract_findings(self, by_source: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Extract key findings per source agent."""
        findings: Dict[str, Any] = {}

        for source, entries in by_source.items():
            source_findings: List[str] = []
            for entry in entries:
                data = entry.get("data", {})
                # Extract summary or insight
                summary = data.get("summary") or data.get("insight") or data.get("recommendation")
                if summary:
                    source_findings.append(str(summary)[:300])
                elif data.get("anomaly_count"):
                    source_findings.append(f"{data['anomaly_count']} anomalies detected")
                elif data.get("event_count"):
                    source_findings.append(f"{data['event_count']} events detected")

            findings[source] = {
                "event_count": len(entries),
                "findings": source_findings,
                "worst_severity": self._worst_severity([e.get("severity", "info") for e in entries]),
            }

        return findings

    @staticmethod
    def _worst_severity(severities: List[str]) -> str:
        order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
        worst = max(severities, key=lambda s: order.get(s, 0), default="info")
        return worst
