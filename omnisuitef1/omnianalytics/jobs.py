"""Background job state machine for long-running analysis tasks."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from omnianalytics._types import JobState, JobStatus

logger = logging.getLogger(__name__)


class JobManager:
    """Thread-safe job registry with state machine.

    States: queued -> processing -> completed | failed
    """

    def __init__(self, max_jobs: int = 100):
        self._jobs: Dict[str, JobState] = {}
        self._lock = threading.Lock()
        self._max_jobs = max_jobs

    def submit(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ) -> str:
        """Submit a function for background execution. Returns job_id."""
        self.cleanup()

        job_id = uuid.uuid4().hex[:12]
        job = JobState(job_id=job_id)

        with self._lock:
            self._jobs[job_id] = job

        t = threading.Thread(
            target=self._run,
            args=(job_id, func, args, kwargs or {}),
            daemon=True,
        )
        t.start()
        return job_id

    def _run(self, job_id: str, func: Callable, args: tuple, kwargs: dict):
        job = self._jobs.get(job_id)
        if not job:
            return

        job.status = JobStatus.PROCESSING
        job.updated_at = time.time()

        try:
            result = func(*args, **kwargs)
            job.status = JobStatus.COMPLETED
            if hasattr(result, "to_dict"):
                job.result = result.to_dict()
            elif isinstance(result, dict):
                job.result = result
            else:
                job.result = {"output": str(result)}
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            logger.error("Job %s failed: %s", job_id, e)
        finally:
            job.updated_at = time.time()

    def get(self, job_id: str) -> Optional[JobState]:
        with self._lock:
            return self._jobs.get(job_id)

    def update_progress(self, job_id: str, pct: float, phase: str = "", detail: str = ""):
        job = self._jobs.get(job_id)
        if job:
            job.progress_pct = pct
            job.phase = phase
            job.phase_detail = detail
            job.updated_at = time.time()

    def list_jobs(self, status: Optional[JobStatus] = None) -> List[JobState]:
        with self._lock:
            jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def cleanup(self, max_age_s: float = 3600):
        now = time.time()
        with self._lock:
            stale = [
                jid for jid, j in self._jobs.items()
                if j.status in (JobStatus.COMPLETED, JobStatus.FAILED)
                and now - j.updated_at > max_age_s
            ]
            for jid in stale:
                del self._jobs[jid]


_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    global _manager
    if _manager is None:
        _manager = JobManager()
    return _manager
