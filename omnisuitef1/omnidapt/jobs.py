"""Background training job manager — thread-safe submit/run/get/cancel.

Ported from omnianalytics/jobs.py, adapted for training workloads.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional

from omnidapt._types import TrainingJob, TrainingStatus

logger = logging.getLogger(__name__)


class TrainingJobManager:
    """Thread-safe job registry for training tasks.

    States: QUEUED → PREPARING → TRAINING → EVALUATING → COMPLETED | FAILED | CANCELLED
    """

    def __init__(self, max_jobs: int = 50):
        self._jobs: Dict[str, TrainingJob] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._cancel_flags: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._max_jobs = max_jobs

    def submit(self, func: Callable, config=None, args: tuple = (), kwargs: Optional[dict] = None) -> str:
        """Submit a training function for background execution. Returns job_id."""
        self.cleanup()

        job = TrainingJob(config=config)

        with self._lock:
            self._jobs[job.job_id] = job
            cancel_event = threading.Event()
            self._cancel_flags[job.job_id] = cancel_event

        t = threading.Thread(
            target=self._run,
            args=(job.job_id, func, args, kwargs or {}),
            daemon=True,
        )
        self._threads[job.job_id] = t
        t.start()
        return job.job_id

    def _run(self, job_id: str, func: Callable, args: tuple, kwargs: dict):
        job = self._jobs.get(job_id)
        if not job:
            return

        job.status = TrainingStatus.PREPARING
        job.started_at = time.time()

        try:
            result = func(*args, job_id=job_id, cancel_event=self._cancel_flags.get(job_id), **kwargs)

            if self._cancel_flags.get(job_id, threading.Event()).is_set():
                job.status = TrainingStatus.CANCELLED
            else:
                job.status = TrainingStatus.COMPLETED
                job.progress = 100.0

                if hasattr(result, "to_dict"):
                    job.detail = str(result.to_dict())
                elif isinstance(result, dict):
                    if "metrics" in result:
                        from omnidapt._types import TrainingMetrics
                        job.metrics = TrainingMetrics.from_dict(result["metrics"])
                    if "result_path" in result:
                        job.result_path = result["result_path"]

        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error = str(e)
            logger.error("Training job %s failed: %s", job_id, e)
        finally:
            job.completed_at = time.time()

    def get(self, job_id: str) -> Optional[TrainingJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def update_progress(
        self, job_id: str, progress: float,
        status: Optional[TrainingStatus] = None,
        phase: str = "", detail: str = "",
    ):
        job = self._jobs.get(job_id)
        if job:
            job.progress = min(progress, 100.0)
            if status:
                job.status = status
            if phase:
                job.phase = phase
            if detail:
                job.detail = detail

    def list_jobs(self, status: Optional[TrainingStatus] = None) -> List[TrainingJob]:
        with self._lock:
            jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def cancel(self, job_id: str) -> bool:
        """Request cancellation of a running job."""
        flag = self._cancel_flags.get(job_id)
        if flag:
            flag.set()
            job = self._jobs.get(job_id)
            if job and job.status in (TrainingStatus.QUEUED, TrainingStatus.PREPARING, TrainingStatus.TRAINING):
                job.status = TrainingStatus.CANCELLED
                job.completed_at = time.time()
                return True
        return False

    def cleanup(self, max_age_s: float = 3600):
        now = time.time()
        with self._lock:
            stale = [
                jid for jid, j in self._jobs.items()
                if j.status in (TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED)
                and j.completed_at > 0
                and now - j.completed_at > max_age_s
            ]
            for jid in stale:
                del self._jobs[jid]
                self._cancel_flags.pop(jid, None)
                self._threads.pop(jid, None)


# ── Singleton ────────────────────────────────────────────────────────────

_manager: Optional[TrainingJobManager] = None


def get_job_manager() -> TrainingJobManager:
    global _manager
    if _manager is None:
        _manager = TrainingJobManager()
    return _manager
