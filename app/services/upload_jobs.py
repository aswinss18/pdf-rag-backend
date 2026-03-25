"""
In-memory upload job tracking for background PDF processing.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, Optional


_jobs_lock = Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job(user_id: int, filename: str) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "user_id": user_id,
        "filename": filename,
        "status": "queued",
        "message": "Upload received. Chunking and embedding will start shortly.",
        "error": None,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "chunks_created": 0,
        "documents_loaded": 0,
    }
    with _jobs_lock:
        _jobs[job_id] = job
    return job.copy()


def update_job(job_id: str, **updates: Any) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return None
        job.update(updates)
        job["updated_at"] = _utc_now()
        return job.copy()


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return job.copy() if job else None


def clear_user_jobs(user_id: int) -> None:
    with _jobs_lock:
        job_ids = [job_id for job_id, job in _jobs.items() if job.get("user_id") == user_id]
        for job_id in job_ids:
            _jobs.pop(job_id, None)


def get_latest_user_job(user_id: int) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        matching_jobs = [job for job in _jobs.values() if job.get("user_id") == user_id]
        if not matching_jobs:
            return None
        latest = max(matching_jobs, key=lambda job: str(job.get("updated_at", "")))
        return latest.copy()
