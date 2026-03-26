"""
RQ-backed upload job orchestration and status helpers.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from app.queue import get_queue, get_redis_connection
from app.db.vector_store import get_documents
from app.services.rag_service import process_document

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_meta(user_id: int, filename: str) -> Dict[str, Any]:
    now = _utc_now()
    return {
        "user_id": user_id,
        "filename": filename,
        "message": "Upload received. Chunking and embedding will start shortly.",
        "error": None,
        "created_at": now,
        "updated_at": now,
        "chunks_created": 0,
        "documents_loaded": 0,
    }


def _status_from_rq(job: Any) -> str:
    status = job.get_status(refresh=True)
    return {
        "queued": "queued",
        "started": "processing",
        "finished": "completed",
        "failed": "failed",
        "deferred": "queued",
        "scheduled": "queued",
        "stopped": "failed",
        "canceled": "failed",
    }.get(status, status or "queued")


def _serialize_job(job: Any) -> Dict[str, Any]:
    meta = job.get_meta(refresh=True) if hasattr(job, "get_meta") else (job.meta or {})
    return {
        "job_id": job.id,
        "user_id": meta.get("user_id"),
        "filename": meta.get("filename", ""),
        "status": _status_from_rq(job),
        "message": meta.get("message"),
        "error": meta.get("error"),
        "chunks_created": meta.get("chunks_created", 0),
        "documents_loaded": meta.get("documents_loaded", 0),
        "created_at": meta.get("created_at"),
        "updated_at": meta.get("updated_at"),
    }


def _job_collections() -> Iterable[str]:
    from rq.registry import DeferredJobRegistry, FailedJobRegistry, FinishedJobRegistry, ScheduledJobRegistry, StartedJobRegistry

    queue = get_queue()
    registries = [
        queue.job_ids,
        StartedJobRegistry(queue=queue).get_job_ids(),
        FinishedJobRegistry(queue=queue).get_job_ids(),
        FailedJobRegistry(queue=queue).get_job_ids(),
        DeferredJobRegistry(queue=queue).get_job_ids(),
        ScheduledJobRegistry(queue=queue).get_job_ids(),
    ]
    seen = set()
    for collection in registries:
        for job_id in collection:
            if job_id not in seen:
                seen.add(job_id)
                yield job_id


def _update_current_job(**updates: Any) -> None:
    from rq import get_current_job

    job = get_current_job(connection=get_redis_connection())
    if not job:
        return
    job.meta.update(updates)
    job.meta["updated_at"] = _utc_now()
    job.save_meta()


def process_pdf_job(user_id: int, filename: str, file_path: str) -> Dict[str, Any]:
    _update_current_job(
        status="processing",
        filename=filename,
        message="Chunking PDF and generating embeddings.",
        error=None,
    )
    try:
        process_document(user_id, file_path)
        documents = get_documents(user_id)
        unique_docs = sorted({chunk.get("doc", "unknown") for chunk in documents})
        chunks_created = len([chunk for chunk in documents if chunk.get("doc") == filename])
        _update_current_job(
            status="completed",
            filename=filename,
            message="Document processed successfully and is ready for search.",
            error=None,
            chunks_created=chunks_created,
            documents_loaded=len(unique_docs),
        )
        return {"status": "completed", "chunks_created": chunks_created, "documents_loaded": len(unique_docs)}
    except Exception as exc:
        logger.exception("Document processing failed for %s", file_path)
        _update_current_job(
            status="failed",
            filename=filename,
            message="Document processing failed.",
            error=str(exc),
        )
        raise


def enqueue_document_processing(user_id: int, filename: str, file_path: str) -> Dict[str, Any]:
    queue = get_queue()
    job = queue.enqueue(
        process_pdf_job,
        user_id,
        filename,
        file_path,
        meta=_job_meta(user_id, filename),
    )
    return _serialize_job(job)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    from rq.job import Job

    try:
        job = Job.fetch(job_id, connection=get_redis_connection())
    except Exception:
        logger.exception("Failed to fetch RQ job %s", job_id)
        return None
    return _serialize_job(job)


def clear_user_jobs(user_id: int) -> None:
    from rq.job import Job

    for job_id in _job_collections():
        try:
            job = Job.fetch(job_id, connection=get_redis_connection())
        except Exception:
            continue
        if job.meta.get("user_id") == user_id:
            job.delete(delete_dependents=True)


def get_latest_user_job(user_id: int) -> Optional[Dict[str, Any]]:
    from rq.job import Job

    latest_job: Optional[Dict[str, Any]] = None
    latest_updated_at = ""
    for job_id in _job_collections():
        try:
            job = Job.fetch(job_id, connection=get_redis_connection())
        except Exception:
            continue
        if job.meta.get("user_id") != user_id:
            continue
        serialized = _serialize_job(job)
        updated_at = serialized.get("updated_at") or ""
        if updated_at >= latest_updated_at:
            latest_updated_at = updated_at
            latest_job = serialized
    return latest_job
