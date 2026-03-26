"""
Upload and document management routes.
"""

import os
import logging

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from app.core.auth import get_current_user
from app.core.config import settings
from app.services.upload_jobs import clear_user_jobs, enqueue_document_processing, get_job, get_latest_user_job
from app.db.vector_store import clear_documents, get_documents, get_persistence_status
from app.models.schemas import JobStatusResponse, UploadResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def _summarize_documents(chunks):
    """Build per-document statistics from a stable chunk snapshot."""
    doc_info = {}
    for chunk in chunks:
        doc_name = chunk.get("doc", "unknown")
        page = chunk.get("page", 0)

        if doc_name not in doc_info:
            doc_info[doc_name] = {"chunk_count": 0, "pages": set()}

        doc_info[doc_name]["chunk_count"] += 1
        doc_info[doc_name]["pages"].add(page)

    for info in doc_info.values():
        pages = sorted(info["pages"])
        info["pages"] = pages
        info["page_range"] = f"{min(pages)}-{max(pages)}" if pages else "unknown"

    return doc_info


@router.post("/upload", response_model=UploadResponse, summary="Upload and process a PDF")
async def upload_pdf(
    file: UploadFile = File(...),
    user=Depends(get_current_user),
):
    """Upload a PDF file and enqueue RAG processing with RQ."""
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    try:
        user_upload_dir = os.path.join(settings.upload_dir, user["username"])
        os.makedirs(user_upload_dir, exist_ok=True)
        file_path = os.path.join(user_upload_dir, file.filename)

        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        job = enqueue_document_processing(user["id"], file.filename, file_path)
        return UploadResponse(
            success=True,
            filename=file.filename,
            job_id=job["job_id"],
            chunks_created=0,
            documents_loaded=len({chunk.get("doc", "unknown") for chunk in get_documents(user["id"])}),
            status="queued",
            message=job["message"],
        )
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    finally:
        await file.close()


@router.get("/documents", summary="List loaded documents")
async def list_documents(user=Depends(get_current_user)):
    """List all currently loaded documents with statistics."""
    chunks_snapshot = get_documents(user["id"])
    doc_info = _summarize_documents(chunks_snapshot)
    return {
        "success": True,
        "total_documents": len(doc_info),
        "total_chunks": len(chunks_snapshot),
        "documents": doc_info,
    }


@router.delete("/documents", summary="Clear all loaded documents")
async def clear_all_documents(user=Depends(get_current_user)):
    """Remove all loaded documents and reset the vector store."""
    try:
        clear_documents(user["id"])
        clear_user_jobs(user["id"])
        return {"success": True, "message": "All documents cleared successfully"}
    except Exception as e:
        logger.error(f"Clear failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")


@router.get("/job/{job_id}", response_model=JobStatusResponse, summary="Check upload job status")
async def check_job_status(job_id: str, user=Depends(get_current_user)):
    """Return the current status for an RQ upload job."""
    job = get_job(job_id)
    if not job:
        logger.warning("Job %s was not found in Redis for user_id=%s", job_id, user["id"])
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("user_id") != user["id"]:
        logger.warning(
            "Job %s belongs to user_id=%s but was requested by user_id=%s",
            job_id,
            job.get("user_id"),
            user["id"],
        )
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        success=True,
        job_id=job["job_id"],
        filename=job["filename"],
        status=job["status"],
        message=job.get("message"),
        error=job.get("error"),
        chunks_created=job.get("chunks_created", 0),
        documents_loaded=job.get("documents_loaded", 0),
        created_at=job.get("created_at"),
        updated_at=job.get("updated_at"),
    )


@router.get("/status", summary="System status")
async def get_status(user=Depends(get_current_user)):
    """Get system status including loaded documents and persistence info."""
    documents = get_documents(user["id"])
    unique_docs = set(chunk.get("doc", "unknown") for chunk in documents)
    persistence_status = get_persistence_status(user["id"])
    latest_job = get_latest_user_job(user["id"])
    return {
        "success": True,
        "status": latest_job["status"] if latest_job and latest_job.get("status") in {"queued", "processing"} else ("ready" if documents else "idle"),
        "documents_loaded": len(unique_docs),
        "chunks_in_memory": len(documents),
        "processing": latest_job or {"status": "idle"},
        "persistence": persistence_status,
        "model": settings.model_name,
    }
