"""
Upload and document management routes.
"""

import os
import logging
from threading import Lock
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, Depends, UploadFile, File, HTTPException
from app.core.auth import get_current_user
from app.core.config import settings
from app.services.rag_service import process_document
from app.db.vector_store import clear_documents, get_documents, get_persistence_status
from app.models.schemas import UploadResponse

logger = logging.getLogger(__name__)

router = APIRouter()
_processing_lock = Lock()
_processing_status: Dict[int, Dict[str, Any]] = {}


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


def _set_processing_status(user_id: int, **updates: Any) -> None:
    with _processing_lock:
        current = _processing_status.get(user_id, {}).copy()
        current.update(updates)
        _processing_status[user_id] = current


def _get_processing_status(user_id: int) -> Dict[str, Any]:
    with _processing_lock:
        return _processing_status.get(user_id, {"status": "idle", "active": False}).copy()


def _run_document_processing(user_id: int, filename: str, file_path: str) -> None:
    _set_processing_status(user_id, status="processing", active=True, filename=filename, error=None)
    try:
        process_document(user_id, file_path)
        documents = get_documents(user_id)
        unique_docs = sorted({chunk.get("doc", "unknown") for chunk in documents})
        _set_processing_status(
            user_id,
            status="completed",
            active=False,
            filename=filename,
            error=None,
            chunks_created=len([chunk for chunk in documents if chunk.get("doc") == filename]),
            documents_loaded=len(unique_docs),
        )
    except Exception as exc:
        logger.exception("Background document processing failed for %s", file_path)
        _set_processing_status(
            user_id,
            status="failed",
            active=False,
            filename=filename,
            error=str(exc),
        )


@router.post("/upload", response_model=UploadResponse, summary="Upload and process a PDF")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user=Depends(get_current_user),
):
    """Upload a PDF file and queue RAG processing in the background."""
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    try:
        user_upload_dir = os.path.join(settings.upload_dir, user["username"])
        os.makedirs(user_upload_dir, exist_ok=True)
        file_path = os.path.join(user_upload_dir, file.filename)

        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        _set_processing_status(
            user["id"],
            status="queued",
            active=True,
            filename=file.filename,
            error=None,
        )
        background_tasks.add_task(_run_document_processing, user["id"], file.filename, file_path)
        return UploadResponse(
            success=True,
            filename=file.filename,
            chunks_created=0,
            documents_loaded=len({chunk.get("doc", "unknown") for chunk in get_documents(user["id"])}),
            status="queued",
            message="File uploaded successfully. Chunking and embedding are running in the background.",
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
        _set_processing_status(user["id"], status="idle", active=False, filename=None, error=None)
        return {"success": True, "message": "All documents cleared successfully"}
    except Exception as e:
        logger.error(f"Clear failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")


@router.get("/status", summary="System status")
async def get_status(user=Depends(get_current_user)):
    """Get system status including loaded documents and persistence info."""
    documents = get_documents(user["id"])
    unique_docs = set(chunk.get("doc", "unknown") for chunk in documents)
    persistence_status = get_persistence_status(user["id"])
    processing = _get_processing_status(user["id"])
    return {
        "success": True,
        "status": processing["status"] if processing.get("active") else ("ready" if documents else "idle"),
        "documents_loaded": len(unique_docs),
        "chunks_in_memory": len(documents),
        "processing": processing,
        "persistence": persistence_status,
        "model": settings.model_name,
    }
