"""
Upload and document management routes.
"""

import os
import shutil
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.core.config import settings
from app.services.rag_service import process_document
from app.db.vector_store import documents, clear_documents, get_persistence_status
from app.models.schemas import UploadResponse

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
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file, process it through the RAG pipeline, and store embeddings."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    try:
        os.makedirs(settings.upload_dir, exist_ok=True)
        file_path = os.path.join(settings.upload_dir, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        process_document(file_path)

        unique_docs = set(chunk.get("doc", "unknown") for chunk in documents)
        return UploadResponse(
            success=True,
            filename=file.filename,
            chunks_created=len(documents),
            documents_loaded=len(unique_docs),
        )
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


@router.get("/documents", summary="List loaded documents")
async def list_documents():
    """List all currently loaded documents with statistics."""
    chunks_snapshot = list(documents)
    doc_info = _summarize_documents(chunks_snapshot)
    return {
        "success": True,
        "total_documents": len(doc_info),
        "total_chunks": len(chunks_snapshot),
        "documents": doc_info,
    }


@router.delete("/documents", summary="Clear all loaded documents")
async def clear_all_documents():
    """Remove all loaded documents and reset the vector store."""
    try:
        clear_documents()
        return {"success": True, "message": "All documents cleared successfully"}
    except Exception as e:
        logger.error(f"Clear failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")


@router.get("/status", summary="System status")
async def get_status():
    """Get system status including loaded documents and persistence info."""
    unique_docs = set(chunk.get("doc", "unknown") for chunk in documents)
    persistence_status = get_persistence_status()
    return {
        "success": True,
        "status": "ready" if documents else "idle",
        "documents_loaded": len(unique_docs),
        "chunks_in_memory": len(documents),
        "persistence": persistence_status,
        "model": settings.model_name,
    }
