"""
Document tools for the agent — search and listing.
Moved from core/tools.py. Imports updated to app package paths.
"""

import logging
from typing import Dict, Any
from app.core.auth import get_current_user_id
from app.services.internals.rag_pipeline import ask_question as rag_search
from app.db.vector_store import get_documents
from app.services.internals.multi_document_context import (
    group_chunks_by_document,
    analyze_document_distribution,
)

logger = logging.getLogger(__name__)


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
        info["total_pages"] = len(pages)

    return doc_info


def search_documents(query: str) -> Dict[str, Any]:
    """Search through uploaded PDF documents using hybrid RAG pipeline."""
    try:
        user_id = get_current_user_id()
        documents = get_documents(user_id)
        if not documents:
            return {
                "success": False,
                "error": "No documents are currently loaded. Please upload PDF documents first.",
                "answer": None,
                "document_count": 0,
            }
        answer = rag_search(user_id, query)
        unique_docs = set(chunk.get("doc", "unknown") for chunk in documents)
        return {
            "success": True,
            "answer": answer,
            "document_count": len(unique_docs),
            "total_chunks": len(documents),
            "documents": list(unique_docs),
            "query": query,
        }
    except Exception as e:
        logger.error(f"Error in document search: {e}")
        return {
            "success": False,
            "error": f"Document search failed: {str(e)}",
            "answer": None,
            "document_count": len(set(chunk.get("doc", "unknown") for chunk in documents)) if documents else 0,
        }


def list_available_documents() -> Dict[str, Any]:
    """List all currently loaded documents with their statistics."""
    try:
        user_id = get_current_user_id()
        documents = get_documents(user_id)
        if not documents:
            return {
                "success": True,
                "message": "No documents are currently loaded.",
                "documents": {},
                "total_documents": 0,
                "total_chunks": 0,
            }
        chunks_snapshot = list(documents)
        doc_info = _summarize_documents(chunks_snapshot)
        return {
            "success": True,
            "documents": doc_info,
            "total_documents": len(doc_info),
            "total_chunks": len(chunks_snapshot),
            "message": f"Found {len(doc_info)} documents with {len(chunks_snapshot)} total chunks.",
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return {
            "success": False,
            "error": f"Failed to list documents: {str(e)}",
            "documents": {},
            "total_documents": 0,
            "total_chunks": 0,
        }
