"""
RAG service facade.
Single entry point for all RAG-related operations.
Routes call this; internals live in services/internals/.
"""

import logging
from app.services.internals.rag_pipeline import (
    process_pdf,
    ask_question,
    ask_question_stream,
    ask_question_stream_with_sources,
)

logger = logging.getLogger(__name__)


def process_document(user_id: int, file_path: str) -> None:
    """Process a PDF and store its embeddings."""
    logger.info(f"RAG service: processing document {file_path}")
    process_pdf(user_id, file_path)


def query(user_id: int, question: str) -> str:
    """Run a synchronous RAG query and return the answer string."""
    logger.info(f"RAG service: querying — {question[:60]}")
    return ask_question(user_id, question)


async def query_stream_async(user_id: int, question: str):
    """Async streaming RAG query (yields text chunks)."""
    async for chunk in ask_question_stream(user_id, question):
        yield chunk


def query_stream_with_sources(user_id: int, question: str):
    """Streaming RAG query with source metadata (yields dicts)."""
    return ask_question_stream_with_sources(user_id, question)
