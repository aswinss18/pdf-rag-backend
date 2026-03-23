"""
Per-user vector store backed by SQLite persisted chunks and embeddings.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import faiss
import numpy as np

from app.db import sqlite_store
from app.services.internals.keyword_search import KeywordSearcher

logger = logging.getLogger(__name__)

dimension = 1536


@dataclass
class UserVectorState:
    documents: List[Dict[str, Any]] = field(default_factory=list)
    embeddings: List[List[float]] = field(default_factory=list)
    keyword_searcher: KeywordSearcher = field(default_factory=KeywordSearcher)
    index: faiss.Index = field(default_factory=lambda: faiss.IndexFlatL2(dimension))

    def rebuild(self) -> None:
        self.index = faiss.IndexFlatL2(dimension)
        if self.embeddings:
            vectors = np.array(self.embeddings).astype("float32")
            self.index.add(vectors)
        self.keyword_searcher.build_index(self.documents)


_user_states: Dict[int, UserVectorState] = {}


def _load_user_state(user_id: int) -> UserVectorState:
    if user_id in _user_states:
        return _user_states[user_id]

    rows = sqlite_store.list_document_chunks(user_id)
    state = UserVectorState()
    state.documents = [
        {
            "id": row["id"],
            "doc": row["doc"],
            "page": row["page"],
            "chunk_index": row["chunk_index"],
            "text": row["text"],
        }
        for row in rows
    ]
    state.embeddings = [row["embedding"] for row in rows]
    state.rebuild()
    _user_states[user_id] = state
    return state


def add_embeddings(user_id: int, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
    if not chunks:
        return

    document_name = chunks[0].get("doc", "unknown")
    sqlite_store.replace_document_chunks(user_id, document_name, chunks, embeddings)
    _user_states.pop(user_id, None)
    _load_user_state(user_id)
    logger.info("Stored %s chunks for user_id=%s document=%s", len(chunks), user_id, document_name)


def clear_documents(user_id: int) -> None:
    sqlite_store.clear_document_chunks(user_id)
    _user_states.pop(user_id, None)


def get_documents(user_id: int) -> List[Dict[str, Any]]:
    return list(_load_user_state(user_id).documents)


def search(user_id: int, query_embedding: List[float], k: int = 10) -> List[Dict[str, Any]]:
    state = _load_user_state(user_id)
    if not state.documents:
        return []

    vector = np.array([query_embedding]).astype("float32")
    distances, indices = state.index.search(vector, k)

    results = []
    for i, distance in zip(indices[0], distances[0]):
        if i < len(state.documents):
            document = state.documents[i]
            similarity_score = 1.0 / (1.0 + float(distance))
            results.append(
                {
                    "id": document.get("id"),
                    "text": document["text"],
                    "page": document["page"],
                    "doc": document["doc"],
                    "chunk_index": document.get("chunk_index", i),
                    "similarity_score": similarity_score,
                    "distance": float(distance),
                }
            )
    return results


def keyword_search(user_id: int, query: str, k: int = 10) -> List[Dict[str, Any]]:
    state = _load_user_state(user_id)
    return state.keyword_searcher.search(query, k)


def load_persisted_state() -> bool:
    return True


def get_persistence_status(user_id: int) -> Dict[str, Any]:
    documents = get_documents(user_id)
    unique_docs = sorted({document.get("doc", "unknown") for document in documents})
    return {
        "sqlite_db_path": sqlite_store.settings.sqlite_db_path if hasattr(sqlite_store, "settings") else "",
        "document_count": len(unique_docs),
        "total_chunks": len(documents),
        "documents": unique_docs,
    }
