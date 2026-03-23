"""
SQLite-backed storage for users, document chunks, memories, and chat history.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Sequence

from app.core.config import settings

logger = logging.getLogger(__name__)


def _dict_factory(cursor: sqlite3.Cursor, row: Sequence[Any]) -> Dict[str, Any]:
    return {column[0]: row[index] for index, column in enumerate(cursor.description)}


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(settings.sqlite_db_path, check_same_thread=False)
    connection.row_factory = _dict_factory
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_database() -> None:
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                doc_name TEXT NOT NULL,
                page INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_document_chunks_user_id
            ON document_chunks(user_id);

            CREATE INDEX IF NOT EXISTS idx_document_chunks_user_doc
            ON document_chunks(user_id, doc_name);

            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                importance REAL NOT NULL,
                timestamp REAL NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed REAL NOT NULL,
                metadata_json TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_memories_user_id
            ON memories(user_id);

            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chat_history_user_id
            ON chat_history(user_id);
            """
        )
    logger.info("SQLite database initialized at %s", settings.sqlite_db_path)


def create_user(username: str, password_hash: str) -> Dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as connection:
        cursor = connection.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, created_at),
        )
        user_id = cursor.lastrowid
    return {"id": user_id, "username": username, "password_hash": password_hash, "created_at": created_at}


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    with get_connection() as connection:
        return connection.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
            (username,),
        ).fetchone()


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    with get_connection() as connection:
        return connection.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()


def replace_document_chunks(
    user_id: int,
    document_name: str,
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as connection:
        connection.execute(
            "DELETE FROM document_chunks WHERE user_id = ? AND doc_name = ?",
            (user_id, document_name),
        )
        connection.executemany(
            """
            INSERT INTO document_chunks (
                user_id, doc_name, page, chunk_index, text, embedding_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    user_id,
                    chunk.get("doc", document_name),
                    int(chunk.get("page", 0)),
                    int(chunk.get("chunk_index", index)),
                    chunk.get("text", ""),
                    json.dumps(embedding),
                    created_at,
                )
                for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            ],
        )


def list_document_chunks(user_id: int) -> List[Dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, user_id, doc_name, page, chunk_index, text, embedding_json, created_at
            FROM document_chunks
            WHERE user_id = ?
            ORDER BY doc_name, chunk_index, id
            """,
            (user_id,),
        ).fetchall()

    chunks = []
    for row in rows:
        chunks.append(
            {
                "id": row["id"],
                "user_id": row["user_id"],
                "doc": row["doc_name"],
                "page": row["page"],
                "chunk_index": row["chunk_index"],
                "text": row["text"],
                "embedding": json.loads(row["embedding_json"]),
                "created_at": row["created_at"],
            }
        )
    return chunks


def clear_document_chunks(user_id: int) -> None:
    with get_connection() as connection:
        connection.execute("DELETE FROM document_chunks WHERE user_id = ?", (user_id,))


def list_document_names(user_id: int) -> List[str]:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT DISTINCT doc_name FROM document_chunks WHERE user_id = ? ORDER BY doc_name",
            (user_id,),
        ).fetchall()
    return [row["doc_name"] for row in rows]


def add_memory(
    user_id: int,
    text: str,
    memory_type: str,
    importance: float,
    timestamp: float,
    access_count: int,
    last_accessed: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO memories (
                user_id, text, memory_type, importance, timestamp, access_count, last_accessed, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                text,
                memory_type,
                importance,
                timestamp,
                access_count,
                last_accessed,
                json.dumps(metadata or {}),
            ),
        )
        return int(cursor.lastrowid)


def list_memories(user_id: int) -> List[Dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, text, memory_type, importance, timestamp, access_count, last_accessed, metadata_json
            FROM memories
            WHERE user_id = ?
            ORDER BY timestamp ASC, id ASC
            """,
            (user_id,),
        ).fetchall()

    memories = []
    for row in rows:
        memories.append(
            {
                "id": row["id"],
                "text": row["text"],
                "type": row["memory_type"],
                "importance": row["importance"],
                "timestamp": row["timestamp"],
                "access_count": row["access_count"],
                "last_accessed": row["last_accessed"],
                "metadata": json.loads(row["metadata_json"]),
            }
        )
    return memories


def update_memory_access(memory_id: int, access_count: int, last_accessed: float) -> None:
    with get_connection() as connection:
        connection.execute(
            "UPDATE memories SET access_count = ?, last_accessed = ? WHERE id = ?",
            (access_count, last_accessed, memory_id),
        )


def replace_memories(user_id: int, memories: List[Dict[str, Any]]) -> None:
    with get_connection() as connection:
        connection.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
        connection.executemany(
            """
            INSERT INTO memories (
                id, user_id, text, memory_type, importance, timestamp, access_count, last_accessed, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    memory.get("id"),
                    user_id,
                    memory.get("text", ""),
                    memory.get("type", "fact"),
                    memory.get("importance", 0.5),
                    memory.get("timestamp", 0.0),
                    memory.get("access_count", 0),
                    memory.get("last_accessed", memory.get("timestamp", 0.0)),
                    json.dumps(memory.get("metadata", {})),
                )
                for memory in memories
            ],
        )


def clear_memories(user_id: int) -> None:
    with get_connection() as connection:
        connection.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))


def add_chat_message(user_id: int, role: str, content: str, timestamp: str) -> None:
    with get_connection() as connection:
        connection.execute(
            "INSERT INTO chat_history (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (user_id, role, content, timestamp),
        )


def list_chat_history(user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, role, content, timestamp
            FROM chat_history
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return list(reversed(rows))


def trim_chat_history(user_id: int, keep_last: int = 20) -> None:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id FROM chat_history WHERE user_id = ? ORDER BY id DESC LIMIT -1 OFFSET ?",
            (user_id, keep_last),
        ).fetchall()
        if rows:
            connection.executemany("DELETE FROM chat_history WHERE id = ?", [(row["id"],) for row in rows])


def clear_chat_history(user_id: int) -> None:
    with get_connection() as connection:
        connection.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
