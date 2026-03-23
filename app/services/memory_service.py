"""
Memory service facade.
Single entry point for all memory-related operations.
"""

import logging
import time
from typing import Dict, Any
from app.services.internals.memory import (
    agent_memory,
    get_memory_stats,
    clear_chat_history,
    clear_all_memory,
)

logger = logging.getLogger(__name__)


def get_stats(user_id: int) -> Dict[str, Any]:
    """Return current memory statistics."""
    return get_memory_stats(user_id)


def clear_chat(user_id: int) -> None:
    """Clear short-term chat history."""
    clear_chat_history(user_id)


def clear_all(user_id: int) -> None:
    """Clear all memory (long-term + chat history)."""
    clear_all_memory(user_id)


def cleanup(user_id: int, days: int = 30) -> Dict[str, Any]:
    """Remove memories older than `days` and apply decay to aging ones."""
    return agent_memory.cleanup_old_memories(user_id, days)


def apply_decay(user_id: int) -> Dict[str, Any]:
    """Apply memory decay without removing memories (keeps last 365 days)."""
    return agent_memory.cleanup_old_memories(user_id, days_to_keep=365)


def detailed_info(user_id: int) -> Dict[str, Any]:
    """Return detailed memory info including recent memory samples."""
    stats = get_stats(user_id)
    recent_memories = []
    state = agent_memory._get_state(user_id)
    if len(state.memory_store) > 0:
        sorted_memories = sorted(
            state.memory_store,
            key=lambda x: x.get("timestamp", 0),
            reverse=True,
        )
        for memory in sorted_memories[:5]:
            recent_memories.append(
                {
                    "text": memory["text"][:100] + "..." if len(memory["text"]) > 100 else memory["text"],
                    "importance": memory.get("importance", 0.5),
                    "type": memory.get("type", "unknown"),
                    "access_count": memory.get("access_count", 0),
                    "confidence": memory.get("metadata", {}).get("confidence", "medium"),
                    "source": memory.get("metadata", {}).get("source", "unknown"),
                    "age_days": (time.time() - memory.get("timestamp", time.time())) / (24 * 60 * 60),
                }
            )
    system_health = {
        "total_memories": len(state.memory_store),
        "index_size": state.memory_index.ntotal if state.memory_index else 0,
        "average_quality": stats.get("average_importance", 0),
        "system_status": "healthy" if stats.get("scoring_system") == "advanced_ranking_enabled" else "degraded",
    }
    return {
        "memory_stats": stats,
        "recent_memories": recent_memories,
        "system_health": system_health,
    }
