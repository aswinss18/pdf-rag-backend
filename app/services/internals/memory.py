"""
Per-user memory system backed by SQLite.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from app.db import sqlite_store
from app.services.internals.embeddings import get_embedding

logger = logging.getLogger(__name__)


@dataclass
class UserMemoryState:
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    memory_store: List[Dict[str, Any]] = field(default_factory=list)
    memory_index: faiss.Index = field(default_factory=lambda: faiss.IndexFlatIP(1536))
    embedding_dim: int = 1536

    def rebuild_index(self) -> None:
        self.memory_index = faiss.IndexFlatIP(self.embedding_dim)
        for memory in self.memory_store:
            embedding = get_embedding(memory["text"])
            embedding_array = np.array([embedding]).astype("float32")
            faiss.normalize_L2(embedding_array)
            self.memory_index.add(embedding_array)


class AgentMemory:
    def __init__(self):
        self._states: Dict[int, UserMemoryState] = {}

    def _get_state(self, user_id: int) -> UserMemoryState:
        if user_id in self._states:
            return self._states[user_id]

        state = UserMemoryState()
        state.chat_history = sqlite_store.list_chat_history(user_id, limit=20)
        state.memory_store = sqlite_store.list_memories(user_id)
        if state.memory_store:
            state.rebuild_index()
        self._states[user_id] = state
        return state

    def add_to_chat_history(self, user_id: int, role: str, content: str):
        state = self._get_state(user_id)
        message = {"role": role, "content": content, "timestamp": datetime.utcnow().isoformat()}
        state.chat_history.append(message)
        state.chat_history = state.chat_history[-20:]
        sqlite_store.add_chat_message(user_id, role, content, message["timestamp"])
        sqlite_store.trim_chat_history(user_id, keep_last=20)

    def get_chat_history(self, user_id: int, max_messages: int = 10) -> List[Dict[str, str]]:
        state = self._get_state(user_id)
        recent_history = state.chat_history[-max_messages:] if state.chat_history else []
        return [{"role": msg["role"], "content": msg["content"]} for msg in recent_history]

    def store_memory(self, user_id: int, text: str, memory_type: str = "fact", metadata: Optional[Dict[str, Any]] = None):
        state = self._get_state(user_id)
        try:
            importance_score = self.calculate_importance(text)
            if importance_score < 0.6:
                return

            embedding = get_embedding(text)
            embedding_array = np.array([embedding]).astype("float32")
            faiss.normalize_L2(embedding_array)
            now = time.time()
            memory_entry = {
                "text": text,
                "type": memory_type,
                "importance": importance_score,
                "timestamp": now,
                "access_count": 0,
                "last_accessed": now,
                "metadata": metadata or {},
            }
            memory_id = sqlite_store.add_memory(
                user_id,
                text,
                memory_type,
                importance_score,
                now,
                0,
                now,
                metadata or {},
            )
            memory_entry["id"] = memory_id
            state.memory_store.append(memory_entry)
            state.memory_index.add(embedding_array)
        except Exception as exc:
            logger.error("Error storing memory for user_id=%s: %s", user_id, exc)

    def retrieve_memory(self, user_id: int, query: str, k: int = 3) -> List[Dict[str, Any]]:
        state = self._get_state(user_id)
        try:
            if len(state.memory_store) == 0:
                return []
            query_embedding = get_embedding(query)
            query_array = np.array([query_embedding]).astype("float32")
            faiss.normalize_L2(query_array)
            search_k = min(k * 3, len(state.memory_store))
            scores, indices = state.memory_index.search(query_array, search_k)
            ranked_memories = self._rank_memories(state.memory_store, query, scores[0], indices[0])
            top_memories = ranked_memories[:k]
            now = time.time()
            for memory in top_memories:
                memory["access_count"] = memory.get("access_count", 0) + 1
                memory["last_accessed"] = now
                sqlite_store.update_memory_access(memory["id"], memory["access_count"], memory["last_accessed"])
                for stored_memory in state.memory_store:
                    if stored_memory.get("id") == memory["id"]:
                        stored_memory["access_count"] = memory["access_count"]
                        stored_memory["last_accessed"] = memory["last_accessed"]
                        break
            return top_memories
        except Exception as exc:
            logger.error("Error retrieving memory for user_id=%s: %s", user_id, exc)
            return []

    def _rank_memories(self, memory_store: List[Dict[str, Any]], query: str, similarity_scores, indices) -> List[Dict[str, Any]]:
        current_time = time.time()
        ranked_memories = []
        for i, idx in enumerate(indices):
            if idx == -1:
                continue
            memory = memory_store[idx].copy()
            similarity = float(similarity_scores[i])
            if similarity < 0.3:
                continue
            age_days = (current_time - memory.get("timestamp", current_time)) / (24 * 60 * 60)
            if age_days <= 1:
                recency_score = 1.0
            elif age_days <= 7:
                recency_score = 0.8
            elif age_days <= 30:
                recency_score = 0.6
            elif age_days <= 90:
                recency_score = 0.4
            else:
                recency_score = 0.2
            access_count = memory.get("access_count", 0)
            frequency_score = min(access_count / 10.0, 1.0)
            query_words = set(query.lower().split())
            memory_words = set(memory["text"].lower().split())
            context_boost = min(len(query_words.intersection(memory_words)) * 0.1, 0.3)
            type_boost = 0.1 if memory.get("type") == "fact" else 0.05 if memory.get("type") == "preference" else 0.0
            combined_score = min(
                (
                    0.40 * similarity
                    + 0.30 * memory.get("importance", 0.5)
                    + 0.15 * recency_score
                    + 0.10 * frequency_score
                    + 0.05 * context_boost
                )
                + type_boost,
                1.0,
            )
            if combined_score > 0.5:
                memory.update(
                    {
                        "similarity_score": similarity,
                        "importance_score": memory.get("importance", 0.5),
                        "recency_score": recency_score,
                        "frequency_score": frequency_score,
                        "context_boost": context_boost,
                        "combined_score": combined_score,
                        "age_days": age_days,
                    }
                )
                ranked_memories.append(memory)
        ranked_memories.sort(key=lambda item: item["combined_score"], reverse=True)
        return ranked_memories

    def calculate_importance(self, text: str) -> float:
        score = 0.3
        text_lower = text.lower()
        for keyword in ["salary", "income", "name", "location", "work", "goal", "important"]:
            if keyword in text_lower:
                score += 0.3
        for keyword in ["skill", "experience", "prefer", "project", "team"]:
            if keyword in text_lower:
                score += 0.2
        for keyword in ["weather", "time", "date", "today", "week"]:
            if keyword in text_lower:
                score += 0.1
        if re.search(r"\d+", text):
            score += 0.1
        if any(symbol in text for symbol in ["$", "EUR", "INR", "Rs", "₹"]):
            score += 0.2
        if any(pronoun in text_lower for pronoun in ["my", "i am", "i work", "i live", "i like", "i prefer"]):
            score += 0.15
        if len(text.split()) < 3:
            score *= 0.7
        if len(text.split()) > 10:
            score += 0.1
        return min(score, 1.0)

    def should_store_memory(self, text: str) -> bool:
        return self.calculate_importance(text) > 0.6

    def extract_and_store_facts(self, user_id: int, conversation: str, response: str):
        try:
            if not self.should_store_memory(conversation):
                return
            facts_to_store = []
            lower_conversation = conversation.lower()
            if "my name is" in lower_conversation:
                facts_to_store.append(f"User's name is {lower_conversation.split('my name is')[1].split('.')[0].split(',')[0].strip()}")
            if "i live in" in lower_conversation:
                facts_to_store.append(f"User lives in {lower_conversation.split('i live in')[1].split('.')[0].split(',')[0].strip()}")
            if "i work as" in lower_conversation:
                facts_to_store.append(f"User works as {lower_conversation.split('i work as')[1].split('.')[0].split(',')[0].strip()}")
            elif "i am a" in lower_conversation:
                facts_to_store.append(f"User is a {lower_conversation.split('i am a')[1].split('.')[0].split(',')[0].strip()}")
            for fact in facts_to_store:
                self.store_memory(user_id, fact, "fact", {"source": "conversation", "confidence": "high"})
        except Exception as exc:
            logger.error("Error extracting facts for user_id=%s: %s", user_id, exc)

    def get_memory_context(self, user_id: int, query: str) -> str:
        relevant_memories = self.retrieve_memory(user_id, query, k=5)
        if not relevant_memories:
            return ""
        context_parts = ["RELEVANT MEMORY CONTEXT:"]
        for memory in relevant_memories[:5]:
            context_parts.append(f"- {memory['text']}")
        return "\n".join(context_parts) + "\n"

    def clear_chat_history(self, user_id: int):
        state = self._get_state(user_id)
        state.chat_history = []
        sqlite_store.clear_chat_history(user_id)

    def clear_all_memory(self, user_id: int):
        state = self._get_state(user_id)
        state.chat_history = []
        state.memory_store = []
        state.memory_index = faiss.IndexFlatIP(state.embedding_dim)
        sqlite_store.clear_chat_history(user_id)
        sqlite_store.clear_memories(user_id)

    def get_memory_stats(self, user_id: int) -> Dict[str, Any]:
        state = self._get_state(user_id)
        stats = {
            "chat_history_length": len(state.chat_history),
            "stored_memories": len(state.memory_store),
            "memory_index_size": state.memory_index.ntotal if state.memory_index else 0,
        }
        if state.memory_store:
            stats["average_importance"] = round(
                sum(memory.get("importance", 0.5) for memory in state.memory_store) / len(state.memory_store),
                3,
            )
            stats["scoring_system"] = "advanced_ranking_enabled"
        return stats

    def cleanup_old_memories(self, user_id: int, days_to_keep: int = 30):
        state = self._get_state(user_id)
        current_time = time.time()
        cutoff_timestamp = current_time - (days_to_keep * 24 * 60 * 60)
        kept_memories = []
        decay_applied_count = 0
        for memory in state.memory_store:
            age_days = (current_time - memory.get("timestamp", current_time)) / (24 * 60 * 60)
            original_importance = memory.get("importance", 0.5)
            decayed_importance = self._apply_memory_decay(original_importance, age_days)
            if decayed_importance != original_importance:
                memory["importance"] = decayed_importance
                decay_applied_count += 1
            if memory.get("timestamp", 0) > cutoff_timestamp or decayed_importance > 0.3:
                kept_memories.append(memory)
        removed_count = len(state.memory_store) - len(kept_memories)
        state.memory_store = kept_memories
        state.rebuild_index() if kept_memories else None
        if not kept_memories:
            state.memory_index = faiss.IndexFlatIP(state.embedding_dim)
        sqlite_store.replace_memories(user_id, kept_memories)
        return {"removed": removed_count, "kept": len(kept_memories), "decay_applied": decay_applied_count}

    def _apply_memory_decay(self, importance: float, age_days: float) -> float:
        if age_days < 1:
            return importance
        if age_days <= 7:
            decay_factor = 0.95
        elif age_days <= 30:
            decay_factor = 0.85
        elif age_days <= 90:
            decay_factor = 0.70
        elif age_days <= 180:
            decay_factor = 0.50
        else:
            decay_factor = 0.30
        decayed_importance = importance * decay_factor
        if importance > 0.9:
            decayed_importance = max(decayed_importance, 0.6)
        elif importance > 0.8:
            decayed_importance = max(decayed_importance, 0.4)
        return max(decayed_importance, 0.0)


agent_memory = AgentMemory()


def add_to_chat_history(user_id: int, role: str, content: str):
    agent_memory.add_to_chat_history(user_id, role, content)


def get_chat_history(user_id: int, max_messages: int = 10) -> List[Dict[str, str]]:
    return agent_memory.get_chat_history(user_id, max_messages)


def store_memory(user_id: int, text: str, memory_type: str = "fact", metadata: Optional[Dict[str, Any]] = None):
    agent_memory.store_memory(user_id, text, memory_type, metadata)


def retrieve_memory(user_id: int, query: str, k: int = 3) -> List[Dict[str, Any]]:
    return agent_memory.retrieve_memory(user_id, query, k)


def get_memory_context(user_id: int, query: str) -> str:
    return agent_memory.get_memory_context(user_id, query)


def get_memory_stats(user_id: int) -> Dict[str, Any]:
    return agent_memory.get_memory_stats(user_id)


def clear_chat_history(user_id: int):
    agent_memory.clear_chat_history(user_id)


def clear_all_memory(user_id: int):
    agent_memory.clear_all_memory(user_id)


def extract_and_store_facts(user_id: int, conversation: str, response: str):
    agent_memory.extract_and_store_facts(user_id, conversation, response)
