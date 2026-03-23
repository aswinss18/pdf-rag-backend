"""
Agent service facade.
Single entry point for all agent-related operations.
"""

import logging
from typing import Dict, Any, List, Optional
from app.services.internals.agent import run_agent, run_agent_stream

logger = logging.getLogger(__name__)


def run(user_id: int, username: str, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Run the ReAct agent on a query."""
    logger.info(f"Agent service: run — {query[:60]}")
    return run_agent(user_id, username, query, conversation_history)


def run_stream(user_id: int, username: str, query: str, conversation_history: Optional[List[Dict[str, str]]] = None):
    """Run the ReAct agent with streaming response."""
    logger.info(f"Agent service: run_stream — {query[:60]}")
    return run_agent_stream(user_id, username, query, conversation_history)
