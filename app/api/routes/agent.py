"""
Agent routes — ReAct pattern with tool calling.
"""

import json
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from app.core.auth import get_current_user
from app.models.schemas import AgentRequest, AgentResponse
from app.services.agent_service import run, run_stream
from app.services.usage_service import can_make_request, record_usage

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/agent", response_model=AgentResponse, summary="ReAct agent query")
async def agent(request: AgentRequest, user=Depends(get_current_user)):
    """Run the ReAct agent on a query. The agent can call tools like document search."""
    if not can_make_request(user["id"]):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Daily limit reached",
        )

    try:
        result = run(user["id"], user["username"], request.query, request.conversation_history)
        usage = record_usage(user["id"], int(result.get("tokens_used", 0)))
        return AgentResponse(
            success=result.get("success", False),
            query=request.query,
            answer=result.get("answer", ""),
            tools_used=result.get("tools_used", 0),
            tool_calls=result.get("tool_calls", []),
            reasoning_steps=result.get("reasoning_steps", []),
            memory_used=result.get("memory_used", False),
            usage=usage,
        )
    except Exception as e:
        logger.error(f"/agent failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent-stream", summary="Streaming ReAct agent query")
async def agent_stream(request: AgentRequest, user=Depends(get_current_user)):
    """Stream the ReAct agent response (Server-Sent Events)."""
    if not can_make_request(user["id"]):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Daily limit reached",
        )

    def generate():
        usage_recorded = False
        try:
            for chunk in run_stream(user["id"], user["username"], request.query, request.conversation_history):
                if chunk.get("type") == "metadata" and not usage_recorded:
                    chunk["usage"] = record_usage(user["id"], int(chunk.get("tokens_used", 0)))
                    usage_recorded = True
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            logger.error(f"/agent-stream failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
