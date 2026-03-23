"""
Memory management routes.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from app.core.auth import get_current_user
from app.services import memory_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/memory/stats", summary="Memory statistics")
async def memory_stats(user=Depends(get_current_user)):
    """Get current memory system statistics."""
    try:
        return {"success": True, **memory_service.get_stats(user["id"])}
    except Exception as e:
        logger.error(f"/memory/stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/info", summary="Detailed memory info")
async def memory_info(user=Depends(get_current_user)):
    """Get detailed memory info including recent samples."""
    try:
        return {"success": True, **memory_service.detailed_info(user["id"])}
    except Exception as e:
        logger.error(f"/memory/info failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/memory/chat", summary="Clear chat history")
async def clear_chat(user=Depends(get_current_user)):
    """Clear the short-term chat history."""
    try:
        memory_service.clear_chat(user["id"])
        return {"success": True, "message": "Chat history cleared"}
    except Exception as e:
        logger.error(f"/memory/chat DELETE failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/memory/all", summary="Clear all memory")
async def clear_all_memory(user=Depends(get_current_user)):
    """Clear all memory (long-term + chat history)."""
    try:
        memory_service.clear_all(user["id"])
        return {"success": True, "message": "All memory cleared"}
    except Exception as e:
        logger.error(f"/memory/all DELETE failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/cleanup", summary="Cleanup old memories")
async def cleanup_memory(days: int = 30, user=Depends(get_current_user)):
    """Remove memories older than `days` and apply decay."""
    try:
        result = memory_service.cleanup(user["id"], days)
        return {"success": True, **result}
    except Exception as e:
        logger.error(f"/memory/cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
