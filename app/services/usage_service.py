"""
Usage tracking and daily request limiting helpers.
"""

from datetime import datetime
from typing import Any, Dict

from app.db.sqlite_store import get_usage_for_date, upsert_usage

DAILY_REQUEST_LIMIT = 20


def get_today_string() -> str:
    return datetime.now().date().isoformat()


def get_usage_snapshot(user_id: int) -> Dict[str, Any]:
    today = get_today_string()
    usage = get_usage_for_date(user_id, today)

    requests_used = int(usage["requests"]) if usage else 0
    tokens_used = int(usage["tokens"]) if usage else 0

    return {
        "date": today,
        "requests_used": requests_used,
        "requests_limit": DAILY_REQUEST_LIMIT,
        "requests_remaining": max(DAILY_REQUEST_LIMIT - requests_used, 0),
        "tokens_used": tokens_used,
    }


def can_make_request(user_id: int) -> bool:
    usage = get_usage_snapshot(user_id)
    return usage["requests_used"] < usage["requests_limit"]


def record_usage(user_id: int, tokens_used: int) -> Dict[str, Any]:
    usage = upsert_usage(
        user_id=user_id,
        date=get_today_string(),
        requests_delta=1,
        tokens_delta=max(tokens_used, 0),
    )
    requests_used = int(usage["requests"])
    tokens_total = int(usage["tokens"])
    return {
        "date": usage["date"],
        "requests_used": requests_used,
        "requests_limit": DAILY_REQUEST_LIMIT,
        "requests_remaining": max(DAILY_REQUEST_LIMIT - requests_used, 0),
        "tokens_used": tokens_total,
    }
