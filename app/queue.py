"""
Shared Redis and RQ queue helpers.

RQ imports are intentionally lazy so the FastAPI app can still import on
Windows, where some RQ versions currently fail at module import time.
"""

from redis import Redis

from app.core.config import settings


def get_redis_connection() -> Redis:
    return Redis.from_url(settings.redis_url)


def get_queue():
    try:
        from rq import Queue
    except ValueError as exc:
        raise RuntimeError(
            "RQ could not be imported in this environment. Run the API/worker in Docker or Linux/WSL,"
            " or install an RQ build that supports native Windows."
        ) from exc

    return Queue(settings.rq_queue_name, connection=get_redis_connection())
