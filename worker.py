"""
RQ worker entrypoint.

Run with:
python worker.py
"""

from app.queue import get_queue, get_redis_connection


if __name__ == "__main__":
    from rq import Worker

    redis_conn = get_redis_connection()
    queue = get_queue()
    worker = Worker([queue], connection=redis_conn)
    worker.work()
