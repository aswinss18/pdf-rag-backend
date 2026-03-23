"""
Authentication helpers for password hashing, JWT handling, and request context.
"""

import logging
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings
from app.db.sqlite_store import get_user_by_id

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

_current_user_id: ContextVar[Optional[int]] = ContextVar("current_user_id", default=None)
_current_username: ContextVar[Optional[str]] = ContextVar("current_username", default=None)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    return pwd_context.verify(plain_password, password_hash)


def create_access_token(data: Dict[str, Any]) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {**data, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def set_current_user_context(user_id: int, username: str) -> None:
    _current_user_id.set(user_id)
    _current_username.set(username)


def get_current_user_id() -> int:
    user_id = _current_user_id.get()
    if user_id is None:
        raise RuntimeError("User context is not available")
    return user_id


def get_current_username() -> str:
    username = _current_username.get()
    if username is None:
        raise RuntimeError("Username context is not available")
    return username


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication token",
    )

    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        user_id = int(payload.get("user_id"))
        username = payload.get("sub")
    except (JWTError, TypeError, ValueError) as exc:
        logger.warning("JWT decode failed: %s", exc)
        raise unauthorized

    if not username:
        raise unauthorized

    user = get_user_by_id(user_id)
    if not user or user["username"] != username:
        raise unauthorized

    set_current_user_context(user["id"], user["username"])
    return {"id": user["id"], "username": user["username"]}
