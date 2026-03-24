"""
Authentication helpers for password hashing, JWT handling, and request context.
"""

import base64
import hashlib
import hmac
import json
import logging
import secrets
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings
from app.db.sqlite_store import get_user_by_id

logger = logging.getLogger(__name__)

security = HTTPBearer()

_current_user_id: ContextVar[Optional[int]] = ContextVar("current_user_id", default=None)
_current_username: ContextVar[Optional[str]] = ContextVar("current_username", default=None)


class JWTError(Exception):
    """Raised when a JWT cannot be decoded or validated."""


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _encode_jwt(payload: Dict[str, Any], secret: str, algorithm: str) -> str:
    if algorithm != "HS256":
        raise ValueError(f"Unsupported JWT algorithm: {algorithm}")

    header = {"alg": algorithm, "typ": "JWT"}
    encoded_header = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    encoded_payload = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{encoded_header}.{encoded_payload}.{_b64url_encode(signature)}"


def _decode_jwt(token: str, secret: str, algorithms: list[str]) -> Dict[str, Any]:
    if "HS256" not in algorithms:
        raise JWTError("Unsupported JWT algorithm")

    try:
        encoded_header, encoded_payload, encoded_signature = token.split(".")
    except ValueError as exc:
        raise JWTError("Invalid token format") from exc

    try:
        header = json.loads(_b64url_decode(encoded_header))
        payload = json.loads(_b64url_decode(encoded_payload))
    except (json.JSONDecodeError, ValueError) as exc:
        raise JWTError("Invalid token encoding") from exc

    if header.get("alg") != "HS256":
        raise JWTError("Unsupported JWT algorithm")

    signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
    expected_signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    actual_signature = _b64url_decode(encoded_signature)

    if not hmac.compare_digest(expected_signature, actual_signature):
        raise JWTError("Invalid token signature")

    exp = payload.get("exp")
    if exp is None:
        raise JWTError("Missing token expiration")

    try:
        exp_timestamp = int(exp)
    except (TypeError, ValueError) as exc:
        raise JWTError("Invalid token expiration") from exc

    now_timestamp = int(datetime.now(timezone.utc).timestamp())
    if exp_timestamp < now_timestamp:
        raise JWTError("Token has expired")

    return payload


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    derived_key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return f"pbkdf2_sha256$100000${_b64url_encode(salt)}${_b64url_encode(derived_key)}"


def verify_password(plain_password: str, password_hash: str) -> bool:
    try:
        algorithm, iterations, encoded_salt, encoded_hash = password_hash.split("$", 3)
    except ValueError:
        return False

    if algorithm != "pbkdf2_sha256":
        return False

    try:
        salt = _b64url_decode(encoded_salt)
        expected_hash = _b64url_decode(encoded_hash)
        iteration_count = int(iterations)
    except (ValueError, TypeError):
        return False

    candidate_hash = hashlib.pbkdf2_hmac(
        "sha256",
        plain_password.encode("utf-8"),
        salt,
        iteration_count,
    )
    return hmac.compare_digest(candidate_hash, expected_hash)


def create_access_token(data: Dict[str, Any]) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {**data, "exp": int(expire.timestamp())}
    return _encode_jwt(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


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
        payload = _decode_jwt(
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
