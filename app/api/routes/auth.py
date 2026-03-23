"""
Authentication routes.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.auth import create_access_token, get_current_user, hash_password, verify_password
from app.db.sqlite_store import create_user, get_user_by_username
from app.models.schemas import AuthRequest, AuthResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/register", response_model=AuthResponse, summary="Create a user account")
async def register(request: AuthRequest):
    username = request.username.strip().lower()
    if not username or not request.password.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username and password are required")

    if get_user_by_username(username):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists")

    user = create_user(username, hash_password(request.password))
    access_token = create_access_token({"sub": user["username"], "user_id": user["id"]})
    return AuthResponse(
        success=True,
        access_token=access_token,
        token_type="bearer",
        username=user["username"],
    )


@router.post("/login", response_model=AuthResponse, summary="Login and get an access token")
async def login(request: AuthRequest):
    username = request.username.strip().lower()
    user = get_user_by_username(username)

    if not user or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")

    access_token = create_access_token({"sub": user["username"], "user_id": user["id"]})
    return AuthResponse(
        success=True,
        access_token=access_token,
        token_type="bearer",
        username=user["username"],
    )


@router.get("/me", summary="Get the authenticated user")
async def me(user=Depends(get_current_user)):
    return {"success": True, "user": user}
