"""
Centralized configuration using environment variables.
All modules import settings from here instead of calling os.getenv() directly.
"""

import os
from dotenv import load_dotenv

# Load .env file at import time
load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    upload_dir: str = os.getenv("UPLOAD_DIR", "uploaded/")
    cache_dir: str = os.getenv("CACHE_DIR", "cache/")
    persistence_dir: str = os.getenv("PERSISTENCE_DIR", "persistence/")
    sqlite_db_path: str = os.getenv("SQLITE_DB_PATH", "persistence/app.db")
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
    allowed_origins: list = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://pdf-rag-app-xi.vercel.app",
        "https://*.vercel.app",
        "*"  # Allow all origins for development - remove in production
    ]

    def __init__(self):
        # Ensure required directories exist
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.persistence_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.sqlite_db_path) or ".", exist_ok=True)


settings = Settings()
